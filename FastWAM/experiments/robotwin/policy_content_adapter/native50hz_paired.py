#!/usr/bin/env python3
"""Strict native-50Hz paired RoboTwin collection and LeRobot-v2.1 export.

The simulator runs at 250 Hz.  Published observations are native simulator
states captured every fifth physics step; interpolation, duplication, and
30-to-50 Hz conversion are forbidden.  Four scene variants (C/R1/R2/R3)
must replay one physical trajectory exactly before they can be exported.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
COLLECTOR_PATH = ROBOTWIN_ROOT / "script" / "collect_paired_random_background.py"
BASE_TASK_PATH = ROBOTWIN_ROOT / "envs" / "_base_task.py"
DEFAULT_COLLECTION_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "policy_native50hz_paired.yml"
)
ROBOTWIN_CAMERA_CONFIG = ROBOTWIN_ROOT / "task_config" / "_camera_config.yml"
POLICY_CAMERA_TYPE = "Large_D435"

PHYSICS_HZ = 250
SAMPLE_EVERY_PHYSICS_STEPS = 5
FPS = 50
TIMESTAMP_DELTA_SECONDS = 0.02
ACTION_HORIZON = 32
STATE_BANK_ANCHORS_PER_TRAJECTORY = 8
STATE_ACTION_DIM = 14
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
VARIANT_DIRS = (
    "clean",
    "style_00_seed_0",
    "style_01_seed_1",
    "style_02_seed_2",
)
SCENE_VARIANTS = ("C", "R1", "R2", "R3")
CAMERA_PATHS = {
    "observation.images.cam_high": "observation/head_camera/rgb",
    "observation.images.cam_left_wrist": "observation/left_camera/rgb",
    "observation.images.cam_right_wrist": "observation/right_camera/rgb",
}
MOTOR_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
TASK_INSTRUCTIONS = {
    "place_a2b_left": "use appropriate arm to place object A on the left of object B",
    "open_microwave": "Use one arm to open the microwave.",
    "move_stapler_pad": "use appropriate arm to move the stapler to a colored mat",
}
SPLIT_COUNTS = {"train": 30, "val": 10, "test": 10}
CONTRACT_FILENAME = "native50hz_collection_contract.json"
PAIRED_MANIFEST_PATH = "meta/paired_contents.jsonl"


class Native50HzContractError(RuntimeError):
    """A collection/export cannot prove the native-50Hz paired contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Native50HzContractError(message)


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Native50HzContractError(f"cannot parse JSON {target}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {target}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - collection environment dependency
        raise Native50HzContractError("PyYAML is required for collection config audit") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Native50HzContractError(f"cannot parse YAML {path}: {exc}") from exc
    _require(isinstance(value, dict), f"YAML root must be an object: {path}")
    return value


def validate_collection_config(path: str | Path = DEFAULT_COLLECTION_CONFIG) -> dict[str, Any]:
    """Validate the checked-in sampling contract without importing SAPIEN."""

    config_path = Path(path).expanduser().resolve()
    _require(config_path.is_file(), f"native-50Hz config not found: {config_path}")
    value = _read_yaml(config_path)
    _require(value.get("save_freq") == 5, "collection config save_freq must be 5")
    _require(
        math.isclose(float(value.get("timestep", -1)), 1.0 / PHYSICS_HZ, abs_tol=1e-12),
        "collection config timestep must be exactly 1/250 second",
    )
    camera = value.get("camera")
    _require(isinstance(camera, Mapping), "collection config camera mapping is missing")
    _require(camera.get("collect_head_camera") is True, "head camera must be collected")
    _require(camera.get("collect_wrist_camera") is True, "both wrist cameras must be collected")
    _require(
        camera.get("head_camera_type") == POLICY_CAMERA_TYPE,
        f"head_camera_type must be {POLICY_CAMERA_TYPE!r} for native {IMAGE_WIDTH}x{IMAGE_HEIGHT}",
    )
    _require(
        camera.get("wrist_camera_type") == POLICY_CAMERA_TYPE,
        f"wrist_camera_type must be {POLICY_CAMERA_TYPE!r} for native {IMAGE_WIDTH}x{IMAGE_HEIGHT}",
    )
    _require(
        ROBOTWIN_CAMERA_CONFIG.is_file(),
        f"RoboTwin camera config not found: {ROBOTWIN_CAMERA_CONFIG}",
    )
    camera_catalog = _read_yaml(ROBOTWIN_CAMERA_CONFIG)
    selected_camera = camera_catalog.get(POLICY_CAMERA_TYPE)
    _require(
        isinstance(selected_camera, Mapping),
        f"RoboTwin camera type {POLICY_CAMERA_TYPE!r} is missing",
    )
    _require(
        int(selected_camera.get("w", -1)) == IMAGE_WIDTH
        and int(selected_camera.get("h", -1)) == IMAGE_HEIGHT,
        f"{POLICY_CAMERA_TYPE} must resolve to {IMAGE_WIDTH}x{IMAGE_HEIGHT}",
    )
    contract = value.get("policy_native50hz_contract")
    _require(isinstance(contract, Mapping), "policy_native50hz_contract is missing")
    expected = {
        "physics_hz": PHYSICS_HZ,
        "sample_every_physics_steps": SAMPLE_EVERY_PHYSICS_STEPS,
        "fps": FPS,
        "timestamp_delta_seconds": TIMESTAMP_DELTA_SECONDS,
        "action_horizon": ACTION_HORIZON,
        "action_dim": STATE_ACTION_DIM,
        "scene_variants": list(SCENE_VARIANTS),
        "interpolation": "forbidden",
        "camera_type": POLICY_CAMERA_TYPE,
        "image_shape_hwc": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
    }
    _require(
        dict(contract) == expected,
        "native-50Hz YAML contract differs from compiled constants",
    )
    return {
        "status": "PASS",
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "camera_catalog_path": str(ROBOTWIN_CAMERA_CONFIG.resolve()),
        "camera_catalog_sha256": sha256_file(ROBOTWIN_CAMERA_CONFIG),
        **expected,
    }


def collection_contract_value(
    *,
    task: str,
    requested_contents: int,
    config_path: str | Path = DEFAULT_COLLECTION_CONFIG,
) -> dict[str, Any]:
    _require(task in OFFICIAL_TASKS, f"unsupported policy task: {task}")
    _require(1 <= int(requested_contents) <= 50, "requested_contents must be in [1, 50]")
    config_audit = validate_collection_config(config_path)
    for source in (COLLECTOR_PATH, BASE_TASK_PATH):
        _require(source.is_file(), f"required RoboTwin source is missing: {source}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "task": task,
        "requested_contents": int(requested_contents),
        "physics_hz": PHYSICS_HZ,
        "physics_timestep_seconds": 1.0 / PHYSICS_HZ,
        "sampling": {
            "method": "native_global_physics_step_decimation",
            "sample_every_physics_steps": SAMPLE_EVERY_PHYSICS_STEPS,
            "fps": FPS,
            "timestamp_delta_seconds": TIMESTAMP_DELTA_SECONDS,
            "interpolation": "forbidden",
            "frame_duplication": "forbidden",
            "30_to_50_interpolation": "explicitly_forbidden",
        },
        "trajectory_contract": {
            "plan_count_per_content": 1,
            "deterministic_replays": list(SCENE_VARIANTS),
            "simulator_snapshot_support_claimed": False,
            "identity_evidence": "exact state/action traces plus task identity",
        },
        "output_contract": {
            "state_dim": STATE_ACTION_DIM,
            "action_dim": STATE_ACTION_DIM,
            "future_action_steps": ACTION_HORIZON,
            "camera_keys": list(CAMERA_PATHS),
            "image_shape_hwc": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        },
        "sources": {
            "collector": {
                "path": str(COLLECTOR_PATH.resolve()),
                "sha256": sha256_file(COLLECTOR_PATH),
            },
            "base_task": {
                "path": str(BASE_TASK_PATH.resolve()),
                "sha256": sha256_file(BASE_TASK_PATH),
                "timestep_source_line_contract": "scene.set_timestep(kwargs.get('timestep', 1 / 250))",
            },
            "config": config_audit,
        },
    }


def _load_legacy_validator() -> Any:
    script_dir = ROBOTWIN_ROOT / "script"
    _require(script_dir.is_dir(), f"RoboTwin script directory is missing: {script_dir}")
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    module = importlib.import_module("validate_paired_random_background")
    validator = getattr(module, "validate_content_dir", None)
    _require(callable(validator), "legacy strict paired validator is not importable")
    return validator


def _decode_jpeg(value: Any, *, label: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Native50HzContractError("OpenCV is required to decode RoboTwin JPEG frames") from exc
    encoded = np.frombuffer(np.asarray(value).tobytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    _require(image is not None, f"cannot decode JPEG frame {label}")
    _require(
        image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        f"camera frame {label} has shape {image.shape}, expected {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}",
    )
    return image


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    _require(path.is_file(), f"trace archive is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise Native50HzContractError(f"cannot load safe NPZ {path}: {exc}") from exc


def _array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _audit_native_variant(variant_dir: Path, *, decode_all_frames: bool) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise Native50HzContractError("h5py is required for raw paired validation") from exc

    metadata = _read_json(variant_dir / "metadata.json")
    state_trace = _load_npz_arrays(variant_dir / "state_trace.npz")
    action_trace = _load_npz_arrays(variant_dir / "action_trace.npz")
    _require("frame_trace_index" in state_trace, f"frame trace missing in {variant_dir}")
    trace_indices = np.asarray(state_trace["frame_trace_index"])
    _require(trace_indices.ndim == 1, f"frame_trace_index must be 1D in {variant_dir}")
    _require(
        len(trace_indices) >= ACTION_HORIZON + STATE_BANK_ANCHORS_PER_TRAJECTORY + 1,
        f"too few native frames for eight endpoint-safe state-bank anchors in {variant_dir}",
    )
    expected_trace = np.arange(len(trace_indices), dtype=np.int64) * SAMPLE_EVERY_PHYSICS_STEPS
    _require(
        np.array_equal(trace_indices, expected_trace),
        f"{variant_dir} is not native every-{SAMPLE_EVERY_PHYSICS_STEPS}-physics-step data",
    )

    hdf5_path = variant_dir / "data" / "episode0.hdf5"
    _require(hdf5_path.is_file(), f"raw episode HDF5 is missing: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as handle:
        joint_path = "joint_action/vector"
        _require(joint_path in handle, f"{joint_path} missing in {hdf5_path}")
        joints = np.asarray(handle[joint_path][()])
        _require(
            joints.shape == (len(trace_indices), STATE_ACTION_DIM),
            f"joint vector shape mismatch in {hdf5_path}: {joints.shape}",
        )
        _require(np.all(np.isfinite(joints)), f"non-finite joint vector in {hdf5_path}")
        camera_hashes: dict[str, str] = {}
        for camera_key, raw_path in CAMERA_PATHS.items():
            _require(raw_path in handle, f"camera {raw_path} missing in {hdf5_path}")
            dataset = handle[raw_path]
            _require(dataset.shape == (len(trace_indices),), f"camera length mismatch for {raw_path}")
            digest = hashlib.sha256()
            positions: Sequence[int]
            positions = range(len(dataset)) if decode_all_frames else (0, len(dataset) - 1)
            decoded_positions = set()
            for frame_index in range(len(dataset)):
                raw = np.asarray(dataset[frame_index]).tobytes()
                digest.update(raw)
                if frame_index in positions and frame_index not in decoded_positions:
                    _decode_jpeg(raw, label=f"{variant_dir.name}/{camera_key}/{frame_index}")
                    decoded_positions.add(frame_index)
            camera_hashes[camera_key] = digest.hexdigest()

    _require(int(metadata.get("frame_count", -1)) == len(trace_indices), "metadata frame_count mismatch")
    converted_frames = len(trace_indices) - 1
    # One sample consumes 33 states (t..t+32) and 32 future actions.
    valid_windows = converted_frames - ACTION_HORIZON
    _require(valid_windows >= 1, f"no {ACTION_HORIZON}-step future-action window in {variant_dir}")
    return {
        "metadata": metadata,
        "trace_indices": trace_indices,
        "state_trace": state_trace,
        "action_trace": action_trace,
        "joint_vector": joints,
        "raw_frame_count": len(trace_indices),
        "converted_frame_count": converted_frames,
        "valid_future_action_windows": valid_windows,
        "camera_encoded_sha256": camera_hashes,
        "hdf5_sha256": sha256_file(hdf5_path),
    }


def validate_raw_content(
    content_dir: str | Path,
    *,
    expected_task: str,
    expected_content_id: int,
    decode_all_frames: bool = True,
    run_legacy_validator: bool = True,
) -> dict[str, Any]:
    """Prove exact C/R1/R2/R3 identity and native 250-to-50Hz sampling."""

    root = Path(content_dir).expanduser().resolve()
    _require(root.is_dir(), f"content directory not found: {root}")
    _require(expected_task in OFFICIAL_TASKS, f"unsupported task {expected_task}")
    expected_name = f"content_{int(expected_content_id):06d}"
    _require(root.name == expected_name, f"content directory must be named {expected_name}")
    for name in VARIANT_DIRS:
        _require((root / name).is_dir(), f"missing scene variant {name} in {root}")
    _require((root / "COMPLETE.json").is_file(), f"missing COMPLETE.json in {root}")

    legacy_report: dict[str, Any] | None = None
    if run_legacy_validator:
        validator = _load_legacy_validator()
        clean_metadata = _read_json(root / "clean" / "metadata.json")
        legacy_report = validator(
            root,
            expected_task=expected_task,
            expected_content_id=int(expected_content_id),
            expected_content_seed=int(clean_metadata["content_seed"]),
            expected_style_seeds=(0, 1, 2),
            require_complete=True,
        )
        _require(
            isinstance(legacy_report, Mapping) and legacy_report.get("valid") is True,
            f"legacy exact replay validator rejected {root}: "
            f"{legacy_report.get('errors') if isinstance(legacy_report, Mapping) else legacy_report}",
        )

    variants = {
        name: _audit_native_variant(root / name, decode_all_frames=decode_all_frames)
        for name in VARIANT_DIRS
    }
    clean = variants["clean"]
    for name in VARIANT_DIRS[1:]:
        candidate = variants[name]
        _require(
            np.array_equal(clean["trace_indices"], candidate["trace_indices"]),
            f"native sample indices differ between clean and {name}",
        )
        _require(
            _array_bytes_equal(clean["joint_vector"], candidate["joint_vector"]),
            f"14D state/action source differs between clean and {name}",
        )
        for artifact in ("state_trace", "action_trace"):
            clean_arrays = clean[artifact]
            candidate_arrays = candidate[artifact]
            _require(set(clean_arrays) == set(candidate_arrays), f"{artifact} keys differ in {name}")
            for key in clean_arrays:
                _require(
                    _array_bytes_equal(clean_arrays[key], candidate_arrays[key]),
                    f"{artifact}/{key} differs between clean and {name}",
                )

    frame_count = int(clean["converted_frame_count"])
    return {
        "status": "PASS",
        "task": expected_task,
        "content_id": int(expected_content_id),
        "content_dir": str(root),
        "scene_variants": dict(zip(SCENE_VARIANTS, VARIANT_DIRS, strict=True)),
        "physics_hz": PHYSICS_HZ,
        "sample_every_physics_steps": SAMPLE_EVERY_PHYSICS_STEPS,
        "fps": FPS,
        "timestamp_delta_seconds": TIMESTAMP_DELTA_SECONDS,
        "interpolation_used": False,
        "raw_native_frame_count": int(clean["raw_frame_count"]),
        "converted_frame_count": frame_count,
        "valid_future_action_windows": frame_count - ACTION_HORIZON,
        "exact_state_action_trace_identity": True,
        "legacy_exact_replay_audit": legacy_report,
    }


def _expected_split(content_id: int) -> str:
    if content_id < SPLIT_COUNTS["train"]:
        return "train"
    if content_id < SPLIT_COUNTS["train"] + SPLIT_COUNTS["val"]:
        return "val"
    return "test"


def validate_raw_task_root(
    task_root: str | Path,
    *,
    expected_task: str,
    expected_contents: int,
    decode_all_frames: bool = True,
    run_legacy_validator: bool = True,
) -> dict[str, Any]:
    root = Path(task_root).expanduser().resolve()
    _require(root.is_dir(), f"raw task root not found: {root}")
    _require(1 <= int(expected_contents) <= 50, "expected_contents must be in [1, 50]")
    contract = _read_json(root / CONTRACT_FILENAME)
    _require(contract.get("status") == "PASS", "collection contract is not PASS")
    _require(contract.get("task") == expected_task, "collection contract task mismatch")
    _require(
        int(contract.get("requested_contents", -1)) == int(expected_contents),
        "collection contract requested_contents mismatch",
    )
    sampling = contract.get("sampling")
    _require(isinstance(sampling, Mapping), "collection sampling contract is missing")
    _require(sampling.get("interpolation") == "forbidden", "interpolation is not forbidden")
    _require(
        sampling.get("sample_every_physics_steps") == SAMPLE_EVERY_PHYSICS_STEPS,
        "collection decimation contract mismatch",
    )
    sources = contract.get("sources")
    _require(isinstance(sources, Mapping), "collection source hashes are missing")
    for key, expected_path in (("collector", COLLECTOR_PATH), ("base_task", BASE_TASK_PATH)):
        declaration = sources.get(key)
        _require(isinstance(declaration, Mapping), f"collection source {key} is missing")
        _require(
            declaration.get("sha256") == sha256_file(expected_path),
            f"collection source {key} changed after collection",
        )

    contents_root = root / "contents"
    _require(contents_root.is_dir(), f"contents directory is missing: {contents_root}")
    actual_names = sorted(path.name for path in contents_root.iterdir() if path.is_dir())
    expected_names = [f"content_{index:06d}" for index in range(int(expected_contents))]
    _require(actual_names == expected_names, "raw contents are not exact/contiguous")
    content_reports = [
        validate_raw_content(
            contents_root / name,
            expected_task=expected_task,
            expected_content_id=index,
            decode_all_frames=decode_all_frames,
            run_legacy_validator=run_legacy_validator,
        )
        for index, name in enumerate(expected_names)
    ]
    split_counts = {
        split: sum(_expected_split(index) == split for index in range(int(expected_contents)))
        for split in SPLIT_COUNTS
    }
    if int(expected_contents) == 50:
        _require(split_counts == SPLIT_COUNTS, "full physical-trajectory split is not 30/10/10")
    return {
        "status": "PASS",
        "task": expected_task,
        "task_root": str(root),
        "content_count": int(expected_contents),
        "scene_episode_count": int(expected_contents) * 4,
        "physical_trajectory_split": split_counts,
        "all_contents_native_50hz": True,
        "all_four_scene_replays_exact": True,
        "minimum_converted_frames": min(item["converted_frame_count"] for item in content_reports),
        "minimum_future_action_windows": min(
            item["valid_future_action_windows"] for item in content_reports
        ),
        "contents": content_reports,
        "collection_contract_sha256": sha256_file(root / CONTRACT_FILENAME),
    }


def _numeric_stats(array: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(array)
    _require(values.shape[0] > 0, "cannot compute stats for an empty array")
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "count": [int(values.shape[0])],
    }


def _decode_camera_frames(hdf5_path: Path, raw_path: str, count: int) -> list[np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise Native50HzContractError("h5py is required for LeRobot export") from exc
    frames: list[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as handle:
        dataset = handle[raw_path]
        _require(len(dataset) >= count, f"not enough camera frames in {hdf5_path}:{raw_path}")
        for index in range(count):
            frames.append(_decode_jpeg(dataset[index], label=f"{hdf5_path}:{raw_path}:{index}"))
    return frames


def _encode_av1(frames: Sequence[np.ndarray], output_path: Path) -> dict[str, Any]:
    _require(bool(frames), f"refusing to encode an empty video: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libsvtav1",
        "-preset",
        "8",
        "-crf",
        "30",
        "-svtav1-params",
        "lp=16",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    # SVT-AV1 writes its own warnings directly to stderr even with ffmpeg's
    # ``-loglevel error``.  Do not leave stderr as an undrained PIPE while this
    # process synchronously streams raw frames to stdin: a full stderr pipe and
    # a full stdin pipe otherwise deadlock each other.  A temporary file keeps
    # the complete diagnostic without a bounded pipe buffer or a second in-RAM
    # copy of the (potentially >1 GiB) raw frame stream.
    with tempfile.TemporaryFile(prefix="fastwam-svtav1-", suffix=".stderr") as stderr_file:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr_file)
        except OSError as exc:
            raise Native50HzContractError("ffmpeg is required for LeRobot video export") from exc
        assert process.stdin is not None
        try:
            for frame in frames:
                _require(
                    frame.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3) and frame.dtype == np.uint8,
                    "decoded video frame has an invalid shape/dtype",
                )
                process.stdin.write(np.ascontiguousarray(frame).tobytes(order="C"))
            process.stdin.close()
            return_code = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
        except Exception:
            process.kill()
            process.wait()
            raise
    _require(return_code == 0, f"ffmpeg failed for {output_path}: {stderr.decode(errors='replace')}")
    return _probe_video(output_path, expected_frames=len(frames))


def _probe_video(path: Path, *, expected_frames: int | None = None) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        stream = json.loads(completed.stdout)["streams"][0]
    except Exception as exc:
        raise Native50HzContractError(f"cannot audit video {path}: {exc}") from exc
    numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
    rate = numerator / denominator
    frames = int(stream["nb_read_frames"])
    _require((int(stream["height"]), int(stream["width"])) == (IMAGE_HEIGHT, IMAGE_WIDTH), "video dimensions mismatch")
    _require(math.isclose(rate, FPS, abs_tol=1e-9), f"video fps is {rate}, expected {FPS}")
    _require(stream["codec_name"] == "av1", f"video codec is {stream['codec_name']}, expected av1")
    if expected_frames is not None:
        _require(frames == expected_frames, f"video frame count {frames} != {expected_frames}")
    return {
        "codec": stream["codec_name"],
        "pix_fmt": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": rate,
        "frames": frames,
    }


def _write_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    global_start: int,
    task_index: int,
) -> dict[str, np.ndarray]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise Native50HzContractError("pyarrow is required for LeRobot v2.1 export") from exc
    length = int(state.shape[0])
    _require(state.shape == action.shape == (length, STATE_ACTION_DIM), "state/action shape mismatch")
    timestamp = _native_float32_timestamp_grid(length)
    _require_native_float32_timestamp_grid(timestamp, label="generated timestamp")
    frame_index = np.arange(length, dtype=np.int64)
    episode_indices = np.full(length, episode_index, dtype=np.int64)
    global_indices = np.arange(global_start, global_start + length, dtype=np.int64)
    task_indices = np.full(length, task_index, dtype=np.int64)
    vector_type = pa.list_(pa.float32(), STATE_ACTION_DIM)
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=vector_type),
            "action": pa.array(action.tolist(), type=vector_type),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(global_indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return {
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_indices,
        "index": global_indices,
        "task_index": task_indices,
    }


def _native_float32_timestamp_grid(length: int) -> np.ndarray:
    """Return LeRobot's exact float32 representation of native 50 Hz ticks."""

    _require(int(length) == length and length >= 0, "timestamp length must be non-negative")
    return (np.arange(int(length), dtype=np.float64) / np.float64(FPS)).astype(np.float32)


def _require_native_float32_timestamp_grid(timestamp: np.ndarray, *, label: str) -> None:
    """Fail closed unless every stored timestamp is exactly its native 50 Hz tick.

    A float32 column cannot represent 0.02 exactly.  For longer episodes its
    adjacent differences therefore alternate between nearby values (for
    example 0.01999855 and 0.02000046).  Comparing every difference to a fixed
    absolute tolerance incorrectly rejects valid native samples.  Exact
    comparison against ``float32(i / 50)`` is both stricter and faithful to the
    official LeRobot float32 timestamp schema.
    """

    values = np.asarray(timestamp)
    _require(values.ndim == 1, f"{label} is not one-dimensional")
    expected = _native_float32_timestamp_grid(int(values.shape[0]))
    _require(
        np.array_equal(values, expected),
        f"{label} is not the exact float32 encoding of native 50 Hz ticks",
    )


def _feature_spec(video_info: Mapping[str, Any]) -> dict[str, Any]:
    vector_names = [list(MOTOR_NAMES)]
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": vector_names},
        "action": {"dtype": "float32", "shape": [14], "names": vector_names},
    }
    for camera_key in CAMERA_PATHS:
        features[camera_key] = {
            "dtype": "video",
            "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": IMAGE_HEIGHT,
                "video.width": IMAGE_WIDTH,
                "video.codec": video_info["codec"],
                "video.pix_fmt": video_info["pix_fmt"],
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n")


def export_paired_lerobot_v21(
    raw_task_roots: Mapping[str, str | Path],
    *,
    output_root: str | Path,
    expected_contents: int,
    decode_all_frames_during_raw_audit: bool = True,
) -> dict[str, Any]:
    """Transactionally export three-task four-scene data as LeRobot v2.1."""

    _require(tuple(raw_task_roots) == OFFICIAL_TASKS, f"raw task order must be exactly {OFFICIAL_TASKS}")
    output = Path(output_root).expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite LeRobot root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    _require(not staging.exists(), f"staging path already exists: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    published = False
    try:
        raw_audits = {
            task: validate_raw_task_root(
                raw_task_roots[task],
                expected_task=task,
                expected_contents=expected_contents,
                decode_all_frames=decode_all_frames_during_raw_audit,
                run_legacy_validator=True,
            )
            for task in OFFICIAL_TASKS
        }
        for task_index, task in enumerate(OFFICIAL_TASKS):
            _append_jsonl(
                staging / "meta" / "tasks.jsonl",
                {"task_index": task_index, "task": TASK_INSTRUCTIONS[task]},
            )

        total_frames = 0
        episode_index = 0
        first_video_info: dict[str, Any] | None = None
        paired_rows: list[dict[str, Any]] = []
        for task_index, task in enumerate(OFFICIAL_TASKS):
            raw_root = Path(raw_task_roots[task]).expanduser().resolve()
            for content_id in range(int(expected_contents)):
                content_dir = raw_root / "contents" / f"content_{content_id:06d}"
                variant_episode_ids: dict[str, int] = {}
                variant_lengths: dict[str, int] = {}
                clean_state: np.ndarray | None = None
                clean_action: np.ndarray | None = None
                for scene_variant, directory_name in zip(SCENE_VARIANTS, VARIANT_DIRS, strict=True):
                    variant_dir = content_dir / directory_name
                    hdf5_path = variant_dir / "data" / "episode0.hdf5"
                    try:
                        import h5py
                    except ImportError as exc:  # pragma: no cover
                        raise Native50HzContractError("h5py is required for export") from exc
                    with h5py.File(hdf5_path, "r") as handle:
                        joints = np.asarray(handle["joint_action/vector"][()], dtype=np.float32)
                    _require(joints.ndim == 2 and joints.shape[1] == STATE_ACTION_DIM, "joint shape invalid")
                    # Current observed qpos predicts the next native 50Hz qpos.
                    # This is a one-step temporal shift, never interpolation.
                    state = np.ascontiguousarray(joints[:-1])
                    action = np.ascontiguousarray(joints[1:])
                    length = int(state.shape[0])
                    _require(
                        length >= ACTION_HORIZON + STATE_BANK_ANCHORS_PER_TRAJECTORY,
                        "exported episode cannot provide eight endpoint-safe 33-state anchors",
                    )
                    if clean_state is None:
                        clean_state, clean_action = state, action
                    else:
                        _require(np.array_equal(state, clean_state), f"state differs in {task}/{content_id}/{scene_variant}")
                        _require(np.array_equal(action, clean_action), f"action differs in {task}/{content_id}/{scene_variant}")

                    chunk = episode_index // 1000
                    parquet_path = (
                        staging / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
                    )
                    scalar_arrays = _write_parquet(
                        parquet_path,
                        state=state,
                        action=action,
                        episode_index=episode_index,
                        global_start=total_frames,
                        task_index=task_index,
                    )
                    video_hashes: dict[str, str] = {}
                    for camera_key, raw_path in CAMERA_PATHS.items():
                        frames = _decode_camera_frames(hdf5_path, raw_path, length)
                        video_path = (
                            staging
                            / "videos"
                            / f"chunk-{chunk:03d}"
                            / camera_key
                            / f"episode_{episode_index:06d}.mp4"
                        )
                        video_info = _encode_av1(frames, video_path)
                        if first_video_info is None:
                            first_video_info = video_info
                        video_hashes[camera_key] = sha256_file(video_path)

                    stats = {
                        "observation.state": _numeric_stats(state),
                        "action": _numeric_stats(action),
                        **{key: _numeric_stats(value) for key, value in scalar_arrays.items()},
                    }
                    _append_jsonl(
                        staging / "meta" / "episodes.jsonl",
                        {
                            "episode_index": episode_index,
                            "tasks": [TASK_INSTRUCTIONS[task]],
                            "length": length,
                        },
                    )
                    _append_jsonl(
                        staging / "meta" / "episodes_stats.jsonl",
                        {"episode_index": episode_index, "stats": stats},
                    )
                    variant_episode_ids[scene_variant] = episode_index
                    variant_lengths[scene_variant] = length
                    episode_index += 1
                    total_frames += length

                _require(len(set(variant_lengths.values())) == 1, "four variants have unequal lengths")
                paired_rows.append(
                    {
                        "schema_version": 1,
                        "task": task,
                        "content_id": content_id,
                        "split": _expected_split(content_id),
                        "physical_trajectory_id": f"{task}/content_{content_id:06d}",
                        "scene_variant_episode_indices": variant_episode_ids,
                        "equal_length": next(iter(variant_lengths.values())),
                        "state_action_exactly_equal": True,
                        "source_content_complete_sha256": sha256_file(content_dir / "COMPLETE.json"),
                        "sampling": {
                            "physics_hz": PHYSICS_HZ,
                            "every_physics_steps": SAMPLE_EVERY_PHYSICS_STEPS,
                            "fps": FPS,
                            "interpolation": False,
                        },
                    }
                )

        _require(first_video_info is not None, "export produced no videos")
        for row in paired_rows:
            _append_jsonl(staging / PAIRED_MANIFEST_PATH, row)
        total_episodes = episode_index
        info = {
            "codebase_version": "v2.1",
            "robot_type": "aloha",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(OFFICIAL_TASKS),
            "total_videos": total_episodes * len(CAMERA_PATHS),
            "total_chunks": (total_episodes + 999) // 1000,
            "chunks_size": 1000,
            "fps": FPS,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _feature_spec(first_video_info),
        }
        atomic_write_json(staging / "meta" / "info.json", info)

        policy_metadata = {
            "protocol_id": "policy_native50hz_four_scene_v1",
            "variant_names": list(VARIANT_DIRS),
            "view_count": 4,
            "r3_role": "training_positive",
            "camera_count": 3,
            "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "native_fps": FPS,
            "action_steps": ACTION_HORIZON,
            "action_dim": STATE_ACTION_DIM,
            "temporal_resampling": "none",
            "native_action_targets": True,
            "split": "all",
        }
        policy_groups = []
        for row in paired_rows:
            scene_ids = row["scene_variant_episode_indices"]
            policy_groups.append(
                {
                    "task": row["task"],
                    "content_id": row["content_id"],
                    "split": row["split"],
                    "trajectory_id": row["physical_trajectory_id"],
                    "episode_length": int(row["equal_length"]),
                    "valid_action_anchor_count": int(row["equal_length"]) - ACTION_HORIZON,
                    "episodes": {
                        directory_name: scene_ids[scene_variant]
                        for scene_variant, directory_name in zip(
                            SCENE_VARIANTS, VARIANT_DIRS, strict=True
                        )
                    },
                }
            )
        policy_manifest_path = staging / "meta" / "policy_native_action_manifest.json"
        policy_manifest = {
            "schema": "policy_native_action_manifest_v1",
            "schema_version": 1,
            **policy_metadata,
            "dataset_root": str(output),
            "groups": policy_groups,
        }
        atomic_write_json(policy_manifest_path, policy_manifest)
        policy_manifest_sha = sha256_file(policy_manifest_path)
        policy_audit_path = staging / "meta" / "policy_native_action_audit.json"
        atomic_write_json(
            policy_audit_path,
            {
                "status": "PASS",
                **policy_metadata,
                "dataset_root": str(output),
                "manifest_sha256": policy_manifest_sha,
                "checks": {
                    "three_camera_sync": True,
                    "native_50hz": True,
                    "action_window_32x14": True,
                    "state_window_33x14": True,
                    "cross_scene_state_exact": True,
                    "cross_scene_action_exact": True,
                    "temporal_resampling_absent": True,
                    "endpoint_safe_state_bank_supported": True,
                },
                "source_raw_audits": raw_audits,
            },
        )
        policy_audit_sha = sha256_file(policy_audit_path)
        from experiments.robotwin.policy_content_adapter.data import (
            PolicyPhysicalStateAnchor,
            physical_state_inventory_sha256,
            policy_state_bank_offsets,
        )
        from experiments.robotwin.policy_content_adapter.protocol import (
            POLICY_STATE_BANK_SAMPLING_ALGORITHM,
            POLICY_STATE_BANK_SAMPLING_VERSION,
            POLICY_STATE_BANK_SCHEMA,
            POLICY_STATE_BANK_SCHEMA_VERSION,
            POLICY_STATE_BANK_SEED,
            POLICY_STATES_PER_TRAJECTORY,
        )

        anchors: list[PolicyPhysicalStateAnchor] = []
        for group in policy_groups:
            if group["split"] != "train":
                continue
            for frame_offset in policy_state_bank_offsets(
                task=group["task"],
                content_id=group["content_id"],
                episode_length=group["episode_length"],
                seed=POLICY_STATE_BANK_SEED,
            ):
                anchors.append(
                    PolicyPhysicalStateAnchor(
                        task=group["task"],
                        content_id=group["content_id"],
                        trajectory_id=group["trajectory_id"],
                        frame_offset=frame_offset,
                    )
                )
        _require(bool(anchors), "export produced no train state-bank anchors")
        state_bank_path = staging / "meta" / "policy_paired_state_bank.json"
        atomic_write_json(
            state_bank_path,
            {
                "schema": POLICY_STATE_BANK_SCHEMA,
                "schema_version": POLICY_STATE_BANK_SCHEMA_VERSION,
                **{**policy_metadata, "split": "train"},
                "paired_action_manifest_sha256": policy_manifest_sha,
                "paired_action_audit_sha256": policy_audit_sha,
                "sampling": {
                    "algorithm": POLICY_STATE_BANK_SAMPLING_ALGORITHM,
                    "version": POLICY_STATE_BANK_SAMPLING_VERSION,
                    "seed": POLICY_STATE_BANK_SEED,
                    "states_per_trajectory": POLICY_STATES_PER_TRAJECTORY,
                    "endpoint_rule": "33_state_frames_and_32_actions_without_padding",
                    "short_trajectory_policy": "fail_closed",
                },
                "states": [
                    {**anchor.as_dict(), "physical_state_id": anchor.physical_state_id}
                    for anchor in anchors
                ],
                "physical_state_inventory_sha256": physical_state_inventory_sha256(anchors),
            },
        )
        atomic_write_json(
            staging / "meta" / "native50hz_export_contract.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "interpolation": "forbidden_and_not_used",
                "source_raw_audits": raw_audits,
                "paired_manifest_sha256": sha256_file(staging / PAIRED_MANIFEST_PATH),
            },
        )
        export_audit = validate_lerobot_v21_root(
            staging,
            expected_contents=expected_contents,
            require_output_name=False,
            validate_policy_consumer_contract=False,
        )
        atomic_write_json(staging / "meta" / "export_audit.json", export_audit)
        os.replace(staging, output)
        published = True
        from experiments.robotwin.policy_content_adapter.data import (
            audit_native_paired_action_contract,
            verify_native_paired_action_manifest,
            verify_policy_state_bank,
        )

        policy_contract_audit = audit_native_paired_action_contract(
            dataset_root=output,
            manifest_path=output / "meta" / "policy_native_action_manifest.json",
            audit_path=output / "meta" / "policy_native_action_audit.json",
            expected_tasks=OFFICIAL_TASKS,
            require_full_protocol_counts=int(expected_contents) == 50,
        )
        verified_native = verify_native_paired_action_manifest(
            output / "meta" / "policy_native_action_manifest.json",
            dataset_root=output,
            audit_path=output / "meta" / "policy_native_action_audit.json",
        )
        verified_state_bank = verify_policy_state_bank(
            output / "meta" / "policy_paired_state_bank.json",
            native_manifest=verified_native,
            expected_tasks=OFFICIAL_TASKS,
        )
        export_audit["root"] = str(output)
        export_audit["policy_native_action_contract"] = policy_contract_audit
        export_audit["policy_state_bank"] = {
            "path": str(verified_state_bank.path),
            "sha256": verified_state_bank.sha256,
            "physical_state_inventory_sha256": (
                verified_state_bank.physical_state_inventory_sha256
            ),
            "anchor_count": len(verified_state_bank.anchors),
        }
        return export_audit
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if published and output.exists():
            # ``output`` was created transactionally by this invocation and
            # has not been returned to the caller.  Roll it back if the final
            # consumer-contract audit fails, so a rerun cannot mistake it for
            # a valid existing export.
            shutil.rmtree(output)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"JSONL file is missing: {path}")
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            try:
                value = json.loads(line)
            except Exception as exc:
                raise Native50HzContractError(f"invalid JSONL {path}:{line_index + 1}: {exc}") from exc
            _require(isinstance(value, dict), f"JSONL record is not an object: {path}:{line_index + 1}")
            values.append(value)
    return values


def _read_parquet_vectors(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise Native50HzContractError("pyarrow is required for LeRobot validation") from exc
    table = pq.read_table(
        path,
        columns=["observation.state", "action", "timestamp", "frame_index"],
    )
    _require(
        pa.types.is_float32(table.schema.field("timestamp").type),
        f"parquet timestamp physical type is not float32: {path}",
    )
    _require(
        pa.types.is_int64(table.schema.field("frame_index").type),
        f"parquet frame_index physical type is not int64: {path}",
    )
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float32)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    return state, action, timestamp, frame_index


def validate_lerobot_v21_root(
    root: str | Path,
    *,
    expected_contents: int,
    require_output_name: bool = False,
    validate_policy_consumer_contract: bool = True,
) -> dict[str, Any]:
    """Strictly validate the exported LeRobot root and paired manifest."""

    dataset_root = Path(root).expanduser().resolve()
    _require(dataset_root.is_dir(), f"LeRobot root not found: {dataset_root}")
    info = _read_json(dataset_root / "meta" / "info.json")
    expected_episodes = len(OFFICIAL_TASKS) * int(expected_contents) * len(SCENE_VARIANTS)
    _require(info.get("codebase_version") == "v2.1", "LeRobot codebase_version must be v2.1")
    _require(info.get("fps") == FPS, "LeRobot fps must be 50")
    _require(info.get("total_episodes") == expected_episodes, "LeRobot episode count mismatch")
    _require(info.get("total_videos") == expected_episodes * 3, "LeRobot video count mismatch")
    features = info.get("features")
    _require(isinstance(features, Mapping), "LeRobot features mapping is missing")
    required_features = {
        "observation.state",
        "action",
        *CAMERA_PATHS.keys(),
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    _require(set(features) == required_features, "LeRobot feature keys differ from official contract")
    _require(features["observation.state"]["shape"] == [14], "state dimension is not 14")
    _require(features["action"]["shape"] == [14], "action dimension is not 14")
    _require(features["timestamp"]["dtype"] == "float32", "timestamp metadata is not float32")
    _require(features["frame_index"]["dtype"] == "int64", "frame_index metadata is not int64")
    for camera_key in CAMERA_PATHS:
        _require(features[camera_key]["shape"] == [480, 640, 3], f"camera shape mismatch: {camera_key}")
        _require(features[camera_key]["dtype"] == "video", f"camera is not video: {camera_key}")
        _require(
            features[camera_key].get("info", {}).get("video.codec") == "av1",
            f"camera codec is not official-compatible AV1: {camera_key}",
        )

    tasks = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    _require(
        tasks
        == [
            {"task_index": index, "task": TASK_INSTRUCTIONS[task]}
            for index, task in enumerate(OFFICIAL_TASKS)
        ],
        "LeRobot task metadata differs from deterministic three-task contract",
    )
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    episode_stats = _read_jsonl(dataset_root / "meta" / "episodes_stats.jsonl")
    _require(len(episodes) == len(episode_stats) == expected_episodes, "episode metadata count mismatch")
    paired = _read_jsonl(dataset_root / PAIRED_MANIFEST_PATH)
    _require(len(paired) == len(OFFICIAL_TASKS) * int(expected_contents), "paired content count mismatch")

    parquet_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    minimum_length: int | None = None
    for expected_episode, record in enumerate(episodes):
        _require(record.get("episode_index") == expected_episode, "episode indices are not contiguous")
        length = int(record.get("length", -1))
        _require(
            length >= ACTION_HORIZON + STATE_BANK_ANCHORS_PER_TRAJECTORY,
            f"episode {expected_episode} cannot provide eight endpoint-safe 33-state anchors",
        )
        minimum_length = length if minimum_length is None else min(minimum_length, length)
        chunk = expected_episode // 1000
        parquet_path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{expected_episode:06d}.parquet"
        state, action, timestamp, frame_index = _read_parquet_vectors(parquet_path)
        _require(state.shape == action.shape == (length, STATE_ACTION_DIM), "parquet vector shape mismatch")
        _require(len(timestamp) == length, "parquet timestamp length mismatch")
        _require(
            np.array_equal(frame_index, np.arange(length, dtype=np.int64)),
            f"episode {expected_episode} frame_index is not contiguous from zero",
        )
        _require(np.isclose(timestamp[0], 0.0, atol=1e-8), "episode timestamp does not start at zero")
        _require_native_float32_timestamp_grid(
            timestamp,
            label=f"episode {expected_episode} timestamp",
        )
        parquet_cache[expected_episode] = (state, action, timestamp)
        for camera_key in CAMERA_PATHS:
            video_path = dataset_root / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{expected_episode:06d}.mp4"
            _probe_video(video_path, expected_frames=length)

    for row_index, row in enumerate(paired):
        task = OFFICIAL_TASKS[row_index // int(expected_contents)]
        content_id = row_index % int(expected_contents)
        _require(row.get("task") == task and row.get("content_id") == content_id, "paired manifest ordering mismatch")
        _require(row.get("split") == _expected_split(content_id), "paired manifest split mismatch")
        episode_map = row.get("scene_variant_episode_indices")
        _require(isinstance(episode_map, Mapping), "paired episode map is missing")
        _require(tuple(episode_map) == SCENE_VARIANTS, "paired scene variant order mismatch")
        ids = [int(episode_map[key]) for key in SCENE_VARIANTS]
        clean_state, clean_action, _ = parquet_cache[ids[0]]
        for episode_id in ids[1:]:
            state, action, _ = parquet_cache[episode_id]
            _require(np.array_equal(state, clean_state), "paired exported states differ")
            _require(np.array_equal(action, clean_action), "paired exported actions differ")
        _require(int(row.get("equal_length", -1)) == len(clean_state), "paired length declaration mismatch")
        _require(row.get("state_action_exactly_equal") is True, "paired equality declaration is false")
        sampling = row.get("sampling")
        _require(isinstance(sampling, Mapping) and sampling.get("interpolation") is False, "paired interpolation declaration invalid")

    if int(expected_contents) == 50:
        split_counts = {
            split: sum(row["split"] == split for row in paired) // len(OFFICIAL_TASKS)
            for split in SPLIT_COUNTS
        }
        _require(split_counts == SPLIT_COUNTS, "LeRobot physical split is not 30/10/10")
    result = {
        "status": "PASS",
        "root": str(dataset_root),
        "codebase_version": "v2.1",
        "fps": FPS,
        "timestamp_delta_seconds": TIMESTAMP_DELTA_SECONDS,
        "state_dim": STATE_ACTION_DIM,
        "action_dim": STATE_ACTION_DIM,
        "camera_keys": list(CAMERA_PATHS),
        "episode_count": expected_episodes,
        "content_count_per_task": int(expected_contents),
        "minimum_episode_length": minimum_length,
        "future_action_horizon": ACTION_HORIZON,
        "all_pairs_exact": True,
        "interpolation_used": False,
        "paired_manifest_sha256": sha256_file(dataset_root / PAIRED_MANIFEST_PATH),
    }
    if validate_policy_consumer_contract:
        from experiments.robotwin.policy_content_adapter.data import (
            audit_native_paired_action_contract,
            verify_native_paired_action_manifest,
            verify_policy_state_bank,
        )

        result["policy_native_action_contract"] = audit_native_paired_action_contract(
            dataset_root=dataset_root,
            manifest_path=dataset_root / "meta" / "policy_native_action_manifest.json",
            audit_path=dataset_root / "meta" / "policy_native_action_audit.json",
            expected_tasks=OFFICIAL_TASKS,
            require_full_protocol_counts=int(expected_contents) == 50,
        )
        native_manifest = verify_native_paired_action_manifest(
            dataset_root / "meta" / "policy_native_action_manifest.json",
            dataset_root=dataset_root,
            audit_path=dataset_root / "meta" / "policy_native_action_audit.json",
        )
        state_bank = verify_policy_state_bank(
            dataset_root / "meta" / "policy_paired_state_bank.json",
            native_manifest=native_manifest,
            expected_tasks=OFFICIAL_TASKS,
        )
        result["policy_state_bank"] = {
            "path": str(state_bank.path),
            "sha256": state_bank.sha256,
            "physical_state_inventory_sha256": state_bank.physical_state_inventory_sha256,
            "anchor_count": len(state_bank.anchors),
        }
    return result


__all__ = [
    "ACTION_HORIZON",
    "CAMERA_PATHS",
    "CONTRACT_FILENAME",
    "DEFAULT_COLLECTION_CONFIG",
    "FPS",
    "Native50HzContractError",
    "PHYSICS_HZ",
    "SAMPLE_EVERY_PHYSICS_STEPS",
    "SCENE_VARIANTS",
    "STATE_BANK_ANCHORS_PER_TRAJECTORY",
    "TIMESTAMP_DELTA_SECONDS",
    "VARIANT_DIRS",
    "atomic_write_json",
    "collection_contract_value",
    "export_paired_lerobot_v21",
    "sha256_file",
    "validate_collection_config",
    "validate_lerobot_v21_root",
    "validate_raw_content",
    "validate_raw_task_root",
]
