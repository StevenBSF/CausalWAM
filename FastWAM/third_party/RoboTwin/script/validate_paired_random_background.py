#!/usr/bin/env python3
"""Strict validator for paired RoboTwin random-background episodes.

The validator deliberately uses exact comparisons.  It never applies a numeric
tolerance and it never repairs or rewrites an episode.  The only writes made by
the command-line aggregate are the explicitly selected JSON report and JSONL
manifest.

``validate_content_dir`` is also used by the generator before a staged content
directory is published, so ``COMPLETE.json`` is optional by default.  Aggregate
validation requires that marker for every published content directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paired_task_adapters import (  # noqa: E402
    RIGID_STATE_FIELDS,
    SPLIT_COUNTS,
    SUPPORTED_TASKS,
    TOTAL_CONTENTS,
    TaskAdapterError,
    canonical_json_sha256,
    derive_success,
    split_for_content,
    validate_success_spec,
    validate_task_identity,
)


SCHEMA_VERSION = 2
DEFAULT_TASK = "grab_roller"
DEFAULT_STYLE_SEEDS = (0, 1, 2)
REQUIRED_VARIANT_FILES = (
    "metadata.json",
    "initial_state.npz",
    "action_trace.npz",
    "state_trace.npz",
    "data/episode0.hdf5",
    "video/episode0.mp4",
)
SOURCE_TRAJECTORY = "_traj_data/episode0.pkl"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CONTENT_RE = re.compile(r"^content_(\d{6})$")
STYLE_RE = re.compile(r"^style_(\d{2})_seed_(-?\d+)$")
PCI_ADDRESS_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
ACTION_TRACE_KEY_ORDER = (
    "semantic_action",
    "left_drive_target",
    "right_drive_target",
    "left_drive_velocity",
    "right_drive_velocity",
)
STATE_TRACE_KEY_ORDER = (
    "left_qpos",
    "right_qpos",
    "left_qvel",
    "right_qvel",
    "left_eef",
    "right_eef",
    "task_state",
    "semantic_action",
    "left_gripper_open",
    "right_gripper_open",
    "left_gripper_closed",
    "right_gripper_closed",
    "frame_trace_index",
)
ACTION_TRACE_KEYS = frozenset(ACTION_TRACE_KEY_ORDER)
STATE_TRACE_KEYS = frozenset(STATE_TRACE_KEY_ORDER)
INITIAL_STATE_KEYS = STATE_TRACE_KEYS
TRACE_TO_HDF5 = {
    "left_qpos": "robot_state/left_qpos",
    "right_qpos": "robot_state/right_qpos",
    "left_qvel": "robot_state/left_qvel",
    "right_qvel": "robot_state/right_qvel",
    "left_eef": "endpose/left_endpose",
    "right_eef": "endpose/right_endpose",
}
_MISSING = object()


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _json_value(value: Any) -> Any:
    """Return a JSON-safe diagnostic value without dumping large arrays."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _issue(
    target: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    variant: str | None = None,
    artifact: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {"code": code, "message": message}
    if variant is not None:
        item["variant"] = variant
    if artifact is not None:
        item["artifact"] = artifact
    if details:
        item["details"] = _json_value(details)
    target.append(item)


def _walk_nodes(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if prefix:
        yield prefix, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_nodes(child, prefix + (str(key),))


def _find_value(metadata: Mapping[str, Any], aliases: Sequence[str], default: Any = _MISSING) -> Any:
    """Find a metadata value while allowing sensible nesting/key punctuation."""
    normalised_aliases = [_normalise_key(alias) for alias in aliases]
    best: tuple[int, int, Any] | None = None
    for path, value in _walk_nodes(metadata):
        full = _normalise_key(".".join(path))
        leaf = _normalise_key(path[-1])
        for order, alias in enumerate(normalised_aliases):
            score = -1
            if full == alias:
                score = 500
            elif leaf == alias:
                score = 400
            elif full.endswith(alias):
                score = 300
            elif alias.endswith(full):
                score = 200
            if score >= 0:
                candidate = (score, -order, value)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    return default if best is None else best[2]


def _find_mapping(metadata: Mapping[str, Any], aliases: Sequence[str]) -> Mapping[str, Any] | None:
    value = _find_value(metadata, aliases)
    return value if isinstance(value, Mapping) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return None


def _as_json_int(value: Any) -> int | None:
    """Accept only a JSON integer, not a boolean or numeric string alias."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "enabled", "1"}:
            return True
        if lowered in {"false", "no", "off", "disabled", "none", "0"}:
            return False
    return None


def _setting_enabled(value: Any) -> bool:
    parsed = _as_bool(value)
    if parsed is not None:
        return parsed
    if value is None:
        return False
    if isinstance(value, (float, np.floating, int, np.integer)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null", "false", "off", "disabled", "0"}
    if isinstance(value, Mapping):
        return any(_setting_enabled(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return any(_setting_enabled(child) for child in value)
    return bool(value)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("top-level JSON value must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_array_bytes(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.strip()) is not None


HASH_ALIASES: dict[str, tuple[str, ...]] = {
    "initial_state": ("initial_state", "initial_state.npz", "initialstate", "initialstatenpz"),
    "action_trace": ("action_trace", "action_trace.npz", "actiontrace", "actiontracenpz"),
    "state_trace": ("state_trace", "state_trace.npz", "statetrace", "statetracenpz"),
    "hdf5": ("hdf5", "episode_hdf5", "data/episode0.hdf5", "episode0.hdf5", "episode0hdf5"),
    "video": ("video", "episode_video", "video/episode0.mp4", "episode0.mp4", "episode0mp4"),
    "source_trajectory": (
        "source_trajectory",
        "trajectory_pickle",
        "_traj_data/episode0.pkl",
        "episode0.pkl",
        "sourcetrajectory",
    ),
    "head_rgb": ("head_rgb", "head_camera_rgb", "headrgb", "headcamerargb"),
}


def _declared_hash(metadata: Mapping[str, Any], kind: str, *, source: bool = False) -> Any:
    aliases = HASH_ALIASES[kind]
    containers: list[Mapping[str, Any]] = []
    if source:
        source_map = _find_mapping(
            metadata,
            ("source_hashes", "source_hash", "source.clean_hashes", "source.artifact_hashes"),
        )
        if source_map is not None:
            containers.append(source_map)
    else:
        hashes = _find_mapping(metadata, ("hashes", "artifact_hashes", "sha256", "artifact_sha256"))
        if hashes is not None:
            containers.append(hashes)
    for container in containers:
        found = _find_value(container, aliases)
        if found is not _MISSING:
            if isinstance(found, Mapping):
                nested = _find_value(found, ("sha256", "hash", "digest"))
                if nested is not _MISSING:
                    return nested
            return found
    suffix_aliases: list[str] = []
    for alias in aliases:
        suffix_aliases.extend((f"{alias}_sha256", f"{alias}_hash", f"{alias}_digest"))
    if source:
        suffix_aliases.extend(f"source_{alias}_sha256" for alias in aliases)
    return _find_value(metadata, tuple(suffix_aliases))


def _require_declared_hash(
    metadata: Mapping[str, Any],
    kind: str,
    actual: str,
    errors: list[dict[str, Any]],
    variant: str,
    *,
    source: bool = False,
) -> None:
    declared = _declared_hash(metadata, kind, source=source)
    label = f"source {kind}" if source else kind
    if declared is _MISSING:
        _issue(
            errors,
            "metadata_hash_missing",
            f"metadata does not declare a SHA-256 for {label}",
            variant=variant,
            artifact="metadata.json",
        )
        return
    if not _valid_hash(declared):
        _issue(
            errors,
            "metadata_hash_invalid",
            f"metadata SHA-256 for {label} is not 64 hexadecimal characters",
            variant=variant,
            artifact="metadata.json",
            details={"declared": declared},
        )
        return
    if str(declared).lower() != actual.lower():
        _issue(
            errors,
            "metadata_hash_mismatch",
            f"metadata SHA-256 for {label} does not match the bytes on disk",
            variant=variant,
            artifact="metadata.json",
            details={"declared": str(declared).lower(), "actual": actual.lower()},
        )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    # The explicit flag is important: an object array must make validation fail,
    # rather than causing arbitrary pickle loading through NPZ.
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("NPZ contains no arrays")
        if len(set(archive.files)) != len(archive.files):
            raise ValueError("NPZ contains duplicate array names")
        for key in archive.files:
            array = np.asarray(archive[key])
            if array.dtype.hasobject:
                raise ValueError(f"array {key!r} has object dtype")
            arrays[str(key)] = array
    return arrays


def _exact_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _numeric_diff(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "numeric_comparable": False,
    }
    if left.shape != right.shape:
        return result
    if left.dtype.kind not in "buifc" or right.dtype.kind not in "buifc":
        return result
    result["numeric_comparable"] = True
    if left.size == 0:
        result.update({"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "mismatch_count": 0, "element_count": 0})
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(left.astype(np.complex128 if left.dtype.kind == "c" else np.float64)
                            - right.astype(np.complex128 if right.dtype.kind == "c" else np.float64))
    finite = np.isfinite(difference)
    if np.any(finite):
        result["max_abs_diff"] = float(np.max(difference[finite]))
        result["mean_abs_diff"] = float(np.mean(difference[finite], dtype=np.float64))
    if np.any(~finite):
        result["non_finite_diff_count"] = int(np.count_nonzero(~finite))
    try:
        unequal = np.not_equal(left, right)
        result["mismatch_count"] = int(np.count_nonzero(unequal))
        result["element_count"] = int(left.size)
    except (TypeError, ValueError):
        pass
    return result


def _array_mismatch_details(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    details: dict[str, Any] = {
        "clean_shape": list(left.shape),
        "variant_shape": list(right.shape),
        "clean_dtype": str(left.dtype),
        "variant_dtype": str(right.dtype),
        "byte_identical": False,
    }
    details.update(_numeric_diff(left, right))
    return details


def _compare_array_mappings(
    clean: Mapping[str, np.ndarray],
    variant: Mapping[str, np.ndarray],
    *,
    artifact: str,
    variant_name: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_keys = set(clean)
    variant_keys = set(variant)
    comparison: dict[str, Any] = {
        "variant": variant_name,
        "artifact": artifact,
        "equal": True,
        "checked_arrays": 0,
        "mismatches": [],
    }
    missing = sorted(clean_keys - variant_keys)
    extra = sorted(variant_keys - clean_keys)
    if missing or extra:
        comparison["equal"] = False
        structural = {"missing_arrays": missing, "extra_arrays": extra}
        comparison["mismatches"].append(structural)
        _issue(
            errors,
            "array_keys_mismatch",
            f"{artifact} array names differ from clean",
            variant=variant_name,
            artifact=artifact,
            details=structural,
        )
    for key in sorted(clean_keys & variant_keys):
        comparison["checked_arrays"] += 1
        if _exact_array_equal(clean[key], variant[key]):
            continue
        comparison["equal"] = False
        details = {"array": key, **_array_mismatch_details(clean[key], variant[key])}
        comparison["mismatches"].append(details)
        _issue(
            errors,
            "array_exact_mismatch",
            f"{artifact}:{key} is not byte-for-byte identical to clean",
            variant=variant_name,
            artifact=artifact,
            details=details,
        )
    return comparison


def _is_rgb_leaf(path: str) -> bool:
    return any("rgb" in _normalise_key(component) for component in path.split("/"))


def _hdf5_inventory(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("h5py is required to validate episode HDF5 files") from exc

    leaves: dict[str, dict[str, Any]] = {}
    head_candidates: list[str] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                leaves[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
                normalised = _normalise_key(name)
                if "headcamera" in normalised and "rgb" in _normalise_key(name.split("/")[-1]):
                    head_candidates.append(name)

        handle.visititems(visitor)
        if not leaves:
            raise ValueError("HDF5 contains no datasets")
        if not head_candidates:
            raise ValueError("HDF5 has no head-camera RGB dataset")
        head_path = sorted(head_candidates, key=lambda name: (name.count("/"), len(name), name))[0]
        head_array = np.asarray(handle[head_path][()])
        head_hash = _sha256_array_bytes(head_array)
        head_shape = list(head_array.shape)

    counts = {name: int(info["shape"][0]) for name, info in leaves.items() if info["shape"]}
    unique_counts = sorted(set(counts.values()))
    frame_count = unique_counts[0] if len(unique_counts) == 1 else None
    non_rgb = sorted(name for name in leaves if not _is_rgb_leaf(name))
    return {
        "leaves": leaves,
        "non_rgb_leaves": non_rgb,
        "head_rgb_path": head_path,
        "head_rgb_sha256": head_hash,
        "head_rgb_shape": head_shape,
        "frame_count": frame_count,
        "dataset_frame_counts": counts,
        "frame_count_values": unique_counts,
    }


def _compare_hdf5_non_rgb(
    clean_path: Path,
    variant_path: Path,
    clean_inventory: Mapping[str, Any],
    variant_inventory: Mapping[str, Any],
    variant_name: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required to compare HDF5 files") from exc

    clean_names = set(clean_inventory["non_rgb_leaves"])
    variant_names = set(variant_inventory["non_rgb_leaves"])
    comparison: dict[str, Any] = {
        "variant": variant_name,
        "artifact": "data/episode0.hdf5 (non-RGB)",
        "equal": True,
        "checked_arrays": 0,
        "mismatches": [],
    }
    missing = sorted(clean_names - variant_names)
    extra = sorted(variant_names - clean_names)
    if missing or extra:
        comparison["equal"] = False
        details = {"missing_leaves": missing, "extra_leaves": extra}
        comparison["mismatches"].append(details)
        _issue(
            errors,
            "hdf5_structure_mismatch",
            "non-RGB HDF5 leaf names differ from clean",
            variant=variant_name,
            artifact="data/episode0.hdf5",
            details=details,
        )
    with h5py.File(clean_path, "r") as clean_handle, h5py.File(variant_path, "r") as variant_handle:
        for name in sorted(clean_names & variant_names):
            comparison["checked_arrays"] += 1
            clean_array = np.asarray(clean_handle[name][()])
            variant_array = np.asarray(variant_handle[name][()])
            if _exact_array_equal(clean_array, variant_array):
                continue
            comparison["equal"] = False
            details = {"leaf": name, **_array_mismatch_details(clean_array, variant_array)}
            comparison["mismatches"].append(details)
            _issue(
                errors,
                "hdf5_exact_mismatch",
                f"non-RGB HDF5 leaf {name!r} is not byte-for-byte identical to clean",
                variant=variant_name,
                artifact="data/episode0.hdf5",
                details=details,
            )
    return comparison


def _video_frame_count(path: Path) -> int:
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("video cannot be opened")
        count = 0
        try:
            while capture.grab():
                count += 1
        finally:
            capture.release()
        if count <= 0:
            raise ValueError("video contains no decodable frames")
        return count

    # The lightweight validation environment need not have Python OpenCV even
    # though RoboTwin's collection environment does.  ffprobe's count_frames
    # decodes/counts the selected stream and provides an equally strict,
    # read-only fallback.
    try:
        completed = subprocess.run(
            (
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("neither OpenCV nor a working ffprobe can count video frames") from exc
    output = completed.stdout.strip().splitlines()
    if len(output) != 1 or not output[0].isdigit() or int(output[0]) <= 0:
        raise ValueError(f"ffprobe returned an invalid decoded frame count: {completed.stdout!r}")
    return int(output[0])


def _load_source_trajectory(path: Path) -> tuple[Mapping[str, Any], dict[str, int]]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
        trailing = handle.read()
    if trailing:
        raise ValueError("trajectory pickle has trailing bytes after the first object")
    if not isinstance(value, Mapping):
        raise ValueError("trajectory pickle must contain a mapping")
    counts: dict[str, int] = {}
    for arm, key in (("left", "left_joint_path"), ("right", "right_joint_path")):
        paths = value.get(key)
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes, bytearray)):
            raise ValueError(f"trajectory {key!r} must be a sequence")
        counts[arm] = len(paths)
        for index, entry in enumerate(paths):
            if not isinstance(entry, Mapping):
                raise ValueError(f"trajectory {key}[{index}] is not a mapping")
            status = entry.get("status")
            if status != "Success":
                raise ValueError(f"trajectory {key}[{index}] status is {status!r}, not 'Success'")
            position = entry.get("position")
            if not isinstance(position, np.ndarray) or position.size == 0:
                raise ValueError(f"trajectory {key}[{index}] has no non-empty position array")
            for entry_key, entry_value in entry.items():
                if isinstance(entry_value, np.ndarray):
                    if entry_value.dtype.hasobject:
                        raise ValueError(f"trajectory {key}[{index}].{entry_key} has object dtype")
                    if entry_value.dtype.kind in "fc" and not np.all(np.isfinite(entry_value)):
                        raise ValueError(f"trajectory {key}[{index}].{entry_key} contains non-finite values")
    if sum(counts.values()) == 0:
        raise ValueError("trajectory must contain at least one left/right joint-path entry")
    return value, counts


def _validate_task_path_arms(
    task_name: str,
    source_counts: Mapping[str, int],
    initial_state: Mapping[str, np.ndarray] | None,
    task_state_layout: Sequence[str] | None,
    *,
    errors: list[dict[str, Any]],
) -> None:
    """Bind source path occupancy to the arm selected by each task policy."""

    expected_active: set[str] | None = None
    if task_name == "grab_roller":
        expected_active = {"left", "right"}
    elif task_name == "open_microwave":
        expected_active = {"left"}
    elif task_name in {"place_a2b_left", "move_stapler_pad"}:
        position_key = (
            "object_A_px" if task_name == "place_a2b_left" else "stapler_px"
        )
        task_state = None if initial_state is None else initial_state.get("task_state")
        if (
            task_state_layout is None
            or position_key not in task_state_layout
            or task_state is None
            or task_state.ndim != 2
            or task_state.shape[0] != 1
            or task_state.shape[1] != len(task_state_layout)
        ):
            _issue(
                errors,
                "task_active_arm_unavailable",
                f"cannot derive the active arm for {task_name} from initial task_state",
                artifact="initial_state.npz",
            )
            return
        px = float(task_state[0, task_state_layout.index(position_key)])
        expected_active = {"right" if px > 0 else "left"}

    if expected_active is None:
        return
    actual_active = {arm for arm in ("left", "right") if source_counts[arm] > 0}
    if actual_active != expected_active:
        _issue(
            errors,
            "task_source_active_arms_mismatch",
            "source trajectory arm occupancy does not match the task's deterministic arm policy",
            artifact=SOURCE_TRAJECTORY,
            details={
                "task": task_name,
                "expected_active_arms": sorted(expected_active),
                "actual_active_arms": sorted(actual_active),
                "source_counts": source_counts,
            },
        )


def _metadata_error(
    errors: list[dict[str, Any]],
    variant: str,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    _issue(errors, code, message, variant=variant, artifact="metadata.json", details=details)


def _validate_render_device_metadata(
    metadata: Mapping[str, Any],
    *,
    variant: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate the exact physical-render-device provenance contract."""

    raw = metadata.get("render_device", _MISSING)
    if not isinstance(raw, Mapping):
        _metadata_error(
            errors,
            variant,
            "canonical_render_device_missing",
            "metadata render_device must be the canonical device mapping",
            {"value": None if raw is _MISSING else raw},
        )
        return None

    expected_keys = {
        "requested_alias",
        "physical_gpu_index",
        "cuda_visible_devices",
        "name",
        "logical_cuda_id",
        "pci_string",
        "can_render",
    }
    if set(raw) != expected_keys:
        _metadata_error(
            errors,
            variant,
            "canonical_render_device_layout_invalid",
            "metadata render_device keys are not canonical",
            {
                "expected": sorted(expected_keys),
                "actual": sorted(str(key) for key in raw),
            },
        )

    requested_alias = raw.get("requested_alias", _MISSING)
    pci_string = raw.get("pci_string", _MISSING)
    expected_alias = (
        f"pci:{pci_string}"
        if isinstance(pci_string, str) and PCI_ADDRESS_RE.fullmatch(pci_string)
        else None
    )
    if requested_alias != expected_alias:
        _metadata_error(
            errors,
            variant,
            "canonical_render_device_alias_invalid",
            "render_device requested_alias must exactly bind its canonical PCI address",
            {
                "value": None if requested_alias is _MISSING else requested_alias,
                "expected": expected_alias,
            },
        )

    cuda_visible_devices = raw.get("cuda_visible_devices", _MISSING)
    visible_gpu_index: int | None = None
    if not isinstance(cuda_visible_devices, str) or re.fullmatch(
        r"[0-9]+", cuda_visible_devices
    ) is None:
        _metadata_error(
            errors,
            variant,
            "canonical_cuda_visible_devices_invalid",
            "render_device cuda_visible_devices must be a single numeric GPU index",
            {
                "value": None
                if cuda_visible_devices is _MISSING
                else cuda_visible_devices
            },
        )
    else:
        visible_gpu_index = int(cuda_visible_devices)

    physical_gpu_index = _as_json_int(raw.get("physical_gpu_index", _MISSING))
    if physical_gpu_index is None or physical_gpu_index < 0:
        _metadata_error(
            errors,
            variant,
            "canonical_physical_gpu_index_invalid",
            "render_device physical_gpu_index must be a non-negative JSON integer",
            {"value": raw.get("physical_gpu_index")},
        )
    elif visible_gpu_index is not None and physical_gpu_index != visible_gpu_index:
        _metadata_error(
            errors,
            variant,
            "render_device_physical_gpu_mismatch",
            "render_device physical_gpu_index must match cuda_visible_devices",
            {
                "physical_gpu_index": physical_gpu_index,
                "cuda_visible_devices": cuda_visible_devices,
            },
        )

    logical_cuda_id = _as_json_int(raw.get("logical_cuda_id", _MISSING))
    if logical_cuda_id != 0:
        _metadata_error(
            errors,
            variant,
            "canonical_logical_cuda_id_invalid",
            "render_device logical_cuda_id must be the JSON integer 0",
            {"value": raw.get("logical_cuda_id")},
        )

    if not isinstance(pci_string, str) or PCI_ADDRESS_RE.fullmatch(pci_string) is None:
        _metadata_error(
            errors,
            variant,
            "canonical_render_device_pci_invalid",
            "render_device pci_string must be a canonical lowercase PCI address",
            {"value": None if pci_string is _MISSING else pci_string},
        )

    name = raw.get("name", _MISSING)
    if not isinstance(name, str) or not name.strip():
        _metadata_error(
            errors,
            variant,
            "canonical_render_device_name_invalid",
            "render_device name must be a non-empty string",
            {"value": None if name is _MISSING else name},
        )

    can_render = raw.get("can_render", _MISSING)
    if can_render is not True:
        _metadata_error(
            errors,
            variant,
            "canonical_render_device_can_render_invalid",
            "render_device can_render must be the boolean true",
            {"value": None if can_render is _MISSING else can_render},
        )

    # Preserve the exact JSON values in the per-variant summary.  Besides
    # making reports auditable, this allows the cross-variant phase to require
    # all four replays of one content trajectory to use the same physical GPU.
    return {str(key): value for key, value in raw.items()}


def _canonical_task_state_layout(task_name: str, value: Any) -> tuple[list[str] | None, list[str]]:
    """Validate the exact adapter-defined column order for one task."""

    if not isinstance(value, list) or any(not isinstance(field, str) for field in value):
        return None, ["task_state_layout must be a JSON list of strings"]
    layout = list(value)
    if not layout or len(set(layout)) != len(layout):
        return layout, ["task_state_layout must be non-empty and contain unique fields"]

    rigid_prefixes = {
        "grab_roller": ("roller",),
        "place_a2b_left": ("object_A", "target_B"),
        "move_stapler_pad": ("stapler", "pad"),
    }
    if task_name in rigid_prefixes:
        expected = [
            f"{prefix}_{field}"
            for prefix in rigid_prefixes[task_name]
            for field in RIGID_STATE_FIELDS
        ]
        if layout != expected:
            return layout, [f"{task_name} task_state_layout is not canonical"]
        return layout, []

    if task_name == "open_microwave":
        root = [
            "microwave_root_px",
            "microwave_root_py",
            "microwave_root_pz",
            "microwave_root_qw",
            "microwave_root_qx",
            "microwave_root_qy",
            "microwave_root_qz",
        ]
        remaining = len(layout) - len(root)
        if remaining < 2 or remaining % 2:
            return layout, [
                "open_microwave task_state_layout must contain root pose plus equal, non-empty qpos/qvel fields"
            ]
        joint_count = remaining // 2
        expected = [
            *root,
            *[f"microwave_qpos_{index}" for index in range(joint_count)],
            *[f"microwave_qvel_{index}" for index in range(joint_count)],
        ]
        if layout != expected:
            return layout, ["open_microwave task_state_layout is not canonical"]
        return layout, []

    return layout, [f"unsupported task {task_name!r}"]


def _validate_metadata_trace_schema(
    metadata: Mapping[str, Any],
    *,
    task_state_layout: list[str] | None,
    variant: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    raw = metadata.get("trace_schema", _MISSING)
    if not isinstance(raw, Mapping):
        _metadata_error(
            errors,
            variant,
            "canonical_trace_schema_missing",
            "metadata trace_schema must be a canonical mapping",
        )
        return None

    expected_keys = {
        "action_trace_keys",
        "state_trace_keys",
        "task_state_layout",
        "initial_state_row",
        "npz_allow_pickle",
    }
    if set(raw) != expected_keys:
        _metadata_error(
            errors,
            variant,
            "canonical_trace_schema_keys_mismatch",
            "metadata trace_schema keys are not canonical",
            {"expected": sorted(expected_keys), "actual": sorted(str(key) for key in raw)},
        )
    if raw.get("action_trace_keys") != list(ACTION_TRACE_KEY_ORDER):
        _metadata_error(
            errors,
            variant,
            "canonical_action_trace_schema_mismatch",
            "trace_schema action_trace_keys must exactly match the canonical ordered keys",
            {"value": raw.get("action_trace_keys")},
        )
    if raw.get("state_trace_keys") != list(STATE_TRACE_KEY_ORDER):
        _metadata_error(
            errors,
            variant,
            "canonical_state_trace_schema_mismatch",
            "trace_schema state_trace_keys must exactly match the canonical ordered keys",
            {"value": raw.get("state_trace_keys")},
        )
    if task_state_layout is None or raw.get("task_state_layout") != task_state_layout:
        _metadata_error(
            errors,
            variant,
            "trace_schema_task_state_layout_mismatch",
            "trace_schema task_state_layout must exactly equal the top-level layout",
            {
                "top_level": task_state_layout,
                "trace_schema": raw.get("task_state_layout"),
            },
        )
    if _as_json_int(raw.get("initial_state_row", _MISSING)) != 0:
        _metadata_error(
            errors,
            variant,
            "canonical_initial_state_row_invalid",
            "trace_schema initial_state_row must be the JSON integer 0",
            {"value": raw.get("initial_state_row")},
        )
    if raw.get("npz_allow_pickle", _MISSING) is not False:
        _metadata_error(
            errors,
            variant,
            "canonical_npz_allow_pickle_invalid",
            "trace_schema npz_allow_pickle must be the boolean false",
            {"value": raw.get("npz_allow_pickle")},
        )
    return dict(raw)


def _validate_domain_randomization(
    metadata: Mapping[str, Any],
    *,
    is_clean: bool,
    variant: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    settings = _find_mapping(
        metadata,
        ("domain_randomization", "domain_randomisation", "randomization", "randomisation", "dr"),
    )
    if settings is None:
        _metadata_error(errors, variant, "domain_randomization_missing", "domain-randomization settings are missing")
        return None

    flat: dict[str, Any] = {}
    for path, value in _walk_nodes(settings):
        if isinstance(value, Mapping):
            continue
        flat[_normalise_key(".".join(path))] = value

    background_matches = [value for key, value in flat.items() if "randombackground" in key]
    if not background_matches:
        _metadata_error(
            errors,
            variant,
            "background_setting_missing",
            "domain-randomization metadata does not declare random_background",
        )
    else:
        expected = not is_clean
        parsed = [_as_bool(value) for value in background_matches]
        if any(value is None for value in parsed) or any(value != expected for value in parsed):
            _metadata_error(
                errors,
                variant,
                "background_setting_invalid",
                f"random_background must be {expected} for this variant",
                {"values": background_matches},
            )

    mandatory_off = {
        "random_light": ("randomlight", "lightrandom"),
        "clutter": ("clutteredtable", "clutter"),
        "camera": ("randomheadcameradis", "randomcamera", "camerarandom", "camerapose"),
        "table_height": ("randomtableheight", "tableheightrandom"),
        "embodiment": ("randomembodiment", "embodimentrandom"),
    }
    for label, aliases in mandatory_off.items():
        matches = [(key, value) for key, value in flat.items() if any(alias in key for alias in aliases)]
        if not matches:
            _metadata_error(
                errors,
                variant,
                "domain_setting_missing",
                f"domain-randomization metadata does not explicitly disable {label}",
            )
        for key, value in matches:
            if _setting_enabled(value):
                _metadata_error(
                    errors,
                    variant,
                    "forbidden_domain_randomization",
                    f"forbidden domain randomization {key!r} is enabled",
                    {"value": value},
                )

    allowed_truthy = ("randombackground", "onlybackground")
    allowed_control = ("cleanbackgroundrate",)
    for key, value in flat.items():
        if any(token in key for token in allowed_truthy + allowed_control):
            continue
        if _setting_enabled(value):
            _metadata_error(
                errors,
                variant,
                "unexpected_domain_randomization",
                f"only random_background may be enabled, but {key!r} is truthy",
                {"value": value},
            )
    if not is_clean:
        for key, value in flat.items():
            if "cleanbackgroundrate" in key:
                try:
                    if float(value) != 0.0:
                        _metadata_error(
                            errors,
                            variant,
                            "clean_background_rate_nonzero",
                            "style variants require clean_background_rate == 0 so both textures are applied",
                            {"value": value},
                        )
                except (TypeError, ValueError):
                    _metadata_error(
                        errors,
                        variant,
                        "clean_background_rate_invalid",
                        "clean_background_rate must be numeric",
                        {"value": value},
                    )
    return dict(settings)


def _texture_fields(metadata: Mapping[str, Any], surface: str) -> dict[str, Any]:
    """Read the runner's canonical top-level texture fields only.

    The metadata validator separately checks the duplicated ``textures`` map.
    Keeping artifact resolution on the canonical top-level values prevents a
    contradictory fuzzy alias or nested value from hiding a tampered runner
    field.
    """

    return {
        "id": metadata.get(f"{surface}_texture_id", _MISSING),
        "identifier": metadata.get(f"{surface}_texture", _MISSING),
        "path": metadata.get(f"{surface}_texture_file", _MISSING),
        "sha256": metadata.get(f"{surface}_texture_sha256", _MISSING),
    }


def _resolve_texture_file(
    fields: Mapping[str, Any],
    *,
    variant_dir: Path,
    content_dir: Path,
) -> Path | None:
    script_dir = Path(__file__).resolve().parent
    robotwin_root = script_dir.parent
    texture_root = robotwin_root / "assets" / "background_texture"
    raw_candidates: list[str] = []
    for key in ("path", "identifier"):
        value = fields.get(key, _MISSING)
        if value is not _MISSING and value is not None and not isinstance(value, Mapping):
            raw_candidates.append(str(value))
    candidates: list[Path] = []
    for raw in raw_candidates:
        candidate = Path(raw)
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.extend(
                (
                    variant_dir / candidate,
                    content_dir / candidate,
                    robotwin_root / candidate,
                    texture_root / candidate,
                )
            )
        if candidate.suffix == "":
            base_candidates = list(candidates[-4:] if not candidate.is_absolute() else candidates[-1:])
            for base in base_candidates:
                candidates.extend((base.with_suffix(".png"), base.with_suffix(".jpg"), base.with_suffix(".jpeg")))
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normpath(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _count_from_mapping(mapping: Mapping[str, Any], aliases: Sequence[str]) -> int | None:
    return _as_int(_find_value(mapping, aliases))


def _validate_path_consumption(
    metadata: Mapping[str, Any],
    source_counts: Mapping[str, int],
    *,
    variant: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    consumption = metadata.get("path_consumption", _MISSING)
    if not isinstance(consumption, Mapping):
        _metadata_error(
            errors,
            variant,
            "path_consumption_missing",
            "metadata path_consumption must be the canonical mapping",
        )
        consumption = {}
    expected_keys = {"left", "right", "fully_consumed"}
    if set(consumption) != expected_keys:
        _metadata_error(
            errors,
            variant,
            "path_consumption_layout_invalid",
            "metadata path_consumption must contain exactly left/right/fully_consumed",
            {"keys": sorted(str(key) for key in consumption)},
        )
    summary: dict[str, Any] = {}
    for arm in ("left", "right"):
        arm_data = consumption.get(arm, _MISSING)
        if not isinstance(arm_data, Mapping) or set(arm_data) != {"available", "consumed"}:
            _metadata_error(
                errors,
                variant,
                "path_consumption_arm_layout_invalid",
                f"metadata path_consumption.{arm} must contain exactly available/consumed",
                {"value": None if arm_data is _MISSING else arm_data},
            )
            arm_data = {}
        available = _as_json_int(arm_data.get("available", _MISSING))
        consumed = _as_json_int(arm_data.get("consumed", _MISSING))
        summary[arm] = {"available": available, "consumed": consumed, "source": source_counts[arm]}
        if available is None or consumed is None:
            _metadata_error(
                errors,
                variant,
                "path_consumption_missing",
                f"metadata must declare available and consumed {arm} path-entry counts",
            )
            continue
        if available != source_counts[arm]:
            _metadata_error(
                errors,
                variant,
                "path_entry_count_mismatch",
                f"declared {arm} available path count differs from the source pickle",
                {"declared": available, "source": source_counts[arm]},
            )
        if consumed != available or consumed != source_counts[arm]:
            _metadata_error(
                errors,
                variant,
                "path_not_fully_consumed",
                f"replay did not consume every {arm} source path entry exactly once",
                {"available": available, "consumed": consumed, "source": source_counts[arm]},
            )
    fully = consumption.get("fully_consumed", _MISSING)
    fully_bool = fully if isinstance(fully, bool) else None
    summary["fully_consumed"] = fully_bool
    if fully_bool is not True:
        _metadata_error(
            errors,
            variant,
            "path_fully_consumed_missing_or_false",
            "metadata must explicitly state that the replay path was fully consumed",
            {"value": None if fully is _MISSING else fully},
        )
    return summary


def _find_array(arrays: Mapping[str, np.ndarray], aliases: Sequence[str]) -> tuple[str, np.ndarray] | None:
    alias_norm = [_normalise_key(alias) for alias in aliases]
    exact: list[tuple[str, np.ndarray]] = []
    partial: list[tuple[str, np.ndarray]] = []
    for key, array in arrays.items():
        normalised = _normalise_key(key)
        if normalised in alias_norm:
            exact.append((key, array))
        elif any(alias in normalised or normalised in alias for alias in alias_norm):
            partial.append((key, array))
    choices = exact or partial
    return sorted(choices, key=lambda item: item[0])[0] if choices else None


def _trace_length(
    arrays: Mapping[str, np.ndarray],
    *,
    trace_name: str,
    variant: str,
    errors: list[dict[str, Any]],
) -> int | None:
    """Require one unambiguous leading time dimension for a trace archive."""
    lengths: dict[str, int] = {}
    for key, array in arrays.items():
        normalised = _normalise_key(key)
        if "frametraceindex" in normalised or "frameindices" in normalised:
            continue
        if array.ndim == 0:
            _issue(
                errors,
                "trace_scalar_array",
                f"{trace_name} array {key!r} has no leading time dimension",
                variant=variant,
                artifact=f"{trace_name}.npz",
                details={"dtype": str(array.dtype), "shape": list(array.shape)},
            )
            continue
        lengths[key] = int(array.shape[0])
    unique = sorted(set(lengths.values()))
    if not lengths:
        _issue(
            errors,
            "trace_time_dimension_missing",
            f"{trace_name}.npz has no time-indexed arrays",
            variant=variant,
            artifact=f"{trace_name}.npz",
        )
        return None
    if len(unique) != 1:
        _issue(
            errors,
            "trace_lengths_inconsistent",
            f"all {trace_name} arrays must have the same leading time dimension",
            variant=variant,
            artifact=f"{trace_name}.npz",
            details={"lengths": lengths},
        )
        return None
    if unique[0] <= 0:
        _issue(
            errors,
            "trace_empty",
            f"{trace_name}.npz has zero time steps",
            variant=variant,
            artifact=f"{trace_name}.npz",
        )
        return None
    return unique[0]


def _final_bool(array: np.ndarray) -> bool | None:
    if array.size == 0:
        return None
    value = np.asarray(array).reshape(-1)[-1]
    return _as_bool(value)


def _validate_canonical_trace_schema(
    initial_state: Mapping[str, np.ndarray],
    action_trace: Mapping[str, np.ndarray],
    state_trace: Mapping[str, np.ndarray],
    *,
    task_state_width: int | None,
    variant: str,
    errors: list[dict[str, Any]],
) -> None:
    """Require the exact fixed-array schema emitted by PairTraceRecorder."""

    archives = (
        ("initial_state", initial_state, INITIAL_STATE_KEYS),
        ("action_trace", action_trace, ACTION_TRACE_KEYS),
        ("state_trace", state_trace, STATE_TRACE_KEYS),
    )
    for archive_name, arrays, expected_keys in archives:
        actual_keys = set(arrays)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            _issue(
                errors,
                "canonical_trace_keys_mismatch",
                f"{archive_name}.npz does not have the exact canonical array names",
                variant=variant,
                artifact=f"{archive_name}.npz",
                details={"missing": missing, "extra": extra},
            )

        for key, array in arrays.items():
            if key == "semantic_action":
                if array.dtype.kind not in "SU" or array.ndim != 1:
                    _issue(
                        errors,
                        "semantic_action_array_invalid",
                        f"{archive_name}.npz:{key} must be a one-dimensional fixed-width string array",
                        variant=variant,
                        artifact=f"{archive_name}.npz",
                        details={"shape": list(array.shape), "dtype": str(array.dtype)},
                    )
                continue

            if array.dtype.kind not in "buifc":
                _issue(
                    errors,
                    "canonical_numeric_array_invalid",
                    f"{archive_name}.npz:{key} must have a numeric dtype",
                    variant=variant,
                    artifact=f"{archive_name}.npz",
                    details={"shape": list(array.shape), "dtype": str(array.dtype)},
                )
                continue
            try:
                finite = np.isfinite(array)
            except TypeError:
                finite = np.zeros(array.shape, dtype=np.bool_)
            if not np.all(finite):
                _issue(
                    errors,
                    "trace_numeric_non_finite",
                    f"{archive_name}.npz:{key} contains NaN or infinity",
                    variant=variant,
                    artifact=f"{archive_name}.npz",
                    details={
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "non_finite_count": int(array.size - np.count_nonzero(finite)),
                    },
                )

            if key == "frame_trace_index":
                if array.dtype.kind not in "iu" or array.ndim != 1:
                    _issue(
                        errors,
                        "frame_trace_index_invalid",
                        f"{archive_name}.npz:{key} must be a one-dimensional integer array",
                        variant=variant,
                        artifact=f"{archive_name}.npz",
                        details={"shape": list(array.shape), "dtype": str(array.dtype)},
                    )
            elif key in {
                "left_gripper_open",
                "right_gripper_open",
                "left_gripper_closed",
                "right_gripper_closed",
            }:
                if array.dtype.kind != "b" or array.ndim != 1:
                    _issue(
                        errors,
                        "gripper_semantic_array_invalid",
                        f"{archive_name}.npz:{key} must be a one-dimensional boolean array",
                        variant=variant,
                        artifact=f"{archive_name}.npz",
                        details={"shape": list(array.shape), "dtype": str(array.dtype)},
                    )
            elif array.dtype.kind != "f" or array.ndim != 2 or array.shape[1] == 0:
                _issue(
                    errors,
                    "canonical_vector_array_invalid",
                    f"{archive_name}.npz:{key} must be a non-empty two-dimensional floating array",
                    variant=variant,
                    artifact=f"{archive_name}.npz",
                    details={"shape": list(array.shape), "dtype": str(array.dtype)},
                )

    for archive_name, arrays in (("initial_state", initial_state), ("state_trace", state_trace)):
        for key, width in (("left_eef", 7), ("right_eef", 7)):
            array = arrays.get(key)
            if array is not None and (array.ndim != 2 or array.shape[1:] != (width,)):
                _issue(
                    errors,
                    "canonical_vector_width_invalid",
                    f"{archive_name}.npz:{key} must have width {width}",
                    variant=variant,
                    artifact=f"{archive_name}.npz",
                    details={"shape": list(array.shape), "required_width": width},
                )

        task_state = arrays.get("task_state")
        if (
            task_state is not None
            and task_state_width is not None
            and (task_state.ndim != 2 or task_state.shape[1:] != (task_state_width,))
        ):
            _issue(
                errors,
                "task_state_layout_width_mismatch",
                f"{archive_name}.npz:task_state width must equal task_state_layout length",
                variant=variant,
                artifact=f"{archive_name}.npz",
                details={
                    "shape": list(task_state.shape),
                    "layout_width": task_state_width,
                },
            )

        for arm in ("left", "right"):
            opened = arrays.get(f"{arm}_gripper_open")
            closed = arrays.get(f"{arm}_gripper_closed")
            if (
                opened is not None
                and closed is not None
                and opened.dtype.kind == "b"
                and closed.dtype.kind == "b"
                and opened.shape == closed.shape
                and np.any(opened & closed)
            ):
                _issue(
                    errors,
                    "gripper_semantics_contradictory",
                    f"{archive_name}.npz marks the {arm} gripper open and closed simultaneously",
                    variant=variant,
                    artifact=f"{archive_name}.npz",
                )

    for arm in ("left", "right"):
        qpos = state_trace.get(f"{arm}_qpos")
        qvel = state_trace.get(f"{arm}_qvel")
        target = action_trace.get(f"{arm}_drive_target")
        velocity = action_trace.get(f"{arm}_drive_velocity")
        widths = {
            name: int(array.shape[1])
            for name, array in (
                ("qpos", qpos),
                ("qvel", qvel),
                ("drive_target", target),
                ("drive_velocity", velocity),
            )
            if array is not None and array.ndim == 2
        }
        if len(widths) == 4 and len(set(widths.values())) != 1:
            _issue(
                errors,
                "joint_vector_width_mismatch",
                f"canonical {arm} qpos/qvel and drive vectors must have the same width",
                variant=variant,
                artifact="state_trace.npz",
                details={"widths": widths},
            )

    non_frame_state_keys = STATE_TRACE_KEYS - {"frame_trace_index"}
    if set(initial_state) - {"frame_trace_index"} == non_frame_state_keys:
        for key in sorted(non_frame_state_keys):
            initial_array = initial_state[key]
            state_array = state_trace.get(key)
            if initial_array.ndim == 0 or initial_array.shape[0] != 1:
                _issue(
                    errors,
                    "initial_state_row_count_invalid",
                    f"initial_state.npz:{key} must contain exactly one leading row",
                    variant=variant,
                    artifact="initial_state.npz",
                    details={"shape": list(initial_array.shape)},
                )
                continue
            if state_array is None or state_array.ndim == 0:
                continue
            expected = state_array[:1]
            if not _exact_array_equal(initial_array, expected):
                _issue(
                    errors,
                    "initial_state_row_mismatch",
                    f"initial_state.npz:{key} is not bitwise identical to state_trace row 0",
                    variant=variant,
                    artifact="initial_state.npz",
                    details={"array": key, **_array_mismatch_details(expected, initial_array)},
                )

    initial_frame = initial_state.get("frame_trace_index")
    if initial_frame is not None and not _exact_array_equal(
        initial_frame, np.asarray([0], dtype=np.int64)
    ):
        _issue(
            errors,
            "initial_frame_trace_index_invalid",
            "initial_state.npz:frame_trace_index must be exactly int64 [0]",
            variant=variant,
            artifact="initial_state.npz",
            details={"shape": list(initial_frame.shape), "dtype": str(initial_frame.dtype)},
        )


def _validate_trace_semantics(
    initial_state: Mapping[str, np.ndarray],
    action_trace: Mapping[str, np.ndarray],
    state_trace: Mapping[str, np.ndarray],
    *,
    task_name: str,
    task_state_layout: Sequence[str] | None,
    task_success_spec: Mapping[str, Any] | None,
    frame_count: int | None,
    variant: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_canonical_trace_schema(
        initial_state,
        action_trace,
        state_trace,
        task_state_width=(
            None if task_state_layout is None else len(task_state_layout)
        ),
        variant=variant,
        errors=errors,
    )

    action_length = _trace_length(
        action_trace,
        trace_name="action_trace",
        variant=variant,
        errors=errors,
    )
    state_length = _trace_length(
        state_trace,
        trace_name="state_trace",
        variant=variant,
        errors=errors,
    )
    if action_length is not None and state_length is not None and state_length != action_length + 1:
        _issue(
            errors,
            "trace_length_relation_invalid",
            "state trace must contain the initial state plus exactly one state per action",
            variant=variant,
            artifact="state_trace.npz",
            details={
                "action_length": action_length,
                "state_length": state_length,
                "required_relation": "state_length == action_length + 1",
            },
        )

    action_semantic = (
        ("semantic_action", action_trace["semantic_action"])
        if "semantic_action" in action_trace
        else None
    )
    state_semantic = (
        ("semantic_action", state_trace["semantic_action"])
        if "semantic_action" in state_trace
        else None
    )
    if action_semantic is None or state_semantic is None:
        _issue(
            errors,
            "semantic_action_missing",
            "both traces must contain semantic_action labels",
            variant=variant,
            artifact="action_trace.npz",
        )
    else:
        action_labels = np.asarray(action_semantic[1]).reshape(-1)
        state_labels = np.asarray(state_semantic[1]).reshape(-1)
        if state_labels.size == 0 or _normalise_key(state_labels[0]) != "initial":
            _issue(
                errors,
                "state_initial_label_missing",
                "state semantic_action must begin with the 'initial' label",
                variant=variant,
                artifact="state_trace.npz",
            )
        if any(_normalise_key(value) == "initial" for value in action_labels):
            _issue(
                errors,
                "initial_action_forbidden",
                "action_trace must not contain an initial pseudo-action",
                variant=variant,
                artifact="action_trace.npz",
            )
        if state_labels.size != action_labels.size + 1 or not np.array_equal(
            state_labels[1:].astype(str), action_labels.astype(str)
        ):
            _issue(
                errors,
                "semantic_action_alignment_invalid",
                "state semantic labels after 'initial' must exactly match action semantic labels",
                variant=variant,
                artifact="state_trace.npz",
                details={
                    "action_label_count": int(action_labels.size),
                    "state_label_count": int(state_labels.size),
                },
            )

    frame_entry = (
        ("frame_trace_index", state_trace["frame_trace_index"])
        if "frame_trace_index" in state_trace
        else None
    )
    frame_summary: dict[str, Any] = {"key": None, "count": None, "first": None, "last": None}
    if frame_entry is None:
        _issue(
            errors,
            "frame_trace_index_missing",
            "neither trace NPZ contains frame_trace_index",
            variant=variant,
            artifact="state_trace.npz",
        )
    else:
        key, indices = frame_entry
        frame_summary["key"] = key
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            _issue(
                errors,
                "frame_trace_index_invalid",
                "frame_trace_index must be a one-dimensional integer array",
                variant=variant,
                artifact="state_trace.npz",
                details={"key": key, "shape": list(indices.shape), "dtype": str(indices.dtype)},
            )
        else:
            frame_summary["count"] = int(indices.size)
            if indices.size:
                frame_summary["first"] = int(indices[0])
                frame_summary["last"] = int(indices[-1])
            if indices.size == 0 or int(indices[0]) < 0 or np.any(np.diff(indices.astype(np.int64)) < 0):
                _issue(
                    errors,
                    "frame_trace_index_decreases",
                    "frame_trace_index must be non-empty, non-negative, and non-decreasing",
                    variant=variant,
                    artifact="state_trace.npz",
                )
            if state_length is not None and indices.size and int(indices[-1]) >= state_length:
                _issue(
                    errors,
                    "frame_trace_index_out_of_range",
                    "frame_trace_index contains an index outside state_trace",
                    variant=variant,
                    artifact="state_trace.npz",
                    details={"last_index": int(indices[-1]), "state_length": state_length},
                )
            if frame_count is not None and indices.size != frame_count:
                _issue(
                    errors,
                    "frame_trace_index_count_mismatch",
                    "frame_trace_index length does not equal the HDF5/video frame count",
                    variant=variant,
                    artifact="state_trace.npz",
                    details={"trace_indices": int(indices.size), "frames": frame_count},
                )

    final_gripper: dict[str, dict[str, bool | None]] = {
        "open": {"left": None, "right": None},
        "closed": {"left": None, "right": None},
    }
    for status in ("open", "closed"):
        for arm in ("left", "right"):
            key = f"{arm}_gripper_{status}"
            array = state_trace.get(key)
            if array is None:
                _issue(
                    errors,
                    "semantic_gripper_state_missing",
                    f"state_trace.npz has no semantic {arm}-gripper-{status} array",
                    variant=variant,
                    artifact="state_trace.npz",
                )
                continue
            final_gripper[status][arm] = _final_bool(array)
            if final_gripper[status][arm] is None:
                _issue(
                    errors,
                    "semantic_gripper_state_invalid",
                    f"final semantic {arm}-gripper-{status} state is not boolean",
                    variant=variant,
                    artifact="state_trace.npz",
                    details={"key": key, "dtype": str(array.dtype)},
                )

    success_details: dict[str, Any] = {"derived_success": False}
    success_issue_emitted = False
    task_state = state_trace.get("task_state")
    if task_state is None:
        _issue(
            errors,
            "task_state_missing",
            "state_trace.npz has no task_state array from which success can be derived",
            variant=variant,
            artifact="state_trace.npz",
        )
        success_issue_emitted = True
    elif task_state_layout is None or task_success_spec is None:
        _issue(
            errors,
            "success_metadata_unavailable",
            "canonical task_state_layout/task_success_spec is unavailable",
            variant=variant,
            artifact="metadata.json",
        )
        success_issue_emitted = True
    else:
        try:
            success_details = derive_success(
                task_name,
                task_state,
                task_state_layout,
                left_gripper_open=final_gripper["open"]["left"],
                right_gripper_open=final_gripper["open"]["right"],
                spec=task_success_spec,
            )
        except (TaskAdapterError, KeyError, TypeError, ValueError) as exc:
            _issue(
                errors,
                "independent_success_derivation_invalid",
                f"cannot independently derive {task_name} success: {exc}",
                variant=variant,
                artifact="state_trace.npz",
            )
            success_issue_emitted = True
        if task_name == "grab_roller" and not (
            final_gripper["closed"]["left"] is True
            and final_gripper["closed"]["right"] is True
        ):
            success_details["derived_success"] = False
            success_details["requires_both_grippers_closed"] = True
        if success_details.get("derived_success") is not True:
            _issue(
                errors,
                "independent_success_failed",
                f"final {task_name} success is false when independently derived from task_state",
                variant=variant,
                artifact="state_trace.npz",
                details={
                    **success_details,
                    "left_gripper_open": final_gripper["open"]["left"],
                    "right_gripper_open": final_gripper["open"]["right"],
                    "left_gripper_closed": final_gripper["closed"]["left"],
                    "right_gripper_closed": final_gripper["closed"]["right"],
                },
            )
            success_issue_emitted = True

    if success_details.get("derived_success") is not True and not success_issue_emitted:
        _issue(
            errors,
            "independent_success_failed",
            f"final {task_name} success could not be independently established",
            variant=variant,
            artifact="state_trace.npz",
            details=success_details,
        )

    return {
        "action_length": action_length,
        "state_length": state_length,
        "frame_trace_index": frame_summary,
        "task": task_name,
        "left_gripper_open": final_gripper["open"]["left"],
        "right_gripper_open": final_gripper["open"]["right"],
        "left_gripper_closed": final_gripper["closed"]["left"],
        "right_gripper_closed": final_gripper["closed"]["right"],
        **success_details,
    }


def _validate_trace_hdf5_alignment(
    state_trace: Mapping[str, np.ndarray],
    hdf5_path: Path,
    *,
    variant: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind each saved frame to its exact physical state in native HDF5.

    HDF5 preserves SAPIEN's native float32 qpos/qvel while the trace recorder
    promotes vectors to float64.  Equality is therefore exact by numeric value
    and shape, with no tolerance, but intentionally does not require equal
    serialization dtypes across the two formats.
    """

    summary: dict[str, Any] = {"checked_arrays": 0, "equal": True, "datasets": {}}
    indices = state_trace.get("frame_trace_index")
    if indices is None or indices.ndim != 1 or indices.dtype.kind not in "iu":
        _issue(
            errors,
            "hdf5_alignment_frame_map_invalid",
            "cannot align HDF5 without a canonical one-dimensional integer frame_trace_index",
            variant=variant,
            artifact="state_trace.npz",
        )
        summary["equal"] = False
        return summary
    indices_i64 = indices.astype(np.int64, copy=False)
    if indices_i64.size == 0 or np.any(indices_i64 < 0):
        _issue(
            errors,
            "hdf5_alignment_frame_map_invalid",
            "cannot align HDF5 with an empty or negative frame_trace_index",
            variant=variant,
            artifact="state_trace.npz",
        )
        summary["equal"] = False
        return summary

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("h5py is required to align state traces with HDF5") from exc

    with h5py.File(hdf5_path, "r") as handle:
        for trace_key, dataset_path in TRACE_TO_HDF5.items():
            trace_array = state_trace.get(trace_key)
            if trace_array is None:
                # The canonical-key check already reports the missing trace key,
                # but retain an explicit failed binding in the alignment report.
                summary["equal"] = False
                summary["datasets"][trace_key] = {"path": dataset_path, "equal": False}
                continue
            if dataset_path not in handle:
                _issue(
                    errors,
                    "hdf5_trace_dataset_missing",
                    f"HDF5 is missing canonical trace binding dataset /{dataset_path}",
                    variant=variant,
                    artifact="data/episode0.hdf5",
                    details={"trace_key": trace_key, "dataset": f"/{dataset_path}"},
                )
                summary["equal"] = False
                summary["datasets"][trace_key] = {"path": dataset_path, "equal": False}
                continue
            if trace_array.ndim == 0 or int(indices_i64[-1]) >= trace_array.shape[0]:
                _issue(
                    errors,
                    "hdf5_alignment_frame_map_out_of_range",
                    f"frame_trace_index cannot index state_trace.npz:{trace_key}",
                    variant=variant,
                    artifact="state_trace.npz",
                    details={
                        "last_index": int(indices_i64[-1]),
                        "trace_shape": list(trace_array.shape),
                    },
                )
                summary["equal"] = False
                summary["datasets"][trace_key] = {"path": dataset_path, "equal": False}
                continue

            hdf5_array = np.asarray(handle[dataset_path][()])
            selected_trace = np.asarray(trace_array[indices_i64])
            if hdf5_array.dtype.kind not in "buifc":
                _issue(
                    errors,
                    "hdf5_trace_dataset_non_numeric",
                    f"HDF5 /{dataset_path} must have a numeric dtype",
                    variant=variant,
                    artifact="data/episode0.hdf5",
                    details={"shape": list(hdf5_array.shape), "dtype": str(hdf5_array.dtype)},
                )
                summary["equal"] = False
                summary["datasets"][trace_key] = {"path": dataset_path, "equal": False}
                continue
            if not np.all(np.isfinite(hdf5_array)):
                _issue(
                    errors,
                    "hdf5_trace_dataset_non_finite",
                    f"HDF5 /{dataset_path} contains NaN or infinity",
                    variant=variant,
                    artifact="data/episode0.hdf5",
                    details={
                        "shape": list(hdf5_array.shape),
                        "dtype": str(hdf5_array.dtype),
                        "non_finite_count": int(
                            hdf5_array.size - np.count_nonzero(np.isfinite(hdf5_array))
                        ),
                    },
                )
                summary["equal"] = False

            equal = hdf5_array.shape == selected_trace.shape and np.array_equal(
                hdf5_array, selected_trace
            )
            summary["checked_arrays"] += 1
            summary["datasets"][trace_key] = {
                "path": dataset_path,
                "equal": bool(equal),
                "hdf5_shape": list(hdf5_array.shape),
                "trace_shape": list(selected_trace.shape),
                "hdf5_dtype": str(hdf5_array.dtype),
                "trace_dtype": str(selected_trace.dtype),
            }
            if not equal:
                _issue(
                    errors,
                    "hdf5_trace_exact_mismatch",
                    f"HDF5 /{dataset_path} is not exactly equal to {trace_key}[frame_trace_index]",
                    variant=variant,
                    artifact="data/episode0.hdf5",
                    details={
                        "trace_key": trace_key,
                        "dataset": f"/{dataset_path}",
                        "zero_tolerance": True,
                        **_array_mismatch_details(selected_trace, hdf5_array),
                    },
                )
                summary["equal"] = False
    return summary


def _validate_hdf5_semantic_coverage(
    inventory: Mapping[str, Any], variant: str, errors: list[dict[str, Any]]
) -> None:
    leaves = [_normalise_key(name) for name in inventory["non_rgb_leaves"]]
    requirements = {
        "robot/qpos": ("robotstate", "jointaction", "qpos"),
        "end-effector pose": ("endpose", "endeffector", "eepose"),
    }
    for label, tokens in requirements.items():
        if not any(any(token in leaf for token in tokens) for leaf in leaves):
            _issue(
                errors,
                "hdf5_semantic_leaf_missing",
                f"HDF5 has no {label} leaf available for strict pairing validation",
                variant=variant,
                artifact="data/episode0.hdf5",
            )


def _validate_exact_variant_layout(
    variant_dir: Path, *, is_clean: bool, variant: str, errors: list[dict[str, Any]]
) -> dict[str, Path]:
    expected_top = {"metadata.json", "initial_state.npz", "action_trace.npz", "state_trace.npz", "data", "video"}
    if is_clean:
        expected_top.add("_traj_data")
    if variant_dir.is_dir():
        actual_top = {entry.name for entry in variant_dir.iterdir()}
        extra = sorted(actual_top - expected_top)
        missing_top = sorted(expected_top - actual_top)
        if extra or missing_top:
            _issue(
                errors,
                "variant_layout_mismatch",
                "variant top-level layout is not exact",
                variant=variant,
                details={"missing": missing_top, "extra": extra},
            )
    paths = {relative: variant_dir / relative for relative in REQUIRED_VARIANT_FILES}
    if is_clean:
        paths[SOURCE_TRAJECTORY] = variant_dir / SOURCE_TRAJECTORY
    for relative, path in paths.items():
        if not path.is_file():
            _issue(
                errors,
                "required_file_missing",
                f"required file {relative!r} is missing",
                variant=variant,
                artifact=relative,
            )
    for directory, expected_file in (("data", "episode0.hdf5"), ("video", "episode0.mp4")):
        folder = variant_dir / directory
        if folder.is_dir():
            actual = sorted(item.name for item in folder.iterdir())
            if actual != [expected_file]:
                _issue(
                    errors,
                    "episode_directory_not_exact",
                    f"{directory}/ must contain exactly {expected_file}",
                    variant=variant,
                    artifact=directory,
                    details={"entries": actual},
                )
    if is_clean:
        folder = variant_dir / "_traj_data"
        if folder.is_dir():
            actual = sorted(item.name for item in folder.iterdir())
            if actual != ["episode0.pkl"]:
                _issue(
                    errors,
                    "trajectory_directory_not_exact",
                    "_traj_data/ must contain exactly episode0.pkl",
                    variant=variant,
                    artifact="_traj_data",
                    details={"entries": actual},
                )
    return paths


def _validate_source_episode(value: Any) -> bool:
    integer = _as_int(value)
    if integer is not None:
        return integer == 0
    if isinstance(value, str):
        normalised = value.replace("\\", "/").rstrip("/").lower()
        return normalised in {"clean", "episode0", "clean/episode0"} or normalised.endswith(
            "clean/_traj_data/episode0.pkl"
        )
    return False


def _same_json_scalar(left: Any, right: Any) -> bool:
    """Compare duplicated canonical JSON scalar fields without bool/int aliasing."""

    return type(left) is type(right) and left == right


def _validate_canonical_texture_metadata(
    metadata: Mapping[str, Any],
    *,
    variant: str,
    is_clean: bool,
    errors: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate the exact redundant texture schema emitted by the runner."""

    nested_textures = metadata.get("textures", _MISSING)
    if not isinstance(nested_textures, Mapping):
        _metadata_error(
            errors,
            variant,
            "canonical_textures_mapping_invalid",
            "metadata textures must be the canonical wall/table mapping",
            {"value": None if nested_textures is _MISSING else nested_textures},
        )
        nested_textures = {}
    elif set(nested_textures) != {"wall", "table"}:
        _metadata_error(
            errors,
            variant,
            "canonical_textures_layout_invalid",
            "metadata textures must contain exactly wall and table",
            {"keys": sorted(str(key) for key in nested_textures)},
        )

    summary: dict[str, dict[str, Any]] = {}
    nested_keys = {"id", "name", "file", "sha256"}
    for surface in ("wall", "table"):
        top_names = {
            "id": f"{surface}_texture_id",
            "name": f"{surface}_texture",
            "file": f"{surface}_texture_file",
            "sha256": f"{surface}_texture_sha256",
        }
        missing_top = sorted(name for name in top_names.values() if name not in metadata)
        if missing_top:
            _metadata_error(
                errors,
                variant,
                "canonical_texture_fields_missing",
                f"metadata is missing canonical top-level {surface} texture fields",
                {"missing": missing_top},
            )
        top = {field: metadata.get(name, _MISSING) for field, name in top_names.items()}
        public_top = {
            field: None if value is _MISSING else value for field, value in top.items()
        }

        nested = nested_textures.get(surface, _MISSING)
        if not isinstance(nested, Mapping):
            _metadata_error(
                errors,
                variant,
                "canonical_texture_surface_invalid",
                f"metadata textures.{surface} must be a mapping",
                {"value": None if nested is _MISSING else nested},
            )
            nested = {}
        elif set(nested) != nested_keys:
            _metadata_error(
                errors,
                variant,
                "canonical_texture_surface_layout_invalid",
                f"metadata textures.{surface} must contain exactly id/name/file/sha256",
                {"keys": sorted(str(key) for key in nested)},
            )

        for field in sorted(nested_keys):
            nested_value = nested.get(field, _MISSING)
            top_value = top[field]
            if (
                nested_value is _MISSING
                or top_value is _MISSING
                or not _same_json_scalar(nested_value, top_value)
            ):
                _metadata_error(
                    errors,
                    variant,
                    "canonical_texture_duplicate_mismatch",
                    f"metadata textures.{surface}.{field} must exactly equal {top_names[field]}",
                    {
                        "top": None if top_value is _MISSING else top_value,
                        "nested": None if nested_value is _MISSING else nested_value,
                    },
                )

        if is_clean:
            non_null = {
                field: value
                for field, value in top.items()
                if value is _MISSING or value is not None
            }
            if non_null:
                _metadata_error(
                    errors,
                    variant,
                    "canonical_clean_texture_not_null",
                    f"clean canonical {surface} texture fields must all be null",
                    {"fields": public_top},
                )
        else:
            texture_id = _as_json_int(top["id"])
            if texture_id is None or texture_id < 0:
                _metadata_error(
                    errors,
                    variant,
                    "canonical_texture_id_invalid",
                    f"style {surface}_texture_id must be a non-negative JSON integer",
                    {"value": public_top["id"]},
                )
            else:
                expected_name = f"seen/{texture_id}"
                expected_file = f"assets/background_texture/seen/{texture_id}.png"
                if top["name"] != expected_name:
                    _metadata_error(
                        errors,
                        variant,
                        "canonical_texture_name_invalid",
                        f"{surface}_texture must be derived exactly from {surface}_texture_id",
                        {"expected": expected_name, "actual": public_top["name"]},
                    )
                if top["file"] != expected_file:
                    _metadata_error(
                        errors,
                        variant,
                        "canonical_texture_file_invalid",
                        f"{surface}_texture_file must be the canonical seen-texture path",
                        {"expected": expected_file, "actual": public_top["file"]},
                    )
            if not _valid_hash(top["sha256"]):
                _metadata_error(
                    errors,
                    variant,
                    "canonical_texture_hash_invalid",
                    f"{surface}_texture_sha256 must be a valid SHA-256",
                    {"value": public_top["sha256"]},
                )

        summary[surface] = public_top
    if not is_clean:
        wall_id = _as_json_int(summary["wall"]["id"])
        table_id = _as_json_int(summary["table"]["id"])
        if wall_id is not None and table_id is not None and wall_id == table_id:
            _metadata_error(
                errors,
                variant,
                "canonical_style_texture_ids_not_distinct",
                "style wall/table texture IDs must be distinct",
                {"wall_texture_id": wall_id, "table_texture_id": table_id},
            )
    return summary


def _validate_canonical_replay_metadata(
    metadata: Mapping[str, Any],
    *,
    variant: str,
    is_clean: bool,
    expected_task: str,
    expected_content_id: int | None,
    expected_style_index: int | None,
    expected_style_seed: int | None,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed on the exact provenance fields written by the runner."""

    canonical_variant = metadata.get("variant", _MISSING)
    if not isinstance(canonical_variant, str) or canonical_variant != variant:
        _metadata_error(
            errors,
            variant,
            "canonical_variant_mismatch",
            "metadata variant must exactly match its directory name",
            {
                "expected": variant,
                "actual": None if canonical_variant is _MISSING else canonical_variant,
            },
        )

    style_index_raw = metadata.get("style_index", _MISSING)
    if is_clean:
        style_index = None
        if style_index_raw is not None:
            _metadata_error(
                errors,
                variant,
                "canonical_clean_style_index_not_null",
                "clean metadata style_index must be present and null",
                {"value": None if style_index_raw is _MISSING else style_index_raw},
            )
    else:
        style_index = _as_json_int(style_index_raw)
        if style_index is None or style_index != expected_style_index:
            _metadata_error(
                errors,
                variant,
                "canonical_style_index_mismatch",
                "metadata style_index must exactly match the style directory ordinal",
                {
                    "expected": expected_style_index,
                    "actual": None if style_index_raw is _MISSING else style_index_raw,
                },
            )

    canonical_style_seed_raw = metadata.get("style_seed", _MISSING)
    if is_clean:
        canonical_style_seed = None
        if canonical_style_seed_raw is not None:
            _metadata_error(
                errors,
                variant,
                "canonical_clean_style_seed_not_null",
                "clean metadata canonical style_seed must be present and null",
                {
                    "value": None
                    if canonical_style_seed_raw is _MISSING
                    else canonical_style_seed_raw
                },
            )
    else:
        canonical_style_seed = _as_json_int(canonical_style_seed_raw)
        if canonical_style_seed is None or canonical_style_seed != expected_style_seed:
            _metadata_error(
                errors,
                variant,
                "canonical_style_seed_mismatch",
                "metadata canonical style_seed must exactly match the style directory seed",
                {
                    "expected": expected_style_seed,
                    "actual": None
                    if canonical_style_seed_raw is _MISSING
                    else canonical_style_seed_raw,
                },
            )

    split = metadata.get("split", _MISSING)
    expected_split: str | None = None
    if expected_content_id is not None:
        try:
            if expected_content_id < 0:
                raise TaskAdapterError("content_id is negative")
            expected_split = split_for_content(expected_content_id)
        except TaskAdapterError as exc:
            _metadata_error(
                errors,
                variant,
                "content_id_outside_fixed_split",
                str(exc),
                {"content_id": expected_content_id},
            )
    if not isinstance(split, str) or expected_split is None or split != expected_split:
        _metadata_error(
            errors,
            variant,
            "canonical_split_mismatch",
            "metadata split must exactly match the fixed content-ID split",
            {
                "expected": expected_split,
                "actual": None if split is _MISSING else split,
            },
        )

    identity_raw = metadata.get("task_identity", _MISSING)
    task_identity = dict(identity_raw) if isinstance(identity_raw, Mapping) else None
    identity_errors = validate_task_identity(expected_task, identity_raw)
    for message in identity_errors:
        _metadata_error(
            errors,
            variant,
            "canonical_task_identity_invalid",
            message,
            {"value": None if identity_raw is _MISSING else identity_raw},
        )
    identity_hash = metadata.get("task_identity_sha256", _MISSING)
    expected_identity_hash = (
        canonical_json_sha256(task_identity) if task_identity is not None else None
    )
    if (
        not _valid_hash(identity_hash)
        or expected_identity_hash is None
        or str(identity_hash).lower() != expected_identity_hash
    ):
        _metadata_error(
            errors,
            variant,
            "canonical_task_identity_hash_mismatch",
            "metadata task_identity_sha256 must be the canonical JSON hash of task_identity",
            {
                "declared": None if identity_hash is _MISSING else identity_hash,
                "expected": expected_identity_hash,
            },
        )

    success_spec_raw = metadata.get("task_success_spec", _MISSING)
    task_success_spec = (
        dict(success_spec_raw) if isinstance(success_spec_raw, Mapping) else None
    )
    success_spec_errors = validate_success_spec(expected_task, success_spec_raw)
    for message in success_spec_errors:
        _metadata_error(
            errors,
            variant,
            "canonical_task_success_spec_invalid",
            message,
            {"value": None if success_spec_raw is _MISSING else success_spec_raw},
        )
    success_spec_hash = metadata.get("task_success_spec_sha256", _MISSING)
    expected_success_spec_hash = (
        canonical_json_sha256(task_success_spec) if task_success_spec is not None else None
    )
    if (
        not _valid_hash(success_spec_hash)
        or expected_success_spec_hash is None
        or str(success_spec_hash).lower() != expected_success_spec_hash
    ):
        _metadata_error(
            errors,
            variant,
            "canonical_task_success_spec_hash_mismatch",
            "metadata task_success_spec_sha256 must be the canonical JSON hash of task_success_spec",
            {
                "declared": None if success_spec_hash is _MISSING else success_spec_hash,
                "expected": expected_success_spec_hash,
            },
        )

    task_state_layout, layout_errors = _canonical_task_state_layout(
        expected_task, metadata.get("task_state_layout", _MISSING)
    )
    for message in layout_errors:
        _metadata_error(
            errors,
            variant,
            "canonical_task_state_layout_invalid",
            message,
            {"value": metadata.get("task_state_layout")},
        )
    trace_schema = _validate_metadata_trace_schema(
        metadata,
        task_state_layout=task_state_layout,
        variant=variant,
        errors=errors,
    )

    need_plan = metadata.get("need_plan", _MISSING)
    if need_plan is not False:
        _metadata_error(
            errors,
            variant,
            "canonical_need_plan_not_false",
            "published replay metadata need_plan must be the boolean false",
            {"value": None if need_plan is _MISSING else need_plan},
        )

    rng_state = metadata.get("rng_state_sha256_after_setup", _MISSING)
    if not _valid_hash(rng_state):
        _metadata_error(
            errors,
            variant,
            "canonical_rng_state_hash_invalid",
            "metadata rng_state_sha256_after_setup must be a valid SHA-256",
            {"value": None if rng_state is _MISSING else rng_state},
        )

    style_rng_raw = metadata.get("style_rng", _MISSING)
    style_rng: dict[str, Any] | None = None
    if is_clean:
        if style_rng_raw is not None:
            _metadata_error(
                errors,
                variant,
                "canonical_clean_style_rng_not_null",
                "clean metadata style_rng must be present and null",
                {"value": None if style_rng_raw is _MISSING else style_rng_raw},
            )
    elif not isinstance(style_rng_raw, Mapping):
        _metadata_error(
            errors,
            variant,
            "canonical_style_rng_invalid",
            "style metadata style_rng must be the canonical RNG mapping",
            {"value": None if style_rng_raw is _MISSING else style_rng_raw},
        )
    else:
        expected_rng_keys = {"implementation", "seed", "state_sha256_after_sampling"}
        if set(style_rng_raw) != expected_rng_keys:
            _metadata_error(
                errors,
                variant,
                "canonical_style_rng_layout_invalid",
                "metadata style_rng must contain exactly "
                "implementation/seed/state_sha256_after_sampling",
                {"keys": sorted(str(key) for key in style_rng_raw)},
            )
        implementation = style_rng_raw.get("implementation", _MISSING)
        rng_seed = _as_json_int(style_rng_raw.get("seed", _MISSING))
        rng_hash = style_rng_raw.get("state_sha256_after_sampling", _MISSING)
        if implementation != "numpy.random.default_rng":
            _metadata_error(
                errors,
                variant,
                "canonical_style_rng_implementation_invalid",
                "metadata style_rng implementation must be numpy.random.default_rng",
                {"value": None if implementation is _MISSING else implementation},
            )
        if rng_seed is None or rng_seed != expected_style_seed:
            _metadata_error(
                errors,
                variant,
                "canonical_style_rng_seed_mismatch",
                "metadata style_rng seed must exactly match the style directory seed",
                {
                    "expected": expected_style_seed,
                    "actual": style_rng_raw.get("seed"),
                },
            )
        if not _valid_hash(rng_hash):
            _metadata_error(
                errors,
                variant,
                "canonical_style_rng_state_hash_invalid",
                "metadata style_rng state_sha256_after_sampling must be a valid SHA-256",
                {"value": None if rng_hash is _MISSING else rng_hash},
            )
        style_rng = {
            "implementation": None if implementation is _MISSING else implementation,
            "seed": rng_seed,
            "state_sha256_after_sampling": None if rng_hash is _MISSING else rng_hash,
        }

    textures = _validate_canonical_texture_metadata(
        metadata,
        variant=variant,
        is_clean=is_clean,
        errors=errors,
    )
    return {
        "variant": None if canonical_variant is _MISSING else canonical_variant,
        "style_index": style_index,
        "canonical_style_seed": canonical_style_seed,
        "split": None if split is _MISSING else split,
        "task_identity": task_identity,
        "task_identity_sha256": None if identity_hash is _MISSING else identity_hash,
        "task_success_spec": task_success_spec,
        "task_success_spec_sha256": (
            None if success_spec_hash is _MISSING else success_spec_hash
        ),
        "task_state_layout": task_state_layout,
        "trace_schema": trace_schema,
        "need_plan": need_plan if isinstance(need_plan, bool) else None,
        "rng_state_sha256_after_setup": None if rng_state is _MISSING else rng_state,
        "style_rng": style_rng,
        "textures": textures,
    }


def _validate_variant_metadata(
    metadata: Mapping[str, Any],
    *,
    variant: str,
    is_clean: bool,
    expected_task: str,
    expected_content_id: int | None,
    expected_content_seed: int | None,
    expected_style_index: int | None,
    expected_style_seed: int | None,
    source_counts: Mapping[str, int],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical = _validate_canonical_replay_metadata(
        metadata,
        variant=variant,
        is_clean=is_clean,
        expected_task=expected_task,
        expected_content_id=expected_content_id,
        expected_style_index=expected_style_index,
        expected_style_seed=expected_style_seed,
        errors=errors,
    )
    schema_version = _as_json_int(metadata.get("schema_version", _MISSING))
    if schema_version != SCHEMA_VERSION:
        _metadata_error(
            errors,
            variant,
            "metadata_schema_version_mismatch",
            f"metadata schema_version must be exactly {SCHEMA_VERSION}",
            {"value": metadata.get("schema_version")},
        )
    task = metadata.get("task", _MISSING)
    if not isinstance(task, str) or task != expected_task:
        _metadata_error(
            errors,
            variant,
            "task_mismatch",
            f"task must be exactly {expected_task!r}",
            {"value": None if task is _MISSING else task},
        )
    content_id_raw = metadata.get("content_id", _MISSING)
    content_id = _as_json_int(content_id_raw)
    if content_id is None:
        _metadata_error(errors, variant, "content_id_missing", "metadata content_id must be an integer")
    elif expected_content_id is not None and content_id != expected_content_id:
        _metadata_error(
            errors,
            variant,
            "content_id_mismatch",
            "metadata content_id does not match the content directory",
            {"expected": expected_content_id, "actual": content_id},
        )
    content_seed_raw = metadata.get("content_seed", _MISSING)
    content_seed = _as_json_int(content_seed_raw)
    if content_seed is None:
        _metadata_error(errors, variant, "content_seed_missing", "metadata content_seed must be an integer")
    elif expected_content_seed is not None and content_seed != expected_content_seed:
        _metadata_error(
            errors,
            variant,
            "content_seed_mismatch",
            "metadata content_seed differs from the expected value",
            {"expected": expected_content_seed, "actual": content_seed},
        )

    style_seed_raw = metadata.get("style_seed", _MISSING)
    if style_seed_raw is _MISSING:
        _metadata_error(errors, variant, "style_seed_missing", "metadata must contain style_seed (null for clean)")
        style_seed = None
    elif is_clean:
        style_seed = None
        if style_seed_raw is not None:
            _metadata_error(
                errors,
                variant,
                "clean_style_seed_not_null",
                "clean metadata style_seed must be null",
                {"value": style_seed_raw},
            )
    else:
        style_seed = _as_json_int(style_seed_raw)
        if style_seed is None or style_seed != expected_style_seed:
            _metadata_error(
                errors,
                variant,
                "style_seed_mismatch",
                "metadata style_seed does not match the style directory",
                {"expected": expected_style_seed, "actual": style_seed_raw},
            )

    intervention = metadata.get("intervention", _MISSING)
    intervention_norm = _normalise_key(intervention) if intervention is not _MISSING else ""
    allowed_clean = {"none", "clean", "nointervention"}
    if (is_clean and intervention_norm not in allowed_clean) or (
        not is_clean and intervention_norm != "randombackground"
    ):
        _metadata_error(
            errors,
            variant,
            "intervention_mismatch",
            "intervention must be 'none' for clean and 'random_background' for styles",
            {"value": None if intervention is _MISSING else intervention},
        )

    source_episode = _find_value(
        metadata,
        ("source_clean_episode", "source_episode", "source.clean_episode", "source.episode"),
    )
    if source_episode is _MISSING or not _validate_source_episode(source_episode):
        _metadata_error(
            errors,
            variant,
            "source_clean_episode_invalid",
            "metadata must identify clean episode0 as the replay source",
            {"value": None if source_episode is _MISSING else source_episode},
        )
    source_path = _find_value(
        metadata,
        ("source_trajectory_path", "source.trajectory_path", "trajectory_source", "source_path"),
    )
    if source_path is _MISSING or not isinstance(source_path, str) or not source_path.replace("\\", "/").endswith(
        "_traj_data/episode0.pkl"
    ):
        _metadata_error(
            errors,
            variant,
            "source_trajectory_path_invalid",
            "metadata source trajectory path must end in _traj_data/episode0.pkl",
            {"value": None if source_path is _MISSING else source_path},
        )
    source_hdf5_path = _find_value(
        metadata,
        ("source_clean_hdf5_path", "source.clean_hdf5_path", "source_hdf5_path"),
    )
    if (
        source_hdf5_path is _MISSING
        or not isinstance(source_hdf5_path, str)
        or not source_hdf5_path.replace("\\", "/").endswith("clean/data/episode0.hdf5")
    ):
        _metadata_error(
            errors,
            variant,
            "source_clean_hdf5_path_invalid",
            "metadata source clean HDF5 path must end in clean/data/episode0.hdf5",
            {"value": None if source_hdf5_path is _MISSING else source_hdf5_path},
        )

    success_raw = _find_value(metadata, ("success", "episode_success", "final_success"))
    success = _as_bool(success_raw)
    if success is not True:
        _metadata_error(
            errors,
            variant,
            "metadata_success_false",
            "metadata must explicitly report a successful final episode",
            {"value": None if success_raw is _MISSING else success_raw},
        )

    frame_count_raw = _find_value(metadata, ("frame_count", "frames", "episode_frame_count"))
    frame_count = _as_int(frame_count_raw)
    if frame_count is None or frame_count <= 0:
        _metadata_error(
            errors,
            variant,
            "frame_count_invalid",
            "metadata frame_count must be a positive integer",
            {"value": None if frame_count_raw is _MISSING else frame_count_raw},
        )
    frame_trace_raw = _find_value(
        metadata,
        ("frame_trace_index", "frame_trace_indices", "saved_frame_trace_index"),
    )
    frame_trace_index: list[int] | None = None
    if not isinstance(frame_trace_raw, Sequence) or isinstance(frame_trace_raw, (str, bytes, bytearray)):
        _metadata_error(
            errors,
            variant,
            "metadata_frame_trace_index_missing",
            "metadata must contain the saved-frame to state-trace index sequence",
        )
    else:
        converted = [_as_int(value) for value in frame_trace_raw]
        if any(value is None for value in converted):
            _metadata_error(
                errors,
                variant,
                "metadata_frame_trace_index_invalid",
                "metadata frame_trace_index must contain only integers",
                {"value": frame_trace_raw},
            )
        else:
            frame_trace_index = [int(value) for value in converted if value is not None]
            if not frame_trace_index or any(
                right < left for left, right in zip(frame_trace_index, frame_trace_index[1:])
            ):
                _metadata_error(
                    errors,
                    variant,
                    "metadata_frame_trace_index_decreases",
                    "metadata frame_trace_index must be non-empty and non-decreasing",
                    {"value": frame_trace_index},
                )
            if frame_count is not None and len(frame_trace_index) != frame_count:
                _metadata_error(
                    errors,
                    variant,
                    "metadata_frame_trace_index_count_mismatch",
                    "metadata frame_trace_index length must equal frame_count",
                    {"indices": len(frame_trace_index), "frame_count": frame_count},
                )

    action_rows_raw = _find_value(metadata, ("action_rows", "action_count", "action_trace_rows"))
    trace_rows_raw = _find_value(metadata, ("trace_rows", "state_rows", "state_trace_rows"))
    action_rows = _as_int(action_rows_raw)
    trace_rows = _as_int(trace_rows_raw)
    if action_rows is None or action_rows <= 0:
        _metadata_error(
            errors,
            variant,
            "metadata_action_rows_invalid",
            "metadata action_rows must be a positive integer",
            {"value": None if action_rows_raw is _MISSING else action_rows_raw},
        )
    if trace_rows is None or trace_rows <= 0:
        _metadata_error(
            errors,
            variant,
            "metadata_trace_rows_invalid",
            "metadata trace_rows must be a positive integer",
            {"value": None if trace_rows_raw is _MISSING else trace_rows_raw},
        )
    if action_rows is not None and trace_rows is not None and trace_rows != action_rows + 1:
        _metadata_error(
            errors,
            variant,
            "metadata_trace_length_relation_invalid",
            "metadata must report trace_rows == action_rows + 1",
            {"action_rows": action_rows, "trace_rows": trace_rows},
        )

    planner_calls_raw = _find_value(metadata, ("planner_calls", "planning_calls", "planner_call_count"))
    planner_calls = _as_int(planner_calls_raw)
    if planner_calls != 0:
        _metadata_error(
            errors,
            variant,
            "planner_calls_nonzero",
            "published clean/style variants are replays and must make exactly zero planner calls",
            {"value": None if planner_calls_raw is _MISSING else planner_calls_raw},
        )

    render_device = _validate_render_device_metadata(
        metadata,
        variant=variant,
        errors=errors,
    )
    domain = _validate_domain_randomization(
        metadata,
        is_clean=is_clean,
        variant=variant,
        errors=errors,
    )
    path_consumption = _validate_path_consumption(
        metadata,
        source_counts,
        variant=variant,
        errors=errors,
    )
    return {
        "schema_version": schema_version,
        "task": None if task is _MISSING else task,
        "content_id": content_id,
        "content_seed": content_seed,
        "variant": canonical["variant"],
        "style_index": canonical["style_index"],
        "style_seed": style_seed,
        "split": canonical["split"],
        "intervention": None if intervention is _MISSING else intervention,
        "task_identity": canonical["task_identity"],
        "task_identity_sha256": canonical["task_identity_sha256"],
        "task_success_spec": canonical["task_success_spec"],
        "task_success_spec_sha256": canonical["task_success_spec_sha256"],
        "task_state_layout": canonical["task_state_layout"],
        "trace_schema": canonical["trace_schema"],
        "need_plan": canonical["need_plan"],
        "rng_state_sha256_after_setup": canonical["rng_state_sha256_after_setup"],
        "style_rng": canonical["style_rng"],
        "textures": canonical["textures"],
        "success": success,
        "frame_count": frame_count,
        "frame_trace_index": frame_trace_index,
        "action_rows": action_rows,
        "trace_rows": trace_rows,
        "planner_calls": planner_calls,
        "render_device": render_device,
        "source_clean_episode": None if source_episode is _MISSING else source_episode,
        "source_trajectory_path": None if source_path is _MISSING else source_path,
        "source_clean_hdf5_path": None if source_hdf5_path is _MISSING else source_hdf5_path,
        "domain_randomization": domain,
        "path_consumption": path_consumption,
    }


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0)


def _validate_complete_marker(
    marker: Mapping[str, Any],
    *,
    expected_task: str,
    expected_content_id: int | None,
    expected_content_seed: int | None,
    expected_style_seeds: Sequence[int],
    source_trajectory_sha256: str | None,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the canonical publication commit record without fuzzy aliases."""

    artifact = "COMPLETE.json"
    schema_version = _as_json_int(marker.get("schema_version", _MISSING))
    if schema_version != SCHEMA_VERSION:
        _issue(
            errors,
            "complete_marker_schema_mismatch",
            f"COMPLETE.json schema_version must be exactly {SCHEMA_VERSION}",
            artifact=artifact,
            details={"value": marker.get("schema_version")},
        )

    task = marker.get("task", _MISSING)
    if task is _MISSING or not isinstance(task, str) or task != expected_task:
        _issue(
            errors,
            "complete_marker_task_mismatch",
            f"COMPLETE.json task must be exactly {expected_task!r}",
            artifact=artifact,
            details={"value": None if task is _MISSING else task},
        )

    content_id = _as_json_int(marker.get("content_id", _MISSING))
    if content_id is None or (expected_content_id is not None and content_id != expected_content_id):
        _issue(
            errors,
            "complete_marker_id_mismatch",
            "COMPLETE.json content_id is missing or incorrect",
            artifact=artifact,
            details={"expected": expected_content_id, "actual": content_id},
        )

    content_seed = _as_json_int(marker.get("content_seed", _MISSING))
    if content_seed is None or (
        expected_content_seed is not None and content_seed != expected_content_seed
    ):
        _issue(
            errors,
            "complete_marker_seed_mismatch",
            "COMPLETE.json content_seed is missing or incorrect",
            artifact=artifact,
            details={"expected": expected_content_seed, "actual": content_seed},
        )

    expected_split: str | None = None
    if expected_content_id is not None and 0 <= expected_content_id < TOTAL_CONTENTS:
        expected_split = split_for_content(expected_content_id)
    split = marker.get("split", _MISSING)
    if split != expected_split:
        _issue(
            errors,
            "complete_marker_split_mismatch",
            "COMPLETE.json split must exactly match the fixed content-ID split",
            artifact=artifact,
            details={
                "expected": expected_split,
                "actual": None if split is _MISSING else split,
            },
        )

    identity_raw = marker.get("task_identity", _MISSING)
    task_identity = dict(identity_raw) if isinstance(identity_raw, Mapping) else None
    for message in validate_task_identity(expected_task, identity_raw):
        _issue(
            errors,
            "complete_marker_task_identity_invalid",
            message,
            artifact=artifact,
            details={"value": None if identity_raw is _MISSING else identity_raw},
        )
    identity_hash = marker.get("task_identity_sha256", _MISSING)
    expected_identity_hash = (
        canonical_json_sha256(task_identity) if task_identity is not None else None
    )
    if (
        not _valid_hash(identity_hash)
        or expected_identity_hash is None
        or str(identity_hash).lower() != expected_identity_hash
    ):
        _issue(
            errors,
            "complete_marker_task_identity_hash_mismatch",
            "COMPLETE.json task_identity_sha256 must hash its task_identity",
            artifact=artifact,
            details={
                "declared": None if identity_hash is _MISSING else identity_hash,
                "expected": expected_identity_hash,
            },
        )

    success_spec_hash = marker.get("task_success_spec_sha256", _MISSING)
    if not _valid_hash(success_spec_hash):
        _issue(
            errors,
            "complete_marker_task_success_spec_hash_invalid",
            "COMPLETE.json task_success_spec_sha256 must be a valid SHA-256",
            artifact=artifact,
            details={
                "value": None if success_spec_hash is _MISSING else success_spec_hash
            },
        )

    task_state_layout, layout_errors = _canonical_task_state_layout(
        expected_task, marker.get("task_state_layout", _MISSING)
    )
    for message in layout_errors:
        _issue(
            errors,
            "complete_marker_task_state_layout_invalid",
            message,
            artifact=artifact,
            details={"value": marker.get("task_state_layout")},
        )

    raw_style_seeds = marker.get("style_seeds", _MISSING)
    parsed_style_seeds: list[int] | None = None
    if isinstance(raw_style_seeds, list):
        converted = [_as_json_int(value) for value in raw_style_seeds]
        if all(value is not None for value in converted):
            parsed_style_seeds = [int(value) for value in converted if value is not None]
    expected_styles = [int(value) for value in expected_style_seeds]
    if parsed_style_seeds != expected_styles:
        _issue(
            errors,
            "complete_marker_style_seeds_mismatch",
            "COMPLETE.json style_seeds must exactly match the three ordered style seeds",
            artifact=artifact,
            details={"expected": expected_styles, "actual": parsed_style_seeds},
        )

    declared_source_hash = marker.get("source_trajectory_sha256", _MISSING)
    if not _valid_hash(declared_source_hash) or (
        source_trajectory_sha256 is not None
        and str(declared_source_hash).lower() != source_trajectory_sha256.lower()
    ):
        _issue(
            errors,
            "complete_marker_source_hash_mismatch",
            "COMPLETE.json source_trajectory_sha256 is missing, invalid, or stale",
            artifact=artifact,
            details={
                "declared": None if declared_source_hash is _MISSING else declared_source_hash,
                "actual": source_trajectory_sha256,
            },
        )

    rng_state = marker.get("rng_state_sha256_after_setup", _MISSING)
    if not _valid_hash(rng_state):
        _issue(
            errors,
            "complete_marker_rng_state_hash_invalid",
            "COMPLETE.json rng_state_sha256_after_setup must be a valid SHA-256",
            artifact=artifact,
            details={"value": None if rng_state is _MISSING else rng_state},
        )

    completed_at = marker.get("completed_at", _MISSING)
    if not _is_utc_timestamp(completed_at):
        _issue(
            errors,
            "complete_marker_completed_at_invalid",
            "COMPLETE.json completed_at must be a timezone-aware UTC ISO-8601 timestamp",
            artifact=artifact,
            details={"value": None if completed_at is _MISSING else completed_at},
        )

    validation = marker.get("validation", _MISSING)
    validation_valid: bool | None = None
    if isinstance(validation, Mapping):
        validation_valid = validation.get("valid") if isinstance(validation.get("valid"), bool) else None
    if validation_valid is not True:
        _issue(
            errors,
            "complete_marker_validation_false",
            "COMPLETE.json validation.valid must be the boolean true",
            artifact=artifact,
            details={"validation": None if validation is _MISSING else validation},
        )
    if isinstance(validation, Mapping):
        validation_errors = validation.get("errors", _MISSING)
        if validation_errors != []:
            _issue(
                errors,
                "complete_marker_validation_errors",
                "COMPLETE.json validation.errors must be an empty list",
                artifact=artifact,
                details={
                    "errors": None if validation_errors is _MISSING else validation_errors
                },
            )
        validation_warnings = validation.get("warnings", _MISSING)
        if not isinstance(validation_warnings, list):
            _issue(
                errors,
                "complete_marker_validation_warnings_invalid",
                "COMPLETE.json validation.warnings must be a list",
                artifact=artifact,
                details={
                    "warnings": None
                    if validation_warnings is _MISSING
                    else validation_warnings
                },
            )

    return {
        "schema_version": schema_version,
        "task": None if task is _MISSING else task,
        "content_id": content_id,
        "content_seed": content_seed,
        "split": None if split is _MISSING else split,
        "task_identity": task_identity,
        "task_identity_sha256": None if identity_hash is _MISSING else identity_hash,
        "task_success_spec_sha256": (
            None if success_spec_hash is _MISSING else success_spec_hash
        ),
        "task_state_layout": task_state_layout,
        "style_seeds": parsed_style_seeds,
        "source_trajectory_sha256": (
            None if declared_source_hash is _MISSING else declared_source_hash
        ),
        "rng_state_sha256_after_setup": None if rng_state is _MISSING else rng_state,
        "completed_at": None if completed_at is _MISSING else completed_at,
        "validation_valid": validation_valid,
    }


def validate_content_dir(
    content_dir: str | os.PathLike[str],
    *,
    expected_task: str = DEFAULT_TASK,
    expected_content_id: int | None = None,
    expected_content_seed: int | None = None,
    expected_style_seeds: Sequence[int] = DEFAULT_STYLE_SEEDS,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate one clean + three-style content group without changing it.

    Args:
        content_dir: ``.../contents/content_XXXXXX`` directory.
        expected_task: Exact task name required in every metadata file.
        expected_content_id: Optional exact ID.  When omitted, it is inferred
            from a canonical ``content_XXXXXX`` directory name if possible.
        expected_content_seed: Optional exact content RNG seed.
        expected_style_seeds: Exact ordered style-seed set.  Directory ordinal
            is the position in this sequence.
        require_complete: Require a published ``COMPLETE.json`` marker.  Leave
            false when validating a staged directory before publication.

    Returns:
        A JSON-serialisable report.  No dataset file is ever modified.
    """
    directory = Path(content_dir).expanduser().resolve()
    style_seeds = tuple(int(seed) for seed in expected_style_seeds)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "content_dir": str(directory),
        "expected_task": expected_task,
        "expected_content_id": expected_content_id,
        "expected_content_seed": expected_content_seed,
        "expected_style_seeds": list(style_seeds),
        "valid": False,
        "errors": errors,
        "warnings": warnings,
        "variants": [],
        "comparisons": [],
    }
    if len(style_seeds) != 3 or len(set(style_seeds)) != len(style_seeds):
        _issue(
            errors,
            "expected_style_seeds_invalid",
            "strict paired content requires exactly three distinct expected style seeds",
            details={"style_seeds": list(style_seeds)},
        )
        return report
    if expected_task not in SUPPORTED_TASKS:
        _issue(
            errors,
            "unsupported_task",
            "strict paired validation supports only the adapter-declared tasks",
            details={"task": expected_task, "supported": list(SUPPORTED_TASKS)},
        )
        return report
    if not directory.is_dir():
        _issue(errors, "content_directory_missing", "content directory does not exist", details={"path": directory})
        return report
    match = CONTENT_RE.fullmatch(directory.name)
    inferred_id = int(match.group(1)) if match else None
    if expected_content_id is None:
        expected_content_id = inferred_id
        report["expected_content_id"] = expected_content_id
    elif inferred_id is not None and inferred_id != expected_content_id:
        _issue(
            errors,
            "content_directory_id_mismatch",
            "content directory name does not match expected_content_id",
            details={"directory_id": inferred_id, "expected": expected_content_id},
        )
    if expected_content_id is None or expected_content_id < 0 or expected_content_id >= TOTAL_CONTENTS:
        _issue(
            errors,
            "content_id_outside_fixed_split",
            f"content ID must be in the fixed range 0..{TOTAL_CONTENTS - 1}",
            details={"content_id": expected_content_id},
        )
    # A staged group is intentionally named with attempt/seed suffixes.  The
    # generator supplies expected_content_id while validating it before the
    # atomic publish rename.  Published/aggregate validation remains strict.
    if match is None and (require_complete or expected_content_id is None):
        _issue(
            errors,
            "content_directory_name_invalid",
            "content directory must be named content_XXXXXX",
            details={"name": directory.name},
        )

    expected_variants = ["clean"] + [f"style_{index:02d}_seed_{seed}" for index, seed in enumerate(style_seeds)]
    allowed_content_entries = set(expected_variants) | {"COMPLETE.json"}
    actual_entries = {entry.name for entry in directory.iterdir()}
    missing_variants = sorted(set(expected_variants) - actual_entries)
    extra_entries = sorted(actual_entries - allowed_content_entries)
    if missing_variants or extra_entries:
        _issue(
            errors,
            "content_layout_mismatch",
            "content directory does not have the exact expected variant layout",
            details={"missing": missing_variants, "extra": extra_entries},
        )
    complete_path = directory / "COMPLETE.json"
    if require_complete and not complete_path.is_file():
        _issue(errors, "complete_marker_missing", "published content is missing COMPLETE.json")
    complete_marker: Mapping[str, Any] | None = None
    if complete_path.exists():
        try:
            complete_marker = _read_json(complete_path)
        except Exception as exc:
            _issue(
                errors,
                "complete_marker_invalid",
                f"cannot parse COMPLETE.json: {exc}",
                artifact="COMPLETE.json",
            )

    clean_dir = directory / "clean"
    clean_paths = _validate_exact_variant_layout(clean_dir, is_clean=True, variant="clean", errors=errors)
    source_path = clean_paths[SOURCE_TRAJECTORY]
    source_counts: dict[str, int] | None = None
    source_hash: str | None = None
    if source_path.is_file():
        try:
            _, source_counts = _load_source_trajectory(source_path)
            source_hash = _sha256_file(source_path)
            report["source_trajectory"] = {
                "path": str(source_path),
                "sha256": source_hash,
                "left_path_entries": source_counts["left"],
                "right_path_entries": source_counts["right"],
            }
        except Exception as exc:
            _issue(
                errors,
                "source_trajectory_invalid",
                f"cannot validate source trajectory pickle: {exc}",
                variant="clean",
                artifact=SOURCE_TRAJECTORY,
            )
    if source_counts is None:
        # Continue collecting useful diagnostics without claiming that zero is a
        # valid source count; the source error above keeps the group invalid.
        source_counts = {"left": 0, "right": 0}
    if complete_marker is not None:
        report["complete_marker"] = _validate_complete_marker(
            complete_marker,
            expected_task=expected_task,
            expected_content_id=expected_content_id,
            expected_content_seed=expected_content_seed,
            expected_style_seeds=style_seeds,
            source_trajectory_sha256=source_hash,
            errors=errors,
        )

    variant_specs: list[tuple[str, Path, bool, int | None, int | None]] = [
        ("clean", clean_dir, True, None, None)
    ]
    variant_specs.extend(
        (
            f"style_{index:02d}_seed_{seed}",
            directory / f"style_{index:02d}_seed_{seed}",
            False,
            index,
            seed,
        )
        for index, seed in enumerate(style_seeds)
    )
    runtime: dict[str, dict[str, Any]] = {}
    for variant_name, variant_dir, is_clean, style_index, style_seed in variant_specs:
        if not variant_dir.is_dir():
            _issue(
                errors,
                "variant_directory_missing",
                "expected variant directory is missing",
                variant=variant_name,
                details={"path": variant_dir},
            )
            continue
        paths = _validate_exact_variant_layout(
            variant_dir,
            is_clean=is_clean,
            variant=variant_name,
            errors=errors,
        )
        record: dict[str, Any] = {
            "name": variant_name,
            "path": str(variant_dir),
            "is_clean": is_clean,
            "expected_style_index": style_index,
            "expected_style_seed": style_seed,
            "files": {},
        }
        internal: dict[str, Any] = {"paths": paths}
        metadata: Mapping[str, Any] | None = None
        if paths["metadata.json"].is_file():
            try:
                metadata = _read_json(paths["metadata.json"])
                internal["metadata"] = metadata
                record["metadata"] = _validate_variant_metadata(
                    metadata,
                    variant=variant_name,
                    is_clean=is_clean,
                    expected_task=expected_task,
                    expected_content_id=expected_content_id,
                    expected_content_seed=expected_content_seed,
                    expected_style_index=style_index,
                    expected_style_seed=style_seed,
                    source_counts=source_counts,
                    errors=errors,
                )
            except Exception as exc:
                _issue(
                    errors,
                    "metadata_invalid",
                    f"cannot parse/validate metadata.json: {exc}",
                    variant=variant_name,
                    artifact="metadata.json",
                )

        file_map = {
            "initial_state": "initial_state.npz",
            "action_trace": "action_trace.npz",
            "state_trace": "state_trace.npz",
            "hdf5": "data/episode0.hdf5",
            "video": "video/episode0.mp4",
        }
        actual_hashes: dict[str, str] = {}
        for kind, relative in file_map.items():
            artifact_path = paths[relative]
            if not artifact_path.is_file():
                continue
            try:
                digest = _sha256_file(artifact_path)
                actual_hashes[kind] = digest
                record["files"][relative] = {"sha256": digest, "size_bytes": artifact_path.stat().st_size}
                if metadata is not None:
                    _require_declared_hash(metadata, kind, digest, errors, variant_name)
            except Exception as exc:
                _issue(
                    errors,
                    "artifact_hash_failed",
                    f"cannot hash {relative}: {exc}",
                    variant=variant_name,
                    artifact=relative,
                )
        if metadata is not None and source_hash is not None:
            _require_declared_hash(metadata, "source_trajectory", source_hash, errors, variant_name)
            for source_kind in ("source_trajectory", "initial_state", "action_trace", "state_trace"):
                expected_source_hash = source_hash if source_kind == "source_trajectory" else None
                if source_kind != "source_trajectory":
                    # Clean hashes become available after this loop.  Defer style
                    # source-artifact validation to the cross-variant phase.
                    continue
                _require_declared_hash(
                    metadata,
                    source_kind,
                    expected_source_hash,
                    errors,
                    variant_name,
                    source=True,
                )
        internal["actual_hashes"] = actual_hashes

        for kind, relative in (
            ("initial_state", "initial_state.npz"),
            ("action_trace", "action_trace.npz"),
            ("state_trace", "state_trace.npz"),
        ):
            if not paths[relative].is_file():
                continue
            try:
                arrays = _load_npz(paths[relative])
                internal[kind] = arrays
                record[kind] = {
                    "arrays": {
                        key: {"shape": list(array.shape), "dtype": str(array.dtype)}
                        for key, array in sorted(arrays.items())
                    }
                }
            except Exception as exc:
                _issue(
                    errors,
                    "npz_invalid",
                    f"cannot safely load {relative} with allow_pickle=False: {exc}",
                    variant=variant_name,
                    artifact=relative,
                )

        if is_clean and "initial_state" in internal and isinstance(
            record.get("metadata"), Mapping
        ):
            _validate_task_path_arms(
                expected_task,
                source_counts,
                internal["initial_state"],
                record["metadata"].get("task_state_layout"),
                errors=errors,
            )

        if paths["data/episode0.hdf5"].is_file():
            try:
                inventory = _hdf5_inventory(paths["data/episode0.hdf5"])
                internal["hdf5_inventory"] = inventory
                record["hdf5"] = {
                    "frame_count": inventory["frame_count"],
                    "frame_count_values": inventory["frame_count_values"],
                    "dataset_count": len(inventory["leaves"]),
                    "non_rgb_dataset_count": len(inventory["non_rgb_leaves"]),
                    "head_rgb_path": inventory["head_rgb_path"],
                    "head_rgb_sha256": inventory["head_rgb_sha256"],
                    "head_rgb_shape": inventory["head_rgb_shape"],
                }
                if inventory["frame_count"] is None or inventory["frame_count"] <= 0:
                    _issue(
                        errors,
                        "hdf5_frame_counts_inconsistent",
                        "all HDF5 leaves must have the same positive leading frame dimension",
                        variant=variant_name,
                        artifact="data/episode0.hdf5",
                        details={"counts": inventory["dataset_frame_counts"]},
                    )
                _validate_hdf5_semantic_coverage(inventory, variant_name, errors)
                if metadata is not None:
                    _require_declared_hash(
                        metadata,
                        "head_rgb",
                        inventory["head_rgb_sha256"],
                        errors,
                        variant_name,
                    )
            except Exception as exc:
                _issue(
                    errors,
                    "hdf5_invalid",
                    f"cannot inspect HDF5: {exc}",
                    variant=variant_name,
                    artifact="data/episode0.hdf5",
                )

        if paths["video/episode0.mp4"].is_file():
            try:
                video_frames = _video_frame_count(paths["video/episode0.mp4"])
                internal["video_frame_count"] = video_frames
                record["video_frame_count"] = video_frames
            except Exception as exc:
                _issue(
                    errors,
                    "video_invalid",
                    f"cannot count decoded video frames: {exc}",
                    variant=variant_name,
                    artifact="video/episode0.mp4",
                )

        frame_count = None
        if "hdf5_inventory" in internal:
            frame_count = internal["hdf5_inventory"]["frame_count"]
        if metadata is not None and "metadata" in record:
            declared_frames = record["metadata"]["frame_count"]
            if frame_count is not None and declared_frames != frame_count:
                _metadata_error(
                    errors,
                    variant_name,
                    "metadata_frame_count_mismatch",
                    "metadata frame_count differs from HDF5",
                    {"metadata": declared_frames, "hdf5": frame_count},
                )
        if frame_count is not None and "video_frame_count" in internal and internal["video_frame_count"] != frame_count:
            _issue(
                errors,
                "video_frame_count_mismatch",
                "decoded video frame count differs from HDF5",
                variant=variant_name,
                artifact="video/episode0.mp4",
                details={"video": internal["video_frame_count"], "hdf5": frame_count},
            )
        if all(kind in internal for kind in ("initial_state", "action_trace", "state_trace")):
            metadata_summary = record.get("metadata", {})
            semantic_checks = _validate_trace_semantics(
                internal["initial_state"],
                internal["action_trace"],
                internal["state_trace"],
                task_name=expected_task,
                task_state_layout=metadata_summary.get("task_state_layout"),
                task_success_spec=metadata_summary.get("task_success_spec"),
                frame_count=frame_count,
                variant=variant_name,
                errors=errors,
            )
            record["semantic_checks"] = semantic_checks
            if "metadata" in record:
                for field, actual in (
                    ("action_rows", semantic_checks["action_length"]),
                    ("trace_rows", semantic_checks["state_length"]),
                ):
                    declared = metadata_summary.get(field)
                    if actual is not None and declared != actual:
                        _metadata_error(
                            errors,
                            variant_name,
                            f"metadata_{field}_mismatch",
                            f"metadata {field} differs from the trace NPZ",
                            {"metadata": declared, "trace": actual},
                        )
                frame_entry = (
                    ("frame_trace_index", internal["state_trace"]["frame_trace_index"])
                    if "frame_trace_index" in internal["state_trace"]
                    else None
                )
                if frame_entry is not None:
                    actual_indices = [int(value) for value in np.asarray(frame_entry[1]).reshape(-1)]
                    if metadata_summary.get("frame_trace_index") != actual_indices:
                        _metadata_error(
                            errors,
                            variant_name,
                            "metadata_frame_trace_index_mismatch",
                            "metadata frame_trace_index differs exactly from the trace NPZ",
                            {
                                "metadata": metadata_summary.get("frame_trace_index"),
                                "trace": actual_indices,
                            },
                        )
        if "state_trace" in internal and paths["data/episode0.hdf5"].is_file():
            record["hdf5_trace_alignment"] = _validate_trace_hdf5_alignment(
                internal["state_trace"],
                paths["data/episode0.hdf5"],
                variant=variant_name,
                errors=errors,
            )
        if metadata is not None:
            textures: dict[str, Any] = {}
            for surface in ("wall", "table"):
                fields = _texture_fields(metadata, surface)
                public_fields = {
                    key: None if value is _MISSING else value for key, value in fields.items()
                }
                if is_clean:
                    non_null = [
                        value
                        for value in fields.values()
                        if value is not _MISSING and value is not None and value != ""
                    ]
                    if non_null:
                        _metadata_error(
                            errors,
                            variant_name,
                            "clean_texture_not_null",
                            f"clean {surface} texture identifier/path/hash must all be null",
                            {"fields": public_fields},
                        )
                else:
                    identifier = fields["identifier"]
                    digest = fields["sha256"]
                    if identifier is _MISSING or identifier is None or str(identifier).strip() == "":
                        _metadata_error(
                            errors,
                            variant_name,
                            "style_texture_missing",
                            f"style {surface} texture must be non-null",
                        )
                    resolved = _resolve_texture_file(fields, variant_dir=variant_dir, content_dir=directory)
                    if resolved is None:
                        _metadata_error(
                            errors,
                            variant_name,
                            "texture_file_missing",
                            f"cannot resolve the selected {surface} texture file",
                            {"fields": public_fields},
                        )
                    else:
                        actual_texture_hash = _sha256_file(resolved)
                        fields = dict(fields)
                        fields["resolved_path"] = str(resolved)
                        fields["actual_sha256"] = actual_texture_hash
                        public_fields.update(
                            {"resolved_path": str(resolved), "actual_sha256": actual_texture_hash}
                        )
                        if not _valid_hash(digest):
                            _metadata_error(
                                errors,
                                variant_name,
                                "texture_hash_missing_or_invalid",
                                f"metadata {surface} texture SHA-256 is missing or invalid",
                                {"declared": None if digest is _MISSING else digest},
                            )
                        elif str(digest).lower() != actual_texture_hash:
                            _metadata_error(
                                errors,
                                variant_name,
                                "texture_hash_mismatch",
                                f"metadata {surface} texture SHA-256 does not match the selected file",
                                {"declared": str(digest).lower(), "actual": actual_texture_hash},
                            )
                textures[surface] = fields
                public_fields = {key: _json_value(value) for key, value in public_fields.items()}
                record.setdefault("textures", {})[surface] = public_fields
            internal["textures"] = textures
        runtime[variant_name] = internal
        report["variants"].append(record)

    clean = runtime.get("clean")
    if complete_marker is not None and clean is not None and isinstance(
        clean.get("metadata"), Mapping
    ):
        clean_metadata = clean["metadata"]
        marker_seed = _as_int(complete_marker.get("content_seed", _MISSING))
        clean_seed = _as_int(_find_value(clean_metadata, ("content_seed", "content.seed")))
        marker_id = _as_int(complete_marker.get("content_id", _MISSING))
        clean_id = _as_int(_find_value(clean_metadata, ("content_id", "content.id")))
        marker_rng_state = complete_marker.get("rng_state_sha256_after_setup", _MISSING)
        clean_rng_state = clean_metadata.get("rng_state_sha256_after_setup", _MISSING)
        marker_split = complete_marker.get("split", _MISSING)
        clean_split = clean_metadata.get("split", _MISSING)
        marker_identity_hash = complete_marker.get("task_identity_sha256", _MISSING)
        clean_identity_hash = clean_metadata.get("task_identity_sha256", _MISSING)
        marker_success_spec_hash = complete_marker.get(
            "task_success_spec_sha256", _MISSING
        )
        clean_success_spec_hash = clean_metadata.get(
            "task_success_spec_sha256", _MISSING
        )
        marker_layout = complete_marker.get("task_state_layout", _MISSING)
        clean_layout = clean_metadata.get("task_state_layout", _MISSING)
        for field, marker_value, clean_value in (
            ("content_id", marker_id, clean_id),
            ("content_seed", marker_seed, clean_seed),
            ("split", marker_split, clean_split),
            ("task_identity_sha256", marker_identity_hash, clean_identity_hash),
            (
                "task_success_spec_sha256",
                marker_success_spec_hash,
                clean_success_spec_hash,
            ),
            ("task_state_layout", marker_layout, clean_layout),
            ("rng_state_sha256_after_setup", marker_rng_state, clean_rng_state),
        ):
            if (
                marker_value is _MISSING
                or clean_value is _MISSING
                or marker_value is None
                or clean_value is None
                or marker_value != clean_value
            ):
                _issue(
                    errors,
                    f"complete_marker_{field}_metadata_mismatch",
                    f"COMPLETE.json {field} does not match clean/metadata.json",
                    artifact="COMPLETE.json",
                    details={
                        "marker": None if marker_value is _MISSING else marker_value,
                        "clean_metadata": None if clean_value is _MISSING else clean_value,
                    },
                )
    if clean is not None:
        clean_meta = clean.get("metadata")
        clean_hashes = clean.get("actual_hashes", {})
        clean_source_artifacts = dict(clean_hashes)
        if "hdf5_inventory" in clean:
            clean_source_artifacts["head_rgb"] = clean["hdf5_inventory"]["head_rgb_sha256"]

        def validate_clean_hdf5_reference(metadata: Mapping[str, Any], variant_name: str) -> None:
            if "hdf5" not in clean_source_artifacts:
                return
            declared = _find_value(
                metadata,
                (
                    "source_clean_hdf5_sha256",
                    "source.clean_hdf5_sha256",
                    "source_hdf5_sha256",
                ),
            )
            if not _valid_hash(declared):
                _metadata_error(
                    errors,
                    variant_name,
                    "source_clean_hdf5_hash_missing_or_invalid",
                    "metadata must declare a valid source_clean_hdf5_sha256",
                    {"value": None if declared is _MISSING else declared},
                )
            elif str(declared).lower() != clean_source_artifacts["hdf5"]:
                _metadata_error(
                    errors,
                    variant_name,
                    "source_clean_hdf5_hash_mismatch",
                    "source_clean_hdf5_sha256 does not match clean/data/episode0.hdf5",
                    {
                        "declared": str(declared).lower(),
                        "actual": clean_source_artifacts["hdf5"],
                    },
                )

        if clean_meta is not None:
            validate_clean_hdf5_reference(clean_meta, "clean")
            for source_kind in (
                "initial_state",
                "action_trace",
                "state_trace",
                "hdf5",
                "video",
                "head_rgb",
            ):
                if source_kind in clean_source_artifacts:
                    _require_declared_hash(
                        clean_meta,
                        source_kind,
                        clean_source_artifacts[source_kind],
                        errors,
                        "clean",
                        source=True,
                    )
        for variant_name, _, is_clean, _, _ in variant_specs:
            if is_clean or variant_name not in runtime:
                continue
            other = runtime[variant_name]
            other_meta = other.get("metadata")
            if other_meta is not None:
                validate_clean_hdf5_reference(other_meta, variant_name)
                for source_kind in (
                    "initial_state",
                    "action_trace",
                    "state_trace",
                    "hdf5",
                    "video",
                    "head_rgb",
                ):
                    if source_kind in clean_source_artifacts:
                        _require_declared_hash(
                            other_meta,
                            source_kind,
                            clean_source_artifacts[source_kind],
                            errors,
                            variant_name,
                            source=True,
                        )
            for kind, artifact in (
                ("initial_state", "initial_state.npz"),
                ("action_trace", "action_trace.npz"),
                ("state_trace", "state_trace.npz"),
            ):
                if kind in clean and kind in other:
                    report["comparisons"].append(
                        _compare_array_mappings(
                            clean[kind],
                            other[kind],
                            artifact=artifact,
                            variant_name=variant_name,
                            errors=errors,
                        )
                    )
            if "hdf5_inventory" in clean and "hdf5_inventory" in other:
                report["comparisons"].append(
                    _compare_hdf5_non_rgb(
                        clean["paths"]["data/episode0.hdf5"],
                        other["paths"]["data/episode0.hdf5"],
                        clean["hdf5_inventory"],
                        other["hdf5_inventory"],
                        variant_name,
                        errors,
                    )
                )

    # Cross-variant metadata invariants are checked explicitly in addition to
    # byte comparisons, so a malformed/mislabelled group cannot pass.
    metadata_summaries = {
        record["name"]: record.get("metadata")
        for record in report["variants"]
        if isinstance(record.get("metadata"), Mapping)
    }
    for field in (
        "task",
        "content_id",
        "content_seed",
        "split",
        "source_clean_episode",
        "task_identity",
        "task_identity_sha256",
        "task_success_spec",
        "task_success_spec_sha256",
        "task_state_layout",
        "rng_state_sha256_after_setup",
        "render_device",
    ):
        values = {name: summary.get(field) for name, summary in metadata_summaries.items()}
        if len(values) != len(expected_variants):
            _issue(
                errors,
                "cross_variant_metadata_incomplete",
                f"metadata field {field!r} is unavailable for one or more variants",
                artifact="metadata.json",
                details={"values": values},
            )
        elif len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            _issue(
                errors,
                "cross_variant_metadata_mismatch",
                f"metadata field {field!r} is not identical across clean/styles",
                artifact="metadata.json",
                details={"values": values},
            )

    if complete_marker is not None and metadata_summaries:
        marker_pairs = (
            ("task_identity", complete_marker.get("task_identity", _MISSING)),
            (
                "task_identity_sha256",
                complete_marker.get("task_identity_sha256", _MISSING),
            ),
            (
                "task_success_spec_sha256",
                complete_marker.get("task_success_spec_sha256", _MISSING),
            ),
            ("task_state_layout", complete_marker.get("task_state_layout", _MISSING)),
            ("split", complete_marker.get("split", _MISSING)),
        )
        clean_summary = metadata_summaries.get("clean", {})
        for field, marker_value in marker_pairs:
            clean_value = clean_summary.get(field, _MISSING)
            if marker_value is _MISSING or clean_value is _MISSING or marker_value != clean_value:
                _issue(
                    errors,
                    f"complete_marker_{field}_metadata_mismatch",
                    f"COMPLETE.json {field} does not match clean/metadata.json",
                    artifact="COMPLETE.json",
                    details={
                        "marker": None if marker_value is _MISSING else marker_value,
                        "clean_metadata": None if clean_value is _MISSING else clean_value,
                    },
                )

    style_runtimes = [(name, runtime[name]) for name in expected_variants[1:] if name in runtime]
    for surface in ("wall", "table"):
        identifiers: list[tuple[str, str]] = []
        hashes: list[tuple[str, str]] = []
        for name, data in style_runtimes:
            fields = data.get("textures", {}).get(surface, {})
            identifier = fields.get("identifier", _MISSING)
            digest = fields.get("actual_sha256", _MISSING)
            if identifier is not _MISSING and identifier is not None:
                identifiers.append((name, str(identifier)))
            if digest is not _MISSING:
                hashes.append((name, str(digest)))
        if len(identifiers) == 3 and len({value for _, value in identifiers}) != 3:
            _issue(
                errors,
                "style_texture_not_pairwise_distinct",
                f"the three style {surface} texture identifiers must be pairwise distinct",
                details={"values": dict(identifiers)},
            )
        if len(hashes) == 3 and len({value for _, value in hashes}) != 3:
            _issue(
                errors,
                "style_texture_bytes_not_pairwise_distinct",
                f"the three style {surface} texture files must have pairwise-distinct SHA-256 values",
                details={"values": dict(hashes)},
            )

    head_hashes: list[tuple[str, str]] = []
    for name in expected_variants:
        data = runtime.get(name)
        if data and "hdf5_inventory" in data:
            head_hashes.append((name, data["hdf5_inventory"]["head_rgb_sha256"]))
    if len(head_hashes) == 4 and len({digest for _, digest in head_hashes}) != 4:
        duplicated = {name: digest for name, digest in head_hashes}
        _issue(
            errors,
            "head_rgb_not_pairwise_different",
            "clean and all three styles must have pairwise-different head-camera RGB bytes",
            artifact="data/episode0.hdf5",
            details={"sha256": duplicated},
        )
    report["head_rgb_sha256"] = dict(head_hashes)
    report["valid"] = not errors
    report["summary"] = {
        "expected_variants": 4,
        "inspected_variants": len(runtime),
        "comparison_count": len(report["comparisons"]),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "valid": report["valid"],
    }
    return report


def _atomic_write_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def validate_dataset(
    root: str | os.PathLike[str],
    *,
    expected_task: str = DEFAULT_TASK,
    expected_contents: int = TOTAL_CONTENTS,
    expected_content_ids: Sequence[int] | None = None,
    expected_content_seeds: Sequence[int] | None = None,
    expected_style_seeds: Sequence[int] = DEFAULT_STYLE_SEEDS,
    require_complete: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate an exact published dataset and return report + valid variant rows.

    The returned manifest contains both clean and random-background rows.  Each
    row carries its fixed split, allowing the CLI to emit six split manifests.
    """
    dataset_root = Path(root).expanduser().resolve()
    contents_root = dataset_root if dataset_root.name == "contents" else dataset_root / "contents"
    errors: list[dict[str, Any]] = []
    if expected_task not in SUPPORTED_TASKS:
        _issue(
            errors,
            "unsupported_task",
            "strict paired validation supports only the adapter-declared tasks",
            details={"task": expected_task, "supported": list(SUPPORTED_TASKS)},
        )
    if expected_content_ids is None and (
        expected_contents < 0 or expected_contents > TOTAL_CONTENTS
    ):
        _issue(
            errors,
            "expected_contents_outside_fixed_split",
            f"expected_contents must be between 0 and {TOTAL_CONTENTS}",
            details={"expected_contents": expected_contents},
        )
    if expected_content_ids is None:
        content_ids = tuple(range(expected_contents))
    else:
        content_ids = tuple(int(value) for value in expected_content_ids)
        expected_contents = len(content_ids)
    content_seeds = None if expected_content_seeds is None else tuple(int(seed) for seed in expected_content_seeds)
    if content_seeds is not None and len(content_seeds) != len(content_ids):
        _issue(
            errors,
            "expected_content_seeds_length",
            "expected_content_seeds must have one entry per expected content ID",
            details={"seeds": list(content_seeds), "ids": list(content_ids)},
        )
    if len(set(content_ids)) != len(content_ids):
        _issue(errors, "expected_content_ids_duplicate", "expected content IDs must be distinct")
    invalid_ids = [
        content_id for content_id in content_ids if content_id < 0 or content_id >= TOTAL_CONTENTS
    ]
    if invalid_ids:
        _issue(
            errors,
            "expected_content_ids_outside_fixed_split",
            f"content IDs must be within 0..{TOTAL_CONTENTS - 1}",
            details={"ids": invalid_ids},
        )
    if content_seeds is not None and len(set(content_seeds)) != len(content_seeds):
        _issue(errors, "expected_content_seeds_duplicate", "expected content seeds must be distinct")

    expected_names = {f"content_{content_id:06d}" for content_id in content_ids}
    actual_names: set[str] = set()
    malformed_entries: list[str] = []
    if not contents_root.is_dir():
        _issue(
            errors,
            "contents_directory_missing",
            "dataset has no contents/ directory",
            details={"path": contents_root},
        )
    else:
        for entry in contents_root.iterdir():
            if entry.is_dir() and CONTENT_RE.fullmatch(entry.name):
                actual_names.add(entry.name)
            else:
                malformed_entries.append(entry.name)
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing:
            _issue(
                errors,
                "contents_incomplete",
                "one or more expected content directories are missing",
                details={"missing": missing},
            )
        if extra:
            _issue(
                errors,
                "extra_contents",
                "unexpected content directories are present",
                details={"extra": extra},
            )
        if malformed_entries:
            _issue(
                errors,
                "unexpected_contents_entries",
                "contents/ has entries that are not canonical content directories",
                details={"entries": sorted(malformed_entries)},
            )

    reports: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    observed_seeds: dict[int, int] = {}
    observed_render_devices: dict[int, dict[str, Any]] = {}
    style_seeds = tuple(int(seed) for seed in expected_style_seeds)
    for index, content_id in enumerate(content_ids):
        content_path = contents_root / f"content_{content_id:06d}"
        expected_seed = content_seeds[index] if content_seeds is not None and index < len(content_seeds) else None
        content_report = validate_content_dir(
            content_path,
            expected_task=expected_task,
            expected_content_id=content_id,
            expected_content_seed=expected_seed,
            expected_style_seeds=style_seeds,
            require_complete=require_complete,
        )
        reports.append(content_report)
        clean_record = next(
            (record for record in content_report.get("variants", []) if record.get("name") == "clean"),
            None,
        )
        content_seed = None
        if clean_record and isinstance(clean_record.get("metadata"), Mapping):
            content_seed = clean_record["metadata"].get("content_seed")
            render_device = clean_record["metadata"].get("render_device")
            if isinstance(render_device, Mapping):
                observed_render_devices[content_id] = dict(render_device)
        if isinstance(content_seed, int):
            observed_seeds[content_id] = content_seed
        if content_report.get("valid"):
            split = split_for_content(content_id)
            clean_path = str(Path("contents") / f"content_{content_id:06d}" / "clean")
            render_device = observed_render_devices.get(content_id)
            manifest.append(
                {
                    "task": expected_task,
                    "content_id": content_id,
                    "content_seed": content_seed,
                    "split": split,
                    "style_seed": None,
                    "intervention": "none",
                    "variant": "clean",
                    "path": clean_path,
                    "clean_path": clean_path,
                    "render_device": render_device,
                }
            )
            for style_index, style_seed in enumerate(style_seeds):
                variant_name = f"style_{style_index:02d}_seed_{style_seed}"
                manifest.append(
                    {
                        "task": expected_task,
                        "content_id": content_id,
                        "content_seed": content_seed,
                        "split": split,
                        "style_seed": style_seed,
                        "intervention": "random_background",
                        "variant": variant_name,
                        "path": str(
                            Path("contents") / f"content_{content_id:06d}" / variant_name
                        ),
                        "clean_path": clean_path,
                        "render_device": render_device,
                    }
                )
    reverse_seeds: dict[int, list[int]] = {}
    for content_id, seed in observed_seeds.items():
        reverse_seeds.setdefault(seed, []).append(content_id)
    duplicates = {seed: ids for seed, ids in reverse_seeds.items() if len(ids) > 1}
    if duplicates:
        _issue(
            errors,
            "duplicate_content_seeds",
            "content seeds must be unique across successful content trajectories",
            details={"duplicates": duplicates},
        )

    dataset_render_device: dict[str, Any] | None = None
    if observed_render_devices:
        canonical_devices = {
            json.dumps(device, sort_keys=True)
            for device in observed_render_devices.values()
        }
        if len(canonical_devices) != 1 or len(observed_render_devices) != len(content_ids):
            _issue(
                errors,
                "cross_content_render_device_mismatch",
                "all contents in one task dataset must use the same physical render GPU",
                details={"render_devices": observed_render_devices},
            )
        else:
            dataset_render_device = next(iter(observed_render_devices.values()))

    invalid_content_ids = [
        content_id for content_id, content_report in zip(content_ids, reports) if not content_report.get("valid")
    ]
    expected_random_variants = len(content_ids) * len(style_seeds)
    clean_manifest = [row for row in manifest if row["variant"] == "clean"]
    random_manifest = [row for row in manifest if row["variant"] != "clean"]
    expected_split_counts = {
        split: {
            "clean": sum(split_for_content(content_id) == split for content_id in content_ids),
            "random": len(style_seeds)
            * sum(split_for_content(content_id) == split for content_id in content_ids),
        }
        for split in SPLIT_COUNTS
        if all(0 <= content_id < TOTAL_CONTENTS for content_id in content_ids)
    }
    valid_split_counts = {
        split: {
            "clean": sum(
                row["split"] == split and row["variant"] == "clean" for row in manifest
            ),
            "random": sum(
                row["split"] == split and row["variant"] != "clean" for row in manifest
            ),
        }
        for split in SPLIT_COUNTS
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "contents_root": str(contents_root),
        "expected_task": expected_task,
        "expected_content_ids": list(content_ids),
        "expected_content_seeds": None if content_seeds is None else list(content_seeds),
        "expected_style_seeds": list(style_seeds),
        "valid": not errors
        and not invalid_content_ids
        and len(clean_manifest) == len(content_ids)
        and len(random_manifest) == expected_random_variants
        and valid_split_counts == expected_split_counts,
        "errors": errors,
        "invalid_content_ids": invalid_content_ids,
        "observed_content_seeds": observed_seeds,
        "render_device": dataset_render_device,
        "expected_split_counts": expected_split_counts,
        "valid_split_counts": valid_split_counts,
        "contents": reports,
        "summary": {
            "expected_contents": len(content_ids),
            "found_expected_content_dirs": len(expected_names & actual_names),
            "valid_contents": len(content_ids) - len(invalid_content_ids),
            "invalid_contents": len(invalid_content_ids),
            "expected_random_variants": expected_random_variants,
            "valid_clean_variants": len(clean_manifest),
            "valid_random_variants": len(random_manifest),
            "valid_variants": len(manifest),
            "split_counts": valid_split_counts,
            "aggregate_error_count": len(errors),
        },
    }
    return report, manifest


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate clean + three random-background RoboTwin replay variants.",
    )
    parser.add_argument("root", nargs="?", help="dataset output root (the parent of contents/)")
    parser.add_argument("--root", dest="root_option", help="dataset output root (alternative to positional root)")
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        default=DEFAULT_TASK,
        help=f"exact task name (default: {DEFAULT_TASK})",
    )
    parser.add_argument(
        "--expected-contents",
        "--expected-content-count",
        type=int,
        default=TOTAL_CONTENTS,
        help=f"exact number of expected content groups (default: {TOTAL_CONTENTS})",
    )
    parser.add_argument(
        "--content-ids",
        type=_parse_csv_ints,
        help="optional exact comma-separated content IDs (otherwise 0..N-1)",
    )
    parser.add_argument(
        "--content-seeds",
        type=_parse_csv_ints,
        help="optional exact comma-separated content RNG seeds, ordered like content IDs",
    )
    parser.add_argument(
        "--style-seeds",
        type=_parse_csv_ints,
        default=DEFAULT_STYLE_SEEDS,
        help="exact ordered comma-separated style seeds (default: 0,1,2)",
    )
    parser.add_argument("--report", help="report JSON path (default: ROOT/validation_report.json)")
    parser.add_argument(
        "--manifest",
        help="combined valid clean+random JSONL path (default: ROOT/valid_variants.jsonl)",
    )
    parser.add_argument(
        "--split-manifest-dir",
        help="six clean/random split JSONLs directory (default: ROOT/split_manifests)",
    )
    parser.add_argument(
        "--allow-missing-complete",
        action="store_true",
        help="allow staged content groups without COMPLETE.json (published validation requires it by default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.root and args.root_option:
        parser.error("provide the dataset root either positionally or with --root, not both")
    root_value = args.root_option or args.root
    if not root_value:
        parser.error("a dataset root is required")
    if args.expected_contents < 0:
        parser.error("--expected-contents must be non-negative")
    root = Path(root_value).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else root / "validation_report.json"
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else root / "valid_variants.jsonl"
    split_manifest_dir = (
        Path(args.split_manifest_dir).expanduser().resolve()
        if args.split_manifest_dir
        else root / "split_manifests"
    )
    report, manifest = validate_dataset(
        root,
        expected_task=args.task,
        expected_contents=args.expected_contents,
        expected_content_ids=args.content_ids,
        expected_content_seeds=args.content_seeds,
        expected_style_seeds=args.style_seeds,
        require_complete=not args.allow_missing_complete,
    )
    _atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    manifest_text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in manifest)
    _atomic_write_text(manifest_path, manifest_text)
    split_manifest_paths: dict[str, str] = {}
    for split in SPLIT_COUNTS:
        for kind in ("clean", "random"):
            rows = [
                row
                for row in manifest
                if row["split"] == split
                and (row["variant"] == "clean") == (kind == "clean")
            ]
            path = split_manifest_dir / f"{split}_{kind}.jsonl"
            _atomic_write_text(
                path,
                "".join(
                    json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                    for row in rows
                ),
            )
            split_manifest_paths[f"{split}_{kind}"] = str(path)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "valid": report["valid"],
                **summary,
                "report": str(report_path),
                "manifest": str(manifest_path),
                "split_manifests": split_manifest_paths,
            },
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
