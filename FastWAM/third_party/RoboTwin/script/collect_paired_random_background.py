#!/usr/bin/env python3
"""Transactional strict-pair collection for supported RoboTwin tasks.

This runner deliberately has a narrower contract than RoboTwin's generic
collector:

* a successful clean planning pass creates one source trajectory;
* clean and styles 0, 1, and 2 all replay that trajectory with
  ``need_plan=False``;
* style sampling uses a private ``numpy.random.Generator`` and explicit wall
  and table IDs, so it cannot advance the task's global RNG;
* a content group is published only after every replay succeeds and the
  importable strict validator accepts the staged group.

Run this file from the RoboTwin root.  RoboTwin's asset loaders intentionally
use paths relative to that working directory.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml

from paired_task_adapters import (
    SPLIT_COUNTS,
    SUPPORTED_TASKS,
    TOTAL_CONTENTS,
    canonical_json_sha256,
    capture_success_spec,
    capture_task_identity,
    capture_task_state,
    split_for_content,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = ROBOTWIN_ROOT / "task_config" / "paired_random_background.yml"
TEXTURE_ROOT = ROBOTWIN_ROOT / "assets" / "background_texture" / "seen"

DEFAULT_TASK = "grab_roller"
TASK_NAME = DEFAULT_TASK
TASK_CONFIG_NAME = "paired_random_background"
STYLE_SEEDS = (0, 1, 2)
MAX_CONTENTS = TOTAL_CONTENTS
SCHEMA_VERSION = 2

_CONTENT_RE = re.compile(r"^content_(\d{6})$")
_STYLE_DIRS = tuple(f"style_{index:02d}_seed_{seed}" for index, seed in enumerate(STYLE_SEEDS))
_NVIDIA_PCI_RE = re.compile(
    r"^(?P<domain>[0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):"
    r"(?P<bus>[0-9A-Fa-f]{2}):(?P<device>[0-9A-Fa-f]{2})\."
    r"(?P<function>[0-7])$"
)


class PairCollectionError(RuntimeError):
    """A candidate content group did not meet the strict publication gate."""


class PairValidationError(PairCollectionError):
    """The importable validator rejected a fully generated staging group."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON in the target directory and atomically replace ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numpy_global_rng_sha256() -> str:
    """Hash legacy NumPy global RNG state without consuming it."""

    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    digest = hashlib.sha256()
    digest.update(algorithm.encode("ascii"))
    digest.update(np.asarray(keys, dtype="<u4").tobytes(order="C"))
    digest.update(np.asarray([position, has_gauss], dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray([cached_gaussian], dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _hdf5_head_rgb_sha256(path: Path) -> str:
    """Hash contiguous head-camera RGB bytes exactly as the validator does."""

    import h5py

    dataset_path = "observation/head_camera/rgb"
    with h5py.File(path, "r") as handle:
        if dataset_path not in handle:
            raise PairCollectionError(f"missing HDF5 dataset {dataset_path!r} in {path}")
        dataset = handle[dataset_path]
        values = np.asarray(dataset[()])
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _require_robotwin_cwd() -> None:
    actual = Path.cwd().resolve()
    expected = ROBOTWIN_ROOT.resolve()
    if actual != expected:
        raise SystemExit(
            "collect_paired_random_background.py must run from the RoboTwin root; "
            f"expected cwd {expected}, got {actual}.\n"
            f"Use: cd {expected} && python script/{Path(__file__).name}"
        )


def _canonical_nvidia_pci_address(value: str) -> str:
    match = _NVIDIA_PCI_RE.fullmatch(value.strip())
    if match is None:
        raise PairCollectionError(f"nvidia-smi returned an invalid PCI address: {value!r}")
    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        raise PairCollectionError(f"nvidia-smi PCI domain is out of range: {value!r}")
    return (
        f"{domain:04x}:{match.group('bus').lower()}:"
        f"{match.group('device').lower()}.{match.group('function')}"
    )


def _require_single_numeric_visible_gpu() -> tuple[int, str]:
    """Pin CUDA and Vulkan rendering to one auditable physical GPU."""

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or re.fullmatch(r"[0-9]+", value) is None:
        raise PairCollectionError(
            "paired collection requires CUDA_VISIBLE_DEVICES to contain exactly "
            "one numeric physical GPU index"
        )
    physical_gpu_index = int(value)
    declared = os.environ.get("ROBOTWIN_PHYSICAL_GPU_INDEX")
    if declared is not None and declared != value:
        raise PairCollectionError(
            "ROBOTWIN_PHYSICAL_GPU_INDEX must exactly match CUDA_VISIBLE_DEVICES"
        )
    os.environ["ROBOTWIN_PHYSICAL_GPU_INDEX"] = value
    device_order = os.environ.get("CUDA_DEVICE_ORDER")
    if device_order not in (None, "PCI_BUS_ID"):
        raise PairCollectionError(
            "CUDA_DEVICE_ORDER must be PCI_BUS_ID for physical GPU index auditing"
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    command = [
        "nvidia-smi",
        "--id",
        value,
        "--query-gpu=pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PairCollectionError(
            f"failed to resolve physical GPU {value} with nvidia-smi"
        ) from exc
    addresses = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(addresses) != 1:
        raise PairCollectionError(
            f"nvidia-smi returned {len(addresses)} PCI addresses for GPU {value}: "
            f"{addresses!r}"
        )
    pci_address = _canonical_nvidia_pci_address(addresses[0])
    expected_pci = os.environ.get("ROBOTWIN_EXPECTED_GPU_PCI")
    if expected_pci is not None and expected_pci.lower() != pci_address:
        raise PairCollectionError(
            "ROBOTWIN_EXPECTED_GPU_PCI disagrees with nvidia-smi: "
            f"declared={expected_pci!r}, actual={pci_address!r}"
        )
    os.environ["ROBOTWIN_EXPECTED_GPU_PCI"] = pci_address
    os.environ["ROBOTWIN_RENDER_DEVICE_ALIAS"] = f"pci:{pci_address}"
    return physical_gpu_index, pci_address


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise PairCollectionError(f"expected a mapping in {path}")
    return value


def _strict_domain_randomization(random_background: bool) -> dict[str, Any]:
    return {
        "random_background": bool(random_background),
        "cluttered_table": False,
        "clean_background_rate": 0,
        "random_head_camera_dis": 0,
        "random_table_height": 0,
        "random_light": False,
        "crazy_random_light_rate": 0,
        "random_embodiment": False,
    }


def _validate_static_config(config: Mapping[str, Any]) -> None:
    if list(config.get("embodiment", [])) != ["aloha-agilex"]:
        raise PairCollectionError("paired config must pin embodiment to [aloha-agilex]")
    actual = dict(config.get("domain_randomization", {}))
    expected = _strict_domain_randomization(False)
    if actual != expected:
        raise PairCollectionError(
            "paired config contains an unapproved domain-randomization setting: "
            f"expected {expected!r}, got {actual!r}"
        )
    camera = config.get("camera")
    if not isinstance(camera, Mapping):
        raise PairCollectionError("paired config is missing camera settings")


def _get_embodiment_config(robot_directory: Path) -> dict[str, Any]:
    return _read_yaml(robot_directory / "config.yml")


def _load_base_args() -> dict[str, Any]:
    args = _read_yaml(CONFIG_PATH)
    _validate_static_config(args)

    embodiment_types = _read_yaml(ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml")
    embodiment = list(args["embodiment"])

    def robot_directory(kind: str) -> Path:
        try:
            configured = embodiment_types[kind]["file_path"]
        except (KeyError, TypeError) as exc:
            raise PairCollectionError(f"unknown embodiment {kind!r}") from exc
        if configured is None:
            raise PairCollectionError(f"embodiment {kind!r} has no file_path")
        path = Path(configured)
        return path if path.is_absolute() else ROBOTWIN_ROOT / path

    if len(embodiment) == 1:
        left_directory = right_directory = robot_directory(embodiment[0])
        args["dual_arm_embodied"] = True
        embodiment_name = embodiment[0]
    elif len(embodiment) == 3:
        left_directory = robot_directory(embodiment[0])
        right_directory = robot_directory(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise PairCollectionError("embodiment must contain either one or three entries")

    args.update(
        {
            "task_name": TASK_NAME,
            "task_config": TASK_CONFIG_NAME,
            "embodiment_name": embodiment_name,
            "left_robot_file": str(left_directory),
            "right_robot_file": str(right_directory),
            "left_embodiment_config": _get_embodiment_config(left_directory),
            "right_embodiment_config": _get_embodiment_config(right_directory),
        }
    )
    return args


@dataclass(frozen=True)
class TextureChoice:
    style_index: int
    style_seed: int
    wall_id: int
    table_id: int
    wall_file: Path
    table_file: Path
    private_rng_state_sha256: str

    @property
    def ids(self) -> tuple[int, int]:
        return self.wall_id, self.table_id


def _numeric_seen_texture_ids() -> np.ndarray:
    ids: list[int] = []
    for path in TEXTURE_ROOT.glob("*.png"):
        try:
            ids.append(int(path.stem))
        except ValueError as exc:
            raise PairCollectionError(f"non-numeric background texture filename: {path.name}") from exc
    ids.sort()
    if len(ids) < 2:
        raise PairCollectionError(f"need at least two seen background textures in {TEXTURE_ROOT}")
    if len(set(ids)) != len(ids):
        raise PairCollectionError(f"duplicate numeric texture IDs in {TEXTURE_ROOT}")
    return np.asarray(ids, dtype=np.int64)


def _derive_texture_choices() -> tuple[TextureChoice, ...]:
    """Derive exactly three styles without touching ``np.random`` global state."""

    texture_ids = _numeric_seen_texture_ids()
    choices: list[TextureChoice] = []
    for style_index, style_seed in enumerate(STYLE_SEEDS):
        private_rng = np.random.default_rng(style_seed)
        wall_id, table_id = (
            int(value) for value in private_rng.choice(texture_ids, size=2, replace=False)
        )
        if wall_id == table_id:
            raise AssertionError("replace=False returned duplicate texture IDs")
        wall_file = TEXTURE_ROOT / f"{wall_id}.png"
        table_file = TEXTURE_ROOT / f"{table_id}.png"
        if not wall_file.is_file() or not table_file.is_file():
            raise PairCollectionError("private style RNG selected a missing texture file")
        choices.append(
            TextureChoice(
                style_index=style_index,
                style_seed=style_seed,
                wall_id=wall_id,
                table_id=table_id,
                wall_file=wall_file,
                table_file=table_file,
                private_rng_state_sha256=_hash_json(private_rng.bit_generator.state),
            )
        )

    pairs = {(choice.wall_id, choice.table_id) for choice in choices}
    if len(pairs) != len(STYLE_SEEDS):
        raise PairCollectionError("style seeds 0, 1, and 2 did not produce three distinct texture pairs")
    return tuple(choices)


def _to_float_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise PairCollectionError(f"cannot record numeric field {name}") from exc
    if not np.all(np.isfinite(result)):
        raise PairCollectionError(f"non-finite value in trace field {name}")
    return result


def _joint_drive_vector(joints: Iterable[Any], *, velocity: bool) -> np.ndarray:
    method_name = "get_drive_velocity_target" if velocity else "get_drive_target"
    attribute_names = (
        ("drive_velocity_target", "velocity_target") if velocity else ("drive_target", "target")
    )
    values: list[np.ndarray] = []
    for joint in joints:
        getter = getattr(joint, method_name, None)
        value: Any = None
        if callable(getter):
            value = getter()
        else:
            for attribute_name in attribute_names:
                if hasattr(joint, attribute_name):
                    value = getattr(joint, attribute_name)
                    break
        if value is None:
            raise PairCollectionError(
                f"joint {getattr(joint, 'get_name', lambda: '<unknown>')()} exposes no {method_name}"
            )
        values.append(_to_float_vector(value, name=method_name))
    if not values:
        raise PairCollectionError("robot articulation has no active joints")
    return np.concatenate(values)


def _roller_velocity(env: Any, *, angular: bool) -> np.ndarray:
    wrapper = env.roller
    entity = getattr(wrapper, "actor", wrapper)
    getter_names = ("get_angular_velocity",) if angular else ("get_linear_velocity", "get_velocity")
    attribute_names = ("angular_velocity",) if angular else ("linear_velocity", "velocity")

    sources = [wrapper, entity]
    get_components = getattr(entity, "get_components", None)
    if callable(get_components):
        sources.extend(list(get_components()))

    for source in sources:
        for getter_name in getter_names:
            getter = getattr(source, getter_name, None)
            if callable(getter):
                return _to_float_vector(getter(), name=f"roller.{getter_name}")
        for attribute_name in attribute_names:
            if hasattr(source, attribute_name):
                return _to_float_vector(
                    getattr(source, attribute_name), name=f"roller.{attribute_name}"
                )
    kind = "angular" if angular else "linear"
    raise PairCollectionError(f"roller rigid body exposes no {kind} velocity")


def _semantic_action(env: Any, payload: Mapping[str, Any], fallback: str) -> str:
    value = payload.get("semantic_action")
    if value is None:
        value = getattr(env, "_pair_trace_semantic_action", None)
    if value is None:
        value = fallback
    return str(value)


class PairTraceRecorder:
    """Generic Base_Task hook that records every traced physics step."""

    ACTION_KEYS = (
        "semantic_action",
        "left_drive_target",
        "right_drive_target",
        "left_drive_velocity",
        "right_drive_velocity",
    )
    STATE_KEYS = (
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
    )

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.task_state_layout: list[str] | None = None
        self._action_rows: dict[str, list[Any]] = {key: [] for key in self.ACTION_KEYS}
        self._state_rows: dict[str, list[Any]] = {key: [] for key in self.STATE_KEYS}
        self._pending_semantic_action = "physics_step"
        self.frame_trace_indices: list[int] = []
        self.frame_numbers: list[int] = []
        self.frame_pkl_paths: list[str] = []
        self._started = False

    def _snapshot(self, env: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        robot = env.robot
        left_entity = robot.left_entity
        right_entity = robot.right_entity
        left_joints = left_entity.get_active_joints()
        right_joints = right_entity.get_active_joints()

        action = {
            "left_drive_target": _joint_drive_vector(left_joints, velocity=False),
            "right_drive_target": _joint_drive_vector(right_joints, velocity=False),
            "left_drive_velocity": _joint_drive_vector(left_joints, velocity=True),
            "right_drive_velocity": _joint_drive_vector(right_joints, velocity=True),
        }

        task_state, task_state_layout = capture_task_state(env, self.task_name)
        if self.task_state_layout is None:
            self.task_state_layout = list(task_state_layout)
        elif self.task_state_layout != list(task_state_layout):
            raise PairCollectionError("task_state layout changed during replay")
        state = {
            "left_qpos": _to_float_vector(left_entity.get_qpos(), name="left_qpos"),
            "right_qpos": _to_float_vector(right_entity.get_qpos(), name="right_qpos"),
            "left_qvel": _to_float_vector(left_entity.get_qvel(), name="left_qvel"),
            "right_qvel": _to_float_vector(right_entity.get_qvel(), name="right_qvel"),
            "left_eef": _to_float_vector(robot.get_left_ee_pose(), name="left_eef"),
            "right_eef": _to_float_vector(robot.get_right_ee_pose(), name="right_eef"),
            "task_state": task_state,
            "left_gripper_open": bool(env.is_left_gripper_open()),
            "right_gripper_open": bool(env.is_right_gripper_open()),
            "left_gripper_closed": bool(env.is_left_gripper_close()),
            "right_gripper_closed": bool(env.is_right_gripper_close()),
        }
        return action, state

    def _append_action(self, action: Mapping[str, Any], semantic_action: str) -> None:
        action = dict(action)
        action["semantic_action"] = semantic_action
        for key in self.ACTION_KEYS:
            self._action_rows[key].append(action[key])

    def _append_state(self, state: Mapping[str, Any], semantic_action: str) -> None:
        state = dict(state)
        state["semantic_action"] = semantic_action
        for key in self.STATE_KEYS:
            self._state_rows[key].append(state[key])

    def start(self, env: Any) -> None:
        if self._started:
            raise PairCollectionError("PairTraceRecorder.start called more than once")
        _, initial_state = self._snapshot(env)
        self._append_state(initial_state, "initial")
        self._started = True

    def before_step(self, env: Any, **payload: Any) -> None:
        if not self._started:
            raise PairCollectionError("trace hook fired before recorder.start")
        self._pending_semantic_action = _semantic_action(env, payload, "physics_step")
        action, _ = self._snapshot(env)
        self._append_action(action, self._pending_semantic_action)

    def after_step(self, env: Any, **payload: Any) -> None:
        if not self._started:
            raise PairCollectionError("trace hook fired before recorder.start")
        semantic = _semantic_action(env, payload, self._pending_semantic_action)
        _, state = self._snapshot(env)
        self._append_state(state, semantic)
        self._pending_semantic_action = "physics_step"

    def on_frame_saved(
        self,
        env: Any,
        *,
        frame_index: Optional[int] = None,
        pkl_path: Optional[str] = None,
        **_: Any,
    ) -> None:
        if not self._started:
            raise PairCollectionError("frame hook fired before recorder.start")
        expected_frame = len(self.frame_trace_indices)
        actual_frame = expected_frame if frame_index is None else int(frame_index)
        if actual_frame != expected_frame:
            raise PairCollectionError(
                f"non-contiguous saved frame index: expected {expected_frame}, got {actual_frame}"
            )
        self.frame_numbers.append(actual_frame)
        self.frame_trace_indices.append(len(self._state_rows["left_qpos"]) - 1)
        self.frame_pkl_paths.append("" if pkl_path is None else str(pkl_path))

    # Alias for hook implementations that use the shorter event name.
    def on_frame(self, env: Any, **payload: Any) -> None:
        self.on_frame_saved(env, **payload)

    def __call__(self, event: str, env: Any, **payload: Any) -> None:
        callback = getattr(self, event, None)
        if callback is None and event == "on_frame":
            callback = self.on_frame_saved
        if callback is None:
            raise PairCollectionError(f"unknown pair trace hook event {event!r}")
        callback(env, **payload)

    @staticmethod
    def _fixed_array(values: Sequence[Any], *, key: str) -> np.ndarray:
        if key == "semantic_action":
            return np.asarray(values, dtype="<U64")
        if key in (
            "left_gripper_open",
            "right_gripper_open",
            "left_gripper_closed",
            "right_gripper_closed",
        ):
            return np.asarray(values, dtype=np.bool_)
        try:
            return np.stack(values, axis=0)
        except ValueError as exc:
            shapes = [np.shape(value) for value in values]
            raise PairCollectionError(f"non-fixed trace shape for {key}: {shapes}") from exc

    def write(self, variant_dir: Path) -> dict[str, Any]:
        if not self._started or not self._state_rows["left_qpos"]:
            raise PairCollectionError("cannot write an empty pair trace")
        if not self.frame_trace_indices:
            raise PairCollectionError("no frame callbacks were recorded")
        action_count = len(self._action_rows["semantic_action"])
        state_count = len(self._state_rows["semantic_action"])
        if state_count != action_count + 1:
            raise PairCollectionError(
                "trace must contain initial state plus one post-step state per action: "
                f"actions={action_count}, states={state_count}"
            )

        action_arrays = {
            key: self._fixed_array(self._action_rows[key], key=key) for key in self.ACTION_KEYS
        }
        state_arrays = {
            key: self._fixed_array(self._state_rows[key], key=key) for key in self.STATE_KEYS
        }
        state_arrays["frame_trace_index"] = np.asarray(self.frame_trace_indices, dtype=np.int64)

        action_path = variant_dir / "action_trace.npz"
        state_path = variant_dir / "state_trace.npz"
        initial_path = variant_dir / "initial_state.npz"
        np.savez(action_path, **action_arrays)
        np.savez(state_path, **state_arrays)

        initial_arrays = {
            key: array[:1] for key, array in state_arrays.items() if key != "frame_trace_index"
        }
        initial_arrays["frame_trace_index"] = np.asarray([0], dtype=np.int64)
        np.savez(initial_path, **initial_arrays)

        # Prove that none of the fixed-array archives needs pickle support.
        for path in (action_path, state_path, initial_path):
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    _ = archive[key]

        return {
            "action_trace_path": action_path,
            "state_trace_path": state_path,
            "initial_state_path": initial_path,
            "action_rows": action_count,
            "trace_rows": state_count,
            "frame_count": len(self.frame_trace_indices),
            "frame_trace_index": list(self.frame_trace_indices),
            "task_state_layout": list(self.task_state_layout or ()),
        }


def _variant_args(
    base_args: Mapping[str, Any],
    *,
    save_path: Path,
    random_background: bool,
    content_seed: int,
    style_seed: Optional[int],
    background_texture_ids: Optional[tuple[int, int]],
    need_plan: bool,
    save_data: bool,
) -> dict[str, Any]:
    args = copy.deepcopy(dict(base_args))
    args.update(
        {
            "save_path": str(save_path),
            "episode_num": 1,
            "render_freq": 0,
            "need_plan": bool(need_plan),
            "save_data": bool(save_data),
            "content_seed": int(content_seed),
            "style_seed": style_seed,
            "background_texture_ids": background_texture_ids,
            # The collector resolves the requested physical nvidia-smi index
            # to its PCI address before importing SAPIEN.  PCI binding avoids
            # Vulkan selecting a different physical GPU from CUDA.
            "render_device_alias": os.environ["ROBOTWIN_RENDER_DEVICE_ALIAS"],
        }
    )
    args["domain_randomization"] = _strict_domain_randomization(random_background)
    return args


def _load_trajectory(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise PairCollectionError(f"trajectory is not a mapping: {path}")
    total_entries = 0
    for key in ("left_joint_path", "right_joint_path"):
        entries = value.get(key)
        if not isinstance(entries, list):
            raise PairCollectionError(f"trajectory {key} is not a list: {path}")
        total_entries += len(entries)
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or entry.get("status") != "Success":
                raise PairCollectionError(
                    f"trajectory {key}[{entry_index}] is not a successful planned path"
                )
    if total_entries == 0:
        raise PairCollectionError(f"trajectory has no planned path entries: {path}")
    return value


def _safe_close_env(task: Any) -> None:
    try:
        task.clear_pair_trace_hook()
    except Exception:
        pass
    try:
        # A full run reconstructs one planning scene plus four replay scenes
        # per accepted content.  RoboTwin's stock collector periodically clears
        # SAPIEN's renderer cache for this reason; the paired run does it after
        # every scene so the finite 50-content job cannot accumulate textures.
        task.close_env(clear_cache=True)
    except TypeError:
        # Keep the helper usable with older/custom task shims whose close_env
        # predates the clear_cache keyword.
        try:
            task.close_env()
        except Exception:
            pass
    except Exception:
        pass


@dataclass(frozen=True)
class PlannedSource:
    trajectory_path: Path
    trajectory_sha256: str
    trajectory: dict[str, Any]
    task_identity: dict[str, Any]
    task_identity_sha256: str
    task_success_spec: dict[str, Any]
    task_state_layout: tuple[str, ...]
    planner_calls: int


def _plan_clean_source(
    task: Any,
    base_args: Mapping[str, Any],
    *,
    clean_dir: Path,
    content_seed: int,
) -> PlannedSource:
    """Plan once; this pass is not a published variant and records no frames."""

    clean_dir.mkdir(parents=True, exist_ok=False)
    args = _variant_args(
        base_args,
        save_path=clean_dir,
        random_background=False,
        content_seed=content_seed,
        style_seed=None,
        background_texture_ids=None,
        need_plan=True,
        save_data=False,
    )
    setup_complete = False
    try:
        task.setup_demo(now_ep_num=0, seed=content_seed, **args)
        setup_complete = True
        if task.wall_texture is not None or task.table_texture is not None:
            raise PairCollectionError("clean planning pass unexpectedly selected a texture")
        task_identity = capture_task_identity(task, TASK_NAME)
        task_success_spec = capture_success_spec(task, TASK_NAME)
        _, task_state_layout = capture_task_state(task, TASK_NAME)
        task.play_once()
        success = bool(task.plan_success and task.check_success())
        if not success:
            raise PairCollectionError("clean planning pass did not finish successfully")
        planner_calls = int(getattr(task, "planner_call_count", -1))
        if planner_calls <= 0:
            raise PairCollectionError(
                f"clean planning pass must call a planner, observed planner_call_count={planner_calls}"
            )
        task.save_traj_data(0)
        trajectory_path = clean_dir / "_traj_data" / "episode0.pkl"
        trajectory = _load_trajectory(trajectory_path)
        return PlannedSource(
            trajectory_path=trajectory_path,
            trajectory_sha256=_sha256_file(trajectory_path),
            trajectory=trajectory,
            task_identity=task_identity,
            task_identity_sha256=canonical_json_sha256(task_identity),
            task_success_spec=task_success_spec,
            task_state_layout=tuple(task_state_layout),
            planner_calls=planner_calls,
        )
    finally:
        if setup_complete:
            _safe_close_env(task)


def _relative_to_content(path: Path, content_dir: Path) -> str:
    return path.resolve().relative_to(content_dir.resolve()).as_posix()


def _cache_frame_count(variant_dir: Path) -> int:
    cache_dir = variant_dir / ".cache" / "episode0"
    if not cache_dir.is_dir():
        raise PairCollectionError(f"missing replay frame cache: {cache_dir}")
    numbered: list[int] = []
    for path in cache_dir.glob("*.pkl"):
        if path.stem.isdigit():
            numbered.append(int(path.stem))
    numbered.sort()
    if numbered != list(range(len(numbered))):
        raise PairCollectionError(f"non-contiguous replay frame cache in {cache_dir}")
    return len(numbered)


def _remove_variant_cache(task: Any, variant_dir: Path) -> None:
    cache_dir = variant_dir / ".cache" / "episode0"
    try:
        task.remove_data_cache()
    except Exception:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    cache_root = variant_dir / ".cache"
    if cache_root.is_dir() and not any(cache_root.iterdir()):
        cache_root.rmdir()


@dataclass(frozen=True)
class VariantResult:
    directory_name: str
    metadata: dict[str, Any]
    hashes: dict[str, str]
    rng_state_sha256_after_setup: str
    task_identity_sha256: str
    task_success_spec_sha256: str
    task_state_layout: tuple[str, ...]


def _replay_variant(
    task: Any,
    base_args: Mapping[str, Any],
    *,
    content_dir: Path,
    variant_dir: Path,
    content_id: int,
    content_seed: int,
    planned: PlannedSource,
    texture: Optional[TextureChoice],
    clean_source_hashes: Optional[Mapping[str, str]],
    clean_hdf5_sha256: Optional[str],
) -> VariantResult:
    is_clean = texture is None
    variant_name = "clean" if is_clean else variant_dir.name
    style_seed = None if is_clean else texture.style_seed
    expected_texture_ids = None if is_clean else texture.ids

    if not is_clean:
        variant_dir.mkdir(parents=True, exist_ok=False)

    args = _variant_args(
        base_args,
        save_path=variant_dir,
        random_background=not is_clean,
        content_seed=content_seed,
        style_seed=style_seed,
        background_texture_ids=expected_texture_ids,
        need_plan=False,
        save_data=True,
    )

    recorder = PairTraceRecorder(TASK_NAME)
    setup_complete = False
    try:
        task.setup_demo(now_ep_num=0, seed=content_seed, **args)
        setup_complete = True
        rng_after_setup = _numpy_global_rng_sha256()

        if bool(task.need_plan):
            raise PairCollectionError(f"{variant_name}: setup changed need_plan to true")
        task_identity = capture_task_identity(task, TASK_NAME)
        task_identity_sha256 = canonical_json_sha256(task_identity)
        task_success_spec = capture_success_spec(task, TASK_NAME)
        task_success_spec_sha256 = canonical_json_sha256(task_success_spec)
        if task_identity_sha256 != planned.task_identity_sha256:
            raise PairCollectionError(
                f"{variant_name}: task identity changed from "
                f"{planned.task_identity!r} to {task_identity!r}"
            )
        if task_success_spec != planned.task_success_spec:
            raise PairCollectionError(
                f"{variant_name}: task success specification changed across replay"
            )

        actual_texture_ids = getattr(task, "background_texture_ids", None)
        if is_clean:
            if task.wall_texture is not None or task.table_texture is not None:
                raise PairCollectionError("clean replay unexpectedly selected background textures")
            if actual_texture_ids is not None:
                raise PairCollectionError("clean replay retained explicit background texture IDs")
        else:
            if tuple(actual_texture_ids or ()) != expected_texture_ids:
                raise PairCollectionError(
                    f"{variant_name}: explicit texture IDs changed: "
                    f"expected {expected_texture_ids}, got {actual_texture_ids}"
                )
            expected_names = (f"seen/{texture.wall_id}", f"seen/{texture.table_id}")
            if (task.wall_texture, task.table_texture) != expected_names:
                raise PairCollectionError(
                    f"{variant_name}: texture names changed: expected {expected_names}, "
                    f"got {(task.wall_texture, task.table_texture)}"
                )

        task.set_path_lst(
            {
                "need_plan": False,
                "left_joint_path": copy.deepcopy(planned.trajectory["left_joint_path"]),
                "right_joint_path": copy.deepcopy(planned.trajectory["right_joint_path"]),
            }
        )
        recorder.start(task)
        if tuple(recorder.task_state_layout or ()) != planned.task_state_layout:
            raise PairCollectionError(
                f"{variant_name}: task-state layout changed across replay"
            )
        task.set_pair_trace_hook(recorder)
        task.play_once()

        success = bool(task.plan_success and task.check_success())
        available = {
            "left": len(planned.trajectory["left_joint_path"]),
            "right": len(planned.trajectory["right_joint_path"]),
        }
        consumed = {"left": int(task.left_cnt), "right": int(task.right_cnt)}
        fully_consumed = consumed == available
        planner_calls = int(getattr(task, "planner_call_count", -1))

        # This gate is deliberately before HDF5/video merge or publication.
        if not success:
            raise PairCollectionError(f"{variant_name}: replay did not finish successfully")
        if not fully_consumed:
            raise PairCollectionError(
                f"{variant_name}: source paths were not consumed exactly: "
                f"available={available}, consumed={consumed}"
            )
        if planner_calls != 0:
            raise PairCollectionError(
                f"{variant_name}: replay called a planner {planner_calls} time(s)"
            )
        if bool(task.need_plan):
            raise PairCollectionError(f"{variant_name}: need_plan became true during replay")

        trace_info = recorder.write(variant_dir)
        cache_frames = _cache_frame_count(variant_dir)
        if trace_info["frame_count"] != cache_frames or int(task.FRAME_IDX) != cache_frames:
            raise PairCollectionError(
                f"{variant_name}: inconsistent frame accounting: "
                f"hook={trace_info['frame_count']}, cache={cache_frames}, env={task.FRAME_IDX}"
            )

        replay_facts = {
            "rng_after_setup": rng_after_setup,
            "success": success,
            "available": available,
            "consumed": consumed,
            "fully_consumed": fully_consumed,
            "planner_calls": planner_calls,
            "trace_info": trace_info,
            "task_identity": task_identity,
            "task_identity_sha256": task_identity_sha256,
            "task_success_spec": task_success_spec,
            "task_success_spec_sha256": task_success_spec_sha256,
            "wall_texture": task.wall_texture,
            "table_texture": task.table_texture,
            "render_device": dict(task.render_device_info),
        }
    finally:
        try:
            task.clear_pair_trace_hook()
        finally:
            if setup_complete:
                _safe_close_env(task)

    # Success has already been checked.  Only now create the normal RoboTwin
    # HDF5/video artifacts and discard the per-frame pickle cache.
    task.merge_pkl_to_hdf5_video()
    _remove_variant_cache(task, variant_dir)

    hdf5_path = variant_dir / "data" / "episode0.hdf5"
    video_path = variant_dir / "video" / "episode0.mp4"
    if not hdf5_path.is_file() or not video_path.is_file():
        raise PairCollectionError(f"{variant_name}: merge did not produce HDF5 and video")

    artifact_paths = {
        "initial_state": variant_dir / "initial_state.npz",
        "action_trace": variant_dir / "action_trace.npz",
        "state_trace": variant_dir / "state_trace.npz",
        "hdf5": hdf5_path,
        "video": video_path,
    }
    hashes = {name: _sha256_file(path) for name, path in artifact_paths.items()}
    hashes["head_rgb"] = _hdf5_head_rgb_sha256(hdf5_path)

    # The clean variant is its own published source episode.  Style variants
    # point to these clean hashes, never to one another.
    if is_clean:
        source_hashes = {
            "source_trajectory": planned.trajectory_sha256,
            **hashes,
        }
        source_clean_hdf5_sha256 = hashes["hdf5"]
    else:
        if clean_source_hashes is None or clean_hdf5_sha256 is None:
            raise AssertionError("style replay requires completed clean source hashes")
        source_hashes = dict(clean_source_hashes)
        source_clean_hdf5_sha256 = clean_hdf5_sha256

    wall_texture = replay_facts["wall_texture"]
    table_texture = replay_facts["table_texture"]
    wall_texture_file = None if is_clean else _relative_to_content(texture.wall_file, ROBOTWIN_ROOT)
    table_texture_file = None if is_clean else _relative_to_content(texture.table_file, ROBOTWIN_ROOT)
    wall_texture_sha256 = None if is_clean else _sha256_file(texture.wall_file)
    table_texture_sha256 = None if is_clean else _sha256_file(texture.table_file)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_NAME,
        "content_id": int(content_id),
        "content_seed": int(content_seed),
        "split": split_for_content(content_id),
        "variant": variant_name,
        "style_index": None if is_clean else int(texture.style_index),
        "style_seed": style_seed,
        "intervention": "none" if is_clean else "random_background",
        "source_clean_episode": 0,
        "source_clean_hdf5_path": "clean/data/episode0.hdf5",
        "source_clean_hdf5_sha256": source_clean_hdf5_sha256,
        "source_trajectory_path": "clean/_traj_data/episode0.pkl",
        "source_trajectory_sha256": planned.trajectory_sha256,
        "source_hashes": source_hashes,
        "task_identity": replay_facts["task_identity"],
        "task_identity_sha256": replay_facts["task_identity_sha256"],
        "task_success_spec": replay_facts["task_success_spec"],
        "task_success_spec_sha256": replay_facts["task_success_spec_sha256"],
        "task_state_layout": replay_facts["trace_info"]["task_state_layout"],
        "render_device": replay_facts["render_device"],
        "wall_texture": wall_texture,
        "table_texture": table_texture,
        "wall_texture_id": None if is_clean else int(texture.wall_id),
        "table_texture_id": None if is_clean else int(texture.table_id),
        "wall_texture_file": wall_texture_file,
        "table_texture_file": table_texture_file,
        "wall_texture_sha256": wall_texture_sha256,
        "table_texture_sha256": table_texture_sha256,
        "textures": {
            "wall": {
                "id": None if is_clean else int(texture.wall_id),
                "name": wall_texture,
                "file": wall_texture_file,
                "sha256": wall_texture_sha256,
            },
            "table": {
                "id": None if is_clean else int(texture.table_id),
                "name": table_texture,
                "file": table_texture_file,
                "sha256": table_texture_sha256,
            },
        },
        "domain_randomization": _strict_domain_randomization(not is_clean),
        "success": replay_facts["success"],
        "frame_count": replay_facts["trace_info"]["frame_count"],
        "action_rows": replay_facts["trace_info"]["action_rows"],
        "trace_rows": replay_facts["trace_info"]["trace_rows"],
        "frame_trace_index": replay_facts["trace_info"]["frame_trace_index"],
        "path_entries": replay_facts["available"],
        "path_entries_consumed": replay_facts["consumed"],
        "path_fully_consumed": replay_facts["fully_consumed"],
        "path_consumption": {
            "left": {
                "available": replay_facts["available"]["left"],
                "consumed": replay_facts["consumed"]["left"],
            },
            "right": {
                "available": replay_facts["available"]["right"],
                "consumed": replay_facts["consumed"]["right"],
            },
            "fully_consumed": replay_facts["fully_consumed"],
        },
        "planner_calls": replay_facts["planner_calls"],
        "need_plan": False,
        "rng_state_sha256_after_setup": replay_facts["rng_after_setup"],
        "style_rng": None
        if is_clean
        else {
            "implementation": "numpy.random.default_rng",
            "seed": int(texture.style_seed),
            "state_sha256_after_sampling": texture.private_rng_state_sha256,
        },
        "hashes": hashes,
        "artifacts": {
            name: _relative_to_content(path, variant_dir) for name, path in artifact_paths.items()
        },
        "trace_schema": {
            "action_trace_keys": list(PairTraceRecorder.ACTION_KEYS),
            "state_trace_keys": [*PairTraceRecorder.STATE_KEYS, "frame_trace_index"],
            "task_state_layout": replay_facts["trace_info"]["task_state_layout"],
            "initial_state_row": 0,
            "npz_allow_pickle": False,
        },
    }
    _atomic_write_json(variant_dir / "metadata.json", metadata)
    return VariantResult(
        directory_name=variant_name,
        metadata=metadata,
        hashes=hashes,
        rng_state_sha256_after_setup=replay_facts["rng_after_setup"],
        task_identity_sha256=replay_facts["task_identity_sha256"],
        task_success_spec_sha256=replay_facts["task_success_spec_sha256"],
        task_state_layout=tuple(replay_facts["trace_info"]["task_state_layout"]),
    )


def _import_validator() -> Any:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    module = importlib.import_module("validate_paired_random_background")
    validator = getattr(module, "validate_content_dir", None)
    if not callable(validator):
        raise PairCollectionError(
            "validate_paired_random_background.validate_content_dir is not importable"
        )
    return validator


def _strict_validate_stage(
    stage_dir: Path,
    *,
    content_id: int,
    content_seed: int,
) -> dict[str, Any]:
    validator = _import_validator()
    report = validator(
        stage_dir,
        expected_task=TASK_NAME,
        expected_content_id=content_id,
        expected_content_seed=content_seed,
        expected_style_seeds=STYLE_SEEDS,
    )
    if not isinstance(report, Mapping) or not bool(report.get("valid", False)):
        errors = report.get("errors", []) if isinstance(report, Mapping) else [repr(report)]
        raise PairValidationError(f"strict stage validation failed: {errors}")
    return dict(report)


def _build_content_group(
    task: Any,
    base_args: Mapping[str, Any],
    *,
    stage_dir: Path,
    content_id: int,
    content_seed: int,
    textures: Sequence[TextureChoice],
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=False)
    clean_dir = stage_dir / "clean"
    planned = _plan_clean_source(
        task,
        base_args,
        clean_dir=clean_dir,
        content_seed=content_seed,
    )

    clean = _replay_variant(
        task,
        base_args,
        content_dir=stage_dir,
        variant_dir=clean_dir,
        content_id=content_id,
        content_seed=content_seed,
        planned=planned,
        texture=None,
        clean_source_hashes=None,
        clean_hdf5_sha256=None,
    )
    clean_source_hashes = dict(clean.metadata["source_hashes"])
    clean_hdf5_sha256 = clean.hashes["hdf5"]

    variants = [clean]
    for texture, directory_name in zip(textures, _STYLE_DIRS):
        variants.append(
            _replay_variant(
                task,
                base_args,
                content_dir=stage_dir,
                variant_dir=stage_dir / directory_name,
                content_id=content_id,
                content_seed=content_seed,
                planned=planned,
                texture=texture,
                clean_source_hashes=clean_source_hashes,
                clean_hdf5_sha256=clean_hdf5_sha256,
            )
        )

    if _sha256_file(planned.trajectory_path) != planned.trajectory_sha256:
        raise PairCollectionError("source trajectory changed during replay")
    rng_hashes = {variant.rng_state_sha256_after_setup for variant in variants}
    if len(rng_hashes) != 1:
        raise PairCollectionError(
            "clean/style setup consumed different global NumPy RNG streams: "
            f"{sorted(rng_hashes)}"
        )
    identity_hashes = {variant.task_identity_sha256 for variant in variants}
    if identity_hashes != {planned.task_identity_sha256}:
        raise PairCollectionError("task identity changed across variants")
    success_spec_hashes = {variant.task_success_spec_sha256 for variant in variants}
    if success_spec_hashes != {canonical_json_sha256(planned.task_success_spec)}:
        raise PairCollectionError("task success specification changed across variants")
    state_layouts = {variant.task_state_layout for variant in variants}
    if state_layouts != {planned.task_state_layout}:
        raise PairCollectionError("task-state layout changed across variants")

    validation = _strict_validate_stage(
        stage_dir,
        content_id=content_id,
        content_seed=content_seed,
    )
    return {
        "planned_source": planned,
        "variants": variants,
        "validation": validation,
        "rng_state_sha256_after_setup": next(iter(rng_hashes)),
    }


class OutputLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "OutputLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            raise PairCollectionError(
                f"another paired collection process holds {self.path}"
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()


def _slug(value: str, limit: int = 60) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return (normalized or "rejected")[:limit]


def _reject_path(path: Path, rejected_root: Path, reason: str) -> Path:
    rejected_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = rejected_root / (
        f"{path.name}__{_slug(reason)}__{stamp}__{uuid.uuid4().hex[:8]}"
    )
    os.replace(path, destination)
    marker = (
        destination / "REJECTED.json"
        if destination.is_dir()
        else destination.with_name(f"{destination.name}.REJECTED.json")
    )
    _atomic_write_json(
        marker,
        {"rejected_at": _utc_now(), "reason": reason, "original_name": path.name},
    )
    return destination


def _reject_stale_staging(staging_root: Path, rejected_root: Path) -> list[str]:
    rejected: list[str] = []
    if not staging_root.exists():
        return rejected
    for path in sorted(staging_root.iterdir()):
        destination = _reject_path(path, rejected_root, "stale-staging-from-interrupted-run")
        rejected.append(str(destination))
    return rejected


def _content_id_from_path(path: Path) -> int:
    match = _CONTENT_RE.fullmatch(path.name)
    if match is None:
        raise PairCollectionError(f"invalid published content directory name: {path.name}")
    return int(match.group(1))


def _read_content_identity(content_dir: Path) -> tuple[int, int]:
    content_id = _content_id_from_path(content_dir)
    metadata_path = content_dir / "clean" / "metadata.json"
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata_id = int(metadata["content_id"])
        content_seed = int(metadata["content_seed"])
    except Exception as exc:
        raise PairCollectionError(f"cannot read content identity from {metadata_path}") from exc
    if metadata_id != content_id:
        raise PairCollectionError(
            f"directory content ID {content_id} disagrees with metadata ID {metadata_id}"
        )
    return content_id, content_seed


def _scan_published_contents(
    contents_root: Path,
    rejected_root: Path,
    *,
    requested_contents: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Validate published groups; quarantine anything incomplete or invalid."""

    validator = _import_validator()
    valid: dict[int, dict[str, Any]] = {}
    seed_owner: dict[int, int] = {}
    rejected: list[dict[str, Any]] = []
    if not contents_root.exists():
        return valid, rejected

    for path in sorted(contents_root.iterdir()):
        try:
            content_id, content_seed = _read_content_identity(path)
            if content_id not in range(requested_contents):
                raise PairCollectionError(
                    f"published content ID {content_id} is outside the requested "
                    f"range [0, {requested_contents})"
                )
            if not (path / "COMPLETE.json").is_file():
                raise PairCollectionError("published group has no COMPLETE.json")
            report = validator(
                path,
                expected_task=TASK_NAME,
                expected_content_id=content_id,
                expected_content_seed=content_seed,
                expected_style_seeds=STYLE_SEEDS,
            )
            if not bool(report.get("valid", False)):
                raise PairValidationError(str(report.get("errors", [])))
            if content_seed in seed_owner:
                raise PairCollectionError(
                    f"content seed {content_seed} is duplicated by content IDs "
                    f"{seed_owner[content_seed]} and {content_id}"
                )
            seed_owner[content_seed] = content_id
            valid[content_id] = {
                "content_id": content_id,
                "content_seed": content_seed,
                "split": split_for_content(content_id),
                "path": path.relative_to(contents_root.parent).as_posix(),
                "complete_sha256": _sha256_file(path / "COMPLETE.json"),
                "valid": True,
            }
        except Exception as exc:
            destination = _reject_path(path, rejected_root, f"invalid-published-{type(exc).__name__}")
            rejected.append({"path": str(destination), "error": str(exc)})
    return valid, rejected


def _load_run_state(
    path: Path,
    start_seed: int,
    requested_contents: int,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception as exc:
        raise PairCollectionError(f"cannot read resumable run state {path}") from exc
    previous_task = state.get("task", TASK_NAME)
    if previous_task != TASK_NAME:
        raise PairCollectionError(
            f"output root belongs to task {previous_task!r}; refusing task {TASK_NAME!r}"
        )
    if int(state.get("start_seed", start_seed)) != start_seed:
        raise PairCollectionError(
            f"output root was initialized with start_seed={state.get('start_seed')}; "
            f"refusing incompatible --start-seed {start_seed}"
        )
    previous_request = int(state.get("requested_contents", requested_contents))
    if previous_request > requested_contents:
        raise PairCollectionError(
            f"output root previously requested {previous_request} contents; refusing to "
            f"shrink it to {requested_contents}"
        )
    return state


def _manifest_value(
    *,
    output_root: Path,
    requested_contents: int,
    start_seed: int,
    texture_choices: Sequence[TextureChoice],
    contents: Mapping[int, Mapping[str, Any]],
    attempts_total: int,
    rejected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    requested_split_counts = {
        split: sum(
            split_for_content(content_id) == split
            for content_id in range(requested_contents)
        )
        for split in SPLIT_COUNTS
    }
    completed_split_counts = {
        split: {
            "clean": sum(
                split_for_content(content_id) == split for content_id in contents
            ),
            "random": 3
            * sum(split_for_content(content_id) == split for content_id in contents),
        }
        for split in SPLIT_COUNTS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_NAME,
        "task_config": TASK_CONFIG_NAME,
        "output_root": str(output_root),
        "requested_contents": requested_contents,
        "maximum_contents": MAX_CONTENTS,
        "start_seed": start_seed,
        "style_seeds": list(STYLE_SEEDS),
        "requested_split": {
            split: {"clean": count, "random": 3 * count}
            for split, count in requested_split_counts.items()
        },
        "final_fixed_split": {
            split: {"clean": count, "random": 3 * count}
            for split, count in SPLIT_COUNTS.items()
        },
        "completed_split_counts": completed_split_counts,
        "styles": [
            {
                "style_index": choice.style_index,
                "style_seed": choice.style_seed,
                "wall_texture_id": choice.wall_id,
                "table_texture_id": choice.table_id,
                "private_rng_state_sha256_after_sampling": choice.private_rng_state_sha256,
            }
            for choice in texture_choices
        ],
        "completed_contents": [contents[key] for key in sorted(contents)],
        "completed_count": len(contents),
        "attempts_total": attempts_total,
        "rejected": list(rejected),
        "updated_at": _utc_now(),
    }


def _run_state_value(
    *,
    status: str,
    requested_contents: int,
    start_seed: int,
    next_seed: int,
    attempts_total: int,
    attempts_this_run: int,
    completed_ids: Sequence[int],
    error: Optional[str] = None,
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_NAME,
        "status": status,
        "pid": os.getpid(),
        "requested_contents": requested_contents,
        "start_seed": start_seed,
        "next_seed": next_seed,
        "attempts_total": attempts_total,
        "attempts_this_run": attempts_this_run,
        "completed_content_ids": list(sorted(completed_ids)),
        "updated_at": _utc_now(),
    }
    if error is not None:
        value["error"] = error
    return value


def _next_missing_content_id(completed: Mapping[int, Any], requested_contents: int) -> Optional[int]:
    for content_id in range(requested_contents):
        if content_id not in completed:
            return content_id
    return None


def collect(
    *,
    task_name: str = DEFAULT_TASK,
    num_contents: int,
    output_root: Path,
    start_seed: int,
    max_attempts: int,
) -> None:
    global TASK_NAME

    if task_name not in SUPPORTED_TASKS:
        raise PairCollectionError(
            f"unsupported task {task_name!r}; choose one of {', '.join(SUPPORTED_TASKS)}"
        )
    TASK_NAME = task_name
    if not 1 <= num_contents <= MAX_CONTENTS:
        raise PairCollectionError(f"num_contents must be in [1, {MAX_CONTENTS}]")
    if start_seed < 0:
        raise PairCollectionError("start_seed must be non-negative")
    if max_attempts < 1:
        raise PairCollectionError("max_attempts must be positive")

    output_root = output_root.resolve()
    contents_root = output_root / "contents"
    staging_root = output_root / ".staging"
    rejected_root = output_root / "rejected"
    manifest_path = output_root / "manifest.json"
    run_state_path = output_root / "run_state.json"
    texture_choices = _derive_texture_choices()
    base_args = _load_base_args()

    with OutputLock(output_root / ".paired_random_background.lock"):
        contents_root.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        rejected_root.mkdir(parents=True, exist_ok=True)

        rejected_records: list[dict[str, Any]] = [
            {"path": path, "reason": "stale staging"}
            for path in _reject_stale_staging(staging_root, rejected_root)
        ]
        prior_state = _load_run_state(run_state_path, start_seed, num_contents)
        completed, invalid_published = _scan_published_contents(
            contents_root,
            rejected_root,
            requested_contents=num_contents,
        )
        rejected_records.extend(invalid_published)

        prior_attempts_total = int(prior_state.get("attempts_total", 0))
        completed_seeds = [int(value["content_seed"]) for value in completed.values()]
        recovered_next_seed = max([start_seed - 1, *completed_seeds]) + 1
        next_seed = max(int(prior_state.get("next_seed", start_seed)), recovered_next_seed)
        attempts_total = prior_attempts_total
        attempts_this_run = 0

        _atomic_write_json(
            manifest_path,
            _manifest_value(
                output_root=output_root,
                requested_contents=num_contents,
                start_seed=start_seed,
                texture_choices=texture_choices,
                contents=completed,
                attempts_total=attempts_total,
                rejected=rejected_records,
            ),
        )
        _atomic_write_json(
            run_state_path,
            _run_state_value(
                status="running",
                requested_contents=num_contents,
                start_seed=start_seed,
                next_seed=next_seed,
                attempts_total=attempts_total,
                attempts_this_run=attempts_this_run,
                completed_ids=list(completed),
            ),
        )

        task: Any = None
        try:
            task_module = importlib.import_module(f"envs.{TASK_NAME}")
            task_class = getattr(task_module, TASK_NAME, None)
            if task_class is None:
                raise PairCollectionError(f"RoboTwin task {TASK_NAME!r} is not importable")
            task = task_class()

            while (content_id := _next_missing_content_id(completed, num_contents)) is not None:
                if attempts_this_run >= max_attempts:
                    raise PairCollectionError(
                        f"max_attempts={max_attempts} exhausted with "
                        f"{len(completed)}/{num_contents} valid content groups"
                    )

                content_seed = next_seed
                next_seed += 1
                attempts_total += 1
                attempts_this_run += 1
                stage_dir = staging_root / (
                    f"content_{content_id:06d}__seed_{content_seed}__attempt_{attempts_total:06d}"
                )
                print(
                    f"[paired] content={content_id:06d} seed={content_seed} "
                    f"attempt={attempts_this_run}/{max_attempts}",
                    flush=True,
                )

                try:
                    built = _build_content_group(
                        task,
                        base_args,
                        stage_dir=stage_dir,
                        content_id=content_id,
                        content_seed=content_seed,
                        textures=texture_choices,
                    )

                    # COMPLETE.json is intentionally the final staged write.
                    complete = {
                        "schema_version": SCHEMA_VERSION,
                        "task": TASK_NAME,
                        "content_id": content_id,
                        "content_seed": content_seed,
                        "split": split_for_content(content_id),
                        "style_seeds": list(STYLE_SEEDS),
                        "source_trajectory_sha256": built["planned_source"].trajectory_sha256,
                        "task_identity": built["planned_source"].task_identity,
                        "task_identity_sha256": (
                            built["planned_source"].task_identity_sha256
                        ),
                        "task_success_spec_sha256": canonical_json_sha256(
                            built["planned_source"].task_success_spec
                        ),
                        "task_state_layout": list(
                            built["planned_source"].task_state_layout
                        ),
                        "rng_state_sha256_after_setup": built["rng_state_sha256_after_setup"],
                        "validation": {
                            "valid": True,
                            "errors": built["validation"].get("errors", []),
                            "warnings": built["validation"].get("warnings", []),
                        },
                        "completed_at": _utc_now(),
                    }
                    _atomic_write_json(stage_dir / "COMPLETE.json", complete)

                    destination = contents_root / f"content_{content_id:06d}"
                    if destination.exists():
                        raise PairCollectionError(f"refusing to overwrite {destination}")
                    os.replace(stage_dir, destination)
                    completed[content_id] = {
                        "content_id": content_id,
                        "content_seed": content_seed,
                        "split": split_for_content(content_id),
                        "path": destination.relative_to(output_root).as_posix(),
                        "complete_sha256": _sha256_file(destination / "COMPLETE.json"),
                        "valid": True,
                    }
                    print(f"[paired] published {destination}", flush=True)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if stage_dir.exists():
                        rejected_path = _reject_path(stage_dir, rejected_root, type(exc).__name__)
                        rejected_records.append(
                            {
                                "content_id": content_id,
                                "content_seed": content_seed,
                                "path": str(rejected_path),
                                "error": error,
                            }
                        )
                    else:
                        rejected_records.append(
                            {
                                "content_id": content_id,
                                "content_seed": content_seed,
                                "path": None,
                                "error": error,
                            }
                        )
                    print(f"[paired] rejected seed {content_seed}: {error}", file=sys.stderr, flush=True)
                    traceback.print_exc()
                finally:
                    _safe_close_env(task)
                    _atomic_write_json(
                        manifest_path,
                        _manifest_value(
                            output_root=output_root,
                            requested_contents=num_contents,
                            start_seed=start_seed,
                            texture_choices=texture_choices,
                            contents=completed,
                            attempts_total=attempts_total,
                            rejected=rejected_records,
                        ),
                    )
                    _atomic_write_json(
                        run_state_path,
                        _run_state_value(
                            status="running",
                            requested_contents=num_contents,
                            start_seed=start_seed,
                            next_seed=next_seed,
                            attempts_total=attempts_total,
                            attempts_this_run=attempts_this_run,
                            completed_ids=list(completed),
                        ),
                    )

            expected_ids = set(range(num_contents))
            if set(completed) != expected_ids:
                raise PairCollectionError(
                    "refusing to mark an inexact content set complete: "
                    f"expected={sorted(expected_ids)}, actual={sorted(completed)}"
                )
            completed_seeds = [int(value["content_seed"]) for value in completed.values()]
            if len(set(completed_seeds)) != len(completed_seeds):
                raise PairCollectionError(
                    "refusing to mark a dataset with duplicate content seeds complete"
                )

            _atomic_write_json(
                run_state_path,
                _run_state_value(
                    status="complete",
                    requested_contents=num_contents,
                    start_seed=start_seed,
                    next_seed=next_seed,
                    attempts_total=attempts_total,
                    attempts_this_run=attempts_this_run,
                    completed_ids=list(completed),
                ),
            )
        except Exception as exc:
            _atomic_write_json(
                run_state_path,
                _run_state_value(
                    status="failed",
                    requested_contents=num_contents,
                    start_seed=start_seed,
                    next_seed=next_seed,
                    attempts_total=attempts_total,
                    attempts_this_run=attempts_this_run,
                    completed_ids=list(completed),
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise
        finally:
            if task is not None:
                _safe_close_env(task)


def _bounded_contents(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_CONTENTS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_CONTENTS}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect transactional clean + 3-style strict pairs for a RoboTwin task."
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        default=DEFAULT_TASK,
        help=f"RoboTwin task ID (default: {DEFAULT_TASK})",
    )
    parser.add_argument(
        "--num-contents",
        type=_bounded_contents,
        default=MAX_CONTENTS,
        help=f"number of valid content trajectories (default/max: {MAX_CONTENTS})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="transactional dataset root (default: data/<task>/paired_random_background)",
    )
    parser.add_argument("--start-seed", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=1000,
        help="maximum candidate content seeds tried by this invocation",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _require_robotwin_cwd()
    if str(ROBOTWIN_ROOT) not in sys.path:
        sys.path.insert(0, str(ROBOTWIN_ROOT))

    try:
        physical_gpu_index, pci_address = _require_single_numeric_visible_gpu()
        render_device_alias = os.environ["ROBOTWIN_RENDER_DEVICE_ALIAS"]
        print(
            f"[paired] requested physical GPU {physical_gpu_index} "
            f"at PCI {pci_address} as SAPIEN device {render_device_alias}",
            flush=True,
        )

        # Match RoboTwin's supported collection initialization sequence while
        # proving the explicit CUDA/Vulkan device is usable.
        from test_render import Sapien_TEST

        Sapien_TEST(render_device_alias=render_device_alias)
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        output_root = args.output_root
        if output_root is None:
            output_root = Path("data") / args.task / "paired_random_background"
        collect(
            task_name=args.task,
            num_contents=args.num_contents,
            output_root=output_root,
            start_seed=args.start_seed,
            max_attempts=args.max_attempts,
        )
    except Exception as exc:
        print(f"paired collection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
