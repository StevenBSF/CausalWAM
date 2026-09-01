#!/usr/bin/env python3
"""Fail-closed reader for the native RoboTwin paired-rendering dataset.

The physical sample identity is ``(task, content_id, frame_idx)``.  ``frame_idx``
is an ordinal in the saved HDF5 episode, while ``trace_idx`` maps it to the
physics-step state in ``state_trace.npz``.  The distinction matters because
multiple saved frames may legitimately refer to the same physics state.

This module intentionally reads the native paired HDF5/NPZ layout.  It does not
convert or repair data and it never reads ``.staging`` or ``rejected``.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
VARIANTS = (
    "clean",
    "style_00_seed_0",
    "style_01_seed_1",
    "style_02_seed_2",
)
E0_E1_PROTOCOL = "e0_e1"
R3_HOLDOUT_PROTOCOL = "r3_holdout_v1"
# User-facing aliases remain accepted, but E2 and E3 intentionally share one
# canonical data protocol.  Their sole experimental difference is proprio mode.
E2_PROTOCOL = "e2"
E3_PROTOCOL = "e3"
R3_VARIANT = "style_02_seed_2"
SEEN_VARIANTS = VARIANTS[:3]
UNSEEN_TEST_VARIANTS = (VARIANTS[0], R3_VARIANT)
VARIANT_PROTOCOLS: dict[str, dict[str, tuple[str, ...]]] = {
    E0_E1_PROTOCOL: {split: VARIANTS for split in ("train", "val", "test")},
    R3_HOLDOUT_PROTOCOL: {
        "train": SEEN_VARIANTS,
        "val": SEEN_VARIANTS,
        "test": UNSEEN_TEST_VARIANTS,
    },
}
PROTOCOL_ALIASES = {E2_PROTOCOL: R3_HOLDOUT_PROTOCOL, E3_PROTOCOL: R3_HOLDOUT_PROTOCOL}
STYLE_SEEDS = (0, 1, 2)
USED_CAMERAS = ("head_camera", "left_camera", "right_camera")
SPLITS = ("train", "val", "test")
EXPECTED_CONTENTS = 50
CONTENT_RE = re.compile(r"^content_(\d{6})$")
EXPECTED_DOMAIN_RANDOMIZATION = {
    "clean_background_rate": 0,
    "cluttered_table": False,
    "crazy_random_light_rate": 0,
    "random_embodiment": False,
    "random_head_camera_dis": 0,
    "random_light": False,
    "random_table_height": 0,
}

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
    "frame_trace_index",
)
INITIAL_KEYS = STATE_KEYS
HDF_TRACE_BINDINGS = {
    "left_qpos": "robot_state/left_qpos",
    "right_qpos": "robot_state/right_qpos",
    "left_qvel": "robot_state/left_qvel",
    "right_qvel": "robot_state/right_qvel",
    "left_eef": "endpose/left_endpose",
    "right_eef": "endpose/right_endpose",
}


class PairedDataError(RuntimeError):
    """The dataset does not prove the correspondence required by E0/E1."""


def canonical_protocol(protocol: str) -> str:
    canonical = PROTOCOL_ALIASES.get(str(protocol), str(protocol))
    if canonical not in VARIANT_PROTOCOLS:
        raise PairedDataError(
            f"unknown experiment protocol {protocol!r}; expected one of "
            f"{tuple((*VARIANT_PROTOCOLS, *PROTOCOL_ALIASES))}"
        )
    return canonical


def variants_for_protocol(protocol: str, split: str) -> tuple[str, ...]:
    """Return the only rendering set admissible for an experiment/split.

    E2 and E3 deliberately share this mapping: R3 is absent from every
    train/validation sample and is materialized only beside Clean for test.
    Keeping this policy in the data layer prevents a caller from extracting R3
    and merely promising not to use its cached tokens later.
    """

    canonical = canonical_protocol(protocol)
    if split not in SPLITS:
        raise PairedDataError(f"split must be one of {SPLITS}")
    return VARIANT_PROTOCOLS[canonical][split]


def split_for_content(content_id: int) -> str:
    if 0 <= content_id < 30:
        return "train"
    if 30 <= content_id < 40:
        return "val"
    if 40 <= content_id < 50:
        return "test"
    raise PairedDataError(f"content_id {content_id} is outside the fixed range 0..49")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _array_value_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Exact numeric-value equality while allowing intentional dtype promotion."""
    return left.shape == right.shape and np.array_equal(left, right)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PairedDataError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PairedDataError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


@lru_cache(maxsize=12)
def _load_npz_cached(path_text: str) -> dict[str, np.ndarray]:
    path = Path(path_text)
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(bool(archive.files), f"empty NPZ: {path}")
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except PairedDataError:
        raise
    except Exception as exc:
        raise PairedDataError(f"cannot safely load NPZ {path}: {exc}") from exc
    for key, value in arrays.items():
        _require(not value.dtype.hasobject, f"object dtype is forbidden: {path}:{key}")
    return arrays


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    # Return the cached immutable-by-convention arrays.  Consumers never receive
    # these arrays directly; sample state values are copied into Python floats.
    return _load_npz_cached(str(path.resolve()))


def _assert_array_mappings_equal(
    clean: Mapping[str, np.ndarray],
    other: Mapping[str, np.ndarray],
    *,
    label: str,
) -> None:
    _require(set(clean) == set(other), f"{label}: NPZ keys differ")
    for key in sorted(clean):
        _require(
            _array_bytes_equal(clean[key], other[key]),
            f"{label}:{key} is not byte-identical to clean",
        )


def _is_rgb_leaf(path: str) -> bool:
    return any(
        "rgb" in component.lower().replace("_", "") for component in path.split("/")
    )


def _hdf_inventory(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    try:
        with h5py.File(path, "r") as handle:

            def visit(name: str, item: Any) -> None:
                if isinstance(item, h5py.Dataset):
                    result[name] = (
                        tuple(int(value) for value in item.shape),
                        str(item.dtype),
                    )

            handle.visititems(visit)
    except Exception as exc:
        raise PairedDataError(f"cannot inspect HDF5 {path}: {exc}") from exc
    _require(bool(result), f"HDF5 contains no datasets: {path}")
    return result


def _required_hdf_paths() -> set[str]:
    paths = set(HDF_TRACE_BINDINGS.values()) | {"joint_action/vector"}
    for camera in USED_CAMERAS:
        prefix = f"observation/{camera}"
        paths.update(
            {
                f"{prefix}/rgb",
                f"{prefix}/cam2world_gl",
                f"{prefix}/extrinsic_cv",
                f"{prefix}/intrinsic_cv",
            }
        )
    return paths


@dataclass(frozen=True, order=True)
class PhysicalKey:
    task: str
    content_id: int
    frame_idx: int

    def as_string(self) -> str:
        return f"{self.task}/content_{self.content_id:06d}/frame_{self.frame_idx:06d}"


@dataclass(frozen=True)
class PairedTrajectory:
    task: str
    content_id: int
    content_seed: int
    split: str
    content_dir: Path
    variant_names: tuple[str, ...]
    variant_dirs: tuple[Path, ...]
    hdf5_paths: tuple[Path, ...]
    state_trace_paths: tuple[Path, ...]
    action_trace_paths: tuple[Path, ...]
    frame_count: int
    frame_trace_index: tuple[int, ...]
    task_state_layout: tuple[str, ...]
    non_rgb_leaves: tuple[str, ...]


@dataclass(frozen=True)
class FrameRef:
    trajectory_index: int
    key: PhysicalKey
    trace_idx: int


COMMON_METADATA_FIELDS = (
    "task",
    "content_id",
    "content_seed",
    "split",
    "source_clean_episode",
    "source_clean_hdf5_path",
    "source_trajectory_sha256",
    "rng_state_sha256_after_setup",
    "task_identity",
    "task_identity_sha256",
    "task_success_spec",
    "task_success_spec_sha256",
    "task_state_layout",
    "render_device",
    "frame_count",
    "action_rows",
    "trace_rows",
    "frame_trace_index",
)


def _assert_exact_variant_layout(
    content_dir: Path, *, active_variants: Sequence[str] = VARIANTS
) -> None:
    active = tuple(str(variant) for variant in active_variants)
    _require(
        bool(active)
        and active[0] == "clean"
        and len(set(active)) == len(active)
        and set(active) <= set(VARIANTS),
        f"{content_dir}: invalid active variants {active}",
    )
    # The atomically published source still has the canonical four-variant
    # directory inventory.  Listing its names is safe; only active directories
    # are entered or opened below.
    expected = {*VARIANTS, "COMPLETE.json"}
    actual = {entry.name for entry in content_dir.iterdir()}
    _require(
        actual == expected,
        f"{content_dir}: expected only {sorted(expected)}, got {sorted(actual)}",
    )
    for variant in active:
        directory = content_dir / variant
        expected_top = {
            "metadata.json",
            "initial_state.npz",
            "action_trace.npz",
            "state_trace.npz",
            "data",
            "video",
        }
        if variant == "clean":
            expected_top.add("_traj_data")
        actual_top = {entry.name for entry in directory.iterdir()}
        _require(
            actual_top == expected_top,
            f"{directory}: non-canonical variant layout {sorted(actual_top)}",
        )
        for subdirectory, filename in (
            ("data", "episode0.hdf5"),
            ("video", "episode0.mp4"),
        ):
            subdir = directory / subdirectory
            _require(subdir.is_dir(), f"missing directory: {subdir}")
            _require(
                {entry.name for entry in subdir.iterdir()} == {filename},
                f"{subdir}: expected exactly {filename}",
            )
        if variant == "clean":
            source = directory / "_traj_data"
            _require(
                {entry.name for entry in source.iterdir()} == {"episode0.pkl"},
                f"{source}: expected exactly episode0.pkl",
            )


def _assert_metadata_hash(metadata: Mapping[str, Any], key: str, path: Path) -> None:
    hashes = metadata.get("hashes")
    _require(isinstance(hashes, dict), f"{path.parent}: metadata.hashes is missing")
    declared = hashes.get(key)
    _require(
        isinstance(declared, str) and len(declared) == 64,
        f"invalid declared {key} hash",
    )
    actual = _sha256(path)
    _require(declared.lower() == actual, f"stale {key} hash for {path}")


def _hdf_rgb_sha256(path: Path, camera: str) -> str:
    dataset_path = f"observation/{camera}/rgb"
    with h5py.File(path, "r") as handle:
        _require(dataset_path in handle, f"{path}: missing {dataset_path}")
        array = np.asarray(handle[dataset_path][()])
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _assert_domain_metadata(
    metadata: Mapping[str, Any], *, is_clean: bool, label: str
) -> None:
    domain = metadata.get("domain_randomization")
    _require(isinstance(domain, dict), f"{label}: missing domain_randomization")
    expected = {
        **EXPECTED_DOMAIN_RANDOMIZATION,
        "random_background": not is_clean,
    }
    _require(domain == expected, f"{label}: unexpected domain randomization {domain}")
    _require(
        metadata.get("need_plan") is False, f"{label}: published replay must not plan"
    )
    _require(
        metadata.get("planner_calls") == 0, f"{label}: published replay called planner"
    )
    _require(
        metadata.get("path_fully_consumed") is True,
        f"{label}: source trajectory was not fully consumed",
    )
    consumption = metadata.get("path_consumption")
    _require(isinstance(consumption, dict), f"{label}: path_consumption is missing")
    _require(
        consumption.get("fully_consumed") is True, f"{label}: path consumption false"
    )
    for arm in ("left", "right"):
        arm_value = consumption.get(arm)
        _require(
            isinstance(arm_value, dict), f"{label}: {arm} path consumption missing"
        )
        _require(
            arm_value.get("available") == arm_value.get("consumed"),
            f"{label}: {arm} path was not consumed exactly",
        )


def _validate_trace_schema(
    initial: Mapping[str, np.ndarray],
    action: Mapping[str, np.ndarray],
    state: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
    label: str,
) -> tuple[int, ...]:
    _require(
        tuple(action) == ACTION_KEYS, f"{label}: action_trace key order/schema mismatch"
    )
    _require(
        tuple(state) == STATE_KEYS, f"{label}: state_trace key order/schema mismatch"
    )
    _require(
        tuple(initial) == INITIAL_KEYS,
        f"{label}: initial_state key order/schema mismatch",
    )

    action_rows = int(metadata["action_rows"])
    trace_rows = int(metadata["trace_rows"])
    frame_count = int(metadata["frame_count"])
    _require(
        trace_rows == action_rows + 1, f"{label}: trace_rows must equal action_rows + 1"
    )
    for key in ACTION_KEYS:
        _require(
            action[key].ndim >= 1 and action[key].shape[0] == action_rows,
            f"{label}:{key} action length",
        )
    for key in STATE_KEYS:
        expected = frame_count if key == "frame_trace_index" else trace_rows
        _require(
            state[key].ndim >= 1 and state[key].shape[0] == expected,
            f"{label}:{key} state length",
        )
    for key in INITIAL_KEYS:
        _require(
            initial[key].ndim >= 1 and initial[key].shape[0] == 1,
            f"{label}:{key} initial length",
        )

    for key in set(STATE_KEYS) - {"frame_trace_index"}:
        _require(
            _array_bytes_equal(initial[key], state[key][:1]),
            f"{label}: initial {key} != state row 0",
        )
    _require(
        np.array_equal(initial["frame_trace_index"], np.asarray([0], dtype=np.int64)),
        f"{label}: initial frame_trace_index must be int64 [0]",
    )
    _require(
        np.array_equal(state["semantic_action"][1:], action["semantic_action"]),
        f"{label}: semantic action sequence is misaligned",
    )
    indices = state["frame_trace_index"]
    _require(
        indices.dtype.kind in "iu" and indices.ndim == 1,
        f"{label}: invalid frame_trace_index",
    )
    indices_i64 = indices.astype(np.int64, copy=False)
    _require(indices_i64.size == frame_count, f"{label}: frame map length mismatch")
    _require(np.all(indices_i64 >= 0), f"{label}: negative frame trace index")
    _require(
        np.all(indices_i64[1:] >= indices_i64[:-1]),
        f"{label}: decreasing frame trace index",
    )
    _require(
        int(indices_i64[-1]) < trace_rows, f"{label}: frame trace index out of bounds"
    )
    metadata_indices = tuple(int(value) for value in metadata["frame_trace_index"])
    result = tuple(int(value) for value in indices_i64)
    _require(
        metadata_indices == result, f"{label}: metadata frame map differs from NPZ"
    )

    layout = tuple(metadata["task_state_layout"])
    _require(state["task_state"].ndim == 2, f"{label}: task_state must be rank two")
    _require(
        state["task_state"].shape[1] == len(layout),
        f"{label}: task_state layout width mismatch",
    )
    for arrays in (initial, action, state):
        for key, array in arrays.items():
            if array.dtype.kind in "fc":
                _require(np.all(np.isfinite(array)), f"{label}:{key} contains NaN/inf")
    return result


def _validate_hdf5_group(
    paths: tuple[Path, ...],
    *,
    variant_names: tuple[str, ...],
    frame_count: int,
    clean_state: Mapping[str, np.ndarray],
    frame_trace_index: tuple[int, ...],
    label: str,
) -> tuple[str, ...]:
    _require(
        len(paths) == len(variant_names) and bool(paths),
        f"{label}: HDF5 paths/variant names differ",
    )
    inventories = tuple(_hdf_inventory(path) for path in paths)
    required = _required_hdf_paths()
    clean_names = set(inventories[0])
    _require(
        required <= clean_names,
        f"{label}: missing HDF5 leaves {sorted(required - clean_names)}",
    )
    for index, inventory in enumerate(inventories):
        _require(
            set(inventory) == clean_names,
            f"{label}/{variant_names[index]}: HDF5 leaf set differs",
        )
        for name, (shape, _) in inventory.items():
            _require(
                bool(shape) and shape[0] == frame_count,
                f"{label}/{variant_names[index]}:{name} frame count",
            )

    non_rgb = tuple(sorted(name for name in clean_names if not _is_rgb_leaf(name)))
    rgb_hashes: dict[str, list[str]] = {camera: [] for camera in USED_CAMERAS}
    with h5py.File(paths[0], "r") as clean_handle:
        for other_index in range(1, len(paths)):
            with h5py.File(paths[other_index], "r") as other_handle:
                for name in non_rgb:
                    clean_array = np.asarray(clean_handle[name][()])
                    other_array = np.asarray(other_handle[name][()])
                    _require(
                        _array_bytes_equal(clean_array, other_array),
                        f"{label}/{variant_names[other_index]}:{name} non-RGB data differs",
                    )
        indices = np.asarray(frame_trace_index, dtype=np.int64)
        for trace_key, hdf_path in HDF_TRACE_BINDINGS.items():
            hdf_array = np.asarray(clean_handle[hdf_path][()])
            selected = np.asarray(clean_state[trace_key][indices])
            _require(
                hdf_array.shape == selected.shape
                and np.array_equal(hdf_array, selected),
                f"{label}:{hdf_path} != state_trace[{trace_key}][frame_trace_index]",
            )

    decoded_shapes: dict[str, list[tuple[int, ...]]] = {
        camera: [] for camera in USED_CAMERAS
    }
    for variant_index, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            for camera in USED_CAMERAS:
                rgb_path = f"observation/{camera}/rgb"
                array = np.asarray(handle[rgb_path][()])
                _require(
                    array.ndim == 1 and array.shape[0] == frame_count,
                    f"{label}:{rgb_path} shape",
                )
                _require(
                    array.dtype.kind in "SUV",
                    f"{label}:{rgb_path} must store encoded bytes",
                )
                rgb_hashes[camera].append(hashlib.sha256(array.tobytes()).hexdigest())
                for boundary_index in sorted({0, frame_count - 1}):
                    decoded = _decode_rgb(
                        array[boundary_index],
                        label=(
                            f"{label}/{variant_names[variant_index]}/{camera}/"
                            f"frame_{boundary_index:06d}"
                        ),
                    )
                    decoded_shapes[camera].append(
                        tuple(int(value) for value in decoded.shape)
                    )
    for camera, hashes in rgb_hashes.items():
        _require(
            len(set(hashes)) == len(variant_names),
            f"{label}:{camera} RGB is not pairwise different across variants",
        )
        _require(
            len(set(decoded_shapes[camera])) == 1,
            f"{label}:{camera} decoded RGB dimensions differ across variants/frames",
        )
    return non_rgb


def _validate_content(
    content_dir: Path,
    task: str,
    content_id: int,
    *,
    active_variants: Sequence[str] = VARIANTS,
) -> PairedTrajectory:
    label = f"{task}/content_{content_id:06d}"
    variant_names = tuple(str(variant) for variant in active_variants)
    _assert_exact_variant_layout(content_dir, active_variants=variant_names)
    marker = _read_json(content_dir / "COMPLETE.json")
    _require(marker.get("schema_version") == 2, f"{label}: unsupported COMPLETE schema")
    _require(marker.get("task") == task, f"{label}: COMPLETE task mismatch")
    _require(
        marker.get("content_id") == content_id, f"{label}: COMPLETE content ID mismatch"
    )
    _require(
        marker.get("split") == split_for_content(content_id),
        f"{label}: COMPLETE split mismatch",
    )
    _require(
        marker.get("style_seeds") == list(STYLE_SEEDS),
        f"{label}: COMPLETE style seeds mismatch",
    )
    validation = marker.get("validation")
    _require(
        isinstance(validation, dict)
        and validation.get("valid") is True
        and not validation.get("errors"),
        f"{label}: COMPLETE does not prove successful staged validation",
    )

    variant_dirs = tuple(content_dir / name for name in variant_names)
    metadata = tuple(_read_json(path / "metadata.json") for path in variant_dirs)
    clean_metadata = metadata[0]
    for field in COMMON_METADATA_FIELDS:
        values = [item.get(field) for item in metadata]
        _require(
            all(value == values[0] for value in values[1:]),
            f"{label}: metadata {field} differs",
        )
    _require(
        clean_metadata.get("schema_version") == 2,
        f"{label}: unsupported metadata schema",
    )
    _require(clean_metadata.get("task") == task, f"{label}: metadata task mismatch")
    _require(
        clean_metadata.get("content_id") == content_id,
        f"{label}: metadata content ID mismatch",
    )
    _require(
        clean_metadata.get("split") == split_for_content(content_id),
        f"{label}: metadata split mismatch",
    )
    _require(
        clean_metadata.get("success") is True,
        f"{label}: clean replay was not successful",
    )
    _require(
        clean_metadata.get("variant") == "clean",
        f"{label}: clean variant label mismatch",
    )
    _require(
        clean_metadata.get("intervention") == "none",
        f"{label}: clean intervention mismatch",
    )
    _require(
        clean_metadata.get("style_seed") is None,
        f"{label}: clean style seed must be null",
    )
    _assert_domain_metadata(clean_metadata, is_clean=True, label=f"{label}/clean")
    for item, variant in zip(metadata[1:], variant_names[1:], strict=True):
        canonical_index = VARIANTS.index(variant) - 1
        _require(item.get("success") is True, f"{label}/{variant}: replay unsuccessful")
        _require(
            item.get("variant") == variant,
            f"{label}/{variant}: metadata variant mismatch",
        )
        _require(
            item.get("intervention") == "random_background",
            f"{label}/{variant}: intervention",
        )
        _require(
            item.get("style_index") == canonical_index,
            f"{label}/{variant}: style index",
        )
        _require(
            item.get("style_seed") == STYLE_SEEDS[canonical_index],
            f"{label}/{variant}: style seed",
        )
        _assert_domain_metadata(item, is_clean=False, label=f"{label}/{variant}")
    for field in (
        "content_seed",
        "task_identity",
        "task_identity_sha256",
        "task_state_layout",
    ):
        _require(
            marker.get(field) == clean_metadata.get(field),
            f"{label}: COMPLETE {field} mismatch",
        )

    source_path = variant_dirs[0] / "_traj_data/episode0.pkl"
    source_hash = _sha256(source_path)
    _require(
        marker.get("source_trajectory_sha256") == source_hash,
        f"{label}: source trajectory hash",
    )
    _require(
        clean_metadata.get("source_trajectory_sha256") == source_hash,
        f"{label}: metadata source hash",
    )

    artifact_names = {
        "initial_state": "initial_state.npz",
        "action_trace": "action_trace.npz",
        "state_trace": "state_trace.npz",
        "hdf5": "data/episode0.hdf5",
        "video": "video/episode0.mp4",
    }
    for variant_index, directory in enumerate(variant_dirs):
        for hash_name, relative in artifact_names.items():
            _assert_metadata_hash(
                metadata[variant_index], hash_name, directory / relative
            )
        declared_head_hash = metadata[variant_index].get("hashes", {}).get("head_rgb")
        actual_head_hash = _hdf_rgb_sha256(
            directory / "data/episode0.hdf5", "head_camera"
        )
        _require(
            declared_head_hash == actual_head_hash,
            f"{label}/{variant_names[variant_index]}: stale head RGB hash",
        )

    initial_paths = tuple(path / "initial_state.npz" for path in variant_dirs)
    action_paths = tuple(path / "action_trace.npz" for path in variant_dirs)
    state_paths = tuple(path / "state_trace.npz" for path in variant_dirs)
    initials = tuple(_load_npz(path) for path in initial_paths)
    actions = tuple(_load_npz(path) for path in action_paths)
    states = tuple(_load_npz(path) for path in state_paths)
    frame_map = _validate_trace_schema(
        initials[0],
        actions[0],
        states[0],
        metadata=clean_metadata,
        label=f"{label}/clean",
    )
    for variant_index in range(1, len(variant_names)):
        variant_label = f"{label}/{variant_names[variant_index]}"
        _validate_trace_schema(
            initials[variant_index],
            actions[variant_index],
            states[variant_index],
            metadata=metadata[variant_index],
            label=variant_label,
        )
        _assert_array_mappings_equal(
            initials[0], initials[variant_index], label=f"{variant_label}/initial"
        )
        _assert_array_mappings_equal(
            actions[0], actions[variant_index], label=f"{variant_label}/action"
        )
        _assert_array_mappings_equal(
            states[0], states[variant_index], label=f"{variant_label}/state"
        )

    hdf5_paths = tuple(path / "data/episode0.hdf5" for path in variant_dirs)
    non_rgb = _validate_hdf5_group(
        hdf5_paths,
        variant_names=variant_names,
        frame_count=int(clean_metadata["frame_count"]),
        clean_state=states[0],
        frame_trace_index=frame_map,
        label=label,
    )
    return PairedTrajectory(
        task=task,
        content_id=content_id,
        content_seed=int(clean_metadata["content_seed"]),
        split=split_for_content(content_id),
        content_dir=content_dir,
        variant_names=variant_names,
        variant_dirs=variant_dirs,
        hdf5_paths=hdf5_paths,
        state_trace_paths=state_paths,
        action_trace_paths=action_paths,
        frame_count=int(clean_metadata["frame_count"]),
        frame_trace_index=frame_map,
        task_state_layout=tuple(
            str(value) for value in clean_metadata["task_state_layout"]
        ),
        non_rgb_leaves=non_rgb,
    )


def _validate_task_root_state(
    task_root: Path,
    *,
    task: str,
    ids: Sequence[int],
    formal: bool,
    deep_content_ids: Sequence[int] | None = None,
) -> None:
    root_manifest = _read_json(task_root / "manifest.json")
    run_state = _read_json(task_root / "run_state.json")
    _require(root_manifest.get("task") == task, f"{task}: root manifest task mismatch")
    _require(run_state.get("task") == task, f"{task}: run_state task mismatch")
    entries = root_manifest.get("completed_contents")
    _require(isinstance(entries, list), f"{task}: completed_contents is missing")
    by_id: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        _require(isinstance(entry, dict), f"{task}: malformed completed content entry")
        content_id = entry.get("content_id")
        _require(
            isinstance(content_id, int) and not isinstance(content_id, bool),
            f"{task}: malformed manifest content_id",
        )
        _require(
            content_id not in by_id,
            f"{task}: duplicate root manifest content_id {content_id}",
        )
        by_id[content_id] = entry
    _require(
        set(by_id) == set(ids),
        f"{task}: root manifest IDs differ from published contents",
    )
    deep_ids = set(ids if deep_content_ids is None else deep_content_ids)
    _require(deep_ids <= set(ids), f"{task}: deep content IDs are not published")
    seeds: list[int] = []
    for content_id in ids:
        entry = by_id[content_id]
        expected_path = f"contents/content_{content_id:06d}"
        _require(
            entry.get("path") == expected_path,
            f"{task}/{content_id}: root manifest path",
        )
        _require(
            entry.get("split") == split_for_content(content_id),
            f"{task}/{content_id}: root split",
        )
        _require(
            entry.get("valid") is True,
            f"{task}/{content_id}: root manifest invalid",
        )
        seed = entry.get("content_seed")
        _require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"{task}/{content_id}: seed",
        )
        seeds.append(seed)
        if content_id in deep_ids:
            marker = task_root / expected_path / "COMPLETE.json"
            _require(
                entry.get("complete_sha256") == _sha256(marker),
                f"{task}/{content_id}: COMPLETE hash",
            )
    _require(
        len(set(seeds)) == len(seeds),
        f"{task}: successful content seeds are not unique",
    )
    completed_ids = run_state.get("completed_content_ids")
    _require(completed_ids == list(ids), f"{task}: run_state published IDs differ")
    _require(
        root_manifest.get("completed_count") == len(ids),
        f"{task}: root completed count",
    )
    if formal:
        _require(
            run_state.get("status") == "complete",
            f"{task}: formal run is not complete",
        )
        _require(
            run_state.get("requested_contents") == EXPECTED_CONTENTS,
            f"{task}: run target",
        )
        _require(
            root_manifest.get("requested_contents") == EXPECTED_CONTENTS,
            f"{task}: manifest target",
        )
        expected_counts = {
            "train": {"clean": 30, "random": 90},
            "val": {"clean": 10, "random": 30},
            "test": {"clean": 10, "random": 30},
        }
        _require(
            root_manifest.get("completed_split_counts") == expected_counts,
            f"{task}: formal split counts are not 30/10/10 physical trajectories",
        )


def _decode_rgb(encoded: Any, *, label: str) -> np.ndarray:
    if isinstance(encoded, np.ndarray):
        payload = encoded.tobytes()
    else:
        payload = bytes(encoded)
    payload = payload.rstrip(b"\x00")
    _require(bool(payload), f"{label}: empty encoded RGB")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            result = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise PairedDataError(f"{label}: cannot decode RGB: {exc}") from exc
    _require(
        result.ndim == 3 and result.shape[2] == 3,
        f"{label}: decoded RGB shape {result.shape}",
    )
    return result


def _deployment_composite(
    head: np.ndarray, left: np.ndarray, right: np.ndarray
) -> torch.Tensor:
    resampling = Image.Resampling.BILINEAR
    head_image = np.asarray(
        Image.fromarray(head).resize((320, 256), resample=resampling), dtype=np.uint8
    )
    left_image = np.asarray(
        Image.fromarray(left).resize((160, 128), resample=resampling), dtype=np.uint8
    )
    right_image = np.asarray(
        Image.fromarray(right).resize((160, 128), resample=resampling), dtype=np.uint8
    )
    bottom = np.concatenate((left_image, right_image), axis=1)
    composite = np.concatenate((head_image, bottom), axis=0)
    _require(
        composite.shape == (384, 320, 3),
        f"unexpected deployment composite {composite.shape}",
    )
    # Keep decoded pixels as uint8 here.  RoboTwin deployment first casts this
    # tensor to the model dtype and only then applies ``2/255 - 1``; normalizing
    # in float32 before a later bf16 cast is observably different for half of
    # the possible pixel values.
    chw = np.ascontiguousarray(composite.transpose(2, 0, 1), dtype=np.uint8)
    return torch.from_numpy(chw)


def _physical_state_by_name(
    state: Mapping[str, np.ndarray],
    trace_idx: int,
    task_layout: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    task_values = np.asarray(state["task_state"][trace_idx], dtype=np.float64).reshape(
        -1
    )
    _require(
        len(task_layout) == task_values.size,
        "task state layout changed at sampling time",
    )
    task_named = {
        str(name): float(value) for name, value in zip(task_layout, task_values)
    }
    physical = {f"task.{name}": value for name, value in task_named.items()}
    for key in (
        "left_qpos",
        "right_qpos",
        "left_qvel",
        "right_qvel",
        "left_eef",
        "right_eef",
    ):
        values = np.asarray(state[key][trace_idx], dtype=np.float64).reshape(-1)
        physical.update(
            {f"robot.{key}.{index}": float(value) for index, value in enumerate(values)}
        )
    for key in (
        "left_gripper_open",
        "right_gripper_open",
        "left_gripper_closed",
        "right_gripper_closed",
    ):
        physical[f"robot.{key}"] = float(bool(state[key][trace_idx]))
    return task_named, physical


def compatible_state_vectors(
    left: Mapping[str, float], right: Mapping[str, float]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Align two variable-layout physical states by their semantic field names."""
    names = tuple(sorted(set(left) & set(right)))
    _require(bool(names), "physical states have no compatible named fields")
    left_values = np.asarray([left[name] for name in names], dtype=np.float64)
    right_values = np.asarray([right[name] for name in names], dtype=np.float64)
    _require(
        np.all(np.isfinite(left_values)) and np.all(np.isfinite(right_values)),
        "non-finite state",
    )
    return names, left_values, right_values


class PairedFrameDataset(Dataset[dict[str, Any]]):
    """Uniform single-frame samples with strictly corresponding renderings.

    Args:
        data_root: RoboTwin ``data/`` directory.  For a single task, its
            ``paired_random_background`` directory is also accepted.
        tasks: Exact task subset.
        split: One of train/val/test.
        states_per_trajectory: Deterministic, uniformly spaced saved frames.
        allow_incomplete: Smoke-only mode.  It accepts a contiguous published
            prefix but still fully validates every selected content group.
        max_trajectories_per_task: Smoke-only cap applied to that prefix.
        content_ids: Smoke-only exact content IDs, useful for selecting a
            validation/test trajectory without weakening correspondence checks.
        protocol: Rendering policy.  ``e0_e1`` preserves the original four
            renderings.  ``e2``/``e3`` expose only Clean/R1/R2 for train/val
            and only Clean/R3 for test.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        tasks: Sequence[str] = TASKS,
        split: str = "train",
        states_per_trajectory: int = 8,
        allow_incomplete: bool = False,
        max_trajectories_per_task: int | None = None,
        content_ids: Sequence[int] | None = None,
        protocol: str = E0_E1_PROTOCOL,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.tasks = tuple(tasks)
        self.split = split
        self.states_per_trajectory = int(states_per_trajectory)
        self.allow_incomplete = bool(allow_incomplete)
        self.protocol = canonical_protocol(str(protocol))
        self.active_variants = variants_for_protocol(self.protocol, self.split)
        requested_content_ids = (
            None if content_ids is None else tuple(int(value) for value in content_ids)
        )
        _require(
            bool(self.tasks) and len(set(self.tasks)) == len(self.tasks),
            "tasks must be distinct",
        )
        _require(set(self.tasks) <= set(TASKS), f"unsupported task in {self.tasks}")
        _require(split in SPLITS, f"split must be one of {SPLITS}")
        _require(
            self.states_per_trajectory > 0, "states_per_trajectory must be positive"
        )
        if max_trajectories_per_task is not None:
            _require(self.allow_incomplete, "max_trajectories_per_task is smoke-only")
            _require(
                max_trajectories_per_task > 0,
                "max_trajectories_per_task must be positive",
            )
        if requested_content_ids is not None:
            _require(self.allow_incomplete, "content_ids is smoke-only")
            _require(
                bool(requested_content_ids)
                and len(set(requested_content_ids)) == len(requested_content_ids),
                "content_ids must be a non-empty distinct list",
            )
            for content_id in requested_content_ids:
                requested_split = split_for_content(content_id)
                _require(
                    requested_split == split,
                    f"content_id {content_id} belongs to {requested_split}, not {split}",
                )
        trajectories: list[PairedTrajectory] = []
        for task in self.tasks:
            task_root = self._resolve_task_root(task)
            contents_root = task_root / "contents"
            _require(
                contents_root.is_dir(), f"missing contents directory: {contents_root}"
            )
            entries = tuple(contents_root.iterdir())
            if self.protocol == R3_HOLDOUT_PROTOCOL:
                # Do not stat/open out-of-split content paths while constructing
                # a strict holdout loader.  The root manifest proves the
                # published ID inventory; selected paths are deeply validated
                # below before use.
                malformed = [
                    entry.name
                    for entry in entries
                    if CONTENT_RE.fullmatch(entry.name) is None
                ]
            else:
                malformed = [
                    entry.name
                    for entry in entries
                    if not entry.is_dir() or CONTENT_RE.fullmatch(entry.name) is None
                ]
            _require(
                not malformed,
                f"{contents_root}: non-canonical entries {sorted(malformed)}",
            )
            ids = sorted(
                int(CONTENT_RE.fullmatch(entry.name).group(1)) for entry in entries
            )  # type: ignore[union-attr]
            published_ids = list(ids)
            if self.allow_incomplete:
                _require(bool(ids), f"{task}: no atomically published content")
                _require(
                    ids == list(range(ids[-1] + 1)),
                    f"{task}: published contents are not a contiguous prefix",
                )
                if requested_content_ids is not None:
                    _require(
                        set(requested_content_ids) <= set(ids),
                        f"{task}: requested smoke content IDs are not all published",
                    )
            else:
                _require(
                    ids == list(range(EXPECTED_CONTENTS)),
                    f"{task}: formal mode requires content IDs 0..49",
                )
                # Formal E0/E1 runs consume only data for which the collection
                # gate emitted its canonical aggregate artifacts.  This check
                # intentionally occurs after the exact-ID assertion so a
                # structurally incomplete root reports its primary defect.
                _require(
                    (task_root / "validation_report.json").is_file(),
                    f"{task}: canonical final validation_report.json is missing",
                )
                _require(
                    (task_root / "valid_variants.jsonl").is_file(),
                    f"{task}: canonical final valid_variants.jsonl is missing",
                )
                _require(
                    (task_root / "split_manifests").is_dir(),
                    f"{task}: canonical final split_manifests are missing",
                )
            if self.protocol == E0_E1_PROTOCOL:
                # Preserve the original E0/E1 indexing/validation behavior:
                # validate the complete published set (or the requested smoke
                # subset), and choose the requested split only after indexing.
                selected_ids = (
                    list(requested_content_ids)
                    if requested_content_ids is not None
                    else list(ids)
                )
                if max_trajectories_per_task is not None:
                    selected_ids = selected_ids[:max_trajectories_per_task]
                deep_content_ids = published_ids
            else:
                # Strict holdout indexing never enters an out-of-split content
                # directory, even though the source release contains all 50.
                selected_ids = (
                    list(requested_content_ids)
                    if requested_content_ids is not None
                    else [
                        content_id
                        for content_id in ids
                        if split_for_content(content_id) == split
                    ]
                )
                if max_trajectories_per_task is not None:
                    selected_ids = selected_ids[:max_trajectories_per_task]
                deep_content_ids = selected_ids
            _require(
                bool(selected_ids), f"{task}: no published contents for split {split!r}"
            )
            _validate_task_root_state(
                task_root,
                task=task,
                # The root manifest describes every atomically published
                # content.  A smoke sampling cap is applied only after that
                # root-level consistency check.
                ids=published_ids,
                formal=not self.allow_incomplete,
                # Hash only COMPLETE markers that this split will consume.  In
                # particular, an E2/E3 train/val loader never opens a test
                # content path merely to construct its index.
                deep_content_ids=deep_content_ids,
            )
            for content_id in selected_ids:
                trajectory = _validate_content(
                    contents_root / f"content_{content_id:06d}",
                    task,
                    content_id,
                    active_variants=self.active_variants,
                )
                trajectories.append(trajectory)

        self.trajectories = tuple(trajectories)
        if not self.allow_incomplete and self.protocol == E0_E1_PROTOCOL:
            for task in self.tasks:
                task_trajectories = [
                    trajectory
                    for trajectory in self.trajectories
                    if trajectory.task == task
                ]
                self._assert_canonical_validation_outputs(
                    task,
                    task_trajectories,
                    active_variants=self.active_variants,
                )
        selected = [
            index
            for index, trajectory in enumerate(self.trajectories)
            if trajectory.split == split
        ]
        _require(
            bool(selected), f"no validated physical trajectories in split {split!r}"
        )
        refs: list[FrameRef] = []
        for trajectory_index in selected:
            trajectory = self.trajectories[trajectory_index]
            _require(
                trajectory.frame_count >= self.states_per_trajectory,
                f"{trajectory.task}/content_{trajectory.content_id:06d}: fewer frames than requested states",
            )
            frame_indices = np.linspace(
                0,
                trajectory.frame_count - 1,
                num=self.states_per_trajectory,
                dtype=np.int64,
            )
            _require(
                len(set(int(value) for value in frame_indices))
                == self.states_per_trajectory,
                "sampling repeated a frame",
            )
            for frame_idx_value in frame_indices:
                frame_idx = int(frame_idx_value)
                refs.append(
                    FrameRef(
                        trajectory_index=trajectory_index,
                        key=PhysicalKey(
                            trajectory.task, trajectory.content_id, frame_idx
                        ),
                        trace_idx=int(trajectory.frame_trace_index[frame_idx]),
                    )
                )
        self.frame_refs = tuple(refs)
        _require(
            len({ref.key for ref in self.frame_refs}) == len(self.frame_refs),
            "duplicate physical sample key",
        )

    def _resolve_task_root(self, task: str) -> Path:
        if len(self.tasks) == 1 and self.data_root.name == "paired_random_background":
            return self.data_root
        return self.data_root / task / "paired_random_background"

    def _assert_canonical_validation_outputs(
        self,
        task: str,
        trajectories: Sequence[PairedTrajectory],
        *,
        active_variants: Sequence[str],
    ) -> None:
        task_root = self._resolve_task_root(task)
        report = _read_json(task_root / "validation_report.json")
        _require(report.get("schema_version") == 2, f"{task}: validator schema mismatch")
        try:
            report_dataset_root = Path(str(report["dataset_root"])).expanduser().resolve()
            report_contents_root = Path(str(report["contents_root"])).expanduser().resolve()
        except KeyError as exc:
            raise PairedDataError(f"{task}: validator root provenance is missing") from exc
        _require(
            report_dataset_root == task_root,
            f"{task}: validator dataset root does not match current data",
        )
        _require(
            report_contents_root == task_root / "contents",
            f"{task}: validator contents root does not match current data",
        )
        _require(report.get("valid") is True, f"{task}: canonical validator report is invalid")
        _require(report.get("expected_task") == task, f"{task}: validator task mismatch")
        _require(
            report.get("expected_content_ids") == list(range(EXPECTED_CONTENTS)),
            f"{task}: validator did not cover exact content IDs 0..49",
        )
        _require(
            report.get("expected_style_seeds") == list(STYLE_SEEDS),
            f"{task}: validator style seeds mismatch",
        )
        _require(not report.get("errors"), f"{task}: validator reports aggregate errors")
        summary = report.get("summary")
        _require(isinstance(summary, dict), f"{task}: validator summary is missing")
        _require(summary.get("expected_contents") == 50, f"{task}: validator expected count")
        _require(summary.get("valid_contents") == 50, f"{task}: validator content count")
        _require(summary.get("valid_clean_variants") == 50, f"{task}: validator clean count")
        _require(summary.get("valid_random_variants") == 150, f"{task}: validator random count")
        _require(summary.get("valid_variants") == 200, f"{task}: validator variant count")
        _require(summary.get("aggregate_error_count") == 0, f"{task}: validator error count")
        _require(not report.get("invalid_content_ids"), f"{task}: validator has invalid contents")

        expected_split_counts = {
            split: {
                "clean": {"train": 30, "val": 10, "test": 10}[split],
                "random": 3 * {"train": 30, "val": 10, "test": 10}[split],
            }
            for split in SPLITS
        }
        _require(
            report.get("expected_split_counts") == expected_split_counts,
            f"{task}: validator expected split counts mismatch",
        )
        _require(
            report.get("valid_split_counts") == expected_split_counts,
            f"{task}: validator valid split counts mismatch",
        )
        _require(
            summary.get("split_counts") == expected_split_counts,
            f"{task}: validator summary split counts mismatch",
        )
        trajectory_by_id = {
            trajectory.content_id: trajectory for trajectory in trajectories
        }
        expected_seeds = {
            str(content_id): trajectory.content_seed
            for content_id, trajectory in trajectory_by_id.items()
        }
        observed_seeds = report.get("observed_content_seeds")
        _require(isinstance(observed_seeds, dict), f"{task}: validator seeds are missing")
        _require(
            {str(key): value for key, value in observed_seeds.items()} == expected_seeds,
            f"{task}: validator content seeds mismatch current data",
        )
        report_render_device = report.get("render_device")
        _require(
            isinstance(report_render_device, dict),
            f"{task}: validator render-device provenance is missing",
        )

        expected_rows = {
            (trajectory.content_id, variant, trajectory.split)
            for trajectory in trajectories
            for variant in active_variants
        }
        observed_rows: set[tuple[int, str, str]] = set()
        combined_rows: dict[tuple[int, str, str], dict[str, Any]] = {}
        with (task_root / "valid_variants.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (int(row["content_id"]), str(row["variant"]), str(row["split"]))
                _require(key[1] in VARIANTS, f"{task}: unknown canonical variant {key[1]}")
                if key not in expected_rows:
                    continue
                _require(key not in observed_rows, f"{task}: duplicate canonical manifest row {key}")
                observed_rows.add(key)
                _require(row.get("task") == task, f"{task}: canonical manifest task mismatch")
                _require(
                    row.get("content_seed") == trajectory_by_id[key[0]].content_seed,
                    f"{task}: canonical content seed mismatch",
                )
                expected_clean = f"contents/content_{key[0]:06d}/clean"
                _require(row.get("clean_path") == expected_clean, f"{task}: canonical clean path")
                expected_path = (
                    expected_clean
                    if key[1] == "clean"
                    else f"contents/content_{key[0]:06d}/{key[1]}"
                )
                _require(row.get("path") == expected_path, f"{task}: canonical variant path")
                expected_seed = None if key[1] == "clean" else STYLE_SEEDS[VARIANTS.index(key[1]) - 1]
                _require(row.get("style_seed") == expected_seed, f"{task}: canonical style seed")
                expected_intervention = "none" if key[1] == "clean" else "random_background"
                _require(
                    row.get("intervention") == expected_intervention,
                    f"{task}: canonical intervention mismatch",
                )
                _require(
                    row.get("render_device") == report_render_device,
                    f"{task}: canonical render-device mismatch",
                )
                combined_rows[key] = row
        _require(observed_rows == expected_rows, f"{task}: canonical manifest coverage mismatch")

        for split in SPLITS:
            for kind in ("clean", "random"):
                path = task_root / "split_manifests" / f"{split}_{kind}.jsonl"
                _require(path.is_file(), f"{task}: missing canonical split manifest {path.name}")
                with path.open(encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle]
                expected_count = {"train": 30, "val": 10, "test": 10}[split]
                if kind == "random":
                    expected_count *= 3
                _require(len(rows) == expected_count, f"{task}: {path.name} row count")
                split_rows: set[tuple[int, str, str]] = set()
                for row in rows:
                    _require(row.get("split") == split, f"{task}: {path.name} split leakage")
                    is_clean = row.get("variant") == "clean"
                    _require(is_clean == (kind == "clean"), f"{task}: {path.name} kind leakage")
                    key = (int(row["content_id"]), str(row["variant"]), str(row["split"]))
                    if key not in expected_rows:
                        continue
                    _require(key not in split_rows, f"{task}: duplicate row in {path.name}")
                    split_rows.add(key)
                    _require(
                        row == combined_rows.get(key),
                        f"{task}: {path.name} row differs from combined manifest",
                    )
                expected_split_rows = {
                    key
                    for key in expected_rows
                    if key[2] == split and ((key[1] == "clean") == (kind == "clean"))
                }
                _require(
                    split_rows == expected_split_rows,
                    f"{task}: {path.name} coverage mismatch",
                )

    def __len__(self) -> int:
        return len(self.frame_refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.frame_refs[index]
        trajectory = self.trajectories[ref.trajectory_index]
        frame_idx = ref.key.frame_idx
        _require(
            trajectory.frame_trace_index[frame_idx] == ref.trace_idx,
            f"{ref.key.as_string()}: stale trace index",
        )

        clean_rows: dict[str, np.ndarray] | None = None
        composites: list[torch.Tensor] = []
        encoded_rgb_hashes: dict[str, list[str]] = {
            camera: [] for camera in USED_CAMERAS
        }
        for variant_index, hdf_path in enumerate(trajectory.hdf5_paths):
            rows: dict[str, np.ndarray] = {}
            images: dict[str, np.ndarray] = {}
            with h5py.File(hdf_path, "r") as handle:
                for name in trajectory.non_rgb_leaves:
                    dataset = handle[name]
                    _require(
                        dataset.shape[0] == trajectory.frame_count,
                        f"{ref.key.as_string()}:{name} frame count changed",
                    )
                    rows[name] = np.asarray(dataset[frame_idx])
                for camera in USED_CAMERAS:
                    path = f"observation/{camera}/rgb"
                    dataset = handle[path]
                    _require(
                        dataset.shape == (trajectory.frame_count,),
                        f"{ref.key.as_string()}:{path} shape changed",
                    )
                    encoded = dataset[frame_idx]
                    payload = bytes(encoded).rstrip(b"\x00")
                    encoded_rgb_hashes[camera].append(
                        hashlib.sha256(payload).hexdigest()
                    )
                    images[camera] = _decode_rgb(
                        encoded,
                        label=(
                            f"{ref.key.as_string()}/"
                            f"{trajectory.variant_names[variant_index]}/{camera}"
                        ),
                    )
            if clean_rows is None:
                clean_rows = rows
            else:
                _require(
                    set(rows) == set(clean_rows),
                    f"{ref.key.as_string()}: non-RGB row keys changed",
                )
                for name in rows:
                    _require(
                        _array_value_equal(
                            np.asarray(clean_rows[name]), np.asarray(rows[name])
                        ),
                        f"{ref.key.as_string()}/"
                        f"{trajectory.variant_names[variant_index]}:{name} row differs",
                    )
            composites.append(
                _deployment_composite(
                    images["head_camera"], images["left_camera"], images["right_camera"]
                )
            )

        for camera, hashes in encoded_rgb_hashes.items():
            _require(
                len(set(hashes)) == len(trajectory.variant_names),
                f"{ref.key.as_string()}:{camera} RGB frame is not pairwise different",
            )
        composite_bytes = [tensor.numpy().tobytes() for tensor in composites]
        _require(
            len(set(composite_bytes)) == len(trajectory.variant_names),
            f"{ref.key.as_string()}: deployment composites are not pairwise different",
        )
        # Persist exact bytes-read evidence so independently extracted E2 and
        # E3 caches can prove their visual inputs were identical.  This guards
        # against a source HDF5 changing between the two extraction passes.
        visual_input_sha256 = {
            variant: {
                "encoded_rgb_by_camera": {
                    camera: encoded_rgb_hashes[camera][variant_index]
                    for camera in USED_CAMERAS
                },
                "deployment_composite": hashlib.sha256(
                    composite_bytes[variant_index]
                ).hexdigest(),
            }
            for variant_index, variant in enumerate(trajectory.variant_names)
        }

        state_archives = tuple(_load_npz(path) for path in trajectory.state_trace_paths)
        clean_state = state_archives[0]
        for variant_index in range(1, len(trajectory.variant_names)):
            _require(
                np.array_equal(
                    clean_state["frame_trace_index"],
                    state_archives[variant_index]["frame_trace_index"],
                ),
                f"{ref.key.as_string()}: frame map differs at sampling time",
            )
            for key in set(STATE_KEYS) - {"frame_trace_index"}:
                _require(
                    _array_value_equal(
                        np.asarray(clean_state[key][ref.trace_idx]),
                        np.asarray(state_archives[variant_index][key][ref.trace_idx]),
                    ),
                    f"{ref.key.as_string()}/{trajectory.variant_names[variant_index]}:"
                    f"{key} state row differs",
                )
        _require(clean_rows is not None, "no HDF5 rows were read")
        for trace_key, hdf_name in HDF_TRACE_BINDINGS.items():
            expected = np.asarray(clean_state[trace_key][ref.trace_idx])
            _require(
                np.asarray(clean_rows[hdf_name]).shape == expected.shape
                and np.array_equal(np.asarray(clean_rows[hdf_name]), expected),
                f"{ref.key.as_string()}:{hdf_name} is not aligned to trace_idx={ref.trace_idx}",
            )

        task_state, physical_state = _physical_state_by_name(
            clean_state, ref.trace_idx, trajectory.task_state_layout
        )
        return {
            "physical_key": ref.key.as_string(),
            "task": trajectory.task,
            "content_id": trajectory.content_id,
            "content_seed": trajectory.content_seed,
            "split": trajectory.split,
            "frame_idx": frame_idx,
            "trace_idx": ref.trace_idx,
            "variant_names": trajectory.variant_names,
            "images": torch.stack(composites, dim=0),  # [active variants,3,384,320]
            "visual_input_sha256": visual_input_sha256,
            # Deployment names this 14-D observed joint vector `proprio` after
            # applying the released z-score statistics.  It is identical
            # across all four renderings because all non-RGB HDF5 leaves were
            # checked byte-for-byte above.
            "proprio_raw": torch.as_tensor(
                np.asarray(clean_rows["joint_action/vector"], dtype=np.float32).copy()
            ),
            "task_state_by_name": task_state,
            "physical_state_by_name": physical_state,
        }

    def manifest_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for ref in self.frame_refs:
            trajectory = self.trajectories[ref.trajectory_index]
            for variant, hdf5_path in zip(
                trajectory.variant_names, trajectory.hdf5_paths, strict=True
            ):
                records.append(
                    {
                        "physical_key": ref.key.as_string(),
                        "task": trajectory.task,
                        "content_id": trajectory.content_id,
                        "content_seed": trajectory.content_seed,
                        "split": trajectory.split,
                        "frame_idx": ref.key.frame_idx,
                        "trace_idx": ref.trace_idx,
                        "variant": variant,
                        "hdf5": str(hdf5_path),
                    }
                )
        variants = {str(record["variant"]) for record in records}
        _require(
            variants == set(self.active_variants),
            "experiment manifest variant coverage changed",
        )
        if self.protocol == R3_HOLDOUT_PROTOCOL and self.split in ("train", "val"):
            _require(R3_VARIANT not in variants, "R3 leaked into a train/val manifest")
        return records

    def write_manifests(
        self, output_dir: str | Path, *, stem: str = "paired_frames"
    ) -> dict[str, Path]:
        """Atomically write equivalent JSONL/CSV manifests outside the data root."""
        from .io_utils import atomic_write_text, write_csv

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        records = self.manifest_records()
        json_path = destination / f"{stem}_{self.split}.jsonl"
        csv_path = destination / f"{stem}_{self.split}.csv"
        atomic_write_text(
            json_path,
            "".join(
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                for record in records
            ),
        )
        write_csv(csv_path, records)
        return {"jsonl": json_path, "csv": csv_path}


def collate_paired_frames(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate images while retaining variable-layout named states as mappings."""
    batch = list(samples)
    _require(bool(batch), "cannot collate an empty batch")
    variant_names = tuple(str(value) for value in batch[0]["variant_names"])
    _require(
        bool(variant_names)
        and all(tuple(sample["variant_names"]) == variant_names for sample in batch),
        "cannot collate samples with different active variants",
    )
    return {
        "physical_key": [sample["physical_key"] for sample in batch],
        "task": [sample["task"] for sample in batch],
        "content_id": torch.tensor(
            [sample["content_id"] for sample in batch], dtype=torch.int64
        ),
        "content_seed": torch.tensor(
            [sample["content_seed"] for sample in batch], dtype=torch.int64
        ),
        "split": [sample["split"] for sample in batch],
        "frame_idx": torch.tensor(
            [sample["frame_idx"] for sample in batch], dtype=torch.int64
        ),
        "trace_idx": torch.tensor(
            [sample["trace_idx"] for sample in batch], dtype=torch.int64
        ),
        "variant_names": variant_names,
        "images": torch.stack([sample["images"] for sample in batch], dim=0),
        "proprio_raw": torch.stack([sample["proprio_raw"] for sample in batch], dim=0),
        "task_state_by_name": [sample["task_state_by_name"] for sample in batch],
        "physical_state_by_name": [
            sample["physical_state_by_name"] for sample in batch
        ],
    }


__all__ = [
    "E0_E1_PROTOCOL",
    "E2_PROTOCOL",
    "E3_PROTOCOL",
    "R3_HOLDOUT_PROTOCOL",
    "PairedDataError",
    "PairedFrameDataset",
    "PhysicalKey",
    "TASKS",
    "R3_VARIANT",
    "SEEN_VARIANTS",
    "UNSEEN_TEST_VARIANTS",
    "USED_CAMERAS",
    "VARIANTS",
    "collate_paired_frames",
    "canonical_protocol",
    "compatible_state_vectors",
    "split_for_content",
    "variants_for_protocol",
]
