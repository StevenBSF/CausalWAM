#!/usr/bin/env python3
"""Task-specific state and success adapters for strict paired collection.

The module intentionally has no RoboTwin/SAPIEN imports.  Collection passes
live task objects to the capture helpers, while the validator only uses the
pure NumPy/JSON helpers at the bottom of the file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SUPPORTED_TASKS = (
    "grab_roller",
    "place_a2b_left",
    "open_microwave",
    "move_stapler_pad",
)

SPLIT_COUNTS = {"train": 30, "val": 10, "test": 10}
TOTAL_CONTENTS = sum(SPLIT_COUNTS.values())

RIGID_STATE_FIELDS = (
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "linear_vx",
    "linear_vy",
    "linear_vz",
    "angular_vx",
    "angular_vy",
    "angular_vz",
)

PLACE_OBJECTS = frozenset(
    {
        "047_mouse",
        "048_stapler",
        "050_bell",
        "057_toycar",
        "073_rubikscube",
        "075_bread",
        "077_phone",
        "081_playingcards",
        "086_woodenblock",
        "107_soap",
        "112_tea-box",
        "113_coffee-box",
    }
)

PAD_COLORS = {
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Black": (0.0, 0.0, 0.0),
    "Gray": (0.5, 0.5, 0.5),
}


class TaskAdapterError(RuntimeError):
    """A supported task does not expose the state required for strict pairing."""


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_for_content(content_id: int) -> str:
    if 0 <= content_id < SPLIT_COUNTS["train"]:
        return "train"
    if content_id < SPLIT_COUNTS["train"] + SPLIT_COUNTS["val"]:
        return "val"
    if content_id < TOTAL_CONTENTS:
        return "test"
    raise TaskAdapterError(
        f"content_id {content_id} is outside the fixed 0..{TOTAL_CONTENTS - 1} split"
    )


def _float_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise TaskAdapterError(f"cannot record numeric task field {name}") from exc
    if not vector.size or not np.all(np.isfinite(vector)):
        raise TaskAdapterError(f"task field {name} is empty or non-finite")
    return vector


def _velocity(actor: Any, *, angular: bool, label: str) -> np.ndarray:
    wrapper = actor
    entity = getattr(wrapper, "actor", wrapper)
    getter_names = ("get_angular_velocity",) if angular else (
        "get_linear_velocity",
        "get_velocity",
    )
    attribute_names = ("angular_velocity",) if angular else (
        "linear_velocity",
        "velocity",
    )
    sources = [wrapper, entity]
    get_components = getattr(entity, "get_components", None)
    if callable(get_components):
        sources.extend(list(get_components()))
    for source in sources:
        for getter_name in getter_names:
            getter = getattr(source, getter_name, None)
            if callable(getter):
                value = _float_vector(getter(), name=f"{label}.{getter_name}")
                if value.size == 3:
                    return value
        for attribute_name in attribute_names:
            if hasattr(source, attribute_name):
                value = _float_vector(
                    getattr(source, attribute_name), name=f"{label}.{attribute_name}"
                )
                if value.size == 3:
                    return value
    kind = "angular" if angular else "linear"
    raise TaskAdapterError(f"{label} exposes no three-vector {kind} velocity")


def _rigid_state(actor: Any, *, label: str) -> np.ndarray:
    pose = actor.get_pose()
    state = np.concatenate(
        (
            _float_vector(pose.p, name=f"{label}.position"),
            _float_vector(pose.q, name=f"{label}.quaternion_wxyz"),
            _velocity(actor, angular=False, label=label),
            _velocity(actor, angular=True, label=label),
        )
    )
    if state.shape != (len(RIGID_STATE_FIELDS),):
        raise TaskAdapterError(f"{label} rigid state has unexpected shape {state.shape}")
    return state


def _rigid_layout(prefix: str) -> list[str]:
    return [f"{prefix}_{field}" for field in RIGID_STATE_FIELDS]


def capture_task_state(env: Any, task_name: str) -> tuple[np.ndarray, list[str]]:
    if task_name == "grab_roller":
        return _rigid_state(env.roller, label="roller"), _rigid_layout("roller")
    if task_name == "place_a2b_left":
        state = np.concatenate(
            (
                _rigid_state(env.object, label="object_A"),
                _rigid_state(env.target_object, label="target_B"),
            )
        )
        return state, [*_rigid_layout("object_A"), *_rigid_layout("target_B")]
    if task_name == "move_stapler_pad":
        state = np.concatenate(
            (
                _rigid_state(env.stapler, label="stapler"),
                _rigid_state(env.pad, label="pad"),
            )
        )
        return state, [*_rigid_layout("stapler"), *_rigid_layout("pad")]
    if task_name == "open_microwave":
        pose = env.microwave.get_pose()
        root = np.concatenate(
            (
                _float_vector(pose.p, name="microwave.position"),
                _float_vector(pose.q, name="microwave.quaternion_wxyz"),
            )
        )
        qpos = _float_vector(env.microwave.get_qpos(), name="microwave.qpos")
        qvel = _float_vector(env.microwave.get_qvel(), name="microwave.qvel")
        if qpos.shape != qvel.shape:
            raise TaskAdapterError(
                f"microwave qpos/qvel shapes differ: {qpos.shape} versus {qvel.shape}"
            )
        layout = [
            "microwave_root_px",
            "microwave_root_py",
            "microwave_root_pz",
            "microwave_root_qw",
            "microwave_root_qx",
            "microwave_root_qy",
            "microwave_root_qz",
            *[f"microwave_qpos_{index}" for index in range(qpos.size)],
            *[f"microwave_qvel_{index}" for index in range(qvel.size)],
        ]
        return np.concatenate((root, qpos, qvel)), layout
    raise TaskAdapterError(f"unsupported strict-pair task {task_name!r}")


def capture_task_identity(env: Any, task_name: str) -> dict[str, Any]:
    if task_name == "grab_roller":
        return {"model_name": "roller", "model_id": int(env.model_id)}
    if task_name == "place_a2b_left":
        return {
            "object_A_model_name": str(env.selected_modelname_A),
            "object_A_model_id": int(env.selected_model_id_A),
            "target_B_model_name": str(env.selected_modelname_B),
            "target_B_model_id": int(env.selected_model_id_B),
        }
    if task_name == "open_microwave":
        return {"model_name": str(env.model_name), "model_id": int(env.model_id)}
    if task_name == "move_stapler_pad":
        return {
            "model_name": "048_stapler",
            "model_id": int(env.stapler_id),
            "pad_color_name": str(env.color_name),
            "pad_color_value": [float(value) for value in env.color_value],
        }
    raise TaskAdapterError(f"unsupported strict-pair task {task_name!r}")


def capture_success_spec(env: Any, task_name: str) -> dict[str, Any]:
    if task_name == "grab_roller":
        return {"type": task_name, "minimum_final_z": 0.8}
    if task_name == "place_a2b_left":
        return {
            "type": task_name,
            "distance_min": 0.08,
            "distance_max": 0.2,
            "maximum_abs_y_difference": 0.05,
            "require_both_grippers_open": True,
        }
    if task_name == "move_stapler_pad":
        return {
            "type": task_name,
            "position_abs_eps": [0.02, 0.02, 0.01],
            "maximum_abs_quaternion_spread": 0.02,
            "require_both_grippers_open": True,
        }
    if task_name == "open_microwave":
        limits = np.asarray(env.microwave.get_qlimits(), dtype=np.float64)
        if limits.ndim != 2 or limits.shape[0] < 1 or limits.shape[1] != 2:
            raise TaskAdapterError(f"microwave joint limits have invalid shape {limits.shape}")
        upper = float(limits[0, 1])
        if not np.isfinite(upper) or upper <= 0:
            raise TaskAdapterError(f"microwave upper joint limit is invalid: {upper}")
        return {
            "type": task_name,
            "joint_index": 0,
            "joint_upper_limit": upper,
            "success_ratio": 0.6,
        }
    raise TaskAdapterError(f"unsupported strict-pair task {task_name!r}")


def validate_task_identity(task_name: str, identity: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(identity, Mapping):
        return ["task_identity must be a mapping"]
    if task_name == "grab_roller":
        if set(identity) != {"model_name", "model_id"}:
            errors.append("grab_roller identity keys are not canonical")
        if identity.get("model_name") != "roller" or identity.get("model_id") not in (0, 2):
            errors.append("grab_roller identity value is invalid")
    elif task_name == "place_a2b_left":
        expected = {
            "object_A_model_name",
            "object_A_model_id",
            "target_B_model_name",
            "target_B_model_id",
        }
        if set(identity) != expected:
            errors.append("place_a2b_left identity keys are not canonical")
        if identity.get("object_A_model_name") not in PLACE_OBJECTS:
            errors.append("place_a2b_left object A model name is invalid")
        if identity.get("target_B_model_name") not in PLACE_OBJECTS:
            errors.append("place_a2b_left target B model name is invalid")
        if identity.get("object_A_model_name") == identity.get("target_B_model_name"):
            errors.append("place_a2b_left A/B model names must differ")
        for key in ("object_A_model_id", "target_B_model_id"):
            if isinstance(identity.get(key), bool) or not isinstance(identity.get(key), int) or identity[key] < 0:
                errors.append(f"place_a2b_left {key} is invalid")
    elif task_name == "open_microwave":
        if set(identity) != {"model_name", "model_id"}:
            errors.append("open_microwave identity keys are not canonical")
        if identity.get("model_name") != "044_microwave" or identity.get("model_id") not in (0, 1):
            errors.append("open_microwave identity value is invalid")
    elif task_name == "move_stapler_pad":
        expected = {"model_name", "model_id", "pad_color_name", "pad_color_value"}
        if set(identity) != expected:
            errors.append("move_stapler_pad identity keys are not canonical")
        if identity.get("model_name") != "048_stapler" or identity.get("model_id") not in range(7):
            errors.append("move_stapler_pad stapler identity is invalid")
        color_name = identity.get("pad_color_name")
        color_value = identity.get("pad_color_value")
        if color_name not in PAD_COLORS or color_value != list(PAD_COLORS.get(color_name, ())):
            errors.append("move_stapler_pad pad color identity is invalid")
    else:
        errors.append(f"unsupported task {task_name!r}")
    return errors


def validate_success_spec(task_name: str, spec: Any) -> list[str]:
    if not isinstance(spec, Mapping) or spec.get("type") != task_name:
        return ["task_success_spec type is invalid"]
    expected = capture_success_spec_schema(task_name)
    if set(spec) != expected:
        return [f"task_success_spec keys differ from canonical {sorted(expected)}"]
    errors: list[str] = []
    if task_name == "grab_roller" and spec.get("minimum_final_z") != 0.8:
        errors.append("grab_roller minimum_final_z must be 0.8")
    elif task_name == "place_a2b_left":
        if (
            spec.get("distance_min") != 0.08
            or spec.get("distance_max") != 0.2
            or spec.get("maximum_abs_y_difference") != 0.05
            or spec.get("require_both_grippers_open") is not True
        ):
            errors.append("place_a2b_left success constants are invalid")
    elif task_name == "move_stapler_pad":
        if (
            spec.get("position_abs_eps") != [0.02, 0.02, 0.01]
            or spec.get("maximum_abs_quaternion_spread") != 0.02
            or spec.get("require_both_grippers_open") is not True
        ):
            errors.append("move_stapler_pad success constants are invalid")
    elif task_name == "open_microwave":
        upper = spec.get("joint_upper_limit")
        if (
            spec.get("joint_index") != 0
            or spec.get("success_ratio") != 0.6
            or isinstance(upper, bool)
            or not isinstance(upper, (int, float))
            or not np.isfinite(upper)
            or upper <= 0
        ):
            errors.append("open_microwave success constants are invalid")
    return errors


def capture_success_spec_schema(task_name: str) -> set[str]:
    schemas = {
        "grab_roller": {"type", "minimum_final_z"},
        "place_a2b_left": {
            "type",
            "distance_min",
            "distance_max",
            "maximum_abs_y_difference",
            "require_both_grippers_open",
        },
        "move_stapler_pad": {
            "type",
            "position_abs_eps",
            "maximum_abs_quaternion_spread",
            "require_both_grippers_open",
        },
        "open_microwave": {
            "type",
            "joint_index",
            "joint_upper_limit",
            "success_ratio",
        },
    }
    try:
        return schemas[task_name]
    except KeyError as exc:
        raise TaskAdapterError(f"unsupported strict-pair task {task_name!r}") from exc


def derive_success(
    task_name: str,
    task_state: np.ndarray,
    layout: Sequence[str],
    *,
    left_gripper_open: bool | None,
    right_gripper_open: bool | None,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    values = np.asarray(task_state, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(layout):
        raise TaskAdapterError(
            f"task_state shape {values.shape} does not match layout width {len(layout)}"
        )
    if not np.all(np.isfinite(values)) or len(set(layout)) != len(layout):
        raise TaskAdapterError("task_state/layout is non-finite or contains duplicate fields")
    indices = {name: index for index, name in enumerate(layout)}
    final = values[-1]

    def vector(prefix: str, names: Sequence[str]) -> np.ndarray:
        try:
            return np.asarray([final[indices[f"{prefix}_{name}"]] for name in names])
        except KeyError as exc:
            raise TaskAdapterError(f"task_state layout is missing {exc.args[0]}") from exc

    if task_name == "grab_roller":
        z = float(vector("roller", ("pz",))[0])
        success = bool(
            z > float(spec["minimum_final_z"])
            and left_gripper_open is False
            and right_gripper_open is False
        )
        return {"derived_success": success, "roller_final_z": z}

    if task_name == "place_a2b_left":
        object_pos = vector("object_A", ("px", "py", "pz"))
        target_pos = vector("target_B", ("px", "py", "pz"))
        distance = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        abs_y = float(abs(object_pos[1] - target_pos[1]))
        success = bool(
            distance < float(spec["distance_max"])
            and distance > float(spec["distance_min"])
            and object_pos[0] < target_pos[0]
            and abs_y < float(spec["maximum_abs_y_difference"])
            and left_gripper_open is True
            and right_gripper_open is True
        )
        return {
            "derived_success": success,
            "final_xy_distance": distance,
            "final_abs_y_difference": abs_y,
            "object_A_left_of_target_B": bool(object_pos[0] < target_pos[0]),
        }

    if task_name == "move_stapler_pad":
        stapler_pos = vector("stapler", ("px", "py", "pz"))
        pad_pos = vector("pad", ("px", "py", "pz"))
        stapler_q = np.abs(vector("stapler", ("qw", "qx", "qy", "qz")))
        position_abs_error = np.abs(stapler_pos - pad_pos)
        quaternion_spread = float(stapler_q.max() - stapler_q.min())
        success = bool(
            np.all(position_abs_error < np.asarray(spec["position_abs_eps"], dtype=np.float64))
            and quaternion_spread < float(spec["maximum_abs_quaternion_spread"])
            and left_gripper_open is True
            and right_gripper_open is True
        )
        return {
            "derived_success": success,
            "final_position_abs_error": position_abs_error.tolist(),
            "final_abs_quaternion_spread": quaternion_spread,
        }

    if task_name == "open_microwave":
        joint_index = int(spec["joint_index"])
        key = f"microwave_qpos_{joint_index}"
        if key not in indices:
            raise TaskAdapterError(f"task_state layout is missing {key}")
        qpos = float(final[indices[key]])
        threshold = float(spec["joint_upper_limit"]) * float(spec["success_ratio"])
        return {
            "derived_success": bool(qpos >= threshold),
            "final_joint_qpos": qpos,
            "required_joint_qpos": threshold,
        }
    raise TaskAdapterError(f"unsupported strict-pair task {task_name!r}")
