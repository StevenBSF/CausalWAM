"""Policy-only data protocol constants and fail-closed metadata checks.

This module intentionally owns its constants.  In particular it does not
import the representation-level E0--E3 protocol: Policy v2 trains with four
*scene versions* (each scene still contains the same three cameras), and R3 is
a training positive rather than a held-out evaluation domain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


POLICY_PROTOCOL_ID = "policy_native50hz_four_scene_v1"
POLICY_MAIN_BASE_KIND = "author_release"
POLICY_MAIN_BASE_LINEAGE_KIND = "policy_author_release_base_lineage"
POLICY_MAIN_BASE_LINEAGE_ID = "fastwam_robotwin_50task_clean_random_author_release_v1"
POLICY_MAIN_CONTROLS = ("c0_original", "c1_architecture_only", "c3_ours")
POLICY_STAGE2_SEEDS = (1, 2, 3)
POLICY_C0_HAS_TRAINING_SEED = False
POLICY_VARIANTS = (
    "clean",
    "style_00_seed_0",
    "style_01_seed_1",
    "style_02_seed_2",
)
POLICY_R3_VARIANT = POLICY_VARIANTS[-1]
POLICY_R3_ROLE = "training_positive"
POLICY_VIEW_COUNT = 4

POLICY_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
POLICY_RAW_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
POLICY_CAMERA_COUNT = 3
POLICY_NATIVE_FPS = 50
POLICY_ACTION_STEPS = 32
POLICY_ACTION_DIM = 14
POLICY_TEMPORAL_RESAMPLING = "none"

POLICY_TOKEN_CACHE_SCHEMA = "policy_frozen_token_cache_v1"
POLICY_TOKEN_CACHE_SCHEMA_VERSION = 3
POLICY_ACTION_MANIFEST_SCHEMA = "policy_native_action_manifest_v1"
POLICY_ACTION_MANIFEST_VERSION = 1
POLICY_ACTION_SAMPLE_SCHEMA = "policy_native_action_group_v1"
POLICY_ACTION_SAMPLE_VERSION = 1
POLICY_STATE_BANK_SCHEMA = "policy_paired_state_bank_v1"
POLICY_STATE_BANK_SCHEMA_VERSION = 1
POLICY_STATE_BANK_SAMPLING_ALGORITHM = "sha256_rank_endpoint_safe_v1"
POLICY_STATE_BANK_SAMPLING_VERSION = 1
POLICY_STATE_BANK_SEED = 42
POLICY_STATES_PER_TRAJECTORY = 8
POLICY_STATE_STEPS = POLICY_ACTION_STEPS + 1

PAIRED_SUPERVISION_MODES = ("none", "action", "contrastive")
POLICY_DATA_SPLITS = ("train", "val", "test")
POLICY_TRAIN_SPLITS = ("train", "val")
POLICY_MANIFEST_SPLIT = "all"
POLICY_CONTENTS_PER_TASK_BY_SPLIT = {"train": 30, "val": 10, "test": 10}


class PolicyProtocolError(ValueError):
    """Metadata cannot prove the native Policy v2 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyProtocolError(message)


def canonical_variants(value: Any) -> tuple[str, ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)),
        "variant_names must be a sequence",
    )
    return tuple(str(item) for item in value)


def policy_split_for_content_id(content_id: int) -> str:
    """Return the immutable 30/10/10 split encoded by a content id.

    Policy paired trajectories are numbered continuously within each task:
    0--29 train, 30--39 validation, and 40--49 test.  Treating the split as a
    free manifest label would allow one physical trajectory to leak between
    splits, so consumers derive and verify it here.
    """

    value = int(content_id)
    _require(value == content_id and value >= 0, "content_id must be a non-negative integer")
    offset = 0
    for split, count in POLICY_CONTENTS_PER_TASK_BY_SPLIT.items():
        upper = offset + int(count)
        if offset <= value < upper:
            return split
        offset = upper
    raise PolicyProtocolError(
        f"content_id must be in [0,{offset - 1}] for the fixed 30/10/10 split, got {value}"
    )


def validate_policy_protocol_metadata(
    metadata: Mapping[str, Any],
    *,
    split: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize the invariant Policy v2 metadata.

    Callers may carry additional provenance fields, but every field checked
    here is mandatory.  This makes a 30 Hz cache, a three-scene E2/E3 cache, or
    a cache that treats R3 as holdout impossible to consume accidentally.
    """

    _require(isinstance(metadata, Mapping), "policy protocol metadata is missing")
    protocol_id = str(metadata.get("protocol_id", metadata.get("protocol", "")))
    _require(
        protocol_id == POLICY_PROTOCOL_ID,
        f"protocol_id must be {POLICY_PROTOCOL_ID!r}, got {protocol_id!r}",
    )
    variants = canonical_variants(
        metadata.get("variant_names", metadata.get("active_variants", ()))
    )
    _require(
        variants == POLICY_VARIANTS,
        f"Policy paired data requires ordered C/R1/R2/R3 {POLICY_VARIANTS}, got {variants}",
    )
    _require(
        int(metadata.get("view_count", -1)) == POLICY_VIEW_COUNT,
        f"view_count must be {POLICY_VIEW_COUNT}",
    )
    _require(
        str(metadata.get("r3_role", "")) == POLICY_R3_ROLE,
        f"r3_role must be {POLICY_R3_ROLE!r}",
    )
    _require(
        int(metadata.get("camera_count", -1)) == POLICY_CAMERA_COUNT,
        f"camera_count must be {POLICY_CAMERA_COUNT}",
    )
    camera_names = canonical_variants(metadata.get("camera_names", ()))
    _require(
        camera_names == POLICY_CAMERA_NAMES,
        f"camera_names must be {POLICY_CAMERA_NAMES}",
    )
    _require(
        int(metadata.get("native_fps", -1)) == POLICY_NATIVE_FPS,
        f"native_fps must be {POLICY_NATIVE_FPS}; temporal interpolation is forbidden",
    )
    _require(
        int(metadata.get("action_steps", -1)) == POLICY_ACTION_STEPS,
        f"action_steps must be {POLICY_ACTION_STEPS}",
    )
    _require(
        int(metadata.get("action_dim", -1)) == POLICY_ACTION_DIM,
        f"action_dim must be {POLICY_ACTION_DIM}",
    )
    _require(
        str(metadata.get("temporal_resampling", "")) == POLICY_TEMPORAL_RESAMPLING,
        "temporal_resampling must be 'none'; 30 Hz to 50 Hz interpolation is forbidden",
    )
    _require(
        metadata.get("native_action_targets") is True,
        "native_action_targets=true is required",
    )
    resolved_split = str(metadata.get("split", split or ""))
    if split is not None:
        _require(resolved_split == split, f"paired split must be {split!r}")
    _require(
        resolved_split in (*POLICY_DATA_SPLITS, POLICY_MANIFEST_SPLIT),
        "paired split must be train/val/test, or 'all' for a full manifest",
    )
    return {
        "protocol_id": protocol_id,
        "variant_names": variants,
        "view_count": POLICY_VIEW_COUNT,
        "r3_role": POLICY_R3_ROLE,
        "camera_count": POLICY_CAMERA_COUNT,
        "camera_names": camera_names,
        "native_fps": POLICY_NATIVE_FPS,
        "action_steps": POLICY_ACTION_STEPS,
        "action_dim": POLICY_ACTION_DIM,
        "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
        "native_action_targets": True,
        "split": resolved_split,
    }


__all__ = [
    "PAIRED_SUPERVISION_MODES",
    "POLICY_C0_HAS_TRAINING_SEED",
    "POLICY_ACTION_DIM",
    "POLICY_ACTION_MANIFEST_SCHEMA",
    "POLICY_ACTION_MANIFEST_VERSION",
    "POLICY_ACTION_SAMPLE_SCHEMA",
    "POLICY_ACTION_SAMPLE_VERSION",
    "POLICY_ACTION_STEPS",
    "POLICY_CAMERA_COUNT",
    "POLICY_CAMERA_NAMES",
    "POLICY_DATA_SPLITS",
    "POLICY_CONTENTS_PER_TASK_BY_SPLIT",
    "POLICY_MANIFEST_SPLIT",
    "POLICY_MAIN_BASE_KIND",
    "POLICY_MAIN_BASE_LINEAGE_ID",
    "POLICY_MAIN_BASE_LINEAGE_KIND",
    "POLICY_MAIN_CONTROLS",
    "POLICY_NATIVE_FPS",
    "POLICY_PROTOCOL_ID",
    "POLICY_RAW_CAMERA_NAMES",
    "POLICY_R3_ROLE",
    "POLICY_R3_VARIANT",
    "POLICY_TEMPORAL_RESAMPLING",
    "POLICY_STATE_BANK_SAMPLING_ALGORITHM",
    "POLICY_STATE_BANK_SAMPLING_VERSION",
    "POLICY_STATE_BANK_SCHEMA",
    "POLICY_STATE_BANK_SCHEMA_VERSION",
    "POLICY_STATE_BANK_SEED",
    "POLICY_STATE_STEPS",
    "POLICY_STATES_PER_TRAJECTORY",
    "POLICY_STAGE2_SEEDS",
    "POLICY_TOKEN_CACHE_SCHEMA",
    "POLICY_TOKEN_CACHE_SCHEMA_VERSION",
    "POLICY_TRAIN_SPLITS",
    "POLICY_VARIANTS",
    "POLICY_VIEW_COUNT",
    "PolicyProtocolError",
    "canonical_variants",
    "policy_split_for_content_id",
    "validate_policy_protocol_metadata",
]
