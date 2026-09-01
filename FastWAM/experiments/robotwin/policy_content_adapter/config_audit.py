"""Fail-closed Policy Protocol v2 config and fairness audits.

Current configs describe release-base C0/C1/C3 adaptation.  The author
release lineage replaces (and must never impersonate) a local Stage-1
completion manifest.
The four old Clean/Clean+Aug/CR/CR+Aug YAMLs remain readable historical
artifacts, but they are classified as legacy schema v2 and can never satisfy a
current execution-ready audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .protocol import (
    POLICY_ACTION_DIM,
    POLICY_ACTION_STEPS,
    POLICY_CAMERA_COUNT,
    POLICY_CAMERA_NAMES,
    POLICY_NATIVE_FPS,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_TEMPORAL_RESAMPLING,
    POLICY_VARIANTS,
    POLICY_VIEW_COUNT,
)
from .p_mode_selection import (
    DEV_EPISODES_PER_CELL,
    PModeSelectionError,
    canonical_sha256,
    formal_config_protocol_projection,
    validate_formal_protocol_lock_manifest_payload,
    validate_seed_bank_descriptor,
    validate_selection_manifest_payload,
)
from .release_lineage import ReleaseLineageError, verify_author_release_lineage


SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSION = 2
CONFIG_KIND = "policy_content_adapter_run"
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
FORMAL_SEEDS = (1, 2, 3)
LEGACY_STAGE1_SEED_PLAN_FILENAME = "stage1_clean_random_base_seeds012_legacy.yaml"
LEGACY_STAGE1_SEED_PLAN_SHA256 = "bc64d65922e44068b772546e0613c23e726d479f85faa394a580d7a2746c8f9c"
FORMAL_MATRIX_ID = "three_task_author_release_c1_c3_v3"
FORMAL_FILENAMES = (
    "formal_c1_architecture_only.yaml",
    "formal_c3_ours.yaml",
)
LEGACY_FORMAL_FILENAMES = (
    "formal_clean.yaml",
    "formal_clean_aug.yaml",
    "formal_clean_random.yaml",
    "formal_clean_random_aug.yaml",
)
SUPPORTED_CONTROLS = {
    "p_v1",
    "p_v2",
    "c0_original",
    "c1_architecture_only",
    "c2_naive_aug",
    "c3_ours",
}
POLICY_CONTROLS = SUPPORTED_CONTROLS - {"c0_original"}
CONTRASTIVE_CONTROLS = {"p_v1", "p_v2", "c3_ours"}
PAIRED_ACTION_CONTROLS = {"c2_naive_aug"}
FORMAL_ROLES = {
    "c1_architecture_only": "architecture_only",
    "c2_naive_aug": "naive_augmentation",
    "c3_ours": "ours",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_PREFIXES = ("__REQUIRED_", "__SELECT_")

# Canonical author-release identities.  These are duplicated in the immutable
# lineage manifest on purpose: changing a manifest and its declared SHA cannot
# silently redirect the formal experiment to another base.
AUTHOR_RELEASE_CHECKPOINT_SHA256 = "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63"
AUTHOR_RELEASE_DATASET_STATS_SHA256 = "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095"
AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256 = "15f1d60e6f662f047385069ec1afe4715f69aeb3137419f9b4f37a811ec55126"
AUTHOR_RELEASE_BASE_MANIFEST_SHA256 = "d90e6d545c04c28e9e73b6b8a9356ec5e9320be4be6f6b7e3b69237a3f38cefc"
FORMAL_RELEASE_RECIPE_AMENDMENT_SHA256 = "d35e1ab6d9825ae2533b31aaa5d3dbf08e09959b32c55d16e2e1f08e0f3d6f45"
PV2_FOLLOWUP_STAGE = "mechanism_followup"
PV2_FOLLOWUP_ROLE = "post_hoc_actiondit_mechanism"
PV2_FOLLOWUP_PROTOCOL_KIND = "policy_pv2_actiondit_followup_protocol"
PV2_FOLLOWUP_PROTOCOL_SCHEMA_VERSION = 1

# Kept so old result readers importing this name do not break.
KNOWN_SMOKE_IDENTITIES = {
    "base_checkpoint_sha256": "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63",
    "dataset_stats_sha256": "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095",
    "head_init_sha256": "e42c5af1b50023c8cea1a17c8b9269038518c83533e94c54cf2a939092f6ae97",
    "paired_cache_sha256": "91f02fcc3490e3c64904585d89b64cf26684f26a1e6374ee7fc1f184b98b6026",
}


class ConfigAuditError(ValueError):
    """A config cannot prove the requested Policy v2 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigAuditError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _required_keys(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in value]
    _require(not missing, f"{label} is missing required keys: {missing}")


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIXES)


def find_placeholders(value: Any, prefix: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if _is_placeholder(value):
        found.append(prefix or "<root>")
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_placeholders(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(find_placeholders(item, f"{prefix}[{index}]"))
    return tuple(found)


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"config does not exist: {resolved}")
    raw = OmegaConf.load(resolved)
    value = OmegaConf.to_container(raw, resolve=True)
    _require(isinstance(value, dict), f"config root must be a mapping: {resolved}")
    return value


def _validate_sha(value: Any, label: str, *, allow_null: bool = False) -> None:
    if value is None and allow_null:
        return
    if _is_placeholder(value):
        return
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase 64-character SHA-256 or a placeholder",
    )


def _expected_paired_mode(control: str) -> str:
    # C1 deliberately consumes the exact same paired cache/batch sequence as
    # C3.  Its lambda is zero, so the only causal difference is whether the
    # paired contrastive objective contributes a gradient.
    if control in CONTRASTIVE_CONTROLS or control == "c1_architecture_only":
        return "contrastive"
    if control in PAIRED_ACTION_CONTROLS:
        return "action"
    return "none"


def _contrastive_gradient_enabled(config: Mapping[str, Any], control: str) -> bool:
    if control == "c3_ours":
        return True
    if control in {"p_v1", "p_v2"}:
        return config.get("selection_role") == "engineering_method_smoke"
    return False


def _is_c1_c3_engineering_smoke(config: Mapping[str, Any]) -> bool:
    """Return whether this is the pre-selection, non-scientific C1/C3 gate.

    The gate is allowed to pin a provisional P-v1 runtime solely to prove that
    both treatment branches can train from the release checkpoint.  It must
    not bind (or impersonate) the later dev-selected P-mode manifest.
    """

    return (
        not bool(config.get("formal", False))
        and str(config.get("stage", "")) == "smoke"
        and str(config.get("control", ""))
        in {"c1_architecture_only", "c3_ours"}
    )


def _is_pv2_actiondit_followup(config: Mapping[str, Any]) -> bool:
    """Return whether a run belongs to the disclosed post-hoc P-v2 study."""

    return (
        not bool(config.get("formal", False))
        and str(config.get("stage", "")) == PV2_FOLLOWUP_STAGE
        and str(config.get("study_role", "")) == PV2_FOLLOWUP_ROLE
        and str(config.get("control", ""))
        in {"c1_architecture_only", "c3_ours"}
    )


def _validate_execution(config: Mapping[str, Any], control: str, formal: bool) -> None:
    execution = _mapping(config.get("execution"), "execution")
    _required_keys(
        execution,
        ("runner", "runnable", "fail_closed", "long_formal_training"),
        "execution",
    )
    for key in ("runnable", "fail_closed", "long_formal_training"):
        _require(isinstance(execution[key], bool), f"execution.{key} must be boolean")
    expected_runner = "author_release_reference" if control == "c0_original" else "policy_content_adapter"
    _require(execution["runner"] == expected_runner, f"{control} requires runner={expected_runner}")
    _require(
        execution["long_formal_training"] is formal,
        "execution.long_formal_training disagrees with formal",
    )
    if execution["fail_closed"]:
        _require(bool(execution.get("blocked_reason")), "fail-closed config requires blocked_reason")


def _validate_artifacts(config: Mapping[str, Any], control: str) -> None:
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    _required_keys(
        artifacts,
        (
            "base_checkpoint_sha256",
            "dataset_stats_sha256",
            "official_task_manifest_sha256",
            "base_lineage_manifest_sha256",
            "release_paired_binding_manifest_sha256",
            "head_init_sha256",
            "paired_action_manifest_sha256",
            "paired_state_bank_sha256",
            "paired_text_cache_sha256",
            "paired_cache_sha256",
            "p_mode_selection_manifest_sha256",
            "simulator_seed_bank_manifest_sha256",
        ),
        "artifacts",
    )
    for key in (
        "base_checkpoint_sha256",
        "dataset_stats_sha256",
        "official_task_manifest_sha256",
    ):
        _validate_sha(artifacts[key], f"artifacts.{key}")
    _validate_sha(artifacts["base_lineage_manifest_sha256"], "artifacts.base_lineage_manifest_sha256")
    if _is_pv2_actiondit_followup(config):
        _required_keys(
            artifacts,
            (
                "mechanism_protocol_manifest_sha256",
                "action_dit_initialization_audit_sha256",
            ),
            "artifacts",
        )
        _validate_sha(
            artifacts["mechanism_protocol_manifest_sha256"],
            "artifacts.mechanism_protocol_manifest_sha256",
        )
        _validate_sha(
            artifacts["action_dit_initialization_audit_sha256"],
            "artifacts.action_dit_initialization_audit_sha256",
        )
    _validate_sha(
        artifacts["release_paired_binding_manifest_sha256"],
        "artifacts.release_paired_binding_manifest_sha256",
        allow_null=control == "c0_original",
    )
    if control == "c0_original":
        _require(
            artifacts["release_paired_binding_manifest_sha256"] is None,
            "C0 must not bind Stage-2 paired data",
        )
    primary_formal = bool(config.get("formal")) and control in {
        "c1_architecture_only",
        "c3_ours",
    }
    if primary_formal:
        _require(
            "formal_recipe_amendment_manifest_sha256" in artifacts,
            "formal config requires artifacts.formal_recipe_amendment_manifest_sha256",
        )
        _validate_sha(
            artifacts["formal_recipe_amendment_manifest_sha256"],
            "artifacts.formal_recipe_amendment_manifest_sha256",
        )
        _require(
            artifacts["formal_recipe_amendment_manifest_sha256"]
            == FORMAL_RELEASE_RECIPE_AMENDMENT_SHA256,
            "formal recipe amendment SHA differs from the locked disclosure",
        )
        _require(
            "formal_protocol_lock_manifest_sha256" in artifacts,
            "formal config requires artifacts.formal_protocol_lock_manifest_sha256",
        )
        _validate_sha(
            artifacts["formal_protocol_lock_manifest_sha256"],
            "artifacts.formal_protocol_lock_manifest_sha256",
        )
    else:
        _require(
            artifacts.get("formal_protocol_lock_manifest_sha256") is None,
            "non-formal config must not bind a formal protocol lock",
        )
    _require(
        artifacts["base_checkpoint_sha256"] == AUTHOR_RELEASE_CHECKPOINT_SHA256,
        "base checkpoint must be the locked author release",
    )
    _require(
        artifacts["dataset_stats_sha256"] == AUTHOR_RELEASE_DATASET_STATS_SHA256,
        "dataset stats must be the locked author release stats",
    )
    _require(
        artifacts["official_task_manifest_sha256"] == AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
        "official task manifest must be the locked release partition",
    )
    _require(
        artifacts["base_lineage_manifest_sha256"] == AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        "base lineage manifest is not the locked author release lineage",
    )
    _validate_sha(artifacts["head_init_sha256"], "artifacts.head_init_sha256", allow_null=True)
    c1_c3_engineering_smoke = _is_c1_c3_engineering_smoke(config)
    _validate_sha(
        artifacts["p_mode_selection_manifest_sha256"],
        "artifacts.p_mode_selection_manifest_sha256",
        allow_null=(
            control in {"p_v1", "p_v2", "c0_original"}
            or c1_c3_engineering_smoke
        ),
    )
    _validate_sha(
        artifacts["simulator_seed_bank_manifest_sha256"],
        "artifacts.simulator_seed_bank_manifest_sha256",
    )
    if control in {"p_v1", "p_v2", "c0_original"} or c1_c3_engineering_smoke:
        _require(
            artifacts["p_mode_selection_manifest_sha256"] is None,
            f"{control} engineering/pre-selection run must not bind a P-mode selection SHA",
        )
    mode = _expected_paired_mode(control)
    _validate_sha(
        artifacts["paired_action_manifest_sha256"],
        "artifacts.paired_action_manifest_sha256",
        allow_null=mode == "none",
    )
    _validate_sha(
        artifacts["paired_state_bank_sha256"],
        "artifacts.paired_state_bank_sha256",
        allow_null=mode == "none",
    )
    _validate_sha(
        artifacts["paired_text_cache_sha256"],
        "artifacts.paired_text_cache_sha256",
        allow_null=mode == "none",
    )
    _validate_sha(
        artifacts["paired_cache_sha256"],
        "artifacts.paired_cache_sha256",
        allow_null=mode != "contrastive",
    )
    if mode == "none":
        _require(artifacts["paired_action_manifest_sha256"] is None, "unpaired control binds paired data")
        _require(artifacts["paired_state_bank_sha256"] is None, "unpaired control binds paired state bank")
        _require(artifacts["paired_text_cache_sha256"] is None, "unpaired control binds paired text cache")
        _require(artifacts["paired_cache_sha256"] is None, "unpaired control binds paired cache")
    elif mode == "action":
        _require(artifacts["paired_cache_sha256"] is None, "C2 must not consume a frozen contrastive cache")


def _validate_policy(config: Mapping[str, Any], control: str) -> None:
    policy = _mapping(config.get("policy"), "policy")
    architecture = _mapping(config.get("architecture"), "architecture")
    enabled = control in POLICY_CONTROLS
    _require(policy.get("enabled") is enabled, f"policy.enabled must be {enabled} for {control}")
    _require(bool(architecture.get("content_head")) is enabled, "content_head disagrees with policy")
    _require(bool(architecture.get("gated_action_adapter")) is enabled, "GCA disagrees with policy")
    freeze = _mapping(policy.get("freeze"), "policy.freeze")
    _required_keys(
        freeze,
        ("vae", "video_backbone", "action_dit", "content_head", "action_adapter"),
        "policy.freeze",
    )
    _require(freeze["vae"] is True and freeze["video_backbone"] is True, "VAE/Video must remain frozen")
    if not enabled:
        _require(policy.get("adapter_count") == 0, "C0 must have zero adapters")
        _require(policy.get("head_init") is None, "C0 must not initialize a head")
        return
    expected = {
        "content_layer": 16,
        "input_token_count": 120,
        "input_dim": 3072,
        "queries": 8,
        "content_dim": 384,
        "attention_heads": 8,
        "adapter_count": 1,
        "action_hidden_dim": 1024,
        "gate_init_exact": 0.0,
    }
    for key, expected_value in expected.items():
        _require(policy.get(key) == expected_value, f"policy.{key} must be {expected_value!r}")
    for key in ("adapter_init_seed", "head_init_seed"):
        value = policy.get(key)
        _require(
            _is_placeholder(value)
            or (isinstance(value, int) and not isinstance(value, bool) and value >= 0),
            f"policy.{key} must be a non-negative integer or placeholder",
        )
    init_mode = str(policy.get("head_init_mode", ""))
    _require(init_mode in {"random", "pretrained"}, "policy.head_init_mode must be random or pretrained")
    if init_mode == "random":
        _require(policy.get("head_init") is None, "random Head initialization must not load a checkpoint")
        _require(config["artifacts"]["head_init_sha256"] is None, "random Head must not bind pretrained SHA")
    else:
        _require(bool(policy.get("head_init")), "pretrained Head requires policy.head_init")
        _validate_sha(config["artifacts"]["head_init_sha256"], "artifacts.head_init_sha256")
    _require(freeze["content_head"] is False and freeze["action_adapter"] is False, "Head/GCA must train")
    regime = policy.get("regime")
    _require(regime in {"p_v1", "p_v2"} or _is_placeholder(regime), "invalid policy.regime")
    if regime == "p_v1":
        _require(freeze["action_dit"] is True, "P-v1 must freeze ActionDiT")
    elif regime == "p_v2":
        _require(freeze["action_dit"] is False, "P-v2 must train ActionDiT")
    else:
        _require(_is_placeholder(freeze["action_dit"]), "unselected regime needs freeze placeholder")
    _require(architecture.get("adapter_injection") == "action_encoder_output", "GCA injection changed")
    _require(
        architecture.get("adapter_residual") == "Xa_plus_tanh_gate_cross_attention",
        "GCA residual changed",
    )
    _require(architecture.get("mean_pool_on_policy_path") is False, "policy path must keep eight queries")


def _validate_official(config: Mapping[str, Any], formal: bool) -> None:
    official = _mapping(config.get("official"), "official")
    _required_keys(
        official,
        (
            "enabled",
            "dataset_root",
            "dataset_stats",
            "canonical_task_manifest",
            "selection_mode",
            "expected_clean_per_task",
            "expected_random_per_task",
            "expected_total_per_task",
            "strict_task_subset",
            "sampling_mode",
            "task_balanced",
            "balanced_tasks",
            "text_cache_dir",
            "on_the_fly_text_smoke",
            "action_supervision",
            "domain_label",
            "domain_verified",
            "domain_evidence",
        ),
        "official",
    )
    _require(official["enabled"] is True and official["action_supervision"] is True, "official action stream required")
    _require(official["selection_mode"] == "full_550_per_task", "Policy v2 requires full_550_per_task")
    _require(official["expected_clean_per_task"] == 50, "official Clean count must be 50/task")
    _require(official["expected_random_per_task"] == 500, "official Random count must be 500/task")
    _require(official["expected_total_per_task"] == 550, "official total must be 550/task")
    _require(official["strict_task_subset"] is True, "official subset must be fail closed")
    _require(official["task_balanced"] is True and official["balanced_tasks"] is True, "official tasks must balance")
    _require(official["sampling_mode"] in {"episode_anchor", "all_frames"}, "invalid official sampling_mode")
    if formal:
        _require(official["sampling_mode"] == "all_frames", "formal official stream must use all_frames")
    _require(official["domain_label"] == "official_clean_plus_random", "official domain label changed")
    _require(isinstance(official["domain_verified"], bool) or _is_placeholder(official["domain_verified"]), "invalid domain_verified")
    if official["on_the_fly_text_smoke"]:
        _require(official["text_cache_dir"] is None, "on-the-fly smoke must not claim cache")
        _require(config["training"]["num_workers"] == 0, "on-the-fly text requires num_workers=0")
    else:
        _require(bool(official["text_cache_dir"]), "precomputed text cache path is required")


def _validate_official_text_cache_binding(config: Mapping[str, Any]) -> None:
    """Verify the small immutable binding, never re-hash the 72-GiB cache."""

    official = _mapping(config.get("official"), "official")
    manifest = official.get("text_cache_binding_manifest")
    expected_sha = _mapping(config.get("artifacts"), "artifacts").get(
        "official_text_cache_binding_manifest_sha256"
    )
    if bool(official.get("on_the_fly_text_smoke", False)):
        _require(manifest is None, "on-the-fly text run must not bind an official text cache")
        _require(expected_sha is None, "on-the-fly text run must not bind an official text-cache SHA")
        return
    _require(bool(manifest), "precomputed official text cache requires a binding manifest")
    _validate_sha(expected_sha, "artifacts.official_text_cache_binding_manifest_sha256")
    from .release_official_text_cache_binding import (
        ReleaseOfficialTextCacheBindingError,
        verify_binding,
    )

    try:
        value = verify_binding(
            manifest,
            expected_sha256=str(expected_sha),
            expected_base_lineage_sha256=config["artifacts"][
                "base_lineage_manifest_sha256"
            ],
            expected_cache_dir=official["text_cache_dir"],
        )
    except ReleaseOfficialTextCacheBindingError as exc:
        raise ConfigAuditError(f"invalid official text-cache binding: {exc}") from exc
    _require(
        value["base_lineage"]["official_manifest_sha256"]
        == config["artifacts"]["official_task_manifest_sha256"],
        "official text-cache binding names a different task manifest",
    )


def _validate_paired(config: Mapping[str, Any], control: str) -> None:
    paired = _mapping(config.get("paired"), "paired")
    mode = _expected_paired_mode(control)
    _required_keys(
        paired,
        (
            "enabled",
            "protocol_id",
            "supervision_mode",
            "action_root",
            "action_manifest",
            "action_audit",
            "state_bank",
            "text_cache_dir",
            "cache",
            "split",
            "layer",
            "token_count",
            "token_dim",
            "variants",
            "view_count",
            "r3_role",
            "camera_names",
            "camera_count",
            "native_fps",
            "action_steps",
            "action_dim",
            "temporal_resampling",
            "native_action_targets",
            "same_physical_state_positives",
            "same_task_state_negatives",
            "action_supervision",
            "contrastive_supervision",
        ),
        "paired",
    )
    _require(paired["enabled"] is (mode != "none"), "paired.enabled disagrees with control")
    _require(paired["supervision_mode"] == mode, f"paired.supervision_mode must be {mode}")
    _require(paired["protocol_id"] == POLICY_PROTOCOL_ID, "paired protocol_id changed")
    _require(tuple(paired["variants"]) == POLICY_VARIANTS, "paired variants must be ordered C/R1/R2/R3")
    _require(paired["view_count"] == POLICY_VIEW_COUNT, "paired view_count must be four scene versions")
    _require(paired["r3_role"] == POLICY_R3_ROLE, "R3 must be training_positive")
    _require(tuple(paired["camera_names"]) == POLICY_CAMERA_NAMES, "paired camera names changed")
    _require(paired["camera_count"] == POLICY_CAMERA_COUNT, "each scene must contain three cameras")
    _require(paired["native_fps"] == POLICY_NATIVE_FPS, "paired data must be native 50 Hz")
    _require(paired["action_steps"] == POLICY_ACTION_STEPS, "paired action window must be 32")
    _require(paired["action_dim"] == POLICY_ACTION_DIM, "paired action dimension must be 14")
    _require(paired["temporal_resampling"] == POLICY_TEMPORAL_RESAMPLING, "temporal interpolation is forbidden")
    _require(paired["native_action_targets"] is True, "native action targets required")
    _require(paired["split"] == "train", "Stage-2 paired training must use train split")
    _require(paired["layer"] == 16 and paired["token_count"] == 120 and paired["token_dim"] == 3072, "Layer-16 cache shape changed")
    _require(paired["same_physical_state_positives"] is True, "paired positives changed")
    _require(paired["same_task_state_negatives"] is True, "paired negatives changed")
    if mode == "none":
        for key in ("action_root", "action_manifest", "action_audit", "state_bank", "text_cache_dir", "cache"):
            _require(paired[key] is None, f"unpaired control must set paired.{key}=null")
        _require(paired["action_supervision"] is False and paired["contrastive_supervision"] is False, "unpaired supervision leaked")
    else:
        for key in ("action_root", "action_manifest", "action_audit", "state_bank", "text_cache_dir"):
            _require(bool(paired[key]), f"paired.{key} is required")
        if mode == "action":
            _require(paired["cache"] is None, "C2 must not load contrastive cache")
            _require(paired["action_supervision"] is True and paired["contrastive_supervision"] is False, "C2 supervision changed")
        else:
            _require(bool(paired["cache"]), "contrastive control requires paired.cache")
            enabled = _contrastive_gradient_enabled(config, control)
            _require(
                paired["action_supervision"] is False
                and paired["contrastive_supervision"] is enabled,
                "paired contrastive gradient switch changed",
            )


def _validate_supervision_and_loss(config: Mapping[str, Any], control: str) -> None:
    mode = _expected_paired_mode(control)
    contrastive_enabled = _contrastive_gradient_enabled(config, control)
    supervision = _mapping(config.get("supervision"), "supervision")
    loss = _mapping(config.get("loss"), "loss")
    _required_keys(supervision, ("streams", "concatenate_datasets", "official_action", "paired_action", "paired_contrastive", "video_loss"), "supervision")
    _require(supervision["concatenate_datasets"] is False, "official/paired streams must stay independent")
    _require(supervision["official_action"] is True and supervision["video_loss"] is False, "Stage-2 supervision changed")
    _require(supervision["streams"] == ("official_only" if mode == "none" else "dual_independent"), "stream mode changed")
    _require(supervision["paired_action"] is (mode == "action"), "paired action flag changed")
    _require(supervision["paired_contrastive"] is contrastive_enabled, "paired contrastive flag changed")
    _require(loss.get("action") == "native_flow_matching_mse", "official action loss changed")
    _require(loss.get("video") is None, "Stage-2 video loss must be null")
    _require(float(loss.get("lambda_paired_action", -1)) == (1.0 if mode == "action" else 0.0), "lambda_paired_action changed")
    _require(float(loss.get("lambda_contrastive", -1)) == (0.1 if contrastive_enabled else 0.0), "lambda_contrastive changed")
    _require(float(loss.get("temperature", -1)) == 0.07, "contrastive temperature changed")
    _require(loss.get("contrastive") == "multi_positive_supcon", "contrastive loss changed")


def _validate_optimizer_training(config: Mapping[str, Any], control: str, formal: bool) -> None:
    optimizer = _mapping(config.get("optimizer"), "optimizer")
    training = _mapping(config.get("training"), "training")
    _required_keys(optimizer, ("name", "lr_scheduler", "trainable_parameter_dtype", "head_adapter_lr", "action_dit_lr", "weight_decay", "betas"), "optimizer")
    _required_keys(training, ("seed", "max_steps", "official_batch_size", "paired_groups_per_batch", "world_size", "gradient_accumulation_steps", "effective_official_global_batch", "effective_paired_groups_per_step", "num_workers", "mixed_precision", "model_dtype", "max_grad_norm", "save_optimizer", "require_cuda", "separate_stream_rng", "preserve_official_sequence_across_controls"), "training")
    if control == "c0_original":
        _require(training["seed"] is None, "C0 has no training seed")
        for key in (
            "max_steps",
            "official_batch_size",
            "paired_groups_per_batch",
            "world_size",
            "gradient_accumulation_steps",
            "effective_official_global_batch",
            "effective_paired_groups_per_step",
        ):
            _require(training[key] == 0, f"C0 training.{key} must be zero")
        _require(training["save_optimizer"] is False, "C0 cannot save optimizer state")
        _require(training["require_cuda"] is True, "C0 rollout must require CUDA")
        return
    _require(str(optimizer["name"]).lower() == "adamw", "optimizer must be AdamW")
    _require(str(optimizer["lr_scheduler"]).lower() == "constant", "Stage-2 LR scheduler must be explicit constant")
    _require(str(optimizer["trainable_parameter_dtype"]).lower() == "fp32", "trainable/master state must be fp32")
    _require(float(optimizer["head_adapter_lr"]) == 1e-4, "Head/GCA LR must be 1e-4")
    _require(float(optimizer["action_dit_lr"]) == 1e-5, "ActionDiT LR must be 1e-5")
    _require(float(optimizer["weight_decay"]) == 0.0, "Stage-2 weight_decay must be zero")
    _require(tuple(float(value) for value in optimizer["betas"]) == (0.9, 0.95), "AdamW betas changed")
    _require(training["require_cuda"] is True, "FastWAM run must require CUDA")
    _require(training["mixed_precision"] == "bf16" and training["model_dtype"] == "bf16", "training precision changed")
    _require(training["separate_stream_rng"] is True, "official and paired RNG streams must be separate")
    _require(training["preserve_official_sequence_across_controls"] is True, "official sequence fairness guard missing")
    for key in ("seed", "max_steps", "official_batch_size", "paired_groups_per_batch", "world_size", "gradient_accumulation_steps", "effective_official_global_batch", "effective_paired_groups_per_step", "num_workers"):
        value = training[key]
        if _is_placeholder(value):
            continue
        lower = 0 if key in {"seed", "num_workers"} else 1
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= lower, f"training.{key} must be integer >= {lower}")
    _require(
        training["gradient_accumulation_steps"] == 1
        or _is_placeholder(training["gradient_accumulation_steps"]),
        "Stage-2 gradient_accumulation_steps must be one",
    )
    if not any(
        _is_placeholder(training[key])
        for key in (
            "official_batch_size",
            "paired_groups_per_batch",
            "world_size",
            "effective_official_global_batch",
            "effective_paired_groups_per_step",
        )
    ):
        _require(
            training["effective_official_global_batch"]
            == training["official_batch_size"] * training["world_size"],
            "effective official global batch does not match local batch x world size",
        )
        _require(
            training["effective_paired_groups_per_step"]
            == training["paired_groups_per_batch"] * training["world_size"],
            "effective paired groups do not match local groups x world size",
        )
    if formal and not _is_placeholder(training["seed"]):
        _require(training["seed"] in FORMAL_SEEDS, f"formal seed must be one of {FORMAL_SEEDS}")
    _require(isinstance(training["max_grad_norm"], (int, float)) and float(training["max_grad_norm"]) > 0, "max_grad_norm must be positive")
    _require(isinstance(training["save_optimizer"], bool), "save_optimizer must be boolean")
    if "save_every" in training:
        save_every = training["save_every"]
        _require(
            isinstance(save_every, int)
            and not isinstance(save_every, bool)
            and save_every >= 0,
            "training.save_every must be a non-negative integer",
        )
        if not _is_placeholder(training["max_steps"]):
            _require(
                save_every <= int(training["max_steps"]),
                "training.save_every cannot exceed max_steps",
            )
        if save_every > 0:
            _require(
                training["save_optimizer"] is True,
                "periodic native checkpoints require save_optimizer=true",
            )
    if "resume" in training:
        resume_value = training["resume"]
        _require(
            resume_value is None
            or resume_value == ""
            or isinstance(resume_value, str),
            "training.resume must be null or a state-directory string",
        )
    if (
        control in {"p_v1", "p_v2"}
        or _is_c1_c3_engineering_smoke(config)
    ) and not _is_placeholder(training["max_steps"]):
        _require(training["max_steps"] >= len(TASKS), "three-task smoke requires at least three steps")


def _validate_evaluation(config: Mapping[str, Any], formal: bool, control: str) -> None:
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    _required_keys(
        evaluation,
        (
            "tasks",
            "required_domains",
            "rollout_protocol_id",
            "simulator_seed_bank_manifest",
            "simulator_seed_bank_id",
            "simulator_seed_bank_purpose",
            "episodes_per_task",
        ),
        "evaluation",
    )
    _require(tuple(evaluation["tasks"]) == TASKS, "evaluation task order changed")
    _require(tuple(evaluation["required_domains"]) == ("clean", "official_random"), "Policy evaluation must be Clean/official Random only")
    _require(bool(evaluation["rollout_protocol_id"]), "rollout_protocol_id required")
    _require(bool(evaluation["simulator_seed_bank_id"]), "simulator_seed_bank_id required")
    _require(bool(evaluation["simulator_seed_bank_manifest"]), "simulator_seed_bank_manifest required")
    stage = str(config.get("stage", ""))
    expected_purpose = {
        "smoke": "engineering_smoke",
        "dev_pilot": "dev_selection",
        PV2_FOLLOWUP_STAGE: "dev_selection",
        "formal": "final_test",
    }.get(stage, "final_test" if control == "c0_original" else "development_analysis")
    _require(
        evaluation["simulator_seed_bank_purpose"] == expected_purpose,
        f"{stage}/{control} seed-bank purpose must be {expected_purpose}",
    )
    episodes = evaluation["episodes_per_task"]
    if not _is_placeholder(episodes):
        _require(isinstance(episodes, int) and episodes > 0, "episodes_per_task must be positive")
        if formal:
            _require(episodes == 100, "formal evaluation requires 100 episodes/task/domain")
        if stage == "dev_pilot":
            _require(
                episodes == DEV_EPISODES_PER_CELL,
                f"dev_pilot evaluation requires {DEV_EPISODES_PER_CELL} episodes/task/domain",
            )
        if stage == PV2_FOLLOWUP_STAGE:
            _require(
                episodes == DEV_EPISODES_PER_CELL,
                f"{PV2_FOLLOWUP_STAGE} requires {DEV_EPISODES_PER_CELL} episodes/task/domain",
            )
        if stage == "smoke":
            _require(episodes == 1, "engineering smoke evaluation requires one episode/task/domain")
    _require("r3_background_only" not in evaluation, "R3 is not a Policy evaluation domain")


def _validate_formal_protocol(config: Mapping[str, Any], control: str) -> None:
    protocol = _mapping(config.get("protocol"), "protocol")
    _required_keys(protocol, ("matrix_id", "comparison_role", "base_parent", "action_distribution", "head_initialization", "seeds"), "protocol")
    _require(protocol["matrix_id"] == FORMAL_MATRIX_ID, "formal matrix_id changed")
    _require(protocol["comparison_role"] == FORMAL_ROLES[control], "formal comparison role changed")
    _require(protocol["base_parent"] == "author_release_50task_clean_random", "formal release parent changed")
    _require(protocol["action_distribution"] == "official_clean_plus_random_full_550", "formal action distribution changed")
    _require(protocol["head_initialization"] == "seeded_random", "formal Head initialization changed")
    _require(
        config["policy"].get("head_init_mode") == "random"
        and config["policy"].get("head_init") is None
        and config["artifacts"].get("head_init_sha256") is None,
        "formal main comparison must actually use seeded random Head/GCA initialization",
    )
    _require(tuple(protocol["seeds"]) == FORMAL_SEEDS, "formal seeds must be [1,2,3]")
    if control in {"c1_architecture_only", "c3_ours"}:
        amendment = config.get("formal_recipe_amendment_manifest")
        _require(
            isinstance(amendment, str)
            and amendment.endswith("formal_release_stage2_recipe_amendment_20260819.json"),
            "formal config must bind the disclosed release Stage-2 recipe amendment",
        )


def validate_config_structure(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
    _require(config.get("kind") == CONFIG_KIND, f"kind must be {CONFIG_KIND!r}")
    control = str(config.get("control", ""))
    _require(control in SUPPORTED_CONTROLS, f"unsupported control {control!r}")
    formal = config.get("formal")
    _require(isinstance(formal, bool), "formal must be boolean")
    stage = str(config.get("stage", ""))
    if formal:
        _require(stage == "formal", "stage/control/formal mismatch")
    elif control in {"p_v1", "p_v2"}:
        _require(stage in {"smoke", "dev_pilot"}, "P-mode stage must be smoke or dev_pilot")
    elif control in {"c1_architecture_only", "c3_ours"}:
        _require(
            stage in {"control", "smoke", PV2_FOLLOWUP_STAGE},
            "C1/C3 stage must be control, smoke, or mechanism_followup",
        )
    else:
        _require(stage == "control", "stage/control/formal mismatch")
    if control in {"p_v1", "p_v2"}:
        expected_selection_role = (
            "engineering_method_smoke" if stage == "smoke" else "c1_lambda0"
        )
        _require(
            config.get("selection_role") == expected_selection_role,
            f"{stage} P-mode selection_role must be {expected_selection_role}",
        )
    else:
        _require("selection_role" not in config, "selection_role is only valid for P-mode runs")
    if _is_pv2_actiondit_followup(config):
        _require(
            config.get("mechanism_protocol_manifest"),
            "P-v2 follow-up requires mechanism_protocol_manifest",
        )
        _require(
            config.get("policy", {}).get("regime") == "p_v2",
            "P-v2 follow-up must train ActionDiT under policy.regime=p_v2",
        )
    else:
        _require(
            config.get("study_role") != PV2_FOLLOWUP_ROLE,
            "post-hoc mechanism role requires mechanism_followup stage",
        )
    _require(tuple(config.get("tasks", ())) == TASKS, f"tasks must be exactly {TASKS}")
    for key in (
        "experiment_id",
        "output_dir",
        "base_checkpoint",
        "model_base_path",
        "base_lineage_manifest",
    ):
        _require(bool(config.get(key)), f"{key} is required")
    paired_binding = config.get("release_paired_binding_manifest")
    if control == "c0_original":
        _require(paired_binding is None, "C0 release_paired_binding_manifest must be null")
    else:
        _require(bool(paired_binding), f"{control} requires release_paired_binding_manifest")
    formal_lock = config.get("formal_protocol_lock_manifest")
    primary_formal = bool(formal) and control in {
        "c1_architecture_only",
        "c3_ours",
    }
    if primary_formal:
        _require(bool(formal_lock), "formal config requires formal_protocol_lock_manifest")
    else:
        _require(formal_lock is None, "non-primary config must not bind a formal protocol lock")
    selection_manifest = config.get("p_mode_selection_manifest")
    if (
        control in {"p_v1", "p_v2", "c0_original"}
        or _is_c1_c3_engineering_smoke(config)
    ):
        _require(selection_manifest is None, f"{control} p_mode_selection_manifest must be null")
    else:
        _require(bool(selection_manifest), f"{control} requires p_mode_selection_manifest")
    _validate_execution(config, control, bool(formal))
    _validate_artifacts(config, control)
    _validate_policy(config, control)
    _validate_official(config, bool(formal))
    _validate_paired(config, control)
    _validate_supervision_and_loss(config, control)
    _validate_optimizer_training(config, control, bool(formal))
    _validate_evaluation(config, bool(formal), control)
    if formal:
        _require(control in FORMAL_ROLES, "formal matrix supports only C1/C2/C3")
        _validate_formal_protocol(config, control)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound_json(
    path_value: Any,
    expected_sha: Any,
    *,
    label: str,
) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser().resolve()
    _require(path.is_file(), f"{label} does not exist: {path}")
    _require(_file_sha256(path) == expected_sha, f"{label} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigAuditError(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _validate_mode_and_seed_bank_bindings(config: Mapping[str, Any]) -> None:
    artifacts = config["artifacts"]
    evaluation = config["evaluation"]
    seed_payload = _load_bound_json(
        evaluation["simulator_seed_bank_manifest"],
        artifacts["simulator_seed_bank_manifest_sha256"],
        label="simulator seed-bank manifest",
    )
    try:
        seed_bank = validate_seed_bank_descriptor(
            seed_payload,
            expected_purpose=str(evaluation["simulator_seed_bank_purpose"]),
        )
    except PModeSelectionError as exc:
        raise ConfigAuditError(f"invalid simulator seed-bank manifest: {exc}") from exc
    _require(
        seed_bank["simulator_seed_bank_id"] == evaluation["simulator_seed_bank_id"],
        "simulator seed-bank id differs from bound manifest",
    )
    _require(
        seed_bank["episodes_per_cell"] == evaluation["episodes_per_task"],
        "simulator seed-bank episodes differ from evaluation",
    )

    control = str(config["control"])
    if (
        control in {"p_v1", "p_v2", "c0_original"}
        or _is_c1_c3_engineering_smoke(config)
    ):
        return
    selection_payload = _load_bound_json(
        config["p_mode_selection_manifest"],
        artifacts["p_mode_selection_manifest_sha256"],
        label="P-mode selection manifest",
    )
    try:
        selection = validate_selection_manifest_payload(selection_payload)
    except PModeSelectionError as exc:
        raise ConfigAuditError(f"invalid P-mode selection manifest: {exc}") from exc
    if _is_pv2_actiondit_followup(config):
        _require(
            selection["winner"] == "p_v1",
            "post-hoc P-v2 study must disclose the original P-v1 winner",
        )
        _require(
            config["policy"]["regime"] == "p_v2",
            "post-hoc mechanism study must intentionally override to P-v2",
        )
        _require(
            seed_bank["simulator_seed"] == 53
            and seed_bank["episodes_per_cell"] == DEV_EPISODES_PER_CELL,
            "P-v2 mechanism pilot must use locked dev seed 53 and 20 episodes/cell",
        )
        old_dev_members = set(selection["dev_seed_bank"]["members"])
        _require(
            old_dev_members.isdisjoint(seed_bank["members"]),
            "P-v2 mechanism dev bank overlaps the old P-mode dev bank",
        )
        _require(
            4_300_000 not in set(seed_bank["members"]),
            "P-v2 mechanism dev bank overlaps the author-stock seed42 start",
        )
        return
    _require(
        config["policy"]["regime"] == selection["winner"],
        "policy.regime differs from P-mode selection winner",
    )
    if evaluation["simulator_seed_bank_purpose"] == "final_test":
        dev_bank = selection["dev_seed_bank"]
        exclusions = {
            item["simulator_seed_bank_id"]: item
            for item in seed_bank["disjoint_from"]
        }
        _require(
            dev_bank["simulator_seed_bank_id"] in exclusions,
            "final_test seed bank does not exclude the selected dev bank",
        )
        excluded = exclusions[dev_bank["simulator_seed_bank_id"]]
        _require(
            excluded["members_sha256"] == dev_bank["members_sha256"],
            "final_test exclusion binds different dev members",
        )


def _validate_release_lineage_binding(config: Mapping[str, Any]) -> None:
    """Prove a runnable control consumes the one locked author release."""

    try:
        lineage = verify_author_release_lineage(
            config["base_lineage_manifest"],
            checkpoint_path=config["base_checkpoint"],
            dataset_stats_path=config["official"]["dataset_stats"],
            official_manifest_path=config["official"]["canonical_task_manifest"],
            expected_manifest_sha256=config["artifacts"]["base_lineage_manifest_sha256"],
        )
    except ReleaseLineageError as exc:
        raise ConfigAuditError(f"invalid author release lineage: {exc}") from exc
    artifacts = config["artifacts"]
    _require(lineage["checkpoint"]["sha256"] == artifacts["base_checkpoint_sha256"], "config checkpoint SHA differs from release lineage")
    _require(lineage["dataset_stats"]["sha256"] == artifacts["dataset_stats_sha256"], "config stats SHA differs from release lineage")
    _require(lineage["official_partition"]["manifest"]["sha256"] == artifacts["official_task_manifest_sha256"], "config official manifest SHA differs from release lineage")


def _validate_release_paired_binding(config: Mapping[str, Any]) -> None:
    """Bind every Stage-2/P-mode run to the same audited 600-scene dataset."""

    if config["control"] == "c0_original":
        return
    # Keep structure-only config audits lightweight; this module pulls in the
    # native dataset validators and torch and is needed only after placeholders
    # have been fully resolved.
    from .release_paired_binding import (
        ReleasePairedBindingError,
        verify_release_paired_binding,
    )

    try:
        binding = verify_release_paired_binding(
            config["release_paired_binding_manifest"],
            expected_sha256=config["artifacts"]["release_paired_binding_manifest_sha256"],
        )
    except ReleasePairedBindingError as exc:
        raise ConfigAuditError(f"invalid release paired binding: {exc}") from exc
    _require(
        binding["base_lineage"]["sha256"]
        == config["artifacts"]["base_lineage_manifest_sha256"],
        "paired binding descends from a different base lineage",
    )
    _require(
        Path(str(binding["paired_dataset"]["root"])).expanduser().resolve()
        == Path(str(config["paired"]["action_root"])).expanduser().resolve(),
        "paired action root differs from the release paired binding",
    )
    meta = binding["meta_artifacts"]
    expected = {
        "policy_native_action_manifest": "paired_action_manifest_sha256",
        "policy_paired_state_bank": "paired_state_bank_sha256",
    }
    for binding_key, artifact_key in expected.items():
        _require(
            meta[binding_key]["sha256"] == config["artifacts"][artifact_key],
            f"{artifact_key} differs from the release paired binding",
        )


def _validate_formal_recipe_amendment_binding(config: Mapping[str, Any]) -> None:
    """Bind executable primary runs to the transparently amended recipe.

    The P-mode dev run completed before a numeric formal recipe artifact
    existed.  The amendment records that deviation and derives the formal
    step count from the already fixed 720-state inventory.  Validating the
    exact file and recipe here prevents a later result-driven rewrite.
    """

    control = str(config["control"])
    if not bool(config["formal"]) or control not in {
        "c1_architecture_only",
        "c3_ours",
    }:
        return
    payload = _load_bound_json(
        config["formal_recipe_amendment_manifest"],
        config["artifacts"]["formal_recipe_amendment_manifest_sha256"],
        label="formal Stage-2 recipe amendment",
    )
    _require(
        payload.get("kind") == "policy_release_stage2_recipe_amendment"
        and payload.get("schema_version") == 2
        and payload.get("status") == "PASS",
        "formal Stage-2 recipe amendment kind/version/status differs",
    )
    recipe = _mapping(payload.get("locked_recipe"), "formal amendment locked_recipe")
    expected = {
        "selected_policy_regime": "p_v1",
        "stage2_training_seeds": [1, 2, 3],
        "max_steps": 1800,
        "official_batch_size_per_rank": 1,
        "paired_groups_per_batch_per_rank": 2,
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "effective_official_global_batch": 1,
        "effective_paired_groups_per_step": 2,
        "head_adapter_lr": 1.0e-4,
        "action_dit_lr_inactive_under_p_v1": 1.0e-5,
        "lr_scheduler": "constant",
        "temperature": 0.07,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
        "mixed_precision": "bf16",
        "trainable_parameter_dtype": "fp32",
    }
    for field, expected_value in expected.items():
        _require(
            recipe.get(field) == expected_value,
            f"formal Stage-2 recipe amendment changed at {field}",
        )
    lambda_by_control = _mapping(
        recipe.get("lambda_contrastive"),
        "formal amendment lambda_contrastive",
    )
    _require(
        lambda_by_control
        == {"c1_architecture_only": 0.0, "c3_ours": 0.1},
        "formal amendment C1/C3 contrastive coefficients changed",
    )
    correction = _mapping(
        payload.get("runtime_incident_correction"),
        "formal amendment runtime_incident_correction",
    )
    _require(
        correction.get("formal_checkpoints_created") == 0
        and correction.get("online_rollouts_started") is False,
        "formal runtime correction no longer describes a pre-result incident",
    )
    audit_correction = _mapping(
        correction.get("audit_semantics_correction"),
        "formal amendment audit_semantics_correction",
    )
    _require(
        audit_correction.get("loss_formula_changed") is False
        and audit_correction.get("scheduler_weight_clamped_or_skipped") is False
        and audit_correction.get("optimizer_step_skipped") is False,
        "formal runtime correction changed the native optimization objective",
    )
    rng_correction = _mapping(
        correction.get("rng_stream_correction"),
        "formal amendment rng_stream_correction",
    )
    _require(
        rng_correction.get("new_policy_id")
        == "stage2_mixed_radix_no_collision_v1"
        and rng_correction.get("cross_seed_collision_free_for_formal_range") is True
        and rng_correction.get("same_seed_c1_c3_exactly_paired") is True
        and rng_correction.get("training_seed_labels_unchanged") == [1, 2, 3],
        "formal runtime RNG correction differs from the reviewed policy",
    )
    source_hashes = _mapping(
        correction.get("bound_runtime_source_sha256"),
        "formal amendment bound_runtime_source_sha256",
    )
    expected_source_files = {
        "experiments/robotwin/policy_content_adapter/train.py",
        "experiments/robotwin/policy_content_adapter/losses.py",
        "experiments/robotwin/policy_content_adapter/release_formal_c1c3_audit.py",
        "experiments/robotwin/policy_content_adapter/training_audit.py",
    }
    _require(
        set(source_hashes) == expected_source_files,
        "formal runtime source binding file set changed",
    )
    project_root = Path(__file__).resolve().parents[3]
    for relative_path, expected_sha256 in source_hashes.items():
        _validate_sha(expected_sha256, f"runtime source {relative_path}")
        _require(
            _file_sha256(project_root / relative_path) == expected_sha256,
            f"formal runtime source drifted at {relative_path}",
        )
    training = _mapping(config["training"], "training")
    optimizer = _mapping(config["optimizer"], "optimizer")
    _require(config["policy"]["regime"] == recipe["selected_policy_regime"], "formal regime differs from amendment")
    _require(training["seed"] in tuple(recipe["stage2_training_seeds"]), "formal seed differs from amendment")
    config_recipe = {
        "max_steps": training["max_steps"],
        "official_batch_size_per_rank": training["official_batch_size"],
        "paired_groups_per_batch_per_rank": training["paired_groups_per_batch"],
        "world_size": training["world_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "effective_official_global_batch": training["effective_official_global_batch"],
        "effective_paired_groups_per_step": training["effective_paired_groups_per_step"],
        "head_adapter_lr": optimizer["head_adapter_lr"],
        "action_dit_lr_inactive_under_p_v1": optimizer["action_dit_lr"],
        "lr_scheduler": optimizer["lr_scheduler"],
        "temperature": config["loss"]["temperature"],
        "optimizer": str(optimizer["name"]).lower(),
        "weight_decay": optimizer["weight_decay"],
        "betas": list(optimizer["betas"]),
        "mixed_precision": training["mixed_precision"],
        "trainable_parameter_dtype": optimizer["trainable_parameter_dtype"],
    }
    for field, actual in config_recipe.items():
        _require(
            actual == recipe[field],
            f"formal executable config differs from amended recipe at {field}",
        )
    _require(
        float(config["loss"]["lambda_contrastive"])
        == float(lambda_by_control[control]),
        "formal executable lambda differs from amended recipe",
    )


def _validate_formal_protocol_lock_binding(config: Mapping[str, Any]) -> None:
    """Match a resolved primary config to its pre-final canonical projection."""

    control = str(config["control"])
    if not bool(config["formal"]) or control not in {
        "c1_architecture_only",
        "c3_ours",
    }:
        return
    payload = _load_bound_json(
        config["formal_protocol_lock_manifest"],
        config["artifacts"]["formal_protocol_lock_manifest_sha256"],
        label="formal protocol lock manifest",
    )
    try:
        lock = validate_formal_protocol_lock_manifest_payload(payload)
    except PModeSelectionError as exc:
        raise ConfigAuditError(f"invalid formal protocol lock manifest: {exc}") from exc
    _require(
        lock["base_lineage_manifest"]["sha256"]
        == config["artifacts"]["base_lineage_manifest_sha256"],
        "formal protocol lock binds a different base lineage",
    )
    _require(
        lock["p_mode_selection_manifest"]["sha256"]
        == config["artifacts"]["p_mode_selection_manifest_sha256"],
        "formal protocol lock binds a different P-mode selection",
    )
    _require(
        lock["selected_policy_regime"] == config["policy"]["regime"],
        "formal protocol lock selected a different policy regime",
    )
    seed = config["training"]["seed"]
    _require(seed in FORMAL_SEEDS, f"resolved formal seed must be one of {FORMAL_SEEDS}")
    row = lock["resolved_configs"][control][FORMAL_SEEDS.index(seed)]
    _require(
        row["training_seed"] == seed and row["control"] == control,
        "formal protocol lock row/seed differs from config",
    )
    _require(
        float(row["lambda_contrastive"])
        == float(config["loss"]["lambda_contrastive"]),
        "formal protocol lock lambda differs from config",
    )
    projection_sha = canonical_sha256(formal_config_protocol_projection(config))
    _require(
        row["protocol_projection_sha256"] == projection_sha,
        "resolved formal config differs from its locked protocol projection",
    )


def _validate_pv2_followup_protocol_binding(config: Mapping[str, Any]) -> None:
    """Validate the immutable, explicitly post-hoc P-v2 mechanism protocol."""

    if not _is_pv2_actiondit_followup(config):
        return
    artifacts = _mapping(config["artifacts"], "artifacts")
    payload = _load_bound_json(
        config["mechanism_protocol_manifest"],
        artifacts["mechanism_protocol_manifest_sha256"],
        label="P-v2 mechanism protocol manifest",
    )
    _require(
        payload.get("kind") == PV2_FOLLOWUP_PROTOCOL_KIND
        and payload.get("schema_version")
        == PV2_FOLLOWUP_PROTOCOL_SCHEMA_VERSION
        and payload.get("status") == "PASS",
        "P-v2 mechanism protocol kind/version/status differs",
    )
    study = _mapping(payload.get("study_classification"), "study_classification")
    _require(
        study.get("role") == PV2_FOLLOWUP_ROLE
        and study.get("post_hoc_after_primary_results") is True
        and study.get("primary_experiment_remains_unchanged") is True,
        "P-v2 follow-up must remain a disclosed post-hoc mechanism study",
    )

    primary = _mapping(payload.get("primary_pv1_result"), "primary_pv1_result")
    for name in ("summary", "completion_audit"):
        identity = _mapping(primary.get(name), f"primary_pv1_result.{name}")
        path = Path(str(identity.get("path", ""))).expanduser().resolve()
        _require(path.is_file(), f"bound P-v1 {name} is missing")
        _validate_sha(identity.get("sha256"), f"primary_pv1_result.{name}.sha256")
        _require(
            _file_sha256(path) == identity["sha256"],
            f"bound P-v1 {name} changed",
        )
        bound = _load_bound_json(path, identity["sha256"], label=f"P-v1 {name}")
        _require(bound.get("status") == "PASS", f"P-v1 {name} is not PASS")

    history = _mapping(
        payload.get("historical_p_mode_selection"),
        "historical_p_mode_selection",
    )
    _require(
        history.get("winner") == "p_v1"
        and history.get("use") == "historical_context_not_treatment_selection",
        "historical P-mode selection role changed",
    )
    _require(
        history.get("sha256")
        == artifacts["p_mode_selection_manifest_sha256"],
        "P-v2 mechanism protocol binds a different historical selection",
    )

    ancestry = _mapping(payload.get("common_ancestry"), "common_ancestry")
    expected_ancestry = {
        "base_checkpoint_sha256": artifacts["base_checkpoint_sha256"],
        "dataset_stats_sha256": artifacts["dataset_stats_sha256"],
        "base_lineage_manifest_sha256": artifacts[
            "base_lineage_manifest_sha256"
        ],
        "release_paired_binding_manifest_sha256": artifacts[
            "release_paired_binding_manifest_sha256"
        ],
        "official_task_manifest_sha256": artifacts[
            "official_task_manifest_sha256"
        ],
        "paired_action_manifest_sha256": artifacts[
            "paired_action_manifest_sha256"
        ],
        "paired_state_bank_sha256": artifacts["paired_state_bank_sha256"],
        "paired_text_cache_sha256": artifacts["paired_text_cache_sha256"],
        "paired_cache_sha256": artifacts["paired_cache_sha256"],
    }
    _require(
        dict(ancestry) == expected_ancestry,
        "P-v2 mechanism common ancestry/artifacts changed",
    )

    init_identity = _mapping(
        payload.get("action_dit_initialization_audit"),
        "action_dit_initialization_audit",
    )
    _require(
        init_identity.get("sha256")
        == artifacts["action_dit_initialization_audit_sha256"],
        "ActionDiT initialization audit SHA differs from config",
    )
    init_payload = _load_bound_json(
        init_identity.get("path"),
        init_identity.get("sha256"),
        label="ActionDiT initialization audit",
    )
    _require(
        init_payload.get("kind") == "policy_actiondit_release_initialization_audit"
        and init_payload.get("schema_version") == 1
        and init_payload.get("status") == "PASS",
        "ActionDiT initialization audit kind/version/status differs",
    )
    _require(
        init_payload.get("checkpoint_sha256")
        == artifacts["base_checkpoint_sha256"]
        and init_payload.get("tensor_count") == 824,
        "ActionDiT initialization audit names a different release payload",
    )
    _validate_sha(
        init_payload.get("action_dit_tensor_sha256"),
        "action_dit_initialization_audit.action_dit_tensor_sha256",
    )

    recipe = _mapping(payload.get("locked_training"), "locked_training")
    expected_recipe = {
        "policy_regime": "p_v2",
        "action_dit_trainable": True,
        "video_dit_frozen": True,
        "vae_frozen": True,
        "t5_frozen": True,
        "training_seeds": [1, 2, 3],
        "pilot_training_seed": 1,
        "max_steps": 1800,
        "official_batch_size": 1,
        "paired_groups_per_batch": 2,
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "head_adapter_lr": 1.0e-4,
        "action_dit_lr": 1.0e-5,
        "lr_scheduler": "constant",
        "mixed_precision": "bf16",
        "trainable_parameter_dtype": "fp32",
        "temperature": 0.07,
    }
    _require(dict(recipe) == expected_recipe, "locked P-v2 training recipe changed")
    training = _mapping(config["training"], "training")
    optimizer = _mapping(config["optimizer"], "optimizer")
    _require(training["seed"] == 1, "mechanism pilot must use training seed 1")
    _require(training["max_steps"] == recipe["max_steps"], "mechanism pilot step count changed")
    for field in (
        "official_batch_size",
        "paired_groups_per_batch",
        "world_size",
        "gradient_accumulation_steps",
        "mixed_precision",
    ):
        _require(training[field] == recipe[field], f"mechanism training.{field} changed")
    for field in ("head_adapter_lr", "action_dit_lr", "lr_scheduler", "trainable_parameter_dtype"):
        _require(optimizer[field] == recipe[field], f"mechanism optimizer.{field} changed")

    treatment = _mapping(payload.get("locked_treatment"), "locked_treatment")
    expected_lambda = 0.0 if config["control"] == "c1_architecture_only" else 0.1
    _require(
        treatment.get("c1_lambda_contrastive") == 0.0
        and treatment.get("c3_lambda_contrastive") == 0.1
        and treatment.get("only_permitted_difference")
        == "contrastive_coefficient_and_gradient",
        "P-v2 treatment allowlist changed",
    )
    _require(
        float(config["loss"]["lambda_contrastive"]) == expected_lambda,
        "P-v2 follow-up lambda differs from locked treatment",
    )

    pilot = _mapping(payload.get("pilot_gate"), "pilot_gate")
    _require(
        pilot.get("simulator_seed") == 53
        and pilot.get("episodes_per_task_domain") == DEV_EPISODES_PER_CELL
        and pilot.get("official_random_macro_delta_min") == 0.03
        and pilot.get("clean_macro_delta_min") == -0.03
        and pilot.get("stop_expansion_on_failure") is True
        and pilot.get("result_driven_tuning_forbidden") is True,
        "P-v2 pilot gate changed",
    )
    _require(
        pilot.get("seed_bank_manifest_sha256")
        == artifacts["simulator_seed_bank_manifest_sha256"]
        and pilot.get("seed_bank_id")
        == config["evaluation"]["simulator_seed_bank_id"],
        "P-v2 pilot seed bank differs from protocol",
    )
    confirmatory = _mapping(
        payload.get("confirmatory_intent"), "confirmatory_intent"
    )
    _require(
        confirmatory.get("only_if_pilot_passes") is True
        and confirmatory.get("simulator_seed") == 59
        and confirmatory.get("episodes_per_task_domain") == 100
        and confirmatory.get("training_seeds") == [1, 2, 3]
        and confirmatory.get("unopened_before_pilot_decision") is True,
        "P-v2 confirmatory intent changed",
    )


def validate_execution_ready(config: Mapping[str, Any]) -> None:
    validate_config_structure(config)
    placeholders = find_placeholders(config)
    _require(not placeholders, f"unresolved config placeholders: {', '.join(placeholders)}")
    execution = _mapping(config["execution"], "execution")
    _require(execution["runnable"] is True, "execution.runnable is not explicitly true")
    _require(execution["fail_closed"] is False, "execution.fail_closed is still true")
    _validate_release_lineage_binding(config)
    _validate_release_paired_binding(config)
    _validate_official_text_cache_binding(config)
    _validate_formal_recipe_amendment_binding(config)
    _validate_formal_protocol_lock_binding(config)
    _validate_pv2_followup_protocol_binding(config)
    _validate_mode_and_seed_bank_bindings(config)
    if bool(config["formal"]):
        _require(config["official"]["domain_verified"] is True, "formal official domain provenance is unverified")


def _common_stage2_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = config["artifacts"]
    return {
        "tasks": config["tasks"],
        "base_checkpoint": config["base_checkpoint"],
        "base_lineage_manifest": config["base_lineage_manifest"],
        "release_paired_binding_manifest": config["release_paired_binding_manifest"],
        "formal_recipe_amendment_manifest": config.get(
            "formal_recipe_amendment_manifest"
        ),
        "formal_protocol_lock_manifest": config.get("formal_protocol_lock_manifest"),
        "p_mode_selection_manifest": config["p_mode_selection_manifest"],
        "mechanism_protocol_manifest": config.get(
            "mechanism_protocol_manifest"
        ),
        "model_base_path": config["model_base_path"],
        "base_checkpoint_sha256": artifacts["base_checkpoint_sha256"],
        "dataset_stats_sha256": artifacts["dataset_stats_sha256"],
        "official_task_manifest_sha256": artifacts["official_task_manifest_sha256"],
        "base_lineage_manifest_sha256": artifacts[
            "base_lineage_manifest_sha256"
        ],
        "release_paired_binding_manifest_sha256": artifacts[
            "release_paired_binding_manifest_sha256"
        ],
        "formal_recipe_amendment_manifest_sha256": artifacts.get(
            "formal_recipe_amendment_manifest_sha256"
        ),
        "formal_protocol_lock_manifest_sha256": artifacts.get(
            "formal_protocol_lock_manifest_sha256"
        ),
        "head_init_sha256": artifacts["head_init_sha256"],
        "p_mode_selection_manifest_sha256": artifacts[
            "p_mode_selection_manifest_sha256"
        ],
        "mechanism_protocol_manifest_sha256": artifacts.get(
            "mechanism_protocol_manifest_sha256"
        ),
        "action_dit_initialization_audit_sha256": artifacts.get(
            "action_dit_initialization_audit_sha256"
        ),
        "simulator_seed_bank_manifest_sha256": artifacts[
            "simulator_seed_bank_manifest_sha256"
        ],
        "official_text_cache_binding_manifest": config["official"].get(
            "text_cache_binding_manifest"
        ),
        "official_text_cache_binding_manifest_sha256": artifacts.get(
            "official_text_cache_binding_manifest_sha256"
        ),
        "policy": config["policy"],
        "architecture": config["architecture"],
        "official": config["official"],
        "optimizer": config["optimizer"],
        "training": config["training"],
        "evaluation": config["evaluation"],
    }


def _assert_common_stage2(configs: Sequence[Mapping[str, Any]], label: str) -> None:
    reference = _common_stage2_identity(configs[0])
    for candidate in configs[1:]:
        _require(_common_stage2_identity(candidate) == reference, f"{label}: unfair common Stage-2 mismatch")


def _primary_pair_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the declared C1/C3 intervention fields.

    Comparing this normalized payload prevents an innocuous-looking new YAML
    field from silently becoming a second treatment difference.
    """

    value = copy.deepcopy(dict(config))
    for key in ("experiment_id", "control", "output_dir"):
        value.pop(key, None)
    if isinstance(value.get("execution"), dict):
        value["execution"].pop("blocked_reason", None)
    if isinstance(value.get("protocol"), dict):
        value["protocol"].pop("comparison_role", None)
    value["paired"].pop("contrastive_supervision", None)
    value["supervision"].pop("paired_contrastive", None)
    value["loss"].pop("lambda_contrastive", None)
    return value


def validate_c1_c3_pair(architecture_only: Mapping[str, Any], ours: Mapping[str, Any]) -> dict[str, Any]:
    validate_config_structure(architecture_only)
    validate_config_structure(ours)
    _require(architecture_only["control"] == "c1_architecture_only", "first config must be C1")
    _require(ours["control"] == "c3_ours", "second config must be C3")
    _assert_common_stage2((architecture_only, ours), "C1/C3")
    _require(architecture_only["paired"]["supervision_mode"] == "contrastive", "C1 must consume the matched paired cache")
    _require(ours["paired"]["supervision_mode"] == "contrastive", "C3 must use paired contrastive")
    for key in ("action_root", "action_manifest", "action_audit", "state_bank", "text_cache_dir", "cache"):
        _require(architecture_only["paired"][key] == ours["paired"][key], f"C1/C3 paired stream mismatch at {key}")
    for key in ("paired_action_manifest_sha256", "paired_state_bank_sha256", "paired_text_cache_sha256", "paired_cache_sha256"):
        _require(architecture_only["artifacts"][key] == ours["artifacts"][key], f"C1/C3 paired artifact mismatch at {key}")
    _require(architecture_only["paired"]["contrastive_supervision"] is False, "C1 must block paired contrastive gradient")
    _require(architecture_only["supervision"]["paired_contrastive"] is False, "C1 paired gradient switch must be off")
    _require(float(architecture_only["loss"]["lambda_contrastive"]) == 0.0, "C1 lambda_contrastive must be zero")
    _require(ours["paired"]["contrastive_supervision"] is True, "C3 paired contrastive gradient must be on")
    _require(ours["supervision"]["paired_contrastive"] is True, "C3 paired gradient switch must be on")
    _require(float(ours["loss"]["lambda_contrastive"]) == 0.1, "C3 lambda_contrastive must be 0.1")
    _require(
        _primary_pair_identity(architecture_only) == _primary_pair_identity(ours),
        "C1/C3 differ outside the declared contrastive lambda/gradient switches",
    )
    return {
        "baseline": architecture_only["experiment_id"],
        "method": ours["experiment_id"],
        "fairness": "PASS",
        "comparison": "C3-C1 paired contrastive total gain",
    }


def validate_c2_c3_pair(naive_aug: Mapping[str, Any], ours: Mapping[str, Any]) -> dict[str, Any]:
    validate_config_structure(naive_aug)
    validate_config_structure(ours)
    _require(naive_aug["control"] == "c2_naive_aug", "first config must be C2")
    _require(ours["control"] == "c3_ours", "second config must be C3")
    _assert_common_stage2((naive_aug, ours), "C2/C3")
    for key in ("action_root", "action_manifest", "action_audit", "state_bank", "text_cache_dir", "protocol_id", "variants", "split"):
        _require(naive_aug["paired"][key] == ours["paired"][key], f"C2/C3 paired dataset mismatch at {key}")
    _require(
        naive_aug["artifacts"]["paired_action_manifest_sha256"]
        == ours["artifacts"]["paired_action_manifest_sha256"],
        "C2/C3 paired manifest SHA mismatch",
    )
    _require(
        naive_aug["artifacts"]["paired_state_bank_sha256"]
        == ours["artifacts"]["paired_state_bank_sha256"],
        "C2/C3 paired state-bank SHA mismatch",
    )
    _require(
        naive_aug["artifacts"]["paired_text_cache_sha256"]
        == ours["artifacts"]["paired_text_cache_sha256"],
        "C2/C3 paired text-cache SHA mismatch",
    )
    _require(naive_aug["paired"]["supervision_mode"] == "action", "C2 must use paired action")
    _require(ours["paired"]["supervision_mode"] == "contrastive", "C3 must use paired contrastive")
    return {
        "baseline": naive_aug["experiment_id"],
        "method": ours["experiment_id"],
        "fairness": "PASS",
        "comparison": "C3-C2 contrastive versus naive action augmentation",
    }


def validate_formal_pair(baseline: Mapping[str, Any], augmented: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible public entry for the primary C1/C3 comparison."""
    _require(bool(baseline.get("formal")) and bool(augmented.get("formal")), "formal pair required")
    return validate_c1_c3_pair(baseline, augmented)


def validate_formal_matrix(config_dir: str | Path | None = None) -> dict[str, Any]:
    directory = Path(config_dir).resolve() if config_dir else Path(__file__).resolve().parent / "configs"
    configs = {name: load_config(directory / name) for name in FORMAL_FILENAMES}
    c1 = configs["formal_c1_architecture_only.yaml"]
    c3 = configs["formal_c3_ours.yaml"]
    for config in configs.values():
        validate_config_structure(config)
    _assert_common_stage2((c1, c3), "formal C1/C3")
    return {
        "status": "PASS",
        "matrix_id": FORMAL_MATRIX_ID,
        "formal_configs": list(FORMAL_FILENAMES),
        "comparisons": [validate_c1_c3_pair(c1, c3)],
        "formal_templates_execution_ready": False,
    }


def _legacy_row(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == LEGACY_SCHEMA_VERSION, f"legacy config {path.name} schema changed")
    execution = _mapping(config.get("execution"), f"legacy {path.name}.execution")
    _require(execution.get("runnable") is False and execution.get("fail_closed") is True, f"legacy config {path.name} must stay fail closed")
    return {
        "file": path.name,
        "control": config.get("control"),
        "formal": config.get("formal"),
        "structure": "LEGACY_V1",
        "execution_ready": False,
        "placeholder_count": len(find_placeholders(config)),
        "not_ready_reason": "superseded by Policy Protocol v2",
    }


def audit_config_directory(config_dir: str | Path | None = None) -> dict[str, Any]:
    directory = Path(config_dir).resolve() if config_dir else Path(__file__).resolve().parent / "configs"
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        config = load_config(path)
        if path.name == LEGACY_STAGE1_SEED_PLAN_FILENAME:
            _require(
                _file_sha256(path) == LEGACY_STAGE1_SEED_PLAN_SHA256,
                "preserved Stage1 seeds012 snapshot SHA-256 changed",
            )
            rows.append(
                {
                    "file": path.name,
                    "control": None,
                    "formal": False,
                    "structure": "LEGACY_STAGE1_SEED_PLAN",
                    "execution_ready": False,
                    "placeholder_count": len(find_placeholders(config)),
                    "not_ready_reason": "immutable cache-binding snapshot; never executable",
                }
            )
            continue
        if path.name in LEGACY_FORMAL_FILENAMES:
            rows.append(_legacy_row(path, config))
            continue
        # Stage-1 has a separate native-training schema and validator.
        if config.get("kind") == "policy_stage1_native_run":
            from .stage1 import validate_stage1_config

            validate_stage1_config(path)
            rows.append(
                {
                    "file": path.name,
                    "control": config.get("control"),
                    "formal": config.get("formal"),
                    "structure": "PASS_STAGE1",
                    "execution_ready": False,
                    "placeholder_count": len(find_placeholders(config)),
                    "not_ready_reason": "Stage-1 long training requires explicit human launch",
                }
            )
            continue
        validate_config_structure(config)
        try:
            validate_execution_ready(config)
        except ConfigAuditError as exc:
            ready = False
            reason = str(exc)
        else:
            ready = True
            reason = None
        rows.append(
            {
                "file": path.name,
                "control": config["control"],
                "formal": config["formal"],
                "structure": "PASS",
                "execution_ready": ready,
                "placeholder_count": len(find_placeholders(config)),
                "not_ready_reason": reason,
            }
        )
    short = {
        name: load_config(directory / name)
        for name in ("c1_architecture_only.yaml", "c2_naive_aug.yaml", "c3_ours.yaml")
    }
    return {
        "status": "PASS",
        "protocol": POLICY_PROTOCOL_ID,
        "configs": rows,
        "short_control_matrix": {
            "c1_c3": validate_c1_c3_pair(short["c1_architecture_only.yaml"], short["c3_ours.yaml"]),
            "c2_c3": validate_c2_c3_pair(short["c2_naive_aug.yaml"], short["c3_ours.yaml"]),
        },
        "formal_matrix": validate_formal_matrix(directory),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--config-dir")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--formal-matrix", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.formal_matrix:
        result: Any = validate_formal_matrix(args.config_dir)
    elif args.config:
        rows = []
        for path in args.config:
            config = load_config(path)
            if config.get("kind") == "policy_stage1_native_run":
                from .stage1 import validate_stage1_config

                validate_stage1_config(path)
            elif args.require_ready:
                validate_execution_ready(config)
            else:
                validate_config_structure(config)
            rows.append({"file": str(Path(path).resolve()), "placeholders": list(find_placeholders(config)), "status": "PASS"})
        result = {"status": "PASS", "configs": rows}
    else:
        result = audit_config_directory(args.config_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHOR_RELEASE_BASE_MANIFEST_SHA256",
    "AUTHOR_RELEASE_CHECKPOINT_SHA256",
    "AUTHOR_RELEASE_DATASET_STATS_SHA256",
    "AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256",
    "CONFIG_KIND",
    "ConfigAuditError",
    "FORMAL_FILENAMES",
    "FORMAL_MATRIX_ID",
    "KNOWN_SMOKE_IDENTITIES",
    "LEGACY_FORMAL_FILENAMES",
    "SCHEMA_VERSION",
    "TASKS",
    "audit_config_directory",
    "find_placeholders",
    "load_config",
    "validate_c1_c3_pair",
    "validate_c2_c3_pair",
    "validate_config_structure",
    "validate_execution_ready",
    "validate_formal_matrix",
    "validate_formal_pair",
]
