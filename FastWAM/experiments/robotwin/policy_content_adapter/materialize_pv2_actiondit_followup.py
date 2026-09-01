"""Materialize the disclosed P-v2 ActionDiT C1/C3 mechanism pilot.

This is deliberately separate from the completed P-v1 primary experiment.
It is CPU-only: it verifies and writes immutable configs/manifests, but never
starts training, deployment, or RoboTwin evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from .config_audit import (
    AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
    AUTHOR_RELEASE_CHECKPOINT_SHA256,
    AUTHOR_RELEASE_DATASET_STATS_SHA256,
    AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
    PV2_FOLLOWUP_PROTOCOL_KIND,
    PV2_FOLLOWUP_PROTOCOL_SCHEMA_VERSION,
    PV2_FOLLOWUP_ROLE,
    PV2_FOLLOWUP_STAGE,
    load_config,
    validate_c1_c3_pair,
    validate_execution_ready,
)
from .materialize_release_engineering_smoke import (
    CONFIG_DIR,
    DEFAULT_BINDING,
    DEFAULT_CACHE,
    DEFAULT_CACHE_AUDIT,
    DEFAULT_LINEAGE,
    DEFAULT_OFFICIAL_TEXT_BINDING,
    DEFAULT_OFFICIAL_TEXT_CACHE,
    DEFAULT_TEXT_CACHE,
    _file_sha256,
    _load_json,
    _write_new_bytes,
    _write_new_json,
    _write_new_yaml,
)
from .model import artifact_identity
from .p_mode_selection import (
    DEV_EPISODES_PER_CELL,
    build_seed_bank_descriptor,
    canonical_sha256,
    validate_seed_bank_descriptor,
    validate_selection_manifest_payload,
)
from .prepare_release_paired_text_cache import verify_release_paired_text_cache
from .release_lineage import verify_author_release_lineage
from .release_official_text_cache_binding import verify_binding as verify_official_binding
from .release_paired_binding import verify_release_paired_binding
from .runtime_utils import PROJECT_ROOT


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1"
).resolve()
DEFAULT_HISTORICAL_SELECTION = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/p_mode_dev_v1_retry1/p_mode_selection.json"
).resolve()
DEFAULT_PRIMARY_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1"
).resolve()
DEFAULT_PRIMARY_SUMMARY = (
    DEFAULT_PRIMARY_ROOT
    / "online_rollouts_author_stock_seed42_v1/aggregate/summary.json"
).resolve()
DEFAULT_PRIMARY_COMPLETION_AUDIT = (
    DEFAULT_PRIMARY_ROOT
    / "online_rollouts_author_stock_seed42_v1/aggregate/completion_audit.json"
).resolve()
DEFAULT_EVALUATOR_SOURCE = (
    PROJECT_ROOT / "third_party/RoboTwin/script/eval_policy.py"
).resolve()

PILOT_TRAINING_SEED = 1
FORMAL_TRAINING_SEEDS = (1, 2, 3)
PILOT_SIMULATOR_SEED = 53
SMOKE_SIMULATOR_SEED = 54
CONFIRMATORY_SIMULATOR_SEED = 59
PILOT_MAX_STEPS = 1800
SMOKE_MAX_STEPS = 3
PILOT_RANDOM_DELTA_MIN = 0.03
PILOT_CLEAN_DELTA_MIN = -0.03


class Pv2FollowupMaterializationError(ValueError):
    """The P-v2 mechanism study cannot be materialized without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2FollowupMaterializationError(message)


def _identity(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }


def action_dit_release_payload_audit(
    checkpoint: str | Path,
    *,
    expected_checkpoint_sha256: str = AUTHOR_RELEASE_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Hash the exact ActionDiT tensor payload used by both treatment arms."""

    path = Path(checkpoint).expanduser().resolve()
    _require(path.is_file(), f"release checkpoint is missing: {path}")
    checkpoint_sha = _file_sha256(path)
    _require(
        checkpoint_sha == expected_checkpoint_sha256,
        "release checkpoint differs before ActionDiT payload hashing",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), "release checkpoint root is not a mapping")
    mot = payload.get("mot")
    _require(isinstance(mot, Mapping), "release checkpoint lacks mot state")
    names = sorted(
        name
        for name, tensor in mot.items()
        if str(name).startswith("mixtures.action.") and isinstance(tensor, torch.Tensor)
    )
    _require(len(names) == 824, f"expected 824 ActionDiT tensors, found {len(names)}")
    digest = hashlib.sha256()
    digest.update(b"fastwam-actiondit-release-tensor-payload-v1\0")
    tensor_bytes = 0
    for name in names:
        tensor = mot[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        tensor_bytes += len(raw)
    return {
        "schema_version": 1,
        "kind": "policy_actiondit_release_initialization_audit",
        "status": "PASS",
        "checkpoint_path": str(path),
        "checkpoint_size_bytes": int(path.stat().st_size),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(payload.get("step", -1)),
        "checkpoint_declared_torch_dtype": str(payload.get("torch_dtype", "")),
        "payload_prefix": "mot.mixtures.action.",
        "hash_schema": "fastwam-actiondit-release-tensor-payload-v1",
        "tensor_count": len(names),
        "tensor_bytes": tensor_bytes,
        "action_dit_tensor_sha256": digest.hexdigest(),
        "strict_load_contract": {
            "mot_strict": True,
            "expected_missing_keys": 0,
            "expected_unexpected_keys": 0,
        },
        "shared_initialization_claim": (
            "C1-Pv2 and C3-Pv2 load this identical immutable ActionDiT payload "
            "before any optimizer step"
        ),
    }


def _pair_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    for key in ("experiment_id", "control", "output_dir"):
        value.pop(key, None)
    value["paired"].pop("contrastive_supervision", None)
    value["supervision"].pop("paired_contrastive", None)
    value["loss"].pop("lambda_contrastive", None)
    return value


def validate_followup_pair(c1: Mapping[str, Any], c3: Mapping[str, Any]) -> dict[str, Any]:
    validate_c1_c3_pair(c1, c3)
    for config, control, coefficient in (
        (c1, "c1_architecture_only", 0.0),
        (c3, "c3_ours", 0.1),
    ):
        _require(config.get("control") == control, f"{control} label differs")
        _require(config.get("stage") == PV2_FOLLOWUP_STAGE, "follow-up stage differs")
        _require(config.get("study_role") == PV2_FOLLOWUP_ROLE, "study role differs")
        _require(config.get("formal") is False, "post-hoc pilot must be non-formal")
        _require(config["policy"]["regime"] == "p_v2", "follow-up must use P-v2")
        _require(
            config["policy"]["freeze"]["action_dit"] is False,
            "ActionDiT must train in both controls",
        )
        _require(
            float(config["loss"]["lambda_contrastive"]) == coefficient,
            f"{control} contrastive coefficient differs",
        )
        _require(config["training"]["seed"] == PILOT_TRAINING_SEED, "pilot seed differs")
        _require(config["training"]["max_steps"] == PILOT_MAX_STEPS, "step budget differs")
        _require(
            config["evaluation"]["episodes_per_task"] == DEV_EPISODES_PER_CELL,
            "pilot episode budget differs",
        )
    _require(
        _pair_projection(c1) == _pair_projection(c3),
        "C1-Pv2/C3-Pv2 differ outside the contrastive treatment allowlist",
    )
    return {
        "status": "PASS",
        "policy_regime": "p_v2",
        "training_seed": PILOT_TRAINING_SEED,
        "max_steps": PILOT_MAX_STEPS,
        "only_permitted_difference": "contrastive_coefficient_and_gradient",
        "shared_recipe_sha256": canonical_sha256(_pair_projection(c1)),
    }


def build_followup_pair(
    *,
    template: Mapping[str, Any],
    output_root: Path,
    mechanism_protocol_manifest: Path,
    mechanism_protocol_manifest_sha256: str,
    action_dit_initialization_audit_sha256: str,
    historical_selection_manifest: Path,
    historical_selection_sha256: str,
    release_paired_binding_manifest: Path,
    release_paired_binding_sha256: str,
    paired_text_cache: Path,
    paired_text_cache_sha256: str,
    paired_cache: Path,
    paired_cache_sha256: str,
    official_text_cache: Path,
    official_text_cache_binding_manifest: Path,
    official_text_cache_binding_manifest_sha256: str,
    seed_bank_manifest: Path,
    seed_bank_manifest_sha256: str,
    seed_bank_id: str,
    max_steps: int = PILOT_MAX_STEPS,
    num_workers: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(max_steps == PILOT_MAX_STEPS, "pilot max_steps is locked to 1800")
    resolved: list[dict[str, Any]] = []
    for control, short, coefficient in (
        ("c1_architecture_only", "c1", 0.0),
        ("c3_ours", "c3", 0.1),
    ):
        value = OmegaConf.to_container(OmegaConf.create(dict(template)), resolve=True)
        _require(isinstance(value, dict), "P-v2 template root must be a mapping")
        value.pop("selection_role", None)
        value.update(
            {
                "experiment_id": f"pv2_actiondit_followup_seed1_{short}_v1",
                "stage": PV2_FOLLOWUP_STAGE,
                "study_role": PV2_FOLLOWUP_ROLE,
                "formal": False,
                "control": control,
                "output_dir": str((output_root / "runs/seed_1" / short).resolve()),
                "mechanism_protocol_manifest": str(
                    mechanism_protocol_manifest.resolve()
                ),
                "p_mode_selection_manifest": str(
                    historical_selection_manifest.resolve()
                ),
                "release_paired_binding_manifest": str(
                    release_paired_binding_manifest.resolve()
                ),
            }
        )
        value["execution"] = {
            "runner": "policy_content_adapter",
            "runnable": True,
            "fail_closed": False,
            "long_formal_training": False,
        }
        value["artifacts"].update(
            {
                "base_checkpoint_sha256": AUTHOR_RELEASE_CHECKPOINT_SHA256,
                "dataset_stats_sha256": AUTHOR_RELEASE_DATASET_STATS_SHA256,
                "official_task_manifest_sha256": AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
                "base_lineage_manifest_sha256": AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
                "release_paired_binding_manifest_sha256": release_paired_binding_sha256,
                "head_init_sha256": None,
                "paired_text_cache_sha256": paired_text_cache_sha256,
                "paired_cache_sha256": paired_cache_sha256,
                "p_mode_selection_manifest_sha256": historical_selection_sha256,
                "simulator_seed_bank_manifest_sha256": seed_bank_manifest_sha256,
                "official_text_cache_binding_manifest_sha256": (
                    official_text_cache_binding_manifest_sha256
                ),
                "mechanism_protocol_manifest_sha256": (
                    mechanism_protocol_manifest_sha256
                ),
                "action_dit_initialization_audit_sha256": (
                    action_dit_initialization_audit_sha256
                ),
                "formal_protocol_lock_manifest_sha256": None,
            }
        )
        value["policy"].update(
            {
                "regime": "p_v2",
                "head_init_mode": "random",
                "head_init_seed": PILOT_TRAINING_SEED,
                "adapter_init_seed": PILOT_TRAINING_SEED,
                "head_init": None,
            }
        )
        value["policy"]["freeze"].update(
            {
                "vae": True,
                "video_backbone": True,
                "action_dit": False,
                "content_head": False,
                "action_adapter": False,
            }
        )
        value["official"].update(
            {
                "sampling_mode": "all_frames",
                "text_cache_dir": str(official_text_cache.resolve()),
                "text_cache_binding_manifest": str(
                    official_text_cache_binding_manifest.resolve()
                ),
                "on_the_fly_text_smoke": False,
                "domain_verified": True,
            }
        )
        value["paired"].update(
            {
                "text_cache_dir": str(paired_text_cache.resolve()),
                "cache": str(paired_cache.resolve()),
                "contrastive_supervision": coefficient > 0.0,
            }
        )
        value["supervision"]["paired_contrastive"] = coefficient > 0.0
        value["loss"]["lambda_contrastive"] = coefficient
        value["optimizer"].update(
            {
                "head_adapter_lr": 1.0e-4,
                "action_dit_lr": 1.0e-5,
                "lr_scheduler": "constant",
                "trainable_parameter_dtype": "fp32",
            }
        )
        value["training"].update(
            {
                "seed": PILOT_TRAINING_SEED,
                "max_steps": max_steps,
                "official_batch_size": 1,
                "paired_groups_per_batch": 2,
                "world_size": 1,
                "gradient_accumulation_steps": 1,
                "effective_official_global_batch": 1,
                "effective_paired_groups_per_step": 2,
                "num_workers": int(num_workers),
                "mixed_precision": "bf16",
                "model_dtype": "bf16",
                "save_optimizer": False,
            }
        )
        value["evaluation"].update(
            {
                "simulator_seed_bank_manifest": str(seed_bank_manifest.resolve()),
                "simulator_seed_bank_id": seed_bank_id,
                "simulator_seed_bank_purpose": "dev_selection",
                "episodes_per_task": DEV_EPISODES_PER_CELL,
            }
        )
        resolved.append(value)
    c1, c3 = resolved
    validate_followup_pair(c1, c3)
    return c1, c3


def build_smoke_pair_from_pilot(
    c1: Mapping[str, Any],
    c3: Mapping[str, Any],
    *,
    output_root: Path,
    seed_bank_manifest: Path,
    seed_bank_manifest_sha256: str,
    seed_bank_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive a three-step P-v2 engineering gate without weakening pilot locks."""

    resolved: list[dict[str, Any]] = []
    for source, short in ((c1, "c1"), (c3, "c3")):
        value = copy.deepcopy(dict(source))
        value["experiment_id"] = f"pv2_actiondit_followup_{short}_smoke_v1"
        value["stage"] = "smoke"
        value["study_role"] = "engineering_gate_for_post_hoc_actiondit_mechanism"
        value["output_dir"] = str((output_root / "smoke/runs" / short).resolve())
        value["p_mode_selection_manifest"] = None
        value["artifacts"]["p_mode_selection_manifest_sha256"] = None
        value["artifacts"]["simulator_seed_bank_manifest_sha256"] = (
            seed_bank_manifest_sha256
        )
        value["training"].update(
            {
                "max_steps": SMOKE_MAX_STEPS,
                "num_workers": 0,
                "save_optimizer": False,
            }
        )
        value["evaluation"].update(
            {
                "simulator_seed_bank_manifest": str(seed_bank_manifest.resolve()),
                "simulator_seed_bank_id": seed_bank_id,
                "simulator_seed_bank_purpose": "engineering_smoke",
                "episodes_per_task": 1,
            }
        )
        resolved.append(value)
    smoke_c1, smoke_c3 = resolved
    validate_c1_c3_pair(smoke_c1, smoke_c3)
    for config in resolved:
        _require(config["policy"]["regime"] == "p_v2", "smoke regime changed")
        _require(
            config["policy"]["freeze"]["action_dit"] is False,
            "smoke must train ActionDiT",
        )
        _require(config["training"]["max_steps"] == 3, "smoke must run 3 steps")
    return smoke_c1, smoke_c3


def materialize(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    lineage_manifest: str | Path = DEFAULT_LINEAGE,
    release_paired_binding_manifest: str | Path = DEFAULT_BINDING,
    paired_text_cache: str | Path = DEFAULT_TEXT_CACHE,
    paired_cache: str | Path = DEFAULT_CACHE,
    paired_cache_audit: str | Path = DEFAULT_CACHE_AUDIT,
    evaluator_source: str | Path = DEFAULT_EVALUATOR_SOURCE,
    official_text_cache: str | Path = DEFAULT_OFFICIAL_TEXT_CACHE,
    official_text_cache_binding: str | Path = DEFAULT_OFFICIAL_TEXT_BINDING,
    historical_selection: str | Path = DEFAULT_HISTORICAL_SELECTION,
    primary_summary: str | Path = DEFAULT_PRIMARY_SUMMARY,
    primary_completion_audit: str | Path = DEFAULT_PRIMARY_COMPLETION_AUDIT,
) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    _require(not destination.exists(), f"refusing to reuse output root: {destination}")
    lineage_path = Path(lineage_manifest).expanduser().resolve()
    binding_path = Path(release_paired_binding_manifest).expanduser().resolve()
    text_cache_path = Path(paired_text_cache).expanduser().resolve()
    cache_path = Path(paired_cache).expanduser().resolve()
    cache_audit_path = Path(paired_cache_audit).expanduser().resolve()
    evaluator_path = Path(evaluator_source).expanduser().resolve()
    official_text_path = Path(official_text_cache).expanduser().resolve()
    official_binding_path = Path(official_text_cache_binding).expanduser().resolve()
    selection_path = Path(historical_selection).expanduser().resolve()
    primary_summary_path = Path(primary_summary).expanduser().resolve()
    primary_audit_path = Path(primary_completion_audit).expanduser().resolve()

    template = load_config(CONFIG_DIR / "p_v2_dev_pilot.yaml")
    checkpoint = Path(template["base_checkpoint"]).expanduser().resolve()
    stats = Path(template["official"]["dataset_stats"]).expanduser().resolve()
    official_manifest = Path(
        template["official"]["canonical_task_manifest"]
    ).expanduser().resolve()
    lineage = verify_author_release_lineage(
        lineage_path,
        checkpoint_path=checkpoint,
        dataset_stats_path=stats,
        official_manifest_path=official_manifest,
        expected_manifest_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
    )
    binding = verify_release_paired_binding(
        binding_path,
        expected_sha256=template["artifacts"][
            "release_paired_binding_manifest_sha256"
        ],
    )
    paired_text = verify_release_paired_text_cache(
        text_cache_path,
        expected_base_lineage_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        expected_release_paired_binding_sha256=binding[
            "binding_manifest_identity"
        ]["sha256"],
    )
    cache_identity = artifact_identity(cache_path)
    cache_audit = _load_json(cache_audit_path, "Layer-16 cache audit")
    _require(cache_audit.get("status") == "PASS", "Layer-16 cache audit is not PASS")
    _require(
        cache_audit.get("cache", {}).get("sha256") == cache_identity["sha256"],
        "Layer-16 cache differs from its audit",
    )
    _require(
        cache_audit.get("layer16_shape") == [2880, 120, 3072],
        "Layer-16 cache shape changed",
    )
    official_binding_sha = _file_sha256(official_binding_path)
    official_binding = verify_official_binding(
        official_binding_path,
        expected_sha256=official_binding_sha,
        expected_base_lineage_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        expected_cache_dir=official_text_path,
    )

    selection_raw = _load_json(selection_path, "historical P-mode selection")
    selection = validate_selection_manifest_payload(selection_raw)
    _require(selection["winner"] == "p_v1", "historical P-mode winner is not P-v1")
    selection_identity = _identity(selection_path)
    summary = _load_json(primary_summary_path, "P-v1 primary summary")
    completion = _load_json(primary_audit_path, "P-v1 completion audit")
    _require(
        summary.get("status") == "PASS" and summary.get("record_count") == 36,
        "P-v1 primary summary is incomplete",
    )
    _require(
        completion.get("status") == "PASS"
        and completion.get("online_rollout_complete") is True
        and completion.get("record_count") == 36,
        "P-v1 primary completion audit is incomplete",
    )
    primary_summary_identity = _identity(primary_summary_path)
    primary_completion_identity = _identity(primary_audit_path)

    seed_bank = build_seed_bank_descriptor(
        simulator_seed=PILOT_SIMULATOR_SEED,
        episodes_per_cell=DEV_EPISODES_PER_CELL,
        evaluator_source=evaluator_path,
        purpose="dev_selection",
    )
    validate_seed_bank_descriptor(seed_bank, expected_purpose="dev_selection")
    _require(
        set(seed_bank["members"]).isdisjoint(selection["dev_seed_bank"]["members"]),
        "new seed53 pilot bank overlaps old seed23 dev bank",
    )
    _require(
        4_300_000 not in set(seed_bank["members"]),
        "new seed53 pilot bank overlaps the author-stock seed42 start",
    )
    seed_bank_path = (destination / "manifests/dev_seed53_bank.json").resolve()
    seed_bank_bytes = (json.dumps(seed_bank, indent=2, sort_keys=True) + "\n").encode()
    seed_bank_sha = hashlib.sha256(seed_bank_bytes).hexdigest()
    smoke_seed_bank = build_seed_bank_descriptor(
        simulator_seed=SMOKE_SIMULATOR_SEED,
        episodes_per_cell=1,
        evaluator_source=evaluator_path,
        purpose="engineering_smoke",
    )
    validate_seed_bank_descriptor(
        smoke_seed_bank, expected_purpose="engineering_smoke"
    )
    smoke_seed_bank_path = (
        destination / "manifests/engineering_smoke_seed54_bank.json"
    ).resolve()
    smoke_seed_bank_bytes = (
        json.dumps(smoke_seed_bank, indent=2, sort_keys=True) + "\n"
    ).encode()
    smoke_seed_bank_sha = hashlib.sha256(smoke_seed_bank_bytes).hexdigest()

    action_init = action_dit_release_payload_audit(checkpoint)
    action_init_path = (
        destination / "manifests/action_dit_initialization_audit.json"
    ).resolve()
    action_init_bytes = (
        json.dumps(action_init, indent=2, sort_keys=True) + "\n"
    ).encode()
    action_init_sha = hashlib.sha256(action_init_bytes).hexdigest()

    common_ancestry = {
        "base_checkpoint_sha256": AUTHOR_RELEASE_CHECKPOINT_SHA256,
        "dataset_stats_sha256": AUTHOR_RELEASE_DATASET_STATS_SHA256,
        "base_lineage_manifest_sha256": AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        "release_paired_binding_manifest_sha256": binding[
            "binding_manifest_identity"
        ]["sha256"],
        "official_task_manifest_sha256": AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
        "paired_action_manifest_sha256": template["artifacts"][
            "paired_action_manifest_sha256"
        ],
        "paired_state_bank_sha256": template["artifacts"][
            "paired_state_bank_sha256"
        ],
        "paired_text_cache_sha256": paired_text["directory_identity"]["sha256"],
        "paired_cache_sha256": cache_identity["sha256"],
    }
    source_paths = {
        "config_audit.py": Path(__file__).with_name("config_audit.py"),
        "materialize_pv2_actiondit_followup.py": Path(__file__).resolve(),
        "train.py": Path(__file__).with_name("train.py"),
        "losses.py": Path(__file__).with_name("losses.py"),
    }
    protocol = {
        "schema_version": PV2_FOLLOWUP_PROTOCOL_SCHEMA_VERSION,
        "kind": PV2_FOLLOWUP_PROTOCOL_KIND,
        "status": "PASS",
        "study_classification": {
            "role": PV2_FOLLOWUP_ROLE,
            "post_hoc_after_primary_results": True,
            "primary_experiment_remains_unchanged": True,
            "may_replace_primary_conclusion": False,
            "interpretation": (
                "mechanism study of contrastive supervision when ActionDiT is trainable"
            ),
        },
        "primary_pv1_result": {
            "root": str(DEFAULT_PRIMARY_ROOT),
            "summary": primary_summary_identity,
            "completion_audit": primary_completion_identity,
            "reported_macro_delta_percentage_points": {
                "clean": -0.444444444444451,
                "official_random": 0.22222222222221997,
            },
        },
        "historical_p_mode_selection": {
            **selection_identity,
            "winner": "p_v1",
            "use": "historical_context_not_treatment_selection",
        },
        "common_ancestry": common_ancestry,
        "action_dit_initialization_audit": {
            "path": str(action_init_path),
            "size_bytes": len(action_init_bytes),
            "sha256": action_init_sha,
            "action_dit_tensor_sha256": action_init[
                "action_dit_tensor_sha256"
            ],
        },
        "locked_training": {
            "policy_regime": "p_v2",
            "action_dit_trainable": True,
            "video_dit_frozen": True,
            "vae_frozen": True,
            "t5_frozen": True,
            "training_seeds": list(FORMAL_TRAINING_SEEDS),
            "pilot_training_seed": PILOT_TRAINING_SEED,
            "max_steps": PILOT_MAX_STEPS,
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
        },
        "locked_treatment": {
            "c1_lambda_contrastive": 0.0,
            "c1_contrastive_gradient": False,
            "c3_lambda_contrastive": 0.1,
            "c3_contrastive_gradient": True,
            "only_permitted_difference": "contrastive_coefficient_and_gradient",
        },
        "pilot_gate": {
            "simulator_seed": PILOT_SIMULATOR_SEED,
            "seed_bank_id": seed_bank["simulator_seed_bank_id"],
            "seed_bank_manifest_path": str(seed_bank_path),
            "seed_bank_manifest_sha256": seed_bank_sha,
            "episodes_per_task_domain": DEV_EPISODES_PER_CELL,
            "official_random_macro_delta_min": PILOT_RANDOM_DELTA_MIN,
            "clean_macro_delta_min": PILOT_CLEAN_DELTA_MIN,
            "both_conditions_required": True,
            "stop_expansion_on_failure": True,
            "result_driven_tuning_forbidden": True,
        },
        "engineering_smoke": {
            "simulator_seed": SMOKE_SIMULATOR_SEED,
            "seed_bank_id": smoke_seed_bank["simulator_seed_bank_id"],
            "seed_bank_manifest_path": str(smoke_seed_bank_path),
            "seed_bank_manifest_sha256": smoke_seed_bank_sha,
            "steps_per_control": SMOKE_MAX_STEPS,
            "scientific_result": False,
        },
        "confirmatory_intent": {
            "only_if_pilot_passes": True,
            "simulator_seed": CONFIRMATORY_SIMULATOR_SEED,
            "episodes_per_task_domain": 100,
            "training_seeds": list(FORMAL_TRAINING_SEEDS),
            "unopened_before_pilot_decision": True,
            "disjoint_from_simulator_seeds": [23, 42, PILOT_SIMULATOR_SEED],
        },
        "source_sha256": {
            name: _file_sha256(path) for name, path in source_paths.items()
        },
    }
    protocol_path = (destination / "manifests/mechanism_protocol.json").resolve()
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()

    c1, c3 = build_followup_pair(
        template=template,
        output_root=destination,
        mechanism_protocol_manifest=protocol_path,
        mechanism_protocol_manifest_sha256=protocol_sha,
        action_dit_initialization_audit_sha256=action_init_sha,
        historical_selection_manifest=selection_path,
        historical_selection_sha256=selection_identity["sha256"],
        release_paired_binding_manifest=binding_path,
        release_paired_binding_sha256=binding["binding_manifest_identity"]["sha256"],
        paired_text_cache=text_cache_path,
        paired_text_cache_sha256=paired_text["directory_identity"]["sha256"],
        paired_cache=cache_path,
        paired_cache_sha256=cache_identity["sha256"],
        official_text_cache=official_text_path,
        official_text_cache_binding_manifest=official_binding_path,
        official_text_cache_binding_manifest_sha256=official_binding_sha,
        seed_bank_manifest=seed_bank_path,
        seed_bank_manifest_sha256=seed_bank_sha,
        seed_bank_id=seed_bank["simulator_seed_bank_id"],
    )
    smoke_c1, smoke_c3 = build_smoke_pair_from_pilot(
        c1,
        c3,
        output_root=destination,
        seed_bank_manifest=smoke_seed_bank_path,
        seed_bank_manifest_sha256=smoke_seed_bank_sha,
        seed_bank_id=smoke_seed_bank["simulator_seed_bank_id"],
    )

    _write_new_bytes(seed_bank_path, seed_bank_bytes)
    _write_new_bytes(smoke_seed_bank_path, smoke_seed_bank_bytes)
    _write_new_bytes(action_init_path, action_init_bytes)
    _write_new_bytes(protocol_path, protocol_bytes)
    c1_path = (destination / "configs/seed_1/c1.yaml").resolve()
    c3_path = (destination / "configs/seed_1/c3.yaml").resolve()
    _write_new_yaml(c1_path, c1)
    _write_new_yaml(c3_path, c3)
    smoke_c1_path = (destination / "smoke/configs/c1.yaml").resolve()
    smoke_c3_path = (destination / "smoke/configs/c3.yaml").resolve()
    _write_new_yaml(smoke_c1_path, smoke_c1)
    _write_new_yaml(smoke_c3_path, smoke_c3)
    emitted_c1 = load_config(c1_path)
    emitted_c3 = load_config(c3_path)
    validate_execution_ready(emitted_c1)
    validate_execution_ready(emitted_c3)
    fairness = validate_followup_pair(emitted_c1, emitted_c3)
    emitted_smoke_c1 = load_config(smoke_c1_path)
    emitted_smoke_c3 = load_config(smoke_c3_path)
    validate_execution_ready(emitted_smoke_c1)
    validate_execution_ready(emitted_smoke_c3)
    smoke_fairness = validate_c1_c3_pair(emitted_smoke_c1, emitted_smoke_c3)

    manifest = {
        "schema_version": 1,
        "kind": "policy_pv2_actiondit_followup_materialization",
        "status": "PASS",
        "scientific_results_present": False,
        "gpu_training_started": False,
        "online_rollout_started": False,
        "primary_pv1_modified": False,
        "configs": {
            "pilot": {"c1": _identity(c1_path), "c3": _identity(c3_path)},
            "smoke": {
                "c1": _identity(smoke_c1_path),
                "c3": _identity(smoke_c3_path),
            },
        },
        "protocol": _identity(protocol_path),
        "action_dit_initialization_audit": _identity(action_init_path),
        "pilot_seed_bank": _identity(seed_bank_path),
        "smoke_seed_bank": _identity(smoke_seed_bank_path),
        "fairness": fairness,
        "smoke_fairness": smoke_fairness,
        "verified_artifacts": {
            "base_lineage_manifest_sha256": lineage["manifest_identity"]["sha256"],
            "release_paired_binding_manifest_sha256": binding[
                "binding_manifest_identity"
            ]["sha256"],
            "paired_text_cache_sha256": paired_text["directory_identity"]["sha256"],
            "paired_cache_sha256": cache_identity["sha256"],
            "paired_cache_audit_sha256": _file_sha256(cache_audit_path),
            "official_text_cache_binding_manifest_sha256": official_binding_sha,
            "official_text_cache_aggregate_payload_sha256": official_binding["cache"][
                "aggregate_payload_sha256"
            ],
            "historical_p_mode_selection_sha256": selection_identity["sha256"],
            "primary_pv1_summary_sha256": primary_summary_identity["sha256"],
            "primary_pv1_completion_audit_sha256": primary_completion_identity[
                "sha256"
            ],
        },
        "resource_plan": {
            "smoke": "single GPU, C1 then C3, at least 3 steps each",
            "pilot_training": "GPU0 C1 and GPU1 C3, independent world_size=1",
            "pilot_rollout": "seed53, 20 episodes per task/domain/control",
            "confirmatory": "forbidden until pilot gate PASS",
        },
    }
    manifest_path = (destination / "materialization_manifest.json").resolve()
    _write_new_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--lineage-manifest", default=str(DEFAULT_LINEAGE))
    parser.add_argument(
        "--release-paired-binding-manifest", default=str(DEFAULT_BINDING)
    )
    parser.add_argument("--paired-text-cache", default=str(DEFAULT_TEXT_CACHE))
    parser.add_argument("--paired-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--paired-cache-audit", default=str(DEFAULT_CACHE_AUDIT))
    parser.add_argument("--evaluator-source", default=str(DEFAULT_EVALUATOR_SOURCE))
    parser.add_argument("--official-text-cache", default=str(DEFAULT_OFFICIAL_TEXT_CACHE))
    parser.add_argument(
        "--official-text-cache-binding", default=str(DEFAULT_OFFICIAL_TEXT_BINDING)
    )
    parser.add_argument(
        "--historical-selection", default=str(DEFAULT_HISTORICAL_SELECTION)
    )
    parser.add_argument("--primary-summary", default=str(DEFAULT_PRIMARY_SUMMARY))
    parser.add_argument(
        "--primary-completion-audit",
        default=str(DEFAULT_PRIMARY_COMPLETION_AUDIT),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        output_root=args.output_root,
        lineage_manifest=args.lineage_manifest,
        release_paired_binding_manifest=args.release_paired_binding_manifest,
        paired_text_cache=args.paired_text_cache,
        paired_cache=args.paired_cache,
        paired_cache_audit=args.paired_cache_audit,
        evaluator_source=args.evaluator_source,
        official_text_cache=args.official_text_cache,
        official_text_cache_binding=args.official_text_cache_binding,
        historical_selection=args.historical_selection,
        primary_summary=args.primary_summary,
        primary_completion_audit=args.primary_completion_audit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATORY_SIMULATOR_SEED",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAL_TRAINING_SEEDS",
    "PILOT_CLEAN_DELTA_MIN",
    "PILOT_MAX_STEPS",
    "PILOT_RANDOM_DELTA_MIN",
    "PILOT_SIMULATOR_SEED",
    "PILOT_TRAINING_SEED",
    "SMOKE_MAX_STEPS",
    "SMOKE_SIMULATOR_SEED",
    "Pv2FollowupMaterializationError",
    "action_dit_release_payload_audit",
    "build_followup_pair",
    "build_smoke_pair_from_pilot",
    "materialize",
    "validate_followup_pair",
]
