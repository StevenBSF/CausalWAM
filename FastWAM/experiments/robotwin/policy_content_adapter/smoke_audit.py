"""Fail-closed post-run audit for the sequential three-task policy smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_audit import TASKS
from .model import (
    EXPECTED_ACTION_DIT_PARAMETER_COUNT,
    EXPECTED_ADAPTER_PARAMETER_COUNT,
    EXPECTED_HEAD_PARAMETER_COUNT,
)
from .protocol import (
    POLICY_ACTION_DIM,
    POLICY_ACTION_STEPS,
    POLICY_CAMERA_COUNT,
    POLICY_CAMERA_NAMES,
    POLICY_NATIVE_FPS,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_STATE_BANK_SAMPLING_ALGORITHM,
    POLICY_STATE_BANK_SEED,
    POLICY_STATES_PER_TRAJECTORY,
    POLICY_VARIANTS,
    POLICY_VIEW_COUNT,
)


EXPECTED_STEPS = 3
REQUIRED_FILES = (
    "requested_config.json",
    "run_config.json",
    "artifact_identities.json",
    "official_subset_audit.json",
    "base_lineage_audit.json",
    "release_paired_binding_audit.json",
    "release_paired_binding_crosscheck.json",
    "identity_audit.json",
    "data_provenance_audit.json",
    "data_distribution_audit.json",
    "gradient_audit.json",
    "parameter_update_audit.json",
    "train_log.csv",
    "training_summary.json",
    "checkpoint.pt",
    "rollout_load_execute.json",
)


class SmokeAuditError(RuntimeError):
    """A completed-looking smoke directory violates the protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeAuditError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SmokeAuditError(f"cannot parse JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact root must be a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    if positive:
        _require(result > 0.0, f"{label} must be positive")
    return result


def _nonzero(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result != 0.0, f"{label} must be nonzero")
    return result


def _shape(value: Any, label: str, tail: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SmokeAuditError(f"{label} is not a JSON shape") from exc
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be a shape sequence",
    )
    shape = tuple(int(item) for item in value)
    _require(len(shape) == len(tail) + 1, f"{label} rank changed: {shape}")
    _require(shape[0] > 0 and shape[1:] == tail, f"{label} changed: {shape}")
    return shape


def _sha256(value: Any, label: str) -> str:
    digest = str(value)
    _require(
        len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        f"{label} is not a lowercase SHA-256",
    )
    return digest


def _audit_gradient_steps(
    gradient: Mapping[str, Any], *, regime: str
) -> list[Mapping[str, Any]]:
    _require(gradient.get("status") == "PASS", "gradient audit status is not PASS")
    _require(gradient.get("regime") == regime, "gradient audit regime mismatch")
    steps = gradient.get("steps")
    _require(
        isinstance(steps, list) and len(steps) == EXPECTED_STEPS,
        f"gradient audit must contain exactly {EXPECTED_STEPS} steps",
    )
    for expected_step, raw_step in enumerate(steps, start=1):
        step = _mapping(raw_step, f"gradient step {expected_step}")
        _require(int(step.get("step", -1)) == expected_step, "gradient step order changed")
        _nonzero(step.get("gate_raw_after_step"), "gate_raw_after_step")
        combined = _mapping(step.get("combined"), "combined gradient report")
        for module in ("content_head", "adapter"):
            report = _mapping(combined.get(module), f"{module} gradient report")
            _require(report.get("all_finite") is True, f"{module} gradients are non-finite")
            _finite(report.get("gradient_norm"), f"{module} gradient norm", positive=True)
        for module in ("video_backbone", "vae"):
            report = _mapping(combined.get(module), f"{module} gradient report")
            _require(
                int(report.get("gradient_tensors", -1)) == 0
                and _finite(report.get("gradient_norm"), f"{module} gradient norm") == 0.0,
                f"frozen {module} received gradients",
            )
        action = _mapping(combined.get("action_dit"), "ActionDiT gradient report")
        action_norm = _finite(action.get("gradient_norm"), "ActionDiT gradient norm")
        if regime == "p_v1":
            _require(action_norm == 0.0, "P-v1 ActionDiT received a gradient")
            _require(int(action.get("gradient_tensors", -1)) == 0, "P-v1 ActionDiT has gradient tensors")
        else:
            _require(action_norm > 0.0, "P-v2 ActionDiT has no gradient")
        probe = _mapping(step.get("action_only_probe"), "action-only gradient probe")
        _require(probe.get("all_finite") is True, "action-only gradient probe is non-finite")
        _finite(probe.get("gate_grad_norm"), "action-only gate gradient", positive=True)
        if expected_step >= 2:
            _finite(
                step.get("action_only_official_content_token_grad_norm"),
                "action-only official Zc gradient",
                positive=True,
            )
            _finite(probe.get("head_grad_norm"), "action-only head gradient", positive=True)
            _finite(
                probe.get("adapter_attention_grad_norm"),
                "action-only GCA gradient",
                positive=True,
            )
    return [_mapping(item, "gradient step") for item in steps]


def _audit_updates(update: Mapping[str, Any], *, regime: str) -> None:
    head_adapter = _mapping(update.get("head_and_adapter"), "head/adapter update audit")
    deltas = _mapping(
        head_adapter.get("max_abs_delta_by_module"), "head/adapter update deltas"
    )
    _finite(deltas.get("content_head"), "Content Head update", positive=True)
    _finite(deltas.get("adapter"), "adapter update", positive=True)
    _require(head_adapter.get("all_finite") is True, "head/adapter updates are non-finite")

    action = _mapping(update.get("action_dit"), "ActionDiT update audit")
    if regime == "p_v1":
        _require(action.get("changed") is False, "P-v1 ActionDiT parameters changed")
        return
    _require(action.get("changed") is True, "P-v2 ActionDiT did not change")
    _require(
        _finite(action.get("changed_fraction"), "P-v2 sampled changed fraction") >= 0.5,
        "P-v2 sampled changed fraction is below 0.5",
    )
    _require(
        int(action.get("required_changed_strata", -1)) >= 6,
        "P-v2 changed fewer than six required ActionDiT strata",
    )
    visibility = _mapping(
        action.get("bf16_deployment_category_visibility"),
        "P-v2 BF16 deployment visibility",
    )
    _require(
        set(visibility) == {"early", "mid", "late", "head"}
        and all(value is True for value in visibility.values()),
        "P-v2 update is not BF16-visible across early/mid/late/head",
    )
    adam = _mapping(action.get("optimizer_exp_avg"), "P-v2 optimizer state audit")
    _require(adam.get("all_finite") is True, "P-v2 optimizer state is non-finite")
    _require(
        _finite(adam.get("nonzero_fraction"), "P-v2 Adam exp_avg nonzero fraction")
        >= 0.5,
        "P-v2 Adam exp_avg nonzero fraction is below 0.5",
    )


def audit_smoke_run(run_dir: str | Path, regime: str) -> dict[str, Any]:
    """Validate all ten requested smoke goals from one completed run directory."""

    normalized_regime = str(regime).lower().replace("-", "_")
    _require(normalized_regime in {"p_v1", "p_v2"}, "regime must be p_v1 or p_v2")
    root = Path(run_dir).expanduser().resolve()
    _require(root.is_dir(), f"smoke run directory does not exist: {root}")
    for relative in REQUIRED_FILES:
        path = root / relative
        _require(path.is_file() and path.stat().st_size > 0, f"missing/empty artifact: {path}")

    config = _load_json(root / "run_config.json")
    summary = _load_json(root / "training_summary.json")
    identity = _load_json(root / "identity_audit.json")
    gradient = _load_json(root / "gradient_audit.json")
    update = _load_json(root / "parameter_update_audit.json")
    provenance = _load_json(root / "data_provenance_audit.json")
    distribution = _load_json(root / "data_distribution_audit.json")
    official = _load_json(root / "official_subset_audit.json")
    base_lineage = _load_json(root / "base_lineage_audit.json")
    release_binding = _load_json(root / "release_paired_binding_audit.json")
    release_binding_crosscheck = _load_json(
        root / "release_paired_binding_crosscheck.json"
    )
    artifacts = _load_json(root / "artifact_identities.json")
    rollout = _load_json(root / "rollout_load_execute.json")

    _require(config.get("formal") is False, "smoke run was marked formal")
    _require(int(config.get("schema_version", -1)) == 3, "smoke is not Policy v2 schema 3")
    _require(config.get("formal_training_auto_started") is False, "formal training auto-started")
    _require(config.get("control") == normalized_regime, "run config regime mismatch")
    _require(
        config.get("selection_role") == "engineering_method_smoke",
        "release-base engineering smoke role changed",
    )
    _require(tuple(config.get("tasks", ())) == TASKS, "run config task order changed")
    policy = _mapping(config.get("policy"), "policy config")
    _require(policy.get("head_init_mode") == "random", "new smoke must use random Head initialization")
    _require(policy.get("head_init") is None, "random Head smoke named a legacy checkpoint")
    paired_config = _mapping(config.get("paired"), "paired config")
    _require(
        paired_config.get("supervision_mode") == "contrastive",
        "P-v1/P-v2 smoke must read the common paired diagnostic stream",
    )
    _require(paired_config.get("protocol_id") == POLICY_PROTOCOL_ID, "paired config protocol changed")
    _require(bool(paired_config.get("state_bank")), "paired config has no shared state bank")
    _require(bool(paired_config.get("text_cache_dir")), "paired config has no text cache")
    loss = _mapping(config.get("loss"), "loss config")
    _require(
        float(loss.get("lambda_contrastive", -1.0)) > 0.0
        and float(loss.get("lambda_paired_action", -1.0)) == 0.0,
        "engineering smoke must exercise the contrastive method path",
    )
    _require(
        base_lineage.get("status") == "PASS"
        and base_lineage.get("kind") == "policy_author_release_base_lineage"
        and base_lineage.get("base_kind") == "author_release"
        and "training_seed" not in base_lineage,
        "author release lineage audit failed",
    )
    _require(
        release_binding.get("status") == "PASS"
        and release_binding_crosscheck.get("status") == "PASS",
        "release/paired binding audit failed",
    )
    lineage_identity = _mapping(
        artifacts.get("base_lineage_manifest"), "base lineage artifact identity"
    )
    _sha256(lineage_identity.get("sha256"), "base lineage artifact")
    state_bank_identity = _mapping(
        artifacts.get("paired_state_bank"), "paired state-bank artifact identity"
    )
    text_cache_identity = _mapping(
        artifacts.get("paired_text_cache"), "paired text-cache artifact identity"
    )
    _sha256(state_bank_identity.get("sha256"), "paired state-bank artifact")
    _sha256(text_cache_identity.get("sha256"), "paired text-cache artifact")
    _require(int(_mapping(config.get("training"), "training").get("max_steps", -1)) == EXPECTED_STEPS, "smoke must run exactly three optimizer steps")
    source = _mapping(
        _mapping(config.get("runtime_provenance"), "runtime provenance").get("fastwam_source"),
        "FastWAM source provenance",
    )
    _require(source.get("status") == "PASS", "training FastWAM source audit did not pass")
    _require(source.get("scope") == "all_python_files_under_src_fastwam", "training source scope is incomplete")
    _require(int(source.get("file_count", 0)) > 0, "training source audit is empty")

    _require(summary.get("status") == "SMOKE_COMPLETE", "training summary is not SMOKE_COMPLETE")
    _require(summary.get("regime") == normalized_regime, "training summary regime mismatch")
    _require(int(summary.get("steps", -1)) == EXPECTED_STEPS, "training summary step count changed")
    runtime_batch = _mapping(
        summary.get("runtime_batch_contract"), "runtime batch contract"
    )
    _require(runtime_batch.get("status") == "PASS", "runtime batch contract did not pass")
    _require(
        int(runtime_batch.get("accelerator_num_processes", -1))
        == int(_mapping(config.get("training"), "training").get("world_size", -2)),
        "actual/configured world size mismatch",
    )
    _require(
        int(runtime_batch.get("gradient_accumulation_steps", -1)) == 1,
        "smoke used gradient accumulation",
    )
    _require(
        int(runtime_batch.get("effective_official_global_batch", -1))
        == int(config["training"]["effective_official_global_batch"])
        and int(runtime_batch.get("effective_paired_groups_per_step", -1))
        == int(config["training"]["effective_paired_groups_per_step"]),
        "runtime effective batch differs from locked config",
    )
    loader_rng = _mapping(
        summary.get("official_loader_rng_contract"), "official loader RNG contract"
    )
    _require(
        loader_rng.get("status") == "PASS"
        and loader_rng.get("identity_loader_is_separate") is True,
        "official loader RNG isolation did not pass",
    )
    _require(
        int(loader_rng.get("training_dataloader_generator_seed", -1))
        != int(loader_rng.get("identity_dataloader_generator_seed", -1)),
        "identity audit consumed the formal loader generator",
    )
    _require(
        _mapping(official.get("loader_rng_contract"), "official subset loader RNG")
        == loader_rng,
        "official audit/summary loader RNG contracts differ",
    )
    _require(summary.get("formal_training_auto_started") is False, "summary claims formal training")
    head_init = _mapping(summary.get("head_init"), "Head initialization summary")
    _require(head_init.get("mode") == "random", "smoke Head was not randomly initialized")
    _require(head_init.get("identity") is None, "smoke Head retained a legacy checkpoint identity")
    _require(isinstance(head_init.get("seed"), int), "smoke Head seed is missing")
    counts = _mapping(summary.get("parameter_counts"), "parameter counts")
    _require(int(counts.get("content_head", -1)) == EXPECTED_HEAD_PARAMETER_COUNT, "Content Head parameter count changed")
    _require(int(counts.get("adapter", -1)) == EXPECTED_ADAPTER_PARAMETER_COUNT, "adapter parameter count changed")
    expected_action = 0 if normalized_regime == "p_v1" else EXPECTED_ACTION_DIT_PARAMETER_COUNT
    _require(int(counts.get("action_dit", -1)) == expected_action, "ActionDiT trainable parameter count changed")
    _require(int(counts.get("total", -1)) == EXPECTED_HEAD_PARAMETER_COUNT + EXPECTED_ADAPTER_PARAMETER_COUNT + expected_action, "total trainable parameter count changed")
    _nonzero(summary.get("final_gate_raw"), "final raw gate")
    _finite(summary.get("final_gate_tanh"), "final tanh gate")
    last = _mapping(summary.get("last_metrics"), "last metrics")
    for name in (
        "loss_total",
        "loss_action",
        "loss_paired_action",
        "loss_contrastive",
        "positive_similarity",
        "negative_similarity",
        "gate_raw",
        "gate_tanh",
        "content_head_grad_norm",
        "adapter_grad_norm",
    ):
        _finite(last.get(name), f"last metric {name}")
    _require(last.get("loss_finite") is True and last.get("gradients_finite") is True, "last loss/gradient finite flags failed")
    _shape(last.get("layer16_shape"), "Layer-16 shape", (120, 3072))
    _shape(last.get("zc_shape"), "Zc shape", (8, 384))

    _require(identity.get("status") == "PASS", "zero-init identity status is not PASS")
    _require(identity.get("native_prefill_kv_bit_exact") is True, "video KV prefill is not bit-exact")
    _require(identity.get("action_output_bit_exact") is True, "zero-gate action output is not bit-exact")
    _require(_finite(identity.get("max_abs_error"), "identity max_abs_error") == 0.0, "identity max_abs_error is nonzero")
    _require(_finite(identity.get("max_rel_error"), "identity max_rel_error") == 0.0, "identity max_rel_error is nonzero")
    _require(_finite(identity.get("gate_raw"), "identity raw gate") == 0.0, "identity gate is not exact zero")
    _require(identity.get("finite") is True, "identity action output is non-finite")
    _shape(identity.get("layer16_shape"), "identity Layer-16 shape", (120, 3072))
    _shape(identity.get("content_token_shape"), "identity Zc shape", (8, 384))

    gradient_steps = _audit_gradient_steps(gradient, regime=normalized_regime)
    _audit_updates(update, regime=normalized_regime)

    _require(tuple(official.get("task_order", ())) == TASKS, "official subset task order changed")
    histogram = _mapping(official.get("task_histogram"), "official task histogram")
    _require(set(histogram) == set(TASKS), "official task histogram is incomplete")
    _require(all(int(_mapping(histogram[task], task).get("episodes", 0)) > 0 for task in TASKS), "official subset contains an empty task")
    _require(set(summary.get("official_task_sequence", ())) == set(TASKS), "official smoke did not cover all three tasks")
    _require(set(summary.get("paired_task_sequence", ())) == set(TASKS), "paired smoke did not cover all three tasks")
    _require(summary.get("paired_supervision_mode") == "contrastive", "summary paired mode changed")
    _require(
        float(summary.get("lambda_contrastive", -1.0)) > 0.0
        and summary.get("paired_contrastive_gradient_enabled") is True,
        "engineering smoke did not exercise contrastive gradients",
    )
    _require(
        summary.get("base_lineage_manifest_sha256") == lineage_identity["sha256"],
        "summary and artifacts bind different author release lineages",
    )
    _sha256(summary.get("official_sample_sequence_sha256"), "official sample sequence")
    _sha256(
        summary.get("paired_physical_state_sequence_sha256"),
        "paired physical-state sequence",
    )
    stream = _mapping(provenance.get("stream_contract"), "dual-stream contract")
    _require(stream.get("concatenated") is False, "official/paired datasets were concatenated")
    _require(stream.get("official_role") == "policy_action_supervision", "official stream role changed")
    _require(stream.get("paired_role") == "content_invariance_supervision", "paired stream role changed")
    _require(stream.get("paired_supervision_mode") == "contrastive", "paired stream mode changed")
    paired = _mapping(provenance.get("paired"), "paired provenance")
    _require(paired.get("protocol_id") == POLICY_PROTOCOL_ID, "paired protocol changed")
    _require(tuple(paired.get("variant_names", ())) == POLICY_VARIANTS, "paired scenes changed")
    _require(int(paired.get("view_count", -1)) == POLICY_VIEW_COUNT, "paired scene count changed")
    _require(paired.get("r3_role") == POLICY_R3_ROLE, "R3 is not a training positive")
    _require(paired.get("r3_training_positive") is True, "R3 positive audit is missing")
    _require(int(paired.get("camera_count", -1)) == POLICY_CAMERA_COUNT, "paired camera count changed")
    _require(tuple(paired.get("camera_names", ())) == POLICY_CAMERA_NAMES, "paired camera keys changed")
    _require(int(paired.get("native_fps", -1)) == POLICY_NATIVE_FPS, "paired data is not native 50 Hz")
    _require(int(paired.get("action_steps", -1)) == POLICY_ACTION_STEPS, "paired action horizon changed")
    _require(int(paired.get("action_dim", -1)) == POLICY_ACTION_DIM, "paired action dimension changed")
    _require(paired.get("temporal_resampling") == "none", "paired data used temporal resampling")
    _require(
        _sha256(paired.get("paired_state_bank_sha256"), "cache state-bank binding")
        == state_bank_identity["sha256"],
        "cache and run bind different state banks",
    )
    _sha256(
        paired.get("physical_state_inventory_sha256"),
        "paired physical-state inventory",
    )
    state_bank = _mapping(
        paired.get("shared_state_bank_contract"), "shared state-bank contract"
    )
    _require(state_bank.get("status") == "PASS", "shared state-bank audit did not pass")
    sampling = _mapping(state_bank.get("sampling"), "state-bank sampling")
    _require(
        sampling.get("algorithm") == POLICY_STATE_BANK_SAMPLING_ALGORITHM
        and int(sampling.get("seed", -1)) == POLICY_STATE_BANK_SEED
        and int(sampling.get("states_per_trajectory", -1)) == POLICY_STATES_PER_TRAJECTORY,
        "shared state-bank deterministic sampling changed",
    )
    _require(
        int(state_bank.get("physical_state_count", -1)) == 720,
        "three-task train state bank must contain exactly 720 physical states",
    )
    extraction = _mapping(paired.get("extraction_contract"), "cache extraction contract")
    _require(
        extraction.get("schema") == "policy_cache_extraction_contract_v1",
        "cache extraction dependency contract changed",
    )
    runtime_artifacts = _mapping(
        extraction.get("runtime_artifacts"), "cache runtime artifacts"
    )
    _require(
        _sha256(_mapping(runtime_artifacts.get("text_cache"), "cache text identity").get("sha256"), "cache text identity")
        == text_cache_identity["sha256"],
        "cache and smoke use different text caches",
    )
    native_prefill = _mapping(
        paired.get("native_prefill_identity_audit"), "cache native prefill audit"
    )
    _require(
        native_prefill.get("status") == "PASS"
        and int(native_prefill.get("checked_states", -1)) == 1
        and float(native_prefill.get("rtol", -1.0)) == 0.0
        and float(native_prefill.get("atol", -1.0)) == 0.0,
        "cache first-state native prefill identity audit failed",
    )
    _require(distribution.get("status") == "DIAGNOSTIC_ONLY_POLICY_NATIVE50HZ", "distribution audit status changed")
    _require(distribution.get("official_clean_claim_supported") is True, "hash-bound Clean range claim was lost")
    _require(distribution.get("official_domain_partition_verified") is True, "official domain partition was not verified")
    _require(distribution.get("intrinsic_metadata_domain_field") is False, "audit falsely claims an intrinsic domain field")
    _require(distribution.get("automatic_data_substitution") is False, "distribution audit substituted data")

    with (root / "train_log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == EXPECTED_STEPS, "train_log.csv must contain exactly three rows")

    _require(rollout.get("status") == "PASS", "rollout load/execute status is not PASS")
    _require(rollout.get("kind") == "no_sapien_one_action_rollout_smoke", "rollout smoke kind changed")
    _require(rollout.get("sapien_imported") is False, "no-SAPIEN smoke imported SAPIEN")
    rollout_tasks = rollout.get("tasks")
    _require(isinstance(rollout_tasks, list) and len(rollout_tasks) == len(TASKS), "rollout task count changed")
    _require(tuple(item.get("task") for item in rollout_tasks if isinstance(item, Mapping)) == TASKS, "rollout task order changed")
    for item in rollout_tasks:
        task = _mapping(item, "rollout task result")
        _require(int(task.get("executed_actions", -1)) == 1, "rollout did not execute exactly one action")
        _require(task.get("action_shape") == [14], "rollout action shape changed")
        _require(task.get("action_finite") is True, "rollout action is non-finite")
        _require(task.get("zc_shape") == [1, 8, 384], "deployed rollout Zc shape changed")
        _require(task.get("zc_finite") is True, "deployed rollout Zc is non-finite")
    rollout_audit = _mapping(rollout.get("checkpoint_audit"), "rollout checkpoint audit")
    rollout_source = _mapping(rollout_audit.get("fastwam_source"), "rollout source audit")
    _require(rollout_source.get("status") == "PASS", "rollout source audit did not pass")
    _require(rollout_source.get("scope") == source.get("scope"), "train/rollout source scopes differ")
    _require(rollout_source.get("source_root") == source.get("source_root"), "train/rollout source roots differ")
    _require(int(rollout_source.get("file_count", -1)) == int(source.get("file_count", -2)), "train/rollout source file counts differ")

    return {
        "status": "PASS",
        "kind": "strict_three_task_policy_smoke_audit",
        "run_dir": str(root),
        "regime": normalized_regime,
        "steps": EXPECTED_STEPS,
        "tasks": list(TASKS),
        "ten_smoke_goals": {
            "checkpoint_load": "PASS",
            "layer16_hook": "PASS",
            "content_head_shape": "PASS",
            "zero_init_identity": "PASS",
            "p_v1_gradient_flow": "PASS" if normalized_regime == "p_v1" else "NOT_APPLICABLE",
            "p_v2_param_groups_and_updates": "PASS" if normalized_regime == "p_v2" else "NOT_APPLICABLE",
            "dual_loader": "PASS",
            "finite_loss": "PASS",
            "short_parameter_update": "PASS",
            "rollout_load_execute": "PASS",
        },
        "identity": {
            "max_abs_error": identity["max_abs_error"],
            "max_rel_error": identity["max_rel_error"],
        },
        "final_gate_raw": summary["final_gate_raw"],
        "gradient_steps": len(gradient_steps),
        "fastwam_source_file_count": int(source["file_count"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--regime", required=True, choices=("p_v1", "p_v2"))
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = audit_smoke_run(args.run_dir, args.regime)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = ["SmokeAuditError", "audit_smoke_run"]
