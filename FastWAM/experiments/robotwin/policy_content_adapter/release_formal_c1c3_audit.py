"""Strict prelaunch and post-training audits for formal release C1/C3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_audit import (
    load_config,
    validate_c1_c3_pair,
    validate_execution_ready,
)
from .materialize_release_engineering_smoke import _write_new_json
from .materialize_release_formal_c1c3 import (
    CONTROLS,
    DEFAULT_MAX_STEPS,
    FORMAL_SEEDS,
    validate_formal_matrix_configs,
)
from .p_mode_selection import (
    validate_formal_protocol_lock_manifest_payload,
    validate_seed_bank_descriptor,
)


class FormalC1C3AuditError(ValueError):
    """Formal C1/C3 artifacts do not prove the locked paired experiment."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalC1C3AuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FormalC1C3AuditError(f"cannot parse {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify_identity(value: Any, label: str) -> Path:
    _require(isinstance(value, Mapping), f"{label} identity is missing")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    _require(path.is_file(), f"{label} file missing: {path}")
    _require(path.stat().st_size == value.get("size_bytes"), f"{label} size changed")
    _require(_sha256(path) == value.get("sha256"), f"{label} SHA-256 changed")
    return path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _gradient_norm(
    combined: Mapping[str, Any], module: str, label: str
) -> float:
    report = _mapping(combined.get(module), f"{label} {module} report")
    _require(report.get("all_finite") is True, f"{label} {module} is non-finite")
    return _finite_number(report.get("gradient_norm"), f"{label} {module} norm")


def _audit_formal_gradient_audit(
    gradient: Mapping[str, Any], *, label: str
) -> dict[str, int]:
    """Validate every formal P-v1 gradient row, including zero-weight steps.

    The native action scheduler can legitimately produce exact-zero endpoint
    weights.  Such a row is not evidence of a broken action path, but it must be
    explicitly labelled and have exactly zero action-only gradients.  The first
    positive-weight row opens the zero-initialized gate; all later positive rows
    must prove that action loss reaches Head, GCA, and official Zc tokens.
    """

    _require(gradient.get("status") == "PASS", f"{label} status is not PASS")
    _require(gradient.get("regime") == "p_v1", f"{label} regime is not P-v1")
    steps = gradient.get("steps")
    _require(isinstance(steps, list), f"{label} steps must be a list")
    _require(
        len(steps) == DEFAULT_MAX_STEPS,
        f"{label} must contain exactly {DEFAULT_MAX_STEPS} rows",
    )

    positive_declared = gradient.get("positive_action_signal_steps")
    zero_declared = gradient.get("zero_action_signal_steps")
    _require(
        isinstance(positive_declared, int) and not isinstance(positive_declared, bool),
        f"{label} positive_action_signal_steps must be an integer",
    )
    _require(
        isinstance(zero_declared, int) and not isinstance(zero_declared, bool),
        f"{label} zero_action_signal_steps must be an integer",
    )
    _require(
        positive_declared + zero_declared == DEFAULT_MAX_STEPS,
        f"{label} positive/zero counts do not sum to {DEFAULT_MAX_STEPS}",
    )
    _require(
        positive_declared >= 2,
        f"{label} needs at least two positive-weight action rows",
    )

    positive_observed = 0
    zero_observed = 0
    for expected_step, raw_row in enumerate(steps, start=1):
        row_label = f"{label} step {expected_step}"
        row = _mapping(raw_row, row_label)
        _require(row.get("step") == expected_step, f"{row_label} order changed")
        signal_positive = row.get("action_supervision_signal_positive")
        zero_weight = row.get("zero_weight_action_step")
        _require(
            isinstance(signal_positive, bool),
            f"{row_label} action supervision flag must be boolean",
        )
        _require(
            isinstance(zero_weight, bool),
            f"{row_label} zero-weight flag must be boolean",
        )
        _require(
            zero_weight is (not signal_positive),
            f"{row_label} zero-weight flag contradicts action supervision",
        )

        combined = _mapping(row.get("combined"), f"{row_label} combined gradients")
        head_norm = _gradient_norm(combined, "content_head", row_label)
        adapter_norm = _gradient_norm(combined, "adapter", row_label)
        attention_norm = _gradient_norm(
            combined,
            "adapter_attention_action_only_by_construction",
            row_label,
        )
        probe = _mapping(row.get("action_only_probe"), f"{row_label} action-only probe")
        _require(
            probe.get("all_finite") is True,
            f"{row_label} action-only probe is non-finite",
        )
        probe_head = _finite_number(
            probe.get("head_grad_norm"), f"{row_label} probe Head norm"
        )
        probe_attention = _finite_number(
            probe.get("adapter_attention_grad_norm"),
            f"{row_label} probe GCA-attention norm",
        )
        probe_gate = _finite_number(
            probe.get("gate_grad_norm"), f"{row_label} probe gate norm"
        )
        content_norm = _finite_number(
            row.get("action_only_official_content_token_grad_norm"),
            f"{row_label} official Zc norm",
        )
        loss_action = _finite_number(
            row.get("loss_action"), f"{row_label} weighted action loss"
        )
        action_weight_min = _finite_number(
            row.get("action_weight_min"), f"{row_label} action weight minimum"
        )
        action_weight_max = _finite_number(
            row.get("action_weight_max"), f"{row_label} action weight maximum"
        )
        action_effective_weight_sum = _finite_number(
            row.get("action_effective_weight_sum"),
            f"{row_label} effective action weight sum",
        )
        zero_reason = row.get("zero_action_signal_reason")
        _require(
            isinstance(zero_reason, str),
            f"{row_label} zero-action reason must be a string",
        )

        if zero_weight:
            zero_observed += 1
            _require(
                zero_reason == "scheduler_zero_weight",
                f"{row_label} zero-weight reason is not scheduler_zero_weight",
            )
            zero_values = {
                "combined adapter": adapter_norm,
                "combined GCA attention": attention_norm,
                "probe Head": probe_head,
                "probe GCA attention": probe_attention,
                "probe gate": probe_gate,
                "official Zc": content_norm,
                "weighted action loss": loss_action,
                "action weight minimum": action_weight_min,
                "action weight maximum": action_weight_max,
                "effective action weight sum": action_effective_weight_sum,
            }
            for name, value in zero_values.items():
                _require(value == 0.0, f"{row_label} zero-weight {name} is nonzero")
            # C3's contrastive branch may still give the Content Head a combined
            # gradient on a zero-action row, so head_norm is intentionally only
            # checked for finiteness above.
            continue

        positive_observed += 1
        _require(
            zero_reason == "none",
            f"{row_label} positive-weight reason is not none",
        )
        _require(
            action_weight_max > 0.0,
            f"{row_label} positive-weight action weight maximum is not positive",
        )
        _require(
            action_effective_weight_sum > 0.0,
            f"{row_label} positive-weight effective action weight sum is not positive",
        )
        _require(
            loss_action >= 0.0,
            f"{row_label} positive-weight action loss is negative",
        )
        _require(adapter_norm > 0.0, f"{row_label} positive-weight GCA norm is zero")
        _require(probe_gate > 0.0, f"{row_label} positive-weight gate probe is zero")
        gate_after_step = _finite_number(
            row.get("gate_raw_after_step"), f"{row_label} gate after step"
        )
        _require(
            abs(gate_after_step) > 0.0,
            f"{row_label} positive-weight step did not open the gate",
        )
        if positive_observed >= 2:
            positive_values = {
                "combined Head": head_norm,
                "combined GCA attention": attention_norm,
                "probe Head": probe_head,
                "probe GCA attention": probe_attention,
                "probe gate": probe_gate,
                "official Zc": content_norm,
            }
            for name, value in positive_values.items():
                _require(
                    value > 0.0,
                    f"{row_label} post-gate positive-weight {name} is zero",
                )

    _require(
        positive_observed == positive_declared,
        f"{label} observed positive-row count differs from its declaration",
    )
    _require(
        zero_observed == zero_declared,
        f"{label} observed zero-row count differs from its declaration",
    )
    return {
        "rows": len(steps),
        "positive_action_signal_steps": positive_observed,
        "zero_action_signal_steps": zero_observed,
    }


def audit_materialization(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _json(manifest_path, "formal materialization manifest")
    _require(
        manifest.get("kind") == "policy_release_formal_c1_c3_materialization"
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "PASS",
        "formal materialization kind/version/status differs",
    )
    _require(manifest.get("selected_policy_regime") == "p_v1", "formal regime is not P-v1")
    _require(manifest.get("stage2_training_seeds") == [1, 2, 3], "formal seeds changed")
    _require(manifest.get("c0_evaluation_requested") is False, "materialization unexpectedly requests C0")
    recipe = manifest.get("recipe")
    _require(isinstance(recipe, Mapping), "formal recipe is missing")
    _require(
        recipe.get("max_steps") == DEFAULT_MAX_STEPS
        and recipe.get("world_size") == 1
        and recipe.get("official_batch_size_per_rank") == 1
        and recipe.get("paired_groups_per_batch_per_rank") == 2,
        "formal recipe differs from the reviewed 1800-step single-GPU contract",
    )

    configs: dict[int, dict[str, dict[str, Any]]] = {}
    config_paths: dict[int, dict[str, str]] = {}
    raw_configs = manifest.get("configs")
    _require(isinstance(raw_configs, Mapping) and set(raw_configs) == {"1", "2", "3"}, "formal config matrix changed")
    for seed in FORMAL_SEEDS:
        identities = raw_configs[str(seed)]
        _require(isinstance(identities, Mapping) and set(identities) == {"c1", "c3"}, f"seed {seed} config pair changed")
        configs[seed] = {}
        config_paths[seed] = {}
        for short, control in (("c1", "c1_architecture_only"), ("c3", "c3_ours")):
            config_path = _verify_identity(identities[short], f"seed {seed}/{short} config")
            config = load_config(config_path)
            validate_execution_ready(config)
            _require(config["control"] == control, f"seed {seed}/{short} control differs")
            configs[seed][control] = config
            config_paths[seed][short] = str(config_path)
        validate_c1_c3_pair(
            configs[seed]["c1_architecture_only"], configs[seed]["c3_ours"]
        )
    recomputed = validate_formal_matrix_configs(configs)
    embedded = manifest.get("matrix_audit")
    _require(isinstance(embedded, Mapping) and embedded.get("status") == "PASS", "embedded formal matrix audit is missing")
    _require(
        [row["expected_initialization"] for row in recomputed["rows"]]
        == [row["expected_initialization"] for row in embedded.get("rows", [])],
        "formal expected initialization identities changed",
    )

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, Mapping), "formal materialization artifacts are missing")
    for key in (
        "recipe_amendment_manifest",
        "p_mode_selection_manifest",
        "dev_seed_bank_manifest",
        "formal_matrix_audit",
        "formal_protocol_lock",
        "final_test_seed_bank",
    ):
        _verify_identity(artifacts.get(key), key)
    lock_path = Path(artifacts["formal_protocol_lock"]["path"])
    lock = validate_formal_protocol_lock_manifest_payload(
        _json(lock_path, "formal protocol lock")
    )
    _require(lock["selected_policy_regime"] == "p_v1", "formal lock regime changed")
    final_path = Path(artifacts["final_test_seed_bank"]["path"])
    final_bank = validate_seed_bank_descriptor(
        _json(final_path, "final-test seed bank"), expected_purpose="final_test"
    )
    _require(
        final_bank["simulator_seed_bank_id"] == artifacts.get("final_test_seed_bank_id"),
        "final-test seed-bank id differs from materialization",
    )
    return {
        "status": "PASS",
        "stage": "prelaunch",
        "materialization_manifest": str(manifest_path),
        "configs": config_paths,
        "formal_protocol_lock_sha256": artifacts["formal_protocol_lock"]["sha256"],
        "final_test_seed_bank_sha256": artifacts["final_test_seed_bank"]["sha256"],
        "final_test_seed_bank_id": final_bank["simulator_seed_bank_id"],
    }


def audit_posttrain(path: str | Path) -> dict[str, Any]:
    prelaunch = audit_materialization(path)
    manifest_path = Path(path).expanduser().resolve()
    manifest = _json(manifest_path, "formal materialization manifest")
    expected_rows = {
        int(row["training_seed"]): row["expected_initialization"]
        for row in manifest["matrix_audit"]["rows"]
    }
    checkpoints: dict[str, dict[str, dict[str, Any]]] = {}
    paired_sequences: dict[str, dict[str, str]] = {}
    for seed in FORMAL_SEEDS:
        summaries: dict[str, dict[str, Any]] = {}
        sequences: dict[str, dict[str, Any]] = {}
        contracts: dict[str, dict[str, Any]] = {}
        checkpoints[str(seed)] = {}
        for short, control, coefficient in (
            ("c1", "c1_architecture_only", 0.0),
            ("c3", "c3_ours", 0.1),
        ):
            config_path = Path(prelaunch["configs"][seed][short])
            config = load_config(config_path)
            root = Path(config["output_dir"]).expanduser().resolve()
            summary = _json(root / "training_summary.json", f"seed {seed}/{short} summary")
            sequence = _json(root / "training_sequence_audit.json", f"seed {seed}/{short} sequence")
            contract = _json(root / "matched_stream_contract.json", f"seed {seed}/{short} stream contract")
            gradient = _json(root / "gradient_audit.json", f"seed {seed}/{short} gradient audit")
            updates = _json(root / "parameter_update_audit.json", f"seed {seed}/{short} update audit")
            checkpoint = root / "checkpoint.pt"
            _require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"seed {seed}/{short} checkpoint missing")
            _require(summary.get("status") == "COMPLETE", f"seed {seed}/{short} formal training incomplete")
            _require(summary.get("control") == control and summary.get("regime") == "p_v1", f"seed {seed}/{short} control/regime differs")
            _require(summary.get("steps") == DEFAULT_MAX_STEPS, f"seed {seed}/{short} step count differs")
            _require(float(summary.get("lambda_contrastive", -1.0)) == coefficient, f"seed {seed}/{short} lambda differs")
            _require(summary.get("paired_contrastive_gradient_enabled") is (coefficient > 0.0), f"seed {seed}/{short} gradient switch differs")
            _require(summary.get("deliverable_status", {}).get("formal_long_training") == "PASS", f"seed {seed}/{short} formal deliverable is not PASS")
            _require(sequence.get("status") == "PASS", f"seed {seed}/{short} sequence audit is not PASS")
            _require(sequence.get("official_sample_count") == DEFAULT_MAX_STEPS, f"seed {seed}/{short} official sample count differs")
            _require(sequence.get("paired_physical_state_count") == 2 * DEFAULT_MAX_STEPS, f"seed {seed}/{short} paired draw count differs")
            _require(contract.get("status") == "PASS", f"seed {seed}/{short} stream contract is not PASS")
            _audit_formal_gradient_audit(
                gradient, label=f"seed {seed}/{short} gradient audit"
            )
            _require(updates.get("head_and_adapter", {}).get("changed_parameter_tensors", 0) > 0, f"seed {seed}/{short} Head/GCA did not update")
            _require(updates.get("action_dit", {}).get("changed") is False, f"seed {seed}/{short} P-v1 ActionDiT changed")
            expected = expected_rows[seed]
            initialization = summary.get("initialization", {})
            for field in (
                "source_fp32_content_head_sha256",
                "source_fp32_adapter_sha256",
            ):
                _require(initialization.get(field) == expected[field], f"seed {seed}/{short} initialization {field} differs")
            for field in (
                "source_fp32_content_head_sha256",
                "source_fp32_adapter_sha256",
                "training_fp32_content_head_sha256",
                "training_fp32_adapter_sha256",
            ):
                _require(isinstance(initialization.get(field), str), f"seed {seed}/{short} initialization {field} missing")
            _require(summary.get("matched_stream_contract_sha256") == contract.get("sha256"), f"seed {seed}/{short} contract SHA binding differs")
            _require(summary.get("training_sequence_audit") == sequence, f"seed {seed}/{short} sequence summary differs")
            summaries[short] = summary
            sequences[short] = sequence
            contracts[short] = contract
            checkpoints[str(seed)][short] = {
                "path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            }
        for field in (
            "official_sample_sequence_sha256",
            "paired_physical_state_sequence_sha256",
            "matched_stream_contract_sha256",
        ):
            _require(summaries["c1"].get(field) == summaries["c3"].get(field), f"seed {seed} C1/C3 {field} differs")
        _require(sequences["c1"] == sequences["c3"], f"seed {seed} C1/C3 sequence files differ")
        _require(contracts["c1"] == contracts["c3"], f"seed {seed} C1/C3 stream contracts differ")
        for field in (
            "source_fp32_content_head_sha256",
            "source_fp32_adapter_sha256",
            "training_fp32_content_head_sha256",
            "training_fp32_adapter_sha256",
        ):
            _require(
                summaries["c1"]["initialization"][field]
                == summaries["c3"]["initialization"][field],
                f"seed {seed} C1/C3 initialization {field} differs",
            )
        paired_sequences[str(seed)] = {
            field: str(summaries["c1"][field])
            for field in (
                "official_sample_sequence_sha256",
                "paired_physical_state_sequence_sha256",
                "matched_stream_contract_sha256",
            )
        }
    return {
        "kind": "policy_release_formal_c1_c3_posttrain_audit",
        "schema_version": 1,
        "status": "PASS",
        "formal_training_complete": True,
        "online_rollout_started": False,
        "c0_evaluation_requested": False,
        "prelaunch": prelaunch,
        "checkpoints": checkpoints,
        "matched_sequences": paired_sequences,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-manifest", required=True)
    parser.add_argument("--stage", choices=("prelaunch", "posttrain"), default="prelaunch")
    parser.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        audit_materialization(args.materialization_manifest)
        if args.stage == "prelaunch"
        else audit_posttrain(args.materialization_manifest)
    )
    if args.output_json:
        _write_new_json(Path(args.output_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FormalC1C3AuditError",
    "audit_materialization",
    "audit_posttrain",
]
