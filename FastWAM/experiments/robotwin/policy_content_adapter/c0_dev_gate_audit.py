#!/usr/bin/env python3
"""Strictly audit the non-formal author-release C0 deployment gate.

The gate runs one episode for each of three tasks under Clean and official
Random.  Those six outcomes are engineering evidence only: they prove that the
fixed release can be packaged through the zero-gate transport and executed in
RoboTwin, but they are never eligible for the final Policy result table.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .config_audit import TASKS
from .eval_robotwin_single import (
    COMPLETED_ROLLOUTS_SCHEMA,
    COMPLETED_ROLLOUTS_SCHEMA_VERSION,
    OFFICIAL_TASK_CONFIGS,
    TASK_CONFIG_TO_DOMAIN,
    _canonical_sha256,
    _phase_name,
)
from .model import artifact_identity
from .native50hz_paired import atomic_write_json
from .p_mode_selection import validate_seed_bank_descriptor
from .rollout_policy import _read_checkpoint_provenance


DOMAINS = ("clean", "official_random")
EPISODES_PER_CELL = 1
PURPOSE = "development_analysis"


class C0DevGateAuditError(ValueError):
    """The C0 dev gate lacks complete, non-formal deployment evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C0DevGateAuditError(message)


def _json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise C0DevGateAuditError(f"cannot parse {label}: {resolved}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, resolved


def _expected_task_domains() -> set[tuple[str, str]]:
    return {(task, domain) for task in TASKS for domain in DOMAINS}


def audit_c0_dev_gate(
    *,
    identity_audit: str | Path,
    transport_checkpoint: str | Path,
    simulator_seed_bank_manifest: str | Path,
    completed_rollouts: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate the exact-zero identity, transport, and all six dev cells."""

    destination = Path(output).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite C0 dev audit: {destination}")
    identity, identity_path = _json(identity_audit, "C0 runtime identity audit")
    bank_raw, bank_path = _json(
        simulator_seed_bank_manifest, "C0 development seed-bank manifest"
    )
    completed, completed_path = _json(completed_rollouts, "completed rollouts")

    _require(identity.get("status") == "PASS", "C0 runtime identity is not PASS")
    _require(
        identity.get("kind") == "policy_c0_zero_gate_runtime_identity",
        "C0 runtime identity kind changed",
    )
    for field in ("native_prefill_kv_bit_exact", "action_output_bit_exact"):
        _require(identity.get(field) is True, f"C0 runtime identity lacks {field}")
    for field in ("max_abs_error", "max_rel_error", "gate_raw"):
        _require(float(identity.get(field, -1.0)) == 0.0, f"C0 identity {field} is not zero")
    official = identity.get("official_selection")
    expected_counts = {
        task: {"clean": 50, "official_random": 500} for task in TASKS
    }
    _require(isinstance(official, Mapping), "C0 identity lacks official selection audit")
    _require(
        official.get("selected_episode_counts_by_domain") == expected_counts,
        "C0 identity did not use the complete 50+500/task official partition",
    )
    cache_artifacts = identity.get("runtime_artifacts")
    _require(isinstance(cache_artifacts, Mapping), "C0 identity lacks runtime artifacts")
    cache = cache_artifacts.get("text_cache")
    _require(
        isinstance(cache, Mapping)
        and cache.get("kind") == "audited_directory_binding"
        and cache.get("directory_bytes_rehashed_for_c0") is False,
        "C0 identity lacks the strict official text-cache binding",
    )

    try:
        bank = validate_seed_bank_descriptor(bank_raw, expected_purpose=PURPOSE)
    except Exception as exc:
        raise C0DevGateAuditError(f"invalid C0 development seed bank: {exc}") from exc
    _require(
        bank["episodes_per_cell"] == EPISODES_PER_CELL,
        "C0 deployment gate must use one episode per task/domain",
    )

    checkpoint_path = Path(transport_checkpoint).expanduser().resolve()
    provenance = _read_checkpoint_provenance(checkpoint_path)
    metadata = provenance["metadata"]
    run_config = metadata["run_config"]
    evaluation = run_config.get("evaluation")
    semantics = run_config.get("c0_semantics")
    training = run_config.get("training")
    _require(run_config.get("kind") == "policy_c0_eval_transport", "wrong C0 transport kind")
    _require(run_config.get("control") == "c0_original", "wrong C0 transport control")
    _require(run_config.get("formal") is False, "C0 dev transport must be non-formal")
    _require(metadata.get("regime") == "p_v1" and metadata.get("step") == 0, "C0 transport changed native step/regime sentinel")
    _require(training == {"seed": None, "stage2_steps": 0}, "C0 transport claims Stage-2 training")
    _require(
        isinstance(semantics, Mapping)
        and semantics.get("stage2_training") is False
        and semantics.get("action_expert_overlay") is False
        and semantics.get("head_gca_effect_on_action") == "none_exact_zero_gate",
        "C0 transport does not prove exact native semantics",
    )
    _require(isinstance(evaluation, Mapping), "C0 transport lacks evaluation contract")
    _require(evaluation.get("tasks") == list(TASKS), "C0 transport task set changed")
    _require(
        evaluation.get("required_domains") == list(DOMAINS),
        "C0 transport domains changed",
    )
    _require(
        evaluation.get("simulator_seed_bank_id") == bank["simulator_seed_bank_id"]
        and evaluation.get("simulator_seed_bank_purpose") == PURPOSE
        and evaluation.get("episodes_per_task") == EPISODES_PER_CELL,
        "C0 transport is bound to another dev rollout protocol",
    )
    recorded_bank_sha = metadata.get("artifact_identities", {}).get(
        "simulator_seed_bank_manifest", {}
    ).get("sha256")
    _require(
        recorded_bank_sha == artifact_identity(bank_path)["sha256"],
        "C0 transport seed-bank bytes differ",
    )
    recorded_identity_sha = metadata.get("artifact_identities", {}).get(
        "c0_runtime_identity_audit", {}
    ).get("sha256")
    _require(
        recorded_identity_sha == artifact_identity(identity_path)["sha256"],
        "C0 transport runtime-identity bytes differ",
    )
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    gate = payload.get("content_adapter", {}).get("gate")
    _require(
        isinstance(gate, torch.Tensor) and float(gate.item()) == 0.0,
        "C0 transport gate is not exactly zero",
    )
    del payload

    _require(
        completed.get("schema") == COMPLETED_ROLLOUTS_SCHEMA
        and completed.get("schema_version") == COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        "completed-rollouts schema changed",
    )
    completed_contract = completed.get("checkpoint_contract")
    _require(isinstance(completed_contract, Mapping), "completed rollouts lack checkpoint contract")
    expected_contract_fields = {
        "control": "c0_original",
        "stage": "control",
        "training_seed": None,
        "policy_regime": None,
        "head_init_sha256": None,
        "gca_init_sha256": None,
        "stage2_recipe_sha256": None,
        "p_mode_selection_manifest_sha256": None,
        "simulator_seed_bank_id": bank["simulator_seed_bank_id"],
        "simulator_seed_bank_purpose": PURPOSE,
        "declared_tasks": list(TASKS),
        "declared_domains": list(DOMAINS),
        "declared_episodes_per_task": EPISODES_PER_CELL,
        "formal_evaluation_eligible": False,
    }
    for field, expected in expected_contract_fields.items():
        _require(
            completed_contract.get(field) == expected,
            f"completed checkpoint contract differs: {field}",
        )
    for field, expected in (
        ("base_checkpoint_sha256", metadata["base_checkpoint"]["sha256"]),
        (
            "dataset_stats_sha256",
            metadata["artifact_identities"]["dataset_stats"]["sha256"],
        ),
        (
            "base_lineage_manifest_sha256",
            metadata["artifact_identities"]["base_lineage_manifest"]["sha256"],
        ),
    ):
        _require(
            completed_contract.get(field) == expected,
            f"completed checkpoint ancestry differs: {field}",
        )
    _require(Path(str(completed.get("checkpoint", ""))).resolve() == checkpoint_path, "completed rollouts used another checkpoint")
    _require(completed.get("episodes_per_task") == EPISODES_PER_CELL, "completed episode count changed")
    _require(completed.get("simulator_seed") == bank["simulator_seed"], "completed simulator seed changed")
    _require(completed.get("simulator_seed_bank") == bank_raw, "completed seed-bank payload differs")
    _require(completed.get("simulator_seed_bank_id") == bank["simulator_seed_bank_id"], "completed seed-bank id differs")
    _require(completed.get("simulator_seed_bank_purpose") == PURPOSE, "completed seed-bank purpose differs")
    _require(completed.get("checkpoint_fairness_identity") is None, "non-formal C0 must not claim a fairness-table identity")
    protocol = completed.get("evaluation_protocol")
    _require(
        isinstance(protocol, Mapping)
        and protocol.get("eligible") is False
        and protocol.get("control") is None,
        "C0 dev results must be ineligible for the formal result table",
    )
    _require(completed.get("evaluation_records") == [], "C0 dev gate emitted formal records")
    gpu_binding = completed.get("gpu_runtime_binding")
    _require(
        isinstance(gpu_binding, Mapping)
        and gpu_binding.get("status") == "PASS"
        and isinstance(gpu_binding.get("physical_gpu_index"), int)
        and str(gpu_binding.get("render_device_alias", "")).startswith("pci:"),
        "C0 dev gate lacks an explicit PASS CUDA/Vulkan/SAPIEN binding",
    )
    sapien_binding = gpu_binding.get("sapien")
    _require(
        isinstance(sapien_binding, Mapping)
        and sapien_binding.get("can_render") is True
        and sapien_binding.get("pci_bus_id") == gpu_binding.get("pci_bus_id"),
        "C0 dev gate lacks matching SAPIEN PCI preflight evidence",
    )
    settings = completed.get("rollout_settings")
    settings_sha = str(completed.get("rollout_settings_sha256", ""))
    _require(
        isinstance(settings, Mapping) and _canonical_sha256(settings) == settings_sha,
        "completed rollout-settings identity differs",
    )
    runs = completed.get("runs")
    _require(isinstance(runs, list) and len(runs) == 6, "C0 dev gate requires exactly six rollout cells")
    cells: set[tuple[str, str]] = set()
    outcomes: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    for index, run in enumerate(runs):
        _require(isinstance(run, Mapping), f"run {index} is not an object")
        task = str(run.get("task", ""))
        task_config = str(run.get("task_config", ""))
        _require(task in TASKS, f"run {index} has unsupported task")
        _require(task_config in OFFICIAL_TASK_CONFIGS, f"run {index} has unsupported task config")
        domain = TASK_CONFIG_TO_DOMAIN[task_config]
        _require(run.get("domain") == domain and run.get("phase") == _phase_name(task_config), f"run {index} domain/phase differs")
        cell = (task, domain)
        _require(cell not in cells, f"duplicate C0 dev cell: {cell}")
        cells.add(cell)
        _require(run.get("episodes") == EPISODES_PER_CELL, f"run {index} episode count differs")
        _require(run.get("simulator_seed") == bank["simulator_seed"], f"run {index} simulator seed differs")
        _require(run.get("simulator_seed_bank_id") == bank["simulator_seed_bank_id"], f"run {index} seed-bank id differs")
        _require(run.get("simulator_seed_bank_purpose") == PURPOSE, f"run {index} seed-bank purpose differs")
        _require(run.get("rollout_settings_sha256") == settings_sha, f"run {index} rollout settings differ")
        _require(
            run.get("physical_gpu_index") == gpu_binding["physical_gpu_index"]
            and run.get("render_device_alias") == gpu_binding["render_device_alias"],
            f"run {index} GPU/render binding differs",
        )
        try:
            rate = float(run["success_rate"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise C0DevGateAuditError(f"run {index} success rate is invalid") from exc
        _require(math.isfinite(rate) and rate in {0.0, 1.0}, f"one-episode run {index} must have binary SR")
        for artifact_field in ("log", "result"):
            _require(Path(str(run.get(artifact_field, ""))).is_file(), f"run {index} {artifact_field} is missing")
        log_text = Path(str(run["log"])).read_text(encoding="utf-8", errors="replace")
        _require(
            "SAPIEN render device:" in log_text,
            f"run {index} lacks explicit SAPIEN render-device evidence",
        )
        outcomes[task][domain] = rate
    _require(cells == _expected_task_domains(), "C0 dev task/domain matrix is incomplete")

    report = {
        "schema_version": 1,
        "kind": "policy_c0_author_release_dev_deployment_gate",
        "status": "PASS",
        "scientific_result": False,
        "formal_test_bank_opened": False,
        "formal_evaluation_records_emitted": False,
        "purpose": "bit_exact_zero_gate_and_six_cell_deployment_only",
        "tasks": list(TASKS),
        "domains": list(DOMAINS),
        "episodes_per_task_domain": EPISODES_PER_CELL,
        "total_rollout_episodes": 6,
        "dev_outcomes_diagnostic_only": outcomes,
        "simulator_seed_bank_id": bank["simulator_seed_bank_id"],
        "artifacts": {
            "identity_audit": artifact_identity(identity_path),
            "transport_checkpoint": artifact_identity(checkpoint_path),
            "simulator_seed_bank_manifest": artifact_identity(bank_path),
            "completed_rollouts": artifact_identity(completed_path),
        },
    }
    atomic_write_json(destination, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-audit", required=True)
    parser.add_argument("--transport-checkpoint", required=True)
    parser.add_argument("--simulator-seed-bank-manifest", required=True)
    parser.add_argument("--completed-rollouts", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_c0_dev_gate(
        identity_audit=args.identity_audit,
        transport_checkpoint=args.transport_checkpoint,
        simulator_seed_bank_manifest=args.simulator_seed_bank_manifest,
        completed_rollouts=args.completed_rollouts,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["C0DevGateAuditError", "audit_c0_dev_gate"]
