"""Fail-closed audits and pilot gate for the P-v2 ActionDiT follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_audit import load_config, validate_c1_c3_pair, validate_execution_ready
from .materialize_pv2_actiondit_followup import (
    PILOT_CLEAN_DELTA_MIN,
    PILOT_MAX_STEPS,
    PILOT_RANDOM_DELTA_MIN,
    PILOT_SIMULATOR_SEED,
    PILOT_TRAINING_SEED,
    SMOKE_MAX_STEPS,
    validate_followup_pair,
)
from .p_mode_selection import TASKS, validate_seed_bank_descriptor
from .pv2_followup_eval100_amendment import (
    PROFILE as EVAL100_PROFILE,
    RUNTIME_EPISODES_PER_CELL as EVAL100_EPISODES_PER_CELL,
    SIMULATOR_SEED as EVAL100_SIMULATOR_SEED,
    validate_eval100_amendment,
)


DOMAINS = ("clean", "official_random")


class Pv2FollowupAuditError(ValueError):
    """One or more P-v2 mechanism-study invariants were not proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2FollowupAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required artifact missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pv2FollowupAuditError(f"cannot parse {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _bound_identity(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(value, Mapping), f"{label} identity must be an object")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    actual = _identity(path)
    _require(actual["sha256"] == value.get("sha256"), f"{label} SHA changed")
    _require(actual["size_bytes"] == value.get("size_bytes"), f"{label} size changed")
    return path, actual


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise Pv2FollowupAuditError(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def audit_materialization(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _json(manifest_path, "materialization manifest")
    _require(
        manifest.get("kind") == "policy_pv2_actiondit_followup_materialization"
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "PASS",
        "materialization kind/version/status differs",
    )
    _require(
        manifest.get("scientific_results_present") is False
        and manifest.get("gpu_training_started") is False
        and manifest.get("online_rollout_started") is False
        and manifest.get("primary_pv1_modified") is False,
        "materialization must predate all P-v2 results",
    )

    configs = manifest.get("configs")
    _require(
        isinstance(configs, Mapping) and set(configs) == {"pilot", "smoke"},
        "materialization must contain pilot and smoke config pairs",
    )
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    identities: dict[str, dict[str, dict[str, Any]]] = {}
    for pair in ("pilot", "smoke"):
        pair_configs = configs[pair]
        _require(
            isinstance(pair_configs, Mapping) and set(pair_configs) == {"c1", "c3"},
            f"{pair} configs must contain exactly C1/C3",
        )
        loaded[pair] = {}
        identities[pair] = {}
        for control in ("c1", "c3"):
            config_path, identity = _bound_identity(
                pair_configs[control], f"{pair}/{control} config"
            )
            config = load_config(config_path)
            validate_execution_ready(config)
            loaded[pair][control] = config
            identities[pair][control] = identity
    followup_fairness = validate_followup_pair(
        loaded["pilot"]["c1"], loaded["pilot"]["c3"]
    )
    smoke_fairness = validate_c1_c3_pair(
        loaded["smoke"]["c1"], loaded["smoke"]["c3"]
    )
    _require(
        followup_fairness == manifest.get("fairness"),
        "pilot fairness proof changed",
    )
    _require(
        smoke_fairness == manifest.get("smoke_fairness"),
        "smoke fairness proof changed",
    )
    for config in loaded["smoke"].values():
        _require(config["policy"]["regime"] == "p_v2", "smoke is not P-v2")
        _require(config["training"]["max_steps"] == 3, "smoke step count changed")

    protocol_path, protocol_identity = _bound_identity(
        manifest["protocol"], "mechanism protocol"
    )
    action_path, action_identity = _bound_identity(
        manifest["action_dit_initialization_audit"],
        "ActionDiT initialization audit",
    )
    action = _json(action_path, "ActionDiT initialization audit")
    _require(
        action.get("status") == "PASS"
        and action.get("tensor_count") == 824
        and action.get("checkpoint_sha256")
        == loaded["pilot"]["c1"]["artifacts"]["base_checkpoint_sha256"],
        "ActionDiT release initialization proof changed",
    )
    _require(
        len(str(action.get("action_dit_tensor_sha256", ""))) == 64,
        "ActionDiT tensor SHA is invalid",
    )
    protocol = _json(protocol_path, "mechanism protocol")
    _require(
        protocol["action_dit_initialization_audit"]["sha256"]
        == action_identity["sha256"],
        "protocol binds a different ActionDiT initialization audit",
    )
    source_sha = protocol.get("source_sha256")
    _require(isinstance(source_sha, Mapping), "mechanism source SHA map is missing")
    source_root = Path(__file__).resolve().parent
    _require(
        set(source_sha)
        == {
            "config_audit.py",
            "losses.py",
            "materialize_pv2_actiondit_followup.py",
            "train.py",
        },
        "mechanism source binding file set changed",
    )
    for name, expected_sha in source_sha.items():
        source_path = source_root / str(name)
        _require(
            source_path.is_file() and _sha256(source_path) == expected_sha,
            f"mechanism source drifted: {name}",
        )
    pilot_bank_path, pilot_bank_identity = _bound_identity(
        manifest["pilot_seed_bank"], "pilot seed bank"
    )
    pilot_bank = validate_seed_bank_descriptor(
        _json(pilot_bank_path, "pilot seed bank"), expected_purpose="dev_selection"
    )
    _require(
        pilot_bank["simulator_seed"] == PILOT_SIMULATOR_SEED
        and pilot_bank["episodes_per_cell"] == 20,
        "pilot seed bank no longer encodes seed53/20 episodes",
    )
    smoke_bank_path, smoke_bank_identity = _bound_identity(
        manifest["smoke_seed_bank"], "smoke seed bank"
    )
    smoke_bank = validate_seed_bank_descriptor(
        _json(smoke_bank_path, "smoke seed bank"),
        expected_purpose="engineering_smoke",
    )
    _require(smoke_bank["episodes_per_cell"] == 1, "smoke bank episode count changed")
    return {
        "status": "PASS",
        "stage": "materialization",
        "manifest": _identity(manifest_path),
        "configs": identities,
        "protocol": protocol_identity,
        "action_dit_initialization_audit": action_identity,
        "action_dit_initial_tensor_sha256": action[
            "action_dit_tensor_sha256"
        ],
        "pilot_seed_bank": pilot_bank_identity,
        "smoke_seed_bank": smoke_bank_identity,
        "fairness": followup_fairness,
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"training log missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _audit_training_run(
    root: Path,
    *,
    control: str,
    expected_steps: int,
    expected_lambda: float,
) -> dict[str, Any]:
    summary = _json(root / "training_summary.json", f"{control} training summary")
    run_config = _json(root / "run_config.json", f"{control} run config")
    gradient = _json(root / "gradient_audit.json", f"{control} gradient audit")
    updates = _json(
        root / "parameter_update_audit.json", f"{control} parameter update audit"
    )
    sequence = _json(
        root / "training_sequence_audit.json", f"{control} sequence audit"
    )
    contract = _json(
        root / "matched_stream_contract.json", f"{control} matched contract"
    )
    checkpoint = root / "checkpoint.pt"
    rows = _csv_rows(root / "train_log.csv")
    _require(summary.get("status") == "SMOKE_COMPLETE", f"{control} status differs")
    _require(summary.get("control") == control, f"{control} summary label differs")
    _require(summary.get("regime") == "p_v2", f"{control} is not P-v2")
    _require(summary.get("steps") == expected_steps, f"{control} steps differ")
    _require(len(rows) == expected_steps, f"{control} CSV row count differs")
    _require(
        float(summary.get("lambda_contrastive", -1.0)) == expected_lambda,
        f"{control} lambda differs",
    )
    _require(
        summary.get("paired_contrastive_gradient_enabled")
        is (expected_lambda > 0.0),
        f"{control} contrastive gradient switch differs",
    )
    _require(
        run_config["policy"]["freeze"]["action_dit"] is False,
        f"{control} ActionDiT is frozen",
    )
    resolved_base = run_config.get("resolved_base_checkpoint_identity")
    _require(
        isinstance(resolved_base, Mapping)
        and resolved_base.get("sha256")
        == run_config["artifacts"]["base_checkpoint_sha256"],
        f"{control} runtime base checkpoint identity is inconsistent",
    )
    release_load = summary.get("base_release_load")
    _require(
        isinstance(release_load, Mapping)
        and release_load.get("step") == 29355
        and release_load.get("mot_tensor_count") == 1649
        and release_load.get("proprio_tensor_count") == 2,
        f"{control} did not prove the strict author release load",
    )
    _require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"{control} checkpoint missing")
    _require(gradient.get("status") == "PASS", f"{control} gradient audit is not PASS")
    gradient_steps = gradient.get("steps")
    _require(
        isinstance(gradient_steps, list) and len(gradient_steps) == expected_steps,
        f"{control} gradient step count differs",
    )
    positive_action_steps = 0
    positive_action_dit_gradient_steps = 0
    for index, step in enumerate(gradient_steps, start=1):
        _require(step.get("step") == index, f"{control} gradient steps are unordered")
        action_report = step.get("combined", {}).get("action_dit", {})
        _require(
            action_report.get("all_finite") is True
            and action_report.get("gradient_tensors") == 824,
            f"{control} ActionDiT gradient audit failed at step {index}",
        )
        if rows[index - 1]["action_supervision_signal_positive"] == "True":
            positive_action_steps += 1
            positive_action_dit_gradient_steps += int(
                float(action_report.get("gradient_norm", 0.0)) > 0.0
            )
    _require(positive_action_steps >= 2, f"{control} lacks positive action supervision")
    _require(
        positive_action_dit_gradient_steps > 0,
        f"{control} never proves an ActionDiT gradient",
    )
    coverage = gradient.get("positive_action_path_coverage")
    _require(
        isinstance(coverage, Mapping)
        and int(coverage.get("positive_weight_steps", -1)) == positive_action_steps
        and all(
            int(coverage.get(name, 0)) > 0
            for name in (
                "gate_positive_steps",
                "adapter_attention_positive_steps",
                "official_content_token_positive_steps",
                "action_dit_positive_steps",
            )
        ),
        f"{control} cumulative action-path coverage failed",
    )

    action_update = updates.get("action_dit")
    _require(isinstance(action_update, Mapping), f"{control} ActionDiT update missing")
    _require(
        action_update.get("changed") is True
        and action_update.get("all_finite") is True
        and float(action_update.get("changed_fraction", 0.0)) >= 0.5
        and int(action_update.get("required_changed_strata", 0)) >= 6,
        f"{control} ActionDiT update contract failed",
    )
    visibility = action_update.get("bf16_deployment_category_visibility")
    _require(
        isinstance(visibility, Mapping)
        and set(visibility) == {"early", "mid", "late", "head"}
        and all(value is True for value in visibility.values()),
        f"{control} ActionDiT update is not BF16-visible in all strata",
    )
    head_adapter = updates.get("head_and_adapter")
    _require(
        isinstance(head_adapter, Mapping)
        and head_adapter.get("all_finite") is True
        and float(head_adapter.get("max_abs_delta_by_module", {}).get("content_head", 0.0)) > 0.0
        and float(head_adapter.get("max_abs_delta_by_module", {}).get("adapter", 0.0)) > 0.0,
        f"{control} Head/GCA update contract failed",
    )
    _require(
        sequence.get("status") == "PASS" and contract.get("status") == "PASS",
        f"{control} stream audit failed",
    )
    return {
        "summary": summary,
        "run_config": run_config,
        "gradient": gradient,
        "updates": updates,
        "sequence": sequence,
        "contract": contract,
        "checkpoint": _identity(checkpoint),
        "base_checkpoint_sha256": resolved_base["sha256"],
        "step_rng_rows_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "step": row["step"],
                        "policy": row["step_rng_policy_id"],
                        "official": row["official_rng_seed"],
                        "paired": row["paired_rng_seed"],
                        "data": row["official_data_seed"],
                        "timestep_min": row["action_timestep_min"],
                        "timestep_max": row["action_timestep_max"],
                    }
                    for row in rows
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def _audit_action_gate(path: Path, control: str) -> dict[str, Any]:
    payload = _json(path, f"{control} compact action gate")
    _require(payload.get("status") == "PASS", f"{control} action gate is not PASS")
    _require(
        payload.get("checkpoint_audit", {}).get("action_expert_overlaid") is True,
        f"{control} deployment omitted trained ActionDiT",
    )
    tasks = payload.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 3, f"{control} action gate task count differs")
    _require(
        {row.get("task") for row in tasks} == set(TASKS),
        f"{control} action gate task set differs",
    )
    for row in tasks:
        _require(
            row.get("action_finite") is True
            and row.get("action_shape") == [14]
            and row.get("executed_actions") == 1
            and row.get("zc_finite") is True,
            f"{control} produced an invalid compact deployment action",
        )
    return {"status": "PASS", "identity": _identity(path), "tasks": tasks}


def audit_trained_pair(
    path: str | Path,
    *,
    pair: str,
    require_action_gate: bool = False,
) -> dict[str, Any]:
    _require(pair in {"smoke", "pilot"}, "pair must be smoke or pilot")
    prelaunch = audit_materialization(path)
    manifest = _json(Path(path).expanduser().resolve(), "materialization manifest")
    expected_steps = SMOKE_MAX_STEPS if pair == "smoke" else PILOT_MAX_STEPS
    runs: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for short, control, coefficient in (
        ("c1", "c1_architecture_only", 0.0),
        ("c3", "c3_ours", 0.1),
    ):
        config_path = Path(manifest["configs"][pair][short]["path"]).resolve()
        config = load_config(config_path)
        configs[short] = config
        root = Path(config["output_dir"]).resolve()
        runs[short] = _audit_training_run(
            root,
            control=control,
            expected_steps=expected_steps,
            expected_lambda=coefficient,
        )
        if require_action_gate:
            runs[short]["action_gate"] = _audit_action_gate(
                root / "pre_online_action_gate.json", control
            )
    if pair == "pilot":
        validate_followup_pair(configs["c1"], configs["c3"])
    else:
        validate_c1_c3_pair(configs["c1"], configs["c3"])
    for field in (
        "official_sample_sequence_sha256",
        "paired_physical_state_sequence_sha256",
        "matched_stream_contract_sha256",
    ):
        _require(
            runs["c1"]["summary"].get(field)
            == runs["c3"]["summary"].get(field),
            f"C1/C3 {field} differs",
        )
    _require(
        runs["c1"]["base_checkpoint_sha256"]
        == runs["c3"]["base_checkpoint_sha256"],
        "C1/C3 ActionDiT initialization sources differ",
    )
    _require(
        runs["c1"]["step_rng_rows_sha256"]
        == runs["c3"]["step_rng_rows_sha256"],
        "C1/C3 per-step RNG/timestep sequence differs",
    )
    for field in (
        "source_fp32_content_head_sha256",
        "source_fp32_adapter_sha256",
        "training_fp32_content_head_sha256",
        "training_fp32_adapter_sha256",
    ):
        _require(
            runs["c1"]["summary"]["initialization"].get(field)
            == runs["c3"]["summary"]["initialization"].get(field),
            f"C1/C3 initialization differs at {field}",
        )
    return {
        "status": "PASS",
        "stage": f"{pair}_posttrain",
        "prelaunch": prelaunch,
        "steps_per_control": expected_steps,
        "shared_action_dit_initial_tensor_sha256": prelaunch[
            "action_dit_initial_tensor_sha256"
        ],
        "shared_head_initial_sha256": runs["c1"]["summary"]["initialization"][
            "source_fp32_content_head_sha256"
        ],
        "shared_gca_initial_sha256": runs["c1"]["summary"]["initialization"][
            "source_fp32_adapter_sha256"
        ],
        "shared_sequences": {
            field: runs["c1"]["summary"][field]
            for field in (
                "official_sample_sequence_sha256",
                "paired_physical_state_sequence_sha256",
                "matched_stream_contract_sha256",
            )
        },
        "shared_step_rng_rows_sha256": runs["c1"]["step_rng_rows_sha256"],
        "runs": {
            short: {
                "checkpoint": row["checkpoint"],
                "action_dit_update": row["updates"]["action_dit"],
                "head_gca_update": row["updates"]["head_and_adapter"],
                "final_gate_raw": row["summary"]["final_gate_raw"],
                "action_gate": row.get("action_gate"),
            }
            for short, row in runs.items()
        },
    }


def evaluate_pilot_gate(
    materialization_manifest: str | Path,
    *,
    c1_rollout_manifest: str | Path,
    c3_rollout_manifest: str | Path,
    evaluation_amendment: str | Path,
) -> dict[str, Any]:
    training = audit_trained_pair(
        materialization_manifest, pair="pilot", require_action_gate=True
    )
    amendment, amendment_path = validate_eval100_amendment(evaluation_amendment)
    amendment_identity = _identity(amendment_path)
    manifests: dict[str, dict[str, Any]] = {}
    cells: dict[str, dict[str, dict[str, float]]] = {}
    for short, path_value, expected_control in (
        ("c1", c1_rollout_manifest, "c1_architecture_only"),
        ("c3", c3_rollout_manifest, "c3_ours"),
    ):
        path = Path(path_value).expanduser().resolve()
        payload = _json(path, f"{short} rollout manifest")
        _require(
            payload.get("schema") == "policy_content_adapter.completed_rollouts",
            f"{short} rollout schema differs",
        )
        _require(
            payload.get("schema_version") == 8,
            f"{short} rollout is not the amended 100-episode schema",
        )
        bound_amendment = payload.get("pv2_followup_eval_amendment")
        _require(
            isinstance(bound_amendment, Mapping),
            f"{short} rollout lacks eval100 amendment identity",
        )
        for field in ("path", "size_bytes", "sha256"):
            _require(
                bound_amendment.get(field) == amendment_identity[field],
                f"{short} eval100 amendment {field} differs",
            )
        _require(
            payload.get("pv2_followup_eval_amendment_id")
            == amendment["amendment_id"]
            and payload.get("evaluation_profile") == EVAL100_PROFILE
            and payload.get("episode_pairing") == "not_claimed",
            f"{short} eval100 profile/disclaimer differs",
        )
        contract = payload.get("checkpoint_contract")
        _require(isinstance(contract, Mapping), f"{short} checkpoint contract missing")
        _require(contract.get("control") == expected_control, f"{short} control differs")
        _require(
            contract.get("stage") == "mechanism_followup"
            and isinstance(contract.get("mechanism_protocol_manifest_sha256"), str)
            and len(contract["mechanism_protocol_manifest_sha256"]) == 64,
            f"{short} lacks the mechanism protocol binding",
        )
        _require(contract.get("policy_regime") == "p_v2", f"{short} is not P-v2")
        _require(
            contract.get("training_seed") == PILOT_TRAINING_SEED
            and contract.get("checkpoint_step") == PILOT_MAX_STEPS,
            f"{short} checkpoint seed/step differs",
        )
        _require(
            payload.get("simulator_seed") == EVAL100_SIMULATOR_SEED
            and payload.get("episodes_per_task") == EVAL100_EPISODES_PER_CELL
            and payload.get("simulator_seed_bank_purpose") == "dev_selection",
            f"{short} rollout does not use amended seed53/100-episode bank",
        )
        _require(
            payload.get("simulator_seed_bank_id")
            == amendment["runtime_evaluation"]["seed_bank_id"],
            f"{short} runtime seed bank differs from eval100 amendment",
        )
        runs = payload.get("runs")
        _require(isinstance(runs, list) and len(runs) == 6, f"{short} needs six cells")
        parsed: dict[str, dict[str, float]] = {task: {} for task in TASKS}
        for run in runs:
            task = str(run.get("task", ""))
            domain = str(run.get("domain", ""))
            _require(task in TASKS and domain in DOMAINS, f"{short} has invalid cell")
            _require(
                run.get("episodes") == EVAL100_EPISODES_PER_CELL,
                f"{short} cell episode count differs",
            )
            _require(
                run.get("pv2_followup_eval_amendment_id")
                == amendment["amendment_id"]
                and run.get("superseded_partial_20_episode_results_used") is False,
                f"{short} cell amendment/partial-result contract differs",
            )
            rate = float(run.get("success_rate"))
            _require(math.isfinite(rate) and 0.0 <= rate <= 1.0, f"{short} SR invalid")
            _require(domain not in parsed[task], f"{short} duplicate cell")
            parsed[task][domain] = rate
        _require(
            all(set(row) == set(DOMAINS) for row in parsed.values()),
            f"{short} rollout matrix incomplete",
        )
        manifests[short] = {"identity": _identity(path), "payload": payload}
        cells[short] = parsed
    _require(
        manifests["c1"]["payload"]["simulator_seed_bank_id"]
        == manifests["c3"]["payload"]["simulator_seed_bank_id"],
        "C1/C3 pilot seed-bank IDs differ",
    )
    for field in (
        "mechanism_protocol_manifest_sha256",
        "official_sample_sequence_sha256",
        "paired_physical_state_sequence_sha256",
        "matched_stream_contract_sha256",
    ):
        _require(
            manifests["c1"]["payload"]["checkpoint_contract"].get(field)
            == manifests["c3"]["payload"]["checkpoint_contract"].get(field),
            f"C1/C3 pilot checkpoint contract differs at {field}",
        )
    macro = {
        short: {
            domain: sum(cells[short][task][domain] for task in TASKS) / len(TASKS)
            for domain in DOMAINS
        }
        for short in ("c1", "c3")
    }
    delta = {
        domain: macro["c3"][domain] - macro["c1"][domain]
        for domain in DOMAINS
    }
    random_pass = delta["official_random"] >= PILOT_RANDOM_DELTA_MIN
    clean_pass = delta["clean"] >= PILOT_CLEAN_DELTA_MIN
    passed = random_pass and clean_pass
    return {
        "schema_version": 1,
        "kind": "policy_pv2_actiondit_followup_pilot_decision",
        "status": "PASS",
        "pilot_gate_passed": passed,
        "next_action": (
            "EXPAND_TO_SEEDS_2_3_AND_CONFIRMATORY_SEED59"
            if passed
            else "STOP_EXPANSION_AND_REPORT_FAILURE_MECHANISM"
        ),
        "post_hoc_mechanism_study": True,
        "episode_pairing": "not_claimed_shared_starting_seed_only",
        "training_audit": training,
        "evaluation_amendment": amendment_identity,
        "evaluation_amendment_id": amendment["amendment_id"],
        "episodes_per_task_domain": EVAL100_EPISODES_PER_CELL,
        "episodes_per_control": 600,
        "rollout_manifests": {
            short: manifests[short]["identity"] for short in ("c1", "c3")
        },
        "cells": cells,
        "macro": macro,
        "delta": delta,
        "locked_thresholds": {
            "official_random_macro_delta_min": PILOT_RANDOM_DELTA_MIN,
            "clean_macro_delta_min": PILOT_CLEAN_DELTA_MIN,
            "both_required": True,
        },
        "conditions": {
            "official_random": random_pass,
            "clean": clean_pass,
        },
        "result_driven_tuning_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-manifest", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("materialization", "smoke", "pilot_posttrain", "pilot_gate"),
    )
    parser.add_argument("--c1-rollout-manifest")
    parser.add_argument("--c3-rollout-manifest")
    parser.add_argument("--evaluation-amendment")
    parser.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "materialization":
        result = audit_materialization(args.materialization_manifest)
    elif args.stage == "smoke":
        result = audit_trained_pair(
            args.materialization_manifest, pair="smoke", require_action_gate=True
        )
    elif args.stage == "pilot_posttrain":
        result = audit_trained_pair(
            args.materialization_manifest, pair="pilot", require_action_gate=True
        )
    else:
        _require(args.c1_rollout_manifest is not None, "pilot_gate needs C1 rollout")
        _require(args.c3_rollout_manifest is not None, "pilot_gate needs C3 rollout")
        _require(
            args.evaluation_amendment is not None,
            "pilot_gate needs the 100-episode evaluation amendment",
        )
        result = evaluate_pilot_gate(
            args.materialization_manifest,
            c1_rollout_manifest=args.c1_rollout_manifest,
            c3_rollout_manifest=args.c3_rollout_manifest,
            evaluation_amendment=args.evaluation_amendment,
        )
    if args.output_json:
        _write_new_json(Path(args.output_json).expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Pv2FollowupAuditError",
    "audit_materialization",
    "audit_trained_pair",
    "evaluate_pilot_gate",
]
