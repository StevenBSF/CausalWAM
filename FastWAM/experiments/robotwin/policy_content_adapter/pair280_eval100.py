#!/usr/bin/env python3
"""Audited six-GPU seed-53/100-episode evaluation for Pair-280 seed1/C3."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .pair280_protocol import PAIR280_PROFILE_ID, PAIR280_TOTAL_STEPS
from .pv2_followup_eval100_amendment import (
    _canonical_sha256,
    _file_identity,
    _load_json,
    _verify_identity,
    _write_new_json,
)
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pair280_seed1_c3_eval100_amendment"
SCHEMA_VERSION = 2
PROFILE = "pair280_seed1_c3_dev_seed53_100ep_v2"
SIMULATOR_SEED = 53
EPISODES_PER_CELL = 100
CHECKPOINT_STEP = PAIR280_TOTAL_STEPS
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
DOMAINS = ("clean", "official_random")
ARTIFACT_ROOT = Path("/root/fastwam_policy_artifacts/pair280_layer16_v1").resolve()
RUN_ROOT = ARTIFACT_ROOT / "seed1_c3_pair280_posttraining_v1"
FORMAL_ROOT = RUN_ROOT / "formal"
EVAL_ROOT = RUN_ROOT / "evaluation_seed53_100ep_v2"
DEFAULT_AMENDMENT = EVAL_ROOT / "manifests/pair280_seed1_c3_seed53_eval100_v2.json"
DEFAULT_ROLLOUT_ROOT = EVAL_ROOT / "online_rollouts/c3"
SOURCE_EVAL_AMENDMENT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "pv2_actiondit_full5ep_v1_retry2/manifests/"
    "seed1_c3_dev_seed53_eval100_v3.json"
).resolve()


class Pair280EvalError(ValueError):
    """Pair-280 online evaluation differs from its immutable contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pair280EvalError(message)


def _checkpoint_contract(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _require(isinstance(payload, dict), "Pair-280 checkpoint root is not a mapping")
    _require(
        payload.get("schema") == "fastwam.policy_content_adapter"
        and payload.get("schema_version") == 3
        and payload.get("regime") == "p_v2"
        and payload.get("step") == CHECKPOINT_STEP,
        "Pair-280 checkpoint schema/regime/step differs",
    )
    run = payload.get("run_config")
    artifacts = payload.get("artifact_identities")
    _require(isinstance(run, Mapping), "Pair-280 checkpoint lacks run_config")
    _require(isinstance(artifacts, Mapping), "Pair-280 checkpoint lacks artifacts")
    execution = run.get("execution")
    training = run.get("training")
    loss = run.get("loss")
    evaluation = run.get("evaluation")
    paired = run.get("paired")
    sequence = run.get("resolved_training_sequence_audit")
    _require(isinstance(execution, Mapping), "Pair-280 execution contract missing")
    _require(isinstance(training, Mapping), "Pair-280 training contract missing")
    _require(isinstance(loss, Mapping), "Pair-280 loss contract missing")
    _require(isinstance(evaluation, Mapping), "Pair-280 evaluation contract missing")
    _require(isinstance(paired, Mapping), "Pair-280 paired contract missing")
    _require(isinstance(sequence, Mapping), "Pair-280 sequence audit missing")
    _require(
        run.get("stage") == "mechanism_followup"
        and run.get("formal") is False
        and run.get("control") == "c3_ours"
        and execution.get("runner") == "policy_pair280_posttraining"
        and execution.get("long_formal_training") is True,
        "Pair-280 stage/control/execution contract differs",
    )
    _require(
        training.get("seed") == 1
        and training.get("max_steps") == CHECKPOINT_STEP
        and training.get("effective_official_global_batch") == 128
        and training.get("world_size") == 8,
        "Pair-280 seed/steps/batch/world-size differs",
    )
    _require(
        float(loss.get("lambda_contrastive", -1.0)) == 0.1
        and paired.get("sampling_profile") == PAIR280_PROFILE_ID,
        "Pair-280 loss/sampling profile differs",
    )
    seed_bank = artifacts.get("simulator_seed_bank_manifest")
    cache = artifacts.get("paired_train_cache")
    state_bank = artifacts.get("paired_state_bank")
    text_cache = artifacts.get("paired_text_cache")
    selection = artifacts.get("p_mode_selection_manifest")
    _require(isinstance(seed_bank, Mapping), "checkpoint seed bank identity missing")
    _require(isinstance(cache, Mapping), "checkpoint Pair-280 cache identity missing")
    _require(isinstance(state_bank, Mapping), "checkpoint Pair-280 state bank missing")
    _require(isinstance(text_cache, Mapping), "checkpoint paired text cache missing")
    _require(isinstance(selection, Mapping), "checkpoint P-mode selection missing")
    protocol_path = Path(str(run.get("pair280_protocol_manifest", ""))).resolve()
    protocol = _file_identity(protocol_path)
    _require(
        protocol["sha256"] == run.get("pair280_protocol_manifest_sha256"),
        "Pair-280 protocol SHA differs",
    )
    return {
        "control": "c3_ours",
        "training_seed": 1,
        "checkpoint_step": CHECKPOINT_STEP,
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "formal_evaluation_eligible": False,
        "lambda_contrastive": 0.1,
        "official_global_batch": 128,
        "world_size": 8,
        "pair280_profile": PAIR280_PROFILE_ID,
        "pair280_protocol_sha256": protocol["sha256"],
        "paired_state_bank_sha256": state_bank.get("sha256"),
        "paired_text_cache_sha256": text_cache.get("sha256"),
        "pair280_cache_sha256": cache.get("sha256"),
        "p_mode_selection_manifest": dict(selection),
        "mechanism_protocol_manifest_sha256": run.get("artifacts", {}).get(
            "mechanism_protocol_manifest_sha256"
        ),
        "official_sample_sequence_sha256": sequence.get(
            "official_sample_sequence_sha256"
        ),
        "paired_physical_state_sequence_sha256": sequence.get(
            "paired_physical_state_sequence_sha256"
        ),
        "matched_stream_contract_sha256": sequence.get(
            "matched_stream_contract_sha256"
        ),
        "simulator_seed_bank_id": evaluation.get("simulator_seed_bank_id"),
        "simulator_seed_bank_manifest_sha256": seed_bank.get("sha256"),
        "declared_episodes_per_task": evaluation.get("episodes_per_task"),
    }


def _selection_ancestry_projection(
    checkpoint_contract: Mapping[str, Any],
) -> dict[str, Any]:
    selection_identity = checkpoint_contract.get("p_mode_selection_manifest")
    _verify_identity(selection_identity, "P-mode selection manifest")
    selection, _ = _load_json(
        selection_identity["path"], "P-mode selection manifest"
    )
    shared = selection.get("shared_candidate_identity")
    _require(isinstance(shared, Mapping), "P-mode selection shared identity missing")
    mapping = {
        "paired_state_bank_sha256": "paired_state_bank_sha256",
        "paired_text_cache_sha256": "paired_text_cache_sha256",
        "paired_cache_sha256": "pair280_cache_sha256",
    }
    historical = {field: shared.get(field) for field in mapping}
    effective = {
        field: checkpoint_contract[contract_field]
        for field, contract_field in mapping.items()
    }
    for label, values in (("historical", historical), ("effective", effective)):
        _require(
            all(
                isinstance(value, str) and len(value) == 64
                for value in values.values()
            ),
            f"Pair-280 {label} selection ancestry is incomplete",
        )
    _require(historical != effective, "Pair-280 ancestry projection is unnecessary")
    return {
        "schema": "policy_pair280_selection_ancestry_projection_v1",
        "status": "PASS",
        "allowed_fields": list(mapping),
        "historical_p_mode_values": historical,
        "effective_pair280_values": effective,
        "reason": "Pair-280 intentionally replaces only paired bank/text/cache after P-mode selection",
        "base_official_and_policy_regime_ancestry_changed": False,
        "policy_result_observed_before_projection": False,
    }


def _training_artifacts() -> dict[str, dict[str, Any]]:
    paths = {
        "formal_completion": RUN_ROOT / "audits/formal_completion.json",
        "formal_summary": FORMAL_ROOT / "training_summary.formal_complete.json",
        "gradient_audit": FORMAL_ROOT / "gradient_audit.json",
        "parameter_update_audit": FORMAL_ROOT / "parameter_update_audit.json",
        "training_sequence_audit": FORMAL_ROOT / "training_sequence_audit.json",
        "final_trainer_state": (
            FORMAL_ROOT
            / "checkpoints/state/step_00018215/trainer_state.json"
        ),
    }
    values = {name: _load_json(path, name)[0] for name, path in paths.items()}
    _require(
        values["formal_completion"].get("status") == "PASS"
        and values["formal_completion"].get("formal_steps") == CHECKPOINT_STEP,
        "Pair-280 completion manifest is not PASS",
    )
    _require(
        values["formal_summary"].get("status") == "COMPLETE"
        and values["formal_summary"].get("steps") == CHECKPOINT_STEP,
        "Pair-280 corrected summary is not complete",
    )
    _require(
        values["gradient_audit"].get("status") == "PASS",
        "Pair-280 gradient audit is not PASS",
    )
    final_state = values["final_trainer_state"]
    _require(
        final_state.get("status") == "PASS"
        and final_state.get("global_step") == CHECKPOINT_STEP
        and final_state.get("world_size") == 8,
        "Pair-280 final trainer state is incomplete",
    )
    return {name: _file_identity(path) for name, path in paths.items()}


def materialize(
    *, output: str | Path = DEFAULT_AMENDMENT
) -> tuple[dict[str, Any], Path]:
    destination = Path(output).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    _require(
        not DEFAULT_ROLLOUT_ROOT.exists(),
        "Pair-280 seed53 rollout output already exists",
    )
    source_amendment, source_path = _load_json(
        SOURCE_EVAL_AMENDMENT, "source seed53 evaluation amendment"
    )
    runtime = source_amendment.get("runtime_evaluation")
    _require(isinstance(runtime, Mapping), "source seed53 runtime contract missing")
    _require(
        runtime.get("simulator_seed") == SIMULATOR_SEED
        and runtime.get("episodes_per_task_domain") == EPISODES_PER_CELL
        and runtime.get("tasks") == list(TASKS)
        and runtime.get("domains") == list(DOMAINS),
        "source seed53/100-episode matrix differs",
    )
    runtime_bank = runtime.get("seed_bank")
    _verify_identity(runtime_bank, "runtime seed53 bank")
    checkpoint = FORMAL_ROOT / "checkpoint.pt"
    checkpoint_contract = _checkpoint_contract(checkpoint)
    selection_projection = _selection_ancestry_projection(checkpoint_contract)
    runtime_sources = {
        name: _file_identity(Path(__file__).with_name(name))
        for name in (
            "pair280_eval100.py",
            "eval_robotwin_pair280.py",
            "eval_robotwin_single.py",
            "run_pair280_eval100_multigpu.sh",
        )
    }
    core = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "profile": PROFILE,
        "study_role": "pair280_seed1_c3_development_evaluation",
        "checkpoint": {
            **_file_identity(checkpoint),
            "control": "c3_ours",
            "training_seed": 1,
            "checkpoint_step": CHECKPOINT_STEP,
        },
        "checkpoint_contract": checkpoint_contract,
        "selection_ancestry_projection": selection_projection,
        "training_artifacts": _training_artifacts(),
        "source_seed53_amendment": _file_identity(source_path),
        "runtime_evaluation": {
            "simulator_seed": SIMULATOR_SEED,
            "episodes_per_task_domain": EPISODES_PER_CELL,
            "episodes_per_checkpoint": 600,
            "tasks": list(TASKS),
            "domains": list(DOMAINS),
            "seed_bank": dict(runtime_bank),
            "seed_bank_id": runtime.get("seed_bank_id"),
            "seed_bank_purpose": "dev_selection",
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
        },
        "runtime_sources": runtime_sources,
        "claim_boundary": {
            "development_bank_already_opened": True,
            "pair280_c3_policy_success_rate_may_be_reported": True,
            "matched_pair280_c1_available": False,
            "c3_minus_c1_causal_claim_allowed": False,
            "episode_level_pairing_claimed": False,
            "result_driven_tuning_allowed": False,
        },
    }
    payload = {
        **core,
        "amendment_id": "pair280-eval100-v2:" + _canonical_sha256(core),
    }
    path = _write_new_json(destination, payload)
    return validate(path)


def validate(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, resolved = _load_json(path, "Pair-280 evaluation amendment")
    _require(
        payload.get("kind") == KIND
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "PASS"
        and payload.get("profile") == PROFILE,
        "Pair-280 amendment kind/version/status/profile differs",
    )
    core = dict(payload)
    amendment_id = core.pop("amendment_id", None)
    _require(
        amendment_id == "pair280-eval100-v2:" + _canonical_sha256(core),
        "Pair-280 amendment ID differs",
    )
    for label in ("checkpoint", "source_seed53_amendment"):
        _verify_identity(payload.get(label), label)
    for label, identity in payload.get("training_artifacts", {}).items():
        _verify_identity(identity, f"training artifact {label}")
    for label, identity in payload.get("runtime_sources", {}).items():
        _verify_identity(identity, f"runtime source {label}")
    _verify_identity(payload["runtime_evaluation"]["seed_bank"], "runtime bank")
    contract = _checkpoint_contract(Path(payload["checkpoint"]["path"]))
    _require(contract == payload["checkpoint_contract"], "checkpoint contract changed")
    _require(
        _selection_ancestry_projection(contract)
        == payload.get("selection_ancestry_projection"),
        "Pair-280 selection ancestry projection changed",
    )
    runtime = payload["runtime_evaluation"]
    _require(
        runtime["simulator_seed"] == SIMULATOR_SEED
        and runtime["episodes_per_task_domain"] == EPISODES_PER_CELL
        and runtime["tasks"] == list(TASKS)
        and runtime["domains"] == list(DOMAINS),
        "Pair-280 runtime evaluation matrix differs",
    )
    return payload, resolved


def matching_checkpoint_row(
    amendment: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    control: str,
    training_seed: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    row = amendment.get("checkpoint")
    _require(isinstance(row, Mapping), "Pair-280 checkpoint row missing")
    _require(
        control == "c3_ours"
        and training_seed == 1
        and checkpoint_step == CHECKPOINT_STEP,
        "Pair-280 runtime control/seed/step differs",
    )
    resolved = Path(checkpoint_path).expanduser().resolve()
    _require(str(resolved) == row.get("path"), "Pair-280 checkpoint path differs")
    _require(resolved.stat().st_size == row.get("size_bytes"), "checkpoint size differs")
    return dict(row)


def summarize_shards(
    *,
    amendment: str | Path = DEFAULT_AMENDMENT,
    rollout_root: str | Path = DEFAULT_ROLLOUT_ROOT,
) -> dict[str, Any]:
    payload, amendment_path = validate(amendment)
    root = Path(rollout_root).expanduser().resolve()
    cells: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    manifests: list[dict[str, Any]] = []
    settings_sha: str | None = None
    for task in TASKS:
        for domain in DOMAINS:
            path = root / "cells" / task / domain / "completed_rollouts.json"
            completed, resolved = _load_json(path, f"{task}/{domain} rollout")
            _require(
                completed.get("checkpoint") == payload["checkpoint"]["path"],
                f"{task}/{domain} checkpoint differs",
            )
            _require(
                completed.get("evaluation_profile") == PROFILE
                and completed.get("pv2_followup_eval_amendment_id")
                == payload["amendment_id"],
                f"{task}/{domain} profile/amendment differs",
            )
            runs = completed.get("runs")
            _require(
                isinstance(runs, list) and len(runs) == 1,
                f"{task}/{domain} must contain one run",
            )
            row = runs[0]
            _require(
                row.get("task") == task
                and row.get("domain") == domain
                and row.get("episodes") == EPISODES_PER_CELL,
                f"{task}/{domain} run contract differs",
            )
            rate = float(row.get("success_rate"))
            _require(0.0 <= rate <= 1.0, f"{task}/{domain} success rate invalid")
            current_settings = str(completed.get("rollout_settings_sha256", ""))
            if settings_sha is None:
                settings_sha = current_settings
            _require(current_settings == settings_sha, "GPU shard settings differ")
            cells[task][domain] = rate
            manifests.append(
                {
                    "task": task,
                    "domain": domain,
                    "physical_gpu_index": row.get("physical_gpu_index"),
                    "identity": _file_identity(resolved),
                }
            )
    macro = {
        domain: sum(cells[task][domain] for task in TASKS) / len(TASKS)
        for domain in DOMAINS
    }
    core = {
        "kind": "policy_pair280_seed1_c3_multigpu_eval100_summary",
        "schema_version": 1,
        "status": "PASS",
        "amendment": _file_identity(amendment_path),
        "completed_shards": manifests,
        "rollout_settings_sha256": settings_sha,
        "cells": cells,
        "macro": macro,
        "episodes_per_cell": EPISODES_PER_CELL,
        "total_episodes": 600,
        "execution": "six_independent_task_domain_cells_on_six_gpus",
        "claim_boundary": payload["claim_boundary"],
    }
    result = {
        **core,
        "summary_id": "pair280-multigpu-eval-summary-v2:"
        + _canonical_sha256(core),
    }
    _write_new_json(root / "summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("materialize")
    create.add_argument("--output", default=str(DEFAULT_AMENDMENT))
    check = sub.add_parser("validate")
    check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    report = sub.add_parser("summarize-shards")
    report.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    report.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    args = parser.parse_args(argv)
    if args.command == "materialize":
        payload, path = materialize(output=args.output)
        result: Any = {
            "status": "PASS",
            "path": str(path),
            "amendment_id": payload["amendment_id"],
        }
    elif args.command == "validate":
        payload, path = validate(args.amendment)
        result = {
            "status": "PASS",
            "path": str(path),
            "amendment_id": payload["amendment_id"],
        }
    else:
        result = summarize_shards(
            amendment=args.amendment, rollout_root=args.rollout_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "matching_checkpoint_row",
    "materialize",
    "summarize_shards",
    "validate",
]
