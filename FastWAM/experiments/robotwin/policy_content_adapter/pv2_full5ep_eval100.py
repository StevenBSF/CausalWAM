"""Audited seed-53 development evaluation for the completed full-5-epoch C3.

The seed-53 bank has already been opened by the 1,800-step mechanism pilot, so
reusing it here measures the training-length trend without opening the reserved
seed-59 confirmatory bank.  This artifact is development evidence only; a
matched full-5-epoch C1 must later use the same bank, while the unopened seed-59
bank remains reserved for the final C1/C3 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from .full5ep_resume_amendment import verify_amendment as verify_resume_amendment
from .pv2_followup_eval100_amendment import (
    _canonical_sha256,
    _file_identity,
    _load_json,
    _verify_identity,
    _write_new_json,
    validate_eval100_amendment,
)
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pv2_full5ep_seed1_c3_eval100_amendment"
SCHEMA_VERSION = 1
PROFILE = "pv2_full5ep_seed1_c3_dev_seed53_100ep_v1"
SIMULATOR_SEED = 53
EPISODES_PER_CELL = 100
CHECKPOINT_STEP = 18_215
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
DOMAINS = ("clean", "official_random")
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "pv2_actiondit_full5ep_v1_retry2"
).resolve()
SHORT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "pv2_actiondit_followup_v1"
).resolve()
DEFAULT_AMENDMENT = DEFAULT_ROOT / "manifests/seed1_c3_dev_seed53_eval100_v1.json"
DEFAULT_ROLLOUT_ROOT = DEFAULT_ROOT / "online_rollouts_dev_seed53_v1/c3"
DEFAULT_PARALLEL_ROOT = DEFAULT_ROOT / "online_rollouts_dev_seed53_multigpu_v1/c3"


class Full5EpochEvalError(ValueError):
    """The full-5-epoch development evaluation contract is not proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Full5EpochEvalError(message)


def _semantic_sha(path: Path) -> str:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    _require(isinstance(value, dict), "full5ep config root must be a mapping")
    return _canonical_sha256(value)


def _checkpoint_contract(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _require(isinstance(payload, dict), "policy checkpoint root must be a mapping")
    _require(
        payload.get("schema") == "fastwam.policy_content_adapter"
        and payload.get("schema_version") == 3
        and payload.get("regime") == "p_v2"
        and payload.get("step") == CHECKPOINT_STEP,
        "checkpoint schema/regime/step differs",
    )
    run = payload.get("run_config")
    _require(isinstance(run, Mapping), "checkpoint lacks run_config")
    _require(
        run.get("stage") == "mechanism_followup"
        and run.get("formal") is False
        and run.get("control") == "c3_ours",
        "checkpoint stage/formal/control differs",
    )
    training = run.get("training")
    loss = run.get("loss")
    evaluation = run.get("evaluation")
    _require(isinstance(training, Mapping), "checkpoint training block missing")
    _require(isinstance(loss, Mapping), "checkpoint loss block missing")
    _require(isinstance(evaluation, Mapping), "checkpoint evaluation block missing")
    _require(
        training.get("seed") == 1
        and training.get("max_steps") == CHECKPOINT_STEP
        and training.get("effective_official_global_batch") == 128
        and loss.get("lambda_contrastive") == 0.1,
        "checkpoint seed/steps/batch/lambda differs",
    )
    artifacts = payload.get("artifact_identities")
    _require(isinstance(artifacts, Mapping), "checkpoint artifact identities missing")
    seed_bank = artifacts.get("simulator_seed_bank_manifest")
    _require(isinstance(seed_bank, Mapping), "checkpoint original seed bank missing")
    return {
        "control": "c3_ours",
        "training_seed": 1,
        "checkpoint_step": CHECKPOINT_STEP,
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "formal_evaluation_eligible": False,
        "mechanism_protocol_manifest_sha256": run.get("artifacts", {}).get(
            "mechanism_protocol_manifest_sha256"
        ),
        "full5ep_protocol_manifest": run.get("full5ep_protocol_manifest"),
        "full5ep_protocol_manifest_sha256": run.get(
            "full5ep_protocol_manifest_sha256"
        ),
        "simulator_seed_bank_id": evaluation.get("simulator_seed_bank_id"),
        "simulator_seed_bank_manifest_sha256": seed_bank.get("sha256"),
        "declared_episodes_per_task": evaluation.get("episodes_per_task"),
    }


def _training_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    run = root / "runs/seed_1/c3"
    paths = {
        "training_summary": run / "training_summary.json",
        "gradient_audit": run / "gradient_audit.json",
        "parameter_update_audit": run / "parameter_update_audit.json",
        "action_gate": run / "pre_online_action_gate.json",
        "final_trainer_state": (
            run / "checkpoints/state/step_00018215/trainer_state.json"
        ),
    }
    values = {name: _load_json(path, name)[0] for name, path in paths.items()}
    _require(
        values["training_summary"].get("steps") == CHECKPOINT_STEP
        and values["training_summary"].get("control") == "c3_ours"
        and values["training_summary"].get("regime") == "p_v2",
        "training summary does not prove completed seed1/C3 P-v2",
    )
    _require(
        values["gradient_audit"].get("status") == "PASS",
        "gradient audit is not PASS",
    )
    _require(
        values["action_gate"].get("status") == "PASS",
        "pre-online action gate is not PASS",
    )
    state = values["final_trainer_state"]
    _require(
        state.get("status") == "PASS"
        and state.get("global_step") == CHECKPOINT_STEP
        and state.get("max_steps") == CHECKPOINT_STEP
        and state.get("world_size") == 8,
        "final native trainer state is incomplete",
    )
    return {name: _file_identity(path) for name, path in paths.items()}


def materialize(
    *,
    root: str | Path = DEFAULT_ROOT,
    output: str | Path = DEFAULT_AMENDMENT,
) -> tuple[dict[str, Any], Path]:
    experiment = Path(root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite amendment: {destination}")
    _require(
        not DEFAULT_PARALLEL_ROOT.exists(),
        "full5ep multi-GPU seed53 rollout already exists",
    )

    old_amendment, old_path = validate_eval100_amendment(
        SHORT_ROOT / "manifests/eval100_user_amendment_v1.json"
    )
    _require(
        old_amendment.get("runtime_evaluation", {}).get("simulator_seed")
        == SIMULATOR_SEED
        and old_amendment.get("runtime_evaluation", {}).get(
            "episodes_per_task_domain"
        )
        == EPISODES_PER_CELL,
        "existing seed53/100 development bank differs",
    )
    checkpoint = experiment / "runs/seed_1/c3/checkpoint.pt"
    checkpoint_contract = _checkpoint_contract(checkpoint)
    config = experiment / "configs/seed_1/c3.yaml"
    config_sha = _semantic_sha(config)
    final_state = _load_json(
        experiment
        / "runs/seed_1/c3/checkpoints/state/step_00018215/trainer_state.json",
        "final trainer state",
    )[0]
    _require(
        final_state.get("requested_config_sha256") == config_sha,
        "final trainer state binds another config",
    )
    protocol = Path(str(checkpoint_contract["full5ep_protocol_manifest"])).resolve()
    protocol_identity = _file_identity(protocol)
    _require(
        protocol_identity["sha256"]
        == checkpoint_contract["full5ep_protocol_manifest_sha256"],
        "checkpoint full5ep protocol SHA differs",
    )
    resume_amendment = experiment / "manifests/step6803_resume_amendment_v3.json"
    verify_resume_amendment(resume_amendment)
    runtime_bank = old_amendment["runtime_evaluation"]["seed_bank"]
    original_bank = old_amendment["original_evaluation"]["seed_bank"]
    _require(
        checkpoint_contract["simulator_seed_bank_manifest_sha256"]
        == original_bank["sha256"],
        "full5ep checkpoint original seed bank differs",
    )
    checkpoint_identity = _file_identity(checkpoint)
    sources = {
        name: _file_identity(Path(__file__).with_name(name))
        for name in (
            "pv2_full5ep_eval100.py",
            "eval_robotwin_pv2_full5ep.py",
            "eval_robotwin_single.py",
        )
    }
    core = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "profile": PROFILE,
        "study_role": "development_training_length_evaluation",
        "experiment_root": str(experiment),
        "checkpoint": {
            **checkpoint_identity,
            "control": "c3_ours",
            "training_seed": 1,
            "checkpoint_step": CHECKPOINT_STEP,
        },
        "checkpoint_contract": checkpoint_contract,
        "config": {**_file_identity(config), "semantic_sha256": config_sha},
        "full5ep_protocol": protocol_identity,
        "resume_amendment": _file_identity(resume_amendment),
        "training_artifacts": _training_artifacts(experiment),
        "source_eval100_amendment": _file_identity(old_path),
        "original_evaluation": {
            "simulator_seed": SIMULATOR_SEED,
            "episodes_per_task_domain": 20,
            "seed_bank": original_bank,
            "seed_bank_id": old_amendment["original_evaluation"]["seed_bank_id"],
        },
        "runtime_evaluation": {
            "simulator_seed": SIMULATOR_SEED,
            "episodes_per_task_domain": EPISODES_PER_CELL,
            "episodes_per_checkpoint": 600,
            "tasks": list(TASKS),
            "domains": list(DOMAINS),
            "seed_bank": runtime_bank,
            "seed_bank_id": old_amendment["runtime_evaluation"]["seed_bank_id"],
            "seed_bank_purpose": "dev_selection",
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
        },
        "runtime_sources": sources,
        "claim_boundary": {
            "development_bank_already_opened_by_1800_step_pilot": True,
            "reserved_seed59_confirmatory_bank_opened": False,
            "matched_full5ep_c1_available": False,
            "c3_minus_c1_claim_allowed": False,
            "result_driven_tuning_allowed": False,
        },
    }
    payload = {
        **core,
        "amendment_id": "pv2-full5ep-eval100-v1:" + _canonical_sha256(core),
    }
    path = _write_new_json(destination, payload)
    return validate(path)


def validate(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, resolved = _load_json(path, "full5ep evaluation amendment")
    _require(
        payload.get("kind") == KIND
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "PASS"
        and payload.get("profile") == PROFILE,
        "amendment kind/version/status/profile differs",
    )
    core = dict(payload)
    amendment_id = core.pop("amendment_id", None)
    _require(
        amendment_id == "pv2-full5ep-eval100-v1:" + _canonical_sha256(core),
        "amendment ID differs",
    )
    for label in (
        "checkpoint",
        "config",
        "full5ep_protocol",
        "resume_amendment",
        "source_eval100_amendment",
    ):
        _verify_identity(payload.get(label), label)
    for label, identity in payload.get("training_artifacts", {}).items():
        _verify_identity(identity, f"training artifact {label}")
    for label, identity in payload.get("runtime_sources", {}).items():
        _verify_identity(identity, f"runtime source {label}")
    _verify_identity(payload["original_evaluation"]["seed_bank"], "original bank")
    _verify_identity(payload["runtime_evaluation"]["seed_bank"], "runtime bank")
    contract = _checkpoint_contract(Path(payload["checkpoint"]["path"]))
    _require(contract == payload["checkpoint_contract"], "checkpoint contract changed")
    _require(
        payload["runtime_evaluation"]["simulator_seed"] == SIMULATOR_SEED
        and payload["runtime_evaluation"]["episodes_per_task_domain"]
        == EPISODES_PER_CELL
        and payload["runtime_evaluation"]["tasks"] == list(TASKS)
        and payload["runtime_evaluation"]["domains"] == list(DOMAINS),
        "runtime evaluation matrix differs",
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
    _require(isinstance(row, Mapping), "amendment checkpoint row missing")
    _require(
        control == "c3_ours"
        and training_seed == 1
        and checkpoint_step == CHECKPOINT_STEP,
        "runtime checkpoint control/seed/step differs",
    )
    resolved = Path(checkpoint_path).expanduser().resolve()
    _require(resolved.is_file(), f"checkpoint missing: {resolved}")
    _require(str(resolved) == row.get("path"), "checkpoint path differs")
    _require(resolved.stat().st_size == row.get("size_bytes"), "checkpoint size differs")
    # validate() immediately re-hashes the checkpoint against row.sha256.  Do
    # not hash the 12-GB file a second time in the same evaluator process.
    return dict(row)


def summarize(
    *, amendment: str | Path = DEFAULT_AMENDMENT, rollout_root: str | Path = DEFAULT_ROLLOUT_ROOT
) -> dict[str, Any]:
    payload, amendment_path = validate(amendment)
    root = Path(rollout_root).expanduser().resolve()
    completed, completed_path = _load_json(
        root / "completed_rollouts.json", "completed full5ep rollout"
    )
    _require(completed.get("checkpoint") == payload["checkpoint"]["path"], "rollout checkpoint differs")
    _require(completed.get("evaluation_profile") == PROFILE, "rollout profile differs")
    _require(
        completed.get("pv2_followup_eval_amendment_id") == payload["amendment_id"],
        "rollout amendment ID differs",
    )
    runs = completed.get("runs")
    _require(isinstance(runs, list) and len(runs) == 6, "rollout must contain six cells")
    cells: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    for row in runs:
        task = str(row.get("task"))
        domain = str(row.get("domain"))
        rate = float(row.get("success_rate"))
        _require(task in TASKS and domain in DOMAINS, "rollout cell task/domain differs")
        _require(row.get("episodes") == EPISODES_PER_CELL, "rollout episode count differs")
        _require(0.0 <= rate <= 1.0 and domain not in cells[task], "rollout success cell invalid")
        cells[task][domain] = rate
    _require(all(set(value) == set(DOMAINS) for value in cells.values()), "rollout matrix incomplete")
    macro = {
        domain: sum(cells[task][domain] for task in TASKS) / len(TASKS)
        for domain in DOMAINS
    }
    core = {
        "kind": "policy_pv2_full5ep_seed1_c3_eval100_summary",
        "schema_version": 1,
        "status": "PASS",
        "amendment": _file_identity(amendment_path),
        "completed_rollouts": _file_identity(completed_path),
        "cells": cells,
        "macro": macro,
        "episodes_per_cell": EPISODES_PER_CELL,
        "total_episodes": 600,
        "claim_boundary": payload["claim_boundary"],
    }
    result = {**core, "summary_id": "pv2-full5ep-eval-summary-v1:" + _canonical_sha256(core)}
    output = root / "summary.json"
    _write_new_json(output, result)
    return result


def summarize_shards(
    *,
    amendment: str | Path = DEFAULT_AMENDMENT,
    rollout_root: str | Path = DEFAULT_PARALLEL_ROOT,
) -> dict[str, Any]:
    payload, amendment_path = validate(amendment)
    root = Path(rollout_root).expanduser().resolve()
    cells: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    manifests: list[dict[str, Any]] = []
    settings_sha: str | None = None
    for task in TASKS:
        for domain in DOMAINS:
            path = root / "cells" / task / domain / "completed_rollouts.json"
            completed, resolved = _load_json(path, f"{task}/{domain} completed rollout")
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
                f"{task}/{domain} shard must contain exactly one run",
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
            _require(
                current_settings == settings_sha,
                "rollout settings differ across GPU shards",
            )
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
        "kind": "policy_pv2_full5ep_seed1_c3_multigpu_eval100_summary",
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
        "summary_id": "pv2-full5ep-multigpu-eval-summary-v1:"
        + _canonical_sha256(core),
    }
    _write_new_json(root / "summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("materialize")
    create.add_argument("--root", default=str(DEFAULT_ROOT))
    create.add_argument("--output", default=str(DEFAULT_AMENDMENT))
    check = sub.add_parser("validate")
    check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    report = sub.add_parser("summarize")
    report.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    report.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    shard_report = sub.add_parser("summarize-shards")
    shard_report.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    shard_report.add_argument("--rollout-root", default=str(DEFAULT_PARALLEL_ROOT))
    args = parser.parse_args(argv)
    if args.command == "materialize":
        payload, path = materialize(root=args.root, output=args.output)
        result: Any = {"status": "PASS", "path": str(path), "amendment_id": payload["amendment_id"]}
    elif args.command == "validate":
        payload, path = validate(args.amendment)
        result = {"status": "PASS", "path": str(path), "amendment_id": payload["amendment_id"]}
    elif args.command == "summarize":
        result = summarize(amendment=args.amendment, rollout_root=args.rollout_root)
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
    "summarize",
    "summarize_shards",
    "validate",
]
