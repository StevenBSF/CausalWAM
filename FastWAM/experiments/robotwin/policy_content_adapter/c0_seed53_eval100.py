"""Audit and summarize six-GPU author-release C0 seed-53 evaluation shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .pv2_followup_eval100_amendment import (
    _canonical_sha256,
    _file_identity,
    _load_json,
    _write_new_json,
)


TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
DOMAINS = ("clean", "official_random")
SIMULATOR_SEED = 53
EPISODES_PER_CELL = 100


class C0Seed53EvaluationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C0Seed53EvaluationError(message)


def _transport_contract(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
    _require(isinstance(payload, dict), "C0 transport checkpoint root must be a mapping")
    _require(
        payload.get("schema") == "fastwam.policy_content_adapter"
        and payload.get("schema_version") == 3
        and payload.get("regime") == "p_v1"
        and payload.get("step") == 0
        and "action_expert" not in payload,
        "C0 transport schema/regime/step/action overlay differs",
    )
    run = payload.get("run_config")
    _require(isinstance(run, Mapping), "C0 transport lacks run_config")
    semantics = run.get("c0_semantics")
    evaluation = run.get("evaluation")
    _require(isinstance(semantics, Mapping), "C0 semantics missing")
    _require(isinstance(evaluation, Mapping), "C0 evaluation contract missing")
    _require(
        run.get("kind") == "policy_c0_eval_transport"
        and run.get("control") == "c0_original"
        and run.get("formal") is False
        and run.get("training", {}).get("stage2_steps") == 0
        and semantics.get("head_gca_effect_on_action") == "none_exact_zero_gate"
        and semantics.get("action_expert_overlay") is False,
        "checkpoint does not prove native zero-Stage-2 C0 semantics",
    )
    _require(
        evaluation.get("simulator_seed_bank_purpose") == "development_analysis"
        and evaluation.get("episodes_per_task") == EPISODES_PER_CELL
        and evaluation.get("tasks") == list(TASKS)
        and evaluation.get("required_domains") == list(DOMAINS),
        "C0 seed53 evaluation matrix differs",
    )
    return {
        "checkpoint": _file_identity(resolved),
        "simulator_seed_bank_id": evaluation["simulator_seed_bank_id"],
        "simulator_seed_bank_manifest": evaluation[
            "simulator_seed_bank_manifest"
        ],
        "base_checkpoint_sha256": run["artifacts"]["base_checkpoint_sha256"],
        "dataset_stats_sha256": run["artifacts"]["dataset_stats_sha256"],
        "runtime_identity_audit_sha256": semantics[
            "runtime_identity_audit_sha256"
        ],
    }


def summarize(
    *,
    checkpoint: str | Path,
    rollout_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    contract = _transport_contract(checkpoint)
    root = Path(rollout_root).expanduser().resolve()
    cells: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    manifests: list[dict[str, Any]] = []
    settings_sha: str | None = None
    for task in TASKS:
        for domain in DOMAINS:
            path = root / "cells" / task / domain / "completed_rollouts.json"
            completed, resolved = _load_json(path, f"C0 {task}/{domain} rollout")
            _require(
                completed.get("checkpoint") == contract["checkpoint"]["path"],
                f"C0 {task}/{domain} checkpoint differs",
            )
            checkpoint_contract = completed.get("checkpoint_contract")
            _require(
                isinstance(checkpoint_contract, Mapping)
                and checkpoint_contract.get("control") == "c0_original"
                and checkpoint_contract.get("checkpoint_step") == 0
                and checkpoint_contract.get("formal_evaluation_eligible") is False,
                f"C0 {task}/{domain} checkpoint contract differs",
            )
            runs = completed.get("runs")
            _require(
                isinstance(runs, list) and len(runs) == 1,
                f"C0 {task}/{domain} shard must contain one run",
            )
            row = runs[0]
            _require(
                row.get("task") == task
                and row.get("domain") == domain
                and row.get("episodes") == EPISODES_PER_CELL
                and row.get("simulator_seed") == SIMULATOR_SEED,
                f"C0 {task}/{domain} run contract differs",
            )
            _require(
                row.get("simulator_seed_bank_id")
                == contract["simulator_seed_bank_id"],
                f"C0 {task}/{domain} seed bank differs",
            )
            rate = float(row.get("success_rate"))
            _require(0.0 <= rate <= 1.0, f"C0 {task}/{domain} rate invalid")
            current_settings = str(completed.get("rollout_settings_sha256", ""))
            if settings_sha is None:
                settings_sha = current_settings
            _require(current_settings == settings_sha, "C0 shard rollout settings differ")
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
        "kind": "policy_author_release_c0_seed53_multigpu_eval100_summary",
        "schema_version": 1,
        "status": "PASS",
        "transport_contract": contract,
        "completed_shards": manifests,
        "rollout_settings_sha256": settings_sha,
        "cells": cells,
        "macro": macro,
        "episodes_per_cell": EPISODES_PER_CELL,
        "total_episodes": 600,
        "episode_pairing": "not_claimed",
        "reference_role": "author_release_same_seed53_development_reference",
    }
    result = {
        **core,
        "summary_id": "c0-seed53-multigpu-eval100-v1:" + _canonical_sha256(core),
    }
    _write_new_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(
        checkpoint=args.checkpoint,
        rollout_root=args.rollout_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
