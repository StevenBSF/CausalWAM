"""Strict 100-episode Motus M1/M3 RoboTwin result aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from .protocol import TASKS


CELL_SCHEMA = "motus_policy_content_adapter_rollout_cell"
SUMMARY_SCHEMA = "motus_policy_content_adapter_evaluation_summary"
CONTROLS = ("m1_architecture_action_control", "m3_ours")
DOMAINS = ("clean", "official_random")


class EvaluationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def validate_cell(cell: Mapping[str, Any]) -> None:
    _require(cell.get("schema") == CELL_SCHEMA, "rollout cell schema changed")
    _require(cell.get("schema_version") == 1, "rollout cell version changed")
    _require(cell.get("status") == "PASS", "rollout cell is not PASS")
    _require(cell.get("control") in CONTROLS, "rollout control changed")
    _require(cell.get("task") in TASKS, "rollout task changed")
    _require(cell.get("domain") in DOMAINS, "rollout domain changed")
    _require(int(cell.get("training_seed", -1)) >= 0, "training seed is invalid")
    _require(cell.get("episode_count") == 100, "each cell must contain 100 episodes")
    successes = int(cell.get("success_count", -1))
    _require(0 <= successes <= 100, "success count is invalid")
    _require(float(cell.get("success_rate", -1)) == successes / 100.0, "success rate/count mismatch")
    _require(cell.get("episode_pairing") == "shared_start_seed_not_exact_pairing", "episode-pairing claim changed")
    for name in ("checkpoint_sha256", "evaluation_settings_sha256"):
        value = str(cell.get(name, ""))
        _require(len(value) == 64, f"{name} is invalid")


def aggregate_cells(
    cells: Sequence[Mapping[str, Any]], *, formal_training_seeds: Sequence[int]
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in formal_training_seeds)
    _require(len(seeds) >= 1 and len(set(seeds)) == len(seeds), "formal seeds are invalid")
    by_key: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    settings_sha = None
    for cell in cells:
        validate_cell(cell)
        key = (
            str(cell["control"]),
            int(cell["training_seed"]),
            str(cell["task"]),
            str(cell["domain"]),
        )
        _require(key not in by_key, f"duplicate rollout cell {key}")
        by_key[key] = cell
        if settings_sha is None:
            settings_sha = cell["evaluation_settings_sha256"]
        _require(cell["evaluation_settings_sha256"] == settings_sha, "evaluation settings differ")
    expected = {
        (control, seed, task, domain)
        for control in CONTROLS
        for seed in seeds
        for task in TASKS
        for domain in DOMAINS
    }
    _require(set(by_key) == expected, "rollout matrix is incomplete or has extra cells")
    rows = []
    macro_by_control_domain: dict[str, dict[str, dict[str, float]]] = {
        control: {domain: {} for domain in DOMAINS} for control in CONTROLS
    }
    for control in CONTROLS:
        for seed in seeds:
            for domain in DOMAINS:
                task_values = {
                    task: float(by_key[(control, seed, task, domain)]["success_rate"])
                    for task in TASKS
                }
                macro = mean(task_values.values())
                macro_by_control_domain[control][domain][str(seed)] = macro
                rows.append(
                    {
                        "control": control,
                        "training_seed": seed,
                        "domain": domain,
                        "task_success_rates": task_values,
                        "macro_success_rate": macro,
                    }
                )
    deltas = {}
    for domain in DOMAINS:
        values = [
            macro_by_control_domain["m3_ours"][domain][str(seed)]
            - macro_by_control_domain["m1_architecture_action_control"][domain][str(seed)]
            for seed in seeds
        ]
        deltas[domain] = {
            "per_training_seed": {
                str(seed): value for seed, value in zip(seeds, values, strict=True)
            },
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }
    summaries = {}
    for control in CONTROLS:
        summaries[control] = {}
        for domain in DOMAINS:
            values = list(macro_by_control_domain[control][domain].values())
            summaries[control][domain] = {
                "per_training_seed": macro_by_control_domain[control][domain],
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
    return {
        "schema": SUMMARY_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "formal_training_seeds": list(seeds),
        "tasks": list(TASKS),
        "domains": list(DOMAINS),
        "episodes_per_cell": 100,
        "cell_count": len(cells),
        "episode_pairing": "shared_start_seed_not_exact_pairing",
        "evaluation_settings_sha256": settings_sha,
        "rows": rows,
        "control_macro": summaries,
        "m3_minus_m1_macro": deltas,
        "primary_comparison": "m3_ours_minus_m1_architecture_action_control",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cells = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.cells]
    result = aggregate_cells(cells, formal_training_seeds=[int(item) for item in args.seeds.split(",")])
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cells": result["cell_count"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
