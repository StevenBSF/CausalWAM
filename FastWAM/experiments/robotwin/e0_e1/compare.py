#!/usr/bin/env python3
"""Compare E0 raw, E1 initialized, and E1 trained metric JSON files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_text, write_csv, write_json


REQUIRED_EXPERIMENTS = ("E0-RawBackbone", "E1-InitHead", "E1-TrainedHead")
DEFAULT_STATE_RETENTION = 0.90


def compare_results(
    paths: list[str | Path],
    output_dir: str | Path,
    *,
    min_state_retention: float = DEFAULT_STATE_RETENTION,
    require_success: bool = False,
) -> list[dict[str, Any]]:
    if not 0.0 <= min_state_retention <= 1.0:
        raise ValueError("min_state_retention must be in [0,1]")
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("metrics"), list):
            raise ValueError(f"Not an E0/E1 metric JSON: {path}")
        sources.append(str(path))
        experiment = str(payload.get("experiment"))
        if experiment in payloads:
            raise ValueError(f"duplicate top-level experiment {experiment}")
        payloads[experiment] = payload
        for row in payload["metrics"]:
            if str(row.get("experiment")) != experiment:
                raise ValueError(
                    f"metric row experiment differs from top-level {experiment}"
                )
            rows.append(dict(row))
    experiments = {str(row.get("experiment")) for row in rows}
    missing = set(REQUIRED_EXPERIMENTS) - experiments
    if missing:
        raise ValueError(f"comparison is missing experiments {sorted(missing)}")
    unexpected = set(payloads) - set(REQUIRED_EXPERIMENTS)
    if unexpected:
        raise ValueError(f"comparison has unexpected experiments {sorted(unexpected)}")
    evaluation_splits = {
        str(payloads[name].get("evaluation_split")) for name in REQUIRED_EXPERIMENTS
    }
    if evaluation_splits != {"test"}:
        raise ValueError(
            "final comparison requires E0/Init/Trained metrics from the held-out "
            f"test split, got {sorted(evaluation_splits)}"
        )
    identities = [payloads[name].get("cache_identity") for name in REQUIRED_EXPERIMENTS]
    if any(not isinstance(identity, Mapping) for identity in identities):
        raise ValueError("metric JSON is missing cache_identity")
    if any(dict(identity) != dict(identities[0]) for identity in identities[1:]):
        raise ValueError("comparison metrics come from different test caches")
    filters = [payloads[name].get("negative_filter") for name in REQUIRED_EXPERIMENTS]
    if any(current != filters[0] for current in filters[1:]):
        raise ValueError("comparison metrics use different negative filters")
    init_head = payloads["E1-InitHead"].get("head")
    trained_head = payloads["E1-TrainedHead"].get("head")
    if not isinstance(init_head, Mapping) or not isinstance(trained_head, Mapping):
        raise ValueError("E1 metric JSON is missing head provenance")
    if init_head.get("initial_head_sha256") != trained_head.get("initial_head_sha256"):
        raise ValueError("E1-InitHead is not the initialization used for training")
    if init_head.get("initialization_seed") != trained_head.get("training_seed"):
        raise ValueError("E1 initialization/training seeds differ")
    layers = {str(row.get("layer")) for row in rows}
    if len(layers) != 1:
        raise ValueError(f"comparison must use one backbone layer, got {sorted(layers)}")
    requested_layers = {
        f"video_block_{int(payloads[name]['layer']):02d}"
        for name in REQUIRED_EXPERIMENTS
    }
    if requested_layers != layers:
        raise ValueError("top-level and metric-row layers disagree")
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["experiment"]))
        if key in keyed:
            raise ValueError(f"duplicate metric row {key}")
        keyed[key] = row
    tasks = sorted({key[0] for key in keyed})
    task_sets = [
        {
            str(row["task"])
            for row in payloads[name]["metrics"]
        }
        for name in REQUIRED_EXPERIMENTS
    ]
    if any(current != task_sets[0] for current in task_sets[1:]):
        raise ValueError("comparison metrics have different task sets")
    for task in tasks:
        if any((task, experiment) not in keyed for experiment in REQUIRED_EXPERIMENTS):
            raise ValueError(f"task {task} is missing one comparison experiment")

    summary: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    for task in tasks:
        init = keyed[(task, "E1-InitHead")]
        trained = keyed[(task, "E1-TrainedHead")]
        style_improved = float(trained["style_distance"]) < float(
            init["style_distance"]
        )
        state_retention = float(trained["state_distance"]) / max(
            float(init["state_distance"]), 1e-8
        )
        state_preserved = state_retention >= min_state_retention
        ratio_improved = float(trained["state_style_ratio"]) > float(
            init["state_style_ratio"]
        )
        retrieval_improved = float(trained["retrieval_r1"]) > float(
            init["retrieval_r1"]
        )
        task_success = all(
            (style_improved, state_preserved, ratio_improved, retrieval_improved)
        )
        success_rows.append(
            {
                "task": task,
                "style_improved": style_improved,
                "state_retention": state_retention,
                "state_preserved": state_preserved,
                "ratio_improved": ratio_improved,
                "retrieval_improved": retrieval_improved,
                "success": task_success,
            }
        )
        for experiment in REQUIRED_EXPERIMENTS:
            row = keyed[(task, experiment)]
            summary.append(
                {
                    "task": task,
                    "layer": row["layer"],
                    "experiment": experiment,
                    "style_distance": row["style_distance"],
                    "state_distance": row["state_distance"],
                    "state_style_ratio": row["state_style_ratio"],
                    "retrieval_r1": row["retrieval_r1"],
                    "retrieval_r5": row["retrieval_r5"],
                    "trained_minus_init_ratio": (
                        float(trained["state_style_ratio"]) - float(init["state_style_ratio"])
                    ),
                    "trained_minus_init_retrieval_r1": (
                        float(trained["retrieval_r1"]) - float(init["retrieval_r1"])
                    ),
                    "e1_success": task_success,
                }
            )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    overall_success = all(row["success"] for row in success_rows)
    write_json(
        destination / "comparison.json",
        {
            "sources": sources,
            "min_state_retention": min_state_retention,
            "overall_success": overall_success,
            "success_criteria": success_rows,
            "rows": summary,
        },
    )
    write_csv(destination / "comparison.csv", summary)
    header = "| Task | Experiment | Style | State | Ratio | R@1 | ΔRatio trained-init |\n"
    separator = "|---|---|---:|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {row['task']} | {row['experiment']} | {row['style_distance']:.6f} | "
        f"{row['state_distance']:.6f} | {row['state_style_ratio']:.3f} | "
        f"{row['retrieval_r1']:.3f} | {row['trained_minus_init_ratio']:.3f} |\n"
        for row in summary
    )
    gate = (
        f"\nOverall E1 success gate: **{'PASS' if overall_success else 'FAIL'}** "
        f"(minimum state retention {min_state_retention:.0%}).\n"
    )
    atomic_write_text(destination / "summary.md", header + separator + body + gate)
    if require_success and not overall_success:
        failed = [row for row in success_rows if not row["success"]]
        raise RuntimeError(f"E1 success gate failed: {failed}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--min-state-retention", type=float, default=DEFAULT_STATE_RETENTION
    )
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    rows = compare_results(
        args.metrics,
        args.output_dir,
        min_state_retention=args.min_state_retention,
        require_success=args.require_success,
    )
    print(f"wrote {len(rows)} comparison rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
