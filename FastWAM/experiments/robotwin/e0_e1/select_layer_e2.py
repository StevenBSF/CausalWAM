#!/usr/bin/env python3
"""Select E2's layer using only seen-style C/R1/R2 validation metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_text, write_csv, write_json


PROTOCOL = "r3_holdout_v1"
EXPECTED_EXPERIMENT = "E2-RawBackbone"
EXPECTED_CANDIDATES = (8, 16, 24)
EXPECTED_VARIANTS = (
    "clean",
    "style_00_seed_0",
    "style_01_seed_1",
)
HOLDOUT_VARIANT = "style_02_seed_2"
MACRO_SUFFIX = "-task-average"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, *, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {field} must be numeric") from error
    _require(math.isfinite(result), f"{source}: {field} must be finite")
    return result


def _load_candidate(path_value: str | Path) -> dict[str, Any]:
    source = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read E2 metric JSON {source}: {error}") from error
    _require(isinstance(payload, Mapping), f"{source}: metric root must be an object")
    _require(payload.get("protocol") == PROTOCOL, f"{source}: protocol mismatch")
    _require(payload.get("evaluation_split") == "val", f"{source}: selection is val-only")
    _require(payload.get("experiment") == EXPECTED_EXPERIMENT, f"{source}: wrong experiment")
    _require(payload.get("proprio_mode") == "observed", f"{source}: E2 must use observed proprio")
    active = tuple(payload.get("active_variants", ()))
    _require(active == EXPECTED_VARIANTS, f"{source}: selection variants must be C/R1/R2")
    _require(payload.get("holdout_variant") == HOLDOUT_VARIANT, f"{source}: holdout mismatch")
    _require(HOLDOUT_VARIANT not in active, f"{source}: R3 leaked into layer selection")
    provenance = payload.get("cache_provenance")
    _require(isinstance(provenance, Mapping), f"{source}: missing cache provenance")
    _require(provenance.get("split") == "val", f"{source}: upstream cache is not val")
    _require(provenance.get("protocol") == PROTOCOL, f"{source}: cache protocol mismatch")
    _require(tuple(provenance.get("active_variants", ())) == EXPECTED_VARIANTS,
             f"{source}: cache is not strict C/R1/R2")
    _require(provenance.get("proprio_mode") == "observed",
             f"{source}: selection cache is not E2 observed-proprio")
    records_variants = set(payload.get("record_variants", active))
    _require(records_variants == set(EXPECTED_VARIANTS),
             f"{source}: metric records are not exactly C/R1/R2")
    identity = payload.get("cache_identity")
    negative_filter = payload.get("negative_filter")
    _require(isinstance(identity, Mapping) and bool(identity), f"{source}: cache identity missing")
    _require(isinstance(negative_filter, Mapping) and bool(negative_filter),
             f"{source}: negative filter missing")
    try:
        layer = int(payload["layer"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: layer must be an integer") from error
    _require(layer in EXPECTED_CANDIDATES, f"{source}: unexpected layer {layer}")
    expected_layer_name = f"video_block_{layer:02d}"
    metrics = payload.get("metrics")
    _require(isinstance(metrics, list) and bool(metrics), f"{source}: metrics missing")
    tasks: list[str] = []
    macro: Mapping[str, Any] | None = None
    task_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(metrics):
        _require(isinstance(row, Mapping), f"{source}: row {index} is not an object")
        _require(row.get("experiment") == EXPECTED_EXPERIMENT,
                 f"{source}: row experiment mismatch")
        _require(row.get("layer") == expected_layer_name, f"{source}: row layer mismatch")
        task = str(row.get("task", ""))
        _require(bool(task), f"{source}: row {index} has no task")
        if task.endswith(MACRO_SUFFIX):
            _require(macro is None, f"{source}: duplicate macro row")
            macro = row
        else:
            _require(task not in tasks, f"{source}: duplicate task {task}")
            tasks.append(task)
            task_rows.append(row)
    _require(task_rows and macro is not None, f"{source}: task/macro rows missing")
    _require(macro.get("task") == f"{len(tasks)}{MACRO_SUFFIX}",
             f"{source}: invalid macro task name")
    retrieval = _finite(macro.get("retrieval_r1"), field="macro retrieval_r1", source=source)
    ratio = _finite(macro.get("state_style_ratio"), field="macro state_style_ratio", source=source)
    _require(0.0 <= retrieval <= 1.0 and ratio >= 0.0, f"{source}: invalid selection metrics")
    for field, reported in (("retrieval_r1", retrieval), ("state_style_ratio", ratio)):
        values = [_finite(row.get(field), field=f"task {field}", source=source) for row in task_rows]
        recomputed = sum(values) / len(values)
        _require(math.isclose(reported, recomputed, rel_tol=1e-9, abs_tol=1e-12),
                 f"{source}: stale macro {field}")
    return {
        "source": str(source),
        "layer": layer,
        "layer_name": expected_layer_name,
        "macro_retrieval_r1": retrieval,
        "macro_state_style_ratio": ratio,
        "task_set": sorted(tasks),
        "cache_identity": dict(identity),
        "negative_filter": dict(negative_filter),
    }


def _descending_ranks(values: Mapping[int, float]) -> dict[int, int]:
    ordered = sorted(set(values.values()), reverse=True)
    ranks: dict[float, int] = {}
    position = 1
    for value in ordered:
        ranks[value] = position
        position += sum(candidate == value for candidate in values.values())
    return {layer: ranks[value] for layer, value in values.items()}


def select_e2_layer(paths: Sequence[str | Path], output_dir: str | Path) -> int:
    candidates = [_load_candidate(path) for path in paths]
    layers = [int(candidate["layer"]) for candidate in candidates]
    _require(len(layers) == 3 and set(layers) == set(EXPECTED_CANDIDATES),
             f"E2 selection requires exactly layers {EXPECTED_CANDIDATES}, got {layers}")
    reference = candidates[0]
    for candidate in candidates[1:]:
        _require(candidate["cache_identity"] == reference["cache_identity"],
                 "candidate metrics use different validation caches")
        _require(candidate["negative_filter"] == reference["negative_filter"],
                 "candidate metrics use different negative filters")
        _require(candidate["task_set"] == reference["task_set"],
                 "candidate metrics use different task sets")
    retrieval_ranks = _descending_ranks(
        {int(item["layer"]): float(item["macro_retrieval_r1"]) for item in candidates}
    )
    ratio_ranks = _descending_ranks(
        {int(item["layer"]): float(item["macro_state_style_ratio"]) for item in candidates}
    )
    rows: list[dict[str, Any]] = []
    for item in candidates:
        layer = int(item["layer"])
        rows.append({
            "layer": layer,
            "layer_name": item["layer_name"],
            "macro_retrieval_r1": item["macro_retrieval_r1"],
            "macro_state_style_ratio": item["macro_state_style_ratio"],
            "retrieval_rank": retrieval_ranks[layer],
            "ratio_rank": ratio_ranks[layer],
            "joint_rank_sum": retrieval_ranks[layer] + ratio_ranks[layer],
            "selected": False,
            "source": item["source"],
        })
    rows.sort(key=lambda row: (
        int(row["joint_rank_sum"]),
        -float(row["macro_retrieval_r1"]),
        -float(row["macro_state_style_ratio"]),
        int(row["layer"]),
    ))
    rows[0]["selected"] = True
    selected = int(rows[0]["layer"])
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "evaluation_split": "val",
        "experiment": EXPECTED_EXPERIMENT,
        "proprio_mode": "observed",
        "active_variants": list(EXPECTED_VARIANTS),
        "holdout_variant": HOLDOUT_VARIANT,
        "r3_used": False,
        "selected_layer": selected,
        "selected_layer_name": f"video_block_{selected:02d}",
        "criterion": {
            "primary": "minimize retrieval_rank + ratio_rank",
            "metric_directions": {
                "macro_retrieval_r1": "higher-is-better",
                "macro_state_style_ratio": "higher-is-better",
            },
            "winner_tiebreak": [
                "higher macro_retrieval_r1",
                "higher macro_state_style_ratio",
                "lower layer number",
            ],
        },
        "cache_identity": reference["cache_identity"],
        "negative_filter": reference["negative_filter"],
        "task_set": reference["task_set"],
        "candidates": rows,
    }
    write_json(destination / "selection.json", payload)
    write_csv(destination / "selection.csv", rows)
    table = (
        "| Layer | Macro R@1 | Macro state/style ratio | R@1 rank | Ratio rank | Sum | Selected |\n"
        "|---:|---:|---:|---:|---:|---:|:---:|\n"
        + "".join(
            f"| {row['layer']} | {row['macro_retrieval_r1']:.6f} | "
            f"{row['macro_state_style_ratio']:.6f} | {row['retrieval_rank']} | "
            f"{row['ratio_rank']} | {row['joint_rank_sum']} | "
            f"{'yes' if row['selected'] else ''} |\n" for row in rows
        )
        + f"\nSelected E2 layer: **{selected}**. R3 was not loaded or used.\n"
    )
    atomic_write_text(destination / "summary.md", table)
    atomic_write_text(destination / "selected_layer.txt", f"{selected}\n")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(f"selected strict E2 validation layer {select_e2_layer(args.metrics, args.output_dir)}")


if __name__ == "__main__":
    main()


__all__ = ["select_e2_layer"]
