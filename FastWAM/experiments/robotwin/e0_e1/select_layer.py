#!/usr/bin/env python3
"""Select one E0 backbone layer from strictly compatible validation metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_text, write_csv, write_json


EXPECTED_EXPERIMENT = "E0-RawBackbone"
MACRO_SUFFIX = "-task-average"


def _finite_float(value: Any, *, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} must be finite")
    return result


def _load_candidate(path_value: str | Path) -> dict[str, Any]:
    source = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read E0 metric JSON {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: metric JSON must contain an object")
    if payload.get("evaluation_split") != "val":
        raise ValueError(
            f"{source}: layer selection requires evaluation_split='val', "
            f"got {payload.get('evaluation_split')!r}"
        )
    if payload.get("experiment") != EXPECTED_EXPERIMENT:
        raise ValueError(
            f"{source}: layer selection accepts only {EXPECTED_EXPERIMENT}, "
            f"got {payload.get('experiment')!r}"
        )
    identity = payload.get("cache_identity")
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError(f"{source}: metric JSON is missing cache_identity")
    negative_filter = payload.get("negative_filter")
    if not isinstance(negative_filter, Mapping) or not negative_filter:
        raise ValueError(f"{source}: metric JSON is missing negative_filter")
    try:
        layer = int(payload["layer"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: top-level layer must be an integer") from error
    if layer < 0:
        raise ValueError(f"{source}: layer must be non-negative")
    expected_layer_name = f"video_block_{layer:02d}"

    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"{source}: metrics must be a non-empty list")
    task_rows: list[Mapping[str, Any]] = []
    macro_rows: list[Mapping[str, Any]] = []
    task_names: set[str] = set()
    for row_index, row in enumerate(metrics):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: metric row {row_index} must be an object")
        if row.get("experiment") != EXPECTED_EXPERIMENT:
            raise ValueError(
                f"{source}: metric row {row_index} is not {EXPECTED_EXPERIMENT}"
            )
        if row.get("layer") != expected_layer_name:
            raise ValueError(
                f"{source}: metric row {row_index} layer disagrees with top-level layer"
            )
        task = str(row.get("task", ""))
        if not task:
            raise ValueError(f"{source}: metric row {row_index} has no task")
        if task.endswith(MACRO_SUFFIX):
            macro_rows.append(row)
        else:
            if task in task_names:
                raise ValueError(f"{source}: duplicate task metric row {task!r}")
            task_names.add(task)
            task_rows.append(row)
    if not task_rows:
        raise ValueError(f"{source}: no per-task metric rows found")
    if len(macro_rows) != 1:
        raise ValueError(
            f"{source}: expected exactly one macro-average row, got {len(macro_rows)}"
        )
    expected_macro_name = f"{len(task_rows)}{MACRO_SUFFIX}"
    macro = macro_rows[0]
    if macro.get("task") != expected_macro_name:
        raise ValueError(
            f"{source}: macro-average task must be {expected_macro_name!r}, "
            f"got {macro.get('task')!r}"
        )

    macro_retrieval = _finite_float(
        macro.get("retrieval_r1"), field="macro retrieval_r1", source=source
    )
    macro_ratio = _finite_float(
        macro.get("state_style_ratio"),
        field="macro state_style_ratio",
        source=source,
    )
    if not 0.0 <= macro_retrieval <= 1.0:
        raise ValueError(f"{source}: macro retrieval_r1 must be in [0,1]")
    if macro_ratio < 0.0:
        raise ValueError(f"{source}: macro state_style_ratio must be non-negative")

    # Do not trust a stale or manually edited macro row: verify it against the
    # unweighted task-level average produced by metrics.macro_average_metrics.
    for field, reported in (
        ("retrieval_r1", macro_retrieval),
        ("state_style_ratio", macro_ratio),
    ):
        values = [
            _finite_float(row.get(field), field=f"task {field}", source=source)
            for row in task_rows
        ]
        recomputed = sum(values) / len(values)
        if not math.isclose(reported, recomputed, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{source}: macro {field}={reported} disagrees with task average "
                f"{recomputed}"
            )

    return {
        "source": str(source),
        "layer": layer,
        "layer_name": expected_layer_name,
        "macro_retrieval_r1": macro_retrieval,
        "macro_state_style_ratio": macro_ratio,
        "task_set": sorted(task_names),
        "cache_identity": dict(identity),
        "negative_filter": dict(negative_filter),
    }


def _descending_ranks(values_by_layer: Mapping[int, float]) -> dict[int, int]:
    """Return competition ranks, with exact metric ties sharing one rank."""

    distinct_values = sorted(set(values_by_layer.values()), reverse=True)
    rank_by_value: dict[float, int] = {}
    position = 1
    for value in distinct_values:
        rank_by_value[value] = position
        position += sum(candidate == value for candidate in values_by_layer.values())
    return {layer: rank_by_value[value] for layer, value in values_by_layer.items()}


def select_layer(
    paths: Sequence[str | Path], output_dir: str | Path
) -> int:
    """Validate, rank, persist, and return the selected E0 layer number."""

    if not paths:
        raise ValueError("at least one E0 validation metric JSON is required")
    candidates = [_load_candidate(path) for path in paths]
    layers = [int(candidate["layer"]) for candidate in candidates]
    if len(set(layers)) != len(layers):
        raise ValueError(f"candidate layers must be unique, got {layers}")

    reference = candidates[0]
    for candidate in candidates[1:]:
        if candidate["cache_identity"] != reference["cache_identity"]:
            raise ValueError("candidate metrics come from different validation caches")
        if candidate["negative_filter"] != reference["negative_filter"]:
            raise ValueError("candidate metrics use different negative filters")
        if candidate["task_set"] != reference["task_set"]:
            raise ValueError("candidate metrics have different task sets")

    retrieval_ranks = _descending_ranks(
        {
            int(candidate["layer"]): float(candidate["macro_retrieval_r1"])
            for candidate in candidates
        }
    )
    ratio_ranks = _descending_ranks(
        {
            int(candidate["layer"]): float(candidate["macro_state_style_ratio"])
            for candidate in candidates
        }
    )
    ranked_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        layer = int(candidate["layer"])
        ranked_rows.append(
            {
                "layer": layer,
                "layer_name": candidate["layer_name"],
                "macro_retrieval_r1": candidate["macro_retrieval_r1"],
                "macro_state_style_ratio": candidate["macro_state_style_ratio"],
                "retrieval_rank": retrieval_ranks[layer],
                "ratio_rank": ratio_ranks[layer],
                "joint_rank_sum": retrieval_ranks[layer] + ratio_ranks[layer],
                "selected": False,
                "source": candidate["source"],
            }
        )
    ranked_rows.sort(
        key=lambda row: (
            int(row["joint_rank_sum"]),
            -float(row["macro_retrieval_r1"]),
            -float(row["macro_state_style_ratio"]),
            int(row["layer"]),
        )
    )
    ranked_rows[0]["selected"] = True
    selected = int(ranked_rows[0]["layer"])

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    criterion = {
        "primary": "minimize retrieval_rank + ratio_rank",
        "metric_directions": {
            "macro_retrieval_r1": "higher-is-better",
            "macro_state_style_ratio": "higher-is-better",
        },
        "metric_rank_ties": "equal values share a competition rank",
        "winner_tiebreak": [
            "higher macro_retrieval_r1",
            "higher macro_state_style_ratio",
            "lower layer number",
        ],
    }
    write_json(
        destination / "selection.json",
        {
            "schema_version": 1,
            "evaluation_split": "val",
            "experiment": EXPECTED_EXPERIMENT,
            "selected_layer": selected,
            "selected_layer_name": f"video_block_{selected:02d}",
            "criterion": criterion,
            "cache_identity": reference["cache_identity"],
            "negative_filter": reference["negative_filter"],
            "task_set": reference["task_set"],
            "candidates": ranked_rows,
        },
    )
    write_csv(destination / "selection.csv", ranked_rows)
    header = "| Layer | Macro Retrieval@1 | Macro state/style ratio | R@1 rank | Ratio rank | Rank sum | Selected |\n"
    separator = "|---:|---:|---:|---:|---:|---:|:---:|\n"
    body = "".join(
        f"| {row['layer']} | {row['macro_retrieval_r1']:.6f} | "
        f"{row['macro_state_style_ratio']:.6f} | {row['retrieval_rank']} | "
        f"{row['ratio_rank']} | {row['joint_rank_sum']} | "
        f"{'yes' if row['selected'] else ''} |\n"
        for row in ranked_rows
    )
    explanation = (
        "\nSelected layer: **"
        f"{selected}**. Candidates are ordered by the sum of descending macro "
        "Retrieval@1 and state/style-ratio ranks. Equal sums are resolved by "
        "higher Retrieval@1, then higher ratio, then lower layer number.\n"
    )
    atomic_write_text(destination / "summary.md", header + separator + body + explanation)
    atomic_write_text(destination / "selected_layer.txt", f"{selected}\n")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    selected = select_layer(args.metrics, args.output_dir)
    print(f"selected validation layer {selected}")


if __name__ == "__main__":
    main()


__all__ = ["select_layer"]
