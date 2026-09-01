#!/usr/bin/env python3
"""Create strict E2/E3 controls and the final E1/E2/E3 R3 comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_text, write_csv, write_json


E2_EXPERIMENTS = (
    "E2-RawBackbone",
    "E2-InitHead",
    "E2-TrainedHead",
)
E3_EXPERIMENTS = (
    "E3-NoProprio-RawBackbone",
    "E3-NoProprio-InitHead",
    "E3-NoProprio-TrainedHead",
)
ALL_CONTROLS = (*E2_EXPERIMENTS, *E3_EXPERIMENTS)
PROTOCOL = "r3_holdout_v1"


def _read_metric(path_value: str | Path) -> tuple[Path, Mapping[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read metric JSON {path}: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metrics"), list):
        raise ValueError(f"not an E2/E3 metric artifact: {path}")
    return path, payload


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _macro(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = [
        row for row in payload["metrics"]
        if str(row.get("task", "")).endswith("-task-average")
    ]
    if len(rows) != 1:
        raise ValueError(f"{payload.get('experiment')}: expected one macro row")
    return rows[0]


def _canonical_unseen(
    row: Mapping[str, Any], *, require_generic_r3_aliases: bool = True
) -> dict[str, float]:
    style = _finite(
        row.get("style_distance_R3", row.get("clean_r3_distance", row.get("style_distance"))),
        "R3 style distance",
    )
    state = _finite(row.get("state_distance"), "state distance")
    if require_generic_r3_aliases:
        ratio = _finite(
            row.get("state_style_ratio_R3", row.get("state_style_ratio")),
            "R3 state/style ratio",
        )
    else:
        # The legacy E1 artifact has only an all-style generic ratio.  Rebuild
        # the R3-only value from its explicit R3 distance and the same epsilon
        # used by metrics.compute_representation_metrics.
        ratio = state / (style + 1e-8)
    r1 = _finite(
        row.get("R3_to_Clean_R@1", row.get("r3_to_clean_retrieval_at1", row.get("retrieval_r1"))),
        "R3 retrieval@1",
    )
    r5 = _finite(
        row.get("R3_to_Clean_R@5", row.get("r3_to_clean_retrieval_at5", row.get("retrieval_r5"))),
        "R3 retrieval@5",
    )
    positive = (
        _finite(row.get("positive_similarity"), "positive similarity")
        if require_generic_r3_aliases
        else 1.0 - style
    )
    negative = _finite(row.get("negative_similarity"), "negative similarity")
    if style < 0 or state < 0 or ratio < 0 or not (0 <= r1 <= 1 and 0 <= r5 <= 1):
        raise ValueError("metric values are outside their valid ranges")
    if require_generic_r3_aliases:
        # E2/E3 payloads must make their generic columns exact R3 aliases.
        for reported, expected, label in (
            (row.get("style_distance"), style, "style_distance"),
            (row.get("state_style_ratio"), ratio, "state_style_ratio"),
            (row.get("retrieval_r1"), r1, "retrieval_r1"),
            (row.get("retrieval_r5"), r5, "retrieval_r5"),
        ):
            if reported is not None and not math.isclose(float(reported), expected, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError(f"test {label} is not the exact R3-only metric")
    return {
        "style_distance_R3": style,
        "state_distance": state,
        "state_style_ratio_R3": ratio,
        "R3_to_Clean_R@1": r1,
        "R3_to_Clean_R@5": r5,
        "positive_similarity": positive,
        "negative_similarity": negative,
    }


def _init_training_consistency(payloads: Mapping[str, Mapping[str, Any]], prefix: str) -> None:
    init_name = f"{prefix}-InitHead" if prefix == "E2" else f"{prefix}-NoProprio-InitHead"
    trained_name = f"{prefix}-TrainedHead" if prefix == "E2" else f"{prefix}-NoProprio-TrainedHead"
    init = payloads[init_name].get("head")
    trained = payloads[trained_name].get("head")
    if not isinstance(init, Mapping) or not isinstance(trained, Mapping):
        raise ValueError(f"{prefix}: init/trained head provenance missing")
    if init.get("initial_head_sha256") != trained.get("initial_head_sha256"):
        raise ValueError(f"{prefix}: InitHead is not the head initialization used for training")
    if init.get("initialization_seed") != trained.get("training_seed"):
        raise ValueError(f"{prefix}: init/training seeds differ")
    if trained.get("checkpoint_kind") != "best_val":
        raise ValueError(f"{prefix}: final test did not use best-val checkpoint")


def compare_e2e3(
    metric_paths: Sequence[str | Path],
    *,
    e1_metric_path: str | Path,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    if len(metric_paths) != len(ALL_CONTROLS):
        raise ValueError(f"expected exactly {len(ALL_CONTROLS)} E2/E3 control metrics")
    payloads: dict[str, Mapping[str, Any]] = {}
    sources: dict[str, str] = {}
    for value in metric_paths:
        path, payload = _read_metric(value)
        experiment = str(payload.get("experiment"))
        if experiment in payloads:
            raise ValueError(f"duplicate metric experiment {experiment}")
        payloads[experiment] = payload
        sources[experiment] = str(path)
    if set(payloads) != set(ALL_CONTROLS):
        raise ValueError(
            f"control experiment set mismatch; missing={sorted(set(ALL_CONTROLS)-set(payloads))}, "
            f"extra={sorted(set(payloads)-set(ALL_CONTROLS))}"
        )
    identities: dict[str, Mapping[str, Any]] = {}
    layers: set[int] = set()
    locks: list[Mapping[str, Any]] = []
    for name, payload in payloads.items():
        if payload.get("schema_version") != 2 or payload.get("protocol") != PROTOCOL:
            raise ValueError(f"{name}: schema/protocol mismatch")
        if payload.get("evaluation_split") != "test":
            raise ValueError(f"{name}: final comparison is test-only")
        if tuple(payload.get("active_variants", ())) != (
            "clean", "style_02_seed_2"
        ) or payload.get("holdout_variant") != "style_02_seed_2":
            raise ValueError(f"{name}: test is not exact C/R3")
        mode = "constant_zero_normalized" if name.startswith("E3-") else "observed"
        if payload.get("proprio_mode") != mode:
            raise ValueError(f"{name}: proprio mode mismatch")
        identity = payload.get("cache_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{name}: cache identity missing")
        identities[name] = identity
        layers.add(int(payload.get("layer", -1)))
        lock = payload.get("decision_lock_identity")
        if not isinstance(lock, Mapping):
            raise ValueError(f"{name}: decision-lock provenance missing")
        locks.append(lock)
    if len(layers) != 1:
        raise ValueError("E2/E3 controls do not use the same selected layer")
    if any(lock != locks[0] for lock in locks[1:]):
        raise ValueError("E2/E3 test metrics do not share one pre-test decision lock")
    for names in (E2_EXPERIMENTS, E3_EXPERIMENTS):
        reference = identities[names[0]]
        if any(identities[name] != reference for name in names[1:]):
            raise ValueError(f"{names[0][:2]} controls use different test caches")
    if identities[E2_EXPERIMENTS[0]] == identities[E3_EXPERIMENTS[0]]:
        raise ValueError("E2/E3 must use distinct observed/no-proprio test caches")
    _init_training_consistency(payloads, "E2")
    _init_training_consistency(payloads, "E3")

    control_rows: list[dict[str, Any]] = []
    for experiment in ALL_CONTROLS:
        payload = payloads[experiment]
        metrics = _canonical_unseen(_macro(payload))
        control_rows.append({
            "experiment": experiment,
            "proprio": "No" if experiment.startswith("E3-") else "Yes",
            "train_styles": "C/R1/R2",
            "test_style": "R3 unseen",
            "layer": int(payload["layer"]),
            **metrics,
        })

    e1_path, e1_payload = _read_metric(e1_metric_path)
    if e1_payload.get("evaluation_split") != "test" or e1_payload.get("experiment") != "E1-TrainedHead":
        raise ValueError("E1 reference must be the formal trained-head test metric")
    # The completed E1 artifact reports generic fields averaged over R1/R2/R3;
    # use its explicit R3 fields so this row is directly comparable to E2/E3.
    e1_metrics = _canonical_unseen(
        _macro(e1_payload), require_generic_r3_aliases=False
    )
    e1_row = {
        "experiment": "E1-TrainedHead",
        "proprio": "Yes",
        "train_styles": "C/R1/R2/R3",
        "test_style": "R3 seen",
        "layer": int(e1_payload["layer"]),
        **e1_metrics,
    }
    trained = {
        row["experiment"]: row
        for row in control_rows
        if row["experiment"] in {"E2-TrainedHead", "E3-NoProprio-TrainedHead"}
    }
    final_rows = [e1_row, trained["E2-TrainedHead"], trained["E3-NoProprio-TrainedHead"]]

    e2_raw = next(row for row in control_rows if row["experiment"] == "E2-RawBackbone")
    e2_init = next(row for row in control_rows if row["experiment"] == "E2-InitHead")
    e2_trained = trained["E2-TrainedHead"]
    e3_raw = next(row for row in control_rows if row["experiment"] == "E3-NoProprio-RawBackbone")
    e3_init = next(row for row in control_rows if row["experiment"] == "E3-NoProprio-InitHead")
    e3_trained = trained["E3-NoProprio-TrainedHead"]
    e2_generalizes = (
        e2_trained["R3_to_Clean_R@1"] > max(e2_raw["R3_to_Clean_R@1"], e2_init["R3_to_Clean_R@1"])
        and e2_trained["state_style_ratio_R3"] > max(e2_raw["state_style_ratio_R3"], e2_init["state_style_ratio_R3"])
    )
    e3_beats_controls = (
        e3_trained["R3_to_Clean_R@1"] > max(e3_raw["R3_to_Clean_R@1"], e3_init["R3_to_Clean_R@1"])
        and e3_trained["state_style_ratio_R3"] > max(e3_raw["state_style_ratio_R3"], e3_init["state_style_ratio_R3"])
    )
    e2_r1_retention = e2_trained["R3_to_Clean_R@1"] / max(e1_row["R3_to_Clean_R@1"], 1e-8)
    e2_ratio_retention = e2_trained["state_style_ratio_R3"] / max(e1_row["state_style_ratio_R3"], 1e-8)
    large_e1_to_e2_drop = e2_r1_retention < 0.75 or e2_ratio_retention < 0.5
    e3_r1_retention = e3_trained["R3_to_Clean_R@1"] / max(e2_trained["R3_to_Clean_R@1"], 1e-8)
    e3_ratio_retention = e3_trained["state_style_ratio_R3"] / max(e2_trained["state_style_ratio_R3"], 1e-8)
    large_no_proprio_drop = e3_r1_retention < 0.75 or e3_ratio_retention < 0.5
    if e2_generalizes and large_e1_to_e2_drop:
        e2_conclusion = (
            "The C/R1/R2-trained content head improves both R3 retrieval and the state/style "
            "ratio over Raw and Init controls, supporting unseen-background generalization; "
            "however, it has a predefined large drop relative to E1, so that generalization is "
            "materially weaker than the seen-style E1 reference."
        )
    elif e2_generalizes:
        e2_conclusion = (
            "The C/R1/R2-trained content head improves both R3 retrieval and the state/style "
            "ratio over Raw and Init controls and does not show a predefined large drop relative "
            "to E1, supporting unseen-background generalization with substantial retention of "
            "the seen-style E1 result."
        )
    elif large_e1_to_e2_drop:
        e2_conclusion = (
            "E2 does not clearly outperform both unseen-style controls and has a predefined large "
            "drop relative to E1; the current evidence does not support generalizable style "
            "invariance and is materially weaker than the seen-style E1 reference."
        )
    else:
        e2_conclusion = (
            "E2 does not clearly outperform both unseen-style controls, although it does not show "
            "a predefined large drop relative to E1; the current E1 result remains consistent "
            "with invariance to seen transformations rather than proven generalizable style "
            "invariance."
        )
    if large_no_proprio_drop:
        e3_conclusion = (
            "Removing state-specific proprio causes a large drop relative to E2; previous state "
            "discrimination relies substantially on proprio-conditioned video tokens."
        )
    elif e3_beats_controls:
        e3_conclusion = (
            "The no-proprio trained head remains above both no-proprio controls without a large "
            "E2-relative drop; proprio shortcuts alone do not explain the result and visual "
            "information contributes substantially."
        )
    else:
        e3_conclusion = (
            "The no-proprio result is inconclusive: it neither shows a predefined large E2-relative "
            "drop nor clearly beats both no-proprio controls on retrieval and ratio."
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    interpretation = {
        "rules_are_descriptive_not_a_significance_test": True,
        "e2_beats_raw_and_init_on_r1_and_ratio": e2_generalizes,
        "e2_r1_retention_vs_e1": e2_r1_retention,
        "e2_ratio_retention_vs_e1": e2_ratio_retention,
        "large_e1_to_e2_drop_rule": "R@1 retention < 0.75 or ratio retention < 0.50",
        "large_e1_to_e2_drop": large_e1_to_e2_drop,
        "e3_beats_raw_and_init_on_r1_and_ratio": e3_beats_controls,
        "e3_r1_retention_vs_e2": e3_r1_retention,
        "e3_ratio_retention_vs_e2": e3_ratio_retention,
        "large_no_proprio_drop_rule": "R@1 retention < 0.75 or ratio retention < 0.50",
        "large_no_proprio_drop": large_no_proprio_drop,
        "e2_conclusion": e2_conclusion,
        "e3_conclusion": e3_conclusion,
    }
    write_json(destination / "comparison.json", {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "sources": {**sources, "E1-TrainedHead": str(e1_path)},
        "decision_lock_identity": dict(locks[0]),
        "selected_layer": next(iter(layers)),
        "controls": control_rows,
        "final_e1_e2_e3": final_rows,
        "interpretation": interpretation,
    })
    write_csv(destination / "controls.csv", control_rows)
    write_csv(destination / "e1_e2_e3.csv", final_rows)
    header = "| Experiment | Proprio | Train Styles | Test Style | Style Dist | State Dist | Ratio | R@1 | R@5 |\n"
    separator = "|---|:---:|---|---|---:|---:|---:|---:|---:|\n"
    final_table = "".join(
        f"| {row['experiment']} | {row['proprio']} | {row['train_styles']} | "
        f"{row['test_style']} | {row['style_distance_R3']:.6f} | "
        f"{row['state_distance']:.6f} | {row['state_style_ratio_R3']:.3f} | "
        f"{row['R3_to_Clean_R@1']:.3f} | {row['R3_to_Clean_R@5']:.3f} |\n"
        for row in final_rows
    )
    controls_table = "".join(
        f"| {row['experiment']} | {row['proprio']} | {row['train_styles']} | "
        f"{row['test_style']} | {row['style_distance_R3']:.6f} | "
        f"{row['state_distance']:.6f} | {row['state_style_ratio_R3']:.3f} | "
        f"{row['R3_to_Clean_R@1']:.3f} | {row['R3_to_Clean_R@5']:.3f} |\n"
        for row in control_rows
    )
    atomic_write_text(
        destination / "summary.md",
        "# Final E1/E2/E3 comparison\n\n"
        + header
        + separator
        + final_table
        + "\n# Strict E2/E3 controls\n\n"
        + header
        + separator
        + controls_table
        + "\nE2: "
        + e2_conclusion
        + "\n\nE3: "
        + e3_conclusion
        + "\n",
    )
    return control_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--e1-metric", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = compare_e2e3(args.metrics, e1_metric_path=args.e1_metric, output_dir=args.output_dir)
    print(f"wrote {len(rows)} strict E2/E3 control rows")


if __name__ == "__main__":
    main()


__all__ = ["compare_e2e3"]
