from __future__ import annotations

import json
from pathlib import Path

from experiments.robotwin.e0_e1.compare_e2e3 import ALL_CONTROLS, compare_e2e3


def _metric(
    path: Path,
    name: str,
    *,
    e3: bool,
    trained: bool = False,
    value: float | None = None,
) -> Path:
    mode = "constant_zero_normalized" if e3 else "observed"
    init_hash = "a" * 64
    head = None
    if name.endswith("InitHead"):
        head = {
            "initial_head_sha256": init_hash,
            "initialization_seed": 0,
        }
    elif name.endswith("TrainedHead"):
        head = {
            "initial_head_sha256": init_hash,
            "training_seed": 0,
            "checkpoint_kind": "best_val",
        }
    value = (.8 if trained else .2) if value is None else value
    row = {
        "task": "1-task-average",
        "style_distance": .1,
        "clean_r3_distance": .1,
        "style_distance_R3": .1,
        "state_distance": .5,
        "state_style_ratio": 5.0 + value,
        "state_style_ratio_R3": 5.0 + value,
        "retrieval_r1": value,
        "retrieval_r5": min(1.0, value + .2),
        "r3_to_clean_retrieval_at1": value,
        "r3_to_clean_retrieval_at5": min(1.0, value + .2),
        "R3_to_Clean_R@1": value,
        "R3_to_Clean_R@5": min(1.0, value + .2),
        "positive_similarity": .9,
        "negative_similarity": .4,
    }
    payload = {
        "schema_version": 2,
        "protocol": "r3_holdout_v1",
        "evaluation_split": "test",
        "experiment": name,
        "proprio_mode": mode,
        "active_variants": ["clean", "style_02_seed_2"],
        "holdout_variant": "style_02_seed_2",
        "layer": 8,
        "cache_identity": {"path": "/e3" if e3 else "/e2"},
        "decision_lock_identity": {"path": "/lock", "sha256": "b" * 64},
        "head": head,
        "metrics": [row],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_comparison_uses_e1_explicit_r3_fields(tmp_path: Path) -> None:
    paths = []
    for name in ALL_CONTROLS:
        paths.append(_metric(
            tmp_path / f"{name}.json", name,
            e3=name.startswith("E3-"), trained=name.endswith("TrainedHead"),
        ))
    e1 = {
        "evaluation_split": "test",
        "experiment": "E1-TrainedHead",
        "layer": 16,
        "metrics": [{
            "task": "3-task-average",
            "style_distance": .8,
            "clean_r3_distance": .12,
            "state_distance": .6,
            "state_style_ratio": 7.0,
            "retrieval_r1": .4,
            "retrieval_r5": .6,
            "r3_to_clean_retrieval_at1": .9,
            "r3_to_clean_retrieval_at5": 1.0,
            "positive_similarity": .88,
            "negative_similarity": .28,
        }],
    }
    e1_path = tmp_path / "e1.json"
    e1_path.write_text(json.dumps(e1), encoding="utf-8")
    compare_e2e3(paths, e1_metric_path=e1_path, output_dir=tmp_path / "out")
    result = json.loads((tmp_path / "out/comparison.json").read_text())
    e1_row = result["final_e1_e2_e3"][0]
    assert e1_row["style_distance_R3"] == .12
    assert e1_row["R3_to_Clean_R@1"] == .9
    assert e1_row["state_style_ratio_R3"] == .6 / (.12 + 1e-8)
    assert e1_row["positive_similarity"] == .88
    summary = (tmp_path / "out/summary.md").read_text()
    assert "# Final E1/E2/E3 comparison" in summary
    assert "# Strict E2/E3 controls" in summary
    assert all(name in summary for name in ALL_CONTROLS)


def test_e2_conclusion_reports_controls_win_and_large_e1_drop(tmp_path: Path) -> None:
    values = {
        "E2-RawBackbone": .2,
        "E2-InitHead": .3,
        # Beats both controls, but retains only 2/3 of E1 R@1.
        "E2-TrainedHead": .6,
        "E3-NoProprio-RawBackbone": .15,
        "E3-NoProprio-InitHead": .25,
        "E3-NoProprio-TrainedHead": .5,
    }
    paths = [
        _metric(
            tmp_path / f"{name}.json",
            name,
            e3=name.startswith("E3-"),
            trained=name.endswith("TrainedHead"),
            value=values[name],
        )
        for name in ALL_CONTROLS
    ]
    e1 = {
        "evaluation_split": "test",
        "experiment": "E1-TrainedHead",
        "layer": 16,
        "metrics": [{
            "task": "3-task-average",
            "clean_r3_distance": .1,
            "state_distance": .5,
            "r3_to_clean_retrieval_at1": .9,
            "r3_to_clean_retrieval_at5": 1.0,
            "negative_similarity": .3,
        }],
    }
    e1_path = tmp_path / "e1.json"
    e1_path.write_text(json.dumps(e1), encoding="utf-8")

    compare_e2e3(paths, e1_metric_path=e1_path, output_dir=tmp_path / "out")
    result = json.loads((tmp_path / "out/comparison.json").read_text())
    interpretation = result["interpretation"]

    assert interpretation["e2_beats_raw_and_init_on_r1_and_ratio"] is True
    assert interpretation["e2_r1_retention_vs_e1"] == .6 / .9
    assert interpretation["e2_ratio_retention_vs_e1"] > .5
    assert interpretation["large_e1_to_e2_drop_rule"] == (
        "R@1 retention < 0.75 or ratio retention < 0.50"
    )
    assert interpretation["large_e1_to_e2_drop"] is True
    assert "improves both R3 retrieval" in interpretation["e2_conclusion"]
    assert "predefined large drop relative to E1" in interpretation["e2_conclusion"]
    assert "materially weaker" in interpretation["e2_conclusion"]
    assert interpretation["e2_conclusion"] in (
        tmp_path / "out/summary.md"
    ).read_text(encoding="utf-8")
