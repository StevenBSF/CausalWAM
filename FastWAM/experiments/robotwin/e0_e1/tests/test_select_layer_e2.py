from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.robotwin.e0_e1.select_layer_e2 import select_e2_layer


def _metric(path: Path, layer: int, retrieval: float, ratio: float, *, r3: bool = False) -> Path:
    variants = ["clean", "style_00_seed_0", "style_01_seed_1"]
    if r3:
        variants.append("style_02_seed_2")
    task_rows = [
        {
            "task": "task_a",
            "layer": f"video_block_{layer:02d}",
            "experiment": "E2-RawBackbone",
            "retrieval_r1": retrieval,
            "state_style_ratio": ratio,
        },
        {
            "task": "task_b",
            "layer": f"video_block_{layer:02d}",
            "experiment": "E2-RawBackbone",
            "retrieval_r1": retrieval,
            "state_style_ratio": ratio,
        },
    ]
    payload = {
        "protocol": "r3_holdout_v1",
        "evaluation_split": "val",
        "experiment": "E2-RawBackbone",
        "proprio_mode": "observed",
        "active_variants": variants,
        "record_variants": variants,
        "holdout_variant": "style_02_seed_2",
        "layer": layer,
        "cache_identity": {"path": "/cache", "size_bytes": 3, "mtime_ns": 4},
        "negative_filter": {"min_temporal_gap": 8, "min_state_distance": 1e-5},
        "cache_provenance": {
            "protocol": "r3_holdout_v1",
            "split": "val",
            "active_variants": variants,
            "proprio_mode": "observed",
        },
        "metrics": task_rows + [{
            "task": "2-task-average",
            "layer": f"video_block_{layer:02d}",
            "experiment": "E2-RawBackbone",
            "retrieval_r1": retrieval,
            "state_style_ratio": ratio,
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_select_e2_layer_requires_three_seen_style_candidates(tmp_path: Path) -> None:
    paths = [
        _metric(tmp_path / "8.json", 8, .2, 2.0),
        _metric(tmp_path / "16.json", 16, .4, 1.5),
        _metric(tmp_path / "24.json", 24, .3, 3.0),
    ]
    selected = select_e2_layer(paths, tmp_path / "out")
    assert selected == 24
    payload = json.loads((tmp_path / "out/selection.json").read_text())
    assert payload["r3_used"] is False
    assert payload["active_variants"] == [
        "clean", "style_00_seed_0", "style_01_seed_1"
    ]


def test_select_e2_layer_rejects_r3_validation(tmp_path: Path) -> None:
    paths = [
        _metric(tmp_path / "8.json", 8, .2, 2.0),
        _metric(tmp_path / "16.json", 16, .4, 1.5, r3=True),
        _metric(tmp_path / "24.json", 24, .3, 3.0),
    ]
    with pytest.raises(ValueError, match="C/R1/R2"):
        select_e2_layer(paths, tmp_path / "out")
