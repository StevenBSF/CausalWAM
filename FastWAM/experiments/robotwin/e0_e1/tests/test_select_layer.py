from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.robotwin.e0_e1.select_layer import select_layer


IDENTITY = {"path": "/cache/val.pt", "size_bytes": 123, "mtime_ns": 456}
NEGATIVE_FILTER = {
    "min_temporal_gap": 8,
    "min_state_distance": 1e-5,
    "num_pairs": 12,
}


def _metric(
    tmp_path: Path,
    *,
    layer: int,
    retrieval: float,
    ratio: float,
    split: str = "val",
    experiment: str = "E0-RawBackbone",
    identity: dict[str, object] | None = None,
    negative_filter: dict[str, object] | None = None,
    tasks: tuple[str, ...] = ("task_a", "task_b"),
) -> Path:
    task_rows = [
        {
            "task": task,
            "layer": f"video_block_{layer:02d}",
            "experiment": experiment,
            "retrieval_r1": retrieval,
            "state_style_ratio": ratio,
        }
        for task in tasks
    ]
    payload = {
        "evaluation_split": split,
        "experiment": experiment,
        "layer": layer,
        "cache_identity": IDENTITY if identity is None else identity,
        "negative_filter": (
            NEGATIVE_FILTER if negative_filter is None else negative_filter
        ),
        "metrics": task_rows
        + [
            {
                "task": f"{len(tasks)}-task-average",
                "layer": f"video_block_{layer:02d}",
                "experiment": experiment,
                "retrieval_r1": retrieval,
                "state_style_ratio": ratio,
            }
        ],
    }
    path = tmp_path / f"metric_{layer}_{len(list(tmp_path.glob('metric_*.json')))}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_select_layer_writes_all_outputs_and_uses_joint_rank(tmp_path: Path) -> None:
    paths = [
        _metric(tmp_path, layer=8, retrieval=0.8, ratio=1.0),
        _metric(tmp_path, layer=16, retrieval=0.9, ratio=2.0),
        _metric(tmp_path, layer=24, retrieval=0.7, ratio=3.0),
    ]
    output_dir = tmp_path / "selection"
    assert select_layer(paths, output_dir) == 16
    assert (output_dir / "selected_layer.txt").read_text() == "16\n"
    assert (output_dir / "selection.csv").is_file()
    assert (output_dir / "summary.md").is_file()
    payload = json.loads((output_dir / "selection.json").read_text())
    assert payload["selected_layer"] == 16
    assert payload["evaluation_split"] == "val"
    assert payload["task_set"] == ["task_a", "task_b"]
    assert [row["layer"] for row in payload["candidates"]] == [16, 24, 8]
    assert payload["candidates"][0]["joint_rank_sum"] == 3
    assert sum(bool(row["selected"]) for row in payload["candidates"]) == 1


def test_select_layer_tiebreak_is_deterministic(tmp_path: Path) -> None:
    paths = [
        _metric(tmp_path, layer=24, retrieval=0.8, ratio=2.0),
        _metric(tmp_path, layer=8, retrieval=0.8, ratio=2.0),
        _metric(tmp_path, layer=16, retrieval=0.8, ratio=2.0),
    ]
    assert select_layer(paths, tmp_path / "selection") == 8


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("split", "evaluation_split='val'"),
        ("experiment", "accepts only E0-RawBackbone"),
        ("identity", "different validation caches"),
        ("filter", "different negative filters"),
        ("tasks", "different task sets"),
    ),
)
def test_select_layer_rejects_invalid_or_incompatible_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    first = _metric(tmp_path, layer=8, retrieval=0.8, ratio=2.0)
    kwargs: dict[str, object] = {}
    if mutation == "split":
        kwargs["split"] = "test"
    elif mutation == "experiment":
        kwargs["experiment"] = "E1-InitHead"
    elif mutation == "identity":
        kwargs["identity"] = {**IDENTITY, "size_bytes": 999}
    elif mutation == "filter":
        kwargs["negative_filter"] = {**NEGATIVE_FILTER, "min_temporal_gap": 99}
    else:
        kwargs["tasks"] = ("task_a", "task_c")
    second = _metric(
        tmp_path, layer=16, retrieval=0.9, ratio=3.0, **kwargs
    )
    with pytest.raises(ValueError, match=message):
        select_layer([first, second], tmp_path / "selection")


def test_select_layer_rejects_stale_macro_average(tmp_path: Path) -> None:
    path = _metric(tmp_path, layer=8, retrieval=0.8, ratio=2.0)
    payload = json.loads(path.read_text())
    payload["metrics"][-1]["retrieval_r1"] = 0.7
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagrees with task average"):
        select_layer([path], tmp_path / "selection")
