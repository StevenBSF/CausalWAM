from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.robotwin.policy_content_adapter import official_data as official_module
from experiments.robotwin.policy_content_adapter import runtime_utils
from experiments.robotwin.policy_content_adapter.official_data import (
    EXPECTED_DATASET_FACTS,
    NativeSplitEpisodeSelection,
    verify_official_task_manifest,
)


CHECKED_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs/official_three_task_manifest.json"
)


def test_fastwam_source_audit_covers_complete_runtime_python_tree() -> None:
    audit = runtime_utils.audit_local_fastwam_source()
    files = audit["files"]
    expected = {
        path.resolve().relative_to(runtime_utils.SRC_ROOT.resolve()).as_posix()
        for path in (runtime_utils.SRC_ROOT / "fastwam").rglob("*.py")
        if path.is_file()
    }
    assert audit["scope"] == "all_python_files_under_src_fastwam"
    assert audit["file_count"] == len(expected)
    assert set(files) == expected
    assert {
        "fastwam/models/wan22/wan_video_dit.py",
        "fastwam/models/wan22/schedulers/scheduler_continuous.py",
        "fastwam/models/wan22/helpers/loader.py",
        "fastwam/datasets/lerobot/processors/fastwam_processor.py",
        "fastwam/datasets/lerobot/utils/normalizer.py",
    }.issubset(files)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "official"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(dict(EXPECTED_DATASET_FACTS)) + "\n", encoding="utf-8"
    )
    (meta / "episodes.jsonl").write_text("episode-contract\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text("task-contract\n", encoding="utf-8")
    expected_files = {}
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        path = meta / name
        expected_files[f"meta/{name}"] = {
            "size_bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
    monkeypatch.setattr(official_module, "EXPECTED_META_FILES", expected_files)
    manifest = json.loads(CHECKED_MANIFEST.read_text(encoding="utf-8"))
    manifest["dataset"]["meta_files"] = copy.deepcopy(expected_files)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    verified = verify_official_task_manifest(manifest_path, root)
    return NativeSplitEpisodeSelection(
        manifest_sha256=verified.manifest_sha256,
        dataset_root=root.resolve(),
        seed=42,
        val_set_proportion=0.01,
        is_training_set=True,
        native_split_episode_count=27_225,
        episode_ids=(11_000, 9_350, 8_250),
        episodes_by_task=(
            ("place_a2b_left", (11_000,)),
            ("open_microwave", (9_350,)),
            ("move_stapler_pad", (8_250,)),
        ),
    )


def test_temporary_native_loader_narrows_and_restores_every_symbol(
    selection: NativeSplitEpisodeSelection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastwam.datasets.lerobot import base_lerobot_dataset as base_module
    from fastwam.datasets.lerobot.lerobot import lerobot_dataset as lerobot_module

    captured = {}

    def fake_multi_dataset(
        dataset_dirs,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerances_s=None,
        download_videos=True,
        video_backend=None,
    ):
        captured["dataset_dirs"] = dataset_dirs
        captured["episodes"] = episodes
        return SimpleNamespace(_datasets=[])

    monkeypatch.setattr(base_module, "MultiLeRobotDataset", fake_multi_dataset)
    before_base_metadata = base_module.LeRobotDatasetMetadata
    before_multi = base_module.MultiLeRobotDataset
    before_lerobot_metadata = lerobot_module.LeRobotDatasetMetadata

    key = str(selection.dataset_root)
    with runtime_utils._temporary_explicit_episode_native_loader(selection):
        assert base_module.LeRobotDatasetMetadata is runtime_utils._InfoOnlyOfficialMetadata
        assert lerobot_module.LeRobotDatasetMetadata is runtime_utils._ExplicitEpisodeMetadata
        result = base_module.MultiLeRobotDataset(
            dataset_dirs=[key],
            episodes={key: [11_000, 7, 9_350, 8_250, 27_000]},
            delta_timestamps={},
        )
        assert result._datasets == []
        assert captured["episodes"] == {key: [11_000, 9_350, 8_250]}

    assert base_module.LeRobotDatasetMetadata is before_base_metadata
    assert base_module.MultiLeRobotDataset is before_multi
    assert lerobot_module.LeRobotDatasetMetadata is before_lerobot_metadata


def test_temporary_native_loader_restores_symbols_on_exception(
    selection: NativeSplitEpisodeSelection,
) -> None:
    from fastwam.datasets.lerobot import base_lerobot_dataset as base_module
    from fastwam.datasets.lerobot.lerobot import lerobot_dataset as lerobot_module

    originals = (
        base_module.LeRobotDatasetMetadata,
        base_module.MultiLeRobotDataset,
        lerobot_module.LeRobotDatasetMetadata,
    )
    with pytest.raises(RuntimeError, match="construction failed"):
        with runtime_utils._temporary_explicit_episode_native_loader(selection):
            raise RuntimeError("construction failed")
    assert (
        base_module.LeRobotDatasetMetadata,
        base_module.MultiLeRobotDataset,
        lerobot_module.LeRobotDatasetMetadata,
    ) == originals


def test_native_loader_rejects_wrong_split_intersection(
    selection: NativeSplitEpisodeSelection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastwam.datasets.lerobot import base_lerobot_dataset as base_module

    def fake_multi_dataset(dataset_dirs, episodes=None, delta_timestamps=None):
        return SimpleNamespace(_datasets=[])

    monkeypatch.setattr(base_module, "MultiLeRobotDataset", fake_multi_dataset)
    key = str(selection.dataset_root)
    with runtime_utils._temporary_explicit_episode_native_loader(selection):
        with pytest.raises(ValueError, match="split intersection differs"):
            base_module.MultiLeRobotDataset(
                dataset_dirs=[key],
                episodes={key: [11_000, 9_350]},
                delta_timestamps={},
            )


def test_indexed_task_metadata_is_lazy_exact_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '\n'.join(
            json.dumps({"task_index": index, "task": task})
            for index, task in enumerate(("alpha", "beta", "gamma"))
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = runtime_utils._IndexedOfficialTasks(path, expected_count=3)
    assert tasks._cache == {}
    assert tasks[1] == "beta"
    assert tasks._cache == {1: "beta"}
    assert runtime_utils._ReverseOfficialTasks(tasks)["beta"] == 1
    with pytest.raises(KeyError):
        _ = tasks[3]


def test_indexed_task_metadata_rejects_line_index_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps({"task_index": 4, "task": "wrong-index"}) + "\n",
        encoding="utf-8",
    )
    tasks = runtime_utils._IndexedOfficialTasks(path, expected_count=1)
    with pytest.raises(ValueError, match="line/index mismatch"):
        _ = tasks[0]
