from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin.policy_content_adapter import official_data as module
from experiments.robotwin.policy_content_adapter.official_data import (
    EXPECTED_DATASET_FACTS,
    EXPECTED_TASK_DOMAIN_EPISODE_RANGES,
    EXPECTED_TASK_EPISODE_RANGES,
    OFFICIAL_TASKS,
    OfficialDataContractError,
    OfficialThreeTaskDataset,
    ThreeTaskRoundRobinSampler,
    select_official_full_550_per_task,
    select_official_episodes_from_native_split,
    verify_official_task_manifest,
)


CHECKED_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs/official_three_task_manifest.json"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def verified_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "official"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = dict(EXPECTED_DATASET_FACTS)
    (meta / "info.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
    (meta / "episodes.jsonl").write_text("episode-contract\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text("task-contract\n", encoding="utf-8")

    expected_files = {}
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        path = meta / name
        expected_files[f"meta/{name}"] = {
            "size_bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
    monkeypatch.setattr(module, "EXPECTED_META_FILES", expected_files)

    manifest = json.loads(CHECKED_MANIFEST.read_text(encoding="utf-8"))
    manifest["dataset"]["meta_files"] = copy.deepcopy(expected_files)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root, manifest_path, manifest


class _FakeInnerDataset:
    def __init__(self, root: Path, episodes: list[int], lengths: list[int]) -> None:
        assert len(episodes) == len(lengths)
        self.root = root
        self.episodes = list(episodes)
        cumulative = []
        total = 0
        for length in lengths:
            total += length
            cumulative.append(total)
        self.episode_data_index = {
            "from": torch.tensor([0] + cumulative[:-1], dtype=torch.long),
            "to": torch.tensor(cumulative, dtype=torch.long),
        }
        self.num_frames = total


class _FakeBaseLerobot:
    def __init__(self, inner: _FakeInnerDataset) -> None:
        self.multi_dataset = SimpleNamespace(_datasets=[inner])
        self._length = inner.num_frames
        self.replacement: int | None = None
        self.raise_on: int | None = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        if self.raise_on == index:
            raise OSError(f"failed frame {index}")
        actual = index if self.replacement is None else self.replacement
        return {"idx": actual, "raw_index": actual}


class _FakeRobotVideoDataset:
    skip_padding_as_possible = False

    def __init__(self, inner: _FakeInnerDataset) -> None:
        self.lerobot_dataset = _FakeBaseLerobot(inner)
        self.get_calls: list[int] = []
        self.public_getitem_calls = 0

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def _get(self, index: int):
        self.get_calls.append(index)
        raw = self.lerobot_dataset[index]
        return {"value": raw["raw_index"]}

    def __getitem__(self, index: int):
        self.public_getitem_calls += 1
        raise AssertionError("strict wrapper must never call native __getitem__")


@pytest.fixture()
def fake_native(verified_fixture):
    root, manifest_path, _ = verified_fixture
    # Include both exact boundary episodes and non-target neighbors.  The
    # shuffled order mirrors the native 99/1 train/val episode selection.
    episodes = [
        0,
        11_000,
        11_550,
        9_350,
        8_250,
        11_549,
        9_899,
        8_799,
    ]
    lengths = [2, 3, 2, 4, 5, 2, 3, 4]
    inner = _FakeInnerDataset(root, episodes, lengths)
    return _FakeRobotVideoDataset(inner), root, manifest_path, episodes, lengths


def _make_subset(fake_native, mode: str) -> OfficialThreeTaskDataset:
    native, root, manifest_path, _, _ = fake_native
    return OfficialThreeTaskDataset(
        native,
        dataset_root=root,
        manifest_path=manifest_path,
        sampling_mode=mode,
    )


def test_checked_manifest_declares_exact_release_contract() -> None:
    value = json.loads(CHECKED_MANIFEST.read_text(encoding="utf-8"))
    assert value["schema_version"] == 2
    assert tuple(value["task_order"]) == OFFICIAL_TASKS
    assert value["domain"]["label"] == "protocol_v2_hash_bound_range_partition"
    assert value["domain"]["verified"] is True
    assert value["domain"]["intrinsic_metadata_domain_field"] is False
    for task, (start, end) in EXPECTED_TASK_EPISODE_RANGES.items():
        assert value["tasks"][task]["episode_start"] == start
        assert value["tasks"][task]["episode_end_inclusive"] == end
        assert value["tasks"][task]["episode_count"] == 550
        for domain, (domain_start, domain_end) in (
            EXPECTED_TASK_DOMAIN_EPISODE_RANGES[task].items()
        ):
            declaration = value["tasks"][task]["domains"][domain]
            assert declaration["episode_start"] == domain_start
            assert declaration["episode_end_inclusive"] == domain_end
            assert declaration["episode_count"] == domain_end - domain_start + 1


def test_verify_manifest_hash_binds_all_three_metadata_files(verified_fixture) -> None:
    root, manifest_path, _ = verified_fixture
    verified = verify_official_task_manifest(manifest_path, root)
    assert verified.dataset_root == root.resolve()
    assert verified.task_names == OFFICIAL_TASKS
    assert {name for name, _, _ in verified.meta_files} == {
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
    }
    provenance = verified.as_provenance()
    assert provenance["domain_verified"] is True
    assert provenance["intrinsic_metadata_domain_field"] is False


def test_explicit_episode_selection_exactly_reproduces_native_split(
    verified_fixture,
) -> None:
    root, manifest_path, _ = verified_fixture
    verified = verify_official_task_manifest(manifest_path, root)
    selection = select_official_episodes_from_native_split(
        verified,
        val_set_proportion=0.01,
        is_training_set=True,
        seed=42,
    )

    # Independent literal reconstruction of the native BaseLerobotDataset
    # algorithm guards against selecting the three ranges before shuffling.
    import numpy as np

    shuffled = list(range(int(EXPECTED_DATASET_FACTS["total_episodes"])))
    np.random.default_rng(42).shuffle(shuffled)
    native_train = shuffled[: int(len(shuffled) * 0.99)]
    manifest_ids = {
        episode
        for start, end in EXPECTED_TASK_EPISODE_RANGES.values()
        for episode in range(start, end + 1)
    }
    expected = tuple(episode for episode in native_train if episode in manifest_ids)
    assert selection.episode_ids == expected
    assert selection.native_split_episode_count == 27_225
    assert selection.task_names == OFFICIAL_TASKS
    assert {task: len(values) for task, values in selection.episodes_by_task} == {
        "place_a2b_left": 546,
        "open_microwave": 543,
        "move_stapler_pad": 549,
    }
    assert len(selection.episode_ids) == 1_638
    assert sum(len(values) for _, values in selection.episodes_by_task) == len(expected)
    assert selection.as_provenance()["only_manifest_split_intersection_loaded"] is True
    assert selection.as_provenance()["loaded_episode_counts_by_task_domain"] == {
        "place_a2b_left": {"clean": 50, "official_random": 496},
        "open_microwave": {"clean": 49, "official_random": 494},
        "move_stapler_pad": {"clean": 50, "official_random": 499},
    }


def test_full_550_per_task_selects_exact_clean_and_official_random_counts(
    verified_fixture,
) -> None:
    root, manifest_path, _ = verified_fixture
    verified = verify_official_task_manifest(manifest_path, root)
    selection = select_official_full_550_per_task(verified, seed=42)
    assert selection.selection_mode == "full_550_per_task"
    assert selection.val_set_proportion == 0.0
    assert selection.is_training_set is True
    assert selection.native_split_episode_count == 27_500
    assert len(selection.episode_ids) == 1_650
    assert selection.episode_ids == tuple(sorted(selection.episode_ids))
    assert {task: len(values) for task, values in selection.episodes_by_task} == {
        task: 550 for task in OFFICIAL_TASKS
    }
    assert selection.as_provenance()["loaded_episode_counts_by_task_domain"] == {
        task: {"clean": 50, "official_random": 500} for task in OFFICIAL_TASKS
    }


def test_explicit_train_and_val_selections_are_disjoint_and_complete(
    verified_fixture,
) -> None:
    root, manifest_path, _ = verified_fixture
    verified = verify_official_task_manifest(manifest_path, root)
    train = select_official_episodes_from_native_split(
        verified,
        val_set_proportion=0.01,
        is_training_set=True,
        seed=42,
    )
    val = select_official_episodes_from_native_split(
        verified,
        val_set_proportion=0.01,
        is_training_set=False,
        seed=42,
    )
    assert train.target_episode_set.isdisjoint(val.target_episode_set)
    assert len(train.target_episode_set | val.target_episode_set) == 3 * 550


@pytest.mark.parametrize("tamper", ["hash", "range", "task_order"])
def test_manifest_declaration_tampering_fails_closed(
    verified_fixture, tamper: str
) -> None:
    root, manifest_path, manifest = verified_fixture
    if tamper == "hash":
        manifest["dataset"]["meta_files"]["meta/tasks.jsonl"]["sha256"] = "0" * 64
    elif tamper == "range":
        manifest["tasks"]["place_a2b_left"]["episode_start"] += 1
        manifest["tasks"]["place_a2b_left"]["episode_count"] -= 1
    else:
        manifest["task_order"] = list(reversed(manifest["task_order"]))
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(OfficialDataContractError):
        verify_official_task_manifest(manifest_path, root)


def test_metadata_content_tampering_fails_closed(verified_fixture) -> None:
    root, manifest_path, _ = verified_fixture
    path = root / "meta/tasks.jsonl"
    original = path.read_bytes()
    assert original
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(OfficialDataContractError, match="SHA-256 mismatch"):
        verify_official_task_manifest(manifest_path, root)


def test_duplicate_manifest_keys_fail_closed(verified_fixture) -> None:
    root, manifest_path, _ = verified_fixture
    manifest_path.write_text('{"schema_version": 1, "schema_version": 1}\n', encoding="utf-8")
    with pytest.raises(OfficialDataContractError, match="duplicate JSON key"):
        verify_official_task_manifest(manifest_path, root)


def test_all_frames_excludes_neighbor_episode_leakage_and_attaches_indices(fake_native) -> None:
    native, _, _, episodes, lengths = fake_native
    subset = _make_subset(fake_native, "all_frames")
    expected_target_frames = sum(
        length
        for episode, length in zip(episodes, lengths, strict=True)
        if any(start <= episode <= end for start, end in EXPECTED_TASK_EPISODE_RANGES.values())
    )
    assert len(subset) == expected_target_frames
    assert set(subset.episodes_by_task["place_a2b_left"]) == {11_000, 11_549}
    all_selected_episodes = {
        subset.record_for_index(index).episode_index for index in range(len(subset))
    }
    assert 0 not in all_selected_episodes
    assert 11_550 not in all_selected_episodes

    item = subset[0]
    record = subset.record_for_index(0)
    assert item["official_task"] == record.task
    assert item["official_domain"] == record.domain
    assert item["official_episode_index"] == record.episode_index
    assert item["official_base_index"] == record.base_index
    assert item["official_subset_index"] == 0
    assert item["task_name"] == record.task
    assert item["episode_index"] == record.episode_index
    assert item["base_index"] == record.base_index
    assert native.get_calls == [record.base_index]
    assert native.public_getitem_calls == 0


def test_episode_anchor_selects_one_lower_midpoint_per_target_episode(fake_native) -> None:
    _, _, _, episodes, _ = fake_native
    subset = _make_subset(fake_native, "episode_anchor")
    target_episode_count = sum(
        any(start <= episode <= end for start, end in EXPECTED_TASK_EPISODE_RANGES.values())
        for episode in episodes
    )
    assert len(subset) == target_episode_count == 6
    assert all(len(values) == 2 for values in subset.indices_by_task.values())
    # Episode 11000 begins after the two-frame episode 0 and has length three:
    # lower midpoint = 2 + (3 - 1) // 2 = 3.
    place_first = subset.record_for_index(subset.indices_by_task["place_a2b_left"][0])
    assert place_first.episode_index == 11_000
    assert place_first.base_index == 3
    assert place_first.frame_offset == 1
    report = subset.audit_report
    json.dumps(report)
    assert report["task_order"] == list(OFFICIAL_TASKS)
    assert report["total_selected_episodes"] == 6
    assert report["total_selected_samples"] == 6
    assert report["task_histogram"]["open_microwave"] == {
        "episodes": 2,
        "samples": 2,
        "domains": {"clean": 1, "official_random": 1},
    }


def test_outer_get_failure_propagates_without_public_getitem_retry(fake_native) -> None:
    native, _, _, _, _ = fake_native
    subset = _make_subset(fake_native, "episode_anchor")
    record = subset.record_for_index(0)
    native.lerobot_dataset.raise_on = record.base_index
    with pytest.raises(OSError, match="failed frame"):
        subset[0]
    assert native.public_getitem_calls == 0


def test_inner_random_replacement_is_detected(fake_native) -> None:
    native, _, _, _, _ = fake_native
    subset = _make_subset(fake_native, "episode_anchor")
    native.lerobot_dataset.replacement = len(native) - 1
    with pytest.raises(OfficialDataContractError, match="replaced requested frame"):
        subset[0]
    assert native.public_getitem_calls == 0


def test_out_of_range_underlying_episode_fails_closed(verified_fixture) -> None:
    root, manifest_path, _ = verified_fixture
    native = _FakeRobotVideoDataset(_FakeInnerDataset(root, [27_500], [2]))
    with pytest.raises(OfficialDataContractError, match="out-of-range canonical episode"):
        OfficialThreeTaskDataset(
            native,
            dataset_root=root,
            manifest_path=manifest_path,
            sampling_mode="all_frames",
        )


def test_inconsistent_episode_index_table_fails_closed(verified_fixture) -> None:
    root, manifest_path, _ = verified_fixture
    inner = _FakeInnerDataset(root, [11_000, 9_350, 8_250], [2, 2, 2])
    inner.episode_data_index["from"][1] += 1
    native = _FakeRobotVideoDataset(inner)
    with pytest.raises(OfficialDataContractError, match="not contiguous"):
        OfficialThreeTaskDataset(
            native,
            dataset_root=root,
            manifest_path=manifest_path,
            sampling_mode="all_frames",
        )


def test_round_robin_sampler_covers_all_tasks_every_three_samples(fake_native) -> None:
    subset = _make_subset(fake_native, "episode_anchor")
    sampler = ThreeTaskRoundRobinSampler(
        subset,
        seed=17,
        num_samples=12,
        shuffle=True,
    )
    indices = list(sampler)
    tasks = [subset.record_for_index(index).task for index in indices]
    assert len(indices) == len(sampler) == 12
    assert sampler.task_schedule == tuple(tasks)
    assert sampler.schedule == sampler.task_schedule
    for start in range(0, 12, 3):
        assert tuple(tasks[start : start + 3]) == OFFICIAL_TASKS
    assert {task: tasks.count(task) for task in OFFICIAL_TASKS} == {
        task: 4 for task in OFFICIAL_TASKS
    }
    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert [subset.record_for_index(index).task for index in second_epoch] == tasks
    assert second_epoch != indices


def test_sampler_default_length_is_exactly_balanced(fake_native) -> None:
    subset = _make_subset(fake_native, "all_frames")
    sampler = ThreeTaskRoundRobinSampler(subset, shuffle=False)
    indices = list(sampler)
    assert len(indices) % 3 == 0
    counts = {task: 0 for task in OFFICIAL_TASKS}
    for index in indices:
        counts[subset.record_for_index(index).task] += 1
    assert len(set(counts.values())) == 1


@pytest.mark.parametrize("bad_mode", ["episodes", "random", ""])
def test_unknown_sampling_mode_fails_closed(fake_native, bad_mode: str) -> None:
    native, root, manifest_path, _, _ = fake_native
    with pytest.raises(OfficialDataContractError, match="sampling_mode"):
        OfficialThreeTaskDataset(
            native,
            dataset_root=root,
            manifest_path=manifest_path,
            sampling_mode=bad_mode,
        )
