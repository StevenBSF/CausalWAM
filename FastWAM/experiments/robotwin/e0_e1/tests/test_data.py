from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.robotwin.e0_e1.data import (
    R3_HOLDOUT_PROTOCOL,
    R3_VARIANT,
    SEEN_VARIANTS,
    UNSEEN_TEST_VARIANTS,
    PairedDataError,
    PairedFrameDataset,
    VARIANTS,
    collate_paired_frames,
    compatible_state_vectors,
    split_for_content,
    variants_for_protocol,
)


FASTWAM_ROOT = Path(__file__).resolve().parents[4]
ROBOTWIN_DATA = FASTWAM_ROOT / "third_party/RoboTwin/data"
PLACE_ROOT = ROBOTWIN_DATA / "place_a2b_left/paired_random_background"


def _real_place_smoke(*, states: int = 2) -> PairedFrameDataset:
    if not (PLACE_ROOT / "contents/content_000000/COMPLETE.json").is_file():
        pytest.skip("the real paired Place A2B Left pilot is unavailable")
    return PairedFrameDataset(
        PLACE_ROOT,
        tasks=("place_a2b_left",),
        split="train",
        states_per_trajectory=states,
        allow_incomplete=True,
        max_trajectories_per_task=1,
    )


def test_split_boundaries() -> None:
    assert split_for_content(0) == "train"
    assert split_for_content(29) == "train"
    assert split_for_content(30) == "val"
    assert split_for_content(39) == "val"
    assert split_for_content(40) == "test"
    assert split_for_content(49) == "test"
    with pytest.raises(PairedDataError):
        split_for_content(50)


def test_real_content_is_strictly_indexed_and_sampled() -> None:
    dataset = _real_place_smoke(states=2)
    assert len(dataset.trajectories) == 1
    assert len(dataset) == 2
    assert [ref.key.frame_idx for ref in dataset.frame_refs] == [0, 150]
    assert [ref.trace_idx for ref in dataset.frame_refs] == [0, 2029]

    sample = dataset[0]
    assert sample["physical_key"] == "place_a2b_left/content_000000/frame_000000"
    assert sample["split"] == "train"
    assert sample["variant_names"] == VARIANTS
    assert sample["images"].shape == (4, 3, 384, 320)
    assert sample["images"].dtype == torch.uint8
    assert int(sample["images"].min()) >= 0
    assert int(sample["images"].max()) <= 255
    assert len({image.numpy().tobytes() for image in sample["images"]}) == 4
    assert sample["proprio_raw"].shape == (14,)
    assert torch.isfinite(sample["proprio_raw"]).all()
    assert "object_A_px" in sample["task_state_by_name"]
    assert "target_B_qz" in sample["task_state_by_name"]
    assert "task.object_A_px" in sample["physical_state_by_name"]
    assert "robot.left_qpos.0" in sample["physical_state_by_name"]


def test_formal_mode_requires_exactly_fifty_canonical_contents(tmp_path: Path) -> None:
    if not (PLACE_ROOT / "contents/content_000000/COMPLETE.json").is_file():
        pytest.skip("the real paired Place A2B Left pilot is unavailable")
    # A temporary directory exposing only one canonical content proves the
    # exact-set check without copying or modifying multi-gigabyte real data.
    root = tmp_path / "paired_random_background"
    contents = root / "contents"
    contents.mkdir(parents=True)
    (contents / "content_000000").symlink_to(
        PLACE_ROOT / "contents/content_000000", target_is_directory=True
    )
    with pytest.raises(PairedDataError, match="requires content IDs 0..49"):
        PairedFrameDataset(
            root,
            tasks=("place_a2b_left",),
            split="train",
            states_per_trajectory=1,
            allow_incomplete=False,
        )


def test_smoke_cap_is_not_allowed_in_formal_mode() -> None:
    with pytest.raises(PairedDataError, match="smoke-only"):
        PairedFrameDataset(
            PLACE_ROOT,
            tasks=("place_a2b_left",),
            split="train",
            states_per_trajectory=1,
            allow_incomplete=False,
            max_trajectories_per_task=1,
        )


def test_exact_smoke_content_ids_can_select_validation_split() -> None:
    if not (PLACE_ROOT / "contents/content_000030/COMPLETE.json").is_file():
        pytest.skip("the real paired Place validation trajectory is unavailable")
    dataset = PairedFrameDataset(
        PLACE_ROOT,
        tasks=("place_a2b_left",),
        split="val",
        states_per_trajectory=2,
        allow_incomplete=True,
        content_ids=(30,),
    )
    assert len(dataset.trajectories) == 1
    assert {trajectory.content_id for trajectory in dataset.trajectories} == {30}
    assert all(ref.key.content_id == 30 for ref in dataset.frame_refs)


def test_manifest_jsonl_csv_and_collate(tmp_path: Path) -> None:
    dataset = _real_place_smoke(states=2)
    outputs = dataset.write_manifests(tmp_path, stem="smoke")
    json_records = [
        json.loads(line) for line in outputs["jsonl"].read_text().splitlines()
    ]
    with outputs["csv"].open(newline="", encoding="utf-8") as handle:
        csv_records = list(csv.DictReader(handle))
    assert len(json_records) == len(csv_records) == len(dataset) * len(VARIANTS)
    assert json_records[0]["physical_key"] == csv_records[0]["physical_key"]
    assert int(csv_records[0]["trace_idx"]) == json_records[0]["trace_idx"]
    assert json_records[0]["variant"] == "clean"
    assert json_records[0]["hdf5"].endswith("/clean/data/episode0.hdf5")
    assert json_records[3]["variant"] == R3_VARIANT
    assert json_records[3]["hdf5"].endswith(
        "/style_02_seed_2/data/episode0.hdf5"
    )

    batch = collate_paired_frames([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 4, 3, 384, 320)
    assert batch["proprio_raw"].shape == (2, 14)
    assert batch["content_id"].dtype == torch.int64
    assert batch["frame_idx"].tolist() == [0, 150]
    assert isinstance(batch["physical_state_by_name"], list)


def test_named_state_alignment_handles_variable_microwave_layouts() -> None:
    left = {
        "task.microwave_root_px": 1.0,
        "task.microwave_qpos_0": 0.2,
        "task.microwave_qvel_0": 0.0,
    }
    right = {
        "task.microwave_root_px": 2.0,
        "task.microwave_qpos_0": 0.3,
        "task.microwave_qpos_1": 0.4,
    }
    names, left_values, right_values = compatible_state_vectors(left, right)
    assert names == ("task.microwave_qpos_0", "task.microwave_root_px")
    np.testing.assert_array_equal(left_values, [0.2, 1.0])
    np.testing.assert_array_equal(right_values, [0.3, 2.0])


def test_sampling_time_assertion_detects_rgb_correspondence_loss() -> None:
    dataset = _real_place_smoke(states=1)
    trajectory = dataset.trajectories[0]
    broken = list(trajectory.hdf5_paths)
    broken[1] = broken[0]
    object.__setattr__(trajectory, "hdf5_paths", tuple(broken))
    with pytest.raises(PairedDataError, match="RGB frame is not pairwise different"):
        _ = dataset[0]


def test_r3_holdout_protocol_mapping_is_exact() -> None:
    assert variants_for_protocol(R3_HOLDOUT_PROTOCOL, "train") == SEEN_VARIANTS
    assert variants_for_protocol(R3_HOLDOUT_PROTOCOL, "val") == SEEN_VARIANTS
    assert variants_for_protocol(R3_HOLDOUT_PROTOCOL, "test") == UNSEEN_TEST_VARIANTS
    # E2/E3 aliases intentionally resolve to the same data intervention.
    assert variants_for_protocol("e2", "train") == SEEN_VARIANTS
    assert variants_for_protocol("e3", "test") == UNSEEN_TEST_VARIANTS


def test_real_r3_holdout_train_materializes_only_seen_variants(tmp_path: Path) -> None:
    if not (PLACE_ROOT / "contents/content_000000/COMPLETE.json").is_file():
        pytest.skip("the real paired Place pilot is unavailable")
    dataset = PairedFrameDataset(
        PLACE_ROOT,
        tasks=("place_a2b_left",),
        split="train",
        states_per_trajectory=1,
        allow_incomplete=True,
        max_trajectories_per_task=1,
        protocol=R3_HOLDOUT_PROTOCOL,
    )
    assert dataset.protocol == R3_HOLDOUT_PROTOCOL
    assert dataset.active_variants == SEEN_VARIANTS
    assert dataset.trajectories[0].variant_names == SEEN_VARIANTS
    assert all(R3_VARIANT not in str(path) for path in dataset.trajectories[0].hdf5_paths)
    sample = dataset[0]
    assert sample["variant_names"] == SEEN_VARIANTS
    assert sample["images"].shape == (3, 3, 384, 320)

    outputs = dataset.write_manifests(tmp_path, stem="e2")
    json_text = outputs["jsonl"].read_text(encoding="utf-8")
    csv_text = outputs["csv"].read_text(encoding="utf-8")
    assert R3_VARIANT not in json_text
    assert R3_VARIANT not in csv_text
    rows = [json.loads(line) for line in json_text.splitlines()]
    assert len(rows) == len(dataset) * 3
    assert {row["variant"] for row in rows} == set(SEEN_VARIANTS)


def test_real_r3_holdout_test_materializes_only_clean_and_r3() -> None:
    if not (PLACE_ROOT / "contents/content_000040/COMPLETE.json").is_file():
        pytest.skip("the real paired Place test trajectory is unavailable")
    dataset = PairedFrameDataset(
        PLACE_ROOT,
        tasks=("place_a2b_left",),
        split="test",
        states_per_trajectory=1,
        allow_incomplete=True,
        content_ids=(40,),
        protocol=R3_HOLDOUT_PROTOCOL,
    )
    sample = dataset[0]
    assert sample["variant_names"] == UNSEEN_TEST_VARIANTS
    assert sample["images"].shape == (2, 3, 384, 320)
    assert all(
        "style_00_seed_0" not in str(path) and "style_01_seed_1" not in str(path)
        for path in dataset.trajectories[0].hdf5_paths
    )
