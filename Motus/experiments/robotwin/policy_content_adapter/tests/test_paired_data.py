from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.robotwin.policy_content_adapter.paired_data import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    MotusPairedObservationDataset,
    build_motus_t_shaped_frame,
    canonical_json_sha256,
    collate_paired_groups,
    validate_paired_observation_manifest,
)
from experiments.robotwin.policy_content_adapter.protocol import (
    CAMERAS,
    PROTOCOL_ID,
    TASKS,
    VARIANTS,
)


def _synthetic_manifest(tmp_path: Path) -> Path:
    records = []
    for index in range(720):
        task = TASKS[index // 240]
        views = []
        for view_index, variant in enumerate(VARIANTS):
            views.append(
                {
                    "variant": variant,
                    "episode_index": index * 4 + view_index,
                    "frame_offset": 0,
                    "camera_paths": {
                        camera: f"/{camera}/{index}_{view_index}.mp4"
                        for camera in CAMERAS
                    },
                    "parquet_path": f"/data/{index}_{view_index}.parquet",
                }
            )
        records.append(
            {
                "physical_state_id": f"{task}/state_{index}",
                "physical_trajectory_id": f"{task}/trajectory_{index // 8}",
                "task": task,
                "content_id": index // 8,
                "frame_offset": 0,
                "views": views,
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "action_supervision_allowed": False,
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "cameras": list(CAMERAS),
        "record_inventory_sha256": canonical_json_sha256(records),
        "records": records,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_validator_accepts_exact_protocol(tmp_path: Path) -> None:
    path = _synthetic_manifest(tmp_path)
    result = validate_paired_observation_manifest(json.loads(path.read_text()))
    assert result["status"] == "PASS"
    assert result["physical_states"] == 720
    assert result["scene_views"] == 2880


def test_t_shape_matches_expected_geometry_and_padding() -> None:
    head = np.full((8, 10, 3), [255, 0, 0], dtype=np.uint8)
    left = np.full((8, 10, 3), [0, 255, 0], dtype=np.uint8)
    right = np.full((8, 10, 3), [0, 0, 255], dtype=np.uint8)
    output = build_motus_t_shaped_frame(
        head, left, right, target_size=(12, 10)
    )
    assert output.shape == (12, 10, 3)
    # 12x10 is the exact 1.2 aspect ratio of the 12x10 composite, no padding.
    assert np.array_equal(output[1, 5], [255, 0, 0])
    assert output[-2, 1, 1] > output[-2, 1, 0]
    assert output[-2, -2, 2] > output[-2, -2, 0]


def test_t_shape_is_pixel_exact_to_author_preprocessing() -> None:
    from data.robotwin2.robotwin_data_convert.robotwin_converter import (
        RobotWinConverter,
    )
    from data.utils.image_utils import resize_with_padding as author_resize

    generator = np.random.default_rng(7)
    frames = [
        generator.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    converter = object.__new__(RobotWinConverter)
    author_composite = converter.resize_and_concatenate_frames(*frames)
    expected = author_resize(author_composite, (384, 320))
    actual = build_motus_t_shaped_frame(*frames)
    assert np.array_equal(actual, expected)


def test_dataset_returns_four_ordered_views_and_collates(tmp_path: Path) -> None:
    path = _synthetic_manifest(tmp_path)

    def decoder(path: str, frame_index: int) -> np.ndarray:
        assert frame_index == 0
        value = sum(path.encode("utf-8")) % 255
        return np.full((8, 10, 3), value, dtype=np.uint8)

    dataset = MotusPairedObservationDataset(path, decoder=decoder)
    first = dataset[0]
    assert first["images"].shape == (4, 3, 384, 320)
    assert first["variants"] == VARIANTS
    collated = collate_paired_groups([first, dataset[1]])
    assert collated["images"].shape == (2, 4, 3, 384, 320)
    assert len(collated["physical_state_ids"]) == 8
    assert len(collated["task_ids"]) == 8
    assert torch.isfinite(collated["images"]).all()
