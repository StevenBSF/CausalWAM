from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.robotwin.policy_content_adapter.official_data import (
    ACTION_STRIDE,
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    MotusOfficialDataset,
    TASK_DOMAIN_RANGES,
    TASK_PROMPTS,
    canonical_json_sha256,
    validate_official_manifest,
)
from experiments.robotwin.policy_content_adapter.protocol import (
    CAMERAS,
    PROTOCOL_ID,
    TASKS,
)


def _manifest(tmp_path: Path) -> Path:
    records = []
    for task in TASKS:
        for domain in ("clean", "official_random"):
            start, end = TASK_DOMAIN_RANGES[task][domain]
            for episode in range(start, end + 1):
                records.append(
                    {
                        "task": task,
                        "domain": domain,
                        "episode_index": episode,
                        "length": 100,
                        "eligible_condition_frames": 20,
                        "parquet_path": f"/{episode}.parquet",
                        "camera_paths": {
                            camera: f"/{camera}/{episode}.mp4" for camera in CAMERAS
                        },
                        "prompt": TASK_PROMPTS[task],
                    }
                )
    stats = tmp_path / "stat.json"
    stats.write_text(
        json.dumps({"robotwin2": {"min": [0.0] * 14, "max": [2.0] * 14}}),
        encoding="utf-8",
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "normalization_stats": {
            "path": str(stats),
            "size_bytes": stats.stat().st_size,
            "sha256": __import__(
                "experiments.robotwin.policy_content_adapter.paired_data",
                fromlist=["sha256_file"],
            ).sha256_file(stats),
        },
        "action_value_contract": {
            "training": "raw_joint_action_vector",
            "state_source": "action_column_matching_motus_qpos_pt",
            "future_action_source": "action_column_matching_motus_qpos_pt",
            "fastwam_observation_state_used": False,
            "deployment": "raw_state_and_raw_predicted_actions",
            "stats_loaded_but_unused_by_author_deploy_path": True,
        },
        "tasks": list(TASKS),
        "temporal_contract": {
            "source_fps": 50,
            "source_stride": 5,
            "action_hz": 10,
            "action_chunk": 16,
            "action_dim": 14,
            "video_frames": 8,
            "video_action_ratio": 2,
            "physical_span_seconds": 1.6,
            "interpolation": False,
            "equivalence": "50Hz_stride5_equals_author_30Hz_stride3_at_10Hz",
        },
        "record_inventory_sha256": canonical_json_sha256(records),
        "records": records,
    }
    path = tmp_path / "official.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_official_manifest_and_time_contract(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    result = validate_official_manifest(json.loads(path.read_text()))
    assert result == {
        "status": "PASS",
        "episodes": 1650,
        "virtual_samples_per_epoch": 16500,
        "record_inventory_sha256": json.loads(path.read_text())["record_inventory_sha256"],
    }


def test_official_dataset_is_deterministic_and_uses_exact_10hz_indices(tmp_path: Path) -> None:
    path = _manifest(tmp_path)

    def decoder(path: str, indices: list[int]):
        del path
        return [np.full((8, 10, 3), index, dtype=np.uint8) for index in indices]

    def parquet_loader(path: str):
        del path
        actions = np.arange(100, dtype=np.float32)[:, None].repeat(14, axis=1)
        states = actions + 1000
        return {"observation.state": states, "action": actions, "frame_index": list(range(100))}

    embeddings = {task: torch.full((3, 4), float(index)) for index, task in enumerate(TASKS)}
    dataset = MotusOfficialDataset(
        path,
        task_embeddings=embeddings,
        training_seed=7,
        decoder=decoder,
        parquet_loader=parquet_loader,
    )
    first = dataset[0]
    again = dataset[0]
    assert first["condition_frame_index"] == again["condition_frame_index"]
    action_indices = first["action_frame_indices"].tolist()
    assert len(action_indices) == 16
    assert all(b - a == ACTION_STRIDE for a, b in zip(action_indices, action_indices[1:]))
    assert first["video_frame_indices"].tolist() == action_indices[1::2]
    assert first["first_frame"].shape == (3, 384, 320)
    assert first["video_frames"].shape == (8, 3, 384, 320)
    # Author RoboTwin training and deployment both consume raw qpos values.
    assert first["initial_state"][0].item() == first["condition_frame_index"]
    assert first["action_sequence"][0, 0].item() == action_indices[0]
    dataset.set_epoch(1)
    # Epoch participates in the stateless condition-frame RNG.
    assert dataset[0]["condition_frame_index"] != first["condition_frame_index"]
