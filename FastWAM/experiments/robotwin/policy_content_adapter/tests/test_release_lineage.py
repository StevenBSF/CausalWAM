from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.release_lineage import (
    ReleaseLineageError,
    validate_author_release_lineage_payload,
    verify_author_release_lineage,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
MANIFEST = CONFIG_DIR / "author_release_base_manifest.json"
MANIFEST_SHA256 = "d90e6d545c04c28e9e73b6b8a9356ec5e9320be4be6f6b7e3b69237a3f38cefc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_static_author_release_lineage_has_locked_semantics_and_identity() -> None:
    value = validate_author_release_lineage_payload(_payload())
    assert _sha256(MANIFEST) == MANIFEST_SHA256
    assert value["base_kind"] == "author_release"
    assert value["checkpoint"]["sha256"] == (
        "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63"
    )
    assert value["dataset_stats"]["sha256"] == (
        "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095"
    )
    assert value["official_partition"]["total_episodes"] == 50 * 550
    assert value["model_contract"]["raw_camera_shape_chw"] == [3, 480, 640]
    assert value["model_contract"]["transformed_camera_shape_chw"] == [3, 240, 320]
    assert value["model_contract"]["final_video_size_hw"] == [384, 320]


def test_author_release_lineage_rejects_a_fabricated_training_seed() -> None:
    payload = copy.deepcopy(_payload())
    payload["training_seed"] = 1
    with pytest.raises(ReleaseLineageError, match="must not declare a training_seed"):
        validate_author_release_lineage_payload(payload)


def test_author_release_lineage_verifies_every_local_artifact() -> None:
    value = verify_author_release_lineage(
        MANIFEST,
        checkpoint_path=(
            "/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/"
            "robotwin_uncond_3cam_384.pt"
        ),
        dataset_stats_path=(
            "/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/"
            "robotwin_uncond_3cam_384_dataset_stats.json"
        ),
        official_manifest_path=CONFIG_DIR / "official_three_task_manifest.json",
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    assert value["manifest_identity"]["kind"] == "file"
    assert value["manifest_identity"]["sha256"] == MANIFEST_SHA256
    assert set(value["source"]["evidence_files"]) == {
        "readme",
        "task_config",
        "data_config",
        "evaluation_config",
    }


def test_author_release_lineage_rejects_wrong_manifest_sha() -> None:
    with pytest.raises(ReleaseLineageError, match="manifest SHA-256 mismatch"):
        verify_author_release_lineage(
            MANIFEST,
            checkpoint_path=_payload()["checkpoint"]["path"],
            dataset_stats_path=_payload()["dataset_stats"]["path"],
            official_manifest_path=CONFIG_DIR / "official_three_task_manifest.json",
            expected_manifest_sha256="0" * 64,
        )
