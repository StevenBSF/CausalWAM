from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.release_paired_binding import (
    BINDING_KIND,
    ReleasePairedBindingError,
    validate_release_paired_binding_payload,
    verify_release_paired_binding,
)
from experiments.robotwin.policy_content_adapter.release_lineage import (
    AUTHOR_RELEASE_LINEAGE_KIND,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload() -> dict:
    return {
        "schema_version": 1,
        "kind": BINDING_KIND,
        "status": "PASS",
        "protocol_id": "policy_native50hz_four_scene_v1",
        "base_lineage": {
            "kind": AUTHOR_RELEASE_LINEAGE_KIND,
            "sha256": _sha("lineage"),
        },
        "paired_dataset": {
            "scene_episode_count": 600,
            "physical_trajectory_count": 150,
            "train_physical_trajectory_count": 90,
            "state_bank_anchor_count": 720,
            "scene_variants_per_state": 4,
            "native_fps": 50,
            "action_steps": 32,
            "action_dim": 14,
            "interpolation_used": False,
            "all_pairs_exact": True,
        },
        "selected_train_artifacts": {
            "episode_count": 360,
            "file_count": 1440,
            "sha256": _sha("selected"),
        },
    }


def test_release_paired_binding_semantics_accept_exact_protocol() -> None:
    value = validate_release_paired_binding_payload(_payload())
    assert value["paired_dataset"]["state_bank_anchor_count"] == 720


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_episode_count", 599),
        ("native_fps", 30),
        ("action_steps", 1),
        ("interpolation_used", True),
        ("all_pairs_exact", False),
    ],
)
def test_release_paired_binding_rejects_protocol_drift(field: str, value: object) -> None:
    payload = _payload()
    payload["paired_dataset"][field] = value
    with pytest.raises(ReleasePairedBindingError):
        validate_release_paired_binding_payload(payload)


def test_release_paired_binding_manifest_sha_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = verify_release_paired_binding(path, expected_sha256=actual)
    assert verified["binding_manifest_identity"]["sha256"] == actual
    with pytest.raises(ReleasePairedBindingError):
        verify_release_paired_binding(path, expected_sha256=_sha("wrong"))
