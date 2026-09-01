from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.robotwin.e0_e1.smoke_proof_e2e3 import (
    ARTIFACTS,
    build_smoke_proof,
    validate_smoke_proof,
    write_smoke_proof,
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_smoke(tmp_path: Path) -> tuple[Path, dict]:
    smoke = tmp_path / "smoke"
    scientific_artifact = smoke / "cache/e2_val.pt"
    scientific_artifact.parent.mkdir(parents=True)
    scientific_artifact.write_bytes(b"validated-cache")
    import hashlib

    stat = scientific_artifact.stat()
    scientific_identity = {
        "path": str(scientific_artifact.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(scientific_artifact.read_bytes()).hexdigest(),
    }
    config = {
        "schema_version": 1,
        "protocol": "r3_holdout_v1",
        "mode": "smoke",
        "tasks": ["place_a2b_left"],
        "layers": [8, 16, 24],
        "states_per_trajectory": 2,
        "train_steps": 1,
        "groups_per_batch": 2,
        "val_every": 1,
        "seed": 0,
        "temperature": 0.07,
        "min_temporal_gap": 8,
        "min_state_distance": 1e-5,
        "data_root": "/data",
        "model_base": "/models",
        "checkpoint": {"path": "/models/checkpoint", "sha256": "1" * 64},
        "dataset_stats": {"path": "/models/stats", "sha256": "2" * 64},
        "experiment_code_sha256": {"runner": "3" * 64},
        "canonical_smoke_proof": None,
    }
    _json(smoke / "run_config.json", config)
    _json(
        smoke / "protocol_audit.json",
        {
            "schema_version": 1,
            "protocol": "r3_holdout_v1",
            "audit_status": "PASS",
            "run_mode": "smoke",
            "tasks": ["place_a2b_left"],
            "run_dir": str(smoke.resolve()),
            "assertions": {"strict_protocol": True, "r3_holdout": True},
            "artifact_identities": {"cache": scientific_identity},
        },
    )
    _json(
        smoke / "deliverables.json",
        {
            "schema_version": 1,
            "protocol": "r3_holdout_v1",
            "status": "COMPLETE_AND_AUDITED",
            "run_dir": str(smoke.resolve()),
        },
    )
    status = smoke / "status"
    status.mkdir(parents=True)
    (status / "state.txt").write_text("SUCCESS\n", encoding="utf-8")
    (status / "SUCCESS").write_text("2026-08-13T00:00:00Z\n", encoding="utf-8")
    (status / "final_audit.done").write_text("2026-08-13T00:00:00Z\n", encoding="utf-8")
    return smoke, config


def test_full_proof_binds_every_canonical_terminal_artifact(tmp_path: Path) -> None:
    smoke, config = _canonical_smoke(tmp_path)
    proof_path = smoke / "canonical_smoke_proof.json"
    write_smoke_proof(smoke, proof_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert set(proof["artifacts"]) == set(ARTIFACTS)
    full_config = {**config, "mode": "full"}
    evidence = validate_smoke_proof(
        proof_path, canonical_smoke_dir=smoke, full_config=full_config
    )
    assert evidence["proof_identity"]["sha256"]

    (smoke / "status/SUCCESS").write_text("forged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="success_marker identity"):
        validate_smoke_proof(
            proof_path, canonical_smoke_dir=smoke, full_config=full_config
        )


def test_proof_rejects_noncanonical_smoke_and_mismatched_full_provenance(
    tmp_path: Path,
) -> None:
    smoke, config = _canonical_smoke(tmp_path)
    config["train_steps"] = 2
    _json(smoke / "run_config.json", config)
    with pytest.raises(ValueError, match="train_steps"):
        build_smoke_proof(smoke)

    config["train_steps"] = 1
    _json(smoke / "run_config.json", config)
    proof_path = smoke / "canonical_smoke_proof.json"
    write_smoke_proof(smoke, proof_path)
    full_config = {**config, "mode": "full", "model_base": "/different"}
    with pytest.raises(ValueError, match="shared provenance differs for model_base"):
        validate_smoke_proof(
            proof_path, canonical_smoke_dir=smoke, full_config=full_config
        )


def test_existing_proof_is_immutable(tmp_path: Path) -> None:
    smoke, _ = _canonical_smoke(tmp_path)
    proof_path = smoke / "canonical_smoke_proof.json"
    write_smoke_proof(smoke, proof_path)
    (smoke / "status/final_audit.done").write_text("later\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existing canonical smoke proof differs"):
        write_smoke_proof(smoke, proof_path)
