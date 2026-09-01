from __future__ import annotations

from pathlib import Path

from experiments.robotwin.policy_content_adapter.lineage import (
    LINEAGE_SCHEMA,
    LINEAGE_VERSION,
    validate_author_release_lineage,
)
from experiments.robotwin.policy_content_adapter.paired_data import (
    canonical_json_sha256,
    sha256_file,
)
from experiments.robotwin.policy_content_adapter.protocol import PROTOCOL_ID


def _identity(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def test_lineage_validator_rehashes_every_bound_file(tmp_path: Path) -> None:
    paths = []
    for index in range(9):
        path = tmp_path / f"f{index}"
        path.write_text(str(index), encoding="utf-8")
        paths.append(path)
    sources = [_identity(paths[8])]
    manifest = {
        "schema": LINEAGE_SCHEMA,
        "schema_version": LINEAGE_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "lineage_id": "test",
        "base_kind": "author_release_stage3_robotwin2",
        "checkpoint": _identity(paths[0]),
        "checkpoint_config": _identity(paths[1]),
        "training_config": _identity(paths[2]),
        "inference_config": _identity(paths[3]),
        "normalization_stats": _identity(paths[4]),
        "wan": {"config": _identity(paths[5]), "vae": _identity(paths[6])},
        "vlm": {"config": _identity(paths[7])},
        "source_files": sources,
        "source_inventory_sha256": canonical_json_sha256(sources),
    }
    result = validate_author_release_lineage(manifest)
    assert result["status"] == "PASS"
    paths[4].write_text("tampered", encoding="utf-8")
    try:
        validate_author_release_lineage(manifest)
    except Exception as exc:
        assert "size changed" in str(exc) or "SHA changed" in str(exc)
    else:
        raise AssertionError("tampered lineage was accepted")
