from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.policy_content_adapter.prepare_release_paired_text_cache import (
    AUDIT_FILENAME,
    AUDIT_KIND,
    ReleasePairedTextCacheError,
    materialize_paired_prompt_payloads,
    paired_prompt_entries,
    verify_release_paired_text_cache,
)
from experiments.robotwin.policy_content_adapter.stage1_text_cache import (
    CONTEXT_DIM,
    CONTEXT_LEN,
)


def _fake_encode(prompts):
    count = len(prompts)
    context = torch.arange(
        count * CONTEXT_LEN * CONTEXT_DIM,
        dtype=torch.float32,
    ).reshape(count, CONTEXT_LEN, CONTEXT_DIM)
    mask = torch.zeros(count, CONTEXT_LEN, dtype=torch.bool)
    mask[:, :8] = True
    return context, mask


def _write_audit(root: Path, cache: dict[str, object]) -> Path:
    payload = {
        "status": "PASS",
        "kind": AUDIT_KIND,
        "schema_version": 1,
        "base_lineage_manifest": {
            "path": "/lineage.json",
            "size_bytes": 1,
            "sha256": "a" * 64,
        },
        "release_paired_binding_manifest": {
            "path": "/binding.json",
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
        "prompts": paired_prompt_entries(),
        "context_len": CONTEXT_LEN,
        "context_dim": CONTEXT_DIM,
        "model": {"encoder_id": "wan22ti2v5b"},
        "cache": cache,
        "implementation": {"sha256": "c" * 64},
    }
    path = root / AUDIT_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_three_prompt_cache_materializes_and_verifies_exact_payload_set(
    tmp_path: Path,
) -> None:
    cache = materialize_paired_prompt_payloads(tmp_path, encode_batch=_fake_encode)
    assert cache["file_count"] == 3
    assert len(paired_prompt_entries()) == 3
    _write_audit(tmp_path, cache)
    report = verify_release_paired_text_cache(
        tmp_path,
        expected_base_lineage_sha256="a" * 64,
        expected_release_paired_binding_sha256="b" * 64,
    )
    assert report["status"] == "PASS"
    assert report["directory_identity"]["file_count"] == 4


def test_three_prompt_cache_rejects_tampered_aggregate(tmp_path: Path) -> None:
    cache = materialize_paired_prompt_payloads(tmp_path, encode_batch=_fake_encode)
    cache["aggregate_payload_sha256"] = "f" * 64
    _write_audit(tmp_path, cache)
    with pytest.raises(ReleasePairedTextCacheError, match="aggregate_payload_sha256"):
        verify_release_paired_text_cache(tmp_path)
