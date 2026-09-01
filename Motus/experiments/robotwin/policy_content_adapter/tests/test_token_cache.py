from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.robotwin.policy_content_adapter.token_cache import (
    FrozenMotusTokenDataset,
    FrozenTokenCacheWriter,
    validate_token_cache,
)
from experiments.robotwin.policy_content_adapter.protocol import TASKS


def _identity(character: str) -> dict:
    return {"path": f"/{character}", "size_bytes": 1, "sha256": character * 64}


def test_sharded_cache_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    writer = FrozenTokenCacheWriter(
        root,
        paired_manifest_identity=_identity("a"),
        base_lineage_identity=_identity("b"),
        capture_layer=16,
        shard_groups=2,
        expected_groups=6,
    )
    for task_index, task in enumerate(TASKS):
        tokens = torch.randn(2, 4, 3, 3072)
        writer.add(
            tokens,
            [f"{task}/s0", f"{task}/s1"],
            [task, task],
        )
    manifest = writer.finalize()
    assert manifest["counts"]["physical_states"] == 6
    result = validate_token_cache(
        root,
        expected_paired_manifest_sha256="a" * 64,
        expected_base_lineage_sha256="b" * 64,
    )
    assert result["physical_states"] == 6 and result["shards"] == 3
    dataset = FrozenMotusTokenDataset(root, verify_shards=True)
    assert len(dataset) == 6
    assert dataset[0]["visual_tokens"].shape == (4, 3, 3072)
    assert dataset[5]["task"] == TASKS[-1]


def test_cache_is_create_only_and_rejects_wrong_group_total(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    writer = FrozenTokenCacheWriter(
        root,
        paired_manifest_identity=_identity("a"),
        base_lineage_identity=_identity("b"),
        capture_layer=16,
        expected_groups=2,
    )
    writer.add(torch.randn(1, 4, 2, 3072), ["s0"], [TASKS[0]])
    with pytest.raises(Exception, match="expected 2"):
        writer.finalize()
    root.mkdir()
    with pytest.raises(FileExistsError):
        FrozenTokenCacheWriter(
            root,
            paired_manifest_identity=_identity("a"),
            base_lineage_identity=_identity("b"),
            capture_layer=16,
        )

