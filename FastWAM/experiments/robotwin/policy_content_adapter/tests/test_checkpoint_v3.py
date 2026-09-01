"""Checkpoint-v3 provenance and artifact tamper tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from experiments.robotwin.e0_e1.backbone import CheckpointAudit
from experiments.robotwin.policy_content_adapter import model as policy_model
from experiments.robotwin.policy_content_adapter.model import (
    POLICY_CHECKPOINT_SCHEMA,
    POLICY_CHECKPOINT_VERSION,
    GatedCrossAttentionAdapter,
    PolicyContentConditioner,
    PolicyContentHead,
    artifact_identity,
    directory_identity,
    load_policy_checkpoint_into_model,
    module_state_sha256,
    save_policy_checkpoint,
    verify_artifact_identity,
)


def _tiny_conditioner() -> PolicyContentConditioner:
    return PolicyContentConditioner(
        head=PolicyContentHead(
            backbone_dim=12,
            embed_dim=8,
            num_queries=2,
            num_heads=2,
        ),
        adapter=GatedCrossAttentionAdapter(
            action_dim=16,
            content_dim=8,
            num_heads=2,
        ),
    )


class _TinyRolloutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_expert = nn.Module()
        self.action_expert.hidden_dim = 16
        self.action_expert.action_encoder = nn.Identity()
        self.mot = nn.Module()
        self.mot.prefill_video_cache = lambda **_kwargs: []


def _patch_tiny_checkpoint_contract(monkeypatch) -> None:
    monkeypatch.setattr(policy_model, "DEFAULT_BACKBONE_DIM", 12)
    monkeypatch.setattr(policy_model, "DEFAULT_EMBED_DIM", 8)
    monkeypatch.setattr(policy_model, "DEFAULT_NUM_QUERIES", 2)
    monkeypatch.setattr(policy_model, "DEFAULT_NUM_HEADS", 2)
    monkeypatch.setattr(policy_model, "DEFAULT_ACTION_DIM", 16)


def test_checkpoint_schema_v3_binds_base_and_rollout_artifact_sha256(tmp_path) -> None:
    base = tmp_path / "base.pt"
    base.write_bytes(b"immutable release checkpoint")
    stats = tmp_path / "dataset_stats.json"
    stats.write_bytes(b'{"mean": [0.0], "std": [1.0]}')
    stats_identity = artifact_identity(stats)
    stats_identity["required_for_rollout"] = True
    conditioner = _tiny_conditioner()
    destination = tmp_path / "policy.pt"

    save_policy_checkpoint(
        destination,
        model=nn.Linear(1, 1),
        conditioner=conditioner,
        base_checkpoint=base,
        regime="p_v1",
        step=3,
        run_config={"experiment": "unit"},
        artifact_identities={"dataset_stats": stats_identity},
    )
    payload = torch.load(destination, map_location="cpu", weights_only=False)

    assert payload["schema"] == POLICY_CHECKPOINT_SCHEMA
    assert payload["schema_version"] == POLICY_CHECKPOINT_VERSION == 3
    assert payload["regime"] == "p_v1"
    assert payload["step"] == 3
    assert payload["base_checkpoint"]["kind"] == "file"
    assert len(payload["base_checkpoint"]["sha256"]) == 64
    assert payload["artifact_identities"]["dataset_stats"] == stats_identity
    assert "action_expert" not in payload

    with pytest.raises(ValueError, match="must bind.*SHA-256"):
        save_policy_checkpoint(
            tmp_path / "unsafe.pt",
            model=nn.Linear(1, 1),
            conditioner=conditioner,
            base_checkpoint=base,
            regime="p_v1",
            step=0,
            run_config={},
            include_base_sha256=False,
        )


def test_file_and_directory_identities_fail_closed_after_equal_size_tampering(
    tmp_path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original-bytes")
    recorded_file = artifact_identity(artifact)
    artifact.write_bytes(b"tampered-bytes")
    assert artifact.stat().st_size == recorded_file["size_bytes"]
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verify_artifact_identity(recorded_file, label="file fixture")

    directory = tmp_path / "tokenizer"
    directory.mkdir()
    vocabulary = directory / "tokenizer.json"
    vocabulary.write_bytes(b"abcdefgh")
    (directory / "config.json").write_bytes(b"{}")
    recorded_directory = directory_identity(directory)
    vocabulary.write_bytes(b"abcdEfgh")
    assert directory_identity(directory)["size_bytes"] == recorded_directory["size_bytes"]
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verify_artifact_identity(recorded_directory, label="directory fixture")


def test_checkpoint_load_recomputes_required_runtime_artifact_before_overlay(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "base.pt"
    base.write_bytes(b"base-state")
    stats = tmp_path / "stats.json"
    stats.write_bytes(b"version-one")
    stats_identity = artifact_identity(stats)
    stats_identity["required_for_rollout"] = True
    checkpoint = tmp_path / "policy.pt"
    save_policy_checkpoint(
        checkpoint,
        model=nn.Linear(1, 1),
        conditioner=_tiny_conditioner(),
        base_checkpoint=base,
        regime="p_v1",
        step=1,
        run_config={},
        artifact_identities={"dataset_stats": stats_identity},
    )

    base_identity = artifact_identity(base)
    fake_audit = CheckpointAudit(
        path=str(base),
        size_bytes=base_identity["size_bytes"],
        mtime_ns=base.stat().st_mtime_ns,
        step=None,
        declared_torch_dtype=None,
        mot_tensor_count=0,
        proprio_tensor_count=0,
        sha256=None,
    )
    monkeypatch.setattr(
        policy_model,
        "strict_load_release_checkpoint",
        lambda *_args, **_kwargs: replace(fake_audit),
    )

    # Preserve byte count so the loader has to recompute and compare SHA-256.
    stats.write_bytes(b"version-two")
    assert stats.stat().st_size == stats_identity["size_bytes"]
    with pytest.raises(ValueError, match="rollout artifact dataset_stats SHA-256 differs"):
        load_policy_checkpoint_into_model(
            nn.Linear(1, 1),
            checkpoint,
            runtime_artifacts={"dataset_stats": stats},
            patch_video_prefill=False,
        )


def test_checkpoint_load_uses_bound_path_for_non_model_rollout_artifact(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "base.pt"
    base.write_bytes(b"base-state")
    seed_bank = tmp_path / "seed_bank.json"
    seed_bank.write_bytes(b'{"status":"PASS"}\n')
    seed_bank_identity = artifact_identity(seed_bank)
    seed_bank_identity["required_for_rollout"] = True
    checkpoint = tmp_path / "policy.pt"
    save_policy_checkpoint(
        checkpoint,
        model=nn.Linear(1, 1),
        conditioner=_tiny_conditioner(),
        base_checkpoint=base,
        regime="p_v1",
        step=1,
        run_config={},
        artifact_identities={
            "simulator_seed_bank_manifest": seed_bank_identity,
        },
    )

    base_identity = artifact_identity(base)
    fake_audit = CheckpointAudit(
        path=str(base),
        size_bytes=base_identity["size_bytes"],
        mtime_ns=base.stat().st_mtime_ns,
        step=None,
        declared_torch_dtype=None,
        mot_tensor_count=0,
        proprio_tensor_count=0,
        sha256=None,
    )
    monkeypatch.setattr(
        policy_model,
        "strict_load_release_checkpoint",
        lambda *_args, **_kwargs: replace(fake_audit),
    )
    _patch_tiny_checkpoint_contract(monkeypatch)

    runtime, _payload, audit = load_policy_checkpoint_into_model(
        _TinyRolloutModel(),
        checkpoint,
        runtime_artifacts={},
        patch_video_prefill=False,
    )
    assert runtime is not None
    verified = audit["verified_runtime_artifacts"][
        "simulator_seed_bank_manifest"
    ]
    assert verified["path"] == str(seed_bank.resolve())
    assert verified["sha256"] == seed_bank_identity["sha256"]
    assert verified["resolution_source"] == "checkpoint_identity"
    assert verified["checkpoint_path"] == str(seed_bank.resolve())


def test_checkpoint_load_rejects_relative_bound_runtime_artifact_without_override(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "base.pt"
    base.write_bytes(b"base-state")
    seed_bank = tmp_path / "seed_bank.json"
    seed_bank.write_bytes(b'{"status":"PASS"}\n')
    seed_bank_identity = artifact_identity(seed_bank)
    seed_bank_identity["path"] = "seed_bank.json"
    seed_bank_identity["required_for_rollout"] = True
    checkpoint = tmp_path / "policy.pt"
    save_policy_checkpoint(
        checkpoint,
        model=nn.Linear(1, 1),
        conditioner=_tiny_conditioner(),
        base_checkpoint=base,
        regime="p_v1",
        step=1,
        run_config={},
        artifact_identities={
            "simulator_seed_bank_manifest": seed_bank_identity,
        },
    )

    base_identity = artifact_identity(base)
    fake_audit = CheckpointAudit(
        path=str(base),
        size_bytes=base_identity["size_bytes"],
        mtime_ns=base.stat().st_mtime_ns,
        step=None,
        declared_torch_dtype=None,
        mot_tensor_count=0,
        proprio_tensor_count=0,
        sha256=None,
    )
    monkeypatch.setattr(
        policy_model,
        "strict_load_release_checkpoint",
        lambda *_args, **_kwargs: replace(fake_audit),
    )

    with pytest.raises(ValueError, match="checkpoint path must be absolute"):
        load_policy_checkpoint_into_model(
            nn.Linear(1, 1),
            checkpoint,
            runtime_artifacts={},
            patch_video_prefill=False,
        )


def test_module_state_sha256_is_deterministic_and_parameter_sensitive() -> None:
    torch.manual_seed(7)
    module = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    module.register_parameter("scalar_gate", nn.Parameter(torch.zeros(())))
    first = module_state_sha256(module)
    second = module_state_sha256(module)
    assert first == second
    assert len(first) == 64

    with torch.no_grad():
        module[0].weight[0, 0].add_(1.0)
    assert module_state_sha256(module) != first
