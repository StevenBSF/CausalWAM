from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from experiments.robotwin.policy_content_adapter import stage1
from experiments.robotwin.policy_content_adapter import stage1_text_cache as module


def _entries(count: int = 3) -> list[dict[str, str]]:
    result = []
    for index in range(count):
        prompt = f"prompt-{index}"
        result.append(
            {
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt": prompt,
                "task": "place_a2b_left",
                "task_index": str(index),
            }
        )
    return sorted(result, key=lambda item: item["sha256"])


def _encoder(prompts):
    batch = len(prompts)
    return (
        torch.zeros(
            (batch, module.CONTEXT_LEN, module.CONTEXT_DIM), dtype=torch.bfloat16
        ),
        torch.zeros((batch, module.CONTEXT_LEN), dtype=torch.bool),
    )


def test_rank_sharding_atomic_write_and_resume(tmp_path: Path) -> None:
    entries = _entries()
    first = module.prepare_prompt_shard(
        entries,
        cache_dir=tmp_path,
        rank=0,
        world_size=2,
        batch_size=2,
        resume=False,
        audit_only=False,
        encode_batch=_encoder,
    )
    second = module.prepare_prompt_shard(
        entries,
        cache_dir=tmp_path,
        rank=1,
        world_size=2,
        batch_size=2,
        resume=False,
        audit_only=False,
        encode_batch=_encoder,
    )
    assert first["assigned_count"] + second["assigned_count"] == len(entries)
    assert first["created_count"] + second["created_count"] == len(entries)
    merged = module.merge_shard_reports(entries, [first, second], cache_dir=tmp_path)
    assert merged["file_count"] == len(entries)
    assert merged["over_length_prompt_count"] == 0
    assert len(list(tmp_path.glob("*.pt"))) == len(entries)
    assert not list(tmp_path.glob(".*.tmp-*"))

    def forbidden(_prompts):
        raise AssertionError("resume must not re-encode valid payloads")

    resumed = module.prepare_prompt_shard(
        entries,
        cache_dir=tmp_path,
        rank=0,
        world_size=1,
        batch_size=2,
        resume=True,
        audit_only=False,
        encode_batch=forbidden,
    )
    assert resumed["created_count"] == 0
    assert resumed["skipped_valid_count"] == len(entries)


def test_audit_rejects_extra_pt_and_over_length_payload(tmp_path: Path) -> None:
    entries = _entries(1)
    report = module.prepare_prompt_shard(
        entries,
        cache_dir=tmp_path,
        rank=0,
        world_size=1,
        batch_size=1,
        resume=False,
        audit_only=False,
        encode_batch=_encoder,
    )
    torch.save(
        {
            "context": torch.zeros(
                (module.CONTEXT_LEN, module.CONTEXT_DIM), dtype=torch.bfloat16
            ),
            "mask": torch.zeros(module.CONTEXT_LEN, dtype=torch.bool),
        },
        tmp_path / "extra.pt",
    )
    with pytest.raises(module.Stage1TextCacheError, match="set mismatch"):
        module.merge_shard_reports(entries, [report], cache_dir=tmp_path)

    bad = tmp_path / "bad.pt"
    torch.save(
        {
            "context": torch.zeros(
                (module.CONTEXT_LEN, module.CONTEXT_DIM), dtype=torch.bfloat16
            ),
            "mask": torch.ones(module.CONTEXT_LEN, dtype=torch.bool),
        },
        bad,
    )
    with pytest.raises(module.Stage1TextCacheError, match="over-length"):
        module.validate_cache_payload(bad)


def test_prompt_template_and_payload_digest_are_unambiguous() -> None:
    assert hashlib.sha256(module.DEFAULT_PROMPT.encode()).hexdigest()
    context = torch.zeros((module.CONTEXT_LEN, module.CONTEXT_DIM), dtype=torch.bfloat16)
    mask = torch.zeros(module.CONTEXT_LEN, dtype=torch.bool)
    first = module._payload_sha256(context, mask)  # noqa: SLF001
    context[0, 0] = 1
    second = module._payload_sha256(context, mask)  # noqa: SLF001
    assert first != second


def test_stage1_long_launch_requires_explicit_cache_audit_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    stats = tmp_path / "stats.json"
    stats.write_text("{}\n", encoding="utf-8")
    stats_sha = hashlib.sha256(stats.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_audit = tmp_path / "cache.audit.json"
    cache_audit.write_text("{}\n", encoding="utf-8")
    cache_audit_sha = hashlib.sha256(cache_audit.read_bytes()).hexdigest()
    model_base = tmp_path / "models"
    vae = (
        model_base
        / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
    )
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"vae")
    dit_dir = model_base / "Wan-AI/Wan2.2-TI2V-5B"
    dit_dir.mkdir(parents=True)
    for index in range(1, 4):
        (dit_dir / f"diffusion_pytorch_model-{index:05d}-of-00003.safetensors").write_bytes(
            b"dit"
        )
    action = tmp_path / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    action.write_bytes(b"action")

    monkeypatch.setattr(
        module,
        "verify_text_cache_audit",
        lambda *args, **kwargs: (
            {
                "cache": {"aggregate_payload_sha256": "a" * 64},
                "inventory": {"sha256": "b" * 64},
                "stage1_config": {
                    "path": str(stage1.DEFAULT_STAGE1_CONFIG.resolve()),
                    "sha256": hashlib.sha256(
                        stage1.DEFAULT_STAGE1_CONFIG.read_bytes()
                    ).hexdigest(),
                },
            },
            [],
        ),
    )
    kwargs = {
        "require_artifacts": True,
        "dataset_root_override": dataset_root,
        "dataset_stats_override": stats,
        "dataset_stats_sha256_override": stats_sha,
        "text_cache_override": cache,
        "text_cache_audit_override": cache_audit,
        "output_dir_override": tmp_path / "new-output",
        "model_base_path_override": model_base,
        "action_dit_init_override": action,
        "training_seed_override": 1,
    }
    with pytest.raises(stage1.Stage1ProtocolError, match="audit SHA-256 is required"):
        stage1.validate_stage1_config(**kwargs)
    report = stage1.validate_stage1_config(
        **kwargs,
        text_cache_audit_sha256_override=cache_audit_sha,
    )
    assert report["artifacts"]["text_embedding_cache_audit"]["sha256"] == cache_audit_sha
