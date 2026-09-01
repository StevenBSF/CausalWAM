from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.policy_content_adapter.runtime import motus_config_kwargs
from experiments.robotwin.policy_content_adapter.train import (
    build_training_scheduler,
    module_tensor_sha256,
    resolve_deepspeed_config,
)


def test_runtime_uses_author_inference_architecture(tmp_path: Path) -> None:
    config = {
        "common": {
            "action_dim": 14,
            "state_dim": 14,
            "num_video_frames": 8,
            "video_height": 384,
            "video_width": 320,
            "global_downsample_rate": 3,
            "video_action_freq_ratio": 2,
        },
        "model": {
            "action_expert": {
                "hidden_size": 1024,
                "ffn_dim_multiplier": 4,
                "norm_eps": 1e-5,
            },
            "und_expert": {
                "hidden_size": 512,
                "ffn_dim_multiplier": 4,
                "norm_eps": 1e-5,
                "vlm": {"input_dim": 2048, "projector_type": "mlp3x_silu"},
            },
        },
    }
    config_path = tmp_path / "config.json"
    checkpoint_config = {
        "common": config["common"],
        "action_expert": config["model"]["action_expert"],
        "und_expert": config["model"]["und_expert"],
    }
    config_path.write_text(json.dumps(checkpoint_config), encoding="utf-8")
    lineage = {
        "checkpoint_config": {"path": str(config_path)},
        "wan": {"root": "/wan", "vae": {"path": "/wan/vae"}},
        "vlm": {"root": "/vlm"},
    }
    kwargs = motus_config_kwargs(lineage, batch_size=2)
    assert kwargs["batch_size"] == 2
    assert kwargs["action_expert_dim"] == 1024
    assert kwargs["action_expert_norm_eps"] == 1e-6
    assert kwargs["num_video_frames"] * kwargs["video_action_freq_ratio"] == 16
    assert kwargs["load_pretrained_backbones"] is False


def test_module_tensor_sha256_supports_bfloat16_scalar() -> None:
    module = torch.nn.Module()
    module.register_buffer("gate", torch.tensor(0.0, dtype=torch.bfloat16))

    initial = module_tensor_sha256(module)
    assert initial == module_tensor_sha256(module)

    module.gate.fill_(1.0)
    assert module_tensor_sha256(module) != initial


def test_deepspeed_config_has_explicit_micro_batch_size() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "deepspeed_zero1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["train_micro_batch_size_per_gpu"] == 1


def test_deepspeed_runtime_values_follow_audited_training_config() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "deepspeed_zero1.json"
    resolved = resolve_deepspeed_config(
        config_path,
        {
            "per_device_batch": 2,
            "gradient_accumulation_steps": 4,
            "world_size": 8,
            "grad_clip_norm": 0.5,
        },
    )

    assert resolved["train_micro_batch_size_per_gpu"] == 2
    assert resolved["gradient_accumulation_steps"] == 4
    assert resolved["train_batch_size"] == 64
    assert resolved["gradient_clipping"] == 0.5


def test_formal_scheduler_matches_motus_author_linear_contract() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=5.0e-5)
    scheduler = build_training_scheduler(
        optimizer,
        {
            "scheduler": "motus_author_linear",
            "warmup_steps": 200,
            "cycle_length": 5_000_000,
            "f_max": 0.99,
            "f_min": 0.4,
            "f_start": 1.0e-6,
        },
    )
    assert scheduler.step_count == 0
    assert optimizer.param_groups[0]["lr"] == 5.0e-5
    scheduler.step()
    expected_multiplier = 1.0e-6 + (0.99 - 1.0e-6) / 200
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        5.0e-5 * expected_multiplier
    )
