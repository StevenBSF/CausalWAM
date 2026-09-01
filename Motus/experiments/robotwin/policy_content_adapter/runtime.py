"""Motus_robotwin2 construction shared by cache, training and rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .lineage import validate_author_release_lineage


def load_lineage(path: str | Path, *, verify_files: bool = False) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_author_release_lineage(manifest, verify_files=verify_files)
    return manifest


def motus_config_kwargs(
    lineage: Mapping[str, Any], *, batch_size: int
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config_path = Path(lineage["checkpoint_config"]["path"])
    checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = {
        "common": checkpoint_config["common"],
        "model": {
            "action_expert": checkpoint_config["action_expert"],
            "und_expert": checkpoint_config["und_expert"],
        },
    }
    common = config["common"]
    model = config["model"]
    action = model["action_expert"]
    understanding = model["und_expert"]
    return {
        "wan_checkpoint_path": lineage["wan"]["root"],
        "vae_path": lineage["wan"]["vae"]["path"],
        "wan_config_path": lineage["wan"]["root"],
        "video_precision": "bfloat16",
        "vlm_checkpoint_path": lineage["vlm"]["root"],
        "und_expert_hidden_size": int(understanding["hidden_size"]),
        "und_expert_ffn_dim_multiplier": int(
            understanding["ffn_dim_multiplier"]
        ),
        "und_expert_norm_eps": float(understanding["norm_eps"]),
        "und_layers_to_extract": None,
        "vlm_adapter_input_dim": int(understanding["vlm"]["input_dim"]),
        "vlm_adapter_projector_type": str(
            understanding["vlm"]["projector_type"]
        ),
        "num_layers": 30,
        "action_state_dim": int(common["state_dim"]),
        "action_dim": int(common["action_dim"]),
        "action_expert_dim": int(action["hidden_size"]),
        "action_expert_ffn_dim_multiplier": int(action["ffn_dim_multiplier"]),
        # Author inference uses 1e-6 even though YAML comments list 1e-5.
        "action_expert_norm_eps": 1e-6,
        "global_downsample_rate": int(common["global_downsample_rate"]),
        "video_action_freq_ratio": int(common["video_action_freq_ratio"]),
        "num_video_frames": int(common["num_video_frames"]),
        "video_loss_weight": 1.0,
        "action_loss_weight": 1.0,
        "batch_size": int(batch_size),
        "video_height": int(common["video_height"]),
        "video_width": int(common["video_width"]),
        "load_pretrained_backbones": False,
        "training_mode": "finetune",
    }


def instantiate_author_release(
    lineage: Mapping[str, Any],
    *,
    batch_size: int,
    local_cuda_index: int,
    strict: bool = True,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Motus_robotwin2 construction requires CUDA")
    torch.cuda.set_device(local_cuda_index)
    # Delay the heavyweight import until after the exact local device is set.
    from models.motus import Motus, MotusConfig

    model = Motus(MotusConfig(**motus_config_kwargs(lineage, batch_size=batch_size)))
    model.load_checkpoint(lineage["checkpoint"]["path"], strict=strict)
    return model
