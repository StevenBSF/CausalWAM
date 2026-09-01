"""Deterministic current-observation content extraction for Motus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .protocol import DEFAULT_BACKBONE_DIM, DEFAULT_CAPTURE_LAYER


class ObservationContentError(RuntimeError):
    pass


def extract_observation_visual_tokens(
    motus_model: Any,
    *,
    first_frame: torch.Tensor,
    language_embeddings: Sequence[torch.Tensor] | torch.Tensor,
    capture_layer: int = DEFAULT_CAPTURE_LAYER,
) -> torch.Tensor:
    """Run a video-only WAN branch at t=0 and return one-based layer features.

    This intentionally does not use the noisy future-video/action/understanding
    tokens from Motus's joint denoising path.  The returned tensor is frozen;
    gradients begin at the Content Head.
    """

    if capture_layer <= 0:
        raise ValueError("capture_layer is one-based and must be positive")
    if first_frame.ndim != 4 or first_frame.shape[1] != 3:
        raise ValueError("first_frame must be [B,3,H,W]")
    if not torch.is_floating_point(first_frame):
        raise TypeError("first_frame must be floating point")
    if not bool(torch.isfinite(first_frame).all().item()):
        raise ValueError("first_frame contains NaN or infinity")
    if bool((first_frame < 0).any().item()) or bool((first_frame > 1).any().item()):
        raise ValueError("first_frame must be normalized to [0,1]")

    video_model = getattr(motus_model, "video_model", None)
    if video_model is None:
        raise TypeError("motus_model has no video_model")
    if isinstance(language_embeddings, torch.Tensor):
        if language_embeddings.ndim != 3:
            raise ValueError("language_embeddings tensor must be [B,L,D]")
        text = [language_embeddings[index] for index in range(first_frame.shape[0])]
    else:
        text = list(language_embeddings)
    if len(text) != first_frame.shape[0]:
        raise ValueError("one language embedding is required per observation")

    device = getattr(motus_model, "device", first_frame.device)
    dtype = getattr(motus_model, "dtype", first_frame.dtype)
    with torch.no_grad():
        frame = first_frame.to(device=device, dtype=dtype)
        normalized = (frame * 2.0 - 1.0).unsqueeze(2)
        latent = video_model.encode_video(normalized)
        patch_embedding = getattr(
            getattr(video_model, "wan_model", None), "patch_embedding", None
        )
        patch_weight = getattr(patch_embedding, "weight", None)
        if patch_weight is not None:
            latent = latent.to(device=patch_weight.device, dtype=patch_weight.dtype)
        else:
            latent = latent.to(device=device, dtype=dtype)
        timestep = torch.zeros(first_frame.shape[0], device=device, dtype=dtype)
        text = [item.to(device=device, dtype=dtype) for item in text]
        # Public protocol uses one-based layer numbering.  The author helper
        # accepts zero-based block indices and appends a final decoded output.
        features = video_model.get_layer_features(
            latent,
            timestep,
            text,
            layer_indices=[capture_layer - 1],
            stop_after_last_requested=True,
        )
    if not isinstance(features, list) or len(features) != 1:
        raise ObservationContentError(
            "WanVideoModel.get_layer_features did not return exactly one requested layer"
        )
    visual = features[0].detach().to(device=device, dtype=dtype)
    if visual.ndim != 3 or visual.shape[0] != first_frame.shape[0]:
        raise ObservationContentError(
            f"captured WAN tokens must be [B,L,D], got {tuple(visual.shape)}"
        )
    if visual.shape[-1] != DEFAULT_BACKBONE_DIM:
        raise ObservationContentError(
            f"expected WAN feature dim {DEFAULT_BACKBONE_DIM}, got {visual.shape[-1]}"
        )
    if not bool(torch.isfinite(visual).all().item()):
        raise ObservationContentError("captured WAN tokens are non-finite")
    return visual
