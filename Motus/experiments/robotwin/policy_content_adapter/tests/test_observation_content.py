from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin.policy_content_adapter.observation_content import (
    extract_observation_visual_tokens,
)


class _DummyVideoModel:
    def __init__(self, *, patch_dtype=torch.float32) -> None:
        self.requested_layers = None
        self.wan_model = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=torch.empty(1, dtype=patch_dtype))
        )

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        assert video.shape[1:3] == (3, 1)
        return torch.ones(video.shape[0], 48, 1, 4, 4, dtype=torch.float32)

    def get_layer_features(
        self, latent, timestep, text, *, layer_indices, stop_after_last_requested
    ):
        self.requested_layers = layer_indices
        assert stop_after_last_requested is True
        return [torch.ones(latent.shape[0], 4, 3072, dtype=latent.dtype)]


def test_observation_only_extractor_uses_one_based_layer_and_detaches() -> None:
    video_model = _DummyVideoModel()
    model = SimpleNamespace(
        video_model=video_model,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    first = torch.rand(2, 3, 8, 8, requires_grad=True)
    text = torch.rand(2, 5, 6)
    tokens = extract_observation_visual_tokens(
        model,
        first_frame=first,
        language_embeddings=text,
        capture_layer=16,
    )
    assert tokens.shape == (2, 4, 3072)
    assert not tokens.requires_grad
    assert video_model.requested_layers == [15]


def test_observation_extractor_rejects_out_of_range_images() -> None:
    model = SimpleNamespace(
        video_model=_DummyVideoModel(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        extract_observation_visual_tokens(
            model,
            first_frame=torch.full((1, 3, 8, 8), 2.0),
            language_embeddings=torch.rand(1, 2, 3),
        )


def test_vae_output_is_cast_to_wan_patch_dtype() -> None:
    video_model = _DummyVideoModel(patch_dtype=torch.bfloat16)
    model = SimpleNamespace(
        video_model=video_model,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    tokens = extract_observation_visual_tokens(
        model,
        first_frame=torch.rand(1, 3, 8, 8),
        language_embeddings=torch.rand(1, 2, 3),
    )
    assert tokens.dtype == torch.bfloat16
