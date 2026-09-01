"""Self-contained deployment-side Content Head/GCA for Motus RoboTwin."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MotusContentHead(nn.Module):
    def __init__(
        self,
        backbone_dim: int = 3072,
        content_dim: int = 384,
        num_queries: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.backbone_dim = int(backbone_dim)
        self.content_dim = int(content_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.normalize_eps = 1e-12
        self.token_projection = nn.Linear(backbone_dim, content_dim)
        self.content_queries = nn.Parameter(torch.empty(num_queries, content_dim))
        self.cross_attention = nn.MultiheadAttention(
            content_dim, num_heads, batch_first=True, dropout=0.0, bias=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(content_dim, content_dim),
            nn.SiLU(),
            nn.Linear(content_dim, content_dim),
        )
        nn.init.normal_(self.content_queries, mean=0.0, std=content_dim**-0.5)

    def forward_content_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.backbone_dim:
            raise ValueError("deployment visual tokens have an invalid shape")
        projected = self.token_projection(visual_tokens)
        queries = (
            self.content_queries.to(device=projected.device, dtype=projected.dtype)
            .unsqueeze(0)
            .expand(projected.shape[0], -1, -1)
        )
        content, _ = self.cross_attention(
            queries, projected, projected, need_weights=False
        )
        return content

    def forward_contrastive(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        content = self.forward_content_tokens(visual_tokens)
        return F.normalize(
            self.mlp(content.mean(dim=1)),
            p=2.0,
            dim=-1,
            eps=self.normalize_eps,
        )


class GatedCrossAttentionAdapter(nn.Module):
    def __init__(
        self, action_dim: int = 1024, content_dim: int = 384, num_heads: int = 8
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.content_dim = int(content_dim)
        self.num_heads = int(num_heads)
        self.cross_attention = nn.MultiheadAttention(
            action_dim,
            num_heads,
            kdim=content_dim,
            vdim=content_dim,
            batch_first=True,
            dropout=0.0,
            bias=True,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, action_tokens, content_tokens):
        delta, _ = self.cross_attention(
            action_tokens, content_tokens, content_tokens, need_weights=False
        )
        return action_tokens + torch.tanh(self.gate).to(delta.dtype) * delta


class MotusPolicyContentConditioner(nn.Module):
    def __init__(self, *, capture_layer: int = 16) -> None:
        super().__init__()
        self.head = MotusContentHead()
        self.adapter = GatedCrossAttentionAdapter()
        self.enabled = True
        self.capture_layer = int(capture_layer)

    def content_tokens(self, visual_tokens):
        return self.head.forward_content_tokens(visual_tokens)

    def inject_action_tokens(self, action_tokens, content_tokens):
        if not self.enabled:
            return action_tokens
        if content_tokens is None:
            raise RuntimeError("deployment adapter has no observation content tokens")
        return self.adapter(action_tokens, content_tokens)


def extract_observation_visual_tokens(
    motus_model,
    *,
    first_frame: torch.Tensor,
    language_embeddings: Sequence[torch.Tensor],
    capture_layer: int,
) -> torch.Tensor:
    with torch.no_grad():
        frame = first_frame.to(device=motus_model.device, dtype=motus_model.dtype)
        latent = motus_model.video_model.encode_video((frame * 2.0 - 1.0).unsqueeze(2))
        patch_weight = motus_model.video_model.wan_model.patch_embedding.weight
        latent = latent.to(device=patch_weight.device, dtype=patch_weight.dtype)
        timestep = torch.zeros(
            frame.shape[0], device=motus_model.device, dtype=motus_model.dtype
        )
        text = [
            item.to(device=motus_model.device, dtype=motus_model.dtype)
            for item in language_embeddings
        ]
        features = motus_model.video_model.get_layer_features(
            latent,
            timestep,
            text,
            layer_indices=[capture_layer - 1],
            stop_after_last_requested=True,
        )
    if len(features) != 1 or features[0].ndim != 3:
        raise RuntimeError("deployment WAN capture returned an invalid payload")
    return features[0].detach().to(
        device=motus_model.device, dtype=motus_model.dtype
    )
