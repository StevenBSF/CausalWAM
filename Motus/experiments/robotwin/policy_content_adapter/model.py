"""Content Head and zero-init action-token adapter for Motus."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_HEADS,
    DEFAULT_BACKBONE_DIM,
    DEFAULT_CAPTURE_LAYER,
    DEFAULT_CONTENT_DIM,
    DEFAULT_CONTENT_HEADS,
    DEFAULT_CONTENT_QUERIES,
)


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


class MotusContentHead(nn.Module):
    """Convert frozen WAN visual tokens into policy and contrastive content."""

    def __init__(
        self,
        *,
        backbone_dim: int = DEFAULT_BACKBONE_DIM,
        content_dim: int = DEFAULT_CONTENT_DIM,
        num_queries: int = DEFAULT_CONTENT_QUERIES,
        num_heads: int = DEFAULT_CONTENT_HEADS,
        normalize_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if backbone_dim <= 0 or content_dim <= 0 or num_queries <= 0:
            raise ValueError("head dimensions and query count must be positive")
        if num_heads <= 0 or content_dim % num_heads:
            raise ValueError("num_heads must divide content_dim")
        if normalize_eps <= 0:
            raise ValueError("normalize_eps must be positive")
        self.backbone_dim = int(backbone_dim)
        self.content_dim = int(content_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.normalize_eps = float(normalize_eps)

        self.token_projection = nn.Linear(self.backbone_dim, self.content_dim)
        self.content_queries = nn.Parameter(
            torch.empty(self.num_queries, self.content_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.content_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
            bias=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.SiLU(),
            nn.Linear(self.content_dim, self.content_dim),
        )
        nn.init.normal_(
            self.content_queries, mean=0.0, std=self.content_dim**-0.5
        )

    def _project(
        self,
        visual_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must be [B,L,D]")
        batch, tokens, feature_dim = visual_tokens.shape
        if batch <= 0 or tokens <= 0 or feature_dim != self.backbone_dim:
            raise ValueError(
                f"expected non-empty [B,L,{self.backbone_dim}], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if not torch.is_floating_point(visual_tokens):
            raise TypeError("visual_tokens must be floating point")
        if not bool(torch.isfinite(visual_tokens).all().item()):
            raise ValueError("visual_tokens contains NaN or infinity")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, tokens):
                raise ValueError("key_padding_mask shape does not match visual tokens")
            if key_padding_mask.dtype is not torch.bool:
                raise TypeError("key_padding_mask must be boolean")
            if bool(key_padding_mask.all(dim=1).any().item()):
                raise ValueError("a sample may not mask every visual token")
            key_padding_mask = key_padding_mask.to(visual_tokens.device)
        return self.token_projection(visual_tokens), key_padding_mask

    def forward_content_tokens(
        self,
        visual_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected, key_padding_mask = self._project(
            visual_tokens, key_padding_mask
        )
        queries = self.content_queries.to(
            device=projected.device, dtype=projected.dtype
        ).unsqueeze(0).expand(projected.shape[0], -1, -1)
        content, _ = self.cross_attention(
            query=queries,
            key=projected,
            value=projected,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        expected = (projected.shape[0], self.num_queries, self.content_dim)
        if tuple(content.shape) != expected:
            raise RuntimeError(f"content shape changed: {tuple(content.shape)}")
        return content

    def forward_contrastive(
        self,
        visual_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        content = self.forward_content_tokens(
            visual_tokens, key_padding_mask=key_padding_mask
        )
        embedding = self.mlp(content.mean(dim=1))
        return F.normalize(
            embedding, p=2.0, dim=-1, eps=self.normalize_eps
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_contrastive(visual_tokens)


class GatedCrossAttentionAdapter(nn.Module):
    """Inject content into Motus Action Expert tokens with an exact zero gate."""

    def __init__(
        self,
        *,
        action_dim: int = DEFAULT_ACTION_DIM,
        content_dim: int = DEFAULT_CONTENT_DIM,
        num_heads: int = DEFAULT_ACTION_HEADS,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or content_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if num_heads <= 0 or action_dim % num_heads:
            raise ValueError("num_heads must divide action_dim")
        self.action_dim = int(action_dim)
        self.content_dim = int(content_dim)
        self.num_heads = int(num_heads)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.action_dim,
            num_heads=self.num_heads,
            kdim=self.content_dim,
            vdim=self.content_dim,
            batch_first=True,
            dropout=0.0,
            bias=True,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self, action_tokens: torch.Tensor, content_tokens: torch.Tensor
    ) -> torch.Tensor:
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_tokens must be [B,T,{self.action_dim}]"
            )
        if content_tokens.ndim != 3 or content_tokens.shape[-1] != self.content_dim:
            raise ValueError(
                f"content_tokens must be [B,Q,{self.content_dim}]"
            )
        if action_tokens.shape[0] != content_tokens.shape[0]:
            raise ValueError("action/content batch sizes differ")
        if not bool(torch.isfinite(action_tokens).all().item()):
            raise ValueError("action_tokens contains NaN or infinity")
        if not bool(torch.isfinite(content_tokens).all().item()):
            raise ValueError("content_tokens contains NaN or infinity")
        delta, _ = self.cross_attention(
            query=action_tokens,
            key=content_tokens,
            value=content_tokens,
            need_weights=False,
        )
        scale = torch.tanh(self.gate).to(delta.dtype)
        return action_tokens + scale * delta

    @property
    def gate_value(self) -> float:
        return float(torch.tanh(self.gate.detach().float()).item())


class MotusPolicyContentConditioner(nn.Module):
    """Own the Head/GCA and enforce one content representation per observation."""

    def __init__(
        self,
        head: MotusContentHead | None = None,
        adapter: GatedCrossAttentionAdapter | None = None,
        *,
        enabled: bool = True,
        capture_layer: int = DEFAULT_CAPTURE_LAYER,
    ) -> None:
        super().__init__()
        self.head = head if head is not None else MotusContentHead()
        self.adapter = adapter if adapter is not None else GatedCrossAttentionAdapter()
        self.enabled = bool(enabled)
        self.capture_layer = int(capture_layer)
        if self.capture_layer <= 0:
            raise ValueError("capture_layer is one-based and must be positive")

    def content_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.head.forward_content_tokens(visual_tokens)

    def contrastive_embedding(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.head.forward_contrastive(visual_tokens)

    def inject_action_tokens(
        self,
        action_tokens: torch.Tensor,
        content_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enabled:
            return action_tokens
        if content_tokens is None:
            raise RuntimeError(
                "Motus content adapter is enabled but this forward did not supply "
                "observation content tokens"
            )
        return self.adapter(action_tokens, content_tokens)


def configure_trainable_parameters(
    motus_model: nn.Module,
    conditioner: MotusPolicyContentConditioner,
    *,
    regime: str,
) -> dict[str, int]:
    """Apply the M-P1/M-P2 freeze contract and return audited parameter counts."""

    if regime not in {"m_p1", "m_p2"}:
        raise ValueError(f"unknown Motus policy regime {regime!r}")
    for parameter in motus_model.parameters():
        parameter.requires_grad_(False)
    for parameter in conditioner.parameters():
        parameter.requires_grad_(True)
    if regime == "m_p2":
        action_expert = getattr(motus_model, "action_expert", None)
        if not isinstance(action_expert, nn.Module):
            raise TypeError("Motus model has no Action Expert module")
        for parameter in action_expert.parameters():
            parameter.requires_grad_(True)
    return {
        "conditioner": parameter_count(conditioner, trainable_only=True),
        "action_expert": parameter_count(
            getattr(motus_model, "action_expert"), trainable_only=True
        ),
        "total_model": parameter_count(motus_model, trainable_only=True),
    }


def optimizer_parameter_groups(
    motus_model: nn.Module,
    conditioner: MotusPolicyContentConditioner,
    *,
    head_adapter_lr: float,
    action_expert_lr: float | None,
) -> list[dict[str, object]]:
    if head_adapter_lr <= 0:
        raise ValueError("head_adapter_lr must be positive")
    groups: list[dict[str, object]] = [
        {
            "name": "content_head_gca",
            "params": [p for p in conditioner.parameters() if p.requires_grad],
            "lr": float(head_adapter_lr),
        }
    ]
    action_parameters = [
        p for p in getattr(motus_model, "action_expert").parameters() if p.requires_grad
    ]
    if action_parameters:
        if action_expert_lr is None or action_expert_lr <= 0:
            raise ValueError("a trainable Action Expert requires a positive LR")
        groups.append(
            {
                "name": "action_expert",
                "params": action_parameters,
                "lr": float(action_expert_lr),
            }
        )
    if any(len(group["params"]) == 0 for group in groups):
        raise RuntimeError("optimizer contains an empty parameter group")
    return groups
