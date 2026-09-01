"""Minimal action-facing content adapter without changing native FastWAM code.

The implementation deliberately keeps the released ``mot`` state-dict schema
untouched.  A runtime hook is attached to the output of
``ActionDiT.action_encoder``.  This is the earliest action-token point shared
by native training and cached RoboTwin rollout.
"""

from __future__ import annotations

import hashlib
import os
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from experiments.robotwin.e0_e1.backbone import (
    run_video_prefill_with_captures,
    strict_load_release_checkpoint,
)
from experiments.robotwin.e0_e1.head import (
    ContrastiveContentHead,
    DEFAULT_BACKBONE_DIM,
    DEFAULT_EMBED_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_QUERIES,
)


POLICY_CHECKPOINT_SCHEMA = "fastwam.policy_content_adapter"
POLICY_CHECKPOINT_VERSION = 3
DEFAULT_CONTENT_LAYER = 16
DEFAULT_CONTENT_TOKEN_COUNT = 120
DEFAULT_ACTION_DIM = 1024
EXPECTED_HEAD_PARAMETER_COUNT = 2_070_144
EXPECTED_ADAPTER_PARAMETER_COUNT = 2_887_681
EXPECTED_ACTION_DIT_PARAMETER_COUNT = 1_020_900_366


def _parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


class PolicyContentHead(ContrastiveContentHead):
    """E1--E3 compatible head exposing all eight content query tokens.

    No parameters are added or renamed relative to
    :class:`ContrastiveContentHead`, so ``payload['head']`` from E1/E2/E3 loads
    strictly.  The policy path uses ``forward_content_tokens`` without pooling;
    the contrastive path preserves the original mean/MLP/L2 behavior.
    """

    def _validate_and_project(
        self,
        visual_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must be [batch,tokens,backbone_dim], got "
                f"{tuple(visual_tokens.shape)}"
            )
        batch_size, sequence_length, feature_dim = visual_tokens.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("visual_tokens must have non-empty batch and token dimensions")
        if feature_dim != self.backbone_dim:
            raise ValueError(
                f"expected visual token dim {self.backbone_dim}, got {feature_dim}"
            )
        if not torch.is_floating_point(visual_tokens):
            raise TypeError("visual_tokens must be floating point")
        if not bool(torch.isfinite(visual_tokens).all().item()):
            raise ValueError("visual_tokens contains NaN or infinity")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "key_padding_mask shape mismatch: "
                    f"{tuple(key_padding_mask.shape)} vs {(batch_size, sequence_length)}"
                )
            if key_padding_mask.dtype is not torch.bool:
                raise TypeError("key_padding_mask must be bool")
            if bool(key_padding_mask.all(dim=1).any().item()):
                raise ValueError("each sample must retain at least one visual token")
            key_padding_mask = key_padding_mask.to(device=visual_tokens.device)
        return self.token_projection(visual_tokens), key_padding_mask

    def forward_content_tokens(
        self,
        visual_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return policy content tokens ``Zc`` shaped ``[B,8,384]``."""

        projected, key_padding_mask = self._validate_and_project(
            visual_tokens, key_padding_mask
        )
        queries = self.content_queries.to(
            device=projected.device, dtype=projected.dtype
        ).unsqueeze(0).expand(projected.shape[0], -1, -1)
        content_tokens, _ = self.cross_attention(
            query=queries,
            key=projected,
            value=projected,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        expected = (projected.shape[0], self.num_queries, self.embed_dim)
        if tuple(content_tokens.shape) != expected:
            raise RuntimeError(
                f"content query output changed shape: {tuple(content_tokens.shape)} vs {expected}"
            )
        return content_tokens

    # A concise alias used by the runtime and audit code.
    content_tokens = forward_content_tokens

    def forward_contrastive(
        self,
        visual_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the E1--E3 mean/MLP/L2 embedding shaped ``[B,384]``."""

        content_tokens = self.forward_content_tokens(
            visual_tokens, key_padding_mask=key_padding_mask
        )
        embedding = self.mlp(content_tokens.mean(dim=1))
        return F.normalize(embedding, p=2.0, dim=-1, eps=self.normalize_eps)

    contrastive_embedding = forward_contrastive

    def forward(
        self,
        visual_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_contrastive(
            visual_tokens, key_padding_mask=key_padding_mask
        )


class GatedCrossAttentionAdapter(nn.Module):
    """One exact zero-init gated cross-attention action adapter."""

    def __init__(
        self,
        action_dim: int = DEFAULT_ACTION_DIM,
        content_dim: int = DEFAULT_EMBED_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or content_dim <= 0:
            raise ValueError("action_dim and content_dim must be positive")
        if num_heads <= 0 or action_dim % num_heads != 0:
            raise ValueError("num_heads must divide action_dim")
        self.action_dim = int(action_dim)
        self.content_dim = int(content_dim)
        self.num_heads = int(num_heads)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.action_dim,
            num_heads=self.num_heads,
            kdim=self.content_dim,
            vdim=self.content_dim,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        # Scalar, exactly zero. tanh(0) is exactly representable in all used dtypes.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self, action_tokens: torch.Tensor, content_tokens: torch.Tensor
    ) -> torch.Tensor:
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_tokens must be [B,T,{self.action_dim}], got {tuple(action_tokens.shape)}"
            )
        if content_tokens.ndim != 3 or content_tokens.shape[-1] != self.content_dim:
            raise ValueError(
                f"content_tokens must be [B,Q,{self.content_dim}], got {tuple(content_tokens.shape)}"
            )
        if action_tokens.shape[0] != content_tokens.shape[0]:
            raise ValueError("action/content batch sizes differ")
        delta, _ = self.cross_attention(
            query=action_tokens,
            key=content_tokens,
            value=content_tokens,
            need_weights=False,
        )
        scale = torch.tanh(self.gate).to(device=delta.device, dtype=delta.dtype)
        return action_tokens + scale * delta

    @property
    def gate_value(self) -> float:
        return float(torch.tanh(self.gate.detach().float()).item())


class PolicyContentConditioner(nn.Module):
    """Own the compatible head, the single GCA, and per-prefill ``Zc`` state."""

    def __init__(
        self,
        head: PolicyContentHead | None = None,
        adapter: GatedCrossAttentionAdapter | None = None,
        *,
        enabled: bool = True,
        content_layer: int = DEFAULT_CONTENT_LAYER,
    ) -> None:
        super().__init__()
        self.head = head if head is not None else PolicyContentHead()
        self.adapter = adapter if adapter is not None else GatedCrossAttentionAdapter()
        self.enabled = bool(enabled)
        self.content_layer = int(content_layer)
        if self.content_layer <= 0:
            raise ValueError("content_layer uses one-based indexing and must be positive")
        self._active_content_tokens: torch.Tensor | None = None
        # A Python scalar populated by a hook on the *original* action-path Zc.
        # Keeping this outside the forward return is important: Accelerate's
        # bf16 wrapper recursively converts returned tensors to new fp32
        # descendants, whose ``.grad`` is unrelated to the tensor used by the
        # loss graph.
        self._action_content_gradient_norm: float | None = None

    def content_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.head.forward_content_tokens(visual_tokens)

    def contrastive(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.head.forward_contrastive(visual_tokens)

    def set_visual_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if (
            self.content_layer == DEFAULT_CONTENT_LAYER
            and self.head.backbone_dim == DEFAULT_BACKBONE_DIM
            and tuple(visual_tokens.shape[1:])
            != (DEFAULT_CONTENT_TOKEN_COUNT, DEFAULT_BACKBONE_DIM)
        ):
            raise ValueError(
                "Layer-16 policy tokens must be [B,120,3072], got "
                f"{tuple(visual_tokens.shape)}"
            )
        content = self.content_tokens(visual_tokens)
        expected = (visual_tokens.shape[0], DEFAULT_NUM_QUERIES, DEFAULT_EMBED_DIM)
        if (
            self.head.num_queries == DEFAULT_NUM_QUERIES
            and self.head.embed_dim == DEFAULT_EMBED_DIM
            and tuple(content.shape) != expected
        ):
            raise RuntimeError(
                f"policy content tokens must be [B,8,384], got {tuple(content.shape)}"
            )
        self._active_content_tokens = content
        return content

    def set_content_tokens(self, content_tokens: torch.Tensor) -> None:
        if content_tokens.ndim != 3 or content_tokens.shape[-1] != self.head.embed_dim:
            raise ValueError("active content tokens have an invalid shape")
        self._active_content_tokens = content_tokens

    def clear_active_content(self) -> None:
        self._active_content_tokens = None

    def arm_action_content_gradient_audit(self, content_tokens: torch.Tensor) -> None:
        """Record the next action-path Zc gradient without returning a live tensor."""

        if not content_tokens.requires_grad:
            raise ValueError("action-path content tokens must require gradients")
        self._action_content_gradient_norm = None

        def capture(gradient: torch.Tensor) -> torch.Tensor:
            norm = gradient.detach().double().square().sum().sqrt()
            value = float(norm.item())
            if not torch.isfinite(norm):
                raise FloatingPointError("official content-token gradient is non-finite")
            self._action_content_gradient_norm = value
            return gradient

        content_tokens.register_hook(capture)

    def consume_action_content_gradient_audit(self) -> float:
        """Return and clear the gradient recorded by the latest official forward."""

        value = self._action_content_gradient_norm
        self._action_content_gradient_norm = None
        if value is None:
            raise RuntimeError("official action-path content gradient hook did not run")
        return float(value)

    def inject_action_tokens(
        self,
        action_tokens: torch.Tensor,
        content_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return action_tokens
        active = self._active_content_tokens if content_tokens is None else content_tokens
        if active is None:
            raise RuntimeError(
                "policy content adapter is enabled but no Layer-16 content was set "
                "before ActionDiT.action_encoder"
            )
        return self.adapter(action_tokens, active)


class PolicyContentRuntime:
    """Reversible hooks joining the experiment conditioner to a FastWAM instance."""

    def __init__(
        self,
        model: nn.Module,
        conditioner: PolicyContentConditioner,
        *,
        patch_video_prefill: bool,
    ) -> None:
        self.model = model
        self.conditioner = conditioner
        self._closed = False
        self._native_prefill = model.mot.prefill_video_cache

        def action_encoder_hook(_module, _inputs, output):
            return self.conditioner.inject_action_tokens(output)

        self._action_hook = model.action_expert.action_encoder.register_forward_hook(
            action_encoder_hook
        )
        self.patch_video_prefill = bool(patch_video_prefill)
        if self.patch_video_prefill:
            runtime = self

            def captured_prefill(mot_self, **kwargs):
                captures, cache = run_video_prefill_with_captures(
                    mot_self,
                    video_tokens=kwargs["video_tokens"],
                    video_freqs=kwargs["video_freqs"],
                    video_t_mod=kwargs["video_t_mod"],
                    video_context_payload=kwargs.get("video_context_payload"),
                    video_attention_mask=kwargs["video_attention_mask"],
                    capture_layers=(runtime.conditioner.content_layer,),
                )
                runtime.conditioner.set_visual_tokens(
                    captures[runtime.conditioner.content_layer]
                )
                return cache

            model.mot.prefill_video_cache = types.MethodType(
                captured_prefill, model.mot
            )
        # Keep a non-Module reference for diagnostics and duplicate-install guards.
        model.__dict__["_policy_content_runtime"] = self

    def capture_video_prefill(self, **kwargs) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        captures, cache = run_video_prefill_with_captures(
            self.model.mot,
            video_tokens=kwargs["video_tokens"],
            video_freqs=kwargs["video_freqs"],
            video_t_mod=kwargs["video_t_mod"],
            video_context_payload=kwargs.get("video_context_payload"),
            video_attention_mask=kwargs["video_attention_mask"],
            capture_layers=(self.conditioner.content_layer,),
        )
        return captures[self.conditioner.content_layer], cache

    def close(self) -> None:
        if self._closed:
            return
        self._action_hook.remove()
        if self.patch_video_prefill:
            self.model.mot.prefill_video_cache = self._native_prefill
        if self.model.__dict__.get("_policy_content_runtime") is self:
            self.model.__dict__.pop("_policy_content_runtime", None)
        self.conditioner.clear_active_content()
        self._closed = True


def install_policy_content_adapter(
    model: nn.Module,
    *,
    head: PolicyContentHead | None = None,
    adapter: GatedCrossAttentionAdapter | None = None,
    enabled: bool = True,
    content_layer: int = DEFAULT_CONTENT_LAYER,
    patch_video_prefill: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PolicyContentRuntime:
    """Install the experiment hook explicitly; native behavior is untouched otherwise."""

    existing = model.__dict__.get("_policy_content_runtime")
    if existing is not None:
        raise RuntimeError("a policy content runtime is already installed on this model")
    action_dim = int(model.action_expert.hidden_dim)
    conditioner = PolicyContentConditioner(
        head=head or PolicyContentHead(),
        adapter=adapter
        or GatedCrossAttentionAdapter(
            action_dim=action_dim,
            content_dim=DEFAULT_EMBED_DIM,
            num_heads=DEFAULT_NUM_HEADS,
        ),
        enabled=enabled,
        content_layer=content_layer,
    )
    if device is None:
        device = getattr(model, "device", next(model.parameters()).device)
    if dtype is None:
        dtype = getattr(model, "torch_dtype", next(model.parameters()).dtype)
    conditioner.to(device=device, dtype=dtype)
    return PolicyContentRuntime(
        model, conditioner, patch_video_prefill=patch_video_prefill
    )


def load_e1_e3_head_checkpoint(
    head: PolicyContentHead, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Strictly load the common E1/E2/E3 ``payload['head']`` schema."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("head"), Mapping):
        raise ValueError(f"content-head checkpoint has no mapping payload['head']: {path}")
    config = payload.get("head_config")
    expected_config = {
        "backbone_dim": head.backbone_dim,
        "embed_dim": head.embed_dim,
        "num_queries": head.num_queries,
        "num_heads": head.num_heads,
    }
    if isinstance(config, Mapping):
        for key, expected in expected_config.items():
            if int(config.get(key, -1)) != int(expected):
                raise ValueError(
                    f"head checkpoint {key} mismatch: {config.get(key)} vs {expected}"
                )
    incompatible = head.load_state_dict(dict(payload["head"]), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict head load failed: {incompatible}")
    return payload


def configure_trainable_modules(
    model: nn.Module,
    conditioner: PolicyContentConditioner,
    regime: str,
) -> dict[str, int]:
    """Apply and audit the exact P-v1/P-v2 freeze contract."""

    regime = str(regime).lower().replace("-", "_")
    if regime not in {"p_v1", "p_v2"}:
        raise ValueError("regime must be p_v1 or p_v2")
    model.eval()
    model.requires_grad_(False)
    conditioner.train()
    conditioner.requires_grad_(True)
    if regime == "p_v2":
        model.action_expert.train()
        model.action_expert.requires_grad_(True)
    else:
        model.action_expert.eval()
    # Shared module aliases must not accidentally enable the video branch.
    model.video_expert.eval()
    model.video_expert.requires_grad_(False)
    model.vae.eval()
    model.vae.requires_grad_(False)
    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder.eval()
        model.text_encoder.requires_grad_(False)
    if getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.eval()
        model.proprio_encoder.requires_grad_(False)

    head_count = _parameter_count(conditioner.head, trainable_only=True)
    adapter_count = _parameter_count(conditioner.adapter, trainable_only=True)
    action_count = _parameter_count(model.action_expert, trainable_only=True)
    if (
        conditioner.head.backbone_dim == DEFAULT_BACKBONE_DIM
        and conditioner.head.embed_dim == DEFAULT_EMBED_DIM
        and conditioner.head.num_queries == DEFAULT_NUM_QUERIES
        and conditioner.head.num_heads == DEFAULT_NUM_HEADS
        and head_count != EXPECTED_HEAD_PARAMETER_COUNT
    ):
        raise RuntimeError(f"unexpected content-head parameter count: {head_count}")
    if (
        conditioner.adapter.action_dim == DEFAULT_ACTION_DIM
        and conditioner.adapter.content_dim == DEFAULT_EMBED_DIM
        and adapter_count != EXPECTED_ADAPTER_PARAMETER_COUNT
    ):
        raise RuntimeError(f"unexpected adapter parameter count: {adapter_count}")
    if regime == "p_v1" and action_count != 0:
        raise RuntimeError("P-v1 unexpectedly left ActionDiT trainable")
    if regime == "p_v2" and action_count == 0:
        raise RuntimeError("P-v2 did not enable ActionDiT")
    if (
        regime == "p_v2"
        and model.action_expert.__class__.__name__ == "ActionDiT"
        and action_count != EXPECTED_ACTION_DIT_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "unexpected released ActionDiT parameter count: "
            f"{action_count} vs {EXPECTED_ACTION_DIT_PARAMETER_COUNT}"
        )
    frozen_video_with_grad = [
        name
        for name, parameter in model.video_expert.named_parameters()
        if parameter.requires_grad
    ]
    if frozen_video_with_grad:
        raise RuntimeError(
            f"video expert is not frozen: {frozen_video_with_grad[:5]}"
        )
    return {
        "content_head": head_count,
        "adapter": adapter_count,
        "action_dit": action_count,
        "total": head_count + adapter_count + action_count,
    }


def build_optimizer_param_groups(
    model: nn.Module,
    conditioner: PolicyContentConditioner,
    regime: str,
    *,
    head_adapter_lr: float = 1e-4,
    action_dit_lr: float = 1e-5,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Return auditable, disjoint P-v1/P-v2 AdamW groups."""

    if head_adapter_lr <= 0 or action_dit_lr <= 0 or weight_decay < 0:
        raise ValueError("learning rates must be positive and weight_decay non-negative")
    regime = str(regime).lower().replace("-", "_")
    groups: list[dict[str, Any]] = [
        {
            "name": "content_head_and_adapter",
            "params": [
                parameter
                for parameter in conditioner.parameters()
                if parameter.requires_grad
            ],
            "lr": float(head_adapter_lr),
            "weight_decay": float(weight_decay),
        }
    ]
    if regime == "p_v2":
        groups.append(
            {
                "name": "action_dit",
                "params": [
                    parameter
                    for parameter in model.action_expert.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(action_dit_lr),
                "weight_decay": float(weight_decay),
            }
        )
    elif regime != "p_v1":
        raise ValueError("regime must be p_v1 or p_v2")
    if any(not group["params"] for group in groups):
        raise RuntimeError("optimizer contains an empty parameter group")
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("optimizer parameter groups overlap")
    return groups


def module_grad_norm(module: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().pow(2).sum().cpu()
    return float(total.sqrt().item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path, *, include_sha256: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {resolved}")
    stat = resolved.stat()
    result: dict[str, Any] = {
        "kind": "file",
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = _sha256_file(resolved)
    return result


def directory_identity(path: str | Path) -> dict[str, Any]:
    """Content-address a small runtime directory such as the tokenizer.

    Relative paths, file sizes, and bytes are all included in the digest.  We
    reject symlinks so a checkpoint cannot silently bind an external mutable
    target through an apparently local tokenizer directory.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"artifact is not a directory: {resolved}")
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for candidate in sorted(resolved.rglob("*"), key=lambda value: value.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"artifact directory contains a symlink: {candidate}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(resolved).as_posix().encode("utf-8")
        stat = candidate.stat()
        digest.update(len(relative).to_bytes(8, byteorder="big", signed=False))
        digest.update(relative)
        digest.update(int(stat.st_size).to_bytes(8, byteorder="big", signed=False))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        file_count += 1
        total_size += int(stat.st_size)
    if file_count == 0:
        raise ValueError(f"artifact directory contains no regular files: {resolved}")
    return {
        "kind": "directory",
        "path": str(resolved),
        "file_count": file_count,
        "size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def artifact_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return file_identity(resolved, include_sha256=True)
    if resolved.is_dir():
        return directory_identity(resolved)
    raise FileNotFoundError(f"artifact does not exist: {resolved}")


def verify_artifact_identity(
    identity: Mapping[str, Any],
    *,
    path_override: str | Path | None = None,
    label: str = "artifact",
) -> dict[str, Any]:
    """Recompute and strictly compare a recorded file/directory identity."""

    if not isinstance(identity, Mapping) or not identity.get("path"):
        raise ValueError(f"{label} identity must be a mapping with a path")
    recorded_sha = str(identity.get("sha256", ""))
    if len(recorded_sha) != 64 or any(ch not in "0123456789abcdef" for ch in recorded_sha):
        raise ValueError(f"{label} identity lacks a lowercase SHA-256")
    path = identity["path"] if path_override is None else path_override
    actual = artifact_identity(path)
    expected_kind = str(identity.get("kind", "file"))
    if actual["kind"] != expected_kind:
        raise ValueError(
            f"{label} kind differs: {actual['kind']!r} vs {expected_kind!r}"
        )
    if int(actual["size_bytes"]) != int(identity.get("size_bytes", -1)):
        raise ValueError(f"{label} size differs from checkpoint identity")
    if expected_kind == "directory" and int(actual["file_count"]) != int(
        identity.get("file_count", -1)
    ):
        raise ValueError(f"{label} file count differs from checkpoint identity")
    if actual["sha256"] != recorded_sha:
        raise ValueError(f"{label} SHA-256 differs from checkpoint identity")
    return actual


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in module.state_dict().items()
    }


def module_state_sha256(module: nn.Module) -> str:
    """Hash a module state deterministically across devices."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        shape_bytes = ",".join(str(int(dim)) for dim in tensor.shape).encode("ascii")
        for field in (name_bytes, dtype_bytes, shape_bytes):
            digest.update(len(field).to_bytes(8, byteorder="big", signed=False))
            digest.update(field)
        # Flatten first: a scalar parameter (the exact-zero GCA gate) cannot be
        # dtype-viewed directly because PyTorch forbids changing element size
        # on a 0-D tensor.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_policy_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    conditioner: PolicyContentConditioner,
    base_checkpoint: str | Path,
    regime: str,
    step: int,
    run_config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    include_base_sha256: bool = True,
    verified_base_identity: Mapping[str, Any] | None = None,
    artifact_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Save a compact overlay; P-v1 does not duplicate the 12 GB base model."""

    regime = str(regime).lower().replace("-", "_")
    if regime not in {"p_v1", "p_v2"}:
        raise ValueError("regime must be p_v1 or p_v2")
    if not include_base_sha256:
        raise ValueError(
            "policy checkpoints must bind the immutable base checkpoint by SHA-256"
        )
    if verified_base_identity is None:
        base_identity = file_identity(base_checkpoint, include_sha256=True)
    else:
        base_path = Path(base_checkpoint).expanduser().resolve()
        recorded_path = Path(str(verified_base_identity.get("path", ""))).expanduser().resolve()
        if recorded_path != base_path:
            raise ValueError("verified base identity path differs from base_checkpoint")
        stat = base_path.stat()
        if int(verified_base_identity.get("size_bytes", -1)) != int(stat.st_size):
            raise ValueError("verified base identity size is stale")
        sha256 = str(verified_base_identity.get("sha256", ""))
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise ValueError("verified base identity lacks a lowercase SHA-256")
        base_identity = dict(verified_base_identity)
        base_identity.setdefault("kind", "file")

    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for name, identity in sorted((artifact_identities or {}).items()):
        if not isinstance(identity, Mapping):
            raise TypeError(f"artifact identity {name!r} is not a mapping")
        body = dict(identity)
        sha256 = str(body.get("sha256", ""))
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise ValueError(f"artifact identity {name!r} lacks a lowercase SHA-256")
        if not body.get("path") or int(body.get("size_bytes", -1)) < 0:
            raise ValueError(f"artifact identity {name!r} lacks path/size")
        body.setdefault("kind", "file")
        body.setdefault("required_for_rollout", False)
        normalized_artifacts[str(name)] = body

    payload: dict[str, Any] = {
        "schema": POLICY_CHECKPOINT_SCHEMA,
        "schema_version": POLICY_CHECKPOINT_VERSION,
        "regime": regime,
        "step": int(step),
        "base_checkpoint": base_identity,
        "artifact_identities": normalized_artifacts,
        "head_config": {
            "backbone_dim": conditioner.head.backbone_dim,
            "embed_dim": conditioner.head.embed_dim,
            "num_queries": conditioner.head.num_queries,
            "num_heads": conditioner.head.num_heads,
        },
        "adapter_config": {
            "action_dim": conditioner.adapter.action_dim,
            "content_dim": conditioner.adapter.content_dim,
            "num_heads": conditioner.adapter.num_heads,
            "content_layer": conditioner.content_layer,
            "gate_initialization": "exact_zero",
        },
        "content_head": _cpu_state(conditioner.head),
        "content_adapter": _cpu_state(conditioner.adapter),
        "run_config": dict(run_config),
    }
    if regime == "p_v2":
        payload["action_expert"] = _cpu_state(model.action_expert)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_policy_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    verify_base: bool = True,
    verify_runtime_artifacts: bool = True,
    runtime_artifacts: Mapping[str, str | Path] | None = None,
    patch_video_prefill: bool = True,
) -> tuple[PolicyContentRuntime, dict[str, Any], dict[str, Any]]:
    """Strictly load base release weights, then the compact policy overlay."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("policy checkpoint root must be a mapping")
    if payload.get("schema") != POLICY_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"not a {POLICY_CHECKPOINT_SCHEMA!r} checkpoint: {checkpoint}"
        )
    if int(payload.get("schema_version", -1)) != POLICY_CHECKPOINT_VERSION:
        raise ValueError("unsupported policy checkpoint schema version")
    base = payload.get("base_checkpoint")
    if not isinstance(base, Mapping) or not base.get("path"):
        raise ValueError("policy checkpoint lacks base_checkpoint identity")
    base_path = Path(str(base["path"])).expanduser().resolve()
    if str(base.get("kind", "file")) != "file":
        raise ValueError("base checkpoint identity must have kind='file'")
    actual_stat = base_path.stat()
    if int(base.get("size_bytes", -1)) != int(actual_stat.st_size):
        raise ValueError("base checkpoint size differs from policy checkpoint identity")
    if verify_base and not base.get("sha256"):
        raise ValueError(
            "verify_base=True requires a SHA-256-bound base checkpoint identity"
        )
    if verify_base:
        actual_sha = _sha256_file(base_path)
        if actual_sha != str(base["sha256"]):
            raise ValueError("base checkpoint SHA-256 differs from policy checkpoint identity")

    # This must precede hook installation because it audits the unchanged mot schema.
    release_audit = strict_load_release_checkpoint(
        model, base_path, compute_sha256=False
    )

    identities = payload.get("artifact_identities")
    if not isinstance(identities, Mapping):
        raise ValueError("policy checkpoint lacks artifact_identities mapping")
    verified_runtime: dict[str, dict[str, Any]] = {}
    if verify_runtime_artifacts:
        supplied = dict(runtime_artifacts or {})
        model_paths = dict(getattr(model, "model_paths", {}) or {})
        aliases = {
            "vae": "vae",
            "text_encoder": "text_encoder",
            "tokenizer": "tokenizer",
        }
        for name, raw_identity in sorted(identities.items()):
            if not isinstance(raw_identity, Mapping):
                raise ValueError(f"artifact identity {name!r} is not a mapping")
            if not bool(raw_identity.get("required_for_rollout", False)):
                continue
            actual_path = supplied.get(str(name))
            resolution_source = "runtime_override"
            if actual_path is None and str(name) in aliases:
                actual_path = model_paths.get(aliases[str(name)])
                resolution_source = "model_component"
            if (
                actual_path in {None, "SKIPPED_PRETRAIN"}
                and str(name) not in aliases
            ):
                # Non-model rollout artifacts (for example the simulator seed
                # bank and formal protocol lock) have no loader alias or CLI
                # override.  Their absolute, content-addressed checkpoint path
                # is therefore the runtime binding.  This remains fail-closed:
                # verify_artifact_identity below recomputes kind, size and
                # SHA-256 before any overlay is installed.
                recorded_path = str(raw_identity.get("path", "")).strip()
                if recorded_path:
                    candidate = Path(recorded_path).expanduser()
                    if not candidate.is_absolute():
                        raise ValueError(
                            f"rollout-required artifact {name!r} checkpoint path "
                            "must be absolute when no runtime override is supplied"
                        )
                    actual_path = candidate.resolve()
                    resolution_source = "checkpoint_identity"
            if actual_path in {None, "SKIPPED_PRETRAIN"}:
                raise ValueError(
                    f"rollout-required artifact {name!r} has no resolved runtime path"
                )
            verified_identity = verify_artifact_identity(
                raw_identity,
                path_override=actual_path,
                label=f"rollout artifact {name}",
            )
            verified_identity["resolution_source"] = resolution_source
            verified_identity["checkpoint_path"] = str(raw_identity["path"])
            verified_runtime[str(name)] = verified_identity
    head_config = dict(payload.get("head_config") or {})
    adapter_config = dict(payload.get("adapter_config") or {})
    required_head = {
        "backbone_dim": DEFAULT_BACKBONE_DIM,
        "embed_dim": DEFAULT_EMBED_DIM,
        "num_queries": DEFAULT_NUM_QUERIES,
        "num_heads": DEFAULT_NUM_HEADS,
    }
    required_adapter = {
        "action_dim": DEFAULT_ACTION_DIM,
        "content_dim": DEFAULT_EMBED_DIM,
        "num_heads": DEFAULT_NUM_HEADS,
        "content_layer": DEFAULT_CONTENT_LAYER,
    }
    for key, expected in required_head.items():
        if int(head_config.get(key, -1)) != expected:
            raise ValueError(
                f"policy prototype requires head {key}={expected}, "
                f"checkpoint has {head_config.get(key)!r}"
            )
    for key, expected in required_adapter.items():
        if int(adapter_config.get(key, -1)) != expected:
            raise ValueError(
                f"policy prototype requires adapter {key}={expected}, "
                f"checkpoint has {adapter_config.get(key)!r}"
            )
    head = PolicyContentHead(
        backbone_dim=int(head_config.get("backbone_dim", DEFAULT_BACKBONE_DIM)),
        embed_dim=int(head_config.get("embed_dim", DEFAULT_EMBED_DIM)),
        num_queries=int(head_config.get("num_queries", DEFAULT_NUM_QUERIES)),
        num_heads=int(head_config.get("num_heads", DEFAULT_NUM_HEADS)),
    )
    adapter = GatedCrossAttentionAdapter(
        action_dim=int(adapter_config.get("action_dim", DEFAULT_ACTION_DIM)),
        content_dim=int(adapter_config.get("content_dim", DEFAULT_EMBED_DIM)),
        num_heads=int(adapter_config.get("num_heads", DEFAULT_NUM_HEADS)),
    )
    if adapter.content_dim != head.embed_dim:
        raise ValueError("checkpoint head/adapter content dimensions differ")
    if adapter.action_dim != int(model.action_expert.hidden_dim):
        raise ValueError(
            "checkpoint adapter action dimension differs from the instantiated ActionDiT"
        )
    head.load_state_dict(dict(payload.get("content_head") or {}), strict=True)
    adapter.load_state_dict(dict(payload.get("content_adapter") or {}), strict=True)
    regime = str(payload.get("regime", ""))
    if regime == "p_v2":
        action_state = payload.get("action_expert")
        if not isinstance(action_state, Mapping):
            raise ValueError("P-v2 checkpoint lacks action_expert weights")
        model.action_expert.load_state_dict(dict(action_state), strict=True)
    elif regime != "p_v1":
        raise ValueError(f"invalid policy regime in checkpoint: {regime!r}")
    runtime = install_policy_content_adapter(
        model,
        head=head,
        adapter=adapter,
        enabled=True,
        content_layer=int(adapter_config.get("content_layer", DEFAULT_CONTENT_LAYER)),
        patch_video_prefill=patch_video_prefill,
    )
    audit = {
        "policy_checkpoint": file_identity(checkpoint, include_sha256=False),
        "base_checkpoint": dict(base),
        "release_load": asdict(release_audit),
        "head_parameter_count": _parameter_count(runtime.conditioner.head),
        "adapter_parameter_count": _parameter_count(runtime.conditioner.adapter),
        "action_expert_overlaid": regime == "p_v2",
        "verified_runtime_artifacts": verified_runtime,
    }
    return runtime, payload, audit


__all__ = [
    "DEFAULT_CONTENT_LAYER",
    "DEFAULT_CONTENT_TOKEN_COUNT",
    "EXPECTED_ACTION_DIT_PARAMETER_COUNT",
    "EXPECTED_ADAPTER_PARAMETER_COUNT",
    "EXPECTED_HEAD_PARAMETER_COUNT",
    "GatedCrossAttentionAdapter",
    "POLICY_CHECKPOINT_SCHEMA",
    "PolicyContentConditioner",
    "PolicyContentHead",
    "PolicyContentRuntime",
    "build_optimizer_param_groups",
    "artifact_identity",
    "configure_trainable_modules",
    "directory_identity",
    "file_identity",
    "install_policy_content_adapter",
    "load_e1_e3_head_checkpoint",
    "load_policy_checkpoint_into_model",
    "module_grad_norm",
    "module_state_sha256",
    "save_policy_checkpoint",
    "verify_artifact_identity",
]
