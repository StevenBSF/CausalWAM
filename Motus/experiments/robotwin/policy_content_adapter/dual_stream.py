"""M1/M3 dual-stream objective with an action-only official branch."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .losses import multi_positive_supcon_loss
from .model import MotusPolicyContentConditioner
from .observation_content import extract_observation_visual_tokens
from .protocol import DEFAULT_TEMPERATURE, PAIRED_VIEW_COUNT, validate_control


class DualStreamError(RuntimeError):
    pass


@dataclass
class DualStreamLoss:
    total: torch.Tensor
    action: torch.Tensor
    contrastive: torch.Tensor
    official_visual_tokens: torch.Tensor
    official_content_tokens: torch.Tensor
    paired_embeddings: torch.Tensor

    def scalar_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().float().item()),
            "loss_action": float(self.action.detach().float().item()),
            "loss_contrastive": float(
                self.contrastive.detach().float().item()
            ),
        }


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"official batch {key!r} must be a tensor")
    return value


def _flatten_group_labels(
    values: Sequence[Any], *, groups: int, name: str
) -> list[Any]:
    items = list(values)
    if len(items) == groups:
        return [value for value in items for _ in range(PAIRED_VIEW_COUNT)]
    if len(items) == groups * PAIRED_VIEW_COUNT:
        return items
    raise ValueError(
        f"{name} must contain G or G*4 labels, got {len(items)} for G={groups}"
    )


def compute_dual_stream_loss(
    *,
    motus_model: Any,
    conditioner: MotusPolicyContentConditioner,
    official_batch: Mapping[str, Any],
    paired_visual_tokens: torch.Tensor,
    paired_physical_state_ids: Sequence[Any],
    paired_task_ids: Sequence[Any],
    control: str,
    lambda_contrastive: float,
    temperature: float = DEFAULT_TEMPERATURE,
    observation_extractor: Callable[..., torch.Tensor] = extract_observation_visual_tokens,
) -> DualStreamLoss:
    """Compute one matched M1/M3 step.

    The official stream supplies only Motus action flow-matching loss.  The
    paired stream supplies only same-state four-view contrastive supervision.
    M1 executes the same paired forward but multiplies it by exactly zero.
    """

    validate_control(
        control=control, lambda_contrastive=float(lambda_contrastive)
    )
    first_frame = _require_tensor(official_batch, "first_frame")
    video_frames = _require_tensor(official_batch, "video_frames")
    actions = _require_tensor(official_batch, "action_sequence")
    state = _require_tensor(official_batch, "initial_state")
    language_embeddings = official_batch.get("language_embedding")
    if not isinstance(language_embeddings, (torch.Tensor, list, tuple)):
        raise TypeError("official language_embedding is missing")

    official_visual = observation_extractor(
        motus_model,
        first_frame=first_frame,
        language_embeddings=language_embeddings,
        capture_layer=conditioner.capture_layer,
    )
    official_content = conditioner.content_tokens(official_visual)
    action_dtype = next(motus_model.action_expert.parameters()).dtype
    if not action_dtype.is_floating_point:
        raise TypeError("Action Expert parameters must use a floating dtype")
    # The author release is loaded in BF16 while the LeRobot contract keeps
    # state/action values in FP32 on disk.  Cast only at the model boundary so
    # the stored data contract remains lossless and linear inputs match the
    # Action Expert weights exactly.
    state = state.to(dtype=action_dtype)
    actions = actions.to(dtype=action_dtype)
    loss_dict = motus_model.training_step(
        first_frame=first_frame,
        video_frames=video_frames,
        state=state,
        actions=actions,
        language_embeddings=language_embeddings,
        vlm_inputs=official_batch.get("vlm_inputs"),
        policy_content_tokens=official_content,
        compute_video_loss=False,
        return_dict=True,
    )
    action_loss = loss_dict.get("action_loss")
    if not isinstance(action_loss, torch.Tensor) or action_loss.ndim != 0:
        raise DualStreamError("Motus did not return a scalar action loss")
    if not bool(torch.isfinite(action_loss).item()):
        raise FloatingPointError("official action loss is non-finite")
    video_loss = loss_dict.get("video_loss")
    if not isinstance(video_loss, torch.Tensor) or video_loss.detach().item() != 0.0:
        raise DualStreamError("policy continuation unexpectedly enabled video loss")

    if paired_visual_tokens.ndim != 4:
        raise ValueError("paired_visual_tokens must be [G,4,L,D]")
    groups, views, token_count, feature_dim = paired_visual_tokens.shape
    if groups <= 0 or views != PAIRED_VIEW_COUNT:
        raise ValueError("paired token batch must contain G non-empty four-view groups")
    flattened_tokens = paired_visual_tokens.reshape(
        groups * views, token_count, feature_dim
    )
    paired_embeddings = conditioner.contrastive_embedding(flattened_tokens)
    state_ids = _flatten_group_labels(
        paired_physical_state_ids,
        groups=groups,
        name="paired_physical_state_ids",
    )
    task_ids = _flatten_group_labels(
        paired_task_ids, groups=groups, name="paired_task_ids"
    )
    contrastive_loss = multi_positive_supcon_loss(
        paired_embeddings,
        state_ids,
        task_ids,
        temperature=float(temperature),
    )
    total = action_loss + contrastive_loss * float(lambda_contrastive)
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("dual-stream total loss is non-finite")
    return DualStreamLoss(
        total=total,
        action=action_loss,
        contrastive=contrastive_loss,
        official_visual_tokens=official_visual,
        official_content_tokens=official_content,
        paired_embeddings=paired_embeddings,
    )


def gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    observed = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        observed = True
        gradient = parameter.grad.detach().double()
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError("parameter gradient is non-finite")
        total += gradient.square().sum().cpu()
    return float(total.sqrt().item()) if observed else 0.0


def audit_dual_stream_gradients(
    *,
    motus_model: Any,
    conditioner: MotusPolicyContentConditioner,
    control: str,
    regime: str,
    step: int,
    gradient_snapshot: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fail closed on frozen leaks and missing expected gradient paths."""

    if gradient_snapshot is None:
        head_norm = gradient_norm(list(conditioner.head.parameters()))
        adapter_norm = gradient_norm(list(conditioner.adapter.parameters()))
        gate_tensor = conditioner.adapter.gate.grad
        if gate_tensor is None or not bool(torch.isfinite(gate_tensor).item()):
            raise DualStreamError("adapter gate did not receive a finite gradient")
        gate_grad = float(gate_tensor.detach().float().item())
        action_norm = gradient_norm(list(motus_model.action_expert.parameters()))
        backend = "pytorch_parameter_grad"
    else:
        required = {
            "content_head_grad_norm",
            "adapter_grad_norm",
            "adapter_gate_grad",
            "action_expert_grad_norm",
        }
        if set(gradient_snapshot) != required:
            raise DualStreamError("distributed gradient snapshot schema changed")
        values = {key: float(value) for key, value in gradient_snapshot.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise DualStreamError("distributed gradient snapshot is non-finite")
        head_norm = values["content_head_grad_norm"]
        adapter_norm = values["adapter_grad_norm"]
        gate_grad = values["adapter_gate_grad"]
        action_norm = values["action_expert_grad_norm"]
        backend = "deepspeed_zero_partition_pre_step_v1"
    if abs(gate_grad) == 0.0:
        raise DualStreamError("adapter gate gradient is zero")
    if adapter_norm <= 0:
        raise DualStreamError("GCA received no gradient")
    if control == "m3_ours" and head_norm <= 0:
        raise DualStreamError("M3 Content Head received no gradient")

    for name, module in (
        ("video", motus_model.video_model),
        ("VLM", motus_model.vlm_model),
        ("understanding", motus_model.und_expert),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise DualStreamError(f"frozen {name} module became trainable")
    video_norm = 0.0
    vlm_norm = 0.0
    und_norm = 0.0
    if video_norm != 0.0 or vlm_norm != 0.0 or und_norm != 0.0:
        raise DualStreamError("a frozen Motus backbone received gradients")
    if regime == "m_p1" and action_norm != 0.0:
        raise DualStreamError("M-P1 Action Expert must stay frozen")
    if regime == "m_p2" and action_norm <= 0.0:
        raise DualStreamError("M-P2 Action Expert received no gradient")
    return {
        "status": "PASS",
        "step": int(step),
        "control": control,
        "regime": regime,
        "content_head_grad_norm": head_norm,
        "adapter_grad_norm": adapter_norm,
        "adapter_gate_grad": gate_grad,
        "action_expert_grad_norm": action_norm,
        "video_grad_norm": video_norm,
        "vlm_grad_norm": vlm_norm,
        "understanding_grad_norm": und_norm,
        "backend": backend,
    }
