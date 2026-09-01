"""Policy v2 official, C2 paired-action, and C3 contrastive objectives."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

from .data import flatten_paired_action_batch
from .model import PolicyContentConditioner, PolicyContentRuntime
from .protocol import POLICY_R3_ROLE, POLICY_VARIANTS, POLICY_VIEW_COUNT


def _labels_to_equality_mask(
    labels: torch.Tensor | Sequence[Hashable],
    *,
    count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        if labels.ndim != 1 or labels.shape[0] != count:
            raise ValueError(f"{name} must have shape [{count}]")
        encoded = labels.to(device=device)
        return encoded[:, None] == encoded[None, :]
    values = list(labels)
    if len(values) != count:
        raise ValueError(f"{name} must contain {count} labels")
    ids: dict[Hashable, int] = {}
    encoded_values: list[int] = []
    for value in values:
        try:
            encoded_values.append(ids.setdefault(value, len(ids)))
        except TypeError as exc:
            raise TypeError(f"every {name} value must be hashable") from exc
    encoded = torch.tensor(encoded_values, dtype=torch.long, device=device)
    return encoded[:, None] == encoded[None, :]


def multi_positive_supcon_loss(
    embeddings: torch.Tensor,
    physical_state_ids: torch.Tensor | Sequence[Hashable],
    task_ids: torch.Tensor | Sequence[Hashable],
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Same-task SupCon with all other scenes of one state as positives."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must be [N,D] with N>=2")
    if not torch.is_floating_point(embeddings) or not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite floating point")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    count = int(embeddings.shape[0])
    task_equal = _labels_to_equality_mask(
        task_ids, count=count, device=embeddings.device, name="task_ids"
    )
    state_equal = _labels_to_equality_mask(
        physical_state_ids,
        count=count,
        device=embeddings.device,
        name="physical_state_ids",
    )
    not_self = ~torch.eye(count, dtype=torch.bool, device=embeddings.device)
    positive = task_equal & state_equal & not_self
    negative = task_equal & ~state_equal
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    if bool((positive_count == 0).any()):
        raise ValueError("every contrastive anchor requires a same-state positive")
    if bool((negative_count == 0).any()):
        raise ValueError("every contrastive anchor requires a same-task different-state negative")
    normalized = F.normalize(embeddings, p=2.0, dim=-1)
    logits = (normalized @ normalized.transpose(0, 1)) / float(temperature)
    denominator = positive | negative
    logits = logits.masked_fill(~denominator, -torch.inf)
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive, 0.0).sum(dim=1)
    loss = -(positive_logits / positive_count - log_denominator).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("multi-positive contrastive loss is non-finite")
    return loss


def assert_kv_cache_equivalent(
    captured_cache: Sequence[Mapping[str, torch.Tensor]],
    native_cache: Sequence[Mapping[str, torch.Tensor]],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Policy-local bit-equivalence check for the mirrored video prefill."""

    if len(captured_cache) != len(native_cache):
        raise AssertionError("captured/native K/V cache lengths differ")
    for layer_index, (captured, native) in enumerate(
        zip(captured_cache, native_cache, strict=True), start=1
    ):
        if set(captured) != {"k", "v"} or set(native) != {"k", "v"}:
            raise AssertionError(f"layer {layer_index} K/V keys are not canonical")
        for key in ("k", "v"):
            try:
                torch.testing.assert_close(captured[key], native[key], rtol=rtol, atol=atol)
            except AssertionError as exc:
                raise AssertionError(
                    f"mirrored prefill differs at layer {layer_index}/{key}: {exc}"
                ) from exc


def _require_tensor(sample: Mapping[str, Any], key: str) -> torch.Tensor:
    value = sample.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"native action batch {key!r} must be a torch.Tensor")
    return value


def prepare_official_action_inputs(
    model,
    sample: Mapping[str, Any],
) -> dict[str, torch.Tensor | None]:
    """Prepare only the current observation and native action targets.

    The official loader still supplies its canonical 9-frame/32-action window.
    Action tokens natively attend only to the first video frame, so this avoids
    constructing a video-prediction objective while preserving the deployed
    current-observation action path.
    """

    video = _require_tensor(sample, "video")
    action = _require_tensor(sample, "action")
    context = _require_tensor(sample, "context")
    context_mask = _require_tensor(sample, "context_mask")
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"official video must be [B,3,T,H,W], got {tuple(video.shape)}")
    if video.shape[2] <= 1 or video.shape[2] % 4 != 1:
        raise ValueError("official video must retain native T>1 and T%4==1 convention")
    if action.ndim != 3 or action.shape[0] != video.shape[0]:
        raise ValueError("official action must be aligned [B,Ta,D]")
    if action.shape[1] % (video.shape[2] - 1) != 0:
        raise ValueError("official action horizon/video transition convention changed")
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError("official context/context_mask must be [B,L,D]/[B,L]")

    device = model.device
    dtype = model.torch_dtype
    # VAE and all context producers are frozen.  Detaching here is intentional;
    # gradients start at the trainable content head and action adapter.
    with torch.no_grad():
        current_frame = video[:, :, 0:1].to(device=device, dtype=dtype)
        first_frame_latents = model._encode_video_latents(current_frame, tiled=False)
        context = context.to(device=device, dtype=dtype, non_blocking=True)
        context_mask = context_mask.to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        if model.proprio_encoder is not None:
            proprio = _require_tensor(sample, "proprio")
            if proprio.ndim != 3 or proprio.shape[0] != video.shape[0]:
                raise ValueError("official proprio must be [B,T,D]")
            context, context_mask = model._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio[:, 0, :].to(device=device, dtype=dtype),
            )
    action = action.to(device=device, dtype=dtype, non_blocking=True)
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is not None:
        if not isinstance(action_is_pad, torch.Tensor):
            raise TypeError("action_is_pad must be a tensor or None")
        if tuple(action_is_pad.shape) != tuple(action.shape[:2]):
            raise ValueError("action_is_pad shape differs from action [B,T]")
        action_is_pad = action_is_pad.to(device=device, dtype=torch.bool)
    return {
        "first_frame_latents": first_frame_latents.detach(),
        "context": context.detach(),
        "context_mask": context_mask.detach(),
        "action": action,
        "action_is_pad": action_is_pad,
    }


def _build_video_prefill_inputs(
    model,
    *,
    first_frame_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_seq_len: int,
) -> tuple[dict[str, Any], torch.Tensor, int]:
    """Build the deployed current-observation prefill inputs without gradients."""

    batch_size = int(first_frame_latents.shape[0])
    with torch.no_grad():
        timestep_video = torch.zeros(
            (batch_size,),
            dtype=first_frame_latents.dtype,
            device=model.device,
        )
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                model.video_expert.fuse_vae_embedding_in_latents
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(action_seq_len),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        prefill_kwargs = {
            "video_tokens": video_pre["tokens"],
            "video_freqs": video_pre["freqs"],
            "video_t_mod": video_pre["t_mod"],
            "video_context_payload": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "video_attention_mask": attention_mask[:video_seq_len, :video_seq_len],
        }
    return prefill_kwargs, attention_mask, video_seq_len


def _predict_action_with_cache(
    model,
    *,
    action_tokens: torch.Tensor,
    timestep_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_kv_cache: list[dict[str, torch.Tensor]],
    attention_mask: torch.Tensor,
    video_seq_len: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Execute the native ActionDiT/MoT tail using an already-built video cache."""

    action_pre = model.action_expert.pre_dit(
        action_tokens=action_tokens,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    pred_tokens = model.mot.forward_action_with_video_cache(
        action_tokens=action_pre["tokens"],
        action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"],
        action_context_payload={
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        },
        video_kv_cache=video_kv_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )
    return model.action_expert.post_dit(pred_tokens, action_pre), action_pre


def zero_init_policy_identity_audit(
    model,
    runtime: PolicyContentRuntime,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the native and installed full action paths at exact ``gate=0``.

    This checks both changes introduced by the experiment: the mirrored prefill
    used to expose Layer-16 and the action-encoder hook used to inject Zc.
    Fixed action inputs/timesteps make the comparison deterministic.
    """

    conditioner = runtime.conditioner
    if float(conditioner.adapter.gate.detach().float().item()) != 0.0:
        raise ValueError("zero-init identity audit requires gate to be exactly zero")
    inputs = prepare_official_action_inputs(model, sample)
    first_frame_latents = inputs["first_frame_latents"]
    context = inputs["context"]
    context_mask = inputs["context_mask"]
    action = inputs["action"]
    assert isinstance(first_frame_latents, torch.Tensor)
    assert isinstance(context, torch.Tensor)
    assert isinstance(context_mask, torch.Tensor)
    assert isinstance(action, torch.Tensor)
    prefill_kwargs, attention_mask, video_seq_len = _build_video_prefill_inputs(
        model,
        first_frame_latents=first_frame_latents,
        context=context,
        context_mask=context_mask,
        action_seq_len=int(action.shape[1]),
    )

    prior_enabled = conditioner.enabled
    if conditioner._active_content_tokens is not None:  # noqa: SLF001
        raise RuntimeError("identity audit requires no pre-existing active content")
    try:
        with torch.no_grad():
            native_cache = runtime._native_prefill(**prefill_kwargs)  # noqa: SLF001
            layer_tokens, captured_cache = runtime.capture_video_prefill(**prefill_kwargs)
            assert_kv_cache_equivalent(captured_cache, native_cache, rtol=0.0, atol=0.0)
            if tuple(layer_tokens.shape[1:]) != (120, 3072):
                raise AssertionError(
                    "deployed Layer-16 tokens must be [B,120,3072], got "
                    f"{tuple(layer_tokens.shape)}"
                )
            fixed_action_input = torch.zeros_like(action)
            fixed_timestep = torch.full(
                (action.shape[0],),
                0.5,
                device=action.device,
                dtype=action.dtype,
            )
            conditioner.enabled = False
            native_pred, native_pre = _predict_action_with_cache(
                model,
                action_tokens=fixed_action_input,
                timestep_action=fixed_timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=native_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            conditioner.enabled = True
            content_tokens = conditioner.set_visual_tokens(layer_tokens)
            adapter_pred, adapter_pre = _predict_action_with_cache(
                model,
                action_tokens=fixed_action_input,
                timestep_action=fixed_timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=captured_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
    finally:
        conditioner.clear_active_content()
        conditioner.enabled = prior_enabled

    difference = (adapter_pred.float() - native_pred.float()).abs()
    max_abs = float(difference.max().item())
    denominator = native_pred.float().abs().clamp_min(torch.finfo(torch.float32).eps)
    max_rel = float((difference / denominator).max().item())
    bit_exact = bool(torch.equal(adapter_pred, native_pred))
    if not bit_exact or max_abs != 0.0 or max_rel != 0.0:
        raise AssertionError(
            "zero-init adapter changed full policy action output: "
            f"max_abs={max_abs}, max_rel={max_rel}"
        )
    return {
        "status": "PASS",
        "native_prefill_kv_bit_exact": True,
        "action_output_bit_exact": bit_exact,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "gate_raw": float(conditioner.adapter.gate.detach().float().item()),
        "gate_tanh": conditioner.adapter.gate_value,
        "layer16_shape": list(layer_tokens.shape),
        "content_token_shape": list(content_tokens.shape),
        "native_action_token_shape": list(native_pre["tokens"].shape),
        "adapter_action_token_shape": list(adapter_pre["tokens"].shape),
        "action_output_shape": list(adapter_pred.shape),
        "finite": bool(torch.isfinite(adapter_pred).all().item()),
    }


def tensor_distribution_summary(tokens: torch.Tensor) -> dict[str, float | int | list[int]]:
    """Return mergeable scalar moments without copying a full token tensor."""

    if tokens.ndim != 3 or not torch.is_floating_point(tokens):
        raise ValueError("distribution tokens must be floating [B,S,D]")
    values = tokens.detach().float()
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError("distribution tokens contain NaN or infinity")
    token_norm = values.norm(p=2, dim=-1)
    return {
        "shape": list(values.shape),
        "element_count": int(values.numel()),
        "sum": float(values.double().sum().item()),
        "sum_squares": float(values.double().square().sum().item()),
        "token_count": int(token_norm.numel()),
        "token_l2_sum": float(token_norm.double().sum().item()),
        "token_l2_sum_squares": float(token_norm.double().square().sum().item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


def _audit_effective_action_weight(
    action_weight: torch.Tensor,
    valid_steps_per_sample: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, bool, str]:
    """Validate native action supervision and classify scheduler endpoints."""

    if valid_steps_per_sample.reshape(-1).numel() != batch_size:
        raise RuntimeError("valid action-step counts do not align with the batch")
    if bool((valid_steps_per_sample <= 0).any().item()):
        raise RuntimeError(
            "official action batch contains a sample with no valid action target"
        )
    action_weight_per_sample = action_weight.reshape(-1)
    if action_weight_per_sample.numel() != batch_size:
        raise RuntimeError("action training weights do not align with the batch")
    if bool((action_weight_per_sample < 0).any().item()):
        raise FloatingPointError("action training weight is negative")
    signal_positive = bool((action_weight_per_sample > 0).any().item())
    return (
        action_weight_per_sample,
        signal_positive,
        "none" if signal_positive else "scheduler_zero_weight",
    )


def official_action_loss(
    model,
    runtime: PolicyContentRuntime | None,
    sample: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Native flow-matching action MSE, with no video loss."""

    inputs = prepare_official_action_inputs(model, sample)
    first_frame_latents = inputs["first_frame_latents"]
    context = inputs["context"]
    context_mask = inputs["context_mask"]
    action = inputs["action"]
    action_is_pad = inputs["action_is_pad"]
    assert isinstance(first_frame_latents, torch.Tensor)
    assert isinstance(context, torch.Tensor)
    assert isinstance(context_mask, torch.Tensor)
    assert isinstance(action, torch.Tensor)

    batch_size = action.shape[0]
    noise_action = torch.randn_like(action)
    timestep_action = model.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=model.device,
        dtype=action.dtype,
    )
    noisy_action = model.train_action_scheduler.add_noise(
        action, noise_action, timestep_action
    )
    target_action = model.train_action_scheduler.training_target(
        action, noise_action, timestep_action
    )

    prefill_kwargs, attention_mask, video_seq_len = _build_video_prefill_inputs(
        model,
        first_frame_latents=first_frame_latents,
        context=context,
        context_mask=context_mask,
        action_seq_len=int(noisy_action.shape[1]),
    )
    with torch.no_grad():
        if runtime is None:
            layer_tokens = None
            video_kv_cache = model.mot.prefill_video_cache(**prefill_kwargs)
        else:
            layer_tokens, video_kv_cache = runtime.capture_video_prefill(
                **prefill_kwargs
            )

    # Head stays outside no_grad.  Its Layer-16 input is deliberately detached
    # from the permanently frozen video branch.
    content_tokens = None
    if runtime is not None:
        assert isinstance(layer_tokens, torch.Tensor)
        content_tokens = runtime.conditioner.set_visual_tokens(layer_tokens.detach())
        if content_tokens.requires_grad:
            runtime.conditioner.arm_action_content_gradient_audit(content_tokens)
    try:
        pred_action, action_pre = _predict_action_with_cache(
            model,
            action_tokens=noisy_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
    finally:
        if runtime is not None:
            runtime.conditioner.clear_active_content()

    action_loss_token = F.mse_loss(
        pred_action.float(), target_action.float(), reduction="none"
    ).mean(dim=2)
    if action_is_pad is not None:
        assert isinstance(action_is_pad, torch.Tensor)
        valid = (~action_is_pad).to(
            device=action_loss_token.device, dtype=action_loss_token.dtype
        )
        valid_steps_per_sample = valid.sum(dim=1)
        action_loss_per_sample = (
            (action_loss_token * valid).sum(dim=1)
            / valid_steps_per_sample.clamp(min=1.0)
        )
    else:
        valid_steps_per_sample = torch.full(
            (batch_size,),
            int(action_loss_token.shape[1]),
            device=action_loss_token.device,
            dtype=action_loss_token.dtype,
        )
        action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = model.train_action_scheduler.training_weight(
        timestep_action
    ).to(device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype)
    (
        action_weight_per_sample,
        action_signal_positive,
        zero_action_signal_reason,
    ) = _audit_effective_action_weight(
        action_weight,
        valid_steps_per_sample,
        batch_size=batch_size,
    )
    effective_action_weight = action_weight_per_sample
    loss = (action_loss_per_sample * action_weight_per_sample).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("official action loss is non-finite")
    return loss, {
        "video_token_shape": (
            list(layer_tokens.shape) if isinstance(layer_tokens, torch.Tensor) else []
        ),
        "content_token_shape": (
            list(content_tokens.shape) if isinstance(content_tokens, torch.Tensor) else []
        ),
        "action_token_shape": list(action_pre["tokens"].shape),
        "loss_action": float(loss.detach().item()),
        # Wan's shifted flow-matching weight is exactly zero at the endpoint.
        # In bf16, a sampled timestep can quantize to that endpoint.  Keep the
        # distinction explicit so the training audit does not confuse a valid
        # zero-supervision batch with a disconnected adapter path.
        "action_timestep_min": float(timestep_action.detach().float().min().item()),
        "action_timestep_max": float(timestep_action.detach().float().max().item()),
        "action_weight_min": float(action_weight_per_sample.detach().min().item()),
        "action_weight_max": float(action_weight_per_sample.detach().max().item()),
        "action_effective_weight_sum": float(
            effective_action_weight.detach().float().sum().item()
        ),
        "action_supervision_signal_positive": action_signal_positive,
        "zero_action_signal_reason": zero_action_signal_reason,
        "action_valid_steps_total": int(
            valid_steps_per_sample.detach().sum().item()
        ),
        "action_unweighted_mse_mean": float(
            action_loss_per_sample.detach().float().mean().item()
        ),
        "official_layer16_distribution": (
            tensor_distribution_summary(layer_tokens)
            if isinstance(layer_tokens, torch.Tensor)
            else None
        ),
    }


def _flatten_paired_batch(
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, list[str], list[str]]:
    tokens = batch.get("tokens")
    if tokens is None:
        tokens = batch.get("visual_tokens")
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("paired batch must contain tensor 'tokens'")
    if tokens.ndim == 4:
        groups, views, sequence_length, dim = tokens.shape
        tokens = tokens.reshape(groups * views, sequence_length, dim)
    elif tokens.ndim == 3:
        groups = None
        views = None
    else:
        raise ValueError("paired tokens must be [G,4,S,D] or [N,S,D]")

    physical = batch.get("physical_state_ids", batch.get("physical_state_id"))
    tasks = batch.get("task_ids", batch.get("task"))
    if physical is None or tasks is None:
        raise ValueError("paired batch lacks physical/task labels")
    physical_values = [str(value) for value in physical]
    task_values = [str(value) for value in tasks]
    if groups is not None:
        if len(physical_values) == groups:
            physical_values = [value for value in physical_values for _ in range(views)]
        if len(task_values) == groups:
            task_values = [value for value in task_values for _ in range(views)]
    if len(physical_values) != tokens.shape[0] or len(task_values) != tokens.shape[0]:
        raise ValueError("paired labels do not align with flattened token views")
    return tokens, physical_values, task_values


def paired_contrastive_loss(
    conditioner: PolicyContentConditioner,
    batch: Mapping[str, Any],
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use C/R1/R2/R3 frozen tokens; every anchor has three positives."""

    if "action" in batch or "state_window" in batch:
        raise ValueError("paired contrastive stream must not contain action supervision")
    if batch.get("supervision_mode", "contrastive") != "contrastive":
        raise ValueError("paired contrastive batch has the wrong supervision_mode")
    if batch.get("r3_role") != POLICY_R3_ROLE:
        raise ValueError("R3 must be marked as a training positive")
    source_tokens = batch.get("tokens", batch.get("visual_tokens"))
    if not isinstance(source_tokens, torch.Tensor):
        raise TypeError("paired batch must contain tensor 'tokens'")
    if source_tokens.ndim == 4:
        variant_names = tuple(str(value) for value in batch.get("variant_names", ()))
        if variant_names != POLICY_VARIANTS or source_tokens.shape[1] != POLICY_VIEW_COUNT:
            raise ValueError(
                "paired contrastive stream must preserve exact ordered C/R1/R2/R3 scenes"
            )
        clean_tokens = source_tokens[:, 0]
    else:
        clean_tokens = None
    if (
        conditioner.content_layer == 16
        and conditioner.head.backbone_dim == 3072
        and tuple(source_tokens.shape[-2:]) != (120, 3072)
    ):
        raise ValueError(
            "Layer-16 paired policy tokens must end in [120,3072], got "
            f"{tuple(source_tokens.shape)}"
        )
    tokens, physical_ids, task_ids = _flatten_paired_batch(batch)
    device = next(conditioner.parameters()).device
    dtype = next(conditioner.parameters()).dtype
    embeddings = conditioner.contrastive(tokens.to(device=device, dtype=dtype))
    loss = multi_positive_supcon_loss(
        embeddings,
        physical_ids,
        task_ids,
        temperature=float(temperature),
    )
    with torch.no_grad():
        similarity = embeddings.float() @ embeddings.float().transpose(0, 1)
        count = len(physical_ids)
        eye = torch.eye(count, dtype=torch.bool, device=similarity.device)
        task_equal = torch.tensor(
            [[a == b for b in task_ids] for a in task_ids],
            dtype=torch.bool,
            device=similarity.device,
        )
        state_equal = torch.tensor(
            [[a == b for b in physical_ids] for a in physical_ids],
            dtype=torch.bool,
            device=similarity.device,
        )
        positive = task_equal & state_equal & ~eye
        negative = task_equal & ~state_equal
        if not bool(positive.any().item()) or not bool(negative.any().item()):
            raise ValueError("paired batch lacks positive or same-task negative pairs")
        positive_similarity = float(similarity[positive].mean().item())
        negative_similarity = float(similarity[negative].mean().item())
    return loss, {
        "loss_contrastive": float(loss.detach().item()),
        "positive_similarity": positive_similarity,
        "negative_similarity": negative_similarity,
        "positives_per_anchor": POLICY_VIEW_COUNT - 1,
        "r3_training_positive": True,
        "paired_clean_layer16_distribution": (
            tensor_distribution_summary(clean_tokens)
            if isinstance(clean_tokens, torch.Tensor)
            else None
        ),
    }


def paired_action_loss(
    model,
    runtime: PolicyContentRuntime,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """C2 native action loss after strict four-scene contract validation.

    The batch is produced by :class:`NativePairedActionDataset`; flattening the
    physical-state and scene axes preserves the exact official FastWAM action
    path.  There is deliberately no video-prediction loss and no interpolation.
    """

    flat = flatten_paired_action_batch(batch)
    loss, diagnostics = official_action_loss(model, runtime, flat)
    paired_distribution = diagnostics.pop("official_layer16_distribution")
    return loss, {
        "loss_paired_action": float(loss.detach().item()),
        "paired_layer16_distribution": paired_distribution,
        "paired_action_video_token_shape": diagnostics["video_token_shape"],
        "paired_action_content_token_shape": diagnostics["content_token_shape"],
        "paired_action_token_shape": diagnostics["action_token_shape"],
        "r3_training_positive": True,
    }


__all__ = [
    "official_action_loss",
    "paired_action_loss",
    "paired_contrastive_loss",
    "prepare_official_action_inputs",
    "tensor_distribution_summary",
    "zero_init_policy_identity_audit",
]
