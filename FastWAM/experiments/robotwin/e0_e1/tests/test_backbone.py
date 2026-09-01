from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from experiments.robotwin.e0_e1.backbone import (
    CheckpointAudit,
    FrozenBackboneError,
    FrozenFastWAMExtractor,
    assert_kv_cache_equivalent,
    run_video_prefill_with_captures,
    strict_load_release_checkpoint,
)


class _TinyBlock(nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.scale = nn.Parameter(torch.tensor(float(layer_index + 1)))


class _TinyExpert(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(i) for i in range(num_layers)])


class _TinyMoT(nn.Module):
    """Small algebraic stand-in exposing the exact MoT prefill interface."""

    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": _TinyExpert(num_layers)})
        self.num_layers = num_layers

    def _build_expert_attention_io(
        self,
        *,
        expert: nn.Module,
        block: _TinyBlock,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del expert, freqs, t_mod
        scale = block.scale
        q = x + scale
        k = x + scale * 2
        v = x + scale * 3
        zeros = torch.zeros_like(x)
        return q, k, v, x, zeros, zeros, zeros, zeros, False

    def _mixed_attention(
        self,
        *,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert bool(attention_mask.all())
        return (q_cat + k_cat + v_cat) / 3

    def _apply_post_with_optional_checkpoint(
        self,
        *,
        block: nn.Module,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        del block, gate_msa, shift_mlp, scale_mlp, gate_mlp
        del use_gradient_checkpointing, context_payload
        return residual_x + mixed_slice

    def prefill_video_cache(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict[str, torch.Tensor] | None,
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        expert = self.mixtures["video"]
        x = video_tokens
        cache: list[dict[str, torch.Tensor]] = []
        for block in expert.blocks:
            q, k, v, residual, g1, shift, scale, g2, checkpoint = (
                self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=video_freqs,
                    t_mod=video_t_mod,
                )
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual,
                gate_msa=g1,
                shift_mlp=shift,
                scale_mlp=scale,
                gate_mlp=g2,
                use_gradient_checkpointing=checkpoint,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            cache.append({"k": k, "v": v})
        return cache


def test_mirrored_prefill_captures_updated_tokens_and_matches_native() -> None:
    mot = _TinyMoT().eval()
    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    freqs = torch.empty(0)
    t_mod = torch.empty(0)
    mask = torch.ones((3, 3), dtype=torch.bool)

    captures, mirrored_cache = run_video_prefill_with_captures(
        mot,
        video_tokens=tokens,
        video_freqs=freqs,
        video_t_mod=t_mod,
        video_context_payload=None,
        video_attention_mask=mask,
        capture_layers=(1, 3, 4),
    )
    native_cache = mot.prefill_video_cache(
        video_tokens=tokens,
        video_freqs=freqs,
        video_t_mod=t_mod,
        video_context_payload=None,
        video_attention_mask=mask,
    )

    assert tuple(captures) == (1, 3, 4)
    assert all(value.shape == tokens.shape for value in captures.values())
    assert not torch.equal(captures[1], tokens)
    assert not torch.equal(captures[1], captures[3])
    assert_kv_cache_equivalent(mirrored_cache, native_cache)


class _TinyReleaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        self.proprio_encoder = nn.Linear(2, 5)


def _checkpoint_payload(model: _TinyReleaseModel) -> dict[str, Any]:
    return {
        "mot": {key: value.detach().clone() for key, value in model.mot.state_dict().items()},
        "proprio_encoder": {
            key: value.detach().clone()
            for key, value in model.proprio_encoder.state_dict().items()
        },
        "step": 17,
        "torch_dtype": "torch.float32",
    }


def test_release_checkpoint_load_is_exact_and_rejects_shape_drift(
    tmp_path: Path,
) -> None:
    source = _TinyReleaseModel()
    valid_path = tmp_path / "valid.pt"
    torch.save(_checkpoint_payload(source), valid_path)

    target = _TinyReleaseModel()
    audit = strict_load_release_checkpoint(target, valid_path)
    assert audit.step == 17
    assert audit.mot_tensor_count == len(source.mot.state_dict())
    assert audit.proprio_tensor_count == len(source.proprio_encoder.state_dict())
    for key, value in source.mot.state_dict().items():
        torch.testing.assert_close(value, target.mot.state_dict()[key])

    bad_payload = _checkpoint_payload(source)
    bad_payload["mot"]["0.weight"] = torch.zeros(1, 1)
    bad_path = tmp_path / "bad_shape.pt"
    torch.save(bad_payload, bad_path)
    with pytest.raises(FrozenBackboneError, match="tensor shapes differ"):
        strict_load_release_checkpoint(_TinyReleaseModel(), bad_path)


class _DummyProcessor:
    shape_meta = {"state": [{"key": "default"}]}


class _FreezableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = _TinyMoT(num_layers=4)
        self.other_weight = nn.Parameter(torch.ones(2))


def test_extractor_keeps_every_backbone_parameter_frozen() -> None:
    model = _FreezableModel().train()
    extractor = FrozenFastWAMExtractor(
        model,
        processor=_DummyProcessor(),
        checkpoint_audit=CheckpointAudit(
            path="synthetic",
            size_bytes=0,
            mtime_ns=0,
            step=None,
            declared_torch_dtype="torch.float32",
            mot_tensor_count=len(model.mot.state_dict()),
            proprio_tensor_count=0,
        ),
        capture_layers=(1, 3),
    )
    extractor.assert_frozen()
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())

    # External mutation cannot silently contaminate E0/E1: the next extraction
    # audit fails closed before doing any backbone work.
    model.other_weight.requires_grad_(True)
    with pytest.raises(FrozenBackboneError, match="frozen backbone invariant"):
        extractor.assert_frozen()


def test_deployment_pixel_operation_order_is_observably_dtype_sensitive() -> None:
    """Guard the uint8 -> bf16 -> scaling contract used by RoboTwin deploy."""

    pixels = torch.arange(256, dtype=torch.uint8)
    deployment = pixels.to(torch.bfloat16) * (2.0 / 255.0) - 1.0
    normalize_first = (pixels.float() * (2.0 / 255.0) - 1.0).to(torch.bfloat16)
    assert not torch.equal(deployment, normalize_first)
    assert int((deployment != normalize_first).sum().item()) > 0


class _AffineStateNormalizer:
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        value = batch["state"]["default"]
        return {"state": {"default": value * 2.0 + 1.0}}


class _ProprioProcessor:
    shape_meta = {
        "state": [{"key": "default", "raw_shape": 2, "shape": 2}]
    }
    proprio_output_dim = 2

    def __init__(self) -> None:
        self.normalizer = _AffineStateNormalizer()

    @staticmethod
    def action_state_transform(batch: dict[str, Any]) -> dict[str, Any]:
        return batch


class _TinyVAE(nn.Module):
    def encode(
        self,
        videos: list[torch.Tensor],
        *,
        device: torch.device,
        tiled: bool,
    ) -> torch.Tensor:
        assert not tiled
        means = torch.stack([video.mean() for video in videos]).to(device=device)
        return means.reshape(-1, 1, 1, 1, 1).expand(-1, 4, 1, 2, 2).clone()


class _TinyVideoExpert(nn.Module):
    fuse_vae_embedding_in_latents = True

    def pre_dit(
        self,
        *,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: None,
        fuse_vae_embedding_in_latents: bool,
    ) -> dict[str, Any]:
        del timestep, action
        assert fuse_vae_embedding_in_latents
        batch_size = x.shape[0]
        sequence_length = 4
        image_value = x.mean(dim=(1, 2, 3, 4)).reshape(batch_size, 1, 1)
        tokens = image_value.expand(-1, sequence_length, 4).clone()
        tokens = tokens + context[:, -1, :4].unsqueeze(1)
        return {
            "tokens": tokens,
            "freqs": torch.empty(0, device=x.device),
            "t_mod": torch.empty(0, device=x.device),
            "context": context,
            "context_mask": context_mask.unsqueeze(1).expand(
                -1, sequence_length, -1
            ),
            "meta": {"tokens_per_frame": sequence_length},
        }

    @staticmethod
    def build_video_to_video_mask(
        *,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert video_seq_len == video_tokens_per_frame
        return torch.ones(
            (video_seq_len, video_seq_len), dtype=torch.bool, device=device
        )


class _ProprioExtractionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = _TinyMoT(num_layers=4)
        self.vae = _TinyVAE()
        self.video_expert = _TinyVideoExpert()
        self.proprio_dim = 2
        self.proprio_encoder = nn.Linear(2, 5)
        with torch.no_grad():
            self.proprio_encoder.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [-1.0, 1.0],
                        [0.5, -0.5],
                    ]
                )
            )
            self.proprio_encoder.bias.copy_(
                torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
            )
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32

    def _append_proprio_to_context(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.proprio_encoder(proprio.to(context.dtype)).unsqueeze(1)
        token_mask = torch.ones(
            (context_mask.shape[0], 1),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat((context, token), dim=1),
            torch.cat((context_mask, token_mask), dim=1),
        )


def _proprio_extractor() -> FrozenFastWAMExtractor:
    model = _ProprioExtractionModel()
    return FrozenFastWAMExtractor(
        model,
        processor=_ProprioProcessor(),
        checkpoint_audit=CheckpointAudit(
            path="synthetic",
            size_bytes=0,
            mtime_ns=0,
            step=None,
            declared_torch_dtype="torch.float32",
            mot_tensor_count=len(model.mot.state_dict()),
            proprio_tensor_count=len(model.proprio_encoder.state_dict()),
        ),
        capture_layers=(1, 3),
    )


def test_constant_zero_normalized_proprio_is_state_independent_and_structural() -> None:
    extractor = _proprio_extractor()
    assert (
        inspect.signature(extractor.extract_current_observations)
        .parameters["proprio_mode"]
        .default
        == "observed"
    )

    raw_a = torch.tensor([1.0, 2.0])
    raw_b = torch.tensor([-3.0, 4.0])
    observed_a, effective_a, observed_provenance = (
        extractor._prepare_proprio_condition(raw_a, batch_size=1)  # noqa: SLF001
    )
    observed_b, zero_b, zero_b_provenance = extractor._prepare_proprio_condition(  # noqa: SLF001
        raw_b,
        batch_size=1,
        proprio_mode="constant_zero_normalized",
    )
    _, zero_a, zero_a_provenance = extractor._prepare_proprio_condition(  # noqa: SLF001
        raw_a,
        batch_size=1,
        proprio_mode="constant_zero_normalized",
    )

    assert observed_provenance["mode"] == "observed"
    assert observed_provenance["all_zero"] is False
    assert observed_provenance["observed_normalized_sha256"] == (
        observed_provenance["effective_normalized_sha256"]
    )
    assert torch.equal(observed_a, effective_a)
    assert not torch.equal(observed_a, observed_b)
    assert torch.equal(zero_a, zero_b)
    assert torch.count_nonzero(zero_a).item() == 0
    assert zero_a.shape == observed_a.shape == observed_b.shape
    assert zero_a.dtype == observed_a.dtype == observed_b.dtype
    assert zero_a_provenance["observed_normalized_sha256"] != (
        zero_b_provenance["observed_normalized_sha256"]
    )
    assert zero_a_provenance["effective_normalized_sha256"] == (
        zero_b_provenance["effective_normalized_sha256"]
    )
    assert zero_a_provenance["all_zero"] is True

    text_context = torch.zeros((1, 3, 5))
    text_mask = torch.ones((1, 3), dtype=torch.bool)
    conditioned, conditioned_mask, context_provenance = extractor._prepare_context(  # noqa: SLF001
        batch_size=1,
        normalized_proprio=zero_a,
        instruction=None,
        context=text_context,
        context_mask=text_mask,
    )
    assert conditioned.shape == (1, 4, 5)
    assert conditioned_mask.shape == (1, 4)
    assert context_provenance["appended_proprio_tokens"] == 1
    torch.testing.assert_close(conditioned[:, :-1], text_context)
    torch.testing.assert_close(
        conditioned[:, -1], extractor.model.proprio_encoder.bias.unsqueeze(0)
    )
    assert bool(conditioned_mask.all())


def test_extract_constant_zero_normalized_changes_only_effective_condition() -> None:
    extractor = _proprio_extractor()
    images = torch.zeros((1, 3, 384, 320), dtype=torch.uint8)
    context = torch.zeros((1, 3, 5))
    context_mask = torch.ones((1, 3), dtype=torch.bool)
    raw_a = torch.tensor([1.0, 2.0])
    raw_b = torch.tensor([-3.0, 4.0])

    observed_a = extractor.extract_current_observations(
        images, raw_a, context=context, context_mask=context_mask
    )
    observed_b = extractor.extract_current_observations(
        images, raw_b, context=context, context_mask=context_mask
    )
    zero_a = extractor.extract_current_observations(
        images,
        raw_a,
        context=context,
        context_mask=context_mask,
        proprio_mode="constant_zero_normalized",
    )
    zero_b = extractor.extract_current_observations(
        images,
        raw_b,
        context=context,
        context_mask=context_mask,
        proprio_mode="constant_zero_normalized",
    )

    assert observed_a.provenance["proprio_mode"] == "observed"
    assert not torch.equal(observed_a.tokens_by_layer[1], observed_b.tokens_by_layer[1])
    for layer in (1, 3):
        assert torch.equal(zero_a.tokens_by_layer[layer], zero_b.tokens_by_layer[layer])
        assert zero_a.tokens_by_layer[layer].shape == observed_a.tokens_by_layer[layer].shape

    condition_a = zero_a.provenance["condition"]["proprio"]
    condition_b = zero_b.provenance["condition"]["proprio"]
    assert condition_a["mode"] == condition_b["mode"] == "constant_zero_normalized"
    assert condition_a["all_zero"] is condition_b["all_zero"] is True
    assert condition_a["observed_normalized_sha256"] != condition_b["observed_normalized_sha256"]
    assert condition_a["effective_normalized_sha256"] == condition_b["effective_normalized_sha256"]
    assert zero_a.provenance["normalized_proprio_sha256"] == condition_a[
        "effective_normalized_sha256"
    ]
