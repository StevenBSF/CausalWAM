"""CPU-only structural tests for the policy content adapter.

These tests deliberately use tiny stand-ins for FastWAM's large frozen modules.
They lock the experiment contract without loading the released 12 GB checkpoint
or requiring CUDA.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from experiments.robotwin.e0_e1.head import ContrastiveContentHead
from experiments.robotwin.policy_content_adapter.model import (
    GatedCrossAttentionAdapter,
    PolicyContentConditioner,
    PolicyContentHead,
    build_optimizer_param_groups,
    configure_trainable_modules,
)


HEAD_PARAMETER_COUNT = 2_070_144
ADAPTER_PARAMETER_COUNT = 2_887_681


def _parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _parameter_ids(parameters) -> set[int]:
    return {id(parameter) for parameter in parameters}


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


class _TinyActionExpert(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_encoder = nn.Linear(5, hidden_dim)
        self.blocks = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, 5)


class _TinyFastWAM(nn.Module):
    """Expose the component names used by the released FastWAM model."""

    def __init__(self) -> None:
        super().__init__()
        self.vae = nn.Linear(3, 4)
        self.video_expert = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.action_expert = _TinyActionExpert()
        self.text_encoder = nn.Linear(7, 8)
        self.proprio_encoder = nn.Linear(2, 8)


def _tiny_conditioner() -> PolicyContentConditioner:
    head = PolicyContentHead(
        backbone_dim=12,
        embed_dim=8,
        num_queries=2,
        num_heads=2,
    )
    adapter = GatedCrossAttentionAdapter(
        action_dim=16,
        content_dim=8,
        num_heads=4,
    )
    return PolicyContentConditioner(head=head, adapter=adapter)


def test_policy_content_head_shapes_normalization_and_legacy_weight_compatibility() -> None:
    torch.manual_seed(11)
    legacy = ContrastiveContentHead(
        backbone_dim=32,
        embed_dim=16,
        num_queries=4,
        num_heads=4,
    ).eval()
    head = PolicyContentHead(
        backbone_dim=32,
        embed_dim=16,
        num_queries=4,
        num_heads=4,
    ).eval()

    # Compatibility means an E1/E2/E3 payload["head"] loads strictly, without
    # renaming or dropping a single tensor.
    assert set(head.state_dict()) == set(legacy.state_dict())
    head.load_state_dict(legacy.state_dict(), strict=True)

    visual_tokens = torch.randn(3, 7, 32)
    content_tokens = head.forward_content_tokens(visual_tokens)
    embedding = head.forward_contrastive(visual_tokens)

    assert content_tokens.shape == (3, 4, 16)
    assert embedding.shape == (3, 16)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=-1),
        torch.ones(3),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        embedding,
        legacy(visual_tokens),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(head(visual_tokens), embedding, rtol=0.0, atol=0.0)


def test_default_parameter_counts_are_protocol_locked() -> None:
    head = PolicyContentHead()
    adapter = GatedCrossAttentionAdapter()

    assert _parameter_count(head) == HEAD_PARAMETER_COUNT
    assert _parameter_count(adapter) == ADAPTER_PARAMETER_COUNT
    assert _parameter_count(PolicyContentConditioner(head=head, adapter=adapter)) == (
        HEAD_PARAMETER_COUNT + ADAPTER_PARAMETER_COUNT
    )


def test_zero_initialized_gate_is_bit_exact_identity() -> None:
    torch.manual_seed(17)
    adapter = GatedCrossAttentionAdapter(
        action_dim=16,
        content_dim=8,
        num_heads=4,
    ).eval()
    action_tokens = torch.randn(2, 6, 16)
    content_tokens = torch.randn(2, 3, 8)

    assert adapter.gate.detach().item() == 0.0
    conditioned = adapter(action_tokens, content_tokens)

    # allclose is insufficient for the zero-init safety contract: the adapter
    # must leave the released policy bit-for-bit unchanged at initialization.
    assert torch.equal(conditioned, action_tokens)
    assert (conditioned - action_tokens).abs().max().item() == 0.0


@pytest.mark.parametrize(
    ("regime", "action_trainable", "expected_group_names"),
    [
        ("p_v1", False, ["content_head_and_adapter"]),
        ("p_v2", True, ["content_head_and_adapter", "action_dit"]),
    ],
)
def test_trainable_configuration_and_optimizer_groups_are_auditable(
    regime: str,
    action_trainable: bool,
    expected_group_names: list[str],
) -> None:
    model = _TinyFastWAM()
    conditioner = _tiny_conditioner()

    configure_trainable_modules(model, conditioner, regime=regime)

    assert all(parameter.requires_grad for parameter in conditioner.parameters())
    assert all(
        parameter.requires_grad is action_trainable
        for parameter in model.action_expert.parameters()
    )
    for frozen_module in (
        model.vae,
        model.video_expert,
        model.text_encoder,
        model.proprio_encoder,
    ):
        assert all(not parameter.requires_grad for parameter in frozen_module.parameters())

    groups = build_optimizer_param_groups(
        model,
        conditioner,
        regime=regime,
        head_adapter_lr=1e-4,
        action_dit_lr=1e-5,
        weight_decay=1e-2,
    )
    assert [group["name"] for group in groups] == expected_group_names
    assert groups[0]["lr"] == pytest.approx(1e-4)
    assert groups[0]["weight_decay"] == pytest.approx(1e-2)
    if action_trainable:
        assert groups[1]["lr"] == pytest.approx(1e-5)
        assert groups[1]["weight_decay"] == pytest.approx(1e-2)

    grouped_parameters = [
        parameter for group in groups for parameter in list(group["params"])
    ]
    assert len(grouped_parameters) == len(_parameter_ids(grouped_parameters))
    expected_trainable = [
        parameter
        for parameter in (*model.parameters(), *conditioner.parameters())
        if parameter.requires_grad
    ]
    assert _parameter_ids(grouped_parameters) == _parameter_ids(expected_trainable)


def test_frozen_action_path_backpropagates_to_gate_and_head_with_contrastive_loss() -> None:
    torch.manual_seed(23)
    model = _TinyFastWAM()
    conditioner = _tiny_conditioner()
    configure_trainable_modules(model, conditioner, regime="p_v1")

    # This frozen tail stands in for all ActionDiT blocks and its action head.
    frozen_tail = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 5))
    frozen_tail.requires_grad_(False)

    visual_tokens = torch.randn(3, 5, 12)
    raw_action = torch.randn(3, 6, 5)
    content_tokens = conditioner.content_tokens(visual_tokens)
    contrastive_embedding = conditioner.contrastive(visual_tokens)
    encoded_action = model.action_expert.action_encoder(raw_action)
    conditioned_action = conditioner.inject_action_tokens(
        encoded_action,
        content_tokens,
    )

    action_loss = frozen_tail(conditioned_action).square().mean()
    # At gate=0 the action branch initially reaches the gate but not the head;
    # the paired contrastive branch supplies the required first-step head grad.
    contrastive_loss = -contrastive_embedding[:, 0].mean()
    loss = action_loss + 0.1 * contrastive_loss
    assert torch.isfinite(loss)
    loss.backward()

    assert conditioner.adapter.gate.grad is not None
    assert abs(float(conditioner.adapter.gate.grad.item())) > 0.0
    assert _grad_norm(conditioner.adapter) > 0.0
    assert _grad_norm(conditioner.head) > 0.0

    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in frozen_tail.parameters())
