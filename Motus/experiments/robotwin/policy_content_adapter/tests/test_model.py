from __future__ import annotations

import torch
from torch import nn

from experiments.robotwin.policy_content_adapter.model import (
    GatedCrossAttentionAdapter,
    MotusContentHead,
    MotusPolicyContentConditioner,
    configure_trainable_parameters,
    optimizer_parameter_groups,
)


def _small_modules():
    head = MotusContentHead(
        backbone_dim=12, content_dim=8, num_queries=3, num_heads=2
    )
    adapter = GatedCrossAttentionAdapter(
        action_dim=16, content_dim=8, num_heads=4
    )
    return head, adapter, MotusPolicyContentConditioner(head, adapter)


def test_head_shapes_and_normalization() -> None:
    head, _, _ = _small_modules()
    visual = torch.randn(4, 7, 12)
    content = head.forward_content_tokens(visual)
    embedding = head.forward_contrastive(visual)
    assert content.shape == (4, 3, 8)
    assert embedding.shape == (4, 8)
    torch.testing.assert_close(
        embedding.norm(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-6
    )


def test_zero_gate_is_bit_exact() -> None:
    _, adapter, _ = _small_modules()
    action = torch.randn(2, 5, 16)
    content = torch.randn(2, 3, 8)
    output = adapter(action, content)
    assert adapter.gate.detach().item() == 0.0
    assert torch.equal(output, action)


def test_gate_opens_head_and_attention_gradient_path() -> None:
    _, adapter, conditioner = _small_modules()
    visual = torch.randn(2, 7, 12)
    action = torch.randn(2, 5, 16, requires_grad=True)

    # At exact zero init only the scalar gate can receive action-path signal.
    content = conditioner.content_tokens(visual)
    conditioner.inject_action_tokens(action, content).square().mean().backward()
    assert adapter.gate.grad is not None
    assert torch.isfinite(adapter.gate.grad)
    assert abs(float(adapter.gate.grad)) > 0
    projection_grad = conditioner.head.token_projection.weight.grad
    assert projection_grad is None or torch.count_nonzero(projection_grad) == 0

    conditioner.zero_grad(set_to_none=True)
    action.grad = None
    with torch.no_grad():
        adapter.gate.fill_(0.25)
    content = conditioner.content_tokens(visual)
    conditioner.inject_action_tokens(action, content).square().mean().backward()
    assert conditioner.head.token_projection.weight.grad is not None
    assert torch.linalg.vector_norm(
        conditioner.head.token_projection.weight.grad
    ).item() > 0
    assert adapter.cross_attention.q_proj_weight.grad is not None
    assert torch.linalg.vector_norm(
        adapter.cross_attention.q_proj_weight.grad
    ).item() > 0


class _DummyMotus(nn.Module):
    def __init__(self, conditioner: nn.Module) -> None:
        super().__init__()
        self.video_model = nn.Linear(4, 4)
        self.vlm_model = nn.Linear(4, 4)
        self.und_expert = nn.Linear(4, 4)
        self.action_expert = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.policy_content_conditioner = conditioner


def test_freeze_contract_and_optimizer_groups() -> None:
    _, _, conditioner = _small_modules()
    model = _DummyMotus(conditioner)
    counts = configure_trainable_parameters(model, conditioner, regime="m_p1")
    assert counts["conditioner"] > 0
    assert counts["action_expert"] == 0
    assert all(not p.requires_grad for p in model.video_model.parameters())

    counts = configure_trainable_parameters(model, conditioner, regime="m_p2")
    assert counts["action_expert"] > 0
    assert all(p.requires_grad for p in model.action_expert.parameters())
    assert all(not p.requires_grad for p in model.video_model.parameters())
    groups = optimizer_parameter_groups(
        model,
        conditioner,
        head_adapter_lr=1e-4,
        action_expert_lr=1e-5,
    )
    assert [group["name"] for group in groups] == [
        "content_head_gca",
        "action_expert",
    ]
    assert [group["lr"] for group in groups] == [1e-4, 1e-5]
