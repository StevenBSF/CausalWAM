from __future__ import annotations

import torch
from torch import nn

from experiments.robotwin.policy_content_adapter.dual_stream import (
    audit_dual_stream_gradients,
    compute_dual_stream_loss,
)
from experiments.robotwin.policy_content_adapter.model import (
    GatedCrossAttentionAdapter,
    MotusContentHead,
    MotusPolicyContentConditioner,
)


class _DummyMotus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.video_model = nn.Linear(1, 1)
        self.vlm_model = nn.Linear(1, 1)
        self.und_expert = nn.Linear(1, 1)
        self.action_expert = nn.Linear(1, 1)
        self.last_kwargs = None

    def training_step(self, **kwargs):
        self.last_kwargs = kwargs
        content = kwargs["policy_content_tokens"]
        # Depends on content and a stand-in action parameter.
        action_loss = (content.mean() * self.anchor - 0.2).square()
        return {
            "action_loss": action_loss,
            "video_loss": action_loss.detach().new_zeros(()),
        }


class _BFloat16ActionDummy(_DummyMotus):
    def __init__(self) -> None:
        super().__init__()
        self.action_expert.to(dtype=torch.bfloat16)


def _batch() -> dict:
    return {
        "first_frame": torch.rand(2, 3, 8, 8),
        "video_frames": torch.rand(2, 8, 3, 8, 8),
        "initial_state": torch.rand(2, 14),
        "action_sequence": torch.rand(2, 16, 14),
        "language_embedding": torch.rand(2, 5, 6),
        "vlm_inputs": {},
    }


def _conditioner() -> MotusPolicyContentConditioner:
    return MotusPolicyContentConditioner(
        MotusContentHead(
            backbone_dim=12, content_dim=8, num_queries=3, num_heads=2
        ),
        GatedCrossAttentionAdapter(action_dim=16, content_dim=8, num_heads=4),
    )


def _extractor(model, *, first_frame, language_embeddings, capture_layer):
    del model, language_embeddings, capture_layer
    return torch.ones(first_frame.shape[0], 7, 12)


def test_m1_and_m3_share_action_loss_but_only_m3_uses_contrastive_gradient() -> None:
    torch.manual_seed(5)
    m1_model, m3_model = _DummyMotus(), _DummyMotus()
    m3_model.load_state_dict(m1_model.state_dict())
    m1, m3 = _conditioner(), _conditioner()
    m3.load_state_dict(m1.state_dict())
    paired = torch.randn(4, 4, 7, 12)
    states = ["s0", "s1", "s2", "s3"]
    tasks = ["t0", "t0", "t1", "t1"]
    result_m1 = compute_dual_stream_loss(
        motus_model=m1_model,
        conditioner=m1,
        official_batch=_batch(),
        paired_visual_tokens=paired,
        paired_physical_state_ids=states,
        paired_task_ids=tasks,
        control="m1_architecture_action_control",
        lambda_contrastive=0.0,
        observation_extractor=_extractor,
    )
    result_m3 = compute_dual_stream_loss(
        motus_model=m3_model,
        conditioner=m3,
        official_batch=_batch(),
        paired_visual_tokens=paired,
        paired_physical_state_ids=states,
        paired_task_ids=tasks,
        control="m3_ours",
        lambda_contrastive=0.1,
        observation_extractor=_extractor,
    )
    torch.testing.assert_close(result_m1.action, result_m3.action)
    torch.testing.assert_close(result_m1.contrastive, result_m3.contrastive)
    assert m1_model.last_kwargs["compute_video_loss"] is False
    assert m3_model.last_kwargs["compute_video_loss"] is False

    result_m1.total.backward()
    result_m3.total.backward()
    m1_mlp = m1.head.mlp[0].weight.grad
    m3_mlp = m3.head.mlp[0].weight.grad
    assert m1_mlp is None or torch.count_nonzero(m1_mlp) == 0
    assert m3_mlp is not None and torch.linalg.vector_norm(m3_mlp).item() > 0


def test_official_state_and_actions_match_action_expert_dtype() -> None:
    model = _BFloat16ActionDummy()
    compute_dual_stream_loss(
        motus_model=model,
        conditioner=_conditioner(),
        official_batch=_batch(),
        paired_visual_tokens=torch.randn(2, 4, 7, 12),
        paired_physical_state_ids=["s0", "s1"],
        paired_task_ids=["t0", "t0"],
        control="m1_architecture_action_control",
        lambda_contrastive=0.0,
        observation_extractor=_extractor,
    )

    assert model.last_kwargs["state"].dtype is torch.bfloat16
    assert model.last_kwargs["actions"].dtype is torch.bfloat16


def test_distributed_gradient_snapshot_is_audited_before_step() -> None:
    model = _DummyMotus()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = audit_dual_stream_gradients(
        motus_model=model,
        conditioner=_conditioner(),
        control="m1_architecture_action_control",
        regime="m_p1",
        step=1,
        gradient_snapshot={
            "content_head_grad_norm": 0.0,
            "adapter_grad_norm": 0.4,
            "adapter_gate_grad": 0.2,
            "action_expert_grad_norm": 0.0,
        },
    )

    assert result["status"] == "PASS"
    assert result["backend"] == "deepspeed_zero_partition_pre_step_v1"
