from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

from experiments.robotwin.policy_content_adapter import losses
from experiments.robotwin.policy_content_adapter.losses import (
    _audit_effective_action_weight,
    paired_action_loss,
    paired_contrastive_loss,
)
from experiments.robotwin.policy_content_adapter.protocol import (
    POLICY_PROTOCOL_ID,
    POLICY_VARIANTS,
)
from experiments.robotwin.policy_content_adapter import train as train_module
from experiments.robotwin.policy_content_adapter.train import PolicyTrainingModule


class _TinyConditioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(5, 4, bias=False)
        self.content_layer = 1
        self.head = SimpleNamespace(backbone_dim=5)

    def contrastive(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(tokens.mean(dim=1)), dim=-1)


def _contrastive_batch() -> dict[str, object]:
    generator = torch.Generator().manual_seed(5)
    return {
        "tokens": torch.randn((2, 4, 3, 5), generator=generator),
        "variant_names": POLICY_VARIANTS,
        "protocol_id": POLICY_PROTOCOL_ID,
        "r3_role": "training_positive",
        "supervision_mode": "contrastive",
        "physical_state_id": ("state_0", "state_1"),
        "task": ("task_a", "task_a"),
    }


def test_four_scene_contrastive_gives_every_anchor_three_positives() -> None:
    conditioner = _TinyConditioner()
    loss, diagnostics = paired_contrastive_loss(conditioner, _contrastive_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert diagnostics["positives_per_anchor"] == 3
    assert diagnostics["r3_training_positive"] is True
    loss.backward()
    assert conditioner.projection.weight.grad is not None
    assert torch.isfinite(conditioner.projection.weight.grad).all()


def test_contrastive_rejects_three_scenes_or_r3_holdout() -> None:
    conditioner = _TinyConditioner()
    three = _contrastive_batch()
    three["tokens"] = three["tokens"][:, :3]
    three["variant_names"] = POLICY_VARIANTS[:3]
    with pytest.raises(ValueError, match="C/R1/R2/R3"):
        paired_contrastive_loss(conditioner, three)

    holdout = _contrastive_batch()
    holdout["r3_role"] = "holdout"
    with pytest.raises(ValueError, match="training positive"):
        paired_contrastive_loss(conditioner, holdout)


def test_action_weight_audit_accepts_only_explicit_scheduler_endpoint() -> None:
    weight, positive, reason = _audit_effective_action_weight(
        torch.tensor(0.0), torch.tensor([32.0]), batch_size=1
    )
    assert weight.tolist() == [0.0]
    assert positive is False
    assert reason == "scheduler_zero_weight"

    weight, positive, reason = _audit_effective_action_weight(
        torch.tensor([0.0, 0.5]), torch.tensor([32.0, 7.0]), batch_size=2
    )
    assert weight.tolist() == [0.0, 0.5]
    assert positive is True
    assert reason == "none"


def test_native_bf16_scheduler_endpoint_has_exact_zero_weight() -> None:
    scheduler = WanContinuousFlowMatchScheduler()
    timestep = torch.tensor(1000.0, dtype=torch.bfloat16)
    assert float(scheduler.training_weight(timestep).item()) == 0.0


def test_action_weight_audit_rejects_all_padding_and_negative_weight() -> None:
    with pytest.raises(RuntimeError, match="no valid action target"):
        _audit_effective_action_weight(
            torch.tensor(0.0), torch.tensor([0.0]), batch_size=1
        )
    with pytest.raises(FloatingPointError, match="negative"):
        _audit_effective_action_weight(
            torch.tensor(-0.1), torch.tensor([32.0]), batch_size=1
        )


def _action_batch() -> dict[str, object]:
    groups, views = 2, 4
    action_one = torch.arange(32 * 14, dtype=torch.float32).reshape(1, 1, 32, 14)
    state_one = torch.arange(33 * 14, dtype=torch.float32).reshape(1, 1, 33, 14)
    proprio_one = state_one[:, :, :-1]
    context_one = torch.ones((1, 1, 5, 6))
    mask_one = torch.ones((1, 1, 5), dtype=torch.bool)
    pad_one = torch.zeros((1, 1, 32), dtype=torch.bool)
    return {
        "video": torch.randn((groups, views, 3, 9, 4, 4)),
        "action": action_one.expand(groups, views, -1, -1).clone(),
        "state_window": state_one.expand(groups, views, -1, -1).clone(),
        "proprio": proprio_one.expand(groups, views, -1, -1).clone(),
        "context": context_one.expand(groups, views, -1, -1).clone(),
        "context_mask": mask_one.expand(groups, views, -1).clone(),
        "action_is_pad": pad_one.expand(groups, views, -1).clone(),
        "protocol_id": POLICY_PROTOCOL_ID,
        "variant_names": POLICY_VARIANTS,
        "view_count": 4,
        "r3_role": "training_positive",
        "camera_count": 3,
        "camera_names": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        "native_fps": 50,
        "action_steps": 32,
        "action_dim": 14,
        "temporal_resampling": "none",
        "native_action_targets": True,
        "split": "train",
        "supervision_mode": "action",
    }


def test_paired_action_loss_flattens_native_four_scene_batch(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_official_action_loss(model, runtime, sample):
        captured.update(sample)
        return torch.tensor(1.25, requires_grad=True), {
            "video_token_shape": [8, 120, 3072],
            "content_token_shape": [8, 8, 384],
            "action_token_shape": [8, 32, 1024],
            "loss_action": 1.25,
            "official_layer16_distribution": {"shape": [8, 120, 3072]},
        }

    monkeypatch.setattr(losses, "official_action_loss", fake_official_action_loss)
    loss, diagnostics = paired_action_loss(object(), object(), _action_batch())
    assert float(loss.detach()) == 1.25
    assert captured["video"].shape == (8, 3, 9, 4, 4)
    assert captured["action"].shape == (8, 32, 14)
    assert diagnostics["loss_paired_action"] == 1.25
    assert diagnostics["r3_training_positive"] is True


def test_paired_action_loss_rejects_nonidentical_r3_target(monkeypatch) -> None:
    batch = _action_batch()
    batch["action"][0, 3, 0, 0] += 1
    with pytest.raises(ValueError, match="action is not exact"):
        paired_action_loss(object(), object(), batch)


@pytest.mark.parametrize(
    ("mode", "lambda_action", "lambda_ctr", "expected"),
    (("none", 0.0, 0.0, 1.0), ("action", 1.0, 0.0, 3.0), ("contrastive", 0.0, 0.1, 1.2)),
)
def test_training_module_keeps_c1_c2_c3_supervision_distinct(
    monkeypatch, mode: str, lambda_action: float, lambda_ctr: float, expected: float
) -> None:
    conditioner = nn.Linear(1, 1)
    runtime = SimpleNamespace(conditioner=conditioner)
    model = nn.Linear(1, 1)

    def fake_official(*_args):
        return torch.tensor(1.0, requires_grad=True), {
            "loss_action": 1.0,
            "official_layer16_distribution": {"shape": [1, 1, 1]},
        }

    def fake_paired_action(*_args):
        return torch.tensor(2.0, requires_grad=True), {
            "loss_paired_action": 2.0,
            "paired_layer16_distribution": {"shape": [1, 1, 1]},
        }

    def fake_contrastive(*_args, **_kwargs):
        return torch.tensor(2.0, requires_grad=True), {
            "loss_contrastive": 2.0,
            "positive_similarity": 0.5,
            "negative_similarity": 0.0,
            "paired_clean_layer16_distribution": {"shape": [1, 1, 1]},
        }

    monkeypatch.setattr(train_module, "official_action_loss", fake_official)
    monkeypatch.setattr(train_module, "paired_action_loss", fake_paired_action)
    monkeypatch.setattr(train_module, "paired_contrastive_loss", fake_contrastive)
    module = PolicyTrainingModule(
        model,
        runtime,
        paired_supervision_mode=mode,
        lambda_contrastive=lambda_ctr,
        lambda_paired_action=lambda_action,
        temperature=0.07,
        training_seed=11,
        process_index=0,
    )
    paired = None if mode == "none" else {"mode": mode}
    total, _, _, diagnostics = module({}, paired)
    assert float(total.detach()) == pytest.approx(expected)
    assert diagnostics["paired_supervision_mode"] == mode
    assert diagnostics["loss_paired_action"] == (2.0 if mode == "action" else 0.0)
    assert diagnostics["step_rng_policy_id"] == train_module.STAGE2_STEP_RNG_POLICY_ID
    assert diagnostics["official_rng_seed"] == train_module.stage2_step_rng_seed(11, 0)
    assert diagnostics["step_rng_training_seed"] == 11
    assert diagnostics["step_rng_step_index"] == 0


def test_c1_training_module_refuses_any_paired_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        train_module,
        "official_action_loss",
        lambda *_args: (
            torch.tensor(1.0, requires_grad=True),
            {"loss_action": 1.0, "official_layer16_distribution": {}},
        ),
    )
    runtime = SimpleNamespace(conditioner=nn.Linear(1, 1))
    module = PolicyTrainingModule(
        nn.Linear(1, 1),
        runtime,
        paired_supervision_mode="none",
        lambda_contrastive=0.0,
        lambda_paired_action=0.0,
        temperature=0.07,
        training_seed=11,
        process_index=0,
    )
    with pytest.raises(ValueError, match="must not consume paired data"):
        module({}, {"unexpected": True})


def test_pair280_inactive_step_is_strict_action_only(monkeypatch) -> None:
    calls = {"official": 0, "contrastive": 0}

    def fake_official(*_args):
        calls["official"] += 1
        return torch.tensor(1.0, requires_grad=True), {
            "loss_action": 1.0,
            "official_layer16_distribution": {"shape": [1, 1, 1]},
        }

    def fake_contrastive(*_args, **_kwargs):
        calls["contrastive"] += 1
        return torch.tensor(2.0, requires_grad=True), {
            "loss_contrastive": 2.0,
            "positive_similarity": 0.5,
            "negative_similarity": 0.0,
            "positives_per_anchor": 3,
            "r3_training_positive": True,
            "paired_clean_layer16_distribution": {"shape": [1, 1, 1]},
        }

    monkeypatch.setattr(train_module, "official_action_loss", fake_official)
    monkeypatch.setattr(train_module, "paired_contrastive_loss", fake_contrastive)
    runtime = SimpleNamespace(conditioner=nn.Linear(1, 1))
    module = PolicyTrainingModule(
        nn.Linear(1, 1),
        runtime,
        paired_supervision_mode="contrastive",
        lambda_contrastive=0.1,
        lambda_paired_action=0.0,
        temperature=0.07,
        training_seed=1,
        process_index=0,
    )

    total, action, contrastive, diagnostics = module(
        {}, None, paired_active=False
    )
    assert total.item() == action.item() == 1.0
    assert contrastive.item() == 0.0
    assert calls == {"official": 1, "contrastive": 0}
    assert diagnostics["paired_contrastive_active"] is False
    assert diagnostics["paired_contrastive_gradient_enabled"] is False
    assert diagnostics["loss_contrastive"] == 0.0
    with pytest.raises(ValueError, match="inactive contrastive step"):
        module({}, {"unexpected": True}, paired_active=False)

    total, _action, contrastive, diagnostics = module(
        {}, {"paired": True}, paired_active=True
    )
    assert total.item() == pytest.approx(1.2)
    assert contrastive.item() == 2.0
    assert calls == {"official": 3, "contrastive": 1}
    assert diagnostics["paired_contrastive_active"] is True
    assert diagnostics["paired_contrastive_gradient_enabled"] is True


def test_c2_paired_rng_cannot_shift_next_official_noise(monkeypatch) -> None:
    official_draws: list[float] = []

    def fake_official(*_args):
        official_draws.append(float(torch.rand(())))
        return torch.tensor(1.0, requires_grad=True), {
            "loss_action": 1.0,
            "official_layer16_distribution": {},
        }

    def fake_paired(*_args):
        # Deliberately consume several global draws inside the paired context.
        torch.rand((20,))
        return torch.tensor(1.0, requires_grad=True), {
            "loss_paired_action": 1.0,
            "paired_layer16_distribution": {},
        }

    monkeypatch.setattr(train_module, "official_action_loss", fake_official)
    monkeypatch.setattr(train_module, "paired_action_loss", fake_paired)

    def build(mode: str) -> PolicyTrainingModule:
        runtime = SimpleNamespace(conditioner=nn.Linear(1, 1))
        return PolicyTrainingModule(
            nn.Linear(1, 1),
            runtime,
            paired_supervision_mode=mode,
            lambda_contrastive=0.0,
            lambda_paired_action=1.0 if mode == "action" else 0.0,
            temperature=0.07,
            training_seed=77,
            process_index=0,
        )

    c1 = build("none")
    _, _, _, c1_step0 = c1({}, None)
    _, _, _, c1_step1 = c1({}, None)
    c1_draws = tuple(official_draws)
    official_draws.clear()
    c2 = build("action")
    _, _, _, c2_step0 = c2({}, {})
    _, _, _, c2_step1 = c2({}, {})
    assert tuple(official_draws) == c1_draws
    assert c1_step0["official_rng_seed"] == c2_step0["official_rng_seed"]
    assert c1_step1["official_rng_seed"] == c2_step1["official_rng_seed"]
    assert c1_step0["official_rng_seed"] == train_module.stage2_step_rng_seed(77, 0)
    assert c1_step1["official_rng_seed"] == train_module.stage2_step_rng_seed(77, 1)
    assert c2_step0["paired_rng_seed"] == train_module.stage2_step_rng_seed(
        77, 0, stream="paired"
    )
    assert c2_step1["paired_rng_seed"] == train_module.stage2_step_rng_seed(
        77, 1, stream="paired"
    )
