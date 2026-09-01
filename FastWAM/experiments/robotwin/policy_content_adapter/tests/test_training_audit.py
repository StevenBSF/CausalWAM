"""Exact CPU audits for the policy-adapter training invariants."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import nn

from experiments.robotwin.policy_content_adapter.losses import (
    tensor_distribution_summary,
)
from experiments.robotwin.policy_content_adapter.model import (
    GatedCrossAttentionAdapter,
    PolicyContentConditioner,
    PolicyContentHead,
)
from experiments.robotwin.policy_content_adapter.training_audit import (
    DistributionAccumulator,
    ParameterSnapshot,
    SampledParameterSnapshot,
    action_path_gradient_probe,
    compare_distributions,
)
from experiments.robotwin.policy_content_adapter import train as train_module


class _PlainModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


def test_prepared_raw_module_resolution_avoids_optional_deepspeed_import() -> None:
    original = nn.Linear(3, 2)
    assert train_module._resolve_prepared_raw_module(original, original) is original
    prepared = _PlainModuleWrapper(_PlainModuleWrapper(original))
    assert train_module._resolve_prepared_raw_module(prepared, original) is original
    with pytest.raises(RuntimeError, match="neither the original"):
        train_module._resolve_prepared_raw_module(nn.Linear(3, 2), original)
    assert "unwrap_model(" not in inspect.getsource(train_module.run)


def _tiny_conditioner() -> PolicyContentConditioner:
    return PolicyContentConditioner(
        head=PolicyContentHead(
            backbone_dim=12,
            embed_dim=8,
            num_queries=2,
            num_heads=2,
        ),
        adapter=GatedCrossAttentionAdapter(
            action_dim=16,
            content_dim=8,
            num_heads=4,
        ),
    )


def _action_loss(conditioner: PolicyContentConditioner) -> torch.Tensor:
    visual_tokens = torch.linspace(-1.0, 1.0, 3 * 5 * 12).reshape(3, 5, 12)
    action_tokens = torch.linspace(0.75, -0.5, 3 * 4 * 16).reshape(3, 4, 16)
    content_tokens = conditioner.content_tokens(visual_tokens)
    conditioned = conditioner.inject_action_tokens(action_tokens, content_tokens)
    # A frozen downstream action projection.  The loss has no contrastive term,
    # so every reported head/MHA gradient can only have traversed the action path.
    frozen_action_tail = torch.linspace(-0.7, 0.9, 16).reshape(1, 1, 16)
    return (conditioned * frozen_action_tail).sum()


def test_action_gradient_probe_locks_exact_zero_gate_then_open_gate_contract() -> None:
    torch.manual_seed(123)
    conditioner = _tiny_conditioner()

    assert conditioner.adapter.gate.detach().item() == 0.0
    zero_gate = action_path_gradient_probe(_action_loss(conditioner), conditioner)

    # tanh(0) blocks the action-only signal to both the content head and GCA
    # weights exactly, while the scalar gate itself must receive a useful signal.
    assert zero_gate["head_grad_norm"] == 0.0
    assert zero_gate["adapter_attention_grad_norm"] == 0.0
    assert zero_gate["gate_grad_norm"] > 0.0
    assert zero_gate["head_gradient_tensors"] > 0
    assert zero_gate["adapter_attention_gradient_tensors"] > 0
    assert zero_gate["gate_gradient_tensors"] == 1
    assert zero_gate["all_finite"] is True

    with torch.no_grad():
        conditioner.adapter.gate.fill_(0.25)
    open_gate = action_path_gradient_probe(_action_loss(conditioner), conditioner)

    assert open_gate["head_grad_norm"] > 0.0
    assert open_gate["adapter_attention_grad_norm"] > 0.0
    assert open_gate["gate_grad_norm"] > 0.0
    assert open_gate["all_finite"] is True


def test_action_gradient_probe_rejects_non_scalar_and_detached_losses() -> None:
    conditioner = _tiny_conditioner()
    with pytest.raises(ValueError, match="differentiable scalar"):
        action_path_gradient_probe(torch.ones(2, requires_grad=True), conditioner)
    with pytest.raises(ValueError, match="differentiable scalar"):
        action_path_gradient_probe(torch.ones(()), conditioner)


def test_action_content_gradient_hook_survives_forward_output_conversion() -> None:
    """Accelerate may replace returned bf16 tensors; audit the graph tensor itself."""

    torch.manual_seed(321)
    conditioner = _tiny_conditioner()
    with torch.no_grad():
        conditioner.adapter.gate.fill_(0.25)
    visual_tokens = torch.randn(2, 5, 12)
    action_tokens = torch.randn(2, 4, 16)
    content_tokens = conditioner.content_tokens(visual_tokens)
    conditioner.arm_action_content_gradient_audit(content_tokens)

    # This is analogous to Accelerate recursively converting a tensor included
    # in the forward return.  It is a distinct descendant and is intentionally
    # not used to form the loss.
    converted_forward_output = content_tokens.to(torch.float64)
    loss = conditioner.inject_action_tokens(action_tokens, content_tokens).square().mean()
    loss.backward()

    assert converted_forward_output.dtype is torch.float64
    assert conditioner.consume_action_content_gradient_audit() > 0.0
    with pytest.raises(RuntimeError, match="gradient hook did not run"):
        conditioner.consume_action_content_gradient_audit()


def test_parameter_snapshot_reports_exact_per_module_updates_and_schema_change() -> None:
    modules = {
        "head": nn.Linear(3, 2, bias=True),
        "adapter": nn.Linear(2, 1, bias=False),
    }
    snapshot = ParameterSnapshot.capture(modules)

    with torch.no_grad():
        modules["adapter"].weight[0, 1].add_(0.125)
    report = snapshot.compare(modules)

    assert report["max_abs_delta"] == pytest.approx(0.125)
    assert report["max_abs_delta_by_module"] == {
        "head": 0.0,
        "adapter": pytest.approx(0.125),
    }
    assert report["changed_parameter_tensors"] == 1
    assert report["all_finite"] is True

    modules["head"].register_parameter("new_parameter", nn.Parameter(torch.zeros(())))
    with pytest.raises(RuntimeError, match="parameter set changed"):
        snapshot.compare(modules)


class _ManyTrainableTensors(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.ones(9, dtype=torch.float32)) for _ in range(6)]
        )
        self.frozen = nn.Parameter(torch.ones(100), requires_grad=False)


def test_sampled_parameter_snapshot_is_bounded_deterministic_and_bf16_aware() -> None:
    expert = _ManyTrainableTensors()
    kwargs = {"max_parameter_tensors": 3, "samples_per_tensor": 4}
    first = SampledParameterSnapshot.capture({"action_expert": expert}, **kwargs)
    second = SampledParameterSnapshot.capture({"action_expert": expert}, **kwargs)

    # Midpoint strata choose tensors 1, 3, 5 out of six and logical flattened
    # elements 1, 3, 5, 7 out of nine.  The frozen tensor is not eligible.
    expected_names = [
        "action_expert.weights.1",
        "action_expert.weights.3",
        "action_expert.weights.5",
    ]
    assert list(first.values) == expected_names
    assert list(second.values) == expected_names
    assert first.eligible_trainable_parameter_tensors == 6
    assert sum(value.numel() for value in first.values.values()) == 12
    assert all(
        torch.equal(indices, torch.tensor([1, 3, 5, 7]))
        for indices in first.indices.values()
    )
    assert all(
        torch.equal(first.indices[name], second.indices[name])
        and torch.equal(first.values[name], second.values[name])
        for name in expected_names
    )

    with torch.no_grad():
        expert.weights[1][1].add_(0.125)  # survives bf16 quantization
        expert.weights[3][3].add_(0.001)  # exact fp32 change, hidden by bf16
        expert.weights[5][0].add_(2.0)  # deliberately outside sampled elements
    report = first.compare({"action_expert": expert})

    assert report["sampling_strategy"] == "deterministic_stratified_midpoint"
    assert report["eligible_trainable_parameter_tensors"] == 6
    assert report["sampled_parameter_tensors"] == 3
    assert report["sampled_parameter_names"] == expected_names
    assert report["sampled_elements"] == 12
    assert report["changed_parameter_tensors"] == 2
    assert report["changed_elements"] == 2
    assert report["changed_fraction"] == pytest.approx(2 / 12)
    assert report["max_abs_delta"] == pytest.approx(0.125)
    assert report["mean_abs_delta"] == pytest.approx((0.125 + 0.001) / 12)
    assert report["deployment_quantization"] == "bfloat16"
    assert report["deployment_visible_changed_elements"] == 1
    assert report["deployment_visible_changed_fraction"] == pytest.approx(1 / 12)
    assert report["quantized_update_retention"] == pytest.approx(1 / 2)
    assert report["by_module"]["action_expert"]["changed_fraction"] == pytest.approx(
        2 / 12
    )
    assert report["by_module"]["action_expert"][
        "deployment_visible_changed_fraction"
    ] == pytest.approx(1 / 12)
    assert report["by_module"]["action_expert"][
        "quantized_update_retention"
    ] == pytest.approx(1 / 2)
    assert report["all_finite"] is True


def test_sampled_parameter_snapshot_fails_closed_on_structure_and_values() -> None:
    expert = _ManyTrainableTensors()
    snapshot = SampledParameterSnapshot.capture(
        {"action_expert": expert}, max_parameter_tensors=6, samples_per_tensor=9
    )
    expert.register_parameter("extra", nn.Parameter(torch.zeros(1)))
    with pytest.raises(RuntimeError, match="trainable parameter set changed"):
        snapshot.compare({"action_expert": expert})

    del expert.extra
    with torch.no_grad():
        expert.weights[0][0] = torch.nan
    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        snapshot.compare({"action_expert": expert})

    frozen = nn.Linear(2, 2)
    frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="no non-empty trainable parameters"):
        SampledParameterSnapshot.capture({"frozen": frozen})
    with pytest.raises(ValueError, match="must be positive"):
        SampledParameterSnapshot.capture(
            {"action_expert": _ManyTrainableTensors()},
            max_parameter_tensors=0,
        )
    with pytest.raises(ValueError, match="required sampled parameters are missing"):
        SampledParameterSnapshot.capture(
            {"action_expert": _ManyTrainableTensors()},
            required_parameter_names=("action_expert.not_present",),
        )


def test_sampled_parameter_snapshot_forces_strata_and_audits_adam_state() -> None:
    expert = _ManyTrainableTensors()
    snapshot = SampledParameterSnapshot.capture(
        {"action_expert": expert},
        max_parameter_tensors=3,
        samples_per_tensor=3,
        required_parameter_names=(
            "action_expert.weights.0",
            "action_expert.weights.5",
        ),
    )
    assert "action_expert.weights.0" in snapshot.values
    assert "action_expert.weights.5" in snapshot.values
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    for parameter in expert.weights:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    report = snapshot.optimizer_state_report(
        {"action_expert": expert}, optimizer
    )
    assert report["sampled_elements"] == 9
    assert report["nonzero_fraction"] == 1.0
    assert report["dtypes"] == ["torch.float32"]


def test_distribution_accumulator_exactly_merges_tensor_moments_and_compares() -> None:
    first = torch.tensor([[[-2.0, -1.0], [0.0, 1.0]]])
    second = torch.tensor([[[2.0, 3.0]]])
    accumulator = DistributionAccumulator()
    accumulator.add(tensor_distribution_summary(first), tasks=["place_a2b_left"])
    accumulator.add(tensor_distribution_summary(second), tasks=["open_microwave"])
    merged = accumulator.finalize()

    flat = torch.cat((first.flatten(), second.flatten())).double()
    token_norms = torch.cat(
        (first.float().norm(dim=-1).flatten(), second.float().norm(dim=-1).flatten())
    ).double()
    assert merged["element_count"] == 6
    assert merged["mean"] == pytest.approx(float(flat.mean().item()))
    # The implementation intentionally reports population moments.
    assert merged["std"] == pytest.approx(float(flat.std(unbiased=False).item()))
    assert merged["token_count"] == 3
    assert merged["token_l2_mean"] == pytest.approx(float(token_norms.mean().item()))
    assert merged["token_l2_std"] == pytest.approx(
        float(token_norms.std(unbiased=False).item())
    )
    assert merged["minimum"] == -2.0
    assert merged["maximum"] == 3.0
    assert merged["observed_shapes"] == [[1, 1, 2], [1, 2, 2]]
    assert merged["task_sequence"] == ["place_a2b_left", "open_microwave"]
    assert merged["task_set"] == ["open_microwave", "place_a2b_left"]

    shifted = dict(merged)
    shifted["mean"] = float(merged["mean"]) + 0.5
    shifted["std"] = float(merged["std"]) * 2.0
    shifted["token_l2_mean"] = float(merged["token_l2_mean"]) + 1.25
    comparison = compare_distributions(merged, shifted)
    assert comparison["absolute_mean_gap"] == pytest.approx(0.5)
    assert comparison["std_ratio_official_over_paired"] == pytest.approx(0.5)
    assert comparison["token_l2_mean_gap"] == pytest.approx(1.25)
    expected_pooled_std = math.sqrt(
        (float(merged["std"]) ** 2 + float(shifted["std"]) ** 2) / 2.0
    )
    assert comparison["standardized_mean_gap"] == pytest.approx(
        0.5 / expected_pooled_std
    )

    restored = DistributionAccumulator.from_state_dict(accumulator.state_dict())
    assert restored.state_dict() == accumulator.state_dict()
    assert restored.finalize() == merged


def test_distribution_accumulator_fails_closed_on_schema_drift_and_empty_data() -> None:
    accumulator = DistributionAccumulator()
    with pytest.raises(ValueError, match="empty distribution"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="schema differs"):
        accumulator.add({"shape": [1, 1, 1]}, tasks=["move_stapler_pad"])
    state = accumulator.state_dict()
    state["minimum"] = 0.0
    with pytest.raises(ValueError, match="empty distribution resume extrema"):
        DistributionAccumulator.from_state_dict(state)
