"""Small, testable audit primitives for policy-adapter smoke training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .model import PolicyContentConditioner


def _gradient_norm(gradients: Iterable[torch.Tensor | None]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is None:
            continue
        detached = gradient.detach()
        if not bool(torch.isfinite(detached).all().item()):
            raise FloatingPointError("gradient contains NaN or infinity")
        total += detached.double().square().sum().cpu()
    return float(total.sqrt().item())


def module_gradient_report(module: nn.Module) -> dict[str, Any]:
    """Describe populated gradients and reject every non-finite value."""

    parameters = list(module.parameters())
    populated = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return {
        "parameter_tensors": len(parameters),
        "trainable_parameter_tensors": sum(parameter.requires_grad for parameter in parameters),
        "gradient_tensors": len(populated),
        "gradient_norm": _gradient_norm(populated),
        "all_finite": True,
    }


def assert_no_parameter_gradients(module: nn.Module, *, label: str) -> None:
    offenders = [name for name, parameter in module.named_parameters() if parameter.grad is not None]
    if offenders:
        raise RuntimeError(f"frozen {label} received gradients: {offenders[:10]}")


def action_path_gradient_probe(
    action_loss: torch.Tensor,
    conditioner: PolicyContentConditioner,
) -> dict[str, Any]:
    """Measure action-loss-only gradients into head, GCA weights, and gate.

    At exact zero initialization only the scalar gate is expected to receive an
    action gradient.  Once an optimizer step opens the gate, both the content
    head and GCA attention weights must receive nonzero action gradients.  This
    explicitly distinguishes action-path use from gradients supplied by the
    paired contrastive branch.
    """

    if action_loss.ndim != 0 or not action_loss.requires_grad:
        raise ValueError("action_loss must be a differentiable scalar")
    head_parameters = [parameter for parameter in conditioner.head.parameters() if parameter.requires_grad]
    attention_parameters = [
        parameter
        for parameter in conditioner.adapter.cross_attention.parameters()
        if parameter.requires_grad
    ]
    gate_parameters = [conditioner.adapter.gate]
    all_parameters = head_parameters + attention_parameters + gate_parameters
    gradients = torch.autograd.grad(
        action_loss,
        all_parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    head_stop = len(head_parameters)
    attention_stop = head_stop + len(attention_parameters)
    head_gradients = gradients[:head_stop]
    attention_gradients = gradients[head_stop:attention_stop]
    gate_gradients = gradients[attention_stop:]
    return {
        "head_grad_norm": _gradient_norm(head_gradients),
        "adapter_attention_grad_norm": _gradient_norm(attention_gradients),
        "gate_grad_norm": _gradient_norm(gate_gradients),
        "head_gradient_tensors": sum(value is not None for value in head_gradients),
        "adapter_attention_gradient_tensors": sum(
            value is not None for value in attention_gradients
        ),
        "gate_gradient_tensors": sum(value is not None for value in gate_gradients),
        "all_finite": True,
    }


@dataclass
class ParameterSnapshot:
    """A compact exact baseline for robust max-absolute update checks."""

    values: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, modules: Mapping[str, nn.Module]) -> "ParameterSnapshot":
        values: dict[str, torch.Tensor] = {}
        for module_name, module in modules.items():
            for parameter_name, parameter in module.named_parameters():
                values[f"{module_name}.{parameter_name}"] = (
                    parameter.detach().to(device="cpu").clone()
                )
        return cls(values=values)

    def compare(self, modules: Mapping[str, nn.Module]) -> dict[str, Any]:
        current: dict[str, torch.Tensor] = {}
        for module_name, module in modules.items():
            for parameter_name, parameter in module.named_parameters():
                current[f"{module_name}.{parameter_name}"] = parameter.detach().to(device="cpu")
        if set(current) != set(self.values):
            raise RuntimeError("parameter set changed after snapshot")
        by_module: dict[str, float] = {name: 0.0 for name in modules}
        global_max = 0.0
        changed_tensors = 0
        for name, before in self.values.items():
            after = current[name]
            difference = (after.float() - before.float()).abs()
            maximum = float(difference.max().item()) if difference.numel() else 0.0
            if not math.isfinite(maximum):
                raise FloatingPointError(f"parameter delta is non-finite for {name}")
            module_name = name.split(".", 1)[0]
            by_module[module_name] = max(by_module[module_name], maximum)
            global_max = max(global_max, maximum)
            changed_tensors += int(maximum > 0.0)
        return {
            "max_abs_delta": global_max,
            "max_abs_delta_by_module": by_module,
            "changed_parameter_tensors": changed_tensors,
            "all_finite": True,
        }

@dataclass(frozen=True)
class _SampledParameterMetadata:
    module_name: str
    parameter_name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int


def _trainable_parameter_records(
    modules: Mapping[str, nn.Module],
) -> list[tuple[str, _SampledParameterMetadata, nn.Parameter]]:
    """Return de-duplicated trainable parameters without copying their values."""

    records: list[tuple[str, _SampledParameterMetadata, nn.Parameter]] = []
    seen_parameters: set[int] = set()
    for module_name in sorted(modules):
        if not module_name:
            raise ValueError("module names must be non-empty")
        module = modules[module_name]
        if not isinstance(module, nn.Module):
            raise TypeError(f"{module_name!r} is not a torch module")
        for parameter_name, parameter in module.named_parameters():
            if not parameter.requires_grad or parameter.numel() == 0:
                continue
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            if not parameter.is_floating_point() or parameter.is_complex():
                raise TypeError(
                    f"sampled update audit only supports real floating-point parameters: "
                    f"{module_name}.{parameter_name} has dtype {parameter.dtype}"
                )
            full_name = f"{module_name}.{parameter_name}"
            records.append(
                (
                    full_name,
                    _SampledParameterMetadata(
                        module_name=module_name,
                        parameter_name=parameter_name,
                        shape=tuple(int(value) for value in parameter.shape),
                        dtype=str(parameter.dtype),
                        numel=int(parameter.numel()),
                    ),
                    parameter,
                )
            )
    return records


def _parameter_structure_sha256(
    records: Sequence[tuple[str, _SampledParameterMetadata, nn.Parameter]],
) -> str:
    digest = hashlib.sha256()
    for full_name, metadata, _parameter in records:
        # Length prefixes make the encoding unambiguous even if a module or
        # parameter name contains punctuation used in the remaining fields.
        encoded_name = full_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded_name)
        digest.update(repr(metadata.shape).encode("ascii"))
        digest.update(metadata.dtype.encode("ascii"))
        digest.update(metadata.numel.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _uniform_midpoint_indices(population_size: int, sample_size: int) -> torch.Tensor:
    """Choose deterministic stratified midpoints without allocating the population."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    count = min(population_size, sample_size)
    if count == population_size:
        return torch.arange(population_size, dtype=torch.int64)
    # Integer arithmetic avoids floating-point rounding drift for large tensors.
    return torch.tensor(
        [((2 * index + 1) * population_size) // (2 * count) for index in range(count)],
        dtype=torch.int64,
    )


def _gather_parameter_samples(
    parameter: nn.Parameter,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather only selected values, including from a non-contiguous parameter."""

    detached = parameter.detach()
    device_indices = indices.to(device=detached.device)
    if detached.is_contiguous():
        sampled = torch.index_select(detached.view(-1), 0, device_indices)
    else:
        coordinates = torch.unravel_index(device_indices, detached.shape)
        sampled = detached[coordinates]
    # `copy=True` ensures the snapshot cannot alias a CPU parameter's storage.
    return sampled.to(device="cpu", copy=True)


@dataclass
class SampledParameterSnapshot:
    """Memory-bounded, deterministic update baseline for a large trainable module.

    Parameter tensors are selected by evenly spaced strata over deterministic
    ``named_parameters`` registration order.  Values are then selected by the
    same midpoint rule over each tensor's flattened logical coordinates.  At
    most ``max_parameter_tensors * samples_per_tensor`` values (and the same
    number of int64 indices) are retained; the full parameter tensors are never
    copied to CPU.  This makes the audit suitable for P-v2's ActionDiT.
    """

    values: dict[str, torch.Tensor]
    indices: dict[str, torch.Tensor]
    metadata: dict[str, _SampledParameterMetadata]
    structure_sha256: str
    eligible_trainable_parameter_tensors: int
    max_parameter_tensors: int
    samples_per_tensor: int
    required_parameter_names: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        modules: Mapping[str, nn.Module],
        *,
        max_parameter_tensors: int = 32,
        samples_per_tensor: int = 1024,
        required_parameter_names: Sequence[str] = (),
    ) -> "SampledParameterSnapshot":
        if max_parameter_tensors <= 0:
            raise ValueError("max_parameter_tensors must be positive")
        if samples_per_tensor <= 0:
            raise ValueError("samples_per_tensor must be positive")
        records = _trainable_parameter_records(modules)
        if not records:
            raise ValueError("no non-empty trainable parameters to sample")
        required_names = tuple(str(name) for name in required_parameter_names)
        if len(required_names) != len(set(required_names)):
            raise ValueError("required_parameter_names contains duplicates")
        record_indices = {name: index for index, (name, _meta, _param) in enumerate(records)}
        missing = [name for name in required_names if name not in record_indices]
        if missing:
            raise ValueError(f"required sampled parameters are missing: {missing}")
        if len(required_names) > max_parameter_tensors:
            raise ValueError("required sampled parameters exceed max_parameter_tensors")
        required_indices = {record_indices[name] for name in required_names}
        remaining_indices = [
            index for index in range(len(records)) if index not in required_indices
        ]
        remaining_budget = min(
            max_parameter_tensors - len(required_indices), len(remaining_indices)
        )
        additional_indices = (
            [
                remaining_indices[index]
                for index in _uniform_midpoint_indices(
                    len(remaining_indices), remaining_budget
                ).tolist()
            ]
            if remaining_budget > 0
            else []
        )
        selected_record_indices = sorted(required_indices.union(additional_indices))
        values: dict[str, torch.Tensor] = {}
        indices: dict[str, torch.Tensor] = {}
        metadata: dict[str, _SampledParameterMetadata] = {}
        for record_index in selected_record_indices:
            full_name, parameter_metadata, parameter = records[record_index]
            parameter_indices = _uniform_midpoint_indices(
                parameter_metadata.numel, samples_per_tensor
            )
            parameter_values = _gather_parameter_samples(parameter, parameter_indices)
            if not bool(torch.isfinite(parameter_values).all().item()):
                raise FloatingPointError(
                    f"parameter contains NaN or infinity at sampled positions: {full_name}"
                )
            values[full_name] = parameter_values
            indices[full_name] = parameter_indices
            metadata[full_name] = parameter_metadata

        return cls(
            values=values,
            indices=indices,
            metadata=metadata,
            structure_sha256=_parameter_structure_sha256(records),
            eligible_trainable_parameter_tensors=len(records),
            max_parameter_tensors=max_parameter_tensors,
            samples_per_tensor=samples_per_tensor,
            required_parameter_names=required_names,
        )

    def compare(self, modules: Mapping[str, nn.Module]) -> dict[str, Any]:
        records = _trainable_parameter_records(modules)
        if (
            len(records) != self.eligible_trainable_parameter_tensors
            or _parameter_structure_sha256(records) != self.structure_sha256
        ):
            raise RuntimeError("trainable parameter set changed after sampled snapshot")
        current_parameters = {name: parameter for name, _metadata, parameter in records}
        if not set(self.values).issubset(current_parameters):
            raise RuntimeError("sampled parameter is missing after snapshot")

        sampled_elements = 0
        changed_elements = 0
        changed_parameter_tensors = 0
        deployment_visible_changed_elements = 0
        absolute_delta_sum = 0.0
        maximum_delta = 0.0
        by_parameter: dict[str, dict[str, Any]] = {}
        module_totals: dict[str, dict[str, float | int]] = {}

        for name, before in self.values.items():
            metadata = self.metadata[name]
            after = _gather_parameter_samples(current_parameters[name], self.indices[name])
            if str(current_parameters[name].dtype) != metadata.dtype:
                # Normally caught by the structure digest; retain a direct error
                # in case snapshot data is loaded from an untrusted serializer.
                raise RuntimeError(f"parameter dtype changed after snapshot: {name}")
            if tuple(current_parameters[name].shape) != metadata.shape:
                raise RuntimeError(f"parameter shape changed after snapshot: {name}")
            if not bool(torch.isfinite(after).all().item()):
                raise FloatingPointError(
                    f"parameter contains NaN or infinity at sampled positions: {name}"
                )

            difference = (after.to(torch.float64) - before.to(torch.float64)).abs()
            element_count = int(difference.numel())
            tensor_changed_elements = int((difference > 0.0).sum().item())
            tensor_maximum = float(difference.max().item()) if element_count else 0.0
            tensor_sum = float(difference.sum().item())
            bf16_changed = after.to(torch.bfloat16) != before.to(torch.bfloat16)
            tensor_deployment_visible = int(bf16_changed.sum().item())

            if not math.isfinite(tensor_maximum) or not math.isfinite(tensor_sum):
                raise FloatingPointError(f"parameter delta is non-finite for {name}")
            sampled_elements += element_count
            changed_elements += tensor_changed_elements
            changed_parameter_tensors += int(tensor_changed_elements > 0)
            deployment_visible_changed_elements += tensor_deployment_visible
            absolute_delta_sum += tensor_sum
            maximum_delta = max(maximum_delta, tensor_maximum)

            by_parameter[name] = {
                "sampled_elements": element_count,
                "changed_elements": tensor_changed_elements,
                "changed_fraction": tensor_changed_elements / element_count,
                "max_abs_delta": tensor_maximum,
                "mean_abs_delta": tensor_sum / element_count,
                "deployment_visible_changed_elements": tensor_deployment_visible,
                "deployment_visible_changed_fraction": (
                    tensor_deployment_visible / element_count
                ),
                "quantized_update_retention": (
                    tensor_deployment_visible / tensor_changed_elements
                    if tensor_changed_elements > 0
                    else None
                ),
            }
            module_total = module_totals.setdefault(
                metadata.module_name,
                {
                    "sampled_parameter_tensors": 0,
                    "changed_parameter_tensors": 0,
                    "sampled_elements": 0,
                    "changed_elements": 0,
                    "deployment_visible_changed_elements": 0,
                    "absolute_delta_sum": 0.0,
                    "max_abs_delta": 0.0,
                },
            )
            module_total["sampled_parameter_tensors"] += 1
            module_total["changed_parameter_tensors"] += int(
                tensor_changed_elements > 0
            )
            module_total["sampled_elements"] += element_count
            module_total["changed_elements"] += tensor_changed_elements
            module_total["deployment_visible_changed_elements"] += (
                tensor_deployment_visible
            )
            module_total["absolute_delta_sum"] += tensor_sum
            module_total["max_abs_delta"] = max(
                float(module_total["max_abs_delta"]), tensor_maximum
            )

        if sampled_elements <= 0:
            raise RuntimeError("sampled snapshot unexpectedly contains no elements")
        by_module: dict[str, dict[str, Any]] = {}
        for module_name, totals in module_totals.items():
            module_elements = int(totals["sampled_elements"])
            by_module[module_name] = {
                "sampled_parameter_tensors": int(totals["sampled_parameter_tensors"]),
                "changed_parameter_tensors": int(totals["changed_parameter_tensors"]),
                "sampled_elements": module_elements,
                "changed_elements": int(totals["changed_elements"]),
                "changed_fraction": int(totals["changed_elements"]) / module_elements,
                "max_abs_delta": float(totals["max_abs_delta"]),
                "mean_abs_delta": float(totals["absolute_delta_sum"])
                / module_elements,
                "deployment_visible_changed_elements": int(
                    totals["deployment_visible_changed_elements"]
                ),
                "deployment_visible_changed_fraction": int(
                    totals["deployment_visible_changed_elements"]
                )
                / module_elements,
                "quantized_update_retention": (
                    int(totals["deployment_visible_changed_elements"])
                    / int(totals["changed_elements"])
                    if int(totals["changed_elements"]) > 0
                    else None
                ),
            }

        return {
            "sampling_strategy": "deterministic_stratified_midpoint",
            "eligible_trainable_parameter_tensors": (
                self.eligible_trainable_parameter_tensors
            ),
            "sampled_parameter_tensors": len(self.values),
            "sampled_parameter_names": list(self.values),
            "required_parameter_names": list(self.required_parameter_names),
            "sampled_elements": sampled_elements,
            "changed_parameter_tensors": changed_parameter_tensors,
            "changed_elements": changed_elements,
            "changed_fraction": changed_elements / sampled_elements,
            "max_abs_delta": maximum_delta,
            "mean_abs_delta": absolute_delta_sum / sampled_elements,
            "deployment_quantization": "bfloat16",
            "deployment_visible_changed_elements": (
                deployment_visible_changed_elements
            ),
            "deployment_visible_changed_fraction": (
                deployment_visible_changed_elements / sampled_elements
            ),
            "quantized_update_retention": (
                deployment_visible_changed_elements / changed_elements
                if changed_elements > 0
                else None
            ),
            "by_module": by_module,
            "by_parameter": by_parameter,
            "all_finite": True,
        }

    def optimizer_state_report(
        self,
        modules: Mapping[str, nn.Module],
        optimizer: torch.optim.Optimizer,
        *,
        state_key: str = "exp_avg",
    ) -> dict[str, Any]:
        """Audit optimizer state at the identical bounded sample coordinates."""

        records = _trainable_parameter_records(modules)
        if (
            len(records) != self.eligible_trainable_parameter_tensors
            or _parameter_structure_sha256(records) != self.structure_sha256
        ):
            raise RuntimeError("trainable parameter set changed after sampled snapshot")
        parameters = {name: parameter for name, _metadata, parameter in records}
        sampled_elements = 0
        nonzero_elements = 0
        dtypes: set[str] = set()
        by_parameter: dict[str, dict[str, Any]] = {}
        for name, indices in self.indices.items():
            parameter = parameters[name]
            state = optimizer.state.get(parameter)
            if not isinstance(state, Mapping) or state_key not in state:
                raise RuntimeError(f"optimizer state {state_key!r} is missing for {name}")
            value = state[state_key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
                parameter.shape
            ):
                raise RuntimeError(f"optimizer state {state_key!r} shape differs for {name}")
            sampled = _gather_parameter_samples(value, indices)
            if not bool(torch.isfinite(sampled).all().item()):
                raise FloatingPointError(f"optimizer state is non-finite for {name}")
            count = int(sampled.numel())
            nonzero = int((sampled != 0).sum().item())
            sampled_elements += count
            nonzero_elements += nonzero
            dtypes.add(str(value.dtype))
            by_parameter[name] = {
                "sampled_elements": count,
                "nonzero_elements": nonzero,
                "nonzero_fraction": nonzero / count,
                "dtype": str(value.dtype),
            }
        if sampled_elements <= 0:
            raise RuntimeError("optimizer sampled-state audit contains no elements")
        return {
            "state_key": state_key,
            "sampled_elements": sampled_elements,
            "nonzero_elements": nonzero_elements,
            "nonzero_fraction": nonzero_elements / sampled_elements,
            "dtypes": sorted(dtypes),
            "by_parameter": by_parameter,
            "all_finite": True,
        }


@dataclass
class DistributionAccumulator:
    """Merge scalar Layer-16 moment summaries emitted by the loss code."""

    element_count: int = 0
    value_sum: float = 0.0
    value_sum_squares: float = 0.0
    token_count: int = 0
    token_l2_sum: float = 0.0
    token_l2_sum_squares: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    observed_shapes: set[tuple[int, ...]] = field(default_factory=set)
    tasks: list[str] = field(default_factory=list)

    def state_dict(self) -> dict[str, Any]:
        """Return the exact merge state required for interrupted-run resume."""

        return {
            "element_count": int(self.element_count),
            "value_sum": float(self.value_sum),
            "value_sum_squares": float(self.value_sum_squares),
            "token_count": int(self.token_count),
            "token_l2_sum": float(self.token_l2_sum),
            "token_l2_sum_squares": float(self.token_l2_sum_squares),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
            "observed_shapes": [
                list(shape) for shape in sorted(self.observed_shapes)
            ],
            "tasks": list(self.tasks),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DistributionAccumulator":
        """Strictly restore :meth:`state_dict` without accepting partial state."""

        required = {
            "element_count",
            "value_sum",
            "value_sum_squares",
            "token_count",
            "token_l2_sum",
            "token_l2_sum_squares",
            "minimum",
            "maximum",
            "observed_shapes",
            "tasks",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("distribution resume state schema differs")
        element_count = int(state["element_count"])
        token_count = int(state["token_count"])
        if element_count < 0 or token_count < 0:
            raise ValueError("distribution resume counts must be non-negative")
        floating = {
            key: float(state[key])
            for key in (
                "value_sum",
                "value_sum_squares",
                "token_l2_sum",
                "token_l2_sum_squares",
                "minimum",
                "maximum",
            )
        }
        if element_count > 0 and not all(math.isfinite(value) for value in floating.values()):
            raise ValueError("non-empty distribution resume state is non-finite")
        if element_count == 0 and not (
            math.isinf(floating["minimum"])
            and floating["minimum"] > 0
            and math.isinf(floating["maximum"])
            and floating["maximum"] < 0
        ):
            raise ValueError("empty distribution resume extrema differ")
        shapes = {
            tuple(int(dimension) for dimension in shape)
            for shape in state["observed_shapes"]
        }
        if any(not shape or any(dimension <= 0 for dimension in shape) for shape in shapes):
            raise ValueError("distribution resume shapes are invalid")
        tasks = [str(task) for task in state["tasks"]]
        return cls(
            element_count=element_count,
            value_sum=floating["value_sum"],
            value_sum_squares=floating["value_sum_squares"],
            token_count=token_count,
            token_l2_sum=floating["token_l2_sum"],
            token_l2_sum_squares=floating["token_l2_sum_squares"],
            minimum=floating["minimum"],
            maximum=floating["maximum"],
            observed_shapes=shapes,
            tasks=tasks,
        )

    def add(self, summary: Mapping[str, Any], *, tasks: Sequence[str]) -> None:
        required = {
            "shape",
            "element_count",
            "sum",
            "sum_squares",
            "token_count",
            "token_l2_sum",
            "token_l2_sum_squares",
            "minimum",
            "maximum",
        }
        if set(summary) != required:
            raise ValueError(
                f"distribution summary schema differs: {sorted(set(summary) ^ required)}"
            )
        self.element_count += int(summary["element_count"])
        self.value_sum += float(summary["sum"])
        self.value_sum_squares += float(summary["sum_squares"])
        self.token_count += int(summary["token_count"])
        self.token_l2_sum += float(summary["token_l2_sum"])
        self.token_l2_sum_squares += float(summary["token_l2_sum_squares"])
        self.minimum = min(self.minimum, float(summary["minimum"]))
        self.maximum = max(self.maximum, float(summary["maximum"]))
        self.observed_shapes.add(tuple(int(value) for value in summary["shape"]))
        self.tasks.extend(str(task) for task in tasks)

    @staticmethod
    def _moments(total: float, total_squares: float, count: int) -> tuple[float, float]:
        if count <= 0:
            raise ValueError("cannot finalize an empty distribution")
        mean = total / count
        variance = max(total_squares / count - mean * mean, 0.0)
        return mean, math.sqrt(variance)

    def finalize(self) -> dict[str, Any]:
        mean, std = self._moments(self.value_sum, self.value_sum_squares, self.element_count)
        norm_mean, norm_std = self._moments(
            self.token_l2_sum,
            self.token_l2_sum_squares,
            self.token_count,
        )
        return {
            "element_count": self.element_count,
            "mean": mean,
            "std": std,
            "token_count": self.token_count,
            "token_l2_mean": norm_mean,
            "token_l2_std": norm_std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "observed_shapes": [list(shape) for shape in sorted(self.observed_shapes)],
            "task_sequence": list(self.tasks),
            "task_set": sorted(set(self.tasks)),
        }


def compare_distributions(
    official: Mapping[str, Any], paired_clean: Mapping[str, Any]
) -> dict[str, float]:
    pooled_std = math.sqrt(
        (float(official["std"]) ** 2 + float(paired_clean["std"]) ** 2) / 2.0
    )
    return {
        "absolute_mean_gap": abs(float(official["mean"]) - float(paired_clean["mean"])),
        "standardized_mean_gap": abs(
            float(official["mean"]) - float(paired_clean["mean"])
        )
        / max(pooled_std, 1e-12),
        "std_ratio_official_over_paired": float(official["std"])
        / max(float(paired_clean["std"]), 1e-12),
        "token_l2_mean_gap": abs(
            float(official["token_l2_mean"])
            - float(paired_clean["token_l2_mean"])
        ),
    }


__all__ = [
    "DistributionAccumulator",
    "ParameterSnapshot",
    "SampledParameterSnapshot",
    "action_path_gradient_probe",
    "assert_no_parameter_gradients",
    "compare_distributions",
    "module_gradient_report",
]
