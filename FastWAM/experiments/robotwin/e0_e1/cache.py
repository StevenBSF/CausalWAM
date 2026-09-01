"""Validated on-disk representation cache schema."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data import (
    R3_HOLDOUT_PROTOCOL,
    R3_VARIANT,
    SEEN_VARIANTS,
    UNSEEN_TEST_VARIANTS,
    VARIANTS,
)
from .io_utils import atomic_torch_save, load_torch
from .metrics import RepresentationRecord


CACHE_SCHEMA_VERSION = 2
LEGACY_CACHE_SCHEMA_VERSION = 1
PROPRIO_MODES = ("observed", "constant_zero_normalized")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sample_variant_names(samples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not samples:
        raise ValueError("cache construction requires at least one physical sample")
    variants = tuple(str(value) for value in samples[0]["variant_names"])
    if (
        not variants
        or variants[0] != "clean"
        or len(set(variants)) != len(variants)
        or not set(variants) <= set(VARIANTS)
    ):
        raise ValueError(f"invalid cache variant order {variants}")
    for sample in samples:
        if tuple(str(value) for value in sample["variant_names"]) != variants:
            raise ValueError("samples use inconsistent variant orders")
    return variants


def _validate_r3_holdout_contract(
    *,
    records: Sequence[Mapping[str, Any]],
    variant_names: tuple[str, ...],
    provenance: Mapping[str, Any],
) -> None:
    if provenance.get("protocol") != R3_HOLDOUT_PROTOCOL:
        return
    split = str(provenance.get("split"))
    expected = SEEN_VARIANTS if split in ("train", "val") else UNSEEN_TEST_VARIANTS
    if split not in ("train", "val", "test"):
        raise ValueError(f"R3 holdout cache has invalid split {split!r}")
    if variant_names != expected:
        raise ValueError(
            f"R3 holdout {split} cache requires variants {expected}, got {variant_names}"
        )
    declared = tuple(str(value) for value in provenance.get("active_variants", ()))
    if declared != variant_names:
        raise ValueError("cache active_variants provenance disagrees with records")
    if provenance.get("holdout_variant") != R3_VARIANT:
        raise ValueError("cache holdout_variant provenance is not canonical R3")
    proprio_mode = str(provenance.get("proprio_mode"))
    if proprio_mode not in PROPRIO_MODES:
        raise ValueError(
            f"R3 holdout cache proprio_mode must be one of {PROPRIO_MODES}"
        )
    backbone = provenance.get("backbone")
    if isinstance(backbone, Mapping) and "proprio_mode" in backbone:
        if str(backbone["proprio_mode"]) != proprio_mode:
            raise ValueError("cache/backbone proprio_mode provenance disagrees")
    conditions = provenance.get("conditions_by_physical_state")
    if not isinstance(conditions, Mapping) or not conditions:
        raise ValueError("R3 holdout cache requires per-state condition provenance")
    expected_state_ids = {str(record.get("physical_state_id")) for record in records}
    if set(conditions) != expected_state_ids:
        raise ValueError("condition provenance keys do not match physical states")
    effective_hashes: set[str] = set()
    for state_id, condition in conditions.items():
        if not isinstance(condition, Mapping):
            raise ValueError(f"condition provenance for {state_id} is not a mapping")
        context = condition.get("context")
        proprio = context.get("proprio") if isinstance(context, Mapping) else None
        if not isinstance(proprio, Mapping):
            raise ValueError(f"condition proprio provenance for {state_id} is missing")
        if str(proprio.get("mode")) != proprio_mode:
            raise ValueError("condition/cache proprio_mode provenance disagrees")
        if proprio_mode == "constant_zero_normalized" and proprio.get("all_zero") is not True:
            raise ValueError("no-proprio condition is not proven exact zero")
        digest = condition.get("normalized_proprio_sha256")
        if not isinstance(digest, str):
            digest = proprio.get("effective_normalized_sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("effective proprio digest is not a SHA-256")
        if (
            proprio.get("effective_normalized_sha256") is not None
            and proprio.get("effective_normalized_sha256") != digest
        ):
            raise ValueError("outer/inner effective proprio hashes disagree")
        effective_hashes.add(digest)
    if proprio_mode == "constant_zero_normalized" and len(effective_hashes) != 1:
        raise ValueError(
            "no-proprio cache must use one identical effective proprio input"
        )
    record_variants = {str(record.get("variant")) for record in records}
    record_splits = {str(record.get("split")) for record in records}
    if record_splits != {split} or record_variants != set(expected):
        raise ValueError("R3 holdout records contradict split/variant provenance")
    if split in ("train", "val") and R3_VARIANT in record_variants:
        raise ValueError("R3 leaked into an R3 holdout train/validation cache")
    lock_identity = provenance.get("decision_lock_identity")
    lock_precedes_test = provenance.get("decision_lock_created_before_test")
    if split == "test":
        if not isinstance(lock_identity, Mapping) or not lock_identity:
            raise ValueError("R3 test cache is missing decision-lock identity")
        if lock_precedes_test is not True:
            raise ValueError("R3 test cache does not prove decision lock preceded extraction")
    elif lock_identity is not None or lock_precedes_test is not None:
        raise ValueError("train/validation cache must not carry a test decision lock")


def build_cache_payload(
    *,
    tokens_by_layer: Mapping[int, torch.Tensor],
    samples: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not tokens_by_layer:
        raise ValueError("at least one captured layer is required")
    variant_names = _sample_variant_names(samples)
    expected = len(samples) * len(variant_names)
    normalized_tokens: dict[str, torch.Tensor] = {}
    pooled: dict[str, torch.Tensor] = {}
    token_shape: tuple[int, int] | None = None
    for layer, tokens in sorted(tokens_by_layer.items()):
        if int(layer) <= 0 or tokens.ndim != 3 or tokens.shape[0] != expected:
            raise ValueError(
                f"layer {layer}: expected tokens [N,S,D] with N={expected}, got {tuple(tokens.shape)}"
            )
        if not torch.isfinite(tokens.float()).all():
            raise ValueError(f"layer {layer}: tokens contain NaN/inf")
        current_shape = (int(tokens.shape[1]), int(tokens.shape[2]))
        if token_shape is None:
            token_shape = current_shape
        elif current_shape != token_shape:
            raise ValueError("candidate layers expose inconsistent token shapes")
        cpu = tokens.detach().to(device="cpu", dtype=torch.float16).contiguous()
        key = str(int(layer))
        normalized_tokens[key] = cpu
        pooled[key] = F.normalize(cpu.float().mean(dim=1), p=2, dim=-1).contiguous()

    records: list[dict[str, Any]] = []
    physical_states: list[dict[str, float]] = []
    proprio_raw: list[torch.Tensor] = []
    visual_inputs: dict[str, dict[str, Any]] = {}
    for sample in samples:
        physical_key = str(sample["physical_key"])
        task = str(sample["task"])
        content_id = int(sample["content_id"])
        frame_idx = int(sample["frame_idx"])
        variants = tuple(str(value) for value in sample["variant_names"])
        if variants != variant_names:
            raise ValueError(f"unexpected variant order {variants}")
        state = {str(key): float(value) for key, value in sample["physical_state_by_name"].items()}
        physical_states.append(state)
        raw = torch.as_tensor(sample["proprio_raw"], dtype=torch.float32).reshape(-1)
        if raw.numel() != 14 or not torch.isfinite(raw).all():
            raise ValueError(f"{physical_key}: invalid 14-D proprio")
        proprio_raw.append(raw)
        visual = sample.get("visual_input_sha256")
        if visual is not None:
            if not isinstance(visual, Mapping) or set(visual) != set(variants):
                raise ValueError(f"{physical_key}: invalid visual input hashes")
            canonical_visual: dict[str, Any] = {}
            for variant in variants:
                item = visual[variant]
                if not isinstance(item, Mapping):
                    raise ValueError(f"{physical_key}/{variant}: visual hashes malformed")
                composite = item.get("deployment_composite")
                cameras = item.get("encoded_rgb_by_camera")
                if (
                    not isinstance(composite, str)
                    or _SHA256_RE.fullmatch(composite) is None
                    or not isinstance(cameras, Mapping)
                    or set(cameras) != {"head_camera", "left_camera", "right_camera"}
                    or any(
                        not isinstance(value, str)
                        or _SHA256_RE.fullmatch(value) is None
                        for value in cameras.values()
                    )
                ):
                    raise ValueError(f"{physical_key}/{variant}: visual hashes malformed")
                canonical_visual[variant] = {
                    "deployment_composite": composite,
                    "encoded_rgb_by_camera": {
                        str(key): str(value) for key, value in sorted(cameras.items())
                    },
                }
            visual_inputs[physical_key] = canonical_visual
        for variant in variants:
            records.append(
                {
                    "task": task,
                    "physical_state_id": physical_key,
                    "trajectory_id": f"{task}/content_{content_id:06d}",
                    "timestep": frame_idx,
                    "trace_idx": int(sample["trace_idx"]),
                    "content_id": content_id,
                    "variant": variant,
                    "split": str(sample["split"]),
                }
            )
    if len(records) != expected:
        raise AssertionError("record construction count mismatch")
    result = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "variant_names": variant_names,
        "variants_per_state": len(variant_names),
        "tokens_by_layer": normalized_tokens,
        "pooled_by_layer": pooled,
        "records": records,
        "physical_states": physical_states,
        "proprio_raw": torch.stack(proprio_raw),
        "visual_input_sha256_by_physical_state": visual_inputs,
        "provenance": dict(provenance),
    }
    _validate_r3_holdout_contract(
        records=records,
        variant_names=variant_names,
        provenance=result["provenance"],
    )
    return result


def validate_cache(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version not in (LEGACY_CACHE_SCHEMA_VERSION, CACHE_SCHEMA_VERSION):
        raise ValueError(f"unsupported cache schema {payload.get('schema_version')!r}")
    records = payload.get("records")
    tokens_by_layer = payload.get("tokens_by_layer")
    pooled_by_layer = payload.get("pooled_by_layer")
    if not isinstance(records, list) or not records:
        raise ValueError("cache records must be a non-empty list")
    if not isinstance(tokens_by_layer, Mapping) or not isinstance(pooled_by_layer, Mapping):
        raise ValueError("cache is missing layer tensors")
    if schema_version == LEGACY_CACHE_SCHEMA_VERSION:
        variant_names = VARIANTS
    else:
        raw_variant_names = payload.get("variant_names")
        if not isinstance(raw_variant_names, (list, tuple)):
            raise ValueError("cache is missing variant_names")
        variant_names = tuple(str(value) for value in raw_variant_names)
        if payload.get("variants_per_state") != len(variant_names):
            raise ValueError("cache variants_per_state disagrees with variant_names")
    if (
        not variant_names
        or variant_names[0] != "clean"
        or len(set(variant_names)) != len(variant_names)
        or not set(variant_names) <= set(VARIANTS)
    ):
        raise ValueError(f"invalid cache variant_names {variant_names}")
    for key, tokens in tokens_by_layer.items():
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] != len(records):
            raise ValueError(f"invalid token tensor at layer {key}")
        pooled = pooled_by_layer.get(key)
        if not isinstance(pooled, torch.Tensor) or pooled.shape != (len(records), tokens.shape[-1]):
            raise ValueError(f"invalid pooled tensor at layer {key}")
        if not torch.isfinite(tokens.float()).all() or not torch.isfinite(pooled).all():
            raise ValueError(f"non-finite cache tensor at layer {key}")
        torch.testing.assert_close(
            pooled.norm(dim=-1), torch.ones(len(records)), rtol=5e-3, atol=5e-3
        )
    if len(records) % len(variant_names) != 0:
        raise ValueError("cache record count is not divisible by variants_per_state")
    groups: dict[tuple[str, str], list[str]] = {}
    for record in records:
        key = (str(record["task"]), str(record["physical_state_id"]))
        groups.setdefault(key, []).append(str(record["variant"]))
    if any(tuple(variants) != variant_names for variants in groups.values()):
        raise ValueError(
            "cache has incomplete, duplicate, or non-canonical physical-state variant groups"
        )
    expected_group_count = len(records) // len(variant_names)
    if len(groups) != expected_group_count:
        raise ValueError("cache physical-state groups are not contiguous and unique")
    for group_index in range(expected_group_count):
        start = group_index * len(variant_names)
        group = records[start : start + len(variant_names)]
        keys = {
            (str(record["task"]), str(record["physical_state_id"]))
            for record in group
        }
        ordered = tuple(str(record["variant"]) for record in group)
        if len(keys) != 1 or ordered != variant_names:
            raise ValueError(
                f"cache physical-state group {group_index} is non-contiguous or misordered"
            )
    physical_states = payload.get("physical_states")
    proprio_raw = payload.get("proprio_raw")
    if not isinstance(physical_states, list) or len(physical_states) != expected_group_count:
        raise ValueError("cache physical_states count does not match variant groups")
    if (
        not isinstance(proprio_raw, torch.Tensor)
        or proprio_raw.shape != (expected_group_count, 14)
        or not torch.isfinite(proprio_raw).all()
    ):
        raise ValueError("cache proprio_raw must be finite [physical_states,14]")
    visual_inputs = payload.get("visual_input_sha256_by_physical_state")
    provenance = payload.get("provenance")
    strict_r3 = (
        isinstance(provenance, Mapping)
        and provenance.get("protocol") == R3_HOLDOUT_PROTOCOL
    )
    if strict_r3 and (not isinstance(visual_inputs, Mapping) or not visual_inputs):
        raise ValueError("R3 holdout cache requires exact visual input hashes")
    if schema_version == CACHE_SCHEMA_VERSION and visual_inputs is not None:
        if not isinstance(visual_inputs, Mapping):
            raise ValueError("cache visual input hashes must be a mapping")
        expected_keys = {
            str(records[index]["physical_state_id"])
            for index in range(0, len(records), len(variant_names))
        }
        if visual_inputs and set(visual_inputs) != expected_keys:
            raise ValueError("cache visual input hash keys do not match physical states")
    if not isinstance(provenance, Mapping):
        raise ValueError("cache provenance must be a mapping")
    _validate_r3_holdout_contract(
        records=records,
        variant_names=variant_names,
        provenance=provenance,
    )


def save_cache(path: str | Path, payload: Mapping[str, Any]) -> Path:
    validate_cache(payload)
    return atomic_torch_save(path, dict(payload))


def load_cache(path: str | Path) -> dict[str, Any]:
    payload = load_torch(path)
    if not isinstance(payload, dict):
        raise ValueError("cache payload must be a dictionary")
    validate_cache(payload)
    return payload


def representation_records(payload: Mapping[str, Any]) -> list[RepresentationRecord]:
    return [
        RepresentationRecord(
            task=str(record["task"]),
            physical_state_id=str(record["physical_state_id"]),
            trajectory_id=str(record["trajectory_id"]),
            timestep=int(record["timestep"]),
            variant=str(record["variant"]),
        )
        for record in payload["records"]
    ]


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "LEGACY_CACHE_SCHEMA_VERSION",
    "PROPRIO_MODES",
    "build_cache_payload",
    "load_cache",
    "representation_records",
    "save_cache",
    "validate_cache",
]
