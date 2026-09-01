#!/usr/bin/env python3
"""Train the E2/E3 content head under the strict R3-holdout protocol.

This module intentionally consumes frozen token caches.  It never imports or
updates the FastWAM backbone.  E2 and E3 share the same training code; the only
allowed experimental difference is the cache ``proprio_mode``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .cache import load_cache
from .head import ContrastiveContentHead, multi_positive_supcon_loss
from .io_utils import (
    atomic_torch_save,
    atomic_write_text,
    file_identity,
    module_state_sha256,
    write_csv,
    write_json,
)
from .negatives import build_state_negative_mask


PROTOCOL = "r3_holdout_v1"
ACTIVE_VARIANTS = (
    "clean",
    "style_00_seed_0",
    "style_01_seed_1",
)
HOLDOUT_VARIANT = "style_02_seed_2"
EXPERIMENT_PROPRIO_MODES = {
    "E2": "observed",
    "E3": "constant_zero_normalized",
}
CHECKPOINT_SCHEMA_VERSION = 2


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_identity(path: str | Path) -> dict[str, Any]:
    """Return a path/stat identity augmented with a content SHA-256."""

    before = file_identity(path)
    digest = _sha256_file(path)
    after = file_identity(path)
    if before != after:
        raise RuntimeError(f"cache changed while hashing: {Path(path).resolve()}")
    return {**after, "sha256": digest}


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _scientific_cache_contract(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the data fields that E2 and E3 must share exactly."""

    records = cache.get("records")
    states = cache.get("physical_states")
    proprio = cache.get("proprio_raw")
    if not isinstance(records, list) or not isinstance(states, list):
        raise ValueError("cache lacks records/physical_states for scientific contract")
    if not isinstance(proprio, torch.Tensor):
        raise ValueError("cache lacks proprio_raw for scientific contract")
    record_fields = (
        "task",
        "physical_state_id",
        "trajectory_id",
        "timestep",
        "trace_idx",
        "content_id",
        "variant",
        "split",
    )
    canonical_records = [
        {field: record[field] for field in record_fields} for record in records
    ]
    token_shapes = {
        str(layer): list(tokens.shape)
        for layer, tokens in sorted(cache["tokens_by_layer"].items())
    }
    provenance = cache.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("cache lacks provenance for scientific contract")
    source_manifest_sha256 = provenance.get("source_manifest_sha256")
    if (
        not isinstance(source_manifest_sha256, str)
        or len(source_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_manifest_sha256)
    ):
        raise ValueError("cache lacks a canonical source-manifest SHA-256")
    visual_inputs = cache.get("visual_input_sha256_by_physical_state")
    if not isinstance(visual_inputs, Mapping) or not visual_inputs:
        raise ValueError("cache lacks exact visual-input SHA-256 evidence")
    return {
        "variant_names": list(cache.get("variant_names", ())),
        "variants_per_state": int(cache.get("variants_per_state", -1)),
        "num_records": len(records),
        "num_physical_states": len(states),
        "physical_state_ids": [
            str(records[index]["physical_state_id"])
            for index in range(0, len(records), int(cache["variants_per_state"]))
        ],
        "records_sha256": _canonical_value_sha256(canonical_records),
        "physical_states_sha256": _canonical_value_sha256(states),
        "proprio_raw_sha256": _tensor_sha256(proprio),
        "proprio_raw_shape": list(proprio.shape),
        "token_shapes_by_layer": token_shapes,
        "source_manifest_sha256": source_manifest_sha256,
        "visual_inputs_sha256": _canonical_value_sha256(visual_inputs),
    }


def _write_training_curve_svg(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    best_step: int,
) -> Path:
    """Write a dependency-free train/validation loss curve as an SVG artifact."""

    train_points = [
        (int(row["step"]), float(row["train_contrastive_loss"])) for row in rows
    ]
    val_points = [
        (int(row["step"]), float(row["val_contrastive_loss"]))
        for row in rows
        if row.get("val_contrastive_loss") is not None
    ]
    if not train_points or not val_points:
        raise ValueError("training curve requires non-empty train and validation values")
    values = [value for _, value in train_points + val_points]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("training curve contains non-finite loss values")

    width, height = 960, 500
    left, right, top, bottom = 82, 30, 48, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    min_step = min(step for step, _ in train_points)
    max_step = max(step for step, _ in train_points)
    min_value = min(0.0, min(values))
    max_value = max(values)
    if max_value <= min_value:
        max_value = min_value + 1.0

    def xy(step: int, value: float) -> tuple[float, float]:
        x = left + (step - min_step) / max(1, max_step - min_step) * plot_width
        y = top + (max_value - value) / (max_value - min_value) * plot_height
        return x, y

    def polyline(points: Sequence[tuple[int, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(*point) for point in points))

    best_values = [value for step, value in val_points if step == best_step]
    if len(best_values) != 1:
        raise ValueError("best validation step is absent or duplicated in curve rows")
    best_x, best_y = xy(best_step, best_values[0])
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="sans-serif" font-size="18">{experiment} train/validation contrastive loss</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>
  <text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">training step</text>
  <text x="19" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 19 {height / 2:.1f})" font-family="sans-serif" font-size="14">SupCon loss</text>
  <text x="{left - 9}" y="{top + 5}" text-anchor="end" font-family="monospace" font-size="12">{max_value:.5g}</text>
  <text x="{left - 9}" y="{top + plot_height + 5}" text-anchor="end" font-family="monospace" font-size="12">{min_value:.5g}</text>
  <text x="{left}" y="{top + plot_height + 22}" text-anchor="middle" font-family="monospace" font-size="12">{min_step}</text>
  <text x="{left + plot_width}" y="{top + plot_height + 22}" text-anchor="middle" font-family="monospace" font-size="12">{max_step}</text>
  <polyline points="{polyline(train_points)}" fill="none" stroke="#2563eb" stroke-width="1.7"/>
  <polyline points="{polyline(val_points)}" fill="none" stroke="#dc2626" stroke-width="2.2"/>
  <line x1="{best_x:.2f}" y1="{top}" x2="{best_x:.2f}" y2="{top + plot_height}" stroke="#16a34a" stroke-width="1.3" stroke-dasharray="6 5"/>
  <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="5" fill="#16a34a"/>
  <text x="{left + 15}" y="{top + 22}" font-family="sans-serif" font-size="13" fill="#2563eb">train</text>
  <text x="{left + 75}" y="{top + 22}" font-family="sans-serif" font-size="13" fill="#dc2626">validation</text>
  <text x="{left + 162}" y="{top + 22}" font-family="sans-serif" font-size="13" fill="#16a34a">best-val step={best_step}</text>
</svg>
"""
    return atomic_write_text(path, svg)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return copy.deepcopy(value)


def _backbone_semantics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("cache provenance backbone must be a mapping")
    semantics = dict(value)
    semantics.pop("native_prefill_verified", None)
    return semantics


def _require_exact_training_protocol(
    cache: Mapping[str, Any],
    *,
    split: str,
    experiment: str,
    proprio_mode: str,
) -> None:
    if split not in {"train", "val"}:
        raise ValueError(f"E2/E3 training cache split must be train/val, got {split!r}")
    expected_mode = EXPERIMENT_PROPRIO_MODES.get(experiment)
    if expected_mode is None:
        raise ValueError(f"experiment must be one of {tuple(EXPERIMENT_PROPRIO_MODES)}")
    if proprio_mode != expected_mode:
        raise ValueError(
            f"{experiment} requires proprio_mode={expected_mode!r}, got {proprio_mode!r}"
        )

    provenance = cache.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("cache is missing provenance")
    if provenance.get("protocol") != PROTOCOL:
        raise ValueError(f"cache protocol must be {PROTOCOL!r}")
    if provenance.get("split") != split:
        raise ValueError(f"cache provenance split must be {split!r}")
    if tuple(provenance.get("active_variants", ())) != ACTIVE_VARIANTS:
        raise ValueError(
            f"{split} cache active_variants must be exactly {ACTIVE_VARIANTS}"
        )
    if provenance.get("holdout_variant") != HOLDOUT_VARIANT:
        raise ValueError(f"cache holdout_variant must be {HOLDOUT_VARIANT!r}")
    if provenance.get("proprio_mode") != proprio_mode:
        raise ValueError("cache proprio_mode does not match the requested experiment")
    if tuple(cache.get("variant_names", ())) != ACTIVE_VARIANTS:
        raise ValueError(f"cache variant_names must be exactly {ACTIVE_VARIANTS}")
    if int(cache.get("variants_per_state", -1)) != len(ACTIVE_VARIANTS):
        raise ValueError("cache variants_per_state must be exactly three")

    records = cache.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("cache records must be a non-empty list")
    record_variants = {str(record.get("variant")) for record in records}
    if HOLDOUT_VARIANT in record_variants:
        raise ValueError(f"R3 holdout leaked into {split} cache records")
    if record_variants != set(ACTIVE_VARIANTS):
        raise ValueError(
            f"{split} record variants must be exactly {ACTIVE_VARIANTS}, "
            f"got {sorted(record_variants)}"
        )
    record_splits = {str(record.get("split")) for record in records}
    if record_splits != {split}:
        raise ValueError(f"cache records do not exclusively belong to split {split!r}")


def _groups_and_states(
    cache: Mapping[str, Any],
) -> tuple[dict[str, list[list[int]]], list[Mapping[str, float]]]:
    """Validate canonical K-view groups and expand state rows to records."""

    records = cache["records"]
    physical_states = cache.get("physical_states")
    if not isinstance(physical_states, list) or not physical_states:
        raise ValueError("cache physical_states must be a non-empty list")
    views_per_state = len(ACTIVE_VARIANTS)
    if len(records) != len(physical_states) * views_per_state:
        raise ValueError("cache records/physical_states do not form three-view groups")

    groups_by_task: dict[str, list[list[int]]] = defaultdict(list)
    state_by_record: list[Mapping[str, float]] = []
    seen_keys: set[tuple[str, str]] = set()
    for group_index, state in enumerate(physical_states):
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"physical state {group_index} is empty or malformed")
        start = group_index * views_per_state
        indices = list(range(start, start + views_per_state))
        group = [records[index] for index in indices]
        variants = tuple(str(record.get("variant")) for record in group)
        if variants != ACTIVE_VARIANTS:
            raise ValueError(
                f"cache group {group_index} does not use canonical three-view order"
            )
        keys = {
            (str(record.get("task")), str(record.get("physical_state_id")))
            for record in group
        }
        if len(keys) != 1:
            raise ValueError(f"cache group {group_index} is not one physical state")
        key = next(iter(keys))
        if key in seen_keys:
            raise ValueError(f"duplicate physical-state group {key}")
        seen_keys.add(key)
        groups_by_task[key[0]].append(indices)
        state_by_record.extend([state] * views_per_state)
    if any(len(groups) < 2 for groups in groups_by_task.values()):
        raise ValueError("every task needs at least two physical-state groups")
    return dict(groups_by_task), state_by_record


def _batch_indices(
    groups_by_task: Mapping[str, Sequence[Sequence[int]]],
    *,
    groups_per_batch: int,
    generator: random.Random,
) -> list[int]:
    task = generator.choice(sorted(groups_by_task))
    groups = list(groups_by_task[task])
    if groups_per_batch > len(groups):
        raise ValueError("--groups-per-batch exceeds a task's unique train states")
    selected = generator.sample(groups, groups_per_batch)
    indices = [int(index) for group in selected for index in group]
    if len(indices) != groups_per_batch * len(ACTIVE_VARIANTS):
        raise AssertionError("three-view batch cardinality mismatch")
    if len(set(indices)) != len(indices):
        raise ValueError("batch repeated a physical state")
    return indices


def _negative_mask(
    cache: Mapping[str, Any],
    state_by_record: Sequence[Mapping[str, float]],
    indices: Sequence[int],
    *,
    min_temporal_gap: int,
    min_state_distance: float,
) -> torch.Tensor:
    records = [cache["records"][index] for index in indices]
    states = [state_by_record[index] for index in indices]
    return torch.from_numpy(
        build_state_negative_mask(
            records,
            states,
            min_temporal_gap=min_temporal_gap,
            min_state_distance=min_state_distance,
        )
    )


@torch.no_grad()
def _diagnostics(
    embeddings: torch.Tensor,
    state_ids: Sequence[str],
    negative_mask: torch.Tensor,
) -> dict[str, float]:
    normalized = F.normalize(embeddings.float(), dim=-1)
    similarity = normalized @ normalized.T
    count = len(state_ids)
    equal = torch.tensor(
        [[left == right for right in state_ids] for left in state_ids],
        dtype=torch.bool,
        device=similarity.device,
    )
    not_self = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    positive = similarity[equal & not_self]
    selected_negative = negative_mask.to(similarity.device)
    if positive.numel() == 0 or not bool(selected_negative.any().item()):
        raise ValueError("diagnostic batch lacks positive or negative pairs")
    negative = similarity[selected_negative]
    result = {
        "positive": float(positive.mean().item()),
        "negative": float(negative.mean().item()),
        "norm": float(embeddings.float().norm(dim=-1).mean().item()),
        "spread": float(1.0 - negative.mean().item()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite representation diagnostics")
    if result["spread"] <= 1e-7:
        raise FloatingPointError(
            f"representation collapse detected: state spread={result['spread']}"
        )
    return result


def _is_better_validation(
    value: float, step: int, *, best_value: float | None, best_step: int | None
) -> bool:
    """Minimize validation loss, resolving exact ties to the earlier step."""

    if not math.isfinite(value) or step <= 0:
        raise ValueError("validation candidate must be finite and have a positive step")
    if best_value is None:
        return True
    if best_step is None:
        raise ValueError("best_step is required when best_value is present")
    return value < best_value or (value == best_value and step < best_step)


def _controlled_training_config(
    *,
    layer: int,
    head_config: Mapping[str, int],
    steps: int,
    groups_per_batch: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    val_every: int,
    seed: int,
    min_temporal_gap: int,
    min_state_distance: float,
) -> dict[str, Any]:
    """Return fields that must be exactly equal between E2 and E3."""

    return {
        "protocol": PROTOCOL,
        "active_variants": list(ACTIVE_VARIANTS),
        "holdout_variant": HOLDOUT_VARIANT,
        "layer": int(layer),
        "head_config": dict(head_config),
        "steps": int(steps),
        "groups_per_batch": int(groups_per_batch),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "loss": {
            "name": "multi_positive_supcon",
            "temperature": float(temperature),
            "positive_views_per_state": len(ACTIVE_VARIANTS),
        },
        "val_every": int(val_every),
        "seed": int(seed),
        "min_temporal_gap": int(min_temporal_gap),
        "min_state_distance": float(min_state_distance),
        "token_precision": "float32",
        "checkpoint_selection": {
            "metric": "val_contrastive_loss",
            "mode": "min",
            "tie_break": "earliest_step",
            "r3_allowed": False,
        },
    }


def train_e2e3_head(
    *,
    experiment: str,
    proprio_mode: str,
    train_cache_path: str | Path,
    val_cache_path: str | Path,
    layer: int,
    output_dir: str | Path,
    steps: int = 1000,
    groups_per_batch: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    temperature: float = 0.07,
    val_every: int = 50,
    seed: int = 0,
    device: str = "cuda",
    min_temporal_gap: int = 8,
    min_state_distance: float = 1e-5,
) -> Path:
    """Train one E2/E3 head and return the selected best-val checkpoint."""

    if steps <= 0 or groups_per_batch < 2 or val_every <= 0:
        raise ValueError("steps/val_every must be positive and groups_per_batch >= 2")
    if learning_rate <= 0 or weight_decay < 0 or temperature <= 0:
        raise ValueError("learning_rate/temperature must be positive and weight_decay non-negative")
    if min_temporal_gap < 0 or min_state_distance < 0:
        raise ValueError("negative thresholds must be non-negative")
    expected_mode = EXPERIMENT_PROPRIO_MODES.get(experiment)
    if expected_mode is None or proprio_mode != expected_mode:
        raise ValueError(
            f"experiment/proprio_mode must be one of {EXPERIMENT_PROPRIO_MODES}"
        )

    train_identity = _cache_identity(train_cache_path)
    val_identity = _cache_identity(val_cache_path)
    train_cache = load_cache(train_cache_path)
    val_cache = load_cache(val_cache_path)
    _require_exact_training_protocol(
        train_cache,
        split="train",
        experiment=experiment,
        proprio_mode=proprio_mode,
    )
    _require_exact_training_protocol(
        val_cache,
        split="val",
        experiment=experiment,
        proprio_mode=proprio_mode,
    )

    train_tasks = {str(record["task"]) for record in train_cache["records"]}
    val_tasks = {str(record["task"]) for record in val_cache["records"]}
    if train_tasks != val_tasks:
        raise ValueError("train/validation task sets differ")
    train_trajectories = {
        (str(record["task"]), str(record["trajectory_id"]))
        for record in train_cache["records"]
    }
    val_trajectories = {
        (str(record["task"]), str(record["trajectory_id"]))
        for record in val_cache["records"]
    }
    leakage = train_trajectories & val_trajectories
    if leakage:
        raise ValueError(f"train/validation physical-trajectory leakage: {sorted(leakage)[:5]}")

    train_provenance = train_cache["provenance"]
    val_provenance = val_cache["provenance"]
    for key in ("backbone", "task_prompt_sha256"):
        if key not in train_provenance or key not in val_provenance:
            raise ValueError(f"train/validation provenance is missing {key}")
        train_value = train_provenance[key]
        val_value = val_provenance[key]
        if key == "backbone":
            train_value = _backbone_semantics(train_value)
            val_value = _backbone_semantics(val_value)
        if train_value != val_value:
            raise ValueError(f"train/validation cache {key} provenance differs")

    layer_key = str(int(layer))
    if (
        layer_key not in train_cache["tokens_by_layer"]
        or layer_key not in val_cache["tokens_by_layer"]
    ):
        raise KeyError(f"layer {layer} is absent from train or validation cache")
    train_tokens = train_cache["tokens_by_layer"][layer_key].detach()
    val_tokens = val_cache["tokens_by_layer"][layer_key].detach()
    if train_tokens.shape[1:] != val_tokens.shape[1:]:
        raise ValueError("train/validation token shapes differ")
    if train_tokens.requires_grad or val_tokens.requires_grad:
        raise AssertionError("frozen cached tokens unexpectedly require gradients")

    train_groups, train_states_by_record = _groups_and_states(train_cache)
    val_groups, val_states_by_record = _groups_and_states(val_cache)
    if any(groups_per_batch > len(groups) for groups in train_groups.values()):
        raise ValueError("--groups-per-batch exceeds a task's unique train states")

    torch.manual_seed(seed)
    generator = random.Random(seed)
    execution_device = torch.device(device)
    head_config = {
        "backbone_dim": int(train_tokens.shape[-1]),
        "embed_dim": 384,
        "num_queries": 8,
        "num_heads": 8,
    }
    head = ContrastiveContentHead(**head_config).to(execution_device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    initial_head_sha256 = module_state_sha256(head)
    controlled_config = _controlled_training_config(
        layer=layer,
        head_config=head_config,
        steps=steps,
        groups_per_batch=groups_per_batch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        val_every=val_every,
        seed=seed,
        min_temporal_gap=min_temporal_gap,
        min_state_distance=min_state_distance,
    )
    controlled_config_sha256 = _canonical_json_sha256(controlled_config)
    training_config = {
        **controlled_config,
        "experiment": experiment,
        "proprio_mode": proprio_mode,
        "device": str(execution_device),
        "train_cache": str(Path(train_cache_path).expanduser().resolve()),
        "val_cache": str(Path(val_cache_path).expanduser().resolve()),
        "train_cache_identity": train_identity,
        "val_cache_identity": val_identity,
    }
    train_scientific_contract = _scientific_cache_contract(train_cache)
    val_scientific_contract = _scientific_cache_contract(val_cache)

    @torch.no_grad()
    def validate_all_tasks() -> dict[str, float]:
        head.eval()
        task_values: list[dict[str, float]] = []
        for task in sorted(val_groups):
            task_indices = [index for group in val_groups[task] for index in group]
            state_ids = [
                str(val_cache["records"][index]["physical_state_id"])
                for index in task_indices
            ]
            task_ids = [task] * len(task_indices)
            negative_mask = _negative_mask(
                val_cache,
                val_states_by_record,
                task_indices,
                min_temporal_gap=min_temporal_gap,
                min_state_distance=min_state_distance,
            ).to(execution_device)
            chunk_size = groups_per_batch * len(ACTIVE_VARIANTS)
            chunks = [
                head(
                    val_tokens.index_select(
                        0, torch.tensor(task_indices[start : start + chunk_size])
                    ).to(execution_device, dtype=torch.float32)
                )
                for start in range(0, len(task_indices), chunk_size)
            ]
            embeddings = torch.cat(chunks, dim=0)
            loss = multi_positive_supcon_loss(
                embeddings,
                state_ids,
                task_ids,
                temperature=temperature,
                negative_mask=negative_mask,
            )
            task_values.append(
                {
                    "loss": float(loss.item()),
                    **_diagnostics(embeddings, state_ids, negative_mask),
                }
            )
        return {
            "loss": sum(row["loss"] for row in task_values) / len(task_values),
            "positive": sum(row["positive"] for row in task_values) / len(task_values),
            "negative": sum(row["negative"] for row in task_values) / len(task_values),
            "norm": sum(row["norm"] for row in task_values) / len(task_values),
            "spread": min(row["spread"] for row in task_values),
        }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    slug = experiment.lower()
    log_rows: list[dict[str, Any]] = []
    best_value: float | None = None
    best_step: int | None = None
    best_head: dict[str, torch.Tensor] | None = None
    best_optimizer: dict[str, Any] | None = None

    for step in range(1, steps + 1):
        indices = _batch_indices(
            train_groups, groups_per_batch=groups_per_batch, generator=generator
        )
        tokens = train_tokens.index_select(0, torch.tensor(indices)).to(
            execution_device, dtype=torch.float32
        )
        state_ids = [
            str(train_cache["records"][index]["physical_state_id"])
            for index in indices
        ]
        task_ids = [str(train_cache["records"][index]["task"]) for index in indices]
        negative_mask = _negative_mask(
            train_cache,
            train_states_by_record,
            indices,
            min_temporal_gap=min_temporal_gap,
            min_state_distance=min_state_distance,
        ).to(execution_device)

        head.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = head(tokens)
        loss = multi_positive_supcon_loss(
            embeddings,
            state_ids,
            task_ids,
            temperature=temperature,
            negative_mask=negative_mask,
        )
        loss.backward()
        gradients = [
            parameter.grad.detach().float().norm().square()
            for parameter in head.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("content head received no gradients")
        gradient_norm = float(torch.stack(gradients).sum().sqrt().item())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise FloatingPointError(f"invalid content-head gradient norm {gradient_norm}")
        optimizer.step()

        head.eval()
        with torch.no_grad():
            updated_embeddings = head(tokens)
        train_diagnostics = _diagnostics(updated_embeddings, state_ids, negative_mask)
        row: dict[str, Any] = {
            "step": step,
            "train_contrastive_loss": float(loss.detach().item()),
            "train_positive_similarity": train_diagnostics["positive"],
            "train_negative_similarity": train_diagnostics["negative"],
            "embedding_norm": train_diagnostics["norm"],
            "embedding_state_spread": train_diagnostics["spread"],
            "gradient_norm": gradient_norm,
            "val_contrastive_loss": None,
            "val_positive_similarity": None,
            "val_negative_similarity": None,
            "val_embedding_norm": None,
            "val_embedding_state_spread": None,
            "is_best": False,
        }
        if step == 1 or step % val_every == 0 or step == steps:
            val = validate_all_tasks()
            row.update(
                {
                    "val_contrastive_loss": val["loss"],
                    "val_positive_similarity": val["positive"],
                    "val_negative_similarity": val["negative"],
                    "val_embedding_norm": val["norm"],
                    "val_embedding_state_spread": val["spread"],
                }
            )
            if _is_better_validation(
                val["loss"], step, best_value=best_value, best_step=best_step
            ):
                best_value = float(val["loss"])
                best_step = step
                best_head = _cpu_tree(head.state_dict())
                best_optimizer = _cpu_tree(optimizer.state_dict())
                row["is_best"] = True
            print(
                f"{experiment} step={step}/{steps} "
                f"train_loss={row['train_contrastive_loss']:.6f} "
                f"val_loss={row['val_contrastive_loss']:.6f} "
                f"best_step={best_step}",
                flush=True,
            )
        log_rows.append(row)

    if best_value is None or best_step is None or best_head is None or best_optimizer is None:
        raise AssertionError("training completed without a best validation checkpoint")
    final_head_sha256 = module_state_sha256(head)
    if final_head_sha256 == initial_head_sha256:
        raise RuntimeError("optimizer completed without changing the content head")
    if file_identity(train_cache_path) != {
        key: train_identity[key] for key in ("path", "size_bytes", "mtime_ns")
    }:
        raise RuntimeError("train cache changed while the content head was training")
    if file_identity(val_cache_path) != {
        key: val_identity[key] for key in ("path", "size_bytes", "mtime_ns")
    }:
        raise RuntimeError("validation cache changed while the content head was training")

    selection = {
        "metric": "val_contrastive_loss",
        "mode": "min",
        "tie_break": "earliest_step",
        "best_step": int(best_step),
        "best_value": float(best_value),
        "r3_used": False,
    }
    common = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment": experiment,
        "protocol": PROTOCOL,
        "proprio_mode": proprio_mode,
        "layer": int(layer),
        "training_steps": int(steps),
        "best_step": int(best_step),
        "best_metric": selection,
        "head_config": head_config,
        "trainable_parameter_count": head.trainable_parameter_count(),
        "temperature": float(temperature),
        "negative_filter": {
            "min_temporal_gap": int(min_temporal_gap),
            "min_state_distance": float(min_state_distance),
        },
        "seed": int(seed),
        "initial_head_sha256": initial_head_sha256,
        "controlled_training_config": controlled_config,
        "controlled_training_config_sha256": controlled_config_sha256,
        "training_config": training_config,
        "train_cache": training_config["train_cache"],
        "val_cache": training_config["val_cache"],
        "train_cache_identity": train_identity,
        "val_cache_identity": val_identity,
        "train_scientific_cache_contract": train_scientific_contract,
        "val_scientific_cache_contract": val_scientific_contract,
    }
    best_checkpoint = {
        **common,
        "checkpoint_kind": "best_val",
        "step": int(best_step),
        "head": best_head,
        "optimizer": best_optimizer,
    }
    final_checkpoint = {
        **common,
        "checkpoint_kind": "final",
        "step": int(steps),
        "head": _cpu_tree(head.state_dict()),
        "optimizer": _cpu_tree(optimizer.state_dict()),
    }
    best_path = atomic_torch_save(
        destination / f"{slug}_best_content_head.pt", best_checkpoint
    )
    final_path = atomic_torch_save(
        destination / f"{slug}_final_content_head.pt", final_checkpoint
    )
    write_json(destination / "train_log.json", log_rows)
    write_csv(destination / "train_log.csv", log_rows)
    curve_path = _write_training_curve_svg(
        destination / "training_curves.svg",
        log_rows,
        experiment=experiment,
        best_step=best_step,
    )
    write_json(
        destination / "training_summary.json",
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "experiment": experiment,
            "protocol": PROTOCOL,
            "proprio_mode": proprio_mode,
            "selected_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "best_step": int(best_step),
            "best_val_contrastive_loss": float(best_value),
            "training_curves": str(curve_path),
            "selection": selection,
            "initial_head_sha256": initial_head_sha256,
            "controlled_training_config_sha256": controlled_config_sha256,
            "training_config": training_config,
        },
    )
    return best_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_PROPRIO_MODES), required=True)
    parser.add_argument(
        "--proprio-mode",
        choices=tuple(EXPERIMENT_PROPRIO_MODES.values()),
        required=True,
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-temporal-gap", type=int, default=8)
    parser.add_argument("--min-state-distance", type=float, default=1e-5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint = train_e2e3_head(
        experiment=args.experiment,
        proprio_mode=args.proprio_mode,
        train_cache_path=args.train_cache,
        val_cache_path=args.val_cache,
        layer=args.layer,
        output_dir=args.output_dir,
        steps=args.steps,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        val_every=args.val_every,
        seed=args.seed,
        device=args.device,
        min_temporal_gap=args.min_temporal_gap,
        min_state_distance=args.min_state_distance,
    )
    print(f"saved selected best-val head: {checkpoint}")


if __name__ == "__main__":
    main()


__all__ = [
    "ACTIVE_VARIANTS",
    "EXPERIMENT_PROPRIO_MODES",
    "HOLDOUT_VARIANT",
    "PROTOCOL",
    "_is_better_validation",
    "train_e2e3_head",
]
