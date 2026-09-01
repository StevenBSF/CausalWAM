#!/usr/bin/env python3
"""Fail-closed artifact validators for the unattended E0/E1 runner.

These checks deliberately reload artifacts instead of trusting ``*.done``
markers.  A successful command means that the artifact is complete and still
belongs to the exact upstream cache/configuration used by this run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .cache import load_cache
from .data import VARIANTS
from .head import ContrastiveContentHead
from .io_utils import file_identity, load_torch, write_json
from .metrics import RESULT_COLUMNS


SCHEMA_VERSION = 1
EXPERIMENTS = ("E0-RawBackbone", "E1-InitHead", "E1-TrainedHead")
SPLIT_CONTENT_IDS = {
    "train": tuple(range(0, 30)),
    "val": tuple(range(30, 40)),
    "test": tuple(range(40, 50)),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RunnerArtifactError(ValueError):
    """An artifact cannot safely be reused by the unattended runner."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerArtifactError(message)


def _read_json(path_value: str | Path) -> tuple[Path, Any]:
    path = Path(path_value).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerArtifactError(f"cannot read JSON {path}: {error}") from error
    return path, value


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique comma-separated values")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique positive comma-separated integers")
    return result


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RunnerArtifactError(f"{field} must be numeric") from error
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _identity(path: str | Path) -> dict[str, Any]:
    try:
        return file_identity(path)
    except OSError as error:
        raise RunnerArtifactError(f"cannot identify artifact {path}: {error}") from error


def _same_identity(actual: Any, path: str | Path, field: str) -> None:
    _require(isinstance(actual, Mapping), f"{field} is missing")
    _require(dict(actual) == _identity(path), f"{field} does not match {Path(path).resolve()}")


def _check_sha256(value: Any, field: str) -> str:
    text = str(value)
    _require(bool(_SHA256_RE.fullmatch(text)), f"{field} must be a lowercase SHA-256")
    return text


def init_config(path: str | Path, config: Mapping[str, Any]) -> Path:
    """Atomically create a run config, or require exact equality on resume."""

    destination = Path(path).expanduser().resolve()
    canonical = dict(config)
    # Paths alone do not make an unattended resume safe: a checkpoint or
    # normalization file may be replaced in-place.  Persist the same immutable
    # identity tuple used for cache lineage and compare it on every invocation.
    for field in ("checkpoint", "dataset_stats"):
        if field in canonical:
            canonical[f"{field}_identity"] = _identity(canonical[field])
    canonical["schema_version"] = SCHEMA_VERSION
    if destination.exists():
        _, previous = _read_json(destination)
        _require(isinstance(previous, Mapping), "existing run_config.json is not an object")
        _require(dict(previous) == canonical, "resume configuration differs from run_config.json")
        return destination
    return write_json(destination, canonical)


def validate_cache_artifact(
    cache_path: str | Path,
    *,
    split: str,
    tasks: Sequence[str],
    layers: Sequence[int],
    states_per_trajectory: int,
    expected_trajectories_per_task: int | None = None,
) -> dict[str, Any]:
    """Validate a complete formal extraction cache and its cardinalities."""

    _require(split in SPLIT_CONTENT_IDS, f"unsupported split {split!r}")
    _require(states_per_trajectory > 0, "states_per_trajectory must be positive")
    expected_content_ids = SPLIT_CONTENT_IDS[split]
    expected_trajectories = (
        len(expected_content_ids)
        if expected_trajectories_per_task is None
        else int(expected_trajectories_per_task)
    )
    _require(expected_trajectories > 0, "expected trajectory count must be positive")
    _require(
        expected_trajectories == len(expected_content_ids),
        "formal cache trajectory count must be 30 for train and 10 for val/test",
    )
    expected_tasks = tuple(tasks)
    expected_layers = tuple(int(layer) for layer in layers)
    cache = load_cache(cache_path)
    records = cache["records"]
    provenance = cache.get("provenance")
    _require(isinstance(provenance, Mapping), "cache provenance is missing")
    _require(provenance.get("split") == split, "cache provenance split mismatch")
    _require(tuple(provenance.get("tasks", ())) == expected_tasks, "cache provenance tasks mismatch")
    _require(
        provenance.get("states_per_trajectory") == states_per_trajectory,
        "cache provenance states_per_trajectory mismatch",
    )
    _require(provenance.get("allow_incomplete") is False, "formal cache used allow_incomplete")
    _require(
        provenance.get("max_trajectories_per_task") is None,
        "formal cache used max_trajectories_per_task",
    )
    _require(provenance.get("content_ids") is None, "formal cache used explicit content_ids")
    for field in ("manifest_jsonl", "manifest_csv"):
        manifest = provenance.get(field)
        _require(isinstance(manifest, str) and Path(manifest).is_file(), f"cache {field} is missing")
    _require(
        set(cache["tokens_by_layer"]) == {str(layer) for layer in expected_layers},
        "cache layer set mismatch",
    )
    backbone = provenance.get("backbone")
    _require(isinstance(backbone, Mapping), "cache backbone provenance is missing")
    _require(
        tuple(backbone.get("capture_layers", ())) == expected_layers,
        "backbone capture_layers mismatch",
    )
    _require(backbone.get("uses_future_video") is False, "cache used future video")
    _require(backbone.get("uses_action_denoising") is False, "cache used action denoising")
    _require(backbone.get("uses_policy_rollout") is False, "cache used a policy rollout")

    expected_states_per_task = expected_trajectories * states_per_trajectory
    expected_states = len(expected_tasks) * expected_states_per_task
    expected_records = expected_states * len(VARIANTS)
    _require(len(records) == expected_records, f"expected {expected_records} records, got {len(records)}")
    _require(
        len(cache.get("physical_states", ())) == expected_states,
        "cache physical-state count mismatch",
    )
    proprio = cache.get("proprio_raw")
    _require(
        isinstance(proprio, torch.Tensor) and tuple(proprio.shape) == (expected_states, 14),
        "cache proprio_raw count/shape mismatch",
    )
    _require(bool(torch.isfinite(proprio).all()), "cache proprio_raw contains NaN/inf")

    group_variants: dict[tuple[str, str], list[str]] = defaultdict(list)
    trajectory_states: dict[tuple[str, str], set[str]] = defaultdict(set)
    content_ids: dict[str, set[int]] = defaultdict(set)
    record_counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"cache record {index} is not an object")
        task = str(record.get("task"))
        _require(task in expected_tasks, f"unexpected cache task {task!r}")
        _require(record.get("split") == split, f"cache record {index} split mismatch")
        state_id = str(record.get("physical_state_id"))
        trajectory_id = str(record.get("trajectory_id"))
        content_id = int(record.get("content_id"))
        group_variants[(task, state_id)].append(str(record.get("variant")))
        trajectory_states[(task, trajectory_id)].add(state_id)
        content_ids[task].add(content_id)
        record_counts[task] += 1
    for key, variants in group_variants.items():
        _require(tuple(variants) == VARIANTS, f"state {key} does not contain canonical four variants")
    for key, state_ids in trajectory_states.items():
        _require(
            len(state_ids) == states_per_trajectory,
            f"trajectory {key} has {len(state_ids)} states, expected {states_per_trajectory}",
        )
    for task in expected_tasks:
        _require(
            content_ids[task] == set(expected_content_ids),
            f"task {task} content IDs do not exactly cover formal {split} split",
        )
        _require(
            record_counts[task] == expected_states_per_task * len(VARIANTS),
            f"task {task} record count mismatch",
        )
    conditions = provenance.get("conditions_by_physical_state")
    _require(
        isinstance(conditions, Mapping) and set(conditions) == {key[1] for key in group_variants},
        "condition provenance does not exactly cover physical states",
    )
    prompts = provenance.get("task_prompt_sha256")
    _require(isinstance(prompts, Mapping) and set(prompts) == set(expected_tasks), "prompt hashes mismatch")
    for task, digest in prompts.items():
        _check_sha256(digest, f"task prompt hash for {task}")
    return cache


def _metric_task_set(payload: Mapping[str, Any], *, source: Path) -> set[str]:
    rows = payload.get("metrics")
    _require(isinstance(rows, list) and rows, f"{source}: metrics must be non-empty")
    task_names: list[str] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"{source}: metric row {index} is not an object")
        missing = set(RESULT_COLUMNS) - set(row)
        _require(not missing, f"{source}: metric row {index} lacks {sorted(missing)}")
        for field in RESULT_COLUMNS[3:]:
            value = _finite(row[field], f"{source}: row {index} {field}")
            if field.startswith("retrieval_"):
                _require(0.0 <= value <= 1.0, f"{source}: {field} is outside [0,1]")
        task_names.append(str(row["task"]))
    macro_names = [task for task in task_names if task.endswith("-task-average")]
    ordinary = [task for task in task_names if not task.endswith("-task-average")]
    _require(len(ordinary) == len(set(ordinary)), f"{source}: duplicate task metric")
    _require(
        macro_names == [f"{len(ordinary)}-task-average"],
        f"{source}: invalid or missing macro-average row",
    )
    return set(ordinary)


def validate_metric_artifact(
    metric_path: str | Path,
    *,
    cache_path: str | Path,
    split: str,
    tasks: Sequence[str],
    layer: int,
    experiment: str,
    seed: int | None = None,
    min_temporal_gap: int | None = None,
    min_state_distance: float | None = None,
) -> dict[str, Any]:
    source, payload = _read_json(metric_path)
    _require(isinstance(payload, Mapping), f"{source}: metric JSON is not an object")
    _require(payload.get("evaluation_split") == split, f"{source}: split mismatch")
    _require(payload.get("experiment") == experiment, f"{source}: experiment mismatch")
    _require(payload.get("layer") == int(layer), f"{source}: layer mismatch")
    _require(str(Path(payload.get("cache", "")).resolve()) == str(Path(cache_path).resolve()), f"{source}: cache path mismatch")
    _same_identity(payload.get("cache_identity"), cache_path, f"{source}: cache_identity")
    cache = load_cache(cache_path)
    _require(payload.get("cache_provenance") == cache.get("provenance"), f"{source}: cache provenance mismatch")
    cache_splits = {str(record["split"]) for record in cache["records"]}
    _require(cache_splits == {split}, f"{source}: upstream cache split mismatch")
    _require(str(layer) in cache["tokens_by_layer"], f"{source}: layer absent from cache")
    _require(_metric_task_set(payload, source=source) == set(tasks), f"{source}: task set mismatch")
    expected_layer_name = f"video_block_{int(layer):02d}"
    for row in payload["metrics"]:
        _require(row["experiment"] == experiment, f"{source}: row experiment mismatch")
        _require(row["layer"] == expected_layer_name, f"{source}: row layer mismatch")
    negative_filter = payload.get("negative_filter")
    _require(isinstance(negative_filter, Mapping), f"{source}: negative_filter is missing")
    _require(int(negative_filter.get("num_pairs", 0)) > 0, f"{source}: no state-negative pairs")
    gap = int(negative_filter.get("min_temporal_gap", -1))
    distance = _finite(negative_filter.get("min_state_distance"), f"{source}: min_state_distance")
    _require(gap >= 0 and distance >= 0, f"{source}: invalid negative filter")
    if min_temporal_gap is not None:
        _require(gap == min_temporal_gap, f"{source}: min_temporal_gap mismatch")
    if min_state_distance is not None:
        _require(distance == min_state_distance, f"{source}: min_state_distance mismatch")

    head = payload.get("head")
    if experiment == "E0-RawBackbone":
        _require(head is None, f"{source}: E0 must not contain a head")
    else:
        _require(isinstance(head, Mapping), f"{source}: E1 head provenance is missing")
        _check_sha256(head.get("initial_head_sha256"), f"{source}: initial head hash")
        _require(int(head.get("trainable_parameter_count", 0)) > 0, f"{source}: invalid head parameter count")
        if experiment == "E1-InitHead":
            _require(seed is not None, f"{source}: initialization seed was not specified")
            _require(head.get("initialization_seed") == int(seed), f"{source}: initialization seed mismatch")
        elif seed is not None:
            _require(head.get("training_seed") == int(seed), f"{source}: training seed mismatch")
    return dict(payload)


def validate_e0_metrics(
    metric_paths: Sequence[str | Path],
    *,
    cache_path: str | Path,
    tasks: Sequence[str],
    layers: Sequence[int],
    min_temporal_gap: int | None = None,
    min_state_distance: float | None = None,
) -> list[dict[str, Any]]:
    _require(len(metric_paths) == len(layers), "E0 metric file count does not match layers")
    payloads = [
        validate_metric_artifact(
            path,
            cache_path=cache_path,
            split="val",
            tasks=tasks,
            layer=layer,
            experiment="E0-RawBackbone",
            min_temporal_gap=min_temporal_gap,
            min_state_distance=min_state_distance,
        )
        for path, layer in zip(metric_paths, layers, strict=True)
    ]
    filters = [payload["negative_filter"] for payload in payloads]
    _require(all(value == filters[0] for value in filters[1:]), "E0 metrics use different negative filters")
    return payloads


def validate_selection_artifact(
    selection_path: str | Path,
    *,
    selected_layer_path: str | Path,
    cache_path: str | Path,
    tasks: Sequence[str],
    layers: Sequence[int],
) -> int:
    source, payload = _read_json(selection_path)
    _require(isinstance(payload, Mapping), f"{source}: selection is not an object")
    _require(payload.get("schema_version") == 1, f"{source}: selection schema mismatch")
    _require(payload.get("evaluation_split") == "val", f"{source}: selection did not use val")
    _require(payload.get("experiment") == "E0-RawBackbone", f"{source}: selection experiment mismatch")
    _same_identity(payload.get("cache_identity"), cache_path, f"{source}: cache_identity")
    _require(set(payload.get("task_set", ())) == set(tasks), f"{source}: task set mismatch")
    candidates = payload.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == len(layers), f"{source}: candidate count mismatch")
    by_layer: dict[int, Mapping[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        _require(isinstance(candidate, Mapping), f"{source}: candidate {index} is invalid")
        layer = int(candidate.get("layer", -1))
        _require(layer not in by_layer, f"{source}: duplicate candidate layer {layer}")
        by_layer[layer] = candidate
    _require(set(by_layer) == set(layers), f"{source}: candidate layer set mismatch")
    selected = int(payload.get("selected_layer", -1))
    _require(selected in by_layer, f"{source}: selected layer is not a candidate")
    _require(payload.get("selected_layer_name") == f"video_block_{selected:02d}", f"{source}: selected layer name mismatch")
    _require(
        [layer for layer, row in by_layer.items() if row.get("selected") is True] == [selected],
        f"{source}: candidate selected flags mismatch",
    )
    selected_path = Path(selected_layer_path).expanduser().resolve()
    try:
        selected_text = selected_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RunnerArtifactError(f"cannot read {selected_path}: {error}") from error
    _require(selected_text == str(selected), "selected_layer.txt disagrees with selection.json")
    reference_filter = payload.get("negative_filter")
    _require(isinstance(reference_filter, Mapping), f"{source}: negative filter is missing")
    for layer, candidate in by_layer.items():
        candidate_payload = validate_metric_artifact(
            str(candidate.get("source", "")),
            cache_path=cache_path,
            split="val",
            tasks=tasks,
            layer=layer,
            experiment="E0-RawBackbone",
        )
        _require(candidate_payload["negative_filter"] == reference_filter, f"{source}: candidate filter mismatch")
    return selected


def _validate_finite_tree(value: Any, field: str) -> None:
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), f"{field} contains NaN/inf")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_tree(child, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_tree(child, f"{field}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{field} contains NaN/inf")


def validate_training_artifact(
    checkpoint_path: str | Path,
    *,
    log_path: str | Path,
    train_cache_path: str | Path,
    val_cache_path: str | Path,
    layer: int,
    steps: int,
    seed: int,
    min_temporal_gap: int,
    min_state_distance: float,
) -> dict[str, Any]:
    try:
        checkpoint = load_torch(checkpoint_path)
    except Exception as error:
        raise RunnerArtifactError(f"cannot load training checkpoint {checkpoint_path}: {error}") from error
    _require(isinstance(checkpoint, Mapping), "training checkpoint is not an object")
    _require(checkpoint.get("schema_version") == 1, "training checkpoint schema mismatch")
    _require(checkpoint.get("step") == steps, "training checkpoint step mismatch")
    _require(checkpoint.get("layer") == layer, "training checkpoint layer mismatch")
    _require(checkpoint.get("seed") == seed, "training checkpoint seed mismatch")
    _same_identity(checkpoint.get("train_cache_identity"), train_cache_path, "checkpoint train_cache_identity")
    _same_identity(checkpoint.get("val_cache_identity"), val_cache_path, "checkpoint val_cache_identity")
    _require(str(Path(checkpoint.get("train_cache", "")).resolve()) == str(Path(train_cache_path).resolve()), "checkpoint train cache path mismatch")
    _require(str(Path(checkpoint.get("val_cache", "")).resolve()) == str(Path(val_cache_path).resolve()), "checkpoint val cache path mismatch")
    expected_filter = {
        "min_temporal_gap": int(min_temporal_gap),
        "min_state_distance": float(min_state_distance),
    }
    _require(checkpoint.get("negative_filter") == expected_filter, "training negative filter mismatch")
    _check_sha256(checkpoint.get("initial_head_sha256"), "checkpoint initial head hash")
    _require(_finite(checkpoint.get("temperature"), "checkpoint temperature") > 0, "checkpoint temperature must be positive")
    config = checkpoint.get("head_config")
    state = checkpoint.get("head")
    _require(isinstance(config, Mapping) and isinstance(state, Mapping), "checkpoint head/config missing")
    try:
        head = ContrastiveContentHead(**dict(config))
        incompatible = head.load_state_dict(state, strict=True)
    except Exception as error:
        raise RunnerArtifactError(f"checkpoint head cannot be strictly loaded: {error}") from error
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "checkpoint head state is incompatible")
    _require(
        checkpoint.get("trainable_parameter_count") == head.trainable_parameter_count(),
        "checkpoint trainable parameter count mismatch",
    )
    _validate_finite_tree(checkpoint.get("head"), "checkpoint.head")
    optimizer = checkpoint.get("optimizer")
    _require(isinstance(optimizer, Mapping) and optimizer.get("state"), "checkpoint optimizer state is missing")
    _validate_finite_tree(optimizer, "checkpoint.optimizer")

    train_cache = load_cache(train_cache_path)
    val_cache = load_cache(val_cache_path)
    _require({str(row["split"]) for row in train_cache["records"]} == {"train"}, "training cache split mismatch")
    _require({str(row["split"]) for row in val_cache["records"]} == {"val"}, "validation cache split mismatch")
    train_tasks = {str(row["task"]) for row in train_cache["records"]}
    val_tasks = {str(row["task"]) for row in val_cache["records"]}
    _require(train_tasks == val_tasks, "training/validation cache task sets differ")
    _require(str(layer) in train_cache["tokens_by_layer"] and str(layer) in val_cache["tokens_by_layer"], "selected layer is absent from train/val cache")

    _, rows = _read_json(log_path)
    _require(isinstance(rows, list) and len(rows) == steps, "training log row count mismatch")
    required_train_fields = (
        "train_contrastive_loss",
        "train_positive_similarity",
        "train_negative_similarity",
        "embedding_norm",
        "embedding_state_spread",
        "gradient_norm",
    )
    for expected_step, row in enumerate(rows, 1):
        _require(isinstance(row, Mapping), f"training log row {expected_step} is invalid")
        _require(row.get("step") == expected_step, f"training log step {expected_step} mismatch")
        for field in required_train_fields:
            value = _finite(row.get(field), f"training log step {expected_step} {field}")
            if field in {"embedding_norm", "embedding_state_spread", "gradient_norm"}:
                _require(value > 0, f"training log step {expected_step} {field} must be positive")
        for field, value in row.items():
            if field.startswith("val_") and value is not None:
                _finite(value, f"training log step {expected_step} {field}")
    for row_name, row in (("first", rows[0]), ("final", rows[-1])):
        for field in (
            "val_contrastive_loss",
            "val_positive_similarity",
            "val_negative_similarity",
            "val_embedding_state_spread",
        ):
            _finite(row.get(field), f"{row_name} training log {field}")
    return dict(checkpoint)


def validate_test_metrics(
    *,
    cache_path: str | Path,
    tasks: Sequence[str],
    layer: int,
    seed: int,
    e0_path: str | Path,
    init_path: str | Path,
    trained_path: str | Path,
) -> list[dict[str, Any]]:
    payloads = [
        validate_metric_artifact(
            path,
            cache_path=cache_path,
            split="test",
            tasks=tasks,
            layer=layer,
            experiment=experiment,
            seed=seed if experiment != "E0-RawBackbone" else None,
        )
        for path, experiment in zip(
            (e0_path, init_path, trained_path), EXPERIMENTS, strict=True
        )
    ]
    filters = [payload["negative_filter"] for payload in payloads]
    _require(all(value == filters[0] for value in filters[1:]), "test metrics use different negative filters")
    init_head = payloads[1]["head"]
    trained_head = payloads[2]["head"]
    _require(
        init_head["initial_head_sha256"] == trained_head["initial_head_sha256"],
        "test Init/Trained initial head hashes differ",
    )
    _require(init_head["initialization_seed"] == trained_head["training_seed"] == seed, "test Init/Trained seeds differ")
    checkpoint_path = trained_head.get("checkpoint")
    _require(isinstance(checkpoint_path, str) and Path(checkpoint_path).is_file(), "trained metric checkpoint is missing")
    checkpoint = load_torch(checkpoint_path)
    _require(isinstance(checkpoint, Mapping), "trained metric checkpoint is invalid")
    _require(checkpoint.get("layer") == layer, "trained metric checkpoint layer mismatch")
    _require(checkpoint.get("seed") == seed, "trained metric checkpoint seed mismatch")
    _require(checkpoint.get("step") == trained_head.get("checkpoint_step"), "trained metric checkpoint step mismatch")
    _require(checkpoint.get("initial_head_sha256") == init_head["initial_head_sha256"], "trained metric checkpoint initialization hash mismatch")
    checkpoint_filter = checkpoint.get("negative_filter")
    _require(
        isinstance(checkpoint_filter, Mapping)
        and checkpoint_filter.get("min_temporal_gap") == filters[0].get("min_temporal_gap")
        and checkpoint_filter.get("min_state_distance") == filters[0].get("min_state_distance"),
        "trained metric/checkpoint negative filters differ",
    )
    return payloads


def validate_comparison_artifact(
    comparison_path: str | Path, *, allow_scientific_fail: bool = False
) -> dict[str, Any]:
    source, payload = _read_json(comparison_path)
    _require(isinstance(payload, Mapping), f"{source}: comparison is not an object")
    success = payload.get("overall_success")
    _require(isinstance(success, bool), f"{source}: overall_success is not boolean")
    criteria = payload.get("success_criteria")
    rows = payload.get("rows")
    sources = payload.get("sources")
    _require(isinstance(criteria, list) and criteria, f"{source}: success criteria are missing")
    _require(isinstance(rows, list) and rows, f"{source}: comparison rows are missing")
    _require(isinstance(sources, list) and len(sources) == 3, f"{source}: comparison sources are invalid")
    _require(all(Path(path).is_file() for path in sources), f"{source}: a metric source is missing")
    criterion_success: list[bool] = []
    for index, criterion in enumerate(criteria):
        _require(isinstance(criterion, Mapping), f"{source}: criterion {index} is invalid")
        _finite(criterion.get("state_retention"), f"{source}: criterion {index} state_retention")
        _require(isinstance(criterion.get("success"), bool), f"{source}: criterion {index} success is invalid")
        criterion_success.append(bool(criterion["success"]))
    _require(success == all(criterion_success), f"{source}: overall_success disagrees with criteria")
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"{source}: row {index} is invalid")
        for field in ("style_distance", "state_distance", "state_style_ratio", "retrieval_r1", "retrieval_r5"):
            _finite(row.get(field), f"{source}: row {index} {field}")
    _require(success or allow_scientific_fail, f"{source}: scientific success gate is false")
    return dict(payload)


def _add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_CONTENT_IDS), required=True)
    parser.add_argument("--tasks", type=_csv_strings, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min-temporal-gap", type=int)
    parser.add_argument("--min-state-distance", type=float)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("init-config")
    config.add_argument("--path", required=True)
    source = config.add_mutually_exclusive_group()
    source.add_argument("--config-json")
    source.add_argument("--config-file")
    for name in ("repo-root", "git-commit", "git-dirty", "python", "model-base", "checkpoint", "dataset-stats"):
        config.add_argument(f"--{name}")
    config.add_argument("--gpu-id", type=int)
    config.add_argument("--tasks", type=_csv_strings)
    config.add_argument("--layers", type=_csv_ints)
    for name in ("states-per-trajectory", "train-steps", "groups-per-batch", "val-every", "seed", "min-temporal-gap"):
        config.add_argument(f"--{name}", type=int)
    for name in ("temperature", "min-state-distance", "min-state-retention"):
        config.add_argument(f"--{name}", type=float)

    cache = commands.add_parser("validate-cache")
    cache.add_argument("--cache", required=True)
    cache.add_argument("--split", choices=tuple(SPLIT_CONTENT_IDS), required=True)
    cache.add_argument("--tasks", type=_csv_strings, required=True)
    cache.add_argument("--layers", type=_csv_ints, required=True)
    cache.add_argument("--states-per-trajectory", type=int, required=True)
    cache.add_argument("--expected-trajectories-per-task", type=int)

    e0 = commands.add_parser("validate-e0-metrics")
    e0.add_argument("--cache", required=True)
    e0.add_argument("--tasks", type=_csv_strings, required=True)
    e0.add_argument("--layers", type=_csv_ints, required=True)
    e0.add_argument("--metrics", nargs="+", required=True)
    e0.add_argument("--min-temporal-gap", type=int)
    e0.add_argument("--min-state-distance", type=float)

    selection = commands.add_parser("validate-selection")
    selection.add_argument("--selection", required=True)
    selection.add_argument("--selected-layer", required=True)
    selection.add_argument("--cache", required=True)
    selection.add_argument("--tasks", type=_csv_strings, required=True)
    selection.add_argument("--layers", type=_csv_ints, required=True)

    metric = commands.add_parser("validate-metric")
    metric.add_argument("--metric", required=True)
    _add_metric_arguments(metric)

    training = commands.add_parser("validate-training")
    training.add_argument("--checkpoint", required=True)
    training.add_argument("--log", "--log-json", dest="log", required=True)
    training.add_argument("--train-cache", required=True)
    training.add_argument("--val-cache", required=True)
    training.add_argument("--layer", type=int, required=True)
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--min-temporal-gap", type=int, required=True)
    training.add_argument("--min-state-distance", type=float, required=True)

    test = commands.add_parser("validate-test-metrics")
    test.add_argument("--cache", required=True)
    test.add_argument("--tasks", type=_csv_strings, required=True)
    test.add_argument("--layer", type=int, required=True)
    test.add_argument("--seed", type=int, required=True)
    test.add_argument("--e0", required=True)
    test.add_argument("--init", required=True)
    test.add_argument("--trained", required=True)

    comparison = commands.add_parser("validate-comparison")
    comparison.add_argument("--comparison", required=True)
    comparison.add_argument("--allow-scientific-fail", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.config_json is not None or args.config_file is not None:
        try:
            value = (
                json.loads(args.config_json)
                if args.config_json is not None
                else json.loads(Path(args.config_file).read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RunnerArtifactError(f"cannot read config JSON: {error}") from error
        _require(isinstance(value, Mapping), "config JSON must contain an object")
        return dict(value)
    names = (
        "repo_root", "git_commit", "git_dirty", "python", "gpu_id", "model_base",
        "checkpoint", "dataset_stats", "tasks", "layers", "states_per_trajectory",
        "train_steps", "groups_per_batch", "val_every", "seed", "temperature",
        "min_temporal_gap", "min_state_distance", "min_state_retention",
    )
    values = {name: getattr(args, name) for name in names}
    missing = [name for name, value in values.items() if value is None]
    _require(not missing, f"init-config is missing fields {missing}")
    values["tasks"] = list(values["tasks"])
    values["layers"] = list(values["layers"])
    return values


def main() -> None:
    args = _parser().parse_args()
    if args.command == "init-config":
        init_config(args.path, _config_from_args(args))
    elif args.command == "validate-cache":
        validate_cache_artifact(
            args.cache,
            split=args.split,
            tasks=args.tasks,
            layers=args.layers,
            states_per_trajectory=args.states_per_trajectory,
            expected_trajectories_per_task=args.expected_trajectories_per_task,
        )
    elif args.command == "validate-e0-metrics":
        validate_e0_metrics(
            args.metrics,
            cache_path=args.cache,
            tasks=args.tasks,
            layers=args.layers,
            min_temporal_gap=args.min_temporal_gap,
            min_state_distance=args.min_state_distance,
        )
    elif args.command == "validate-selection":
        validate_selection_artifact(
            args.selection,
            selected_layer_path=args.selected_layer,
            cache_path=args.cache,
            tasks=args.tasks,
            layers=args.layers,
        )
    elif args.command == "validate-metric":
        validate_metric_artifact(
            args.metric,
            cache_path=args.cache,
            split=args.split,
            tasks=args.tasks,
            layer=args.layer,
            experiment=args.experiment,
            seed=args.seed,
            min_temporal_gap=args.min_temporal_gap,
            min_state_distance=args.min_state_distance,
        )
    elif args.command == "validate-training":
        validate_training_artifact(
            args.checkpoint,
            log_path=args.log,
            train_cache_path=args.train_cache,
            val_cache_path=args.val_cache,
            layer=args.layer,
            steps=args.steps,
            seed=args.seed,
            min_temporal_gap=args.min_temporal_gap,
            min_state_distance=args.min_state_distance,
        )
    elif args.command == "validate-test-metrics":
        validate_test_metrics(
            cache_path=args.cache,
            tasks=args.tasks,
            layer=args.layer,
            seed=args.seed,
            e0_path=args.e0,
            init_path=args.init,
            trained_path=args.trained,
        )
    elif args.command == "validate-comparison":
        validate_comparison_artifact(args.comparison, allow_scientific_fail=args.allow_scientific_fail)
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(args.command)
    print(f"VALID {args.command}")


if __name__ == "__main__":
    main()


__all__ = [
    "RunnerArtifactError",
    "init_config",
    "validate_cache_artifact",
    "validate_comparison_artifact",
    "validate_e0_metrics",
    "validate_metric_artifact",
    "validate_selection_artifact",
    "validate_test_metrics",
    "validate_training_artifact",
]
