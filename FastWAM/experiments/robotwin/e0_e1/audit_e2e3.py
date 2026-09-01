#!/usr/bin/env python3
"""Fail-closed, read-only audit of a completed strict E2/E3 run.

The auditor never recomputes representations or base metrics.  It reloads the
complete artifact chain, independently re-derives every comparison row and
conclusion from the metric inputs, verifies strong identities and protocol
contracts, and writes reports only after every check has succeeded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .cache import load_cache
from .compare_e2e3 import ALL_CONTROLS, _canonical_unseen, _macro
from .decision_lock_e2e3 import strong_file_identity
from .head import ContrastiveContentHead
from .io_utils import file_identity, load_torch, module_state_sha256, write_json
from .select_layer_e2 import _load_candidate
from .smoke_proof_e2e3 import validate_smoke_proof
from .train_e2e3 import _canonical_json_sha256, _scientific_cache_contract


PROTOCOL = "r3_holdout_v1"
SEEN_VARIANTS = ("clean", "style_00_seed_0", "style_01_seed_1")
TEST_VARIANTS = ("clean", "style_02_seed_2")
HOLDOUT_VARIANT = "style_02_seed_2"
LAYERS = (8, 16, 24)
MODES = {"E2": "observed", "E3": "constant_zero_normalized"}
FORMAL_TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
CODE_NAMES = (
    "audit_e2e3.py",
    "backbone.py",
    "cache.py",
    "compare_e2e3.py",
    "data.py",
    "decision_lock_e2e3.py",
    "evaluate_e2e3.py",
    "extract.py",
    "head.py",
    "io_utils.py",
    "metrics.py",
    "negatives.py",
    "prompts.py",
    "select_layer_e2.py",
    "smoke_proof_e2e3.py",
    "train_e2e3.py",
    "run_e2_e3.sh",
)


class E2E3AuditError(ValueError):
    """Raised when a completed run does not satisfy the strict protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E2E3AuditError(message)


def _read_json(path_value: str | Path, *, label: str) -> tuple[Path, Mapping[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E2E3AuditError(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(payload, Mapping), f"{label} root must be an object: {path}")
    return path, payload


def _read_json_array(path_value: str | Path, *, label: str) -> tuple[Path, list[Any]]:
    path = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E2E3AuditError(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(payload, list) and bool(payload), f"{label} must be a non-empty array")
    return path, payload


def _sha256(path_value: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path_value).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(chunk_size):
                digest.update(block)
    except OSError as error:
        raise E2E3AuditError(f"cannot hash artifact {path}: {error}") from error
    return digest.hexdigest()


def _identity(path_value: str | Path, memo: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if path not in memo:
        try:
            memo[path] = strong_file_identity(path)
        except (OSError, RuntimeError) as error:
            raise E2E3AuditError(f"cannot establish immutable identity for {path}: {error}") from error
    return dict(memo[path])


def _identity_matches(
    declared: Any,
    actual: Mapping[str, Any],
    *,
    label: str,
    require_sha256: bool = True,
) -> None:
    _require(isinstance(declared, Mapping), f"{label} identity is missing")
    fields = ("path", "size_bytes", "mtime_ns", "sha256") if require_sha256 else (
        "path", "size_bytes", "mtime_ns"
    )
    for field in fields:
        _require(declared.get(field) == actual.get(field), f"{label} identity {field} mismatch")


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise E2E3AuditError(f"{label} must be numeric") from error
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _run_control_values(config: Mapping[str, Any]) -> dict[str, int | float]:
    """Return the immutable runner controls that every downstream artifact uses."""

    integer_bounds = {
        "train_steps": 1,
        "groups_per_batch": 2,
        "val_every": 1,
        "min_temporal_gap": 0,
    }
    result: dict[str, int | float] = {}
    for field, minimum in integer_bounds.items():
        value = config.get(field)
        _require(
            type(value) is int and value >= minimum,
            f"run config {field} must be an integer >= {minimum}",
        )
        result[field] = value
    seed = config.get("seed")
    _require(type(seed) is int, "run config seed must be an integer")
    result["seed"] = seed
    temperature = _finite(config.get("temperature"), label="run config temperature")
    _require(temperature == 0.07, "run config SupCon temperature must be 0.07")
    result["temperature"] = temperature
    state_distance = _finite(
        config.get("min_state_distance"), label="run config min_state_distance"
    )
    _require(
        state_distance >= 0.0,
        "run config min_state_distance must be non-negative",
    )
    result["min_state_distance"] = state_distance
    return result


def _validate_negative_filter(
    value: Any,
    *,
    config: Mapping[str, Any],
    label: str,
    require_num_pairs: bool,
) -> None:
    controls = _run_control_values(config)
    _require(isinstance(value, Mapping), f"{label} must be an object")
    expected_fields = {"min_temporal_gap", "min_state_distance"}
    if require_num_pairs:
        expected_fields.add("num_pairs")
    _require(set(value) == expected_fields, f"{label} field set mismatch")
    _require(
        type(value.get("min_temporal_gap")) is int
        and value["min_temporal_gap"] == controls["min_temporal_gap"],
        f"{label} min_temporal_gap differs from run config",
    )
    distance = _finite(value.get("min_state_distance"), label=f"{label} min_state_distance")
    _require(
        math.isclose(
            distance,
            float(controls["min_state_distance"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        f"{label} min_state_distance differs from run config",
    )
    if require_num_pairs:
        _require(
            type(value.get("num_pairs")) is int and value["num_pairs"] > 0,
            f"{label} num_pairs must be a positive integer",
        )


def _expected_ids(mode: str, split: str) -> set[int]:
    if mode == "smoke":
        return {"train": {0}, "val": {30}, "test": {40}}[split]
    return {
        "train": set(range(0, 30)),
        "val": set(range(30, 40)),
        "test": set(range(40, 50)),
    }[split]


def _validate_run_config(root: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    path, config = _read_json(root / "run_config.json", label="run config")
    _require(config.get("schema_version") == 1, "run config schema mismatch")
    _require(config.get("protocol") == PROTOCOL, "run config protocol mismatch")
    mode = str(config.get("mode"))
    _require(mode in {"smoke", "full"}, "run config mode must be smoke/full")
    tasks = tuple(str(value) for value in config.get("tasks", ()))
    _require(bool(tasks) and len(set(tasks)) == len(tasks), "run config task set is invalid")
    if mode == "full":
        _require(tasks == FORMAL_TASKS, f"full run must use tasks {FORMAL_TASKS}")
    else:
        _require(tasks == (FORMAL_TASKS[0],), "smoke run must use only place_a2b_left")
    _require(tuple(int(value) for value in config.get("layers", ())) == LAYERS,
             f"candidate layers must be exactly {LAYERS}")
    expected_states = 8 if mode == "full" else 2
    _require(int(config.get("states_per_trajectory", 0)) == expected_states,
             f"{mode} states_per_trajectory must be {expected_states}")
    _run_control_values(config)
    code = config.get("experiment_code_sha256")
    _require(isinstance(code, Mapping) and set(code) == set(CODE_NAMES),
             "run config code identity set is incomplete")
    script_dir = Path(__file__).resolve().parent
    for name in CODE_NAMES:
        _require(code.get(name) == _sha256(script_dir / name),
                 f"experiment code changed since run_config: {name}")
    for field in ("checkpoint", "dataset_stats"):
        declared = config.get(field)
        _require(isinstance(declared, Mapping), f"run config {field} identity missing")
        source = Path(str(declared.get("path", ""))).expanduser().resolve()
        try:
            current = file_identity(source)
        except OSError as error:
            raise E2E3AuditError(f"run config {field} is unavailable: {error}") from error
        _identity_matches(declared, current, label=f"run config {field}", require_sha256=False)
        _require(_is_sha256(declared.get("sha256")), f"run config {field} SHA-256 missing")
        _require(declared["sha256"] == _sha256(source), f"run config {field} SHA-256 mismatch")
    smoke_proof_evidence = None
    if mode == "full":
        declared = config.get("canonical_smoke_proof")
        _require(isinstance(declared, Mapping),
                 "full run config canonical-smoke proof identity is missing")
        canonical_dir = root.parent / "smoke"
        proof_path = canonical_dir / "canonical_smoke_proof.json"
        actual = _identity(proof_path, {})
        _identity_matches(declared, actual, label="full run canonical-smoke proof")
        try:
            smoke_proof_evidence = validate_smoke_proof(
                proof_path,
                canonical_smoke_dir=canonical_dir,
                full_config=config,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise E2E3AuditError(f"canonical smoke proof validation failed: {error}") from error
    else:
        _require(config.get("canonical_smoke_proof") is None,
                 "smoke run config must not declare a canonical-smoke proof")
    return config, {
        "path": str(path),
        "mode": mode,
        "tasks": list(tasks),
        "canonical_smoke_proof": smoke_proof_evidence,
    }


def _manifest_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, Mapping), f"{path}:{line_number}: row is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise E2E3AuditError(f"cannot parse manifest {path}: {error}") from error
    _require(bool(rows), f"manifest is empty: {path}")
    return rows


def _validate_manifest(
    cache: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    memo: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    provenance = cache["provenance"]
    json_path = Path(str(provenance.get("manifest_jsonl", ""))).expanduser().resolve()
    csv_path = Path(str(provenance.get("manifest_csv", ""))).expanduser().resolve()
    json_identity = _identity(json_path, memo)
    csv_identity = _identity(csv_path, memo)
    declared_sha = str(provenance.get("source_manifest_sha256", ""))
    _require(len(declared_sha) == 64 and declared_sha == json_identity["sha256"],
             f"source manifest SHA-256 mismatch: {json_path}")
    rows = _manifest_rows(json_path)
    _require(len(rows) == len(records), f"manifest/cache row count mismatch: {json_path}")
    keys = ("task", "content_id", "split", "trace_idx", "variant")
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        for key in keys:
            _require(row.get(key) == record.get(key),
                     f"manifest/cache {key} mismatch at row {index}: {json_path}")
        _require(row.get("physical_key") == record.get("physical_state_id"),
                 f"manifest/cache physical key mismatch at row {index}: {json_path}")
        _require(row.get("frame_idx") == record.get("timestep"),
                 f"manifest/cache frame mismatch at row {index}: {json_path}")
        hdf5 = Path(str(row.get("hdf5", ""))).expanduser().resolve()
        _require(hdf5.is_file(), f"manifest HDF5 source is unavailable: {hdf5}")
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise E2E3AuditError(f"cannot parse CSV manifest {csv_path}: {error}") from error
    _require(len(csv_rows) == len(rows), f"JSONL/CSV manifest row count mismatch: {csv_path}")
    for index, (left, right) in enumerate(zip(rows, csv_rows, strict=True)):
        for key in left:
            expected = str(left[key])
            _require(right.get(key) == expected,
                     f"JSONL/CSV manifest mismatch at row {index}, field {key}")
    return {"jsonl": json_identity, "csv": csv_identity, "rows": len(rows)}


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_proprio(cache: Mapping[str, Any], *, mode: str, label: str) -> list[str]:
    provenance = cache["provenance"]
    backbone = provenance.get("backbone")
    _require(isinstance(backbone, Mapping), f"{label}: backbone provenance missing")
    _require(backbone.get("proprio_mode") == mode, f"{label}: backbone proprio mode mismatch")
    for flag in ("uses_future_video", "uses_action_denoising", "uses_policy_rollout"):
        _require(backbone.get(flag) is False, f"{label}: unsafe backbone flag {flag}")
    conditions = provenance.get("conditions_by_physical_state")
    _require(isinstance(conditions, Mapping) and bool(conditions),
             f"{label}: per-state proprio audit is missing")
    record_states = {str(record["physical_state_id"]) for record in cache["records"]}
    _require(set(str(key) for key in conditions) == record_states,
             f"{label}: condition keys do not exactly cover physical states")
    effective_hashes: set[str] = set()
    for state_id, condition in conditions.items():
        _require(isinstance(condition, Mapping), f"{label}/{state_id}: malformed condition")
        context = condition.get("context")
        proprio = context.get("proprio") if isinstance(context, Mapping) else None
        _require(isinstance(proprio, Mapping), f"{label}/{state_id}: proprio audit missing")
        _require(proprio.get("mode") == mode, f"{label}/{state_id}: proprio mode mismatch")
        _require(proprio.get("intervention_point") == "post_normalizer_pre_proprio_encoder",
                 f"{label}/{state_id}: proprio intervention point mismatch")
        _require(proprio.get("proprio_token_preserved") is True,
                 f"{label}/{state_id}: proprio token was removed")
        shape = proprio.get("shape")
        _require(isinstance(shape, list) and len(shape) == 2 and int(shape[-1]) == 14,
                 f"{label}/{state_id}: effective proprio shape is invalid")
        observed = str(proprio.get("observed_normalized_sha256", ""))
        effective = str(proprio.get("effective_normalized_sha256", ""))
        _require(_is_sha256(observed) and _is_sha256(effective),
                 f"{label}/{state_id}: proprio SHA-256 is malformed")
        _require(condition.get("normalized_proprio_sha256") == effective,
                 f"{label}/{state_id}: effective proprio hashes disagree")
        effective_hashes.add(effective)
        if mode == "observed":
            _require(observed == effective, f"{label}/{state_id}: E2 altered observed proprio")
        else:
            _require(proprio.get("all_zero") is True,
                     f"{label}/{state_id}: E3 effective proprio is not exact zero")
    if mode == "constant_zero_normalized":
        _require(len(effective_hashes) == 1,
                 f"{label}: E3 effective proprio is not one state-independent constant")
    return sorted(effective_hashes)


def _validate_cache(
    path: Path,
    *,
    experiment: str,
    split: str,
    config: Mapping[str, Any],
    memo: dict[Path, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identity(path, memo)
    try:
        cache = load_cache(path)
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        raise E2E3AuditError(f"invalid {experiment} {split} cache {path}: {error}") from error
    mode = MODES[experiment]
    expected_variants = SEEN_VARIANTS if split in {"train", "val"} else TEST_VARIANTS
    provenance = cache["provenance"]
    _require(provenance.get("protocol") == PROTOCOL, f"{experiment}/{split}: protocol mismatch")
    _require(provenance.get("split") == split, f"{experiment}/{split}: split mismatch")
    _require(provenance.get("proprio_mode") == mode, f"{experiment}/{split}: mode mismatch")
    _require(tuple(provenance.get("active_variants", ())) == expected_variants,
             f"{experiment}/{split}: active variants mismatch")
    _require(tuple(cache.get("variant_names", ())) == expected_variants,
             f"{experiment}/{split}: cache variants mismatch")
    _require(provenance.get("holdout_variant") == HOLDOUT_VARIANT,
             f"{experiment}/{split}: holdout declaration mismatch")
    backbone = provenance.get("backbone")
    _require(isinstance(backbone, Mapping),
             f"{experiment}/{split}: backbone provenance missing")
    declared_checkpoint = config.get("checkpoint")
    actual_checkpoint = backbone.get("checkpoint")
    _require(isinstance(declared_checkpoint, Mapping),
             f"{experiment}/{split}: run checkpoint identity missing")
    _identity_matches(
        actual_checkpoint,
        declared_checkpoint,
        label=f"{experiment}/{split} backbone checkpoint",
    )
    declared_stats = config.get("dataset_stats")
    _require(isinstance(declared_stats, Mapping),
             f"{experiment}/{split}: run dataset-stats identity missing")
    _require(
        backbone.get("dataset_stats_path") == declared_stats.get("path"),
        f"{experiment}/{split}: backbone dataset-stats path mismatch",
    )
    _require(
        backbone.get("dataset_stats_sha256") == declared_stats.get("sha256"),
        f"{experiment}/{split}: backbone dataset-stats SHA-256 mismatch",
    )
    _require(
        backbone.get("model_base_path") == config.get("model_base"),
        f"{experiment}/{split}: backbone model-base path mismatch",
    )
    smoke = str(config["mode"]) == "smoke"
    _require(provenance.get("allow_incomplete") is smoke,
             f"{experiment}/{split}: allow_incomplete does not match run mode")
    expected_content_ids = sorted(_expected_ids(str(config["mode"]), split)) if smoke else None
    _require(provenance.get("content_ids") == expected_content_ids,
             f"{experiment}/{split}: explicit content-ID provenance mismatch")
    records = cache["records"]
    _require({str(row.get("split")) for row in records} == {split},
             f"{experiment}/{split}: mixed record splits")
    _require({str(row.get("variant")) for row in records} == set(expected_variants),
             f"{experiment}/{split}: record variants mismatch")
    if split in {"train", "val"}:
        _require(HOLDOUT_VARIANT not in {str(row.get("variant")) for row in records},
                 f"{experiment}/{split}: R3 leakage")
        _require(provenance.get("decision_lock_identity") is None,
                 f"{experiment}/{split}: pre-test cache carries a decision lock")
    else:
        _require(provenance.get("decision_lock_created_before_test") is True,
                 f"{experiment}/test: lock-before-test proof missing")
    tasks = tuple(str(value) for value in config["tasks"])
    _require(tuple(provenance.get("tasks", ())) == tasks,
             f"{experiment}/{split}: task order mismatch")
    _require({str(row.get("task")) for row in records} == set(tasks),
             f"{experiment}/{split}: task coverage mismatch")
    expected_ids = _expected_ids(str(config["mode"]), split)
    states_per_trajectory = int(config["states_per_trajectory"])
    groups: dict[tuple[str, int], set[str]] = {}
    for row in records:
        key = (str(row["task"]), int(row["content_id"]))
        groups.setdefault(key, set()).add(str(row["physical_state_id"]))
    for task in tasks:
        observed_ids = {content for (record_task, content) in groups if record_task == task}
        _require(observed_ids == expected_ids,
                 f"{experiment}/{split}/{task}: trajectory IDs violate the split")
        for content_id in expected_ids:
            _require(len(groups[(task, content_id)]) == states_per_trajectory,
                     f"{experiment}/{split}/{task}/{content_id}: state count mismatch")
    expected_layers = set(LAYERS) if split in {"train", "val"} else None
    actual_layers = {int(value) for value in cache["tokens_by_layer"]}
    if expected_layers is not None:
        _require(actual_layers == expected_layers,
                 f"{experiment}/{split}: candidate layer cache mismatch")
    manifest = _validate_manifest(cache, records=records, memo=memo)
    effective_hashes = _validate_proprio(cache, mode=mode, label=f"{experiment}/{split}")
    contract = _scientific_cache_contract(cache)
    return cache, {
        "identity": identity,
        "cache_provenance": dict(provenance),
        "manifest": manifest,
        "scientific_contract": contract,
        "effective_proprio_sha256": effective_hashes,
        "layers": sorted(actual_layers),
        "records": len(records),
        "physical_states": len(cache["physical_states"]),
    }


def _validate_selection(
    root: Path,
    *,
    e2_val_identity: Mapping[str, Any],
    tasks: Sequence[str],
    config: Mapping[str, Any] | None = None,
    memo: dict[Path, dict[str, Any]],
) -> tuple[int, Mapping[str, Any], dict[str, Any]]:
    path, selection = _read_json(root / "layer_selection/selection.json", label="selection")
    _require(selection.get("schema_version") == 2 and selection.get("protocol") == PROTOCOL,
             "selection schema/protocol mismatch")
    _require(selection.get("evaluation_split") == "val", "selection is not validation-only")
    _require(selection.get("experiment") == "E2-RawBackbone", "selection was not E2 raw")
    _require(selection.get("proprio_mode") == "observed", "selection did not use observed proprio")
    _require(selection.get("r3_used") is False, "selection does not prove R3 exclusion")
    _require(tuple(selection.get("active_variants", ())) == SEEN_VARIANTS,
             "selection variants are not C/R1/R2")
    _require(tuple(selection.get("task_set", ())) == tuple(sorted(tasks)),
             "selection task set mismatch")
    _identity_matches(selection.get("cache_identity"), e2_val_identity,
                      label="selection E2 validation cache")
    candidates = selection.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == 3,
             "selection requires three candidate records")
    _require({int(row.get("layer", -1)) for row in candidates if isinstance(row, Mapping)} == set(LAYERS),
             "selection candidates must be exactly 8/16/24")
    selected_layer = int(selection.get("selected_layer", -1))
    _require(selected_layer in LAYERS, "selected layer is invalid")
    selected_rows = [row for row in candidates if isinstance(row, Mapping) and row.get("selected") is True]
    _require(len(selected_rows) == 1 and int(selected_rows[0]["layer"]) == selected_layer,
             "selection winner marker mismatch")
    ranked = sorted(candidates, key=lambda row: (
        int(row["joint_rank_sum"]),
        -float(row["macro_retrieval_r1"]),
        -float(row["macro_state_style_ratio"]),
        int(row["layer"]),
    ))
    _require(int(ranked[0]["layer"]) == selected_layer, "selected layer violates declared ranking")
    candidate_evidence: dict[str, Any] = {}
    top_negative_filter = selection.get("negative_filter")
    _require(isinstance(top_negative_filter, Mapping) and bool(top_negative_filter),
             "selection negative filter is missing")
    if config is not None:
        _validate_negative_filter(
            top_negative_filter,
            config=config,
            label="selection negative filter",
            require_num_pairs=True,
        )
    for row in candidates:
        source = Path(str(row.get("source", ""))).expanduser().resolve()
        parsed = _load_candidate(source)
        _require(int(parsed["layer"]) == int(row["layer"]), "candidate source/layer mismatch")
        _require(parsed["task_set"] == sorted(tasks), "candidate metric task set mismatch")
        _identity_matches(parsed["cache_identity"], e2_val_identity,
                          label=f"candidate layer {row['layer']} E2 validation cache")
        _require(parsed["negative_filter"] == top_negative_filter,
                 f"candidate layer {row['layer']} negative filter mismatch")
        _require(math.isclose(float(row["macro_retrieval_r1"]),
                              float(parsed["macro_retrieval_r1"]),
                              rel_tol=1e-12, abs_tol=1e-12),
                 f"candidate layer {row['layer']} retrieval is stale")
        _require(math.isclose(float(row["macro_state_style_ratio"]),
                              float(parsed["macro_state_style_ratio"]),
                              rel_tol=1e-12, abs_tol=1e-12),
                 f"candidate layer {row['layer']} ratio is stale")
        _require(Path(str(parsed["source"])).resolve() == source,
                 f"candidate layer {row['layer']} source path mismatch")
        candidate_evidence[str(int(row["layer"]))] = _identity(source, memo)
    selected_text = (root / "layer_selection/selected_layer.txt").resolve()
    try:
        text_value = int(selected_text.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise E2E3AuditError(f"invalid selected_layer.txt: {error}") from error
    _require(text_value == selected_layer, "selected_layer.txt disagrees with selection.json")
    return selected_layer, selection, {
        "identity": _identity(path, memo),
        "selected_layer_text": _identity(selected_text, memo),
        "candidate_metrics": candidate_evidence,
    }


def _validate_checkpoint_run_controls(
    checkpoint: Mapping[str, Any],
    *,
    experiment: str,
    config: Mapping[str, Any],
    label: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Bind one best/final checkpoint's effective controls to run_config."""

    controls = _run_control_values(config)
    controlled = checkpoint.get("controlled_training_config")
    _require(isinstance(controlled, Mapping), f"{label}: controlled config missing")
    _require(
        checkpoint.get("controlled_training_config_sha256")
        == _canonical_json_sha256(controlled),
        f"{label}: controlled config SHA-256 mismatch",
    )
    controlled_expected = {
        "steps": controls["train_steps"],
        "groups_per_batch": controls["groups_per_batch"],
        "val_every": controls["val_every"],
        "seed": controls["seed"],
        "min_temporal_gap": controls["min_temporal_gap"],
        "min_state_distance": controls["min_state_distance"],
    }
    for field, expected in controlled_expected.items():
        actual = controlled.get(field)
        if isinstance(expected, float):
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=0.0)
            )
        else:
            matches = type(actual) is int and actual == expected
        _require(matches, f"{label}: controlled {field} differs from run config")
    loss = controlled.get("loss")
    loss_temperature = (
        _finite(loss.get("temperature"), label=f"{label} loss temperature")
        if isinstance(loss, Mapping)
        else math.nan
    )
    _require(
        isinstance(loss, Mapping)
        and loss.get("name") == "multi_positive_supcon"
        and math.isclose(
            loss_temperature,
            float(controls["temperature"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and type(loss.get("positive_views_per_state")) is int
        and loss["positive_views_per_state"] == 3,
        f"{label}: SupCon configuration mismatch",
    )
    _require(
        tuple(controlled.get("active_variants", ())) == SEEN_VARIANTS,
        f"{label}: training variants are not C/R1/R2",
    )
    _require(
        controlled.get("holdout_variant") == HOLDOUT_VARIANT,
        f"{label}: training holdout mismatch",
    )
    selection = controlled.get("checkpoint_selection")
    _require(
        isinstance(selection, Mapping) and selection.get("r3_allowed") is False,
        f"{label}: controlled config allowed R3 selection",
    )
    for field, expected in (
        ("training_steps", controls["train_steps"]),
        ("seed", controls["seed"]),
    ):
        _require(
            type(checkpoint.get(field)) is int and checkpoint[field] == expected,
            f"{label}: checkpoint {field} differs from run config",
        )
    temperature = _finite(
        checkpoint.get("temperature"), label=f"{label} checkpoint temperature"
    )
    _require(
        math.isclose(
            temperature,
            float(controls["temperature"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        f"{label}: checkpoint temperature differs from run config",
    )
    _validate_negative_filter(
        checkpoint.get("negative_filter"),
        config=config,
        label=f"{label} checkpoint negative filter",
        require_num_pairs=False,
    )
    training_config = checkpoint.get("training_config")
    _require(isinstance(training_config, Mapping), f"{label}: training config missing")
    for field, expected in controlled.items():
        _require(
            training_config.get(field) == expected,
            f"{label}: training config {field} differs from controlled config",
        )
    _require(
        training_config.get("experiment") == experiment
        and training_config.get("proprio_mode") == MODES[experiment],
        f"{label}: training experiment/proprio mode mismatch",
    )
    return controlled, training_config


def _validate_training(
    root: Path,
    *,
    experiment: str,
    selected_layer: int,
    cache_evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    memo: dict[Path, dict[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    directory = root / experiment.lower()
    best_path = directory / f"{experiment.lower()}_best_content_head.pt"
    final_path = directory / f"{experiment.lower()}_final_content_head.pt"
    try:
        best = load_torch(best_path)
        final = load_torch(final_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise E2E3AuditError(f"cannot load {experiment} checkpoints: {error}") from error
    _require(isinstance(best, Mapping) and isinstance(final, Mapping),
             f"{experiment}: checkpoint root is not an object")
    mode = MODES[experiment]
    common_expected = {
        "schema_version": 2,
        "experiment": experiment,
        "protocol": PROTOCOL,
        "proprio_mode": mode,
        "layer": selected_layer,
    }
    checkpoint_controls: dict[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    for name, checkpoint, kind in (("best", best, "best_val"), ("final", final, "final")):
        for field, value in common_expected.items():
            _require(checkpoint.get(field) == value, f"{experiment}/{name}: {field} mismatch")
        _require(checkpoint.get("checkpoint_kind") == kind,
                 f"{experiment}/{name}: checkpoint kind mismatch")
        _require(isinstance(checkpoint.get("head"), Mapping),
                 f"{experiment}/{name}: head state missing")
        checkpoint_controls[name] = _validate_checkpoint_run_controls(
            checkpoint,
            experiment=experiment,
            config=config,
            label=f"{experiment}/{name}",
        )
    ignored_checkpoint_fields = {"checkpoint_kind", "step", "head", "optimizer"}
    _require(
        {key: value for key, value in best.items() if key not in ignored_checkpoint_fields}
        == {key: value for key, value in final.items() if key not in ignored_checkpoint_fields},
        f"{experiment}: best/final checkpoint metadata differ",
    )
    _require(int(best.get("step", -1)) == int(best.get("best_step", -2)),
             f"{experiment}: best checkpoint step mismatch")
    _require(int(final.get("step", -1)) == int(final.get("training_steps", -2)),
             f"{experiment}: final checkpoint step mismatch")
    _require(int(final.get("best_step", -1)) == int(best["best_step"]),
             f"{experiment}: final/best best-step mismatch")
    best_metric = best.get("best_metric")
    _require(isinstance(best_metric, Mapping), f"{experiment}: best metric missing")
    _require(best_metric.get("metric") == "val_contrastive_loss"
             and best_metric.get("mode") == "min"
             and best_metric.get("tie_break") == "earliest_step"
             and best_metric.get("r3_used") is False,
             f"{experiment}: checkpoint selection is not strict best-validation")
    controlled, training_config = checkpoint_controls["best"]
    controls = _run_control_values(config)
    for split in ("train", "val"):
        current = cache_evidence[split]["identity"]
        _identity_matches(best.get(f"{split}_cache_identity"), current,
                          label=f"{experiment} checkpoint {split} cache")
        _require(best.get(f"{split}_scientific_cache_contract")
                 == cache_evidence[split]["scientific_contract"],
                 f"{experiment}: {split} scientific cache contract is stale")
    head_config = best.get("head_config")
    _require(isinstance(head_config, Mapping), f"{experiment}: head config missing")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(best.get("seed", -1)))
        initial_head = ContrastiveContentHead(**dict(head_config))
    _require(module_state_sha256(initial_head) == best.get("initial_head_sha256"),
             f"{experiment}: declared initialization does not match seed/head config")
    head = ContrastiveContentHead(**dict(head_config))
    incompatible = head.load_state_dict(dict(best["head"]), strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys,
             f"{experiment}: strict head load failed")
    _require(head.trainable_parameter_count() == int(best.get("trainable_parameter_count", -1)),
             f"{experiment}: head capacity provenance mismatch")
    trained_hash = module_state_sha256(head)
    _require(trained_hash != best.get("initial_head_sha256"),
             f"{experiment}: best head equals initialization")
    log_path, log_rows = _read_json_array(
        directory / "train_log.json", label=f"{experiment} train log"
    )
    _require(len(log_rows) == int(best.get("training_steps", -1)),
             f"{experiment}: train log length mismatch")
    steps = [int(row.get("step", -1)) for row in log_rows if isinstance(row, Mapping)]
    _require(steps == list(range(1, len(log_rows) + 1)),
             f"{experiment}: train log steps are non-canonical")
    validation_rows: list[tuple[int, float, bool]] = []
    for row in log_rows:
        _require(isinstance(row, Mapping), f"{experiment}: malformed train log row")
        _finite(row.get("train_contrastive_loss"), label=f"{experiment} train loss")
        value = row.get("val_contrastive_loss")
        if value is not None:
            validation_rows.append((
                int(row["step"]),
                _finite(value, label=f"{experiment} validation loss"),
                row.get("is_best") is True,
            ))
    _require(bool(validation_rows), f"{experiment}: train log has no validation")
    validation_steps = [step for step, _, _ in validation_rows]
    expected_validation_steps = sorted(
        {1, int(controls["train_steps"])}
        | set(
            range(
                int(controls["val_every"]),
                int(controls["train_steps"]) + 1,
                int(controls["val_every"]),
            )
        )
    )
    _require(
        validation_steps == expected_validation_steps,
        f"{experiment}: validation schedule differs from run config",
    )
    expected_best = min(validation_rows, key=lambda item: (item[1], item[0]))
    _require(expected_best[0] == int(best["best_step"]),
             f"{experiment}: checkpoint is not the minimum validation-loss step")
    _require(math.isclose(expected_best[1], float(best_metric.get("best_value")),
                          rel_tol=1e-12, abs_tol=1e-12),
             f"{experiment}: best validation value mismatch")
    marked = [step for step, _, is_best in validation_rows if is_best]
    _require(bool(marked) and int(best["best_step"]) in marked,
             f"{experiment}: train log lacks the selected best marker")
    csv_path = directory / "train_log.csv"
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise E2E3AuditError(f"cannot read {experiment} train CSV: {error}") from error
    _require(len(csv_rows) == len(log_rows), f"{experiment}: JSON/CSV train log length mismatch")
    for row_index, (json_row, csv_row) in enumerate(zip(log_rows, csv_rows, strict=True)):
        _require(set(csv_row) == set(json_row),
                 f"{experiment}: JSON/CSV train log columns differ")
        for field, value in json_row.items():
            expected = "" if value is None else str(value)
            _require(csv_row[field] == expected,
                     f"{experiment}: JSON/CSV log mismatch row {row_index} field {field}")
    curve_path = directory / "training_curves.svg"
    try:
        curve_text = curve_path.read_text(encoding="utf-8")
    except OSError as error:
        raise E2E3AuditError(f"cannot read {experiment} training curve: {error}") from error
    _require("<svg" in curve_text and f"best-val step={best['best_step']}" in curve_text,
             f"{experiment}: training curve does not mark the selected best step")
    summary_path, summary = _read_json(
        directory / "training_summary.json", label=f"{experiment} training summary"
    )
    _require(summary.get("experiment") == experiment and summary.get("protocol") == PROTOCOL,
             f"{experiment}: training summary metadata mismatch")
    _require(int(summary.get("best_step", -1)) == int(best["best_step"]),
             f"{experiment}: summary best step mismatch")
    _require(Path(str(summary.get("selected_checkpoint", ""))).resolve() == best_path.resolve(),
             f"{experiment}: summary selected checkpoint mismatch")
    _require(Path(str(summary.get("final_checkpoint", ""))).resolve() == final_path.resolve(),
             f"{experiment}: summary final checkpoint mismatch")
    _require(Path(str(summary.get("training_curves", ""))).resolve() == curve_path.resolve(),
             f"{experiment}: summary curve path mismatch")
    _require(
        summary.get("training_config") == training_config,
        f"{experiment}: summary training config mismatch",
    )
    _require(
        summary.get("controlled_training_config_sha256")
        == best.get("controlled_training_config_sha256"),
        f"{experiment}: summary controlled-config hash mismatch",
    )
    return best, {
        "best_checkpoint": _identity(best_path, memo),
        "final_checkpoint": _identity(final_path, memo),
        "train_log_json": _identity(log_path, memo),
        "train_log_csv": _identity(csv_path, memo),
        "training_curves": _identity(curve_path, memo),
        "training_summary": _identity(summary_path, memo),
        "best_step": int(best["best_step"]),
        "best_val_contrastive_loss": float(best_metric["best_value"]),
        "trained_head_sha256": trained_hash,
    }


def _validate_lock(
    root: Path,
    *,
    selected_layer: int,
    selection_identity: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
    checkpoint_evidence: Mapping[str, Mapping[str, Any]],
    cache_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    memo: dict[Path, dict[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    path, lock = _read_json(root / "decision_lock.json", label="decision lock")
    identity = _identity(path, memo)
    _require(lock.get("schema_version") == 1 and lock.get("protocol") == PROTOCOL,
             "decision lock schema/protocol mismatch")
    _require(lock.get("r3_access_before_lock") is False,
             "decision lock does not prove pre-test R3 isolation")
    _require(int(lock.get("selected_layer", -1)) == selected_layer,
             "decision lock selected layer mismatch")
    _identity_matches(lock.get("selection_identity"), selection_identity,
                      label="decision-lock selection")
    entries = lock.get("checkpoints")
    _require(isinstance(entries, Mapping) and set(entries) == set(MODES),
             "decision lock checkpoint set mismatch")
    shared = lock.get("shared")
    _require(isinstance(shared, Mapping), "decision lock shared controls missing")
    for experiment in MODES:
        checkpoint = checkpoints[experiment]
        entry = entries[experiment]
        _require(isinstance(entry, Mapping), f"decision lock {experiment} entry missing")
        _identity_matches(entry.get("checkpoint"), checkpoint_evidence[experiment]["best_checkpoint"],
                          label=f"decision lock {experiment} checkpoint")
        for field in ("experiment", "proprio_mode", "best_step",
                      "controlled_training_config_sha256", "initial_head_sha256"):
            _require(entry.get(field) == checkpoint.get(field),
                     f"decision lock {experiment} {field} mismatch")
        declared_caches = lock.get("train_val_cache_identities", {}).get(experiment)
        _require(isinstance(declared_caches, Mapping),
                 f"decision lock {experiment} train/val identities missing")
        for split in ("train", "val"):
            _identity_matches(declared_caches.get(split), cache_evidence[experiment][split]["identity"],
                              label=f"decision lock {experiment}/{split} cache")
        for field in ("controlled_training_config_sha256", "initial_head_sha256"):
            _require(shared.get(field) == checkpoint.get(field),
                     f"decision lock shared {field} mismatch")
    expected_outputs = lock.get("expected_test_outputs")
    _require(isinstance(expected_outputs, Mapping) and set(expected_outputs) == set(MODES),
             "decision lock expected test outputs mismatch")
    for experiment in MODES:
        expected = Path(str(expected_outputs[experiment])).expanduser().resolve()
        actual = (root / f"cache/{experiment.lower()}_test.pt").resolve()
        _require(expected == actual, f"decision lock {experiment} test output path mismatch")
        provenance = cache_evidence[experiment]["test"]["cache_provenance"]
        _identity_matches(provenance.get("decision_lock_identity"), identity,
                          label=f"{experiment} test-cache decision lock")
        _require(int(identity["mtime_ns"]) <= int(cache_evidence[experiment]["test"]["identity"]["mtime_ns"]),
                 f"{experiment} test cache predates decision lock")
    return lock, {"identity": identity, "created_at_utc": lock.get("created_at_utc")}


def _validate_metric(
    path: Path,
    *,
    experiment: str,
    selected_layer: int,
    cache_identity: Mapping[str, Any],
    lock_identity: Mapping[str, Any],
    tasks: Sequence[str],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    config: Mapping[str, Any],
    memo: dict[Path, dict[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    source, metric = _read_json(path, label=f"{experiment} test metric")
    mode = "constant_zero_normalized" if experiment.startswith("E3-") else "observed"
    _require(metric.get("schema_version") == 2 and metric.get("protocol") == PROTOCOL,
             f"{experiment}: metric schema/protocol mismatch")
    _require(metric.get("experiment") == experiment and metric.get("evaluation_split") == "test",
             f"{experiment}: metric experiment/split mismatch")
    _require(metric.get("proprio_mode") == mode, f"{experiment}: metric mode mismatch")
    _require(tuple(metric.get("active_variants", ())) == TEST_VARIANTS
             and tuple(metric.get("record_variants", ())) == TEST_VARIANTS,
             f"{experiment}: metric is not exact C/R3")
    _require(metric.get("holdout_variant") == HOLDOUT_VARIANT,
             f"{experiment}: metric holdout mismatch")
    _require(metric.get("r3_used_for_selection") is False,
             f"{experiment}: metric does not prove R3 exclusion from selection")
    _require(int(metric.get("layer", -1)) == selected_layer,
             f"{experiment}: metric layer mismatch")
    _identity_matches(metric.get("cache_identity"), cache_identity,
                      label=f"{experiment} metric cache")
    _identity_matches(metric.get("decision_lock_identity"), lock_identity,
                      label=f"{experiment} metric decision lock")
    _validate_negative_filter(
        metric.get("negative_filter"),
        config=config,
        label=f"{experiment} metric negative filter",
        require_num_pairs=True,
    )
    protocol = metric.get("metric_protocol")
    _require(isinstance(protocol, Mapping)
             and tuple(protocol.get("style_order", ())) == ("r3",)
             and tuple(protocol.get("required_variants", ())) == ("clean", "r3")
             and protocol.get("query") == "R3" and protocol.get("gallery") == "Clean",
             f"{experiment}: metric protocol is not exact R3-to-Clean")
    rows = metric.get("metrics")
    _require(isinstance(rows, list), f"{experiment}: metric rows missing")
    task_rows = [row for row in rows if isinstance(row, Mapping)
                 and not str(row.get("task", "")).endswith("-task-average")]
    _require({str(row.get("task")) for row in task_rows} == set(tasks),
             f"{experiment}: metric task coverage mismatch")
    macro = _macro(metric)
    canonical = _canonical_unseen(macro)
    for value in canonical.values():
        _finite(value, label=f"{experiment} macro metric")
    head = metric.get("head")
    _require(isinstance(head, Mapping), f"{experiment}: paired checkpoint provenance missing")
    _require(head.get("checkpoint_kind") == "best_val",
             f"{experiment}: metric did not pair the best-val checkpoint")
    paired_identity = head.get("paired_checkpoint_identity", head.get("checkpoint_identity"))
    _identity_matches(
        paired_identity,
        checkpoint_identity,
        label=f"{experiment} metric paired checkpoint",
    )
    expected_seed = _run_control_values(config)["seed"]
    if experiment.endswith("InitHead"):
        _require(
            head.get("initialization_seed") == expected_seed,
            f"{experiment}: initialization seed differs from run config",
        )
    else:
        _require(
            head.get("training_seed") == expected_seed,
            f"{experiment}: training seed differs from run config",
        )
    if experiment.endswith("TrainedHead"):
        _require(int(head.get("best_step", -1)) == int(checkpoint["best_step"]),
                 f"{experiment}: trained metric best step mismatch")
    return metric, {"identity": _identity(source, memo), "macro": canonical}


def _validate_exact_mapping(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(isinstance(actual, Mapping), f"{label} must be an object")
    _require(set(actual) == set(expected), f"{label} field set mismatch")
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if isinstance(expected_value, bool):
            matches = actual_value is expected_value
        elif isinstance(expected_value, int):
            matches = type(actual_value) is int and actual_value == expected_value
        elif isinstance(expected_value, float):
            matches = (
                isinstance(actual_value, (int, float))
                and not isinstance(actual_value, bool)
                and math.isfinite(float(actual_value))
                and math.isclose(
                    float(actual_value), expected_value, rel_tol=1e-12, abs_tol=1e-12
                )
            )
        else:
            matches = actual_value == expected_value
        _require(matches, f"{label} field {field} is stale or inconsistent")


def _validate_exact_rows(
    actual: Any,
    expected: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    _require(isinstance(actual, list), f"{label} must be an array")
    _require(len(actual) == len(expected), f"{label} row count mismatch")
    for index, (actual_row, expected_row) in enumerate(zip(actual, expected, strict=True)):
        _validate_exact_mapping(actual_row, expected_row, label=f"{label} row {index}")


def _validate_csv_rows(
    path: Path,
    expected: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    _require(bool(expected), f"{label} expected rows are empty")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except OSError as error:
        raise E2E3AuditError(f"cannot read {label}: {error}") from error
    expected_fields = list(expected[0])
    _require(fieldnames == expected_fields, f"{label} column order/set mismatch")
    _require(len(rows) == len(expected), f"{label} row count mismatch")
    for row_index, (actual_row, expected_row) in enumerate(zip(rows, expected, strict=True)):
        _require(set(actual_row) == set(expected_row), f"{label} row {row_index} columns differ")
        for field, value in expected_row.items():
            expected_text = "" if value is None else str(value)
            _require(
                actual_row[field] == expected_text,
                f"{label} row {row_index} field {field} is stale or inconsistent",
            )


def _derive_comparison_interpretation(
    controls: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_name = {str(row["experiment"]): row for row in controls}
    _require(set(by_name) == set(ALL_CONTROLS), "cannot derive interpretation without six controls")
    final_by_name = {str(row["experiment"]): row for row in final_rows}
    _require(
        set(final_by_name)
        == {"E1-TrainedHead", "E2-TrainedHead", "E3-NoProprio-TrainedHead"},
        "cannot derive interpretation without final E1/E2/E3 rows",
    )
    e1_trained = final_by_name["E1-TrainedHead"]
    e2_raw = by_name["E2-RawBackbone"]
    e2_init = by_name["E2-InitHead"]
    e2_trained = by_name["E2-TrainedHead"]
    e3_raw = by_name["E3-NoProprio-RawBackbone"]
    e3_init = by_name["E3-NoProprio-InitHead"]
    e3_trained = by_name["E3-NoProprio-TrainedHead"]
    e2_generalizes = (
        float(e2_trained["R3_to_Clean_R@1"])
        > max(float(e2_raw["R3_to_Clean_R@1"]), float(e2_init["R3_to_Clean_R@1"]))
        and float(e2_trained["state_style_ratio_R3"])
        > max(
            float(e2_raw["state_style_ratio_R3"]),
            float(e2_init["state_style_ratio_R3"]),
        )
    )
    e3_beats_controls = (
        float(e3_trained["R3_to_Clean_R@1"])
        > max(float(e3_raw["R3_to_Clean_R@1"]), float(e3_init["R3_to_Clean_R@1"]))
        and float(e3_trained["state_style_ratio_R3"])
        > max(
            float(e3_raw["state_style_ratio_R3"]),
            float(e3_init["state_style_ratio_R3"]),
        )
    )
    e2_r1_retention = float(e2_trained["R3_to_Clean_R@1"]) / max(
        float(e1_trained["R3_to_Clean_R@1"]), 1e-8
    )
    e2_ratio_retention = float(e2_trained["state_style_ratio_R3"]) / max(
        float(e1_trained["state_style_ratio_R3"]), 1e-8
    )
    large_e1_to_e2_drop = e2_r1_retention < 0.75 or e2_ratio_retention < 0.5
    e3_r1_retention = float(e3_trained["R3_to_Clean_R@1"]) / max(
        float(e2_trained["R3_to_Clean_R@1"]), 1e-8
    )
    e3_ratio_retention = float(e3_trained["state_style_ratio_R3"]) / max(
        float(e2_trained["state_style_ratio_R3"]), 1e-8
    )
    large_no_proprio_drop = e3_r1_retention < 0.75 or e3_ratio_retention < 0.5
    if e2_generalizes and large_e1_to_e2_drop:
        e2_conclusion = (
            "The C/R1/R2-trained content head improves both R3 retrieval and the state/style "
            "ratio over Raw and Init controls, supporting unseen-background generalization; "
            "however, it has a predefined large drop relative to E1, so that generalization is "
            "materially weaker than the seen-style E1 reference."
        )
    elif e2_generalizes:
        e2_conclusion = (
            "The C/R1/R2-trained content head improves both R3 retrieval and the state/style "
            "ratio over Raw and Init controls and does not show a predefined large drop relative "
            "to E1, supporting unseen-background generalization with substantial retention of "
            "the seen-style E1 result."
        )
    elif large_e1_to_e2_drop:
        e2_conclusion = (
            "E2 does not clearly outperform both unseen-style controls and has a predefined large "
            "drop relative to E1; the current evidence does not support generalizable style "
            "invariance and is materially weaker than the seen-style E1 reference."
        )
    else:
        e2_conclusion = (
            "E2 does not clearly outperform both unseen-style controls, although it does not show "
            "a predefined large drop relative to E1; the current E1 result remains consistent "
            "with invariance to seen transformations rather than proven generalizable style "
            "invariance."
        )
    if large_no_proprio_drop:
        e3_conclusion = (
            "Removing state-specific proprio causes a large drop relative to E2; previous state "
            "discrimination relies substantially on proprio-conditioned video tokens."
        )
    elif e3_beats_controls:
        e3_conclusion = (
            "The no-proprio trained head remains above both no-proprio controls without a large "
            "E2-relative drop; proprio shortcuts alone do not explain the result and visual "
            "information contributes substantially."
        )
    else:
        e3_conclusion = (
            "The no-proprio result is inconclusive: it neither shows a predefined large E2-relative "
            "drop nor clearly beats both no-proprio controls on retrieval and ratio."
        )
    return {
        "rules_are_descriptive_not_a_significance_test": True,
        "e2_beats_raw_and_init_on_r1_and_ratio": e2_generalizes,
        "e2_r1_retention_vs_e1": e2_r1_retention,
        "e2_ratio_retention_vs_e1": e2_ratio_retention,
        "large_e1_to_e2_drop_rule": "R@1 retention < 0.75 or ratio retention < 0.50",
        "large_e1_to_e2_drop": large_e1_to_e2_drop,
        "e3_beats_raw_and_init_on_r1_and_ratio": e3_beats_controls,
        "e3_r1_retention_vs_e2": e3_r1_retention,
        "e3_ratio_retention_vs_e2": e3_ratio_retention,
        "large_no_proprio_drop_rule": "R@1 retention < 0.75 or ratio retention < 0.50",
        "large_no_proprio_drop": large_no_proprio_drop,
        "e2_conclusion": e2_conclusion,
        "e3_conclusion": e3_conclusion,
    }


def _render_comparison_summary(
    final_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    interpretation: Mapping[str, Any],
) -> str:
    header = "| Experiment | Proprio | Train Styles | Test Style | Style Dist | State Dist | Ratio | R@1 | R@5 |\n"
    separator = "|---|:---:|---|---|---:|---:|---:|---:|---:|\n"

    def table(rows: Sequence[Mapping[str, Any]]) -> str:
        return "".join(
            f"| {row['experiment']} | {row['proprio']} | {row['train_styles']} | "
            f"{row['test_style']} | {float(row['style_distance_R3']):.6f} | "
            f"{float(row['state_distance']):.6f} | "
            f"{float(row['state_style_ratio_R3']):.3f} | "
            f"{float(row['R3_to_Clean_R@1']):.3f} | "
            f"{float(row['R3_to_Clean_R@5']):.3f} |\n"
            for row in rows
        )

    return (
        "# Final E1/E2/E3 comparison\n\n"
        + header
        + separator
        + table(final_rows)
        + "\n# Strict E2/E3 controls\n\n"
        + header
        + separator
        + table(control_rows)
        + "\nE2: "
        + str(interpretation["e2_conclusion"])
        + "\n\nE3: "
        + str(interpretation["e3_conclusion"])
        + "\n"
    )


def _validate_comparison(
    root: Path,
    *,
    metrics: Mapping[str, Mapping[str, Any]],
    metric_evidence: Mapping[str, Mapping[str, Any]],
    selected_layer: int,
    lock_identity: Mapping[str, Any],
    memo: dict[Path, dict[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    path, comparison = _read_json(root / "comparison/comparison.json", label="comparison")
    _require(
        set(comparison) == {
            "schema_version",
            "protocol",
            "sources",
            "decision_lock_identity",
            "selected_layer",
            "controls",
            "final_e1_e2_e3",
            "interpretation",
        },
        "comparison JSON field set mismatch",
    )
    _require(comparison.get("schema_version") == 2 and comparison.get("protocol") == PROTOCOL,
             "comparison schema/protocol mismatch")
    _require(int(comparison.get("selected_layer", -1)) == selected_layer,
             "comparison selected layer mismatch")
    _identity_matches(comparison.get("decision_lock_identity"), lock_identity,
                      label="comparison decision lock")
    sources = comparison.get("sources")
    _require(isinstance(sources, Mapping) and set(sources) == {*ALL_CONTROLS, "E1-TrainedHead"},
             "comparison source set mismatch")
    controls = comparison.get("controls")
    _require(isinstance(controls, list) and len(controls) == len(ALL_CONTROLS),
             "comparison must contain six controls")
    _require(
        [row.get("experiment") for row in controls if isinstance(row, Mapping)]
        == list(ALL_CONTROLS),
        "comparison control experiment order/set mismatch",
    )

    expected_controls: list[dict[str, Any]] = []
    for experiment in ALL_CONTROLS:
        _require(Path(str(sources[experiment])).expanduser().resolve()
                 == Path(str(metric_evidence[experiment]["identity"]["path"])).resolve(),
                 f"comparison source path mismatch for {experiment}")
        payload = metrics[experiment]
        canonical = _canonical_unseen(_macro(payload))
        expected_controls.append({
            "experiment": experiment,
            "proprio": "No" if experiment.startswith("E3-") else "Yes",
            "train_styles": "C/R1/R2",
            "test_style": "R3 unseen",
            "layer": selected_layer,
            **canonical,
        })
    _validate_exact_rows(controls, expected_controls, label="comparison controls")

    e1_source = Path(str(sources["E1-TrainedHead"])).expanduser().resolve()
    _, e1_payload = _read_json(e1_source, label="formal E1 metric")
    _require(
        e1_payload.get("evaluation_split") == "test"
        and e1_payload.get("experiment") == "E1-TrainedHead"
        and isinstance(e1_payload.get("metrics"), list),
        "comparison E1 source is not the formal trained-head test metric",
    )
    e1_metrics = _canonical_unseen(
        _macro(e1_payload), require_generic_r3_aliases=False
    )
    expected_final = [
        {
            "experiment": "E1-TrainedHead",
            "proprio": "Yes",
            "train_styles": "C/R1/R2/R3",
            "test_style": "R3 seen",
            "layer": int(e1_payload["layer"]),
            **e1_metrics,
        },
        expected_controls[ALL_CONTROLS.index("E2-TrainedHead")],
        expected_controls[ALL_CONTROLS.index("E3-NoProprio-TrainedHead")],
    ]
    final = comparison.get("final_e1_e2_e3")
    _require(isinstance(final, list), "comparison final E1/E2/E3 table is missing")
    _validate_exact_rows(final, expected_final, label="comparison final E1/E2/E3")

    expected_interpretation = _derive_comparison_interpretation(
        expected_controls,
        expected_final,
    )
    interpretation = comparison.get("interpretation")
    _validate_exact_mapping(
        interpretation,
        expected_interpretation,
        label="comparison conservative interpretation",
    )

    controls_csv = root / "comparison/controls.csv"
    final_csv = root / "comparison/e1_e2_e3.csv"
    summary_markdown = root / "comparison/summary.md"
    _validate_csv_rows(controls_csv, expected_controls, label="comparison controls CSV")
    _validate_csv_rows(final_csv, expected_final, label="comparison E1/E2/E3 CSV")
    try:
        summary_text = summary_markdown.read_text(encoding="utf-8")
    except OSError as error:
        raise E2E3AuditError(f"cannot read comparison Markdown summary: {error}") from error
    _require(
        summary_text
        == _render_comparison_summary(
            expected_final,
            expected_controls,
            expected_interpretation,
        ),
        "comparison Markdown summary is stale or inconsistent with metric inputs",
    )
    required_files = {
        "comparison_json": path,
        "e1_metric": e1_source,
        "controls_csv": controls_csv,
        "e1_e2_e3_csv": final_csv,
        "summary_markdown": summary_markdown,
    }
    return comparison, {name: _identity(value, memo) for name, value in required_files.items()}


def _backbone_without_intervention(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "backbone provenance is missing")
    result = dict(value)
    result.pop("proprio_mode", None)
    result.pop("native_prefill_verified", None)
    return result


def _assert_e2_e3_cache_equivalence(
    caches: Mapping[str, Mapping[str, Mapping[str, Any]]],
    evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        e2 = caches["E2"][split]
        e3 = caches["E3"][split]
        for field in ("records", "physical_states"):
            _require(e2[field] == e3[field], f"E2/E3 {split} {field} differ")
        _require(torch.equal(e2["proprio_raw"], e3["proprio_raw"]),
                 f"E2/E3 {split} raw proprio differ")
        e2_provenance = e2["provenance"]
        e3_provenance = e3["provenance"]
        _require(e2_provenance.get("source_manifest_sha256")
                 == e3_provenance.get("source_manifest_sha256"),
                 f"E2/E3 {split} visual/source manifest digests differ")
        _require(e2.get("visual_input_sha256_by_physical_state")
                 == e3.get("visual_input_sha256_by_physical_state"),
                 f"E2/E3 {split} actual visual-input digests differ")
        _require(e2_provenance.get("task_prompt_sha256")
                 == e3_provenance.get("task_prompt_sha256"),
                 f"E2/E3 {split} task prompts differ")
        _require(_backbone_without_intervention(e2_provenance.get("backbone"))
                 == _backbone_without_intervention(e3_provenance.get("backbone")),
                 f"E2/E3 {split} backbone semantics differ beyond proprio intervention")
        e2_contract = evidence["E2"][split]["scientific_contract"]
        e3_contract = evidence["E3"][split]["scientific_contract"]
        for field in (
            "variant_names",
            "variants_per_state",
            "num_records",
            "num_physical_states",
            "physical_state_ids",
            "records_sha256",
            "physical_states_sha256",
            "proprio_raw_sha256",
            "proprio_raw_shape",
            "token_shapes_by_layer",
            "source_manifest_sha256",
            "visual_inputs_sha256",
        ):
            _require(e2_contract.get(field) == e3_contract.get(field),
                     f"E2/E3 {split} scientific contract field {field} differs")
        result[split] = {
            "source_manifest_sha256": e2_contract["source_manifest_sha256"],
            "visual_inputs_sha256": e2_contract["visual_inputs_sha256"],
            "records_sha256": e2_contract["records_sha256"],
            "physical_states_sha256": e2_contract["physical_states_sha256"],
            "proprio_raw_sha256": e2_contract["proprio_raw_sha256"],
            "token_shapes_by_layer": e2_contract["token_shapes_by_layer"],
        }
    train_ids = set(evidence["E2"]["train"]["scientific_contract"]["physical_state_ids"])
    val_ids = set(evidence["E2"]["val"]["scientific_contract"]["physical_state_ids"])
    test_ids = set(evidence["E2"]["test"]["scientific_contract"]["physical_state_ids"])
    _require(train_ids and val_ids and test_ids, "one or more physical-state splits are empty")
    _require(not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids),
             "physical-state leakage across train/val/test")
    return result


def _metric_path(root: Path, experiment: str, layer: int) -> Path:
    slug = experiment.lower().replace("-", "_")
    return root / f"test_metrics/{slug}_layer_{layer:02d}.json"


def audit_e2e3_run(
    run_dir: str | Path,
    *,
    require_success_marker: bool = False,
    read_only: bool = False,
) -> tuple[Path, Path]:
    """Validate a complete strict run and atomically write its audit reports."""

    root = Path(run_dir).expanduser().resolve()
    _require(root.is_dir(), f"run directory does not exist: {root}")
    state_path = root / "status/state.txt"
    success_path = root / "status/SUCCESS"
    if require_success_marker:
        try:
            state = state_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise E2E3AuditError(f"runner state marker is missing: {error}") from error
        _require(state == "SUCCESS", "runner state is not SUCCESS")
        try:
            _require(bool(success_path.read_text(encoding="utf-8").strip()),
                     "runner SUCCESS marker is empty")
        except OSError as error:
            raise E2E3AuditError(f"runner SUCCESS marker is missing: {error}") from error

    memo: dict[Path, dict[str, Any]] = {}
    config, config_evidence = _validate_run_config(root)
    caches: dict[str, dict[str, dict[str, Any]]] = {experiment: {} for experiment in MODES}
    cache_evidence: dict[str, dict[str, dict[str, Any]]] = {experiment: {} for experiment in MODES}
    for experiment in MODES:
        for split in ("train", "val", "test"):
            path = root / f"cache/{experiment.lower()}_{split}.pt"
            cache, evidence = _validate_cache(
                path,
                experiment=experiment,
                split=split,
                config=config,
                memo=memo,
            )
            caches[experiment][split] = cache
            cache_evidence[experiment][split] = evidence
    equivalence = _assert_e2_e3_cache_equivalence(caches, cache_evidence)

    selected_layer, selection, selection_evidence = _validate_selection(
        root,
        e2_val_identity=cache_evidence["E2"]["val"]["identity"],
        tasks=tuple(config["tasks"]),
        config=config,
        memo=memo,
    )
    for experiment in MODES:
        _require(cache_evidence[experiment]["test"]["layers"] == [selected_layer],
                 f"{experiment} test cache is not limited to the selected layer")
    checkpoints: dict[str, Mapping[str, Any]] = {}
    training_evidence: dict[str, dict[str, Any]] = {}
    for experiment in MODES:
        checkpoint, evidence = _validate_training(
            root,
            experiment=experiment,
            selected_layer=selected_layer,
            cache_evidence=cache_evidence[experiment],
            config=config,
            memo=memo,
        )
        checkpoints[experiment] = checkpoint
        training_evidence[experiment] = evidence
    _require(checkpoints["E2"]["controlled_training_config_sha256"]
             == checkpoints["E3"]["controlled_training_config_sha256"],
             "E2/E3 controlled training configurations differ")
    _require(checkpoints["E2"]["initial_head_sha256"]
             == checkpoints["E3"]["initial_head_sha256"],
             "E2/E3 head initializations differ")
    _require(checkpoints["E2"]["head_config"] == checkpoints["E3"]["head_config"]
             and checkpoints["E2"]["trainable_parameter_count"]
             == checkpoints["E3"]["trainable_parameter_count"],
             "E2/E3 head capacity differs")
    for split in ("train", "val"):
        _require(checkpoints["E2"][f"{split}_scientific_cache_contract"]
                 == checkpoints["E3"][f"{split}_scientific_cache_contract"],
                 f"E2/E3 checkpoint {split} scientific contracts differ")

    lock, lock_evidence = _validate_lock(
        root,
        selected_layer=selected_layer,
        selection_identity=selection_evidence["identity"],
        checkpoints=checkpoints,
        checkpoint_evidence=training_evidence,
        cache_evidence=cache_evidence,
        memo=memo,
    )
    metrics: dict[str, Mapping[str, Any]] = {}
    metric_evidence: dict[str, dict[str, Any]] = {}
    for experiment in ALL_CONTROLS:
        base = "E3" if experiment.startswith("E3-") else "E2"
        metric, evidence = _validate_metric(
            _metric_path(root, experiment, selected_layer),
            experiment=experiment,
            selected_layer=selected_layer,
            cache_identity=cache_evidence[base]["test"]["identity"],
            lock_identity=lock_evidence["identity"],
            tasks=tuple(config["tasks"]),
            checkpoint=checkpoints[base],
            checkpoint_identity=training_evidence[base]["best_checkpoint"],
            config=config,
            memo=memo,
        )
        metrics[experiment] = metric
        metric_evidence[experiment] = evidence
    comparison, comparison_evidence = _validate_comparison(
        root,
        metrics=metrics,
        metric_evidence=metric_evidence,
        selected_layer=selected_layer,
        lock_identity=lock_evidence["identity"],
        memo=memo,
    )

    artifacts = {
        "run_config": _identity(config_evidence["path"], memo),
        "caches": {experiment: {
            split: cache_evidence[experiment][split]["identity"]
            for split in ("train", "val", "test")
        } for experiment in MODES},
        "manifests": {experiment: {
            split: cache_evidence[experiment][split]["manifest"]
            for split in ("train", "val", "test")
        } for experiment in MODES},
        "selection": selection_evidence,
        "training": training_evidence,
        "decision_lock": lock_evidence,
        "control_metrics": metric_evidence,
        "comparison": comparison_evidence,
    }
    if config_evidence["canonical_smoke_proof"] is not None:
        artifacts["canonical_smoke_proof"] = config_evidence["canonical_smoke_proof"]
    assertions = {
        "strict_protocol": True,
        "full_or_smoke_cardinality": True,
        "r3_absent_from_train_val": True,
        "selection_val_only_c_r1_r2_layers_8_16_24": True,
        "decision_lock_precedes_test": True,
        "test_exactly_c_r3": True,
        "e3_effective_proprio_exact_zero_single_hash": True,
        "proprio_token_preserved": True,
        "e2_e3_same_sources_states_raw_proprio_prompts_backbone": True,
        "e2_e3_same_token_shapes": True,
        "e2_e3_same_head_capacity_seed_config": True,
        "supcon_temperature_0_07": True,
        "best_validation_checkpoints_only": True,
        "six_control_metrics_present": True,
        "comparison_revalidated": True,
        "training_curves_present": True,
        "native_fastwam_not_used_for_future_or_rollout": True,
    }
    audit = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "audit_status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root),
        "run_mode": config["mode"],
        "tasks": list(config["tasks"]),
        "selected_layer": selected_layer,
        "best_steps": {experiment: training_evidence[experiment]["best_step"] for experiment in MODES},
        "best_val_contrastive_loss": {
            experiment: training_evidence[experiment]["best_val_contrastive_loss"]
            for experiment in MODES
        },
        "e3_effective_proprio_sha256": {
            split: cache_evidence["E3"][split]["effective_proprio_sha256"][0]
            for split in ("train", "val", "test")
        },
        "e2_e3_equivalence": equivalence,
        "assertions": assertions,
        "comparison_interpretation": comparison["interpretation"],
        "artifact_identities": artifacts,
    }
    deliverables = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "COMPLETE_AND_AUDITED",
        "run_dir": str(root),
        "selected_layer": selected_layer,
        "best_steps": audit["best_steps"],
        "protocol_audit": str((root / "protocol_audit.json").resolve()),
        "training_curves": {
            experiment: training_evidence[experiment]["training_curves"]["path"]
            for experiment in MODES
        },
        "six_control_metrics": {
            experiment: metric_evidence[experiment]["identity"]["path"]
            for experiment in ALL_CONTROLS
        },
        "comparison": comparison_evidence,
        "decision_lock": lock_evidence,
        "artifact_identities": artifacts,
    }
    audit_path = root / "protocol_audit.json"
    deliverables_path = root / "deliverables.json"
    if read_only:
        _, published_audit = _read_json(audit_path, label="published protocol audit")
        _, published_deliverables = _read_json(
            deliverables_path, label="published deliverables"
        )
        expected_without_time = dict(audit)
        published_without_time = dict(published_audit)
        expected_without_time.pop("audited_at_utc", None)
        published_without_time.pop("audited_at_utc", None)
        _require(
            published_without_time == expected_without_time,
            "published protocol audit does not match current artifact chain",
        )
        _require(
            published_deliverables == deliverables,
            "published deliverables do not match current artifact chain",
        )
    else:
        # No report is touched before all checks above pass.  Each final write
        # is atomic; a failed audit therefore cannot publish a false PASS.
        audit_path = write_json(audit_path, audit)
        deliverables_path = write_json(deliverables_path, deliverables)
    return audit_path, deliverables_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--require-success-marker",
        action="store_true",
        help="require an already-completed runner; omit when this audit is the final pre-SUCCESS gate",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="revalidate existing reports byte-semantically without rewriting them",
    )
    args = parser.parse_args()
    audit, deliverables = audit_e2e3_run(
        args.run_dir,
        require_success_marker=args.require_success_marker,
        read_only=args.read_only,
    )
    print(f"strict E2/E3 audit PASS: {audit}")
    print(f"deliverables manifest: {deliverables}")


if __name__ == "__main__":
    main()


__all__ = ["E2E3AuditError", "audit_e2e3_run"]
