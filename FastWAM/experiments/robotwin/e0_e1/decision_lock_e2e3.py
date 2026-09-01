#!/usr/bin/env python3
"""Freeze E2/E3 model decisions before any held-out R3 test cache exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import file_identity, load_torch, write_json


SCHEMA_VERSION = 1
PROTOCOL = "r3_holdout_v1"
EXPECTED = {
    "E2": "observed",
    "E3": "constant_zero_normalized",
}


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def strong_file_identity(path: str | Path) -> dict[str, Any]:
    before = file_identity(path)
    digest = sha256_file(path)
    after = file_identity(path)
    if before != after:
        raise RuntimeError(f"file changed while hashing: {Path(path).resolve()}")
    return {**after, "sha256": digest}


def _read_json(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {source}")
    return source, payload


def _checkpoint_summary(
    path_value: str | Path,
    *,
    experiment: str,
    selected_layer: int,
) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    payload = load_torch(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not an object: {path}")
    expected_mode = EXPECTED[experiment]
    checks = {
        "schema_version": payload.get("schema_version") == 2,
        "experiment": payload.get("experiment") == experiment,
        "protocol": payload.get("protocol") == PROTOCOL,
        "proprio_mode": payload.get("proprio_mode") == expected_mode,
        "checkpoint_kind": payload.get("checkpoint_kind") == "best_val",
        "layer": payload.get("layer") == selected_layer,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"{path}: invalid best-val checkpoint fields {failed}")
    best_metric = payload.get("best_metric")
    if not isinstance(best_metric, Mapping) or best_metric.get("r3_used") is not False:
        raise ValueError(f"{path}: checkpoint selection does not prove R3 exclusion")
    if best_metric.get("metric") != "val_contrastive_loss" or best_metric.get("mode") != "min":
        raise ValueError(f"{path}: unexpected checkpoint-selection criterion")
    if int(payload.get("step", -1)) != int(payload.get("best_step", -2)):
        raise ValueError(f"{path}: selected checkpoint is not the recorded best step")
    controlled_hash = str(payload.get("controlled_training_config_sha256", ""))
    initial_hash = str(payload.get("initial_head_sha256", ""))
    if len(controlled_hash) != 64 or len(initial_hash) != 64:
        raise ValueError(f"{path}: training hashes are malformed")
    training_config = payload.get("controlled_training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError(f"{path}: controlled training config is missing")
    if training_config.get("holdout_variant") != "style_02_seed_2":
        raise ValueError(f"{path}: controlled config has wrong holdout")
    if tuple(training_config.get("active_variants", ())) != (
        "clean", "style_00_seed_0", "style_01_seed_1"
    ):
        raise ValueError(f"{path}: controlled config is not three-view C/R1/R2")
    selection = training_config.get("checkpoint_selection")
    if not isinstance(selection, Mapping) or selection.get("r3_allowed") is not False:
        raise ValueError(f"{path}: R3 was not forbidden for checkpoint selection")

    cache_identities: dict[str, dict[str, Any]] = {}
    cache_payloads: dict[str, Mapping[str, Any]] = {}
    for split in ("train", "val"):
        identity = payload.get(f"{split}_cache_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{path}: {split} cache identity is missing")
        cache_path = Path(str(payload.get(f"{split}_cache", ""))).expanduser().resolve()
        if str(identity.get("path")) != str(cache_path):
            raise ValueError(f"{path}: {split} cache path/identity mismatch")
        current_stat = file_identity(cache_path)
        for field in ("path", "size_bytes", "mtime_ns"):
            if current_stat[field] != identity.get(field):
                raise ValueError(f"{path}: {split} cache changed after training")
        declared_sha = str(identity.get("sha256", ""))
        if len(declared_sha) != 64:
            raise ValueError(f"{path}: {split} cache has no strong identity")
        actual_sha = sha256_file(cache_path)
        if actual_sha != declared_sha:
            raise ValueError(f"{path}: {split} cache SHA-256 changed after training")
        cache_identities[split] = dict(identity)
        cache_payload = load_torch(cache_path)
        if not isinstance(cache_payload, Mapping):
            raise ValueError(f"{path}: {split} cache is not an object")
        cache_payloads[split] = cache_payload
    scientific_contracts: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        contract = payload.get(f"{split}_scientific_cache_contract")
        if not isinstance(contract, Mapping) or not contract:
            raise ValueError(f"{path}: {split} scientific cache contract is missing")
        scientific_contracts[split] = dict(contract)
    return {
        "checkpoint": strong_file_identity(path),
        "experiment": experiment,
        "proprio_mode": expected_mode,
        "best_step": int(payload["best_step"]),
        "best_val_contrastive_loss": float(best_metric["best_value"]),
        "controlled_training_config_sha256": controlled_hash,
        "initial_head_sha256": initial_hash,
        "train_cache_identity": cache_identities["train"],
        "val_cache_identity": cache_identities["val"],
        "train_scientific_cache_contract": scientific_contracts["train"],
        "val_scientific_cache_contract": scientific_contracts["val"],
        "backbone_semantics_by_split": {
            split: _cache_backbone_semantics(cache_payloads[split])
            for split in ("train", "val")
        },
        "task_prompt_sha256_by_split": {
            split: _cache_prompt_hashes(cache_payloads[split])
            for split in ("train", "val")
        },
    }


def _cache_backbone_semantics(cache: Any) -> dict[str, Any]:
    if not isinstance(cache, Mapping):
        raise ValueError("training cache is not an object")
    provenance = cache.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training cache provenance is missing")
    backbone = provenance.get("backbone")
    if not isinstance(backbone, Mapping):
        raise ValueError("training cache backbone provenance is missing")
    result = dict(backbone)
    # These are the intended condition-specific/audit-only fields.  All model,
    # checkpoint, preprocessing, capture and interface semantics must match.
    result.pop("proprio_mode", None)
    result.pop("native_prefill_verified", None)
    return result


def _cache_prompt_hashes(cache: Mapping[str, Any]) -> dict[str, Any]:
    provenance = cache.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training cache provenance is missing")
    prompts = provenance.get("task_prompt_sha256")
    if not isinstance(prompts, Mapping) or not prompts:
        raise ValueError("training cache task prompt provenance is missing")
    return dict(prompts)


def create_decision_lock(
    *,
    selection_path: str | Path,
    e2_checkpoint: str | Path,
    e3_checkpoint: str | Path,
    e2_test_output: str | Path,
    e3_test_output: str | Path,
    output_path: str | Path,
) -> Path:
    selection_source, selection = _read_json(selection_path)
    if selection.get("schema_version") != 2 or selection.get("protocol") != PROTOCOL:
        raise ValueError("E2 layer selection schema/protocol mismatch")
    if selection.get("evaluation_split") != "val" or selection.get("experiment") != "E2-RawBackbone":
        raise ValueError("layer selection was not E2 raw validation")
    if selection.get("proprio_mode") != "observed" or selection.get("r3_used") is not False:
        raise ValueError("layer selection does not prove observed-proprio/R3 exclusion")
    if tuple(selection.get("active_variants", ())) != (
        "clean", "style_00_seed_0", "style_01_seed_1"
    ):
        raise ValueError("layer selection did not use exactly C/R1/R2")
    selected_layer = int(selection.get("selected_layer", -1))
    if selected_layer not in (8, 16, 24):
        raise ValueError(f"invalid selected layer {selected_layer}")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or {int(row["layer"]) for row in candidates} != {8, 16, 24}:
        raise ValueError("selection did not evaluate exactly layers 8/16/24")

    e2 = _checkpoint_summary(e2_checkpoint, experiment="E2", selected_layer=selected_layer)
    e3 = _checkpoint_summary(e3_checkpoint, experiment="E3", selected_layer=selected_layer)
    if e2["controlled_training_config_sha256"] != e3["controlled_training_config_sha256"]:
        raise ValueError("E2/E3 controlled training configurations differ")
    if e2["initial_head_sha256"] != e3["initial_head_sha256"]:
        raise ValueError("E2/E3 initial content heads differ")
    if e2["backbone_semantics_by_split"] != e3["backbone_semantics_by_split"]:
        raise ValueError("E2/E3 frozen backbone/checkpoint semantics differ")
    if e2["task_prompt_sha256_by_split"] != e3["task_prompt_sha256_by_split"]:
        raise ValueError("E2/E3 task prompt provenance differs")
    for split in ("train", "val"):
        key = f"{split}_scientific_cache_contract"
        if e2[key] != e3[key]:
            raise ValueError(
                f"E2/E3 {split} physical states, raw proprio, record order, or token shapes differ"
            )
    train_contract = e2["train_scientific_cache_contract"]
    val_contract = e2["val_scientific_cache_contract"]
    train_states = set(train_contract.get("physical_state_ids", ()))
    val_states = set(val_contract.get("physical_state_ids", ()))
    if not train_states or not val_states or train_states & val_states:
        raise ValueError("decision lock detected missing or overlapping train/val physical states")
    e2_train = dict(e2["train_cache_identity"])
    e2_val = dict(e2["val_cache_identity"])
    e3_train = dict(e3["train_cache_identity"])
    e3_val = dict(e3["val_cache_identity"])
    # Separate cache identities are mandatory; equal cardinalities/config are
    # established by the shared controlled config, while the intervention is
    # encoded by each checkpoint's proprio_mode.
    all_cache_paths = {item["path"] for item in (e2_train, e2_val, e3_train, e3_val)}
    if len(all_cache_paths) != 4:
        raise ValueError("E2/E3 train/val caches must be physically separate")
    selection_cache = selection.get("cache_identity")
    if not isinstance(selection_cache, Mapping):
        raise ValueError("selection cache identity is missing")
    for field in ("path", "size_bytes", "mtime_ns"):
        if selection_cache.get(field) != e2_val.get(field):
            raise ValueError("E2 selection cache is not the E2 validation cache used for training")
    selection_sha = selection_cache.get("sha256")
    if not isinstance(selection_sha, str) or len(selection_sha) != 64:
        raise ValueError("E2 selection cache is missing its SHA-256 identity")
    if selection_sha != e2_val.get("sha256"):
        raise ValueError("E2 selection cache SHA-256 differs from training validation cache")

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"decision lock already exists and is immutable: {output}"
        )
    for test_output in (e2_test_output, e3_test_output):
        test_path = Path(test_output).expanduser().resolve()
        if test_path.exists():
            raise FileExistsError(
                f"held-out test cache already exists before decision lock: {test_path}"
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_layer": selected_layer,
        "r3_access_before_lock": False,
        "selection": {
            "identity": strong_file_identity(selection_source),
            "cache_identity": dict(selection_cache),
            "active_variants": list(selection["active_variants"]),
            "r3_used": False,
        },
        "selection_identity": strong_file_identity(selection_source),
        "train_val_cache_identities": {
            "E2": {"train": e2_train, "val": e2_val},
            "E3": {"train": e3_train, "val": e3_val},
        },
        "checkpoints": {"E2": e2, "E3": e3},
        "shared": {
            "controlled_training_config_sha256": e2["controlled_training_config_sha256"],
            "initial_head_sha256": e2["initial_head_sha256"],
        },
        "expected_test_outputs": {
            "E2": str(Path(e2_test_output).expanduser().resolve()),
            "E3": str(Path(e3_test_output).expanduser().resolve()),
        },
    }
    return write_json(output, payload)


def load_decision_lock(
    path_value: str | Path,
    *,
    experiment: str | None = None,
    expected_test_output: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source, payload = _read_json(path_value)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported E2/E3 decision lock: {source}")
    if payload.get("r3_access_before_lock") is not False:
        raise ValueError("decision lock does not prove pre-test isolation")
    selected_layer = int(payload.get("selected_layer", -1))
    if selected_layer not in (8, 16, 24):
        raise ValueError("decision lock selected layer is invalid")
    experiments = payload.get("checkpoints")
    if not isinstance(experiments, Mapping) or set(experiments) != set(EXPECTED):
        raise ValueError("decision lock experiment set is invalid")
    if experiment is not None:
        if experiment not in EXPECTED:
            raise ValueError(f"unknown experiment {experiment!r}")
        record = experiments.get(experiment)
        if not isinstance(record, Mapping) or record.get("proprio_mode") != EXPECTED[experiment]:
            raise ValueError(f"decision lock does not contain valid {experiment}")
        if expected_test_output is not None:
            expected = str(Path(expected_test_output).expanduser().resolve())
            outputs = payload.get("expected_test_outputs")
            if not isinstance(outputs, Mapping) or outputs.get(experiment) != expected:
                raise ValueError(f"decision lock does not authorize test output {expected}")
    return dict(payload), strong_file_identity(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--e3-checkpoint", required=True)
    parser.add_argument("--e2-test-output", required=True)
    parser.add_argument("--e3-test-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = create_decision_lock(
        selection_path=args.selection,
        e2_checkpoint=args.e2_checkpoint,
        e3_checkpoint=args.e3_checkpoint,
        e2_test_output=args.e2_test_output,
        e3_test_output=args.e3_test_output,
        output_path=args.output,
    )
    print(f"created immutable pre-test decision lock: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "PROTOCOL",
    "create_decision_lock",
    "load_decision_lock",
    "sha256_file",
    "strong_file_identity",
]
