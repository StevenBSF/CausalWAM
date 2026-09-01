#!/usr/bin/env python3
"""Evaluate strict R3-holdout E2/E3 representation controls.

Validation is deliberately limited to the E2 raw-backbone layer-selection
control over Clean/R1/R2.  Held-out testing requires an immutable decision
lock created after both best-validation checkpoints and accepts only
Clean/R3.  Thus this module cannot accidentally turn R3 into a selection set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import load_cache, representation_records
from .data import compatible_state_vectors
from .head import ContrastiveContentHead
from .io_utils import file_identity, module_state_sha256, write_csv, write_json
from .metrics import compute_representation_metrics
from .negatives import build_state_negative_mask


PROTOCOL = "r3_holdout_v1"
SEEN_VARIANTS = ("clean", "style_00_seed_0", "style_01_seed_1")
TEST_VARIANTS = ("clean", "style_02_seed_2")
HOLDOUT_VARIANT = "style_02_seed_2"
PROPRIO_MODES = {
    "E2": "observed",
    "E3": "constant_zero_normalized",
}
EXPERIMENTS = (
    "E2-RawBackbone",
    "E2-InitHead",
    "E2-TrainedHead",
    "E3-NoProprio-RawBackbone",
    "E3-NoProprio-InitHead",
    "E3-NoProprio-TrainedHead",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _experiment_spec(label: str) -> tuple[str, str, str]:
    if label not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    experiment = "E3" if label.startswith("E3-") else "E2"
    if label.endswith("RawBackbone"):
        control = "raw"
    elif label.endswith("InitHead"):
        control = "init"
    else:
        control = "trained"
    return experiment, PROPRIO_MODES[experiment], control


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _strong_file_identity(path: str | Path) -> dict[str, Any]:
    before = file_identity(path)
    sha256 = _sha256_file(path)
    after = file_identity(path)
    if before != after:
        raise RuntimeError(f"file changed while hashing: {Path(path).resolve()}")
    return {**after, "sha256": sha256}


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validate_proprio_audit(provenance: Mapping[str, Any], mode: str) -> None:
    backbone = _require_mapping(provenance.get("backbone"), label="cache backbone provenance")
    if backbone.get("proprio_mode") != mode:
        raise ValueError("cache backbone proprio_mode mismatch")
    conditions = _require_mapping(
        provenance.get("conditions_by_physical_state"),
        label="conditions_by_physical_state",
    )
    if not conditions:
        raise ValueError("cache contains no per-state condition provenance")
    effective_hashes: set[str] = set()
    for physical_state_id, condition_value in conditions.items():
        condition = _require_mapping(
            condition_value, label=f"condition provenance {physical_state_id}"
        )
        context = _require_mapping(
            condition.get("context"), label=f"condition context {physical_state_id}"
        )
        proprio = _require_mapping(
            context.get("proprio"), label=f"condition proprio {physical_state_id}"
        )
        if proprio.get("mode") != mode:
            raise ValueError(f"{physical_state_id}: proprio intervention mode mismatch")
        if proprio.get("intervention_point") != "post_normalizer_pre_proprio_encoder":
            raise ValueError(f"{physical_state_id}: unexpected proprio intervention point")
        if proprio.get("proprio_token_preserved") is not True:
            raise ValueError(f"{physical_state_id}: proprio token was not preserved")
        observed_hash = str(proprio.get("observed_normalized_sha256", ""))
        effective_hash = str(proprio.get("effective_normalized_sha256", ""))
        if not observed_hash or not effective_hash:
            raise ValueError(f"{physical_state_id}: missing proprio audit hashes")
        if (
            _SHA256_RE.fullmatch(observed_hash) is None
            or _SHA256_RE.fullmatch(effective_hash) is None
        ):
            raise ValueError(f"{physical_state_id}: malformed proprio audit hash")
        outer_hash = condition.get("normalized_proprio_sha256")
        if outer_hash is not None and outer_hash != effective_hash:
            raise ValueError(f"{physical_state_id}: outer/effective proprio hash mismatch")
        effective_hashes.add(effective_hash)
        if mode == "observed":
            if observed_hash != effective_hash:
                raise ValueError(f"{physical_state_id}: E2 changed observed proprio")
        else:
            if proprio.get("all_zero") is not True:
                raise ValueError(f"{physical_state_id}: E3 effective proprio is not zero")
    if mode == "constant_zero_normalized" and len(effective_hashes) != 1:
        raise ValueError("E3 constant-zero proprio hashes differ across physical states")


def _validate_cache_protocol(
    cache: Mapping[str, Any],
    *,
    split: str,
    mode: str,
) -> tuple[str, ...]:
    if split not in {"val", "test"}:
        raise ValueError("E2/E3 representation evaluation accepts only val/test")
    expected_variants = SEEN_VARIANTS if split == "val" else TEST_VARIANTS
    provenance = _require_mapping(cache.get("provenance"), label="cache provenance")
    if provenance.get("protocol") != PROTOCOL:
        raise ValueError(f"cache protocol must be {PROTOCOL!r}")
    if provenance.get("split") != split:
        raise ValueError(f"cache provenance split must be {split!r}")
    if tuple(provenance.get("active_variants", ())) != expected_variants:
        raise ValueError(f"{split} active_variants must be exactly {expected_variants}")
    if provenance.get("holdout_variant") != HOLDOUT_VARIANT:
        raise ValueError("cache holdout_variant mismatch")
    if provenance.get("proprio_mode") != mode:
        raise ValueError("cache proprio_mode does not match the requested control")
    if tuple(cache.get("variant_names", ())) != expected_variants:
        raise ValueError(f"cache variant_names must be exactly {expected_variants}")
    if int(cache.get("variants_per_state", -1)) != len(expected_variants):
        raise ValueError("cache variants_per_state disagrees with the protocol")
    records = cache.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("cache records must be a non-empty list")
    if {str(record.get("split")) for record in records} != {split}:
        raise ValueError(f"cache records are not exclusively from split {split!r}")
    if {str(record.get("variant")) for record in records} != set(expected_variants):
        raise ValueError(f"cache records do not contain exactly {expected_variants}")
    conditions = _require_mapping(
        provenance.get("conditions_by_physical_state"),
        label="conditions_by_physical_state",
    )
    expected_state_ids = {str(record.get("physical_state_id")) for record in records}
    if set(conditions) != expected_state_ids:
        raise ValueError("condition provenance keys do not exactly match cache states")
    if split == "val" and HOLDOUT_VARIANT in {
        str(record.get("variant")) for record in records
    }:
        raise ValueError("R3 leaked into validation records")
    _validate_proprio_audit(provenance, mode)
    return expected_variants


def _standardized_state_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    _, left_values, right_values = compatible_state_vectors(left, right)
    scale = np.maximum(np.maximum(np.abs(left_values), np.abs(right_values)), 1.0)
    return float(np.linalg.norm((left_values - right_values) / scale) / np.sqrt(scale.size))


def _dynamic_state_negative_pairs(
    cache: Mapping[str, Any],
    *,
    variants: Sequence[str],
    min_temporal_gap: int,
    min_state_distance: float,
) -> list[tuple[int, int]]:
    records = cache["records"]
    states = cache.get("physical_states")
    if not isinstance(states, list) or not states:
        raise ValueError("cache physical_states must be a non-empty list")
    views = len(variants)
    if len(records) != len(states) * views:
        raise ValueError("cache records/physical_states do not form K-view groups")
    clean_records: list[Mapping[str, Any]] = []
    clean_indices: list[int] = []
    for group_index in range(len(states)):
        start = group_index * views
        group = records[start : start + views]
        if tuple(str(record.get("variant")) for record in group) != tuple(variants):
            raise ValueError(f"group {group_index} violates canonical variant order")
        keys = {
            (str(record.get("task")), str(record.get("physical_state_id")))
            for record in group
        }
        if len(keys) != 1:
            raise ValueError(f"group {group_index} does not represent one physical state")
        clean_records.append(group[0])
        clean_indices.append(start)
    admissible = build_state_negative_mask(
        clean_records,
        states,
        min_temporal_gap=min_temporal_gap,
        min_state_distance=min_state_distance,
    )
    pairs: list[tuple[int, int]] = []
    for anchor_group, anchor_index in enumerate(clean_indices):
        candidates = np.flatnonzero(admissible[anchor_group]).tolist()
        if not candidates:
            raise ValueError(f"no state negative for cache group {anchor_group}")
        candidate_group = max(
            candidates,
            key=lambda candidate: (
                abs(
                    int(clean_records[candidate]["timestep"])
                    - int(clean_records[anchor_group]["timestep"])
                ),
                _standardized_state_distance(states[anchor_group], states[candidate]),
                -candidate,
            ),
        )
        pairs.append((anchor_index, clean_indices[candidate_group]))
    return pairs


def _load_checkpoint(
    checkpoint_path: str | Path,
    *,
    experiment: str,
    mode: str,
    layer: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("E2/E3 checkpoint must use schema_version=2")
    expected = {
        "experiment": experiment,
        "protocol": PROTOCOL,
        "proprio_mode": mode,
        "checkpoint_kind": "best_val",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint {key} must be {value!r}")
    if int(payload.get("layer", -1)) != layer:
        raise ValueError("checkpoint layer mismatch")
    if int(payload.get("step", -1)) != int(payload.get("best_step", -2)):
        raise ValueError("checkpoint is not the selected best-validation step")
    selection = _require_mapping(payload.get("best_metric"), label="checkpoint best_metric")
    if (
        selection.get("metric") != "val_contrastive_loss"
        or selection.get("mode") != "min"
        or selection.get("tie_break") != "earliest_step"
        or selection.get("r3_used") is not False
    ):
        raise ValueError("checkpoint was not selected strictly by validation loss")
    controlled = _require_mapping(
        payload.get("controlled_training_config"), label="controlled_training_config"
    )
    if controlled.get("protocol") != PROTOCOL:
        raise ValueError("checkpoint controlled protocol mismatch")
    if tuple(controlled.get("active_variants", ())) != SEEN_VARIANTS:
        raise ValueError("checkpoint training variants are not exactly C/R1/R2")
    if controlled.get("holdout_variant") != HOLDOUT_VARIANT:
        raise ValueError("checkpoint holdout variant mismatch")
    if int(controlled.get("layer", -1)) != layer:
        raise ValueError("checkpoint controlled layer mismatch")
    loss = _require_mapping(controlled.get("loss"), label="checkpoint loss config")
    if loss.get("name") != "multi_positive_supcon" or float(loss.get("temperature", -1)) != 0.07:
        raise ValueError("checkpoint must use SupCon temperature 0.07")
    checkpoint_selection = _require_mapping(
        controlled.get("checkpoint_selection"), label="checkpoint selection config"
    )
    if checkpoint_selection.get("r3_allowed") is not False:
        raise ValueError("checkpoint config allowed R3 during selection")
    if not str(payload.get("controlled_training_config_sha256", "")):
        raise ValueError("checkpoint lacks controlled config hash")
    if not str(payload.get("initial_head_sha256", "")):
        raise ValueError("checkpoint lacks initial head hash")
    head_config = _require_mapping(payload.get("head_config"), label="checkpoint head_config")
    return payload, {**file_identity(source), "sha256": _sha256_file(source), "head_config": dict(head_config)}


def _load_decision_lock(
    lock_path: str | Path,
    *,
    cache: Mapping[str, Any],
    cache_path: str | Path,
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    experiment: str,
    mode: str,
    layer: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(lock_path).expanduser().resolve()
    try:
        lock = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read decision lock {source}: {error}") from error
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("decision lock must use schema_version=1")
    if lock.get("protocol") != PROTOCOL or int(lock.get("selected_layer", -1)) != layer:
        raise ValueError("decision lock protocol/layer mismatch")
    for required in ("created_at_utc", "selection_identity", "train_val_cache_identities"):
        if not lock.get(required):
            raise ValueError(f"decision lock is missing {required}")
    checkpoints = _require_mapping(lock.get("checkpoints"), label="decision lock checkpoints")
    entry = _require_mapping(checkpoints.get(experiment), label=f"decision lock {experiment}")
    expected_entry = {
        "experiment": experiment,
        "proprio_mode": mode,
        "best_step": int(checkpoint["best_step"]),
        "controlled_training_config_sha256": checkpoint["controlled_training_config_sha256"],
        "initial_head_sha256": checkpoint["initial_head_sha256"],
    }
    for key, value in expected_entry.items():
        if entry.get(key) != value:
            raise ValueError(f"decision lock checkpoint {key} mismatch")
    recorded_checkpoint_identity = entry.get("checkpoint", entry)
    recorded_checkpoint_identity = _require_mapping(
        recorded_checkpoint_identity,
        label=f"decision lock {experiment} checkpoint identity",
    )
    for key in ("path", "size_bytes", "mtime_ns", "sha256"):
        if recorded_checkpoint_identity.get(key) != checkpoint_identity.get(key):
            raise ValueError(f"decision lock checkpoint identity {key} mismatch")
    shared = _require_mapping(lock.get("shared"), label="decision lock shared controls")
    for key in ("controlled_training_config_sha256", "initial_head_sha256"):
        if shared.get(key) != checkpoint.get(key):
            raise ValueError(f"decision lock shared {key} mismatch")

    expected_outputs = _require_mapping(
        lock.get("expected_test_outputs"), label="decision lock expected_test_outputs"
    )
    expected_output = expected_outputs.get(experiment)
    expected_path = (
        expected_output.get("path") if isinstance(expected_output, Mapping) else expected_output
    )
    if not expected_path or Path(expected_path).expanduser().resolve() != Path(cache_path).expanduser().resolve():
        raise ValueError("test cache path is not the decision-locked expected output")

    lock_identity = _strong_file_identity(source)
    provenance = _require_mapping(cache.get("provenance"), label="test cache provenance")
    recorded_identity = _require_mapping(
        provenance.get("decision_lock_identity"), label="cache decision_lock_identity"
    )
    if dict(recorded_identity) != lock_identity:
        raise ValueError("test cache decision-lock identity mismatch")
    if provenance.get("decision_lock_created_before_test") is not True:
        raise ValueError("test cache does not prove that the decision lock preceded extraction")
    return lock, lock_identity


def _head_from_payload(
    payload: Mapping[str, Any], *, device: torch.device, load_trained: bool
) -> ContrastiveContentHead:
    head = ContrastiveContentHead(**dict(payload["head_config"]))
    if load_trained:
        incompatible = head.load_state_dict(payload["head"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"strict checkpoint load failed: {incompatible}")
    return head.to(device).eval()


def evaluate_e2e3_cache(
    *,
    cache_path: str | Path,
    layer: int,
    experiment: str,
    output_dir: str | Path,
    head_checkpoint: str | Path | None = None,
    decision_lock: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
    min_temporal_gap: int = 8,
    min_state_distance: float = 1e-5,
    inference_batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Evaluate one strict control; final test cannot run without its lock."""

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    base_experiment, mode, control = _experiment_spec(experiment)
    cache_identity = _strong_file_identity(cache_path)
    cache = load_cache(cache_path)
    record_splits = {str(record.get("split")) for record in cache["records"]}
    if len(record_splits) != 1 or next(iter(record_splits)) not in {"val", "test"}:
        raise ValueError("cache must contain exactly one val or test split")
    split = next(iter(record_splits))
    variants = _validate_cache_protocol(cache, split=split, mode=mode)
    if split == "val":
        if experiment != "E2-RawBackbone":
            raise ValueError("validation is reserved for E2 raw layer selection")
        if head_checkpoint is not None or decision_lock is not None:
            raise ValueError("validation layer selection accepts no checkpoint/decision lock")
    elif head_checkpoint is None or decision_lock is None:
        raise ValueError("held-out R3 test requires both head_checkpoint and decision_lock")

    layer_key = str(int(layer))
    if layer_key not in cache["tokens_by_layer"]:
        raise KeyError(f"cache does not contain layer {layer}")
    tokens = cache["tokens_by_layer"][layer_key]
    records = representation_records(cache)
    negative_pairs = _dynamic_state_negative_pairs(
        cache,
        variants=variants,
        min_temporal_gap=min_temporal_gap,
        min_state_distance=min_state_distance,
    )

    checkpoint_payload: dict[str, Any] | None = None
    checkpoint_identity: dict[str, Any] | None = None
    lock_payload: dict[str, Any] | None = None
    lock_identity: dict[str, Any] | None = None
    if split == "test":
        checkpoint_payload, checkpoint_identity = _load_checkpoint(
            head_checkpoint,
            experiment=base_experiment,
            mode=mode,
            layer=int(layer),
        )
        if int(checkpoint_payload["head_config"]["backbone_dim"]) != int(tokens.shape[-1]):
            raise ValueError("checkpoint/cache backbone dimension mismatch")
        lock_payload, lock_identity = _load_decision_lock(
            decision_lock,
            cache=cache,
            cache_path=cache_path,
            checkpoint=checkpoint_payload,
            checkpoint_identity=checkpoint_identity,
            experiment=base_experiment,
            mode=mode,
            layer=int(layer),
        )

    execution_device = torch.device(device)
    head_info: dict[str, Any] | None = None
    if control == "raw":
        embeddings = cache["pooled_by_layer"][layer_key].float()
        if checkpoint_payload is not None:
            head_info = {
                "control": "raw_backbone",
                "paired_checkpoint_identity": checkpoint_identity,
                "initial_head_sha256": checkpoint_payload["initial_head_sha256"],
                "training_seed": int(checkpoint_payload["seed"]),
                "checkpoint_kind": "best_val",
            }
    else:
        if checkpoint_payload is None:
            raise AssertionError("head control reached without test checkpoint")
        if control == "init":
            checkpoint_seed = int(checkpoint_payload["seed"])
            if seed != checkpoint_seed:
                raise ValueError(
                    f"InitHead seed must match training seed {checkpoint_seed}, got {seed}"
                )
            torch.manual_seed(seed)
            head = _head_from_payload(checkpoint_payload, device=execution_device, load_trained=False)
            initial_hash = module_state_sha256(head)
            if initial_hash != checkpoint_payload["initial_head_sha256"]:
                raise ValueError("reconstructed InitHead does not match the trained run initialization")
            head_info = {
                "control": "initial_head",
                "initialization_seed": seed,
                "initial_head_sha256": initial_hash,
                "paired_checkpoint_identity": checkpoint_identity,
                "trainable_parameter_count": head.trainable_parameter_count(),
                "checkpoint_kind": "best_val",
            }
        else:
            head = _head_from_payload(checkpoint_payload, device=execution_device, load_trained=True)
            head_info = {
                "control": "trained_best_validation_head",
                "checkpoint_identity": checkpoint_identity,
                "checkpoint_step": int(checkpoint_payload["step"]),
                "best_step": int(checkpoint_payload["best_step"]),
                "training_seed": int(checkpoint_payload["seed"]),
                "checkpoint_kind": str(checkpoint_payload["checkpoint_kind"]),
                "initial_head_sha256": checkpoint_payload["initial_head_sha256"],
                "controlled_training_config_sha256": checkpoint_payload[
                    "controlled_training_config_sha256"
                ],
                "trainable_parameter_count": head.trainable_parameter_count(),
            }
        with torch.inference_mode():
            batches = [
                head(tokens[start : start + inference_batch_size].to(execution_device, dtype=torch.float32)).cpu()
                for start in range(0, len(tokens), inference_batch_size)
            ]
            embeddings = torch.cat(batches, dim=0)

    canonical_styles = ("r1", "r2") if split == "val" else ("r3",)
    canonical_variants = ("clean", *canonical_styles)
    rows = compute_representation_metrics(
        embeddings,
        records,
        layer=f"video_block_{int(layer):02d}",
        experiment=experiment,
        state_negative_pairs=negative_pairs,
        min_temporal_gap=min_temporal_gap,
        style_order=canonical_styles,
        required_variants=canonical_variants,
    )
    if split == "test":
        for row in rows:
            # Explicit protocol names make downstream tables unambiguous: the
            # generic fields are exact R3 quantities here, never seen-style
            # averages.
            row["style_distance_R3"] = row["clean_r3_distance"]
            row["state_style_ratio_R3"] = row["state_style_ratio"]
            row["R3_to_Clean_R@1"] = row["r3_to_clean_retrieval_at1"]
            row["R3_to_Clean_R@5"] = row["r3_to_clean_retrieval_at5"]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    slug = experiment.lower().replace("-", "_")
    result = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "proprio_mode": mode,
        "active_variants": list(variants),
        "record_variants": list(variants),
        "holdout_variant": HOLDOUT_VARIANT,
        "evaluation_split": split,
        "r3_used_for_selection": False,
        "layer": int(layer),
        "experiment": experiment,
        "cache": str(Path(cache_path).expanduser().resolve()),
        "cache_identity": cache_identity,
        "cache_provenance": cache["provenance"],
        "decision_lock": None if decision_lock is None else str(Path(decision_lock).expanduser().resolve()),
        "decision_lock_identity": lock_identity,
        "decision_lock_payload": lock_payload,
        "head": head_info,
        "negative_filter": {
            "min_temporal_gap": int(min_temporal_gap),
            "min_state_distance": float(min_state_distance),
            "num_pairs": len(negative_pairs),
        },
        "metric_protocol": {
            "style_order": list(canonical_styles),
            "required_variants": list(canonical_variants),
            "query": "R3" if split == "test" else "mean(R1,R2)",
            "gallery": "Clean",
        },
        "metrics": rows,
    }
    output_stem = f"{slug}_layer_{int(layer):02d}"
    write_json(destination / f"{output_stem}.json", result)
    write_csv(destination / f"{output_stem}.csv", rows)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-checkpoint")
    parser.add_argument("--decision-lock")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-temporal-gap", type=int, default=8)
    parser.add_argument("--min-state-distance", type=float, default=1e-5)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = _parser().parse_args()
    rows = evaluate_e2e3_cache(
        cache_path=args.cache,
        layer=args.layer,
        experiment=args.experiment,
        output_dir=args.output_dir,
        head_checkpoint=args.head_checkpoint,
        decision_lock=args.decision_lock,
        seed=args.seed,
        device=args.device,
        min_temporal_gap=args.min_temporal_gap,
        min_state_distance=args.min_state_distance,
        inference_batch_size=args.inference_batch_size,
    )
    for row in rows:
        print(
            f"{row['task']:>22}  style={row['style_distance']:.6f}  "
            f"state={row['state_distance']:.6f}  ratio={row['state_style_ratio']:.3f}  "
            f"R@1={row['retrieval_r1']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()


__all__ = ["EXPERIMENTS", "evaluate_e2e3_cache"]
