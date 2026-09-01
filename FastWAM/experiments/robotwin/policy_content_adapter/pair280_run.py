#!/usr/bin/env python3
"""Materialize, run, and audit seed-1/C3 Pair-280 post-training."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import secrets
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import train as train_module
from .config_audit import load_config
from .materialize_release_engineering_smoke import _write_new_yaml
from .model import artifact_identity
from .pair280_protocol import (
    PAIR280_ACTIVE_STEPS,
    PAIR280_CACHE_STORAGE,
    PAIR280_GROUPS,
    PAIR280_PROFILE_ID,
    PAIR280_STATES_PER_TRAJECTORY,
    PAIR280_TOTAL_STEPS,
    validate_pair280_cache_manifest,
    verify_pair280_state_bank,
    paired_is_active,
)
from .pair280_sampler import (
    PAIR280_SAMPLER_ID,
    audit_state_bank_global_distinct,
)
from .prepare_release_paired_text_cache import verify_release_paired_text_cache
from .release_lineage import verify_author_release_lineage
from .release_official_text_cache_binding import verify_binding as verify_official_text_binding
from .release_paired_binding import verify_release_paired_binding
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pair280_seed1_c3_run"
SCHEMA_VERSION = 1
ARTIFACT_ROOT = Path("/root/fastwam_policy_artifacts/pair280_layer16_v1").resolve()
SOURCE_CONFIG = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "pv2_actiondit_full5ep_v1_retry2/configs/seed_1/c3.yaml"
).resolve()
SOURCE_CONFIG_SHA256 = "6258c58b97bfc473e8bc5736506932ea5f6fb46536feec53a821835bc0f993b7"
PROTOCOL_PATH = ARTIFACT_ROOT / "pair280_protocol.json"
PROTOCOL_SHA256 = "1672a83ba64449fcb917297eb26c51aa9503730cbfe76fbd926483cb1334e2be"
STATE_BANK_PATH = ARTIFACT_ROOT / "pair280_state_bank.json"
STATE_BANK_SHA256 = "d970805d489ce27a9b8b6b516d3f9c215de89940d7711e6fe2f8966627cdf710"
BINDING_PATH = ARTIFACT_ROOT / "pair280_release_binding.json"
BINDING_SHA256 = "d2b17754a9537a099dcd01dce202eb2057c9aef8d01fb84c7e51dfe1bc504084"
TEXT_CACHE_PATH = ARTIFACT_ROOT / "paired_text_cache"
TEXT_CACHE_SHA256 = "56dff1d7f6dd05e7ac2e89570667962d8a58881cfbc416440ed5656a5cffd3df"
CACHE_MANIFEST_PATH = ARTIFACT_ROOT / "cache_manifest.json"
RUN_ROOT = ARTIFACT_ROOT / "seed1_c3_pair280_posttraining_v1"
PAIRED_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1/"
    "full_lerobot_v21"
).resolve()
PAIRED_MANIFEST = PAIRED_ROOT / "meta/policy_native_action_manifest.json"
PAIRED_AUDIT = PAIRED_ROOT / "meta/policy_native_action_audit.json"


class Pair280RunError(ValueError):
    """Pair-280 run config or outputs differ from the preregistered recipe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pair280RunError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _artifact_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    identity = artifact_identity(path)
    _require(identity["sha256"] == expected, f"{label} SHA-256 changed")
    return identity


def _derive_config(source: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    cache_identity = artifact_identity(CACHE_MANIFEST_PATH)
    config = copy.deepcopy(dict(source))
    config["experiment_id"] = (
        "pair280_seed1_c3_pv2_smoke_v1"
        if smoke
        else "pair280_seed1_c3_pv2_full5ep_v1"
    )
    config["output_dir"] = str(RUN_ROOT / ("smoke" if smoke else "formal"))
    config["execution"] = {
        "runner": "policy_pair280_posttraining",
        "runnable": True,
        "fail_closed": False,
        "long_formal_training": not smoke,
    }
    config["release_paired_binding_manifest"] = str(BINDING_PATH)
    config["artifacts"]["release_paired_binding_manifest_sha256"] = BINDING_SHA256
    config["artifacts"]["paired_state_bank_sha256"] = STATE_BANK_SHA256
    config["artifacts"]["paired_text_cache_sha256"] = TEXT_CACHE_SHA256
    config["artifacts"]["paired_cache_sha256"] = cache_identity["sha256"]
    config["paired"]["state_bank"] = str(STATE_BANK_PATH)
    config["paired"]["text_cache_dir"] = str(TEXT_CACHE_PATH)
    config["paired"]["cache"] = str(CACHE_MANIFEST_PATH)
    config["paired"]["cache_format"] = PAIR280_CACHE_STORAGE
    config["paired"]["sampling_profile"] = PAIR280_PROFILE_ID
    config["paired"]["engineering_smoke"] = bool(smoke)
    config["paired"]["schedule"] = {
        "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
        "physical_state_groups": PAIR280_GROUPS,
        "paired_epochs": 10,
        "active_steps": PAIR280_ACTIVE_STEPS,
        "total_steps": PAIR280_TOTAL_STEPS,
        "global_groups_per_active_step": 16,
        "sampler": PAIR280_SAMPLER_ID,
        "active_step_distribution": "floor_difference_v1",
    }
    config["training"]["max_steps"] = 3 if smoke else PAIR280_TOTAL_STEPS
    config["training"]["save_every"] = 3 if smoke else 2_000
    config["training"]["resume"] = None
    config["training"]["epoch_contract"] = {
        "official_dataset_samples": 466_240,
        "official_epochs": 5,
        "official_steps_per_epoch": 3_643,
        "max_steps": PAIR280_TOTAL_STEPS,
        "paired_physical_state_groups": PAIR280_GROUPS,
        "paired_epochs": 10,
        "paired_active_steps": PAIR280_ACTIVE_STEPS,
        "paired_inactive_steps": PAIR280_TOTAL_STEPS - PAIR280_ACTIVE_STEPS,
    }
    config["pair280_protocol_manifest"] = str(PROTOCOL_PATH)
    config["pair280_protocol_manifest_sha256"] = PROTOCOL_SHA256
    sampler_source = (
        PROJECT_ROOT
        / "experiments/robotwin/policy_content_adapter/pair280_sampler.py"
    ).resolve()
    config["pair280_sampler_source"] = str(sampler_source)
    config["pair280_sampler_source_sha256"] = artifact_identity(sampler_source)[
        "sha256"
    ]
    config.pop("full5ep_protocol_manifest", None)
    config.pop("full5ep_protocol_manifest_sha256", None)
    config["full5ep_execution_profile"] = "pair280_global128_exact10"
    return config


def validate_config(path_or_config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    config = (
        copy.deepcopy(dict(path_or_config))
        if isinstance(path_or_config, Mapping)
        else load_config(path_or_config)
    )
    _artifact_sha(SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "source full5ep C3 config")
    _artifact_sha(PROTOCOL_PATH, PROTOCOL_SHA256, "Pair-280 protocol")
    _artifact_sha(STATE_BANK_PATH, STATE_BANK_SHA256, "Pair-280 state bank")
    _artifact_sha(BINDING_PATH, BINDING_SHA256, "Pair-280 release binding")
    _artifact_sha(TEXT_CACHE_PATH, TEXT_CACHE_SHA256, "Pair-280 text cache")
    _artifact_sha(
        Path(config["pair280_sampler_source"]),
        str(config["pair280_sampler_source_sha256"]),
        "Pair-280 sampler source",
    )
    source = load_config(SOURCE_CONFIG)
    smoke = bool(config.get("paired", {}).get("engineering_smoke", False))
    expected = _derive_config(source, smoke=smoke)
    _require(config == expected, "Pair-280 config differs outside the authorized transform")
    train_module.validate_run_config(config)
    lineage = verify_author_release_lineage(
        config["base_lineage_manifest"],
        checkpoint_path=config["base_checkpoint"],
        dataset_stats_path=config["official"]["dataset_stats"],
        official_manifest_path=config["official"]["canonical_task_manifest"],
        expected_manifest_sha256=config["artifacts"]["base_lineage_manifest_sha256"],
    )
    binding = verify_release_paired_binding(
        BINDING_PATH, expected_sha256=BINDING_SHA256
    )
    bank = verify_pair280_state_bank(
        STATE_BANK_PATH,
        paired_root=PAIRED_ROOT,
        paired_manifest=PAIRED_MANIFEST,
        paired_audit=PAIRED_AUDIT,
        expected_sha256=STATE_BANK_SHA256,
    )
    text = verify_release_paired_text_cache(
        TEXT_CACHE_PATH,
        expected_base_lineage_sha256=lineage["manifest_identity"]["sha256"],
        expected_release_paired_binding_sha256=BINDING_SHA256,
    )
    cache = validate_pair280_cache_manifest(
        CACHE_MANIFEST_PATH,
        expected_manifest_sha256=config["artifacts"]["paired_cache_sha256"],
        expected_state_bank_sha256=STATE_BANK_SHA256,
        expected_release_binding_sha256=BINDING_SHA256,
        verify_shard_hashes=False,
    )
    official_text = verify_official_text_binding(
        config["official"]["text_cache_binding_manifest"],
        expected_sha256=config["artifacts"]["official_text_cache_binding_manifest_sha256"],
        expected_base_lineage_sha256=lineage["manifest_identity"]["sha256"],
        expected_cache_dir=config["official"]["text_cache_dir"],
    )
    return {
        "status": "PASS",
        "kind": KIND,
        "smoke": smoke,
        "training_seed": 1,
        "control": "c3_ours",
        "regime": "p_v2",
        "steps": 3 if smoke else PAIR280_TOTAL_STEPS,
        "state_bank_sha256": bank.sha256,
        "release_binding_sha256": binding["binding_manifest_identity"]["sha256"],
        "paired_text_cache_sha256": text["directory_identity"]["sha256"],
        "cache_manifest_sha256": cache["manifest_identity"]["sha256"],
        "official_text_binding_sha256": artifact_identity(
            config["official"]["text_cache_binding_manifest"]
        )["sha256"],
    }


def materialize() -> dict[str, Any]:
    _require(CACHE_MANIFEST_PATH.is_file(), "Pair-280 cache must finish before config materialization")
    materialization = RUN_ROOT / "materialization.json"
    _require(not materialization.exists(), f"refusing to overwrite {materialization}")
    source = load_config(SOURCE_CONFIG)
    state_bank = verify_pair280_state_bank(
        STATE_BANK_PATH,
        paired_root=PAIRED_ROOT,
        paired_manifest=PAIRED_MANIFEST,
        paired_audit=PAIRED_AUDIT,
        expected_sha256=STATE_BANK_SHA256,
    )
    sampler_audit = audit_state_bank_global_distinct(state_bank, seed=1)
    configs: dict[str, Any] = {}
    for name, smoke in (("smoke", True), ("formal", False)):
        config = _derive_config(source, smoke=smoke)
        path = RUN_ROOT / "configs" / f"seed1_c3_{name}.yaml"
        _write_new_yaml(path, config)
        audit = validate_config(path)
        configs[name] = {"identity": artifact_identity(path), "audit": audit}
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "policy_pair280_materialization",
        "status": "PASS",
        "profile_id": PAIR280_PROFILE_ID,
        "source_config": artifact_identity(SOURCE_CONFIG),
        "protocol": artifact_identity(PROTOCOL_PATH),
        "state_bank": artifact_identity(STATE_BANK_PATH),
        "release_binding": artifact_identity(BINDING_PATH),
        "paired_text_cache": artifact_identity(TEXT_CACHE_PATH),
        "cache_manifest": artifact_identity(CACHE_MANIFEST_PATH),
        "sampler_source": artifact_identity(
            PROJECT_ROOT
            / "experiments/robotwin/policy_content_adapter/pair280_sampler.py"
        ),
        "sampler_audit": sampler_audit,
        "configs": configs,
    }
    _write_new_json(materialization, result)
    return result


def run_training(config_path: str | Path, *, resume: str | Path | None) -> Path:
    validate_config(config_path)
    original = train_module.validate_execution_ready
    train_module.validate_execution_ready = validate_config
    try:
        return train_module.run(config_path, resume_from=resume)
    finally:
        train_module.validate_execution_ready = original


def audit_run(run_root: str | Path, *, smoke: bool) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    expected_steps = 3 if smoke else PAIR280_TOTAL_STEPS
    summary = json.loads((root / "training_summary.json").read_text(encoding="utf-8"))
    sequence = json.loads((root / "training_sequence_audit.json").read_text(encoding="utf-8"))
    gradient = json.loads((root / "gradient_audit.json").read_text(encoding="utf-8"))
    _require(int(summary.get("steps", -1)) == expected_steps, "Pair-280 run step count changed")
    _require(summary.get("regime") == "p_v2" and summary.get("control") == "c3_ours", "Pair-280 run identity changed")
    _require(summary.get("paired_sampling_profile") == PAIR280_PROFILE_ID, "Pair-280 summary profile changed")
    expected_paired = 32 if smoke else PAIR280_GROUPS * 10
    _require(int(sequence.get("paired_physical_state_count", -1)) == expected_paired, "Pair-280 paired exposure count changed")
    _require(gradient.get("status") == "PASS", "Pair-280 gradient audit failed")
    with (root / "train_log.csv").open("r", encoding="utf-8", newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    _require(len(train_rows) == expected_steps, "Pair-280 train-log row count changed")
    _require(
        [int(row["step"]) for row in train_rows] == list(range(1, expected_steps + 1)),
        "Pair-280 train-log steps are not exact and contiguous",
    )
    active_rows = [
        row for row in train_rows if row.get("paired_contrastive_active") == "True"
    ]
    expected_active = 2 if smoke else PAIR280_ACTIVE_STEPS
    _require(len(active_rows) == expected_active, "Pair-280 active train-log count changed")
    active_indices = [int(row["paired_active_index"]) for row in active_rows]
    _require(
        active_indices == list(range(expected_active)),
        "Pair-280 active indices are not exact and contiguous",
    )
    exposure_counts: Counter[str] = Counter()
    for row in train_rows:
        active = row.get("paired_contrastive_active") == "True"
        _require(
            active is paired_is_active(int(row["step"])),
            "Pair-280 active/inactive placement differs from the uniform schedule",
        )
        state_ids = [
            value
            for value in row.get("paired_physical_state_ids", "").split(";")
            if value
        ]
        if active:
            _require(len(state_ids) == 16, "Pair-280 active step does not contain 16 groups")
            _require(len(state_ids) == len(set(state_ids)), "Pair-280 active step repeats a state")
            trajectories_by_task: dict[str, list[str]] = {}
            for state_id in state_ids:
                parts = state_id.split("/")
                _require(len(parts) == 3, "Pair-280 physical-state id format changed")
                task, content = parts[:2]
                trajectories_by_task.setdefault(task, []).append(f"{task}/{content}")
                exposure_counts[state_id] += 1
            _require(
                all(
                    len(trajectories) == len(set(trajectories))
                    for trajectories in trajectories_by_task.values()
                ),
                "Pair-280 train log repeats a same-task trajectory within one global step",
            )
        else:
            _require(not state_ids, "Pair-280 inactive step consumed paired state ids")
            _require(
                float(row["loss_contrastive"]) == 0.0,
                "Pair-280 inactive step has nonzero contrastive loss",
            )
            _require(
                row.get("paired_active_index", "") == "",
                "Pair-280 inactive step has a paired active index",
            )
    if not smoke:
        rows = summary.get("paired_task_sequence")
        _require(isinstance(rows, list) and len(rows) == expected_paired, "Pair-280 paired task history changed")
        counts = Counter(rows)
        _require(set(counts.values()) == {84_000}, "Pair-280 task exposure counts are not balanced")
        _require(
            len(exposure_counts) == PAIR280_GROUPS
            and set(exposure_counts.values()) == {10},
            "Pair-280 train log does not prove exactly ten exposures per state",
        )
        checkpoints = root / "checkpoints/state"
        expected_boundaries = list(range(2_000, 18_001, 2_000)) + [18_215]
        for step in expected_boundaries:
            state = checkpoints / f"step_{step:08d}" / "trainer_state.json"
            _require(state.is_file(), f"Pair-280 resume checkpoint missing at step {step}")
            payload = json.loads(state.read_text(encoding="utf-8"))
            _require(int(payload["global_step"]) == step, f"checkpoint {step} metadata changed")
    return {
        "status": "PASS",
        "kind": "policy_pair280_posttraining_audit",
        "smoke": smoke,
        "steps": expected_steps,
        "paired_exposures": expected_paired,
        "paired_active_steps": len(active_rows),
        "paired_unique_states": len(exposure_counts),
        "paired_exposures_per_state": (
            10 if not smoke else "smoke_subset_only"
        ),
        "inactive_steps_action_only_verified": expected_steps - len(active_rows),
        "checkpoint": artifact_identity(root / "checkpoint.pt"),
        "training_summary": artifact_identity(root / "training_summary.json"),
        "training_sequence_audit": artifact_identity(root / "training_sequence_audit.json"),
        "gradient_audit": artifact_identity(root / "gradient_audit.json"),
    }


def _correct_formal_summary_labels(
    summary: Mapping[str, Any],
    requested_config: Mapping[str, Any],
    strict_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a provenance-preserving formal view of a completed summary.

    The first Pair-280 formal run encoded its authority in
    ``execution.long_formal_training`` while the generic summary writer only
    inspected the optional top-level ``formal`` key.  The optimizer work and
    strict Pair-280 audit are unaffected.  This helper permits only that exact
    two-field label correction and records the original values explicitly.
    """

    execution = requested_config.get("execution")
    _require(isinstance(execution, Mapping), "Pair-280 execution contract missing")
    _require(
        execution.get("runner") == "policy_pair280_posttraining"
        and execution.get("long_formal_training") is True,
        "Pair-280 requested config is not a formal post-training run",
    )
    training = requested_config.get("training")
    paired = requested_config.get("paired")
    _require(isinstance(training, Mapping), "Pair-280 training contract missing")
    _require(isinstance(paired, Mapping), "Pair-280 paired contract missing")
    _require(
        int(training.get("max_steps", -1)) == PAIR280_TOTAL_STEPS,
        "Pair-280 formal max steps changed",
    )
    _require(
        paired.get("sampling_profile") == PAIR280_PROFILE_ID,
        "Pair-280 formal sampling profile changed",
    )
    _require(
        strict_audit.get("status") == "PASS"
        and strict_audit.get("smoke") is False
        and int(strict_audit.get("steps", -1)) == PAIR280_TOTAL_STEPS
        and int(strict_audit.get("paired_active_steps", -1))
        == PAIR280_ACTIVE_STEPS
        and int(strict_audit.get("paired_unique_states", -1)) == PAIR280_GROUPS
        and strict_audit.get("paired_exposures_per_state") == 10,
        "strict Pair-280 formal audit is incomplete",
    )
    deliverables = summary.get("deliverable_status")
    _require(isinstance(deliverables, Mapping), "training summary deliverables missing")
    original_status = summary.get("status")
    original_formal = deliverables.get("formal_long_training")
    legacy_labels = (
        original_status == "SMOKE_COMPLETE" and original_formal == "NOT_STARTED"
    )
    corrected_labels = original_status == "COMPLETE" and original_formal == "PASS"
    _require(
        legacy_labels or corrected_labels,
        "training summary has an unrecognized formal-status combination",
    )
    corrected = copy.deepcopy(dict(summary))
    corrected["status"] = "COMPLETE"
    corrected_deliverables = dict(deliverables)
    corrected_deliverables["formal_long_training"] = "PASS"
    corrected["deliverable_status"] = corrected_deliverables
    corrected["formal_label_reconciliation"] = {
        "schema": "policy_pair280_formal_label_reconciliation_v1",
        "status": "PASS",
        "legacy_label_bug_present": legacy_labels,
        "original_status": original_status,
        "original_formal_long_training": original_formal,
        "corrected_status": "COMPLETE",
        "corrected_formal_long_training": "PASS",
        "authorization": "execution.long_formal_training=true",
        "scientific_payload_changed": False,
        "optimizer_or_checkpoint_payload_changed": False,
    }
    return corrected


def finalize_formal_completion(
    run_root: str | Path,
    *,
    summary_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
) -> dict[str, Any]:
    """Create immutable corrected labels and a final completion manifest."""

    root = Path(run_root).expanduser().resolve()
    _require(root.name == "formal", "Pair-280 completion root must be named formal")
    strict_audit = audit_run(root, smoke=False)
    requested_path = root / "requested_config.json"
    summary_path = root / "training_summary.json"
    requested = json.loads(requested_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(isinstance(requested, dict), "requested config root is not an object")
    _require(isinstance(summary, dict), "training summary root is not an object")
    corrected = _correct_formal_summary_labels(summary, requested, strict_audit)
    corrected["formal_label_reconciliation"]["original_training_summary"] = (
        artifact_identity(summary_path)
    )
    corrected["formal_label_reconciliation"]["fixed_sources"] = {
        "train": artifact_identity(Path(train_module.__file__).resolve()),
        "pair280_run": artifact_identity(Path(__file__).resolve()),
    }

    corrected_path = (
        Path(summary_output).expanduser().resolve()
        if summary_output is not None
        else root / "training_summary.formal_complete.json"
    )
    completion_path = (
        Path(manifest_output).expanduser().resolve()
        if manifest_output is not None
        else root.parent / "audits/formal_completion.json"
    )
    _write_new_json(corrected_path, corrected)

    checkpoints: list[dict[str, Any]] = []
    state_root = root / "checkpoints/state"
    expected_boundaries = list(range(2_000, 18_001, 2_000)) + [18_215]
    for step in expected_boundaries:
        state_dir = state_root / f"step_{step:08d}"
        trainer_state_path = state_dir / "trainer_state.json"
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        _require(
            trainer_state.get("status") == "PASS"
            and int(trainer_state.get("global_step", -1)) == step,
            f"checkpoint {step} trainer state is not finalized",
        )
        for filename in (
            "pytorch_model.bin",
            "optimizer.bin",
            "policy_overlay.pt",
            "custom_checkpoint_0.pkl",
        ):
            path = state_dir / filename
            _require(path.is_file() and path.stat().st_size > 0, f"checkpoint {step} lacks {filename}")
        rng_files = sorted(state_dir.glob("random_states_*.pkl"))
        _require(len(rng_files) == 8, f"checkpoint {step} does not contain 8-rank RNG state")
        checkpoints.append(
            {
                "step": step,
                "trainer_state": artifact_identity(trainer_state_path),
                "rng_rank_count": len(rng_files),
            }
        )
    latest = json.loads((state_root / "latest.json").read_text(encoding="utf-8"))
    _require(
        latest.get("status") == "PASS"
        and int(latest.get("global_step", -1)) == PAIR280_TOTAL_STEPS,
        "latest checkpoint does not point to the final Pair-280 step",
    )
    _require(
        not any(state_root.glob(".step_*")),
        "unfinished Pair-280 checkpoint staging directories remain",
    )
    cache_audit_path = ARTIFACT_ROOT / "cache_audit.json"
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    _require(
        cache_audit.get("status") == "PASS"
        and int(cache_audit.get("physical_state_groups", -1)) == PAIR280_GROUPS
        and int(cache_audit.get("shard_count", -1)) == 90,
        "Pair-280 cache audit is not complete",
    )
    smoke_audit_path = root.parent / "audits/smoke.json"
    smoke_audit = json.loads(smoke_audit_path.read_text(encoding="utf-8"))
    _require(
        smoke_audit.get("status") == "PASS" and smoke_audit.get("smoke") is True,
        "Pair-280 8-GPU smoke audit is not PASS",
    )
    original_formal_audit_path = root.parent / "audits/formal.json"
    original_formal_audit = json.loads(
        original_formal_audit_path.read_text(encoding="utf-8")
    )
    _require(
        original_formal_audit.get("status") == "PASS"
        and int(original_formal_audit.get("steps", -1)) == PAIR280_TOTAL_STEPS,
        "original Pair-280 formal audit is not PASS",
    )
    result = {
        "schema_version": 1,
        "kind": "policy_pair280_formal_completion",
        "status": "PASS",
        "run_root": str(root),
        "formal_steps": PAIR280_TOTAL_STEPS,
        "official_epochs": 5,
        "paired_epochs": 10,
        "paired_active_steps": PAIR280_ACTIVE_STEPS,
        "paired_inactive_steps": PAIR280_TOTAL_STEPS - PAIR280_ACTIVE_STEPS,
        "paired_groups": PAIR280_GROUPS,
        "paired_views": PAIR280_GROUPS * 4,
        "paired_exposures_per_state": 10,
        "requested_config": artifact_identity(requested_path),
        "original_training_summary": artifact_identity(summary_path),
        "corrected_training_summary": artifact_identity(corrected_path),
        "original_formal_audit": artifact_identity(original_formal_audit_path),
        "strict_formal_audit_recomputed": strict_audit,
        "final_checkpoint": strict_audit["checkpoint"],
        "resume_checkpoints": checkpoints,
        "cache_manifest": artifact_identity(CACHE_MANIFEST_PATH),
        "cache_audit": artifact_identity(cache_audit_path),
        "smoke_audit": artifact_identity(smoke_audit_path),
        "latest_checkpoint": latest,
        "label_reconciliation": corrected["formal_label_reconciliation"],
    }
    _write_new_json(completion_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", required=True, type=Path)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--resume")
    audit = sub.add_parser("audit-run")
    audit.add_argument("--run-root", required=True, type=Path)
    audit.add_argument("--smoke", action="store_true")
    audit.add_argument("--output", type=Path)
    finalize = sub.add_parser("finalize-formal")
    finalize.add_argument("--run-root", required=True, type=Path)
    finalize.add_argument("--summary-output", type=Path)
    finalize.add_argument("--manifest-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "materialize":
            result = materialize()
        elif args.command == "validate-config":
            result = validate_config(args.config)
        elif args.command == "train":
            destination = run_training(args.config, resume=args.resume)
            result = {"status": "PASS", "output_dir": str(destination)}
        elif args.command == "audit-run":
            result = audit_run(args.run_root, smoke=args.smoke)
            if args.output is not None:
                _write_new_json(args.output.expanduser().resolve(), result)
        else:
            result = finalize_formal_completion(
                args.run_root,
                summary_output=args.summary_output,
                manifest_output=args.manifest_output,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"Pair-280 run failed closed: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
