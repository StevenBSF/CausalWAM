"""Conditionally expand the passed P-v2 pilot to training seeds 2 and 3.

This module is intentionally separate from the source files bound by the
completed seed-1 pilot.  It derives seed-2/3 C1/C3 configs from the immutable
seed-1 pair, proves that only seed/initialization/output metadata changed, and
uses the unmodified trainer through a narrowly validated wrapper.

No artifact is materialized unless the audited 100-episode seed-53 pilot gate
passed.  The seed-59 confirmatory candidate bank is created only at that point.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import train as train_module
from .config_audit import load_config, validate_execution_ready
from .materialize_release_engineering_smoke import _write_new_yaml
from .p_mode_selection import (
    build_seed_bank_descriptor,
    canonical_sha256,
    validate_seed_bank_descriptor,
    validate_selection_manifest_payload,
)
from .pv2_actiondit_followup_audit import (
    _audit_action_gate,
    _audit_training_run,
    evaluate_pilot_gate,
)
from .pv2_followup_eval100_amendment import validate_eval100_amendment
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pv2_actiondit_followup_expansion_protocol"
SCHEMA_VERSION = 1
MATERIALIZATION_KIND = "policy_pv2_actiondit_followup_expansion_materialization"
POSTTRAIN_KIND = "policy_pv2_actiondit_followup_expansion_posttrain_audit"
TRAINING_SEEDS = (2, 3)
ALL_TRAINING_SEEDS = (1, 2, 3)
CONTROLS = {"c1": "c1_architecture_only", "c3": "c3_ours"}
CONFIRMATORY_SIMULATOR_SEED = 59
CONFIRMATORY_EPISODES_PER_CELL = 100
MAX_STEPS = 1800
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "pv2_actiondit_followup_v1"
).resolve()
DEFAULT_EXPANSION_MANIFEST = Path("manifests/seed2_seed3_expansion_protocol_v1.json")
DEFAULT_CONFIRMATORY_BANK = Path("manifests/confirmatory_seed59_bank_v1.json")
DEFAULT_MATERIALIZATION_AUDIT = Path("expansion_materialization_audit.json")
DEFAULT_POSTTRAIN_AUDIT = Path("expansion_posttrain_audit.json")


class Pv2FollowupExpansionError(ValueError):
    """Conditional expansion cannot be proven from immutable pilot evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2FollowupExpansionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required artifact missing: {resolved}")
    before = resolved.stat()
    digest = _sha256(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"artifact changed while hashing: {resolved}",
    )
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _verify_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} identity must be an object")
    actual = _identity(str(value.get("path", "")))
    for field in ("path", "size_bytes", "sha256"):
        expected = (
            str(Path(str(value.get(field, ""))).expanduser().resolve())
            if field == "path"
            else value.get(field)
        )
        _require(actual[field] == expected, f"{label} {field} changed")
    return actual


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pv2FollowupExpansionError(f"cannot parse {label}: {resolved}") from exc
    _require(isinstance(payload, dict), f"{label} root must be an object")
    return payload, resolved


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise Pv2FollowupExpansionError(
                f"refusing to overwrite immutable artifact: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def _dev_exclusion(bank: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seed_bank_descriptor(bank, expected_purpose="dev_selection")
    return {
        "purpose": "dev_selection",
        "simulator_seed_bank_id": normalized["simulator_seed_bank_id"],
        "member_count": normalized["member_count"],
        "members_sha256": normalized["members_sha256"],
        "members": normalized["members"],
    }


def _derive_expansion_config(
    source: Mapping[str, Any],
    *,
    experiment_root: Path,
    expansion_manifest: Path,
    expansion_manifest_sha256: str,
    training_seed: int,
    short: str,
) -> dict[str, Any]:
    _require(training_seed in TRAINING_SEEDS, "expansion seed must be 2 or 3")
    _require(short in CONTROLS, "expansion control must be c1/c3")
    value = copy.deepcopy(dict(source))
    value["experiment_id"] = f"pv2_actiondit_followup_seed{training_seed}_{short}_v1"
    value["output_dir"] = str(
        (experiment_root / "runs" / f"seed_{training_seed}" / short).resolve()
    )
    value["training"]["seed"] = training_seed
    value["policy"]["head_init_seed"] = training_seed
    value["policy"]["adapter_init_seed"] = training_seed
    value["execution"]["runner"] = "policy_content_adapter_pv2_expansion"
    value["mechanism_expansion_manifest"] = str(expansion_manifest.resolve())
    value["mechanism_expansion_manifest_sha256"] = expansion_manifest_sha256
    return value


def _load_and_verify_pilot(root: Path) -> dict[str, Any]:
    decision, decision_path = _load_json(root / "pilot_decision.json", "pilot decision")
    _require(
        decision.get("status") == "PASS"
        and decision.get("pilot_gate_passed") is True
        and decision.get("next_action") == "EXPAND_TO_SEEDS_2_3_AND_CONFIRMATORY_SEED59"
        and decision.get("episodes_per_task_domain") == 100,
        "pilot did not authorize seed2/3 expansion",
    )
    report, report_path = _load_json(
        root / "pilot_report/pilot_report_audit.json", "pilot report audit"
    )
    _require(
        report.get("status") == "PASS"
        and report.get("pilot_gate_passed") is True
        and report.get("seeds_2_3_authorized") is True
        and report.get("confirmatory_seed59_authorized") is True,
        "pilot report does not authorize expansion",
    )
    amendment, amendment_path = validate_eval100_amendment(
        root / "manifests/eval100_user_amendment_v1.json"
    )
    recomputed = evaluate_pilot_gate(
        root / "materialization_manifest.json",
        c1_rollout_manifest=(
            root
            / "pilot_rollouts_100ep_seed53_v1/c1/completed_rollouts.json"
        ),
        c3_rollout_manifest=(
            root
            / "pilot_rollouts_100ep_seed53_v1/c3/completed_rollouts.json"
        ),
        evaluation_amendment=amendment_path,
    )
    for field in (
        "pilot_gate_passed",
        "next_action",
        "macro",
        "delta",
        "conditions",
        "locked_thresholds",
    ):
        _require(recomputed[field] == decision[field], f"pilot decision changed at {field}")
    return {
        "decision": decision,
        "decision_identity": _identity(decision_path),
        "report_identity": _identity(report_path),
        "eval100_amendment": amendment,
        "eval100_amendment_identity": _identity(amendment_path),
        "seed1_posttrain_identity": _identity(root / "pilot_posttrain_audit.json"),
    }


def materialize_expansion(
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    source_root = Path(experiment_root).expanduser().resolve()
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else source_root
    )
    manifest_path = (root / DEFAULT_EXPANSION_MANIFEST).resolve()
    bank_path = (root / DEFAULT_CONFIRMATORY_BANK).resolve()
    audit_path = (root / DEFAULT_MATERIALIZATION_AUDIT).resolve()
    for path in (manifest_path, bank_path, audit_path):
        _require(not path.exists(), f"refusing to reuse expansion artifact: {path}")
    for seed in TRAINING_SEEDS:
        _require(
            not (root / "configs" / f"seed_{seed}").exists()
            and not (root / "runs" / f"seed_{seed}").exists(),
            f"seed {seed} expansion config/run already exists",
        )

    pilot = _load_and_verify_pilot(source_root)
    materialization, _ = _load_json(
        source_root / "materialization_manifest.json", "seed1 materialization"
    )
    protocol_identity = _verify_identity(
        materialization.get("protocol"), "mechanism protocol"
    )
    protocol, _ = _load_json(protocol_identity["path"], "mechanism protocol")
    _require(
        protocol.get("confirmatory_intent", {}).get("simulator_seed")
        == CONFIRMATORY_SIMULATOR_SEED
        and protocol.get("confirmatory_intent", {}).get(
            "episodes_per_task_domain"
        )
        == CONFIRMATORY_EPISODES_PER_CELL
        and protocol.get("confirmatory_intent", {}).get("training_seeds")
        == list(ALL_TRAINING_SEEDS),
        "mechanism protocol confirmatory intent differs",
    )

    selection_identity = _verify_identity(
        protocol.get("historical_p_mode_selection"), "historical selection"
    )
    selection, _ = _load_json(selection_identity["path"], "historical selection")
    validated_selection = validate_selection_manifest_payload(selection)
    seed23_bank = validated_selection["dev_seed_bank"]
    seed53_bank = pilot["eval100_amendment"]["runtime_evaluation"]
    seed53_raw, _ = _load_json(seed53_bank["seed_bank"]["path"], "seed53 bank")
    seed53_bank = validate_seed_bank_descriptor(
        seed53_raw, expected_purpose="dev_selection"
    )
    disjoint = [_dev_exclusion(seed23_bank), _dev_exclusion(seed53_bank)]
    evaluator = PROJECT_ROOT / "third_party/RoboTwin/script/eval_policy.py"
    seed59_bank = build_seed_bank_descriptor(
        simulator_seed=CONFIRMATORY_SIMULATOR_SEED,
        episodes_per_cell=CONFIRMATORY_EPISODES_PER_CELL,
        evaluator_source=evaluator,
        purpose="confirmatory_test",
        disjoint_from=disjoint,
    )
    seed59_bank = validate_seed_bank_descriptor(
        seed59_bank, expected_purpose="confirmatory_test"
    )
    seed59_members = set(seed59_bank["members"])
    _require(
        seed59_members.isdisjoint(range(4_300_000, 4_310_000)),
        "seed59 overlaps author-stock seed42 candidate range",
    )
    bank_bytes = (json.dumps(seed59_bank, indent=2, sort_keys=True) + "\n").encode()
    bank_identity = {
        "path": str(bank_path),
        "size_bytes": len(bank_bytes),
        "sha256": hashlib.sha256(bank_bytes).hexdigest(),
    }

    source_configs = {
        short: _identity(source_root / "configs/seed_1" / f"{short}.yaml")
        for short in CONTROLS
    }
    source_files = {
        "config_audit.py": Path(__file__).with_name("config_audit.py"),
        "p_mode_selection.py": Path(__file__).with_name("p_mode_selection.py"),
        "train.py": Path(__file__).with_name("train.py"),
        "pv2_actiondit_followup_expansion.py": Path(__file__).resolve(),
    }
    core = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "study_role": "conditional_confirmatory_expansion_after_passed_pilot",
        "experiment_root": str(root),
        "pilot_experiment_root": str(source_root),
        "pilot_decision": pilot["decision_identity"],
        "pilot_report_audit": pilot["report_identity"],
        "pilot_eval100_amendment": pilot["eval100_amendment_identity"],
        "seed1_posttrain_audit": pilot["seed1_posttrain_identity"],
        "mechanism_protocol": protocol_identity,
        "historical_selection": selection_identity,
        "authorization": {
            "pilot_gate_passed": True,
            "training_seeds_authorized": list(TRAINING_SEEDS),
            "confirmatory_simulator_seed_authorized": CONFIRMATORY_SIMULATOR_SEED,
            "result_driven_tuning_allowed": False,
        },
        "locked_training": {
            "policy_regime": "p_v2",
            "max_steps": MAX_STEPS,
            "official_batch_size": 1,
            "paired_groups_per_batch": 2,
            "world_size": 1,
            "gradient_accumulation_steps": 1,
            "head_adapter_lr": 1.0e-4,
            "action_dit_lr": 1.0e-5,
            "mixed_precision": "bf16",
            "controls": CONTROLS,
            "only_c1_c3_difference": "contrastive_coefficient_and_gradient",
        },
        "config_derivation": {
            "source_seed": 1,
            "source_configs": source_configs,
            "allowed_changes": [
                "experiment_id",
                "output_dir",
                "training.seed",
                "policy.head_init_seed",
                "policy.adapter_init_seed",
                "execution.runner",
                "mechanism_expansion_manifest",
                "mechanism_expansion_manifest_sha256",
            ],
            "checkpoint_declared_evaluation_bank_remains_seed53_20": True,
            "reason": "training_does_not_consume_online_bank; seed59_is_applied_by_external_confirmatory_amendment",
        },
        "confirmatory_seed_bank": bank_identity,
        "confirmatory_seed_bank_id": seed59_bank["simulator_seed_bank_id"],
        "confirmatory_contract": {
            "simulator_seed": CONFIRMATORY_SIMULATOR_SEED,
            "episodes_per_task_domain": CONFIRMATORY_EPISODES_PER_CELL,
            "tasks": ["place_a2b_left", "open_microwave", "move_stapler_pad"],
            "domains": ["clean", "official_random"],
            "training_seeds": list(ALL_TRAINING_SEEDS),
            "episode_pairing": "not_claimed_shared_starting_seed_only",
            "disjoint_from_simulator_seeds": [23, 42, 53],
        },
        "source_sha256": {
            name: _sha256(path) for name, path in source_files.items()
        },
    }
    manifest = {
        **core,
        "expansion_protocol_id": "pv2-expansion-v1:" + canonical_sha256(core),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    # Write the bank and protocol before configs; config derivation binds the
    # immutable protocol SHA but the protocol intentionally avoids a hash cycle.
    _write_new_json(bank_path, seed59_bank)
    _write_new_json(manifest_path, manifest)
    emitted: dict[str, dict[str, Any]] = {}
    for seed in TRAINING_SEEDS:
        emitted[str(seed)] = {}
        for short in CONTROLS:
            source = load_config(source_configs[short]["path"])
            config = _derive_expansion_config(
                source,
                experiment_root=root,
                expansion_manifest=manifest_path,
                expansion_manifest_sha256=manifest_sha,
                training_seed=seed,
                short=short,
            )
            path = root / "configs" / f"seed_{seed}" / f"{short}.yaml"
            _write_new_yaml(path, config)
            validate_expansion_config(path)
            emitted[str(seed)][short] = _identity(path)
    audit = {
        "kind": MATERIALIZATION_KIND,
        "schema_version": 1,
        "status": "PASS",
        "gpu_training_started": False,
        "online_confirmatory_rollout_started": False,
        "expansion_protocol": _identity(manifest_path),
        "confirmatory_seed_bank": _identity(bank_path),
        "configs": emitted,
        "training_seeds": list(TRAINING_SEEDS),
        "controls": CONTROLS,
    }
    _write_new_json(audit_path, audit)
    return audit


def validate_expansion_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest, resolved = _load_json(path, "expansion protocol")
    _require(
        manifest.get("kind") == KIND
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("status") == "PASS",
        "expansion protocol kind/version/status differs",
    )
    core = dict(manifest)
    protocol_id = core.pop("expansion_protocol_id", None)
    _require(
        protocol_id == "pv2-expansion-v1:" + canonical_sha256(core),
        "expansion protocol id differs",
    )
    decision_identity = _verify_identity(manifest.get("pilot_decision"), "pilot decision")
    decision, _ = _load_json(decision_identity["path"], "pilot decision")
    _require(
        decision.get("pilot_gate_passed") is True
        and decision.get("next_action") == "EXPAND_TO_SEEDS_2_3_AND_CONFIRMATORY_SEED59",
        "pilot no longer authorizes expansion",
    )
    report_identity = _verify_identity(
        manifest.get("pilot_report_audit"), "pilot report audit"
    )
    report, _ = _load_json(report_identity["path"], "pilot report audit")
    _require(
        report.get("seeds_2_3_authorized") is True
        and report.get("confirmatory_seed59_authorized") is True,
        "pilot report authorization differs",
    )
    _verify_identity(manifest.get("pilot_eval100_amendment"), "pilot eval100 amendment")
    _verify_identity(manifest.get("seed1_posttrain_audit"), "seed1 posttrain audit")
    _verify_identity(manifest.get("mechanism_protocol"), "mechanism protocol")
    _verify_identity(manifest.get("historical_selection"), "historical selection")
    bank_identity = _verify_identity(
        manifest.get("confirmatory_seed_bank"), "confirmatory seed59 bank"
    )
    bank, _ = _load_json(bank_identity["path"], "confirmatory seed59 bank")
    bank = validate_seed_bank_descriptor(bank, expected_purpose="confirmatory_test")
    _require(
        bank["simulator_seed"] == CONFIRMATORY_SIMULATOR_SEED
        and bank["episodes_per_cell"] == CONFIRMATORY_EPISODES_PER_CELL
        and bank["simulator_seed_bank_id"]
        == manifest.get("confirmatory_seed_bank_id"),
        "confirmatory seed59 bank contract differs",
    )
    source_sha = manifest.get("source_sha256")
    _require(isinstance(source_sha, Mapping), "expansion source SHA map missing")
    source_root = Path(__file__).resolve().parent
    expected_sources = {
        "config_audit.py",
        "p_mode_selection.py",
        "train.py",
        "pv2_actiondit_followup_expansion.py",
    }
    _require(set(source_sha) == expected_sources, "expansion source set differs")
    for name, digest in source_sha.items():
        _require(_sha256(source_root / name) == digest, f"expansion source drifted: {name}")
    return manifest, resolved


def validate_expansion_config(path_or_config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_config, Mapping):
        config = copy.deepcopy(dict(path_or_config))
    else:
        config = load_config(path_or_config)
    manifest_path = Path(str(config.get("mechanism_expansion_manifest", ""))).expanduser()
    _require(manifest_path.is_absolute(), "expansion config lacks absolute protocol path")
    manifest, resolved_manifest = validate_expansion_manifest(manifest_path)
    _require(
        config.get("mechanism_expansion_manifest_sha256") == _sha256(resolved_manifest),
        "expansion config protocol SHA differs",
    )
    seed = config.get("training", {}).get("seed")
    _require(seed in TRAINING_SEEDS, "expansion config seed must be 2 or 3")
    control = str(config.get("control", ""))
    short_matches = [short for short, label in CONTROLS.items() if label == control]
    _require(len(short_matches) == 1, "expansion config control differs")
    short = short_matches[0]
    source_identity = manifest["config_derivation"]["source_configs"][short]
    source_path = _verify_identity(source_identity, f"seed1 {short} source config")["path"]
    source = load_config(source_path)
    # The original seed-1 config remains the authoritative release/data/protocol
    # audit.  Validate it unchanged, then compare the actual expansion config to
    # the exact deterministic transform authorized by the expansion manifest.
    validate_execution_ready(source)
    expected = _derive_expansion_config(
        source,
        experiment_root=Path(manifest["experiment_root"]),
        expansion_manifest=resolved_manifest,
        expansion_manifest_sha256=_sha256(resolved_manifest),
        training_seed=int(seed),
        short=short,
    )
    _require(config == expected, "expansion config differs outside the authorized transform")
    return {
        "status": "PASS",
        "training_seed": int(seed),
        "control": control,
        "short": short,
        "expansion_protocol_id": manifest["expansion_protocol_id"],
        "source_seed1_config": source_identity,
        "config_projection_sha256": canonical_sha256(config),
    }


def run_expansion_training(config_path: str | Path) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    validate_expansion_config(config_file)
    original_validator = train_module.validate_execution_ready
    train_module.validate_execution_ready = validate_expansion_config
    try:
        return train_module.run(config_file)
    finally:
        train_module.validate_execution_ready = original_validator


def _rng_rows(root: Path) -> list[tuple[int, int, int]]:
    with (root / "train_log.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        (
            int(row["official_rng_seed"]),
            int(row["paired_rng_seed"]),
            int(row["official_data_seed"]),
        )
        for row in rows
    ]


def audit_expansion_posttrain(
    *, experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT
) -> dict[str, Any]:
    root = Path(experiment_root).expanduser().resolve()
    manifest, manifest_path = validate_expansion_manifest(
        root / DEFAULT_EXPANSION_MANIFEST
    )
    by_seed: dict[str, Any] = {}
    all_rng: dict[int, set[int]] = {}
    for seed in TRAINING_SEEDS:
        runs: dict[str, Any] = {}
        configs: dict[str, dict[str, Any]] = {}
        for short, control in CONTROLS.items():
            config_path = root / "configs" / f"seed_{seed}" / f"{short}.yaml"
            config = load_config(config_path)
            validate_expansion_config(config)
            configs[short] = config
            run_root = Path(config["output_dir"])
            run = _audit_training_run(
                run_root,
                control=control,
                expected_steps=MAX_STEPS,
                expected_lambda=0.0 if short == "c1" else 0.1,
            )
            run["action_gate"] = _audit_action_gate(
                run_root / "pre_online_action_gate.json", control
            )
            runs[short] = run
        for field in (
            "official_sample_sequence_sha256",
            "paired_physical_state_sequence_sha256",
            "matched_stream_contract_sha256",
        ):
            _require(
                runs["c1"]["summary"][field] == runs["c3"]["summary"][field],
                f"seed {seed} C1/C3 sequence differs at {field}",
            )
        _require(
            runs["c1"]["step_rng_rows_sha256"]
            == runs["c3"]["step_rng_rows_sha256"],
            f"seed {seed} C1/C3 step RNG rows differ",
        )
        for field in (
            "source_fp32_content_head_sha256",
            "source_fp32_adapter_sha256",
            "training_fp32_content_head_sha256",
            "training_fp32_adapter_sha256",
        ):
            _require(
                runs["c1"]["summary"]["initialization"][field]
                == runs["c3"]["summary"]["initialization"][field],
                f"seed {seed} C1/C3 initialization differs at {field}",
            )
        c1_rng = _rng_rows(Path(configs["c1"]["output_dir"]))
        c3_rng = _rng_rows(Path(configs["c3"]["output_dir"]))
        _require(c1_rng == c3_rng, f"seed {seed} full RNG rows differ")
        all_rng[seed] = {value for row in c1_rng for value in row[:2]}
        by_seed[str(seed)] = {
            "status": "PASS",
            "shared_sequences": {
                field: runs["c1"]["summary"][field]
                for field in (
                    "official_sample_sequence_sha256",
                    "paired_physical_state_sequence_sha256",
                    "matched_stream_contract_sha256",
                )
            },
            "shared_step_rng_rows_sha256": runs["c1"]["step_rng_rows_sha256"],
            "shared_head_initial_sha256": runs["c1"]["summary"]["initialization"][
                "source_fp32_content_head_sha256"
            ],
            "shared_gca_initial_sha256": runs["c1"]["summary"]["initialization"][
                "source_fp32_adapter_sha256"
            ],
            "runs": {
                short: {
                    "checkpoint": runs[short]["checkpoint"],
                    "action_dit_update": runs[short]["updates"]["action_dit"],
                    "head_gca_update": runs[short]["updates"]["head_and_adapter"],
                    "final_gate_raw": runs[short]["summary"]["final_gate_raw"],
                    "action_gate": runs[short]["action_gate"],
                }
                for short in CONTROLS
            },
        }
    _require(all_rng[2].isdisjoint(all_rng[3]), "seed2/3 step RNG streams overlap")
    return {
        "kind": POSTTRAIN_KIND,
        "schema_version": 1,
        "status": "PASS",
        "expansion_protocol": _identity(manifest_path),
        "expansion_protocol_id": manifest["expansion_protocol_id"],
        "steps_per_control": MAX_STEPS,
        "training_seeds": list(TRAINING_SEEDS),
        "controls": CONTROLS,
        "cross_seed_rng_disjoint": True,
        "by_seed": by_seed,
        "online_confirmatory_rollout_started": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    materialize.add_argument("--output-root")
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    audit = subparsers.add_parser("audit-posttrain")
    audit.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    audit.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize_expansion(
            experiment_root=args.experiment_root,
            output_root=args.output_root,
        )
    elif args.command == "validate-config":
        result = validate_expansion_config(args.config)
    elif args.command == "train":
        destination = run_expansion_training(args.config)
        result = {"status": "PASS", "output_dir": str(destination)}
    else:
        result = audit_expansion_posttrain(experiment_root=args.experiment_root)
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else Path(args.experiment_root).expanduser().resolve()
            / DEFAULT_POSTTRAIN_AUDIT
        )
        _write_new_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Pv2FollowupExpansionError",
    "audit_expansion_posttrain",
    "materialize_expansion",
    "run_expansion_training",
    "validate_expansion_config",
    "validate_expansion_manifest",
]
