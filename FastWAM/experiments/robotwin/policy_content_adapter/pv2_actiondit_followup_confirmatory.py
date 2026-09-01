"""Materialize and validate the seed-59 P-v2 confirmatory amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .p_mode_selection import validate_seed_bank_descriptor
from .pv2_actiondit_followup_expansion import (
    ALL_TRAINING_SEEDS,
    CONTROLS,
    DEFAULT_CONFIRMATORY_BANK,
    DEFAULT_EXPANSION_MANIFEST,
    DEFAULT_POSTTRAIN_AUDIT,
    audit_expansion_posttrain,
    validate_expansion_manifest,
)
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pv2_actiondit_followup_confirmatory_amendment"
SCHEMA_VERSION = 1
PROFILE = "pv2_actiondit_seed59_confirmatory_100ep_v1"
SIMULATOR_SEED = 59
EPISODES_PER_CELL = 100
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
DOMAINS = ("clean", "official_random")
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1"
).resolve()
DEFAULT_AMENDMENT = Path("manifests/confirmatory_seed59_amendment_v1.json")


class Pv2ConfirmatoryError(ValueError):
    """The seed-59 confirmatory evaluation contract is not proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2ConfirmatoryError(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required file missing: {resolved}")
    before = resolved.stat()
    digest = _sha256(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {resolved}",
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
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pv2ConfirmatoryError(f"cannot parse {label}: {resolved}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, resolved


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
            raise Pv2ConfirmatoryError(
                f"refusing to overwrite immutable confirmatory amendment: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def _checkpoint_rows(root: Path, expansion_posttrain: Mapping[str, Any]) -> list[dict[str, Any]]:
    seed1, _ = _load_json(root / "pilot_posttrain_audit.json", "seed1 posttrain audit")
    _require(
        seed1.get("status") == "PASS"
        and seed1.get("stage") == "pilot_posttrain"
        and seed1.get("steps_per_control") == 1800,
        "seed1 posttrain audit differs",
    )
    rows: list[dict[str, Any]] = []
    for seed in ALL_TRAINING_SEEDS:
        for short, control in CONTROLS.items():
            if seed == 1:
                raw = seed1.get("runs", {}).get(short, {}).get("checkpoint")
            else:
                raw = (
                    expansion_posttrain.get("by_seed", {})
                    .get(str(seed), {})
                    .get("runs", {})
                    .get(short, {})
                    .get("checkpoint")
                )
            _require(isinstance(raw, Mapping), f"seed{seed}/{short} checkpoint missing")
            actual = _identity(str(raw.get("path", "")))
            for field in ("path", "size_bytes", "sha256"):
                _require(
                    actual[field] == raw.get(field),
                    f"seed{seed}/{short} checkpoint {field} differs",
                )
            rows.append(
                {
                    "training_seed": seed,
                    "short": short,
                    "control": control,
                    "checkpoint_step": 1800,
                    **actual,
                }
            )
    return rows


def materialize_confirmatory_amendment(
    *, experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT
) -> tuple[dict[str, Any], Path]:
    root = Path(experiment_root).expanduser().resolve()
    destination = (root / DEFAULT_AMENDMENT).resolve()
    _require(not destination.exists(), f"confirmatory amendment already exists: {destination}")
    expansion, expansion_path = validate_expansion_manifest(
        root / DEFAULT_EXPANSION_MANIFEST
    )
    saved_posttrain, saved_path = _load_json(
        root / DEFAULT_POSTTRAIN_AUDIT, "expansion posttrain audit"
    )
    recomputed = audit_expansion_posttrain(experiment_root=root)
    _require(saved_posttrain == recomputed, "expansion posttrain audit changed")
    _require(
        saved_posttrain.get("status") == "PASS"
        and saved_posttrain.get("online_confirmatory_rollout_started") is False,
        "expansion training is incomplete or confirmatory rollout already started",
    )
    decision, decision_path = _load_json(root / "pilot_decision.json", "pilot decision")
    _require(
        decision.get("pilot_gate_passed") is True
        and decision.get("next_action") == "EXPAND_TO_SEEDS_2_3_AND_CONFIRMATORY_SEED59",
        "pilot did not authorize seed59",
    )
    bank_identity = _verify_identity(
        expansion.get("confirmatory_seed_bank"), "confirmatory seed59 bank"
    )
    bank, _ = _load_json(bank_identity["path"], "confirmatory seed59 bank")
    bank = validate_seed_bank_descriptor(bank, expected_purpose="confirmatory_test")
    _require(
        bank["simulator_seed"] == SIMULATOR_SEED
        and bank["episodes_per_cell"] == EPISODES_PER_CELL,
        "confirmatory bank is not seed59/100 episodes",
    )
    materialization, _ = _load_json(
        root / "materialization_manifest.json", "seed1 materialization"
    )
    original_bank_identity = _verify_identity(
        materialization.get("pilot_seed_bank"), "checkpoint-declared seed53 bank"
    )
    original_bank, _ = _load_json(original_bank_identity["path"], "original seed53 bank")
    original_bank = validate_seed_bank_descriptor(
        original_bank, expected_purpose="dev_selection"
    )
    _require(
        original_bank["simulator_seed"] == 53
        and original_bank["episodes_per_cell"] == 20,
        "checkpoint-declared seed53 bank differs",
    )
    mechanism_identity = _verify_identity(
        materialization.get("protocol"), "mechanism protocol"
    )
    checkpoints = _checkpoint_rows(root, saved_posttrain)
    source_root = Path(__file__).resolve().parent
    source_paths = {
        "eval_robotwin_pv2_confirmatory.py": source_root
        / "eval_robotwin_pv2_confirmatory.py",
        "eval_robotwin_single.py": source_root / "eval_robotwin_single.py",
        "pv2_actiondit_followup_confirmatory.py": Path(__file__).resolve(),
    }
    core = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "profile": PROFILE,
        "study_role": "conditional_confirmatory_test_after_passed_seed53_pilot",
        "experiment_root": str(root),
        "pilot_decision": _identity(decision_path),
        "expansion_protocol": _identity(expansion_path),
        "expansion_posttrain_audit": _identity(saved_path),
        "mechanism_protocol": mechanism_identity,
        "checkpoints": checkpoints,
        "original_evaluation": {
            "simulator_seed": 53,
            "episodes_per_task_domain": 20,
            "seed_bank": original_bank_identity,
            "seed_bank_id": original_bank["simulator_seed_bank_id"],
            "role": "checkpoint_ancestry_only_not_runtime_confirmatory_bank",
        },
        "runtime_evaluation": {
            "simulator_seed": SIMULATOR_SEED,
            "episodes_per_task_domain": EPISODES_PER_CELL,
            "episodes_per_checkpoint": 600,
            "episodes_all_checkpoints": 3600,
            "seed_bank": bank_identity,
            "seed_bank_id": bank["simulator_seed_bank_id"],
            "seed_bank_purpose": "confirmatory_test",
            "tasks": list(TASKS),
            "domains": list(DOMAINS),
            "training_seeds": list(ALL_TRAINING_SEEDS),
            "controls": CONTROLS,
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
        },
        "claim_boundary": {
            "confirmatory_bank_was_unopened_before_pilot_pass": True,
            "result_driven_tuning_performed": False,
            "exact_episode_pairing_claimed": False,
            "primary_pv1_replaced": False,
        },
        "runtime_source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
    }
    payload = {
        **core,
        "amendment_id": "pv2-confirmatory-v1:" + _canonical_sha256(core),
    }
    path = _write_new_json(destination, payload)
    validated, resolved = validate_confirmatory_amendment(path)
    return validated, resolved


def validate_confirmatory_amendment(
    path: str | Path,
) -> tuple[dict[str, Any], Path]:
    payload, resolved = _load_json(path, "confirmatory amendment")
    _require(
        payload.get("kind") == KIND
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "PASS"
        and payload.get("profile") == PROFILE,
        "confirmatory amendment kind/version/status/profile differs",
    )
    core = dict(payload)
    amendment_id = core.pop("amendment_id", None)
    _require(
        amendment_id == "pv2-confirmatory-v1:" + _canonical_sha256(core),
        "confirmatory amendment id differs",
    )
    decision_identity = _verify_identity(payload.get("pilot_decision"), "pilot decision")
    decision, _ = _load_json(decision_identity["path"], "pilot decision")
    _require(
        decision.get("pilot_gate_passed") is True,
        "pilot decision no longer authorizes confirmatory evaluation",
    )
    validate_expansion_manifest(
        _verify_identity(payload.get("expansion_protocol"), "expansion protocol")["path"]
    )
    posttrain_identity = _verify_identity(
        payload.get("expansion_posttrain_audit"), "expansion posttrain audit"
    )
    posttrain, _ = _load_json(posttrain_identity["path"], "expansion posttrain audit")
    _require(posttrain.get("status") == "PASS", "expansion posttrain audit is not PASS")
    mechanism_identity = _verify_identity(
        payload.get("mechanism_protocol"), "mechanism protocol"
    )
    rows = _checkpoint_rows(Path(payload["experiment_root"]), posttrain)
    _require(payload.get("checkpoints") == rows, "confirmatory checkpoint rows changed")
    original = payload.get("original_evaluation")
    runtime = payload.get("runtime_evaluation")
    _require(isinstance(original, Mapping), "original evaluation block missing")
    _require(isinstance(runtime, Mapping), "runtime evaluation block missing")
    original_identity = _verify_identity(original.get("seed_bank"), "original bank")
    old, _ = _load_json(original_identity["path"], "original bank")
    old = validate_seed_bank_descriptor(old, expected_purpose="dev_selection")
    _require(
        old["simulator_seed"] == 53
        and old["episodes_per_cell"] == 20
        and original.get("seed_bank_id") == old["simulator_seed_bank_id"],
        "original checkpoint bank differs",
    )
    runtime_identity = _verify_identity(runtime.get("seed_bank"), "seed59 bank")
    bank, _ = _load_json(runtime_identity["path"], "seed59 bank")
    bank = validate_seed_bank_descriptor(bank, expected_purpose="confirmatory_test")
    _require(
        bank["simulator_seed"] == SIMULATOR_SEED
        and bank["episodes_per_cell"] == EPISODES_PER_CELL
        and runtime.get("seed_bank_id") == bank["simulator_seed_bank_id"]
        and runtime.get("episodes_per_checkpoint") == 600
        and runtime.get("episodes_all_checkpoints") == 3600
        and runtime.get("tasks") == list(TASKS)
        and runtime.get("domains") == list(DOMAINS)
        and runtime.get("training_seeds") == list(ALL_TRAINING_SEEDS)
        and runtime.get("controls") == CONTROLS,
        "confirmatory runtime matrix differs",
    )
    _require(
        payload.get("mechanism_protocol", {}).get("sha256")
        == mechanism_identity["sha256"],
        "mechanism protocol identity differs",
    )
    source_sha = payload.get("runtime_source_sha256")
    _require(isinstance(source_sha, Mapping), "confirmatory runtime source map missing")
    source_root = Path(__file__).resolve().parent
    expected = {
        "eval_robotwin_pv2_confirmatory.py",
        "eval_robotwin_single.py",
        "pv2_actiondit_followup_confirmatory.py",
    }
    _require(set(source_sha) == expected, "confirmatory runtime source set differs")
    for name, digest in source_sha.items():
        _require(_sha256(source_root / name) == digest, f"confirmatory source drifted: {name}")
    return payload, resolved


def matching_checkpoint_row(
    amendment: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    control: str,
    training_seed: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    actual = _identity(checkpoint_path)
    matches = [
        row
        for row in amendment.get("checkpoints", [])
        if row.get("control") == control
        and row.get("training_seed") == training_seed
        and row.get("checkpoint_step") == checkpoint_step
    ]
    _require(len(matches) == 1, "confirmatory amendment lacks one checkpoint row")
    row = matches[0]
    for field in ("path", "size_bytes", "sha256"):
        _require(row.get(field) == actual[field], f"confirmatory checkpoint {field} differs")
    return dict(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--amendment", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        payload, path = materialize_confirmatory_amendment(
            experiment_root=args.experiment_root
        )
    else:
        payload, path = validate_confirmatory_amendment(args.amendment)
    print(
        json.dumps(
            {
                "status": "PASS",
                "amendment_id": payload["amendment_id"],
                "path": str(path),
                "sha256": _sha256(path),
                "episodes_per_cell": EPISODES_PER_CELL,
                "checkpoint_count": len(payload["checkpoints"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Pv2ConfirmatoryError",
    "matching_checkpoint_row",
    "materialize_confirmatory_amendment",
    "validate_confirmatory_amendment",
]
