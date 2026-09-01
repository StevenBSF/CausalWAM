#!/usr/bin/env python3
"""Fail-closed P-v1/P-v2 selection from completed online-dev rollouts.

The selector accepts exactly one complete ``dev_pilot`` rollout manifest for
each policy regime.  Engineering smoke runs are intentionally ineligible.
The emitted manifest is immutable (exclusive-create) and is the artifact that
    formal C1/C3 configs must bind.  Both candidates must be C1/action-only
    dev pilots (``lambda_contrastive == 0``); C3 evidence is ineligible for
    choosing the shared policy regime.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import secrets
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
DOMAINS = ("clean", "official_random")
REGIMES = ("p_v1", "p_v2")
DEV_EPISODES_PER_CELL = 20
CLEAN_MAX_DROP = 0.05

COMPLETED_ROLLOUTS_SCHEMA = "policy_content_adapter.completed_rollouts"
COMPLETED_ROLLOUTS_SCHEMA_VERSION = 5
P_MODE_SELECTION_KIND = "policy_p_mode_selection"
P_MODE_SELECTION_SCHEMA_VERSION = 2

FORMAL_PROTOCOL_LOCK_KIND = "policy_release_formal_protocol_lock"
FORMAL_PROTOCOL_LOCK_SCHEMA_VERSION = 1

SEED_BANK_SCHEMA = "robotwin.sequential_expert_valid_seed_bank"
SEED_BANK_SCHEMA_VERSION = 3
SEED_BANK_ID_PREFIX = "robotwin-seed-bank-v3:"
SEED_BANK_MEMBER_COUNT = 10_000
SEED_BANK_PURPOSES = {
    "engineering_smoke",
    "dev_selection",
    "development_analysis",
    "confirmatory_test",
    "final_test",
}
SEED_BANK_IDENTITY_FIELDS = (
    "schema",
    "schema_version",
    "purpose",
    "simulator_seed",
    "candidate_start_seed",
    "episodes_per_cell",
    "selection",
    "evaluator_source_size_bytes",
    "evaluator_source_sha256",
    "member_count",
    "members_sha256",
    "members",
    "disjoint_from",
    "lock_ancestry",
)

SELECTION_RULE = {
    "schema": "policy_p_mode_selection_rule_v1",
    "primary_metric": "three_task_macro.official_random",
    "clean_guard_metric": "three_task_macro.clean",
    "clean_max_drop_from_best": CLEAN_MAX_DROP,
    "exact_primary_tie_break": "p_v1",
    "tasks": list(TASKS),
    "domains": list(DOMAINS),
    "episodes_per_task_domain": DEV_EPISODES_PER_CELL,
}

SHARED_IDENTITY_FIELDS = (
    "training_seed",
    "base_checkpoint_sha256",
    "dataset_stats_sha256",
    "base_lineage_manifest_sha256",
    "selection_role",
    "lambda_contrastive",
    "head_init_sha256",
    "gca_init_sha256",
    "stage2_recipe_sha256",
    "runtime_source_sha256",
    "simulator_seed_bank_manifest_sha256",
    "official_manifest_sha256",
    "paired_action_manifest_sha256",
    "paired_state_bank_sha256",
    "paired_text_cache_sha256",
    "paired_cache_sha256",
    "official_sample_sequence_sha256",
    "paired_physical_state_sequence_sha256",
    "matched_stream_contract_sha256",
    "rollout_protocol_id",
    "rollout_settings_sha256",
)


class PModeSelectionError(ValueError):
    """The dev evidence cannot prove a pre-locked P-mode selection."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PModeSelectionError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _as_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    _require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} must be a lowercase 64-character SHA-256",
    )
    return digest


def _as_nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _as_positive_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def _as_rate(value: Any, label: str) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PModeSelectionError(f"{label} must be numeric") from exc
    _require(math.isfinite(rate) and 0.0 <= rate <= 1.0, f"{label} must be within [0, 1]")
    return rate


def _members(value: Any, label: str) -> list[int]:
    _require(isinstance(value, list) and bool(value), f"{label} must be a non-empty list")
    parsed = [_as_nonnegative_int(item, f"{label}[{index}]") for index, item in enumerate(value)]
    _require(len(parsed) == len(set(parsed)), f"{label} contains duplicate seeds")
    _require(parsed == sorted(parsed), f"{label} must be strictly ordered")
    return parsed


def _validate_disjoint_summary(value: Any, index: int) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"disjoint_from[{index}] must be an object")
    purpose = str(value.get("purpose", ""))
    _require(purpose == "dev_selection", "final-test exclusions must be dev_selection banks")
    bank_id = str(value.get("simulator_seed_bank_id", ""))
    _require(bank_id.startswith(SEED_BANK_ID_PREFIX), "disjoint seed-bank id is invalid")
    members = _members(value.get("members"), f"disjoint_from[{index}].members")
    count = _as_positive_int(value.get("member_count"), f"disjoint_from[{index}].member_count")
    _require(count == len(members), "disjoint seed-bank member_count differs")
    members_sha = _as_sha256(
        value.get("members_sha256"), f"disjoint_from[{index}].members_sha256"
    )
    _require(canonical_sha256(members) == members_sha, "disjoint seed-bank members SHA differs")
    return {
        "purpose": purpose,
        "simulator_seed_bank_id": bank_id,
        "member_count": count,
        "members_sha256": members_sha,
        "members": members,
    }


def _embedded_file_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an artifact identity")
    path = str(value.get("path", "")).strip()
    _require(bool(path) and Path(path).expanduser().is_absolute(), f"{label}.path must be absolute")
    size = _as_positive_int(value.get("size_bytes"), f"{label}.size_bytes")
    digest = _as_sha256(value.get("sha256"), f"{label}.sha256")
    return {"path": str(Path(path).expanduser().resolve()), "size_bytes": size, "sha256": digest}


def _validate_seed_bank_lock_ancestry(value: Any, *, purpose: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "seed-bank lock_ancestry must be an object")
    if purpose != "final_test":
        _require(not value, f"{purpose} seed bank must not claim final protocol locks")
        return {}
    _require(
        set(value) == {"p_mode_selection_manifest", "formal_protocol_lock_manifest"},
        "final_test lock_ancestry must bind P-mode selection and formal protocol lock",
    )
    return {
        name: _embedded_file_identity(value[name], f"lock_ancestry.{name}")
        for name in ("p_mode_selection_manifest", "formal_protocol_lock_manifest")
    }


def seed_bank_identity_payload(seed_bank: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in SEED_BANK_IDENTITY_FIELDS if field not in seed_bank]
    _require(not missing, f"seed-bank descriptor lacks identity fields: {missing}")
    return {field: seed_bank[field] for field in SEED_BANK_IDENTITY_FIELDS}


def validate_seed_bank_descriptor(
    value: Any,
    *,
    expected_purpose: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one explicit, purpose-bound seed-bank descriptor."""

    _require(isinstance(value, Mapping), "seed-bank descriptor must be an object")
    _require(value.get("schema") == SEED_BANK_SCHEMA, "seed-bank schema changed")
    _require(
        value.get("schema_version") == SEED_BANK_SCHEMA_VERSION,
        "seed-bank schema_version changed",
    )
    purpose = str(value.get("purpose", ""))
    _require(purpose in SEED_BANK_PURPOSES, f"unsupported seed-bank purpose {purpose!r}")
    if expected_purpose is not None:
        _require(purpose == expected_purpose, f"seed-bank purpose must be {expected_purpose}")
    simulator_seed = _as_nonnegative_int(value.get("simulator_seed"), "simulator_seed")
    start = _as_nonnegative_int(value.get("candidate_start_seed"), "candidate_start_seed")
    episodes = _as_positive_int(value.get("episodes_per_cell"), "episodes_per_cell")
    member_count = _as_positive_int(value.get("member_count"), "member_count")
    _require(
        member_count == SEED_BANK_MEMBER_COUNT,
        f"seed-bank member_count must be {SEED_BANK_MEMBER_COUNT}",
    )
    members = _members(value.get("members"), "members")
    _require(len(members) == member_count, "seed-bank member_count differs from members")
    _require(
        members == list(range(start, start + member_count)),
        "seed-bank members must be the explicit contiguous candidate pool",
    )
    members_sha = _as_sha256(value.get("members_sha256"), "members_sha256")
    _require(canonical_sha256(members) == members_sha, "seed-bank members SHA differs")
    _as_positive_int(value.get("evaluator_source_size_bytes"), "evaluator_source_size_bytes")
    _as_sha256(value.get("evaluator_source_sha256"), "evaluator_source_sha256")
    _require(isinstance(value.get("selection"), str) and value["selection"], "selection is required")
    raw_disjoint = value.get("disjoint_from")
    _require(isinstance(raw_disjoint, list), "seed-bank disjoint_from must be a list")
    disjoint = [_validate_disjoint_summary(item, index) for index, item in enumerate(raw_disjoint)]
    disjoint_ids = [item["simulator_seed_bank_id"] for item in disjoint]
    _require(len(disjoint_ids) == len(set(disjoint_ids)), "duplicate disjoint seed-bank ids")
    own_members = set(members)
    for item in disjoint:
        _require(
            own_members.isdisjoint(item["members"]),
            "seed-bank member sets are not disjoint",
        )
    if purpose in {"final_test", "confirmatory_test"}:
        _require(
            bool(disjoint),
            f"{purpose} seed bank must exclude a dev_selection bank",
        )
    else:
        _require(not disjoint, f"{purpose} seed bank must not carry final-test exclusions")
    lock_ancestry = _validate_seed_bank_lock_ancestry(
        value.get("lock_ancestry"), purpose=purpose
    )

    identity_payload = seed_bank_identity_payload(value)
    expected_id = SEED_BANK_ID_PREFIX + canonical_sha256(identity_payload)
    _require(value.get("simulator_seed_bank_id") == expected_id, "seed-bank identity differs")
    _require(start == 100_000 * (1 + simulator_seed), "candidate_start_seed differs from simulator_seed")
    return {
        **identity_payload,
        "lock_ancestry": lock_ancestry,
        "simulator_seed_bank_id": expected_id,
    }


def _stable_file_identity(path: Path, *, raw_bytes: bytes | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"artifact does not exist: {resolved}")
    before = resolved.stat()
    data = resolved.read_bytes() if raw_bytes is None else raw_bytes
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"artifact changed while hashing: {resolved}",
    )
    _require(len(data) == after.st_size, f"artifact bytes differ from size: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_seed_bank_descriptor(
    *,
    simulator_seed: int,
    episodes_per_cell: int,
    evaluator_source: str | Path,
    purpose: str,
    disjoint_from: Sequence[Mapping[str, Any]] = (),
    lock_ancestry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact descriptor that must exist before training starts."""

    seed = _as_nonnegative_int(simulator_seed, "simulator_seed")
    episodes = _as_positive_int(episodes_per_cell, "episodes_per_cell")
    source = Path(evaluator_source).expanduser().resolve()
    source_identity = _stable_file_identity(source)
    start = 100_000 * (1 + seed)
    members = list(range(start, start + SEED_BANK_MEMBER_COUNT))
    payload = {
        "schema": SEED_BANK_SCHEMA,
        "schema_version": SEED_BANK_SCHEMA_VERSION,
        "purpose": str(purpose),
        "simulator_seed": seed,
        "candidate_start_seed": start,
        "episodes_per_cell": episodes,
        "selection": (
            "ascending_integer_candidates_filtered_by_setup_demo_play_once_"
            "plan_success_and_check_success"
        ),
        "evaluator_source_size_bytes": source_identity["size_bytes"],
        "evaluator_source_sha256": source_identity["sha256"],
        "member_count": len(members),
        "members_sha256": canonical_sha256(members),
        "members": members,
        "disjoint_from": [dict(item) for item in disjoint_from],
        "lock_ancestry": dict(lock_ancestry or {}),
    }
    descriptor = {
        **payload,
        "evaluator_source_path": str(source),
        "simulator_seed_bank_id": SEED_BANK_ID_PREFIX + canonical_sha256(payload),
        "identity_scope": "purpose_explicit_candidate_members_and_acceptance_algorithm",
    }
    return validate_seed_bank_descriptor(descriptor, expected_purpose=str(purpose))


def _exclusive_atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise PModeSelectionError(f"refusing to overwrite manifest: {destination}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_seed_bank_manifest(
    *,
    simulator_seed: int,
    episodes_per_cell: int,
    evaluator_source: str | Path,
    purpose: str,
    output: str | Path,
    disjoint_from_dev_manifest: str | Path | None = None,
    p_mode_selection_manifest: str | Path | None = None,
    formal_protocol_lock_manifest: str | Path | None = None,
) -> dict[str, Any]:
    disjoint: list[dict[str, Any]] = []
    lock_ancestry: dict[str, Any] = {}
    if purpose == "final_test":
        _require(
            disjoint_from_dev_manifest is not None,
            "final_test generation requires --disjoint-from-dev-manifest",
        )
        dev_payload, _ = _load_json_file(
            Path(disjoint_from_dev_manifest), label="dev-selection seed-bank manifest"
        )
        dev = validate_seed_bank_descriptor(dev_payload, expected_purpose="dev_selection")
        disjoint.append(
            {
                "purpose": "dev_selection",
                "simulator_seed_bank_id": dev["simulator_seed_bank_id"],
                "member_count": dev["member_count"],
                "members_sha256": dev["members_sha256"],
                "members": dev["members"],
            }
        )
        _require(
            p_mode_selection_manifest is not None
            and formal_protocol_lock_manifest is not None,
            "final_test generation requires P-mode selection and formal protocol lock manifests",
        )
        selection_payload, selection_raw = _load_json_file(
            Path(p_mode_selection_manifest), label="P-mode selection manifest"
        )
        validate_selection_manifest_payload(selection_payload)
        selection_identity = _stable_file_identity(
            Path(p_mode_selection_manifest), raw_bytes=selection_raw
        )
        lock_payload, lock_raw = _load_json_file(
            Path(formal_protocol_lock_manifest), label="formal protocol lock manifest"
        )
        validated_lock = validate_formal_protocol_lock_manifest_payload(lock_payload)
        lock_identity = _stable_file_identity(
            Path(formal_protocol_lock_manifest), raw_bytes=lock_raw
        )
        _require(
            validated_lock["p_mode_selection_manifest"]["sha256"]
            == selection_identity["sha256"],
            "formal protocol lock binds a different P-mode selection manifest",
        )
        lock_ancestry = {
            "p_mode_selection_manifest": selection_identity,
            "formal_protocol_lock_manifest": lock_identity,
        }
    else:
        _require(
            disjoint_from_dev_manifest is None,
            "only final_test generation accepts a disjoint dev manifest",
        )
        _require(
            p_mode_selection_manifest is None
            and formal_protocol_lock_manifest is None,
            "only final_test generation accepts formal protocol lock ancestry",
        )
    descriptor = build_seed_bank_descriptor(
        simulator_seed=simulator_seed,
        episodes_per_cell=episodes_per_cell,
        evaluator_source=evaluator_source,
        purpose=purpose,
        disjoint_from=disjoint,
        lock_ancestry=lock_ancestry,
    )
    _exclusive_atomic_write_json(output, descriptor)
    return descriptor


def _load_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise PModeSelectionError(f"cannot parse {label}: {resolved}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, raw


def _load_resolved_config_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    """Load a resolved JSON or YAML config while retaining its exact file bytes."""

    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except Exception:
        try:
            import yaml

            value = yaml.safe_load(raw.decode("utf-8"))
        except Exception as exc:
            raise PModeSelectionError(f"cannot parse resolved config {resolved}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, raw


def formal_config_protocol_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the hashable pre-final protocol, excluding unavoidable lock pointers.

    The executable config must bind the lock and final bank, while the lock must
    predate that bank.  Excluding only those pointers (and their operational
    unlock flags) avoids a hash cycle without excluding any method/training
    choice.
    """

    _require(isinstance(value, Mapping), "formal config must be an object")
    projected = copy.deepcopy(dict(value))
    projected.pop("formal_protocol_lock_manifest", None)
    artifacts = projected.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("formal_protocol_lock_manifest_sha256", None)
        artifacts.pop("simulator_seed_bank_manifest_sha256", None)
    evaluation = projected.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation.pop("simulator_seed_bank_manifest", None)
        evaluation.pop("simulator_seed_bank_id", None)
    execution = projected.get("execution")
    if isinstance(execution, dict):
        for field in ("runnable", "fail_closed", "blocked_reason"):
            execution.pop(field, None)
    return projected


def _validated_formal_config(
    path: Path,
    *,
    control: str,
    expected_seed: int,
    base_lineage_sha256: str,
    selection_sha256: str,
    winner: str,
) -> dict[str, Any]:
    payload, raw = _load_resolved_config_file(
        path, label=f"resolved {control} seed-{expected_seed} config"
    )
    projection = formal_config_protocol_projection(payload)
    encoded = json.dumps(projection, sort_keys=True)
    _require("__REQUIRED_" not in encoded and "__SELECT_" not in encoded, "resolved formal config contains placeholders")
    _require(payload.get("formal") is True, "formal config is not marked formal")
    _require(payload.get("stage") == "formal" and payload.get("control") == control, "formal config control/stage differs")
    training = payload.get("training")
    _require(isinstance(training, Mapping) and training.get("seed") == expected_seed, "formal config Stage-2 seed differs")
    artifacts = payload.get("artifacts")
    _require(isinstance(artifacts, Mapping), "formal config lacks artifacts")
    _require(
        artifacts.get("base_lineage_manifest_sha256") == base_lineage_sha256,
        "formal config base lineage SHA differs",
    )
    _require(
        artifacts.get("p_mode_selection_manifest_sha256") == selection_sha256,
        "formal config P-mode selection SHA differs",
    )
    policy = payload.get("policy")
    regime = str(policy.get("regime", "")).lower().replace("-", "_") if isinstance(policy, Mapping) else ""
    _require(regime == winner, "formal config policy regime differs from selected winner")
    loss = payload.get("loss")
    _require(isinstance(loss, Mapping), "formal config lacks loss")
    try:
        lambda_ctr = float(loss.get("lambda_contrastive"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PModeSelectionError("formal config lambda_contrastive must be numeric") from exc
    _require(math.isfinite(lambda_ctr) and lambda_ctr >= 0.0, "formal config lambda_contrastive is invalid")
    if control == "c1_architecture_only":
        _require(lambda_ctr == 0.0, "C1 formal config must set lambda_contrastive=0")
    else:
        _require(lambda_ctr > 0.0, "C3 formal config must enable contrastive supervision")
    return {
        "control": control,
        "training_seed": expected_seed,
        "lambda_contrastive": lambda_ctr,
        "source_config": _stable_file_identity(path, raw_bytes=raw),
        "protocol_projection_sha256": canonical_sha256(projection),
    }


def validate_formal_protocol_lock_manifest_payload(value: Any) -> dict[str, Any]:
    """Validate the immutable artifact that must predate the final-test bank."""

    _require(isinstance(value, Mapping), "formal protocol lock manifest must be an object")
    _require(value.get("kind") == FORMAL_PROTOCOL_LOCK_KIND, "formal protocol lock kind changed")
    _require(value.get("schema_version") == FORMAL_PROTOCOL_LOCK_SCHEMA_VERSION, "formal protocol lock schema changed")
    _require(value.get("status") == "PASS", "formal protocol lock status is not PASS")
    _require(value.get("stage2_training_seeds") == [1, 2, 3], "formal protocol lock seeds must be [1,2,3]")
    lineage = _embedded_file_identity(value.get("base_lineage_manifest"), "base_lineage_manifest")
    selection = _embedded_file_identity(value.get("p_mode_selection_manifest"), "p_mode_selection_manifest")
    matrix = _embedded_file_identity(value.get("formal_matrix_audit"), "formal_matrix_audit")
    winner = str(value.get("selected_policy_regime", ""))
    _require(winner in REGIMES, "formal protocol lock selected regime is invalid")
    configs = value.get("resolved_configs")
    _require(isinstance(configs, Mapping) and set(configs) == {"c1_architecture_only", "c3_ours"}, "formal protocol lock config matrix changed")
    normalized_configs: dict[str, list[dict[str, Any]]] = {}
    for control in ("c1_architecture_only", "c3_ours"):
        rows = configs[control]
        _require(isinstance(rows, list) and len(rows) == 3, f"formal protocol lock requires three {control} configs")
        normalized_rows: list[dict[str, Any]] = []
        for expected_seed, row in zip((1, 2, 3), rows, strict=True):
            _require(isinstance(row, Mapping), f"{control} config identity must be an object")
            _require(row.get("control") == control and row.get("training_seed") == expected_seed, f"{control} config seed/order differs")
            lambda_ctr = _as_rate(row.get("lambda_contrastive"), f"{control} lambda_contrastive")
            if control == "c1_architecture_only":
                _require(lambda_ctr == 0.0, "locked C1 lambda_contrastive must be zero")
            else:
                _require(lambda_ctr > 0.0, "locked C3 lambda_contrastive must be positive")
            source = _embedded_file_identity(
                row.get("source_config"), f"{control}[{expected_seed}].source_config"
            )
            projection_sha = _as_sha256(
                row.get("protocol_projection_sha256"),
                f"{control}[{expected_seed}].protocol_projection_sha256",
            )
            normalized_rows.append({
                "control": control,
                "training_seed": expected_seed,
                "lambda_contrastive": lambda_ctr,
                "source_config": source,
                "protocol_projection_sha256": projection_sha,
            })
        normalized_configs[control] = normalized_rows
    return {
        **dict(value),
        "base_lineage_manifest": lineage,
        "p_mode_selection_manifest": selection,
        "formal_matrix_audit": matrix,
        "resolved_configs": normalized_configs,
    }


def write_formal_protocol_lock_manifest(
    *,
    base_lineage_manifest: str | Path,
    p_mode_selection_manifest: str | Path,
    formal_matrix_audit: str | Path,
    c1_configs: Sequence[str | Path],
    c3_configs: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    """Lock resolved C1/C3 configs before the final-test seed bank exists."""

    _require(len(c1_configs) == 3 and len(c3_configs) == 3, "formal lock requires exactly three C1 and three C3 configs")
    lineage_payload, lineage_raw = _load_json_file(Path(base_lineage_manifest), label="author release lineage")
    _require(
        lineage_payload.get("kind") == "policy_author_release_base_lineage"
        and lineage_payload.get("schema_version") == 1
        and lineage_payload.get("status") == "PASS",
        "author release lineage is not a strict PASS",
    )
    lineage_identity = _stable_file_identity(Path(base_lineage_manifest), raw_bytes=lineage_raw)
    selection_payload, selection_raw = _load_json_file(Path(p_mode_selection_manifest), label="P-mode selection")
    selection = validate_selection_manifest_payload(selection_payload)
    selection_identity = _stable_file_identity(Path(p_mode_selection_manifest), raw_bytes=selection_raw)
    matrix_payload, matrix_raw = _load_json_file(Path(formal_matrix_audit), label="formal matrix audit")
    _require(matrix_payload.get("status") == "PASS", "formal matrix audit status is not PASS")
    matrix_identity = _stable_file_identity(Path(formal_matrix_audit), raw_bytes=matrix_raw)
    winner = str(selection["winner"])
    resolved_configs = {
        "c1_architecture_only": [
            _validated_formal_config(
                Path(path), control="c1_architecture_only", expected_seed=seed,
                base_lineage_sha256=lineage_identity["sha256"],
                selection_sha256=selection_identity["sha256"], winner=winner,
            )
            for seed, path in zip((1, 2, 3), c1_configs, strict=True)
        ],
        "c3_ours": [
            _validated_formal_config(
                Path(path), control="c3_ours", expected_seed=seed,
                base_lineage_sha256=lineage_identity["sha256"],
                selection_sha256=selection_identity["sha256"], winner=winner,
            )
            for seed, path in zip((1, 2, 3), c3_configs, strict=True)
        ],
    }
    manifest = {
        "kind": FORMAL_PROTOCOL_LOCK_KIND,
        "schema_version": FORMAL_PROTOCOL_LOCK_SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_lineage_manifest": lineage_identity,
        "p_mode_selection_manifest": selection_identity,
        "formal_matrix_audit": matrix_identity,
        "selected_policy_regime": winner,
        "stage2_training_seeds": [1, 2, 3],
        "resolved_configs": resolved_configs,
    }
    validate_formal_protocol_lock_manifest_payload(manifest)
    _exclusive_atomic_write_json(output, manifest)
    return manifest


def _checkpoint_identity(payload: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    declared = Path(str(payload.get("checkpoint", ""))).expanduser().resolve()
    embedded = contract.get("checkpoint_identity")
    _require(isinstance(embedded, Mapping), "dev manifest lacks checkpoint_identity")
    embedded_path = Path(str(embedded.get("path", ""))).expanduser().resolve()
    _require(declared == embedded_path, "dev manifest checkpoint paths differ")
    identity = _stable_file_identity(declared)
    _require(identity["size_bytes"] == embedded.get("size_bytes"), "checkpoint size differs")
    return identity


def _result_file_identities(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for run in runs:
        result_path = Path(str(run.get("result", ""))).expanduser().resolve()
        identity = _stable_file_identity(result_path)
        parsed_rate: float | None = None
        for line in result_path.read_text(encoding="utf-8").splitlines():
            try:
                parsed_rate = float(line.strip())
            except ValueError:
                continue
        _require(parsed_rate is not None, f"result file lacks success rate: {result_path}")
        manifest_rate = _as_rate(run.get("success_rate"), "run success_rate")
        _require(
            parsed_rate == manifest_rate,
            f"result file success rate differs from manifest for {run.get('task')}/{run.get('domain')}",
        )
        identities.append(
            {
                "task": str(run["task"]),
                "domain": str(run["domain"]),
                **identity,
                "success_rate": parsed_rate,
            }
        )
    return identities


def _candidate_from_completed_manifest(path: Path, *, expected_regime: str) -> dict[str, Any]:
    payload, raw = _load_json_file(path, label=f"{expected_regime} completed dev manifest")
    _require(payload.get("schema") == COMPLETED_ROLLOUTS_SCHEMA, "completed manifest schema changed")
    _require(
        payload.get("schema_version") == COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        "completed manifest schema_version changed",
    )
    contract = payload.get("checkpoint_contract")
    _require(isinstance(contract, Mapping), "completed manifest lacks checkpoint_contract")
    _require(contract.get("control") == expected_regime, "candidate control/regime differs")
    _require(contract.get("policy_regime") == expected_regime, "candidate policy regime differs")
    _require(contract.get("stage") == "dev_pilot", "mode selection accepts only dev_pilot checkpoints")
    _require(
        contract.get("selection_role") == "c1_lambda0",
        "P-mode selection accepts only C1 lambda=0 dev pilots",
    )
    try:
        lambda_contrastive = float(contract.get("lambda_contrastive"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PModeSelectionError("dev pilot lacks numeric lambda_contrastive") from exc
    _require(
        math.isfinite(lambda_contrastive) and lambda_contrastive == 0.0,
        "P-mode selection dev pilot must set lambda_contrastive=0",
    )
    _require(
        contract.get("p_mode_selection_manifest_sha256") is None,
        "dev_pilot checkpoint must predate P-mode selection",
    )
    _require(payload.get("checkpoint_fairness_identity") is None, "dev_pilot fairness identity must be null")
    protocol = payload.get("evaluation_protocol")
    _require(
        isinstance(protocol, Mapping)
        and protocol.get("eligible") is False
        and protocol.get("control") is None,
        "dev_pilot manifest is incorrectly marked as a formal control",
    )
    _require(payload.get("evaluation_records") == [], "dev_pilot must not emit formal records")

    declared_tasks = tuple(str(item) for item in contract.get("declared_tasks", ()))
    declared_domains = tuple(str(item) for item in contract.get("declared_domains", ()))
    _require(declared_tasks == TASKS, "dev_pilot must declare exactly the three tasks")
    _require(declared_domains == DOMAINS, "dev_pilot must declare Clean and official Random")
    _require(
        contract.get("declared_episodes_per_task") == DEV_EPISODES_PER_CELL
        and payload.get("episodes_per_task") == DEV_EPISODES_PER_CELL,
        f"dev_pilot requires {DEV_EPISODES_PER_CELL} episodes per task/domain",
    )

    seed_bank = validate_seed_bank_descriptor(
        payload.get("simulator_seed_bank"), expected_purpose="dev_selection"
    )
    _require(payload.get("simulator_seed_bank_id") == seed_bank["simulator_seed_bank_id"], "manifest seed-bank id differs")
    _require(payload.get("simulator_seed_bank_purpose") == "dev_selection", "manifest seed-bank purpose differs")
    _require(contract.get("simulator_seed_bank_id") == seed_bank["simulator_seed_bank_id"], "checkpoint seed-bank id differs")
    _require(contract.get("simulator_seed_bank_purpose") == "dev_selection", "checkpoint seed-bank purpose differs")
    seed_manifest_sha = _as_sha256(
        contract.get("simulator_seed_bank_manifest_sha256"),
        "simulator_seed_bank_manifest_sha256",
    )

    settings = payload.get("rollout_settings")
    _require(isinstance(settings, Mapping), "dev manifest lacks rollout_settings")
    settings_sha = _as_sha256(payload.get("rollout_settings_sha256"), "rollout_settings_sha256")
    _require(canonical_sha256(settings) == settings_sha, "rollout settings SHA differs")
    protocol_id = str(payload.get("rollout_protocol_id", ""))
    _require(bool(protocol_id) and contract.get("rollout_protocol_id") == protocol_id, "rollout protocol differs")

    runs = payload.get("runs")
    _require(isinstance(runs, list), "dev manifest runs must be a list")
    cells: dict[tuple[str, str], float] = {}
    normalized_runs: list[Mapping[str, Any]] = []
    for index, run in enumerate(runs):
        _require(isinstance(run, Mapping), f"run {index} must be an object")
        task = str(run.get("task", ""))
        domain = str(run.get("domain", ""))
        _require(task in TASKS and domain in DOMAINS, f"run {index} has unsupported cell")
        key = (task, domain)
        _require(key not in cells, f"duplicate dev cell {key}")
        _require(run.get("episodes") == DEV_EPISODES_PER_CELL, f"run {index} episode count differs")
        _require(run.get("rollout_protocol_id") == protocol_id, f"run {index} protocol differs")
        _require(
            run.get("simulator_seed_bank_id") == seed_bank["simulator_seed_bank_id"],
            f"run {index} seed bank differs",
        )
        _require(run.get("rollout_settings_sha256") == settings_sha, f"run {index} settings differ")
        rate = _as_rate(run.get("success_rate"), f"run {index} success_rate")
        count = round(rate * DEV_EPISODES_PER_CELL)
        _require(
            math.isclose(rate, count / DEV_EPISODES_PER_CELL, abs_tol=1e-12),
            f"run {index} success rate is not an exact episode count",
        )
        cells[key] = rate
        normalized_runs.append(run)
    expected_cells = {(task, domain) for task in TASKS for domain in DOMAINS}
    _require(set(cells) == expected_cells, "dev manifest does not contain the complete 3x2 matrix")

    identity: dict[str, Any] = {
        "training_seed": _as_nonnegative_int(contract.get("training_seed"), "training_seed"),
        "base_checkpoint_sha256": _as_sha256(contract.get("base_checkpoint_sha256"), "base checkpoint SHA"),
        "dataset_stats_sha256": _as_sha256(contract.get("dataset_stats_sha256"), "dataset stats SHA"),
        "base_lineage_manifest_sha256": _as_sha256(
            contract.get("base_lineage_manifest_sha256"), "author release lineage SHA"
        ),
        "selection_role": "c1_lambda0",
        "lambda_contrastive": 0.0,
        "head_init_sha256": _as_sha256(contract.get("head_init_sha256"), "Head init SHA"),
        "gca_init_sha256": _as_sha256(contract.get("gca_init_sha256"), "GCA init SHA"),
        "stage2_recipe_sha256": _as_sha256(contract.get("stage2_recipe_sha256"), "Stage-2 recipe SHA"),
        "runtime_source_sha256": _as_sha256(contract.get("runtime_source_sha256"), "runtime source SHA"),
        "official_sample_sequence_sha256": _as_sha256(
            contract.get("official_sample_sequence_sha256"), "official sample sequence SHA"
        ),
        "paired_physical_state_sequence_sha256": _as_sha256(
            contract.get("paired_physical_state_sequence_sha256"), "paired state sequence SHA"
        ),
        "matched_stream_contract_sha256": _as_sha256(
            contract.get("matched_stream_contract_sha256"), "matched stream contract SHA"
        ),
        "simulator_seed_bank_manifest_sha256": seed_manifest_sha,
        "rollout_protocol_id": protocol_id,
        "rollout_settings_sha256": settings_sha,
    }
    dev_artifacts = contract.get("dev_pilot_artifact_shas")
    _require(
        isinstance(dev_artifacts, Mapping),
        "dev_pilot checkpoint lacks audited paired/official artifact identities",
    )
    for field in (
        "official_manifest_sha256",
        "paired_action_manifest_sha256",
        "paired_state_bank_sha256",
        "paired_text_cache_sha256",
        "paired_cache_sha256",
    ):
        identity[field] = _as_sha256(dev_artifacts.get(field), field)
    recipe = contract.get("stage2_recipe")
    _require(isinstance(recipe, Mapping), "dev checkpoint contract lacks Stage-2 recipe")
    _require(canonical_sha256(recipe) == identity["stage2_recipe_sha256"], "Stage-2 recipe SHA differs")

    task_rows = {
        task: {domain: cells[(task, domain)] for domain in DOMAINS} for task in TASKS
    }
    macro = {
        domain: sum(cells[(task, domain)] for task in TASKS) / len(TASKS)
        for domain in DOMAINS
    }
    return {
        "regime": expected_regime,
        "checkpoint": _checkpoint_identity(payload, contract),
        "result_manifest": _stable_file_identity(path, raw_bytes=raw),
        "result_files": _result_file_identities(normalized_runs),
        "identity": identity,
        "dev_seed_bank_id": seed_bank["simulator_seed_bank_id"],
        "dev_seed_bank_sha256": canonical_sha256(seed_bank_identity_payload(seed_bank)),
        "episodes_per_cell": DEV_EPISODES_PER_CELL,
        "cells": task_rows,
        "three_task_macro": macro,
    }


def _winner_from_metrics(candidates: Mapping[str, Mapping[str, Any]]) -> tuple[str, float, list[str]]:
    best_clean = max(float(candidate["three_task_macro"]["clean"]) for candidate in candidates.values())
    eligible = [
        regime
        for regime in REGIMES
        if float(candidates[regime]["three_task_macro"]["clean"])
        >= best_clean - CLEAN_MAX_DROP
    ]
    _require(bool(eligible), "no P-mode candidate satisfies the Clean guard")
    best_random = max(
        float(candidates[regime]["three_task_macro"]["official_random"])
        for regime in eligible
    )
    tied = [
        regime
        for regime in eligible
        if float(candidates[regime]["three_task_macro"]["official_random"])
        == best_random
    ]
    winner = "p_v1" if "p_v1" in tied else tied[0]
    return winner, best_clean, eligible


def validate_selection_manifest_payload(value: Any) -> dict[str, Any]:
    """Validate a saved selection artifact without trusting its winner field."""

    _require(isinstance(value, Mapping), "P-mode selection manifest must be an object")
    _require(value.get("kind") == P_MODE_SELECTION_KIND, "P-mode selection kind changed")
    _require(
        value.get("schema_version") == P_MODE_SELECTION_SCHEMA_VERSION,
        "P-mode selection schema_version changed",
    )
    _require(value.get("status") == "PASS", "P-mode selection status is not PASS")
    _require(value.get("rule") == SELECTION_RULE, "P-mode selection rule changed")
    seed_bank = validate_seed_bank_descriptor(
        value.get("dev_seed_bank"), expected_purpose="dev_selection"
    )
    _require(
        value.get("dev_seed_bank_sha256")
        == canonical_sha256(seed_bank_identity_payload(seed_bank)),
        "selection dev seed-bank SHA differs",
    )
    raw_candidates = value.get("candidates")
    _require(isinstance(raw_candidates, Mapping), "selection candidates must be an object")
    _require(set(raw_candidates) == set(REGIMES), "selection requires exactly P-v1 and P-v2")
    candidates: dict[str, Mapping[str, Any]] = {}
    shared = value.get("shared_candidate_identity")
    _require(isinstance(shared, Mapping), "selection lacks shared_candidate_identity")
    _require(set(shared) == set(SHARED_IDENTITY_FIELDS), "shared candidate identity fields changed")
    _as_nonnegative_int(shared.get("training_seed"), "shared training_seed")
    _require(shared.get("selection_role") == "c1_lambda0", "selection was not based on C1 lambda=0")
    _require(float(shared.get("lambda_contrastive", -1.0)) == 0.0, "selection lambda_contrastive must be zero")
    for field in (
        "base_checkpoint_sha256",
        "dataset_stats_sha256",
        "base_lineage_manifest_sha256",
        "head_init_sha256",
        "gca_init_sha256",
        "stage2_recipe_sha256",
        "runtime_source_sha256",
        "simulator_seed_bank_manifest_sha256",
        "official_manifest_sha256",
        "paired_action_manifest_sha256",
        "paired_state_bank_sha256",
        "paired_text_cache_sha256",
        "paired_cache_sha256",
        "official_sample_sequence_sha256",
        "paired_physical_state_sequence_sha256",
        "matched_stream_contract_sha256",
        "rollout_settings_sha256",
    ):
        _as_sha256(shared.get(field), f"shared {field}")
    _require(bool(str(shared.get("rollout_protocol_id", ""))), "shared rollout_protocol_id is empty")
    for regime in REGIMES:
        candidate = raw_candidates[regime]
        _require(isinstance(candidate, Mapping), f"{regime} candidate must be an object")
        _require(candidate.get("regime") == regime, f"{regime} candidate label differs")
        _require(candidate.get("identity") == shared, f"{regime} shared identity differs")
        _require(candidate.get("dev_seed_bank_id") == seed_bank["simulator_seed_bank_id"], f"{regime} dev bank differs")
        _require(candidate.get("dev_seed_bank_sha256") == value["dev_seed_bank_sha256"], f"{regime} dev bank SHA differs")
        for artifact_name in ("checkpoint", "result_manifest"):
            artifact = candidate.get(artifact_name)
            _require(isinstance(artifact, Mapping), f"{regime} lacks {artifact_name}")
            _as_positive_int(artifact.get("size_bytes"), f"{regime} {artifact_name} size")
            _as_sha256(artifact.get("sha256"), f"{regime} {artifact_name} SHA")
            _require(bool(str(artifact.get("path", ""))), f"{regime} {artifact_name} path missing")
        cells = candidate.get("cells")
        _require(
            isinstance(cells, Mapping) and set(cells) == set(TASKS),
            f"{regime} cells tasks changed",
        )
        recalculated: dict[str, float] = {}
        for domain in DOMAINS:
            rates = []
            for task in TASKS:
                row = cells[task]
                _require(isinstance(row, Mapping) and set(row) == set(DOMAINS), f"{regime}/{task} domains changed")
                rates.append(_as_rate(row[domain], f"{regime}/{task}/{domain}"))
            recalculated[domain] = sum(rates) / len(rates)
        _require(candidate.get("three_task_macro") == recalculated, f"{regime} macro metrics differ")
        candidates[regime] = candidate
    winner, best_clean, eligible = _winner_from_metrics(candidates)
    _require(value.get("best_clean_macro") == best_clean, "selection best Clean macro differs")
    _require(value.get("eligible_regimes") == eligible, "selection Clean-guard eligibility differs")
    _require(value.get("winner") == winner, "selection winner differs from pre-locked rule")
    return dict(value)


def select_p_mode(
    *,
    p_v1_manifest: str | Path,
    p_v2_manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Select one mode and exclusive-create an immutable PASS manifest."""

    candidates = {
        "p_v1": _candidate_from_completed_manifest(Path(p_v1_manifest), expected_regime="p_v1"),
        "p_v2": _candidate_from_completed_manifest(Path(p_v2_manifest), expected_regime="p_v2"),
    }
    reference = candidates["p_v1"]["identity"]
    _require(candidates["p_v2"]["identity"] == reference, "P-v1/P-v2 candidate identity mismatch")
    _require(
        candidates["p_v1"]["dev_seed_bank_id"] == candidates["p_v2"]["dev_seed_bank_id"],
        "P-v1/P-v2 used different dev seed banks",
    )
    p_v1_payload, _ = _load_json_file(Path(p_v1_manifest), label="P-v1 dev manifest")
    dev_seed_bank = validate_seed_bank_descriptor(
        p_v1_payload.get("simulator_seed_bank"), expected_purpose="dev_selection"
    )
    winner, best_clean, eligible = _winner_from_metrics(candidates)
    source_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "kind": P_MODE_SELECTION_KIND,
        "schema_version": P_MODE_SELECTION_SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selector_source": _stable_file_identity(source_path),
        "rule": SELECTION_RULE,
        "shared_candidate_identity": reference,
        "dev_seed_bank": dev_seed_bank,
        "dev_seed_bank_sha256": canonical_sha256(seed_bank_identity_payload(dev_seed_bank)),
        "candidates": candidates,
        "best_clean_macro": best_clean,
        "eligible_regimes": eligible,
        "winner": winner,
        "winner_reason": (
            "highest official-Random three-task macro among candidates within "
            "0.05 of best Clean; exact Random tie prefers p_v1"
        ),
    }
    validate_selection_manifest_payload(manifest)
    try:
        _exclusive_atomic_write_json(output, manifest)
    except PModeSelectionError as exc:
        raise PModeSelectionError(
            f"refusing to overwrite selection manifest: {Path(output).expanduser().resolve()}"
        ) from exc
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selector = commands.add_parser("select", help="select P-v1/P-v2 from dev-pilot results")
    selector.add_argument("--p-v1-manifest", required=True)
    selector.add_argument("--p-v2-manifest", required=True)
    selector.add_argument("--output", required=True)
    seed_bank = commands.add_parser("seed-bank", help="atomically create a pre-training seed bank")
    seed_bank.add_argument("--purpose", required=True, choices=sorted(SEED_BANK_PURPOSES))
    seed_bank.add_argument("--simulator-seed", required=True, type=int)
    seed_bank.add_argument("--episodes-per-cell", required=True, type=int)
    seed_bank.add_argument("--evaluator-source", required=True)
    seed_bank.add_argument("--disjoint-from-dev-manifest")
    seed_bank.add_argument("--p-mode-selection-manifest")
    seed_bank.add_argument("--formal-protocol-lock-manifest")
    seed_bank.add_argument("--output", required=True)
    formal_lock = commands.add_parser(
        "formal-lock", help="lock resolved C1/C3 configs before creating final-test seeds"
    )
    formal_lock.add_argument("--base-lineage-manifest", required=True)
    formal_lock.add_argument("--p-mode-selection-manifest", required=True)
    formal_lock.add_argument("--formal-matrix-audit", required=True)
    formal_lock.add_argument("--c1-config", action="append", required=True)
    formal_lock.add_argument("--c3-config", action="append", required=True)
    formal_lock.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seed-bank":
        manifest = write_seed_bank_manifest(
            simulator_seed=args.simulator_seed,
            episodes_per_cell=args.episodes_per_cell,
            evaluator_source=args.evaluator_source,
            purpose=args.purpose,
            output=args.output,
            disjoint_from_dev_manifest=args.disjoint_from_dev_manifest,
            p_mode_selection_manifest=args.p_mode_selection_manifest,
            formal_protocol_lock_manifest=args.formal_protocol_lock_manifest,
        )
        result = {"status": "PASS", "simulator_seed_bank_id": manifest["simulator_seed_bank_id"]}
    elif args.command == "select":
        manifest = select_p_mode(
            p_v1_manifest=args.p_v1_manifest,
            p_v2_manifest=args.p_v2_manifest,
            output=args.output,
        )
        result = {"status": "PASS", "winner": manifest["winner"]}
    else:
        manifest = write_formal_protocol_lock_manifest(
            base_lineage_manifest=args.base_lineage_manifest,
            p_mode_selection_manifest=args.p_mode_selection_manifest,
            formal_matrix_audit=args.formal_matrix_audit,
            c1_configs=args.c1_config,
            c3_configs=args.c3_config,
            output=args.output,
        )
        result = {
            "status": "PASS",
            "selected_policy_regime": manifest["selected_policy_regime"],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLEAN_MAX_DROP",
    "DEV_EPISODES_PER_CELL",
    "FORMAL_PROTOCOL_LOCK_KIND",
    "FORMAL_PROTOCOL_LOCK_SCHEMA_VERSION",
    "P_MODE_SELECTION_KIND",
    "P_MODE_SELECTION_SCHEMA_VERSION",
    "PModeSelectionError",
    "SEED_BANK_IDENTITY_FIELDS",
    "SEED_BANK_ID_PREFIX",
    "SEED_BANK_MEMBER_COUNT",
    "SEED_BANK_SCHEMA",
    "SEED_BANK_SCHEMA_VERSION",
    "SELECTION_RULE",
    "canonical_sha256",
    "formal_config_protocol_projection",
    "build_seed_bank_descriptor",
    "seed_bank_identity_payload",
    "select_p_mode",
    "validate_seed_bank_descriptor",
    "validate_selection_manifest_payload",
    "validate_formal_protocol_lock_manifest_payload",
    "write_formal_protocol_lock_manifest",
    "write_seed_bank_manifest",
]
