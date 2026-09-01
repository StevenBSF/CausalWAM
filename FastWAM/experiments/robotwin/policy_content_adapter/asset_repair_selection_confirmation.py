"""Prove that the asset repair cannot change the locked P-mode winner.

The original P-v1/P-v2 dev rollouts were executed before the experiment-local
RoboTwin asset tree was repaired.  Only the ``demo_randomized`` evidence is
invalidated by the missing clutter assets.  The pre-registered selector first
applies a Clean guard and, for this run, that guard leaves P-v1 as the sole
eligible candidate.  Consequently no possible replacement Random score can
change the winner.

This module emits two immutable, CPU-only sidecars:

* a selection confirmation binding the original selection, the six Clean
  result files/logs, the invalid Random logs, and the asset-repair audit; and
* a formal-rollout continuation binding that confirmation to the immutable
  author-stock rollout plan and already-completed valid Clean cells.

Both artifacts are revalidated from their referenced bytes on every formal
rollout launch.  No checkpoint or historical result is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from .p_mode_selection import (
    CLEAN_MAX_DROP,
    COMPLETED_ROLLOUTS_SCHEMA,
    COMPLETED_ROLLOUTS_SCHEMA_VERSION,
    DEV_EPISODES_PER_CELL,
    DOMAINS,
    REGIMES,
    SELECTION_RULE,
    TASKS,
    canonical_sha256,
    validate_selection_manifest_payload,
)
from .release_formal_stock_rollout import (
    audit_stock_completed_cell,
    validate_stock_rollout_plan,
)
from .release_stock_eval_protocol import (
    PROFILE as STOCK_PROFILE,
    validate_stock_eval_amendment,
)


CONFIRMATION_KIND = "policy_pmode_asset_repair_selection_confirmation"
CONFIRMATION_SCHEMA_VERSION = 1
CONTINUATION_KIND = "policy_author_stock_asset_repair_continuation"
CONTINUATION_SCHEMA_VERSION = 1
ASSET_AUDIT_KIND = "robotwin_author_asset_repair_audit"
ASSET_AUDIT_SCHEMA_VERSION = 1

_ASSET_LOAD_ERROR = re.compile(
    r"assets/objects/objaverse/|objects/054_baguette/visual/base2\.glb",
    re.IGNORECASE,
)


class AssetRepairSelectionError(ValueError):
    """The asset-repair P-mode confirmation cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssetRepairSelectionError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required file is missing: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {resolved}",
    )
    return {
        "kind": "file",
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _verify_file_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} identity must be an object")
    path = Path(str(value.get("path", ""))).expanduser()
    _require(path.is_absolute(), f"{label} path must be absolute")
    actual = _stable_file_identity(path)
    for field in ("path", "size_bytes", "sha256"):
        expected = (
            str(Path(str(value.get(field, ""))).expanduser().resolve())
            if field == "path"
            else value.get(field)
        )
        _require(actual[field] == expected, f"{label} {field} changed")
    if "kind" in value:
        _require(value.get("kind") == "file", f"{label} kind changed")
    return actual


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _stable_file_identity(path)
    try:
        value = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssetRepairSelectionError(f"cannot parse {label}: {identity['path']}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, identity


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
            raise AssetRepairSelectionError(
                f"refusing to overwrite immutable sidecar: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def _required_sha256(value: Any, label: str) -> str:
    digest = str(value)
    _require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} must be a lowercase SHA-256",
    )
    return digest


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssetRepairSelectionError(f"cannot parse {label}: {path}") from exc
    _require(isinstance(value, Mapping), f"{label} root must be an object")
    return value


def _validate_domain_config_semantics(robotwin_root: Path) -> dict[str, Any]:
    clean_path = robotwin_root / "task_config/demo_clean.yml"
    random_path = robotwin_root / "task_config/demo_randomized.yml"
    base_task_path = robotwin_root / "envs/_base_task.py"
    clean = _load_yaml(clean_path, "demo_clean config")
    randomized = _load_yaml(random_path, "demo_randomized config")
    clean_domain = clean.get("domain_randomization")
    random_domain = randomized.get("domain_randomization")
    _require(isinstance(clean_domain, Mapping), "demo_clean lacks domain_randomization")
    _require(isinstance(random_domain, Mapping), "demo_randomized lacks domain_randomization")
    _require(
        clean_domain.get("random_background") is False
        and clean_domain.get("cluttered_table") is False
        and float(clean_domain.get("clean_background_rate")) == 1.0,
        "demo_clean unexpectedly enables repaired randomized assets",
    )
    _require(
        random_domain.get("random_background") is True
        and random_domain.get("cluttered_table") is True,
        "demo_randomized no longer enables background/clutter randomization",
    )
    base_task_source = base_task_path.read_text(encoding="utf-8")
    _require(
        "if self.cluttered_table:" in base_task_source
        and "self.get_cluttered_table()" in base_task_source
        and "if self.random_background:" in base_task_source,
        "RoboTwin randomized-asset guards changed",
    )
    return {
        "demo_clean": _stable_file_identity(clean_path),
        "demo_randomized": _stable_file_identity(random_path),
        "base_task_source": _stable_file_identity(base_task_path),
        "clean_semantics": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1.0,
        },
        "randomized_semantics": {
            "random_background": True,
            "cluttered_table": True,
        },
    }


def validate_asset_repair_audit_payload(
    value: Any,
    *,
    audit_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the repair scope and the repaired files needed by Random scenes."""

    _require(isinstance(value, Mapping), "asset repair audit must be an object")
    _require(value.get("kind") == ASSET_AUDIT_KIND, "asset repair audit kind changed")
    _require(
        value.get("schema_version") == ASSET_AUDIT_SCHEMA_VERSION,
        "asset repair audit schema version changed",
    )
    _require(value.get("status") == "PASS", "asset repair audit is not PASS")
    operation = value.get("operation")
    before = value.get("pre_repair")
    after = value.get("post_repair")
    recovery = value.get("rollout_recovery")
    _require(isinstance(operation, Mapping), "asset repair operation is missing")
    _require(isinstance(before, Mapping), "asset repair pre-repair evidence is missing")
    _require(isinstance(after, Mapping), "asset repair post-repair evidence is missing")
    _require(isinstance(recovery, Mapping), "asset repair rollout recovery is missing")
    _require(operation.get("delete_files") is False, "asset repair unexpectedly deleted files")
    _require(operation.get("copy_missing_only") is True, "asset repair was not missing-only")
    _require(int(operation.get("missing_files_copied", -1)) > 0, "asset repair copied no missing files")
    _require(
        int(before.get("observed_missing_objaverse_model_directories", 0)) > 0,
        "asset audit does not identify missing Objaverse clutter assets",
    )
    _require(after.get("missing_source_regular_files") == 0, "asset tree remains incomplete")
    _require(
        after.get("other_regular_file_checksum_differences") == 0,
        "asset tree has unexplained checksum differences",
    )
    _require(
        recovery.get("official_random_cells_started_before_repair_are_invalid") is True,
        "asset audit does not invalidate pre-repair Random evidence",
    )
    _require(
        recovery.get("checkpoint_weights_changed") is False,
        "asset repair unexpectedly changed checkpoint weights",
    )
    source_assets = Path(str(value.get("source_assets", ""))).expanduser().resolve()
    target_assets = Path(str(value.get("target_assets", ""))).expanduser().resolve()
    _require(source_assets.is_dir(), f"author asset source is missing: {source_assets}")
    _require(target_assets.is_dir(), f"repaired asset target is missing: {target_assets}")
    expected_count = int(after.get("target_regular_file_count", -1))
    actual_count = sum(1 for path in target_assets.rglob("*") if path.is_file())
    _require(actual_count == expected_count, "repaired asset file count changed")
    restored = after.get("restored_baguette_glb")
    _require(isinstance(restored, Mapping), "restored baguette identity is missing")
    relative = Path(str(restored.get("path", "")))
    target_glb = _stable_file_identity(target_assets / relative)
    source_glb = _stable_file_identity(source_assets / relative)
    _require(restored.get("matches_author_source") is True, "restored GLB is not source-matched")
    for field in ("size_bytes", "sha256"):
        _require(target_glb[field] == restored.get(field), f"restored GLB {field} changed")
        _require(target_glb[field] == source_glb[field], f"restored/source GLB {field} differs")
    domain_configs = _validate_domain_config_semantics(target_assets.parent)
    result = {
        "status": "PASS",
        "audit": dict(audit_identity) if audit_identity is not None else None,
        "source_assets": str(source_assets),
        "target_assets": str(target_assets),
        "target_regular_file_count": actual_count,
        "restored_baguette_glb": target_glb,
        "domain_config_evidence": domain_configs,
    }
    return result


def validate_asset_repair_audit(path: str | Path) -> dict[str, Any]:
    value, identity = _load_json(path, "asset repair audit")
    return validate_asset_repair_audit_payload(value, audit_identity=identity)


def _parse_result_rate(path: Path) -> float:
    parsed: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            parsed = float(line.strip())
        except ValueError:
            continue
    _require(parsed is not None and math.isfinite(parsed), f"result lacks a finite rate: {path}")
    _require(0.0 <= parsed <= 1.0, f"result rate is outside [0,1]: {path}")
    return parsed


def _same_embedded_identity(
    actual: Mapping[str, Any], embedded: Mapping[str, Any], label: str
) -> None:
    for field in ("path", "size_bytes", "sha256"):
        expected = (
            str(Path(str(embedded.get(field, ""))).expanduser().resolve())
            if field == "path"
            else embedded.get(field)
        )
        _require(actual[field] == expected, f"{label} {field} changed")


def _candidate_clean_evidence(
    selection: Mapping[str, Any], regime: str
) -> dict[str, Any]:
    candidate = selection["candidates"][regime]
    manifest_identity = candidate.get("result_manifest")
    _require(isinstance(manifest_identity, Mapping), f"{regime} result manifest is missing")
    manifest_path = Path(str(manifest_identity.get("path", ""))).expanduser().resolve()
    payload, actual_manifest_identity = _load_json(
        manifest_path, f"{regime} completed dev manifest"
    )
    _same_embedded_identity(
        actual_manifest_identity, manifest_identity, f"{regime} completed manifest"
    )
    _require(payload.get("schema") == COMPLETED_ROLLOUTS_SCHEMA, "dev manifest schema changed")
    _require(
        payload.get("schema_version") == COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        "dev manifest schema version changed",
    )
    contract = payload.get("checkpoint_contract")
    _require(isinstance(contract, Mapping), f"{regime} checkpoint contract is missing")
    _require(
        contract.get("control") == regime
        and contract.get("policy_regime") == regime
        and contract.get("stage") == "dev_pilot"
        and contract.get("selection_role") == "c1_lambda0"
        and float(contract.get("lambda_contrastive", -1.0)) == 0.0,
        f"{regime} is not the locked C1/lambda-zero dev candidate",
    )
    _require(
        payload.get("episodes_per_task") == DEV_EPISODES_PER_CELL,
        f"{regime} dev episode count changed",
    )
    runs = payload.get("runs")
    _require(isinstance(runs, list) and len(runs) == 6, f"{regime} dev matrix changed")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in runs:
        _require(isinstance(row, Mapping), f"{regime} run row is invalid")
        key = (str(row.get("task", "")), str(row.get("domain", "")))
        _require(key not in indexed, f"{regime} duplicate dev cell {key}")
        indexed[key] = row
    _require(
        set(indexed) == {(task, domain) for task in TASKS for domain in DOMAINS},
        f"{regime} dev matrix is incomplete",
    )
    selection_files = candidate.get("result_files")
    _require(isinstance(selection_files, list), f"{regime} selection result files are missing")
    selection_file_by_cell = {
        (str(row.get("task", "")), str(row.get("domain", ""))): row
        for row in selection_files
        if isinstance(row, Mapping)
    }
    _require(len(selection_file_by_cell) == 6, f"{regime} selection result files changed")
    clean_rows: list[dict[str, Any]] = []
    random_rows_ignored: list[dict[str, Any]] = []
    success_count = 0
    for task in TASKS:
        clean_run = indexed[(task, "clean")]
        _require(clean_run.get("task_config") == "demo_clean", f"{regime}/{task} is not Clean")
        _require(
            clean_run.get("episodes") == DEV_EPISODES_PER_CELL,
            f"{regime}/{task} Clean episode count changed",
        )
        result_identity = _stable_file_identity(str(clean_run.get("result", "")))
        selected_identity = selection_file_by_cell[(task, "clean")]
        _same_embedded_identity(
            result_identity, selected_identity, f"{regime}/{task} Clean result"
        )
        rate = _parse_result_rate(Path(result_identity["path"]))
        _require(
            rate == float(clean_run.get("success_rate"))
            == float(candidate["cells"][task]["clean"])
            == float(selected_identity.get("success_rate")),
            f"{regime}/{task} Clean rate bindings differ",
        )
        count = round(rate * DEV_EPISODES_PER_CELL)
        _require(
            math.isclose(rate, count / DEV_EPISODES_PER_CELL, abs_tol=1e-12),
            f"{regime}/{task} Clean rate is not an exact episode count",
        )
        log_identity = _stable_file_identity(str(clean_run.get("log", "")))
        clean_log = Path(log_identity["path"]).read_text(encoding="utf-8", errors="replace")
        load_errors = len(_ASSET_LOAD_ERROR.findall(clean_log))
        _require(load_errors == 0, f"{regime}/{task} Clean log contains repaired-asset errors")
        clean_rows.append(
            {
                "task": task,
                "episodes": DEV_EPISODES_PER_CELL,
                "successes": count,
                "success_rate": rate,
                "result": result_identity,
                "log": log_identity,
                "repaired_asset_error_occurrences": 0,
            }
        )
        success_count += count

        random_run = indexed[(task, "official_random")]
        _require(
            random_run.get("task_config") == "demo_randomized",
            f"{regime}/{task} Random task config changed",
        )
        random_log_identity = _stable_file_identity(str(random_run.get("log", "")))
        random_log = Path(random_log_identity["path"]).read_text(
            encoding="utf-8", errors="replace"
        )
        random_load_errors = len(_ASSET_LOAD_ERROR.findall(random_log))
        _require(
            random_load_errors > 0,
            f"{regime}/{task} Random log does not prove pre-repair asset contamination",
        )
        random_rows_ignored.append(
            {
                "task": task,
                "domain": "official_random",
                "log": random_log_identity,
                "repaired_asset_error_occurrences": random_load_errors,
                "scientific_status": "INVALID_PRE_REPAIR_ASSET_TREE",
                "success_rate_not_used": float(random_run.get("success_rate")),
            }
        )
    total_episodes = DEV_EPISODES_PER_CELL * len(TASKS)
    macro = success_count / total_episodes
    _require(
        math.isclose(macro, float(candidate["three_task_macro"]["clean"]), abs_tol=1e-12),
        f"{regime} Clean macro differs from selection",
    )
    return {
        "regime": regime,
        "completed_manifest": actual_manifest_identity,
        "clean_cells": clean_rows,
        "clean_successes": success_count,
        "clean_episodes": total_episodes,
        "clean_macro": macro,
        "official_random_cells_ignored": random_rows_ignored,
    }


def prove_clean_guard_singleton(
    clean_successes: Mapping[str, int],
    *,
    clean_episodes: int,
    max_drop: float,
) -> dict[str, Any]:
    """Return an exact-fraction proof that the Clean guard has one candidate."""

    _require(set(clean_successes) == set(REGIMES), "Clean proof requires P-v1 and P-v2")
    _require(clean_episodes > 0, "Clean proof requires positive episode count")
    fractions = {
        regime: Fraction(int(clean_successes[regime]), clean_episodes)
        for regime in REGIMES
    }
    best = max(fractions.values())
    drop = Fraction(str(max_drop))
    threshold = best - drop
    eligible = [regime for regime in REGIMES if fractions[regime] >= threshold]
    _require(
        eligible == ["p_v1"],
        "Clean guard is not a P-v1 singleton; Random revalidation is required",
    )
    return {
        "metric": "three_task_macro.clean",
        "max_drop": float(max_drop),
        "clean_successes": {regime: int(clean_successes[regime]) for regime in REGIMES},
        "clean_episodes": clean_episodes,
        "clean_macro_exact": {
            regime: f"{fractions[regime].numerator}/{fractions[regime].denominator}"
            for regime in REGIMES
        },
        "clean_macro": {regime: float(fractions[regime]) for regime in REGIMES},
        "best_clean_exact": f"{best.numerator}/{best.denominator}",
        "eligibility_threshold_exact": (
            f"{threshold.numerator}/{threshold.denominator}"
        ),
        "eligible_regimes": eligible,
        "winner": "p_v1",
        "random_score_quantifier": {
            "p_v1": "arbitrary_in_[0,1]",
            "p_v2": "arbitrary_in_[0,1]",
        },
        "proof": (
            "The selector maximizes official-Random only after the Clean guard. "
            "Because the eligible set is the singleton {p_v1}, every possible "
            "replacement pair of Random scores yields winner p_v1."
        ),
    }


def _confirmation_core(
    *, selection_path: str | Path, asset_audit_path: str | Path
) -> dict[str, Any]:
    selection_payload, selection_identity = _load_json(
        selection_path, "original P-mode selection"
    )
    selection = validate_selection_manifest_payload(selection_payload)
    _require(selection.get("winner") == "p_v1", "original P-mode winner is not P-v1")
    _require(selection.get("rule") == SELECTION_RULE, "P-mode selection rule changed")
    asset_report = validate_asset_repair_audit(asset_audit_path)
    evidence = {
        regime: _candidate_clean_evidence(selection, regime) for regime in REGIMES
    }
    episodes = {evidence[regime]["clean_episodes"] for regime in REGIMES}
    _require(len(episodes) == 1, "P-mode Clean episode totals differ")
    proof = prove_clean_guard_singleton(
        {regime: evidence[regime]["clean_successes"] for regime in REGIMES},
        clean_episodes=episodes.pop(),
        max_drop=CLEAN_MAX_DROP,
    )
    _require(
        selection.get("eligible_regimes") == ["p_v1"],
        "original selection did not record singleton Clean eligibility",
    )
    return {
        "selection_manifest": selection_identity,
        "asset_repair_audit": asset_report["audit"],
        "rule_sha256": canonical_sha256(SELECTION_RULE),
        "rule": SELECTION_RULE,
        "asset_scope_evidence": {
            "source_assets": asset_report["source_assets"],
            "target_assets": asset_report["target_assets"],
            "target_regular_file_count": asset_report["target_regular_file_count"],
            "restored_baguette_glb": asset_report["restored_baguette_glb"],
            "domain_config_evidence": asset_report["domain_config_evidence"],
        },
        "candidate_evidence": evidence,
        "clean_guard_invariance_proof": proof,
        "invalidated_evidence_policy": {
            "official_random_pre_repair": "INVALID_AND_NOT_USED",
            "invalidated_cell_count": len(REGIMES) * len(TASKS),
            "clean_pre_repair": "VALID_CONFIG_AND_LOG_EVIDENCE_BOUND",
            "random_revalidation_required_for_winner": False,
            "random_revalidation": "OPTIONAL_NON_BLOCKING_DIAGNOSTIC",
        },
        "confirmed_winner": "p_v1",
        "formal_resume_compatible": True,
    }


def write_selection_confirmation(
    *,
    selection_path: str | Path,
    asset_audit_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    core = _confirmation_core(
        selection_path=selection_path, asset_audit_path=asset_audit_path
    )
    payload: dict[str, Any] = {
        "kind": CONFIRMATION_KIND,
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **core,
    }
    payload["confirmation_id"] = "pmode-asset-confirmation-v1:" + canonical_sha256(core)
    _write_new_json(output, payload)
    validate_selection_confirmation(output)
    return payload


def validate_selection_confirmation(path: str | Path) -> dict[str, Any]:
    payload, identity = _load_json(path, "asset-repair selection confirmation")
    _require(payload.get("kind") == CONFIRMATION_KIND, "confirmation kind changed")
    _require(
        payload.get("schema_version") == CONFIRMATION_SCHEMA_VERSION,
        "confirmation schema version changed",
    )
    _require(payload.get("status") == "PASS", "selection confirmation is not PASS")
    selection_identity = _verify_file_identity(
        payload.get("selection_manifest"), "confirmation selection manifest"
    )
    asset_identity = _verify_file_identity(
        payload.get("asset_repair_audit"), "confirmation asset repair audit"
    )
    core = _confirmation_core(
        selection_path=selection_identity["path"],
        asset_audit_path=asset_identity["path"],
    )
    for field, expected in core.items():
        _require(payload.get(field) == expected, f"confirmation field changed: {field}")
    expected_id = "pmode-asset-confirmation-v1:" + canonical_sha256(core)
    _require(payload.get("confirmation_id") == expected_id, "confirmation id changed")
    return {
        "status": "PASS",
        "confirmation": identity,
        "confirmation_id": expected_id,
        "confirmed_winner": "p_v1",
        "formal_resume_compatible": True,
        "asset_repair_audit": asset_identity,
        "selection_manifest": selection_identity,
    }


def _continuation_core(
    *,
    confirmation_path: str | Path,
    plan_path: str | Path,
    amendment_path: str | Path,
    rollout_root: str | Path,
    preserved_completed_manifests: Sequence[str | Path],
) -> dict[str, Any]:
    confirmation = validate_selection_confirmation(confirmation_path)
    plan_report = validate_stock_rollout_plan(
        plan_path, require_output_absent=False
    )
    plan = plan_report["payload"]
    amendment, resolved_amendment = validate_stock_eval_amendment(amendment_path)
    amendment_identity = _stable_file_identity(resolved_amendment)
    resolved_rollout = Path(rollout_root).expanduser().resolve()
    _require(
        Path(str(plan.get("rollout_root", ""))).resolve() == resolved_rollout,
        "continuation rollout root differs from stock plan",
    )
    _require(amendment.get("profile") == STOCK_PROFILE, "continuation amendment profile changed")
    _same_embedded_identity(
        amendment_identity,
        plan.get("stock_protocol_amendment", {}),
        "stock plan amendment",
    )
    _require(len(preserved_completed_manifests) == 2, "continuation requires two preserved Clean cells")
    preserved: list[dict[str, Any]] = []
    seen_cells: set[int] = set()
    for path in preserved_completed_manifests:
        audit = audit_stock_completed_cell(plan_path, path)
        record = audit["record"]
        _require(record.get("domain") == "clean", "only completed Clean cells may be preserved")
        _require(record.get("policy_regime") == "p_v1", "preserved cell is not P-v1")
        _require(audit["cell_index"] not in seen_cells, "duplicate preserved completed cell")
        seen_cells.add(audit["cell_index"])
        preserved.append(
            {
                "cell_index": audit["cell_index"],
                "manifest": audit["manifest"],
                "control": record["control"],
                "training_seed": record["training_seed"],
                "task": record["task"],
                "domain": record["domain"],
                "episodes": record["episodes"],
                "success_rate": record["success_rate"],
                "episode_pairing": audit["episode_pairing"],
            }
        )
    preserved.sort(key=lambda row: int(row["cell_index"]))
    _require(
        {(row["training_seed"], row["control"], row["task"]) for row in preserved}
        == {
            (1, "c1_architecture_only", "place_a2b_left"),
            (1, "c1_architecture_only", "move_stapler_pad"),
        },
        "preserved formal Clean cells differ from recovery audit",
    )
    return {
        "evaluation_profile": STOCK_PROFILE,
        "confirmation": confirmation["confirmation"],
        "confirmation_id": confirmation["confirmation_id"],
        "confirmed_policy_regime": "p_v1",
        "asset_repair_audit": confirmation["asset_repair_audit"],
        "original_p_mode_selection": confirmation["selection_manifest"],
        "stock_rollout_plan": plan_report["plan"],
        "stock_protocol_amendment": amendment_identity,
        "rollout_root": str(resolved_rollout),
        "preserved_completed_cells": preserved,
        "recovery_policy": {
            "preserve_only_audited_completed_manifests": True,
            "preserved_completed_cell_count": 2,
            "pre_repair_official_random_attempts": "INVALID_RESTART_FROM_EPISODE_ZERO",
            "partial_attempts_without_completed_manifest": "NON_AUTHORITATIVE",
            "append_only_new_attempts": True,
            "formal_rollout_may_resume": True,
        },
    }


def write_formal_continuation(
    *,
    confirmation_path: str | Path,
    plan_path: str | Path,
    amendment_path: str | Path,
    rollout_root: str | Path,
    preserved_completed_manifests: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    core = _continuation_core(
        confirmation_path=confirmation_path,
        plan_path=plan_path,
        amendment_path=amendment_path,
        rollout_root=rollout_root,
        preserved_completed_manifests=preserved_completed_manifests,
    )
    payload: dict[str, Any] = {
        "kind": CONTINUATION_KIND,
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **core,
    }
    payload["continuation_id"] = "stock-asset-continuation-v1:" + canonical_sha256(core)
    _write_new_json(output, payload)
    validate_formal_continuation(
        output,
        expected_plan=plan_path,
        expected_amendment=amendment_path,
        expected_rollout_root=rollout_root,
    )
    return payload


def validate_formal_continuation(
    path: str | Path,
    *,
    expected_plan: str | Path | None = None,
    expected_amendment: str | Path | None = None,
    expected_rollout_root: str | Path | None = None,
) -> dict[str, Any]:
    payload, identity = _load_json(path, "formal asset-repair continuation")
    _require(payload.get("kind") == CONTINUATION_KIND, "continuation kind changed")
    _require(
        payload.get("schema_version") == CONTINUATION_SCHEMA_VERSION,
        "continuation schema version changed",
    )
    _require(payload.get("status") == "PASS", "formal continuation is not PASS")
    plan_identity = _verify_file_identity(payload.get("stock_rollout_plan"), "continuation plan")
    amendment_identity = _verify_file_identity(
        payload.get("stock_protocol_amendment"), "continuation amendment"
    )
    confirmation_identity = _verify_file_identity(
        payload.get("confirmation"), "continuation confirmation"
    )
    if expected_plan is not None:
        _require(
            plan_identity["path"] == str(Path(expected_plan).expanduser().resolve()),
            "runtime plan differs from continuation",
        )
    if expected_amendment is not None:
        _require(
            amendment_identity["path"]
            == str(Path(expected_amendment).expanduser().resolve()),
            "runtime amendment differs from continuation",
        )
    if expected_rollout_root is not None:
        _require(
            str(Path(str(payload.get("rollout_root", ""))).resolve())
            == str(Path(expected_rollout_root).expanduser().resolve()),
            "runtime rollout root differs from continuation",
        )
    preserved = payload.get("preserved_completed_cells")
    _require(isinstance(preserved, list), "continuation preserved cells are missing")
    paths = []
    for row in preserved:
        _require(isinstance(row, Mapping), "continuation preserved cell is invalid")
        manifest_identity = _verify_file_identity(
            row.get("manifest"), "continuation preserved manifest"
        )
        paths.append(manifest_identity["path"])
    core = _continuation_core(
        confirmation_path=confirmation_identity["path"],
        plan_path=plan_identity["path"],
        amendment_path=amendment_identity["path"],
        rollout_root=str(payload.get("rollout_root", "")),
        preserved_completed_manifests=paths,
    )
    for field, expected in core.items():
        _require(payload.get(field) == expected, f"continuation field changed: {field}")
    expected_id = "stock-asset-continuation-v1:" + canonical_sha256(core)
    _require(payload.get("continuation_id") == expected_id, "continuation id changed")
    return {
        "status": "PASS",
        "continuation": identity,
        "continuation_id": expected_id,
        "confirmation": confirmation_identity,
        "confirmed_policy_regime": "p_v1",
        "formal_rollout_may_resume": True,
        "preserved_completed_cell_count": 2,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    confirm = commands.add_parser("confirm", help="create immutable P-mode confirmation")
    confirm.add_argument("--selection", required=True)
    confirm.add_argument("--asset-repair-audit", required=True)
    confirm.add_argument("--output", required=True)
    validate_confirmation = commands.add_parser(
        "validate-confirmation", help="revalidate P-mode confirmation from raw bytes"
    )
    validate_confirmation.add_argument("--path", required=True)
    continuation = commands.add_parser(
        "bind-continuation", help="bind confirmation to the immutable stock rollout"
    )
    continuation.add_argument("--confirmation", required=True)
    continuation.add_argument("--plan", required=True)
    continuation.add_argument("--amendment", required=True)
    continuation.add_argument("--rollout-root", required=True)
    continuation.add_argument(
        "--preserved-completed-manifest", action="append", required=True
    )
    continuation.add_argument("--output", required=True)
    validate_continuation = commands.add_parser(
        "validate-continuation", help="revalidate the formal continuation binding"
    )
    validate_continuation.add_argument("--path", required=True)
    validate_continuation.add_argument("--plan", required=True)
    validate_continuation.add_argument("--amendment", required=True)
    validate_continuation.add_argument("--rollout-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "confirm":
        report = write_selection_confirmation(
            selection_path=args.selection,
            asset_audit_path=args.asset_repair_audit,
            output=args.output,
        )
        result = {
            "status": "PASS",
            "confirmed_winner": report["confirmed_winner"],
            "formal_resume_compatible": True,
            "confirmation": _stable_file_identity(args.output),
        }
    elif args.command == "validate-confirmation":
        result = validate_selection_confirmation(args.path)
    elif args.command == "bind-continuation":
        report = write_formal_continuation(
            confirmation_path=args.confirmation,
            plan_path=args.plan,
            amendment_path=args.amendment,
            rollout_root=args.rollout_root,
            preserved_completed_manifests=args.preserved_completed_manifest,
            output=args.output,
        )
        result = {
            "status": "PASS",
            "formal_rollout_may_resume": True,
            "continuation_id": report["continuation_id"],
            "continuation": _stable_file_identity(args.output),
        }
    else:
        result = validate_formal_continuation(
            args.path,
            expected_plan=args.plan,
            expected_amendment=args.amendment,
            expected_rollout_root=args.rollout_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AssetRepairSelectionError",
    "CONFIRMATION_KIND",
    "CONFIRMATION_SCHEMA_VERSION",
    "CONTINUATION_KIND",
    "CONTINUATION_SCHEMA_VERSION",
    "prove_clean_guard_singleton",
    "validate_asset_repair_audit",
    "validate_asset_repair_audit_payload",
    "validate_formal_continuation",
    "validate_selection_confirmation",
    "write_formal_continuation",
    "write_selection_confirmation",
]
