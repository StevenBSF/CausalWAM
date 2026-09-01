"""Strict seed-59 aggregation and terminal deliverables for the P-v2 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pv2_actiondit_followup_confirmatory import (
    CONTROLS,
    DOMAINS,
    EPISODES_PER_CELL,
    PROFILE,
    SIMULATOR_SEED,
    TASKS,
    validate_confirmatory_amendment,
)
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pv2_actiondit_followup_final_summary"
SCHEMA_VERSION = 1
COMPLETED_SCHEMA = "policy_content_adapter.completed_rollouts"
COMPLETED_SCHEMA_VERSION = 8
TRAINING_SEEDS = (1, 2, 3)
T_CRITICAL_DF2_95 = 4.302652729911275
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1"
).resolve()
DEFAULT_ROLLOUT_ROOT = Path("confirmatory_rollouts_seed59_v1")
CPU_TEST_LOG = Path("confirmatory_cpu_tests.log")
CPU_TEST_AUDIT = Path("confirmatory_cpu_test_audit.json")


class Pv2FinalError(ValueError):
    """Confirmatory outputs do not prove the full P-v2 protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2FinalError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required file missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pv2FinalError(f"cannot parse {label}: {resolved}") from exc
    _require(isinstance(payload, dict), f"{label} root must be an object")
    return payload, resolved


def _write_new(path: Path, raw: bytes) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise Pv2FinalError(f"refusing to overwrite terminal artifact: {destination}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _write_new(
        path,
        (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def _mean_std_ci(values: Sequence[float]) -> dict[str, Any]:
    _require(len(values) == 3, "confirmatory summary requires exactly three seeds")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    radius = T_CRITICAL_DF2_95 * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "values_by_training_seed": {
            str(seed): value for seed, value in zip(TRAINING_SEEDS, values, strict=True)
        },
        "mean": mean,
        "std": std,
        "student_t_95ci_low": mean - radius,
        "student_t_95ci_high": mean + radius,
        "ci_method": "paired_training_seed_t_interval_df2",
    }


def record_cpu_test_audit(
    *, experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT
) -> dict[str, Any]:
    root = Path(experiment_root).expanduser().resolve()
    log_path = root / CPU_TEST_LOG
    audit_path = root / CPU_TEST_AUDIT
    _require(log_path.is_file(), f"CPU test log missing: {log_path}")
    _require(not audit_path.exists(), f"CPU test audit already exists: {audit_path}")
    text = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"(?:^|\n)(\d+) passed(?:,| in )", text)
    _require(len(matches) == 1, "CPU test log lacks one unambiguous passed count")
    _require(
        re.search(r"(?:^|\s)\d+ failed(?:,|\s)", text) is None
        and "ERRORS" not in text
        and "Interrupted:" not in text,
        "CPU test log contains failures/errors/interruption",
    )
    audit = {
        "kind": "policy_pv2_actiondit_followup_cpu_test_audit",
        "schema_version": 1,
        "status": "PASS",
        "passed": int(matches[0]),
        "failed": 0,
        "command": (
            "python -m pytest -q "
            "experiments/robotwin/policy_content_adapter/tests"
        ),
        "log": _identity(log_path),
        "scope": "full_policy_content_adapter_cpu_suite_before_seed59_rollout",
        "gpu_scientific_work_started_by_this_test_command": False,
    }
    _write_json(audit_path, audit)
    return audit


def _validate_cpu_test_audit(root: Path) -> dict[str, Any]:
    audit, path = _load_json(root / CPU_TEST_AUDIT, "CPU test audit")
    _require(
        audit.get("kind") == "policy_pv2_actiondit_followup_cpu_test_audit"
        and audit.get("schema_version") == 1
        and audit.get("status") == "PASS"
        and isinstance(audit.get("passed"), int)
        and audit.get("passed") > 0
        and audit.get("failed") == 0,
        "CPU test audit kind/version/status/count differs",
    )
    log_identity = audit.get("log")
    _require(isinstance(log_identity, Mapping), "CPU test audit log identity missing")
    actual = _identity(log_identity.get("path", ""))
    for field in ("path", "size_bytes", "sha256"):
        _require(actual[field] == log_identity.get(field), f"CPU test log {field} changed")
    return {**audit, "identity": _identity(path)}


def _exact_success_rate(value: Any) -> tuple[float, int]:
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Pv2FinalError("success_rate must be numeric") from exc
    _require(math.isfinite(rate) and 0.0 <= rate <= 1.0, "success_rate outside [0,1]")
    successes = round(rate * EPISODES_PER_CELL)
    _require(
        math.isclose(rate, successes / EPISODES_PER_CELL, abs_tol=1e-12),
        "success_rate is not an exact 100-episode count",
    )
    return rate, successes


def _manifest_paths(root: Path) -> list[Path]:
    return [
        root
        / DEFAULT_ROLLOUT_ROOT
        / f"seed_{seed}"
        / short
        / "completed_rollouts.json"
        for seed in TRAINING_SEEDS
        for short in CONTROLS
    ]


def aggregate_confirmatory(
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    amendment_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(experiment_root).expanduser().resolve()
    amendment, resolved_amendment = validate_confirmatory_amendment(
        amendment_path
        if amendment_path is not None
        else root / "manifests/confirmatory_seed59_amendment_v1.json"
    )
    amendment_identity = _identity(resolved_amendment)
    cpu_tests = _validate_cpu_test_audit(root)
    checkpoint_rows = {
        (int(row["training_seed"]), str(row["control"])): row
        for row in amendment["checkpoints"]
    }
    cells: dict[tuple[str, int, str, str], float] = {}
    successes: dict[tuple[str, int, str, str], int] = {}
    manifest_identities: dict[str, dict[str, Any]] = {}
    fairness: dict[tuple[int, str], dict[str, Any]] = {}
    settings_shas: set[str] = set()
    bank_ids: set[str] = set()
    protocol_ids: set[str] = set()
    for path in _manifest_paths(root):
        payload, resolved = _load_json(path, "confirmatory completed manifest")
        _require(
            payload.get("schema") == COMPLETED_SCHEMA
            and payload.get("schema_version") == COMPLETED_SCHEMA_VERSION,
            "confirmatory completed schema/version differs",
        )
        bound = payload.get("pv2_followup_eval_amendment")
        _require(isinstance(bound, Mapping), "completed manifest lacks amendment identity")
        for field in ("path", "size_bytes", "sha256"):
            _require(
                bound.get(field) == amendment_identity[field],
                f"completed amendment {field} differs",
            )
        _require(
            payload.get("pv2_followup_eval_amendment_id") == amendment["amendment_id"]
            and payload.get("evaluation_profile") == PROFILE
            and payload.get("episode_pairing") == "not_claimed"
            and payload.get("simulator_seed") == SIMULATOR_SEED
            and payload.get("episodes_per_task") == EPISODES_PER_CELL
            and payload.get("simulator_seed_bank_purpose") == "confirmatory_test"
            and payload.get("simulator_seed_bank_id")
            == amendment["runtime_evaluation"]["seed_bank_id"],
            "completed confirmatory profile/seed/episode contract differs",
        )
        contract = payload.get("checkpoint_contract")
        _require(isinstance(contract, Mapping), "completed checkpoint contract missing")
        seed = contract.get("training_seed")
        control = str(contract.get("control", ""))
        _require(seed in TRAINING_SEEDS and control in CONTROLS.values(), "checkpoint seed/control differs")
        _require(
            contract.get("stage") == "mechanism_followup"
            and contract.get("policy_regime") == "p_v2"
            and contract.get("checkpoint_step") == 1800
            and contract.get("mechanism_protocol_manifest_sha256")
            == amendment["mechanism_protocol"]["sha256"],
            "checkpoint mechanism contract differs",
        )
        expected_checkpoint = checkpoint_rows[(int(seed), control)]
        checkpoint_identity = _identity(payload.get("checkpoint"))
        for field in ("path", "size_bytes", "sha256"):
            _require(
                checkpoint_identity[field] == expected_checkpoint[field],
                f"seed{seed}/{control} checkpoint {field} differs",
            )
        key = (int(seed), control)
        fairness[key] = {
            field: contract.get(field)
            for field in (
                "head_init_sha256",
                "gca_init_sha256",
                "official_sample_sequence_sha256",
                "paired_physical_state_sequence_sha256",
                "matched_stream_contract_sha256",
                "stage2_recipe_sha256",
            )
        }
        runs = payload.get("runs")
        _require(isinstance(runs, list) and len(runs) == 6, "confirmatory manifest needs six cells")
        seen: set[tuple[str, str]] = set()
        for run in runs:
            task = str(run.get("task", ""))
            domain = str(run.get("domain", ""))
            _require(task in TASKS and domain in DOMAINS, "confirmatory cell task/domain differs")
            _require(run.get("episodes") == EPISODES_PER_CELL, "confirmatory cell is not 100 episodes")
            _require((task, domain) not in seen, "duplicate confirmatory cell")
            seen.add((task, domain))
            rate, count = _exact_success_rate(run.get("success_rate"))
            cells[(control, int(seed), task, domain)] = rate
            successes[(control, int(seed), task, domain)] = count
        _require(
            seen == {(task, domain) for task in TASKS for domain in DOMAINS},
            "confirmatory task/domain matrix incomplete",
        )
        manifest_identities[f"seed{seed}_{control}"] = _identity(resolved)
        settings_shas.add(str(payload.get("rollout_settings_sha256", "")))
        bank_ids.add(str(payload.get("simulator_seed_bank_id", "")))
        protocol_ids.add(str(payload.get("rollout_protocol_id", "")))
    _require(len(cells) == 36, "confirmatory matrix must contain exactly 36 cells")
    _require(len(settings_shas) == 1, "confirmatory rollout settings differ")
    _require(len(bank_ids) == 1, "confirmatory seed-bank identities differ")
    _require(len(protocol_ids) == 1, "confirmatory rollout protocol identities differ")
    for seed in TRAINING_SEEDS:
        _require(
            fairness[(seed, "c1_architecture_only")]
            == fairness[(seed, "c3_ours")],
            f"seed {seed} C1/C3 training fairness identity differs",
        )

    controls: dict[str, Any] = {}
    for short, control in CONTROLS.items():
        task_rows: dict[str, Any] = {}
        for task in TASKS:
            task_rows[task] = {
                domain: _mean_std_ci(
                    [cells[(control, seed, task, domain)] for seed in TRAINING_SEEDS]
                )
                for domain in DOMAINS
            }
        macro_by_seed = {
            domain: [
                statistics.fmean(cells[(control, seed, task, domain)] for task in TASKS)
                for seed in TRAINING_SEEDS
            ]
            for domain in DOMAINS
        }
        controls[short] = {
            "control": control,
            "tasks": task_rows,
            "macro": {
                domain: _mean_std_ci(values) for domain, values in macro_by_seed.items()
            },
        }

    deltas: dict[str, Any] = {"tasks": {}, "macro": {}}
    for task in TASKS:
        deltas["tasks"][task] = {}
        for domain in DOMAINS:
            values = [
                cells[("c3_ours", seed, task, domain)]
                - cells[("c1_architecture_only", seed, task, domain)]
                for seed in TRAINING_SEEDS
            ]
            deltas["tasks"][task][domain] = _mean_std_ci(values)
    for domain in DOMAINS:
        values = [
            statistics.fmean(
                cells[("c3_ours", seed, task, domain)] for task in TASKS
            )
            - statistics.fmean(
                cells[("c1_architecture_only", seed, task, domain)] for task in TASKS
            )
            for seed in TRAINING_SEEDS
        ]
        deltas["macro"][domain] = _mean_std_ci(values)

    pilot, pilot_path = _load_json(root / "pilot_decision.json", "pilot decision")
    p_v1_protocol, _ = _load_json(
        amendment["mechanism_protocol"]["path"], "mechanism protocol"
    )
    primary_identity = p_v1_protocol["primary_pv1_result"]["summary"]
    primary, primary_path = _load_json(primary_identity["path"], "P-v1 primary summary")
    _require(_identity(primary_path)["sha256"] == primary_identity["sha256"], "P-v1 primary summary changed")
    random_delta = deltas["macro"]["official_random"]["mean"]
    clean_delta = deltas["macro"]["clean"]["mean"]
    conclusion = (
        "Across three matched Stage-2 seeds under the unopened seed59 bank, "
        f"C3-C1 macro delta is {clean_delta * 100:+.2f} pp Clean and "
        f"{random_delta * 100:+.2f} pp Official Random."
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "study_classification": "post_hoc_mechanism_study",
        "primary_pv1_remains_authoritative": True,
        "confirmatory_amendment": amendment_identity,
        "cpu_test_audit": cpu_tests,
        "confirmatory_profile": PROFILE,
        "training_seeds": list(TRAINING_SEEDS),
        "simulator_seed": SIMULATOR_SEED,
        "episodes_per_task_domain": EPISODES_PER_CELL,
        "record_count": len(cells),
        "total_policy_episodes": len(cells) * EPISODES_PER_CELL,
        "completed_manifests": manifest_identities,
        "rollout_protocol_id": next(iter(protocol_ids)),
        "simulator_seed_bank_id": next(iter(bank_ids)),
        "rollout_settings_sha256": next(iter(settings_shas)),
        "controls": controls,
        "c3_minus_c1": deltas,
        "pilot": {
            "identity": _identity(pilot_path),
            "macro": pilot["macro"],
            "delta": pilot["delta"],
            "gate_passed": pilot["pilot_gate_passed"],
        },
        "p_v1_primary_reference": {
            "identity": _identity(primary_path),
            "status": primary.get("status"),
            "role": "secondary_mechanism_context_not_same_policy_regime",
        },
        "fairness_audit": {
            "status": "PASS",
            "within_seed_c1_c3_identity": {
                str(seed): fairness[(seed, "c1_architecture_only")]
                for seed in TRAINING_SEEDS
            },
            "only_treatment_difference": "contrastive_coefficient_and_gradient",
            "episode_pairing": "not_claimed_shared_starting_seed_only",
        },
        "conclusion": conclusion,
        "claim_boundary": {
            "may_replace_pv1_primary": False,
            "unseen_task_generalization_claimed": False,
            "background_only_causality_claimed": False,
            "r3_policy_success_reported": False,
            "result_driven_tuning_performed": False,
        },
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# P-v2 ActionDiT Follow-up — Confirmatory Result",
        "",
        "This is a post-hoc mechanism study. The immutable P-v1 experiment remains primary.",
        "",
        "## Three-seed macro result",
        "",
        "| Metric | C1 mean ± std | C3 mean ± std | C3−C1 mean ± std | 95% t-CI |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"clean": "Clean", "official_random": "Official Random"}
    for domain in DOMAINS:
        c1 = summary["controls"]["c1"]["macro"][domain]
        c3 = summary["controls"]["c3"]["macro"][domain]
        delta = summary["c3_minus_c1"]["macro"][domain]
        lines.append(
            f"| {labels[domain]} | {c1['mean']*100:.2f} ± {c1['std']*100:.2f}% "
            f"| {c3['mean']*100:.2f} ± {c3['std']*100:.2f}% "
            f"| {delta['mean']*100:+.2f} ± {delta['std']*100:.2f} pp "
            f"| [{delta['student_t_95ci_low']*100:+.2f}, "
            f"{delta['student_t_95ci_high']*100:+.2f}] pp |"
        )
    lines.extend(["", "## Task-level means", ""])
    for task in TASKS:
        lines.extend(
            [
                f"### {task}",
                "",
                "| Metric | C1 mean | C3 mean | C3−C1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for domain in DOMAINS:
            c1 = summary["controls"]["c1"]["tasks"][task][domain]["mean"]
            c3 = summary["controls"]["c3"]["tasks"][task][domain]["mean"]
            delta = summary["c3_minus_c1"]["tasks"][task][domain]["mean"]
            lines.append(
                f"| {labels[domain]} | {c1*100:.2f}% | {c3*100:.2f}% | {delta*100:+.2f} pp |"
            )
        lines.append("")
    lines.extend(["## Conclusion", "", str(summary["conclusion"]), ""])
    return "\n".join(lines)


def write_terminal_deliverables(
    *, experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT
) -> dict[str, Any]:
    root = Path(experiment_root).expanduser().resolve()
    for relative in ("summary.json", "summary.md", "completion_audit.json"):
        _require(not (root / relative).exists(), f"terminal artifact already exists: {relative}")
    summary = aggregate_confirmatory(experiment_root=root)
    summary_json = _write_json(root / "summary.json", summary)
    summary_md = _write_new(root / "summary.md", _markdown(summary).encode("utf-8"))
    completion = {
        "kind": "policy_pv2_actiondit_followup_completion_audit",
        "schema_version": 1,
        "status": "PASS",
        "goal_terminal_condition": "PILOT_PASSED_THREE_SEED_CONFIRMATORY_COMPLETE",
        "pilot_gate_passed": True,
        "seeds_2_3_training_complete": True,
        "confirmatory_seed59_opened_only_after_pilot_pass": True,
        "confirmatory_record_count": 36,
        "confirmatory_total_policy_episodes": 3600,
        "online_rollout_complete": True,
        "result_driven_tuning_performed": False,
        "primary_pv1_modified": False,
        "summary_json": _identity(summary_json),
        "summary_md": _identity(summary_md),
        "confirmatory_amendment": summary["confirmatory_amendment"],
        "cpu_test_audit": summary["cpu_test_audit"]["identity"],
        "completed_manifests": summary["completed_manifests"],
        "source": _identity(Path(__file__)),
    }
    completion_path = _write_json(root / "completion_audit.json", completion)
    return {**summary, "completion_audit": _identity(completion_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--record-cpu-tests", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(
        not (args.write and args.record_cpu_tests),
        "--write and --record-cpu-tests are mutually exclusive",
    )
    if args.record_cpu_tests:
        result = record_cpu_test_audit(experiment_root=args.experiment_root)
    elif args.write:
        result = write_terminal_deliverables(experiment_root=args.experiment_root)
    else:
        result = aggregate_confirmatory(experiment_root=args.experiment_root)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Pv2FinalError",
    "aggregate_confirmatory",
    "record_cpu_test_audit",
    "write_terminal_deliverables",
]
