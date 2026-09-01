"""Prepare and aggregate the strict C1/C3 final online rollout matrix.

The CPU-only ``prepare`` command proves that the six formal checkpoints are
the audited C1/C3 seed pairs, that they all bind the same immutable final-test
seed bank and formal protocol lock, and that the requested rollout destination
does not exist.  It emits a deterministic launch plan but never imports SAPIEN
or touches a GPU.

The shell runner consumes that plan as six candidate-level workers.  Each
worker owns one physical GPU and evaluates one checkpoint sequentially on the
three tasks and the two official domains.  Once all six completed-rollouts
manifests exist, ``aggregate`` reconstructs the 36 strict protocol records and
writes the C3-C1 paired summary.  Every output is create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .eval_robotwin_single import (
    FAIRNESS_RECORD_FIELDS,
    _checkpoint_evaluation_contract,
    _fairness_identity_from_checkpoint_contract,
    aggregate_completed_rollout_manifests,
)
from .evaluation_protocol import (
    DOMAINS,
    TASKS,
    audit_and_summarize,
)
from .formal_episode_protocol import (
    TASK_CONFIG_TO_DOMAIN as FORMAL_TASK_CONFIG_TO_DOMAIN,
    validate_realization_bank,
)
from .p_mode_selection import (
    validate_formal_protocol_lock_manifest_payload,
    validate_seed_bank_descriptor,
)
from .rollout_policy import _resolve_model_base_path
from .runtime_utils import PROJECT_ROOT


SCHEMA_VERSION = 1
FORMAL_SEEDS = (1, 2, 3)
CONTROL_ROWS = (
    ("c1", "c1_architecture_only", 0.0),
    ("c3", "c3_ours", 0.1),
)
EPISODES_PER_CELL = 100
TASK_CONFIGS = ("demo_clean", "demo_randomized")
DEFAULT_GPU_IDS = (0, 1, 2, 4, 5, 6)
DEFAULT_FORMAL_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "formal_c1_c3_release_v1_retry1"
).resolve()
DEFAULT_ROLLOUT_SUBDIR = "online_rollouts_final_test_v1"
DEFAULT_PLAN_NAME = "formal_rollout_plan_v1.json"
DEFAULT_REALIZATION_BANK_RELATIVE = (
    "manifests/final_test_exact_realization_v1/realization_bank.json"
)


class FormalRolloutError(ValueError):
    """The strict final online rollout contract cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalRolloutError(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required file is missing: {resolved}")
    before = resolved.stat()
    digest = _file_sha256(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while it was hashed: {resolved}",
    )
    return {
        "kind": "file",
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FormalRolloutError(f"cannot read {label}: {resolved}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, resolved


def _validate_bound_identity(
    declared: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    _require(isinstance(declared, Mapping), f"{label} identity must be an object")
    declared_path = Path(str(declared.get("path", ""))).expanduser()
    _require(declared_path.is_absolute(), f"{label} path must be absolute")
    declared_path = declared_path.resolve()
    if expected_path is not None:
        _require(
            declared_path == expected_path.resolve(),
            f"{label} path differs: {declared_path} != {expected_path.resolve()}",
        )
    actual = _stable_file_identity(declared_path)
    for field in ("kind", "size_bytes", "sha256"):
        _require(
            declared.get(field) == actual[field],
            f"{label} {field} differs from the immutable artifact",
        )
    return actual


def normalize_gpu_ids(values: str | Sequence[int | str]) -> tuple[int, ...]:
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",")]
    else:
        raw = [str(value).strip() for value in values]
    _require(len(raw) == 6, "formal rollout requires exactly six physical GPU ids")
    _require(all(part.isdigit() for part in raw), "GPU ids must be non-negative integers")
    result = tuple(int(part) for part in raw)
    _require(len(set(result)) == len(result), "formal rollout GPU ids must be unique")
    return result


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _assert_status_done(formal_root: Path) -> None:
    status = formal_root / "formal_c1_c3.status"
    _require(status.is_file(), f"formal training status is missing: {status}")
    value = status.read_text(encoding="utf-8").strip()
    _require(
        value.startswith("DONE formal_training=true online_rollout_started=false "),
        "formal training status does not prove completed training before rollout",
    )


def _load_formal_ancestry(formal_root: Path) -> dict[str, Any]:
    materialization, materialization_path = _load_json(
        formal_root / "materialization_manifest.json", "materialization manifest"
    )
    _require(materialization.get("status") == "PASS", "materialization is not PASS")
    _require(
        materialization.get("formal_lock_status") == "PASS",
        "materialization formal lock is not PASS",
    )
    _require(
        materialization.get("selected_policy_regime") == "p_v1",
        "formal matrix is not locked to selected P-v1",
    )
    _require(
        tuple(materialization.get("stage2_training_seeds", ())) == FORMAL_SEEDS,
        "formal Stage-2 seeds must be exactly 1,2,3",
    )
    _require(
        materialization.get("c0_evaluation_requested") is False,
        "this runner is the C1/C3-only primary protocol and rejects C0",
    )
    _require(
        materialization.get("online_rollout_started") is False,
        "materialization already claims that online rollout started",
    )
    artifacts = materialization.get("artifacts")
    _require(isinstance(artifacts, Mapping), "materialization lacks artifacts")

    final_bank_identity = _validate_bound_identity(
        artifacts.get("final_test_seed_bank"),
        label="final-test seed bank",
        expected_path=formal_root / "manifests/final_test_seed_bank.json",
    )
    formal_lock_identity = _validate_bound_identity(
        artifacts.get("formal_protocol_lock"),
        label="formal protocol lock",
        expected_path=formal_root / "manifests/formal_protocol_lock.json",
    )
    _require(
        artifacts.get("final_test_seed_bank_id"),
        "materialization lacks final-test seed-bank id",
    )

    raw_seed_bank, _ = _load_json(final_bank_identity["path"], "final-test seed bank")
    try:
        seed_bank = validate_seed_bank_descriptor(
            raw_seed_bank,
            expected_purpose="final_test",
        )
    except ValueError as exc:
        raise FormalRolloutError(f"invalid final-test seed bank: {exc}") from exc
    _require(
        seed_bank["episodes_per_cell"] == EPISODES_PER_CELL,
        f"final-test seed bank must contain {EPISODES_PER_CELL} episodes per cell",
    )
    _require(
        seed_bank["simulator_seed_bank_id"]
        == artifacts["final_test_seed_bank_id"],
        "final-test seed-bank id differs from materialization",
    )

    raw_lock, _ = _load_json(formal_lock_identity["path"], "formal protocol lock")
    try:
        formal_lock = validate_formal_protocol_lock_manifest_payload(raw_lock)
    except ValueError as exc:
        raise FormalRolloutError(f"invalid formal protocol lock: {exc}") from exc
    _require(formal_lock.get("status") == "PASS", "formal protocol lock is not PASS")
    _require(
        formal_lock.get("selected_policy_regime") == "p_v1",
        "formal protocol lock does not select P-v1",
    )
    _require(
        tuple(formal_lock.get("stage2_training_seeds", ())) == FORMAL_SEEDS,
        "formal protocol lock seeds differ",
    )
    lock_ancestry = seed_bank.get("lock_ancestry")
    _require(isinstance(lock_ancestry, Mapping), "final-test bank lacks lock ancestry")
    locked_formal = lock_ancestry.get("formal_protocol_lock_manifest")
    locked_selection = lock_ancestry.get("p_mode_selection_manifest")
    _require(
        isinstance(locked_formal, Mapping)
        and locked_formal.get("sha256") == formal_lock_identity["sha256"],
        "final-test bank binds a different formal protocol lock",
    )
    _require(
        isinstance(locked_selection, Mapping)
        and locked_selection.get("sha256")
        == formal_lock["p_mode_selection_manifest"]["sha256"],
        "final-test bank binds a different P-mode selection",
    )
    return {
        "materialization": materialization,
        "materialization_identity": _stable_file_identity(materialization_path),
        "final_test_seed_bank": seed_bank,
        "final_test_seed_bank_identity": final_bank_identity,
        "formal_protocol_lock": formal_lock,
        "formal_protocol_lock_identity": formal_lock_identity,
    }


def build_rollout_plan(
    *,
    formal_root: str | Path,
    rollout_root: str | Path,
    realization_bank: str | Path | None = None,
    gpu_ids: str | Sequence[int | str] = DEFAULT_GPU_IDS,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    formal = Path(formal_root).expanduser().resolve()
    output = Path(rollout_root).expanduser().resolve()
    _require(formal.is_dir(), f"formal output root is missing: {formal}")
    if require_output_absent:
        _require(
            not output.exists(),
            f"refusing to reuse formal rollout output root: {output}",
        )
    _assert_status_done(formal)
    physical_gpus = normalize_gpu_ids(gpu_ids)
    ancestry = _load_formal_ancestry(formal)
    realization_path = (
        Path(realization_bank).expanduser().resolve()
        if realization_bank is not None
        else (formal / DEFAULT_REALIZATION_BANK_RELATIVE).resolve()
    )
    try:
        exact_bank, exact_bank_path = validate_realization_bank(realization_path)
    except Exception as exc:
        raise FormalRolloutError(
            "formal rollout is blocked until the policy-independent exact "
            f"realization bank passes audit: {realization_path}: {exc}"
        ) from exc
    exact_bank_identity = _stable_file_identity(exact_bank_path)
    _require(
        exact_bank["candidate_seed_bank_id"]
        == ancestry["final_test_seed_bank"]["simulator_seed_bank_id"],
        "exact realization derives from a different candidate seed bank",
    )
    _require(
        exact_bank["candidate_seed_bank"]["sha256"]
        == ancestry["final_test_seed_bank_identity"]["sha256"],
        "exact realization candidate-bank SHA differs",
    )
    _require(
        exact_bank["formal_protocol_lock"]["sha256"]
        == ancestry["formal_protocol_lock_identity"]["sha256"],
        "exact realization formal-lock SHA differs",
    )

    posttrain, posttrain_path = _load_json(
        formal / "strict_posttrain_pair_audit.json", "strict posttrain pair audit"
    )
    _require(posttrain.get("status") == "PASS", "strict posttrain audit is not PASS")
    _require(
        posttrain.get("formal_training_complete") is True,
        "strict posttrain audit does not prove complete formal training",
    )
    _require(
        posttrain.get("online_rollout_started") is False,
        "strict posttrain audit already claims online rollout started",
    )
    checkpoint_rows = posttrain.get("checkpoints")
    _require(isinstance(checkpoint_rows, Mapping), "posttrain audit lacks checkpoints")

    seed_bank = ancestry["final_test_seed_bank"]
    seed_bank_sha = ancestry["final_test_seed_bank_identity"]["sha256"]
    formal_lock_sha = ancestry["formal_protocol_lock_identity"]["sha256"]
    candidates: list[dict[str, Any]] = []
    fairness_by_seed: dict[int, dict[str, tuple[Any, ...]]] = {}
    ancestry_identities: set[tuple[Any, ...]] = set()
    protocol_ids: set[str] = set()
    model_bases: set[str] = set()
    index = 0
    for seed in FORMAL_SEEDS:
        raw_pair = checkpoint_rows.get(str(seed))
        _require(isinstance(raw_pair, Mapping), f"posttrain audit lacks seed {seed}")
        fairness_by_seed[seed] = {}
        for short, control, expected_lambda in CONTROL_ROWS:
            declared_checkpoint = raw_pair.get(short)
            checkpoint = _validate_bound_identity(
                declared_checkpoint,
                label=f"seed {seed} {short} checkpoint",
                expected_path=formal / f"runs/seed_{seed}/{short}/checkpoint.pt",
            )
            model_base, provenance = _resolve_model_base_path(
                checkpoint["path"], None
            )
            contract = _checkpoint_evaluation_contract(
                provenance,
                requested_tasks=TASKS,
                requested_domains=DOMAINS,
                episodes_per_task=EPISODES_PER_CELL,
            )
            _require(contract["control"] == control, f"seed {seed} {short} control differs")
            _require(contract["stage"] == "formal", f"seed {seed} {short} is not formal")
            _require(contract["training_seed"] == seed, f"seed {seed} {short} seed differs")
            _require(contract["policy_regime"] == "p_v1", f"seed {seed} {short} is not P-v1")
            _require(
                float(contract["lambda_contrastive"]) == expected_lambda,
                f"seed {seed} {short} contrastive coefficient differs",
            )
            _require(
                contract["checkpoint_step"] == 1800,
                f"seed {seed} {short} is not the locked 1800-step checkpoint",
            )
            _require(
                contract["formal_evaluation_eligible"] is True,
                f"seed {seed} {short} is not formal-evaluation eligible",
            )
            _require(
                tuple(contract["declared_tasks"]) == TASKS,
                f"seed {seed} {short} task matrix differs",
            )
            _require(
                tuple(contract["declared_domains"]) == DOMAINS,
                f"seed {seed} {short} domain matrix differs",
            )
            _require(
                contract["simulator_seed_bank_id"]
                == seed_bank["simulator_seed_bank_id"],
                f"seed {seed} {short} binds a different final-test bank",
            )
            _require(
                contract["simulator_seed_bank_manifest_sha256"] == seed_bank_sha,
                f"seed {seed} {short} final-test bank SHA differs",
            )
            _require(
                contract["formal_protocol_lock_manifest_sha256"] == formal_lock_sha,
                f"seed {seed} {short} formal lock SHA differs",
            )
            dataset_stats = _stable_file_identity(
                formal / f"runs/seed_{seed}/{short}/dataset_stats.json"
            )
            _require(
                dataset_stats["sha256"] == contract["dataset_stats_sha256"],
                f"seed {seed} {short} dataset stats differ from checkpoint",
            )
            fairness = _fairness_identity_from_checkpoint_contract(
                contract,
                evaluation_control=control,
            )
            fairness_tuple = tuple(fairness[field] for field in FAIRNESS_RECORD_FIELDS)
            fairness_by_seed[seed][short] = fairness_tuple
            ancestry_identities.add(
                tuple(
                    fairness[field]
                    for field in (
                        "base_checkpoint_sha256",
                        "dataset_stats_sha256",
                        "base_lineage_manifest_sha256",
                        "runtime_source_sha256",
                    )
                )
            )
            protocol_ids.add(str(contract["rollout_protocol_id"]))
            model_bases.add(str(model_base.resolve()))
            candidates.append(
                {
                    "candidate_index": index,
                    "training_seed": seed,
                    "short_control": short,
                    "control": control,
                    "lambda_contrastive": expected_lambda,
                    "physical_gpu_index": physical_gpus[index],
                    "checkpoint": checkpoint,
                    "dataset_stats": dataset_stats,
                    "model_base_path": str(model_base.resolve()),
                }
            )
            index += 1

    _require(len(ancestry_identities) == 1, "release ancestry differs across checkpoints")
    _require(len(protocol_ids) == 1, "rollout protocol differs across checkpoints")
    _require(len(model_bases) == 1, "model component base differs across checkpoints")
    for seed in FORMAL_SEEDS:
        _require(
            fairness_by_seed[seed]["c1"] == fairness_by_seed[seed]["c3"],
            f"C1/C3 fairness identity differs for seed {seed}",
        )

    rollout_settings = {
        "mixed_precision": "bf16",
        "instruction_type": "unseen",
        "action_horizon": None,
        "replan_steps": 24,
        "num_inference_steps": 10,
        "sigma_shift": None,
        "text_cfg_scale": 1.0,
        "negative_prompt": "",
        "rand_device": "cpu",
        "tiled": False,
        "timing_enabled": False,
        "skip_get_obs_within_replan": True,
    }
    exact_cells = {
        (str(cell["task"]), str(cell["task_config"])): cell
        for cell in exact_bank["cells"]
    }
    waves: list[dict[str, Any]] = []
    cell_index = 0
    for task in TASKS:
        for task_config in TASK_CONFIGS:
            domain = FORMAL_TASK_CONFIG_TO_DOMAIN[task_config]
            realization_cell = exact_cells[(task, task_config)]
            wave_cells: list[dict[str, Any]] = []
            for candidate in candidates:
                seed = int(candidate["training_seed"])
                short = str(candidate["short_control"])
                cell_root = (
                    output
                    / f"cells/{task}/{domain}/seed_{seed}/{short}"
                ).resolve()
                wave_cells.append(
                    {
                        "cell_index": cell_index,
                        "candidate_index": candidate["candidate_index"],
                        "training_seed": seed,
                        "short_control": short,
                        "control": candidate["control"],
                        "physical_gpu_index": candidate["physical_gpu_index"],
                        "task": task,
                        "task_config": task_config,
                        "domain": domain,
                        "realization_cell_id": realization_cell["cell_id"],
                        "ordered_seed_instruction_sha256": realization_cell[
                            "ordered_seed_instruction_sha256"
                        ],
                        "checkpoint": candidate["checkpoint"],
                        "dataset_stats": candidate["dataset_stats"],
                        "model_base_path": candidate["model_base_path"],
                        "cell_root": str(cell_root),
                        "attempt_policy": (
                            "append-only attempt directories; exactly one completed "
                            "manifest is permitted"
                        ),
                    }
                )
                cell_index += 1
            waves.append(
                {
                    "wave_index": len(waves),
                    "task": task,
                    "task_config": task_config,
                    "domain": domain,
                    "realization_cell_id": realization_cell["cell_id"],
                    "ordered_seed_instruction_sha256": realization_cell[
                        "ordered_seed_instruction_sha256"
                    ],
                    "parallel_cells": wave_cells,
                }
            )
    plan = {
        "kind": "policy_release_formal_c1_c3_online_rollout_plan",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "formal_root": str(formal),
        "rollout_root": str(output),
        "materialization_manifest": ancestry["materialization_identity"],
        "strict_posttrain_pair_audit": _stable_file_identity(posttrain_path),
        "formal_protocol_lock": ancestry["formal_protocol_lock_identity"],
        "final_test_seed_bank": ancestry["final_test_seed_bank_identity"],
        "formal_episode_realization_bank": exact_bank_identity,
        "formal_episode_realization_bank_id": exact_bank["realization_bank_id"],
        "selected_policy_regime": "p_v1",
        "stage2_training_seeds": list(FORMAL_SEEDS),
        "controls": [row[1] for row in CONTROL_ROWS],
        "tasks": list(TASKS),
        "task_configs": list(TASK_CONFIGS),
        "domains": list(DOMAINS),
        "episodes_per_cell": EPISODES_PER_CELL,
        "expected_record_count": 36,
        "expected_completed_manifest_count": 36,
        "simulator_seed": int(seed_bank["simulator_seed"]),
        "simulator_seed_bank_id": seed_bank["simulator_seed_bank_id"],
        "rollout_protocol_id": next(iter(protocol_ids)),
        "rollout_settings": rollout_settings,
        "parallelism": {
            "waves": 6,
            "workers_per_wave": 6,
            "physical_gpu_ids": list(physical_gpus),
            "unit": "one task/domain cell per checkpoint and physical GPU",
            "within_wave": "six C1/C3 checkpoint candidates in parallel",
            "between_waves": "six task/domain waves are sequential and cell-resumable",
        },
        "overwrite_policy": "create_only_per_task_domain_checkpoint_cell",
        "candidates": candidates,
        "waves": waves,
    }
    plan["plan_payload_sha256"] = _canonical_sha256(plan)
    return plan


def validate_rollout_plan(
    plan_path: str | Path,
    *,
    require_output_absent: bool,
) -> dict[str, Any]:
    plan, resolved = _load_json(plan_path, "formal rollout plan")
    _require(plan.get("status") == "PASS", "formal rollout plan is not PASS")
    declared_hash = plan.get("plan_payload_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_payload_sha256", None)
    _require(
        declared_hash == _canonical_sha256(unhashed),
        "formal rollout plan payload SHA differs",
    )
    rebuilt = build_rollout_plan(
        formal_root=str(plan.get("formal_root", "")),
        rollout_root=str(plan.get("rollout_root", "")),
        realization_bank=str(
            plan.get("formal_episode_realization_bank", {}).get("path", "")
        ),
        gpu_ids=plan.get("parallelism", {}).get("physical_gpu_ids", ()),
        require_output_absent=require_output_absent,
    )
    _require(plan == rebuilt, "formal rollout plan differs from current immutable inputs")
    return {"status": "PASS", "plan": _stable_file_identity(resolved), "payload": plan}


def audit_completed_cell(
    plan_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Audit one append-only completed attempt for safe wave resumption."""

    validated = validate_rollout_plan(plan_path, require_output_absent=False)
    plan = validated["payload"]
    manifest = Path(manifest_path).expanduser().resolve()
    _require(manifest.is_file(), f"completed cell manifest is missing: {manifest}")
    planned_cells = [
        cell for wave in plan["waves"] for cell in wave["parallel_cells"]
    ]
    matches = []
    for cell in planned_cells:
        root = Path(cell["cell_root"]).resolve()
        try:
            relative = manifest.relative_to(root)
        except ValueError:
            continue
        if (
            len(relative.parts) == 2
            and relative.parts[0].startswith("attempt_")
            and relative.parts[1] == "completed_rollouts.json"
        ):
            matches.append(cell)
    _require(len(matches) == 1, "completed manifest is outside one planned cell attempt")
    cell = matches[0]
    payload, _ = _load_json(manifest, "completed-rollouts manifest")
    _require(payload.get("schema_version") == 6, "completed cell is not exact-replay schema v6")
    _require(
        payload.get("formal_episode_realization_bank")
        == plan["formal_episode_realization_bank"],
        "completed cell realization-bank raw SHA differs",
    )
    _require(
        Path(str(payload.get("checkpoint", ""))).expanduser().resolve()
        == Path(cell["checkpoint"]["path"]).resolve(),
        "completed cell checkpoint differs",
    )
    _require(
        Path(str(payload.get("output_dir", ""))).expanduser().resolve()
        == manifest.parent,
        "completed cell output directory differs",
    )
    runs = payload.get("runs")
    _require(isinstance(runs, list) and len(runs) == 1, "completed cell must contain one run")
    run = runs[0]
    _require(isinstance(run, Mapping), "completed cell run is invalid")
    _require(
        (run.get("task"), run.get("task_config"), run.get("domain"))
        == (cell["task"], cell["task_config"], cell["domain"]),
        "completed cell task/domain differs",
    )
    _require(
        run.get("formal_episode_realization_cell_id")
        == cell["realization_cell_id"],
        "completed cell realization-cell id differs",
    )
    _require(
        run.get("ordered_seed_instruction_sha256")
        == cell["ordered_seed_instruction_sha256"],
        "completed cell ordered seed/instruction SHA differs",
    )
    from .eval_robotwin_single import _records_from_completed_manifest

    records = _records_from_completed_manifest(payload)
    _require(len(records) == 1, "completed cell transport audit returned !=1 record")
    return {
        "status": "PASS",
        "cell_index": cell["cell_index"],
        "training_seed": cell["training_seed"],
        "control": cell["control"],
        "task": cell["task"],
        "domain": cell["domain"],
        "manifest": _stable_file_identity(manifest),
        "record": records[0],
    }


def aggregate_formal_rollouts(plan_path: str | Path) -> dict[str, Any]:
    validated = validate_rollout_plan(plan_path, require_output_absent=False)
    plan = validated["payload"]
    rollout_root = Path(plan["rollout_root"]).resolve()
    _require(rollout_root.is_dir(), f"formal rollout root is missing: {rollout_root}")
    manifests: list[Path] = []
    planned_cells = [
        cell
        for wave in plan["waves"]
        for cell in wave["parallel_cells"]
    ]
    _require(len(planned_cells) == 36, "formal plan no longer contains 36 cells")
    completed_by_cell: dict[int, Path] = {}
    for cell in planned_cells:
        cell_root = Path(cell["cell_root"]).resolve()
        matches = sorted(
            path.resolve() for path in cell_root.glob("attempt_*/completed_rollouts.json")
        )
        _require(
            len(matches) == 1,
            f"cell {cell['cell_index']} must have exactly one completed append-only attempt; "
            f"found {len(matches)}",
        )
        completed_by_cell[int(cell["cell_index"])] = matches[0]
    expected_manifest_paths = set(completed_by_cell.values())
    actual_manifest_paths = {
        path.resolve() for path in rollout_root.rglob("completed_rollouts.json")
    }
    _require(
        actual_manifest_paths == expected_manifest_paths,
        "completed-rollouts manifest set differs from the 36 planned exact cells",
    )
    sequence_by_task_domain: dict[tuple[str, str], set[str]] = {}
    for cell in planned_cells:
        manifest_path = completed_by_cell[int(cell["cell_index"])]
        payload, _ = _load_json(manifest_path, "completed-rollouts manifest")
        _require(
            payload.get("schema_version") == 6,
            f"completed cell {cell['cell_index']} is not exact-replay schema v6",
        )
        _require(
            payload.get("formal_exact_episode_replay") is True,
            f"completed cell {cell['cell_index']} lacks exact-replay proof",
        )
        _require(
            payload.get("formal_episode_realization_bank_id")
            == plan["formal_episode_realization_bank_id"],
            f"completed cell {cell['cell_index']} realization bank differs",
        )
        _require(
            payload.get("formal_episode_realization_bank")
            == plan["formal_episode_realization_bank"],
            f"completed cell {cell['cell_index']} realization-bank SHA differs",
        )
        _require(
            Path(str(payload.get("checkpoint", ""))).expanduser().resolve()
            == Path(cell["checkpoint"]["path"]).resolve(),
            f"completed manifest checkpoint differs for cell {cell['cell_index']}",
        )
        _require(
            Path(str(payload.get("output_dir", ""))).expanduser().resolve()
            == manifest_path.parent,
            f"completed manifest output directory differs from its append-only attempt "
            f"for cell {cell['cell_index']}",
        )
        runs = payload.get("runs")
        _require(
            isinstance(runs, list) and len(runs) == 1,
            f"cell {cell['cell_index']} must contain exactly one completed run",
        )
        run = runs[0]
        _require(isinstance(run, Mapping), f"cell {cell['cell_index']} run is invalid")
        _require(
            (run.get("task"), run.get("task_config"), run.get("domain"))
            == (cell["task"], cell["task_config"], cell["domain"]),
            f"cell {cell['cell_index']} task/domain differs",
        )
        _require(
            run.get("formal_episode_realization_cell_id")
            == cell["realization_cell_id"],
            f"cell {cell['cell_index']} realization cell differs",
        )
        sequence_sha = str(run.get("ordered_seed_instruction_sha256", ""))
        _require(
            sequence_sha == cell["ordered_seed_instruction_sha256"],
            f"cell {cell['cell_index']} ordered seed/instruction SHA differs",
        )
        sequence_by_task_domain.setdefault(
            (cell["task"], cell["domain"]), set()
        ).add(sequence_sha)
        _require(
            payload.get("episodes_per_task") == EPISODES_PER_CELL,
            f"cell {cell['cell_index']} episode count differs",
        )
        _require(
            payload.get("simulator_seed_bank_id") == plan["simulator_seed_bank_id"],
            f"cell {cell['cell_index']} final-test candidate bank differs",
        )
        # Re-read and fully validate trace/outcomes/success-rate through the
        # transport converter before trusting any embedded evaluation record.
        from .eval_robotwin_single import _records_from_completed_manifest

        _require(
            len(_records_from_completed_manifest(payload)) == 1,
            f"cell {cell['cell_index']} transport audit did not return one record",
        )
        manifests.append(manifest_path)

    _require(
        set(sequence_by_task_domain)
        == {(task, domain) for task in TASKS for domain in DOMAINS},
        "formal exact task/domain sequence matrix differs",
    )
    for key, digests in sequence_by_task_domain.items():
        _require(
            len(digests) == 1,
            f"six checkpoints did not replay one identical sequence for {key}",
        )

    evaluation_payload = aggregate_completed_rollout_manifests(manifests)
    _require(
        evaluation_payload.get("formal_exact_episode_replay") is True,
        "aggregate does not prove formal exact episode replay",
    )
    _require(
        evaluation_payload.get("formal_episode_realization_bank_id")
        == plan["formal_episode_realization_bank_id"],
        "aggregate realization-bank id differs",
    )
    _require(
        len(evaluation_payload["records"]) == plan["expected_record_count"],
        "aggregated formal record count differs",
    )
    summary = audit_and_summarize(
        evaluation_payload,
        training_seeds=FORMAL_SEEDS,
        episodes_per_cell=EPISODES_PER_CELL,
    )
    aggregate_root = rollout_root / "aggregate"
    _require(
        not aggregate_root.exists(),
        f"refusing to overwrite formal aggregate root: {aggregate_root}",
    )
    evaluation_path = _write_new_json(
        aggregate_root / "evaluation_records.json", evaluation_payload
    )
    summary_path = _write_new_json(aggregate_root / "summary.json", summary)
    completion = {
        "kind": "policy_release_formal_c1_c3_online_rollout_completion_audit",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "plan": validated["plan"],
        "completed_manifests": [
            _stable_file_identity(path) for path in manifests
        ],
        "evaluation_records": _stable_file_identity(evaluation_path),
        "summary": _stable_file_identity(summary_path),
        "record_count": summary["record_count"],
        "primary_comparison": summary["primary_comparison"],
        "online_rollout_complete": True,
    }
    completion_path = _write_new_json(
        aggregate_root / "completion_audit.json", completion
    )
    return {
        **completion,
        "completion_audit": _stable_file_identity(completion_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="CPU-only immutable prelaunch audit")
    prepare.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    prepare.add_argument("--rollout-root")
    prepare.add_argument("--realization-bank")
    prepare.add_argument("--gpu-ids", default=",".join(map(str, DEFAULT_GPU_IDS)))
    prepare.add_argument("--output-plan")

    audit_plan = subparsers.add_parser("audit-plan", help="revalidate a saved plan")
    audit_plan.add_argument("--plan", required=True)
    audit_plan.add_argument("--allow-existing-rollout-root", action="store_true")

    audit_cell = subparsers.add_parser(
        "audit-cell", help="validate one completed append-only cell attempt"
    )
    audit_cell.add_argument("--plan", required=True)
    audit_cell.add_argument("--manifest", required=True)

    aggregate = subparsers.add_parser("aggregate", help="strictly aggregate six candidates")
    aggregate.add_argument("--plan", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        formal_root = Path(args.formal_root).expanduser().resolve()
        rollout_root = (
            Path(args.rollout_root).expanduser().resolve()
            if args.rollout_root
            else (formal_root / DEFAULT_ROLLOUT_SUBDIR).resolve()
        )
        output_plan = (
            Path(args.output_plan).expanduser().resolve()
            if args.output_plan
            else (formal_root / "manifests" / DEFAULT_PLAN_NAME).resolve()
        )
        report = build_rollout_plan(
            formal_root=formal_root,
            rollout_root=rollout_root,
            realization_bank=args.realization_bank,
            gpu_ids=args.gpu_ids,
            require_output_absent=True,
        )
        destination = _write_new_json(output_plan, report)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "gpu_started": False,
                    "plan": _stable_file_identity(destination),
                    "rollout_root": str(rollout_root),
                    "waves": len(report["waves"]),
                    "cells": report["expected_completed_manifest_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "audit-plan":
        report = validate_rollout_plan(
            args.plan,
            require_output_absent=not args.allow_existing_rollout_root,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "plan": report["plan"],
                    "gpu_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "audit-cell":
        print(
            json.dumps(
                audit_completed_cell(args.plan, args.manifest),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(aggregate_formal_rollouts(args.plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONTROL_ROWS",
    "DEFAULT_GPU_IDS",
    "DEFAULT_PLAN_NAME",
    "DEFAULT_ROLLOUT_SUBDIR",
    "EPISODES_PER_CELL",
    "FORMAL_SEEDS",
    "FormalRolloutError",
    "aggregate_formal_rollouts",
    "audit_completed_cell",
    "build_rollout_plan",
    "normalize_gpu_ids",
    "validate_rollout_plan",
]
