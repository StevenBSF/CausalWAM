"""Plan and aggregate the author-stock seed-42 formal rollout profile.

This profile intentionally follows RoboTwin's stock evaluation behavior: each
checkpoint/task/domain process starts from the same seed-42 candidate range
and independently runs the stock expert filter until 100 episodes are
accepted.  Actual accepted episodes are not claimed to be paired across
models.  The external amendment is immutable and changes evaluation metadata
only; all six trained checkpoint bytes remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .eval_robotwin_single import (
    STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION,
    _records_from_completed_manifest,
    aggregate_completed_rollout_manifests,
)
from .evaluation_protocol import DOMAINS, TASKS, audit_and_summarize
from .release_stock_eval_protocol import (
    DEFAULT_AMENDMENT_RELATIVE,
    DEFAULT_FORMAL_ROOT,
    EPISODES_PER_CELL,
    PROFILE,
    SIMULATOR_SEED,
    TASK_CONFIGS,
    validate_stock_eval_amendment,
)
from .runtime_utils import PROJECT_ROOT


SCHEMA_VERSION = 1
DEFAULT_GPU_IDS = (0, 1, 2, 4, 5, 6)
DEFAULT_OUTPUT_SUBDIR = "online_rollouts_author_stock_seed42_v1"
DEFAULT_PLAN_NAME = "author_stock_seed42_rollout_plan_v1.json"
TASK_CONFIG_TO_DOMAIN = {
    "demo_clean": "clean",
    "demo_randomized": "official_random",
}


class StockRolloutError(ValueError):
    """The author-stock rollout matrix cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StockRolloutError(message)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
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


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StockRolloutError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} root must be an object")
    return payload, resolved


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def normalize_gpu_ids(values: str | Sequence[int | str]) -> tuple[int, ...]:
    raw = (
        [part.strip() for part in values.split(",")]
        if isinstance(values, str)
        else [str(value).strip() for value in values]
    )
    _require(len(raw) == 6, "author-stock rollout requires exactly six GPUs")
    _require(all(part.isdigit() for part in raw), "GPU ids must be non-negative integers")
    parsed = tuple(int(part) for part in raw)
    _require(len(set(parsed)) == 6, "GPU ids must be unique")
    return parsed


def _assert_formal_training_done(formal_root: Path) -> None:
    status = formal_root / "formal_c1_c3.status"
    _require(status.is_file(), f"formal status is missing: {status}")
    _require(
        status.read_text(encoding="utf-8").strip().startswith(
            "DONE formal_training=true online_rollout_started=false "
        ),
        "formal status does not prove completed training before rollout",
    )


def build_stock_rollout_plan(
    *,
    formal_root: str | Path,
    rollout_root: str | Path,
    amendment_path: str | Path | None = None,
    gpu_ids: str | Sequence[int | str] = DEFAULT_GPU_IDS,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    formal = Path(formal_root).expanduser().resolve()
    output = Path(rollout_root).expanduser().resolve()
    _require(formal.is_dir(), f"formal root is missing: {formal}")
    if require_output_absent:
        _require(not output.exists(), f"refusing to reuse rollout root: {output}")
    _assert_formal_training_done(formal)
    physical_gpus = normalize_gpu_ids(gpu_ids)
    amendment_file = (
        Path(amendment_path).expanduser().resolve()
        if amendment_path is not None
        else (formal / DEFAULT_AMENDMENT_RELATIVE).resolve()
    )
    amendment, amendment_file = validate_stock_eval_amendment(amendment_file)
    amendment_identity = _file_identity(amendment_file)
    _require(amendment["profile"] == PROFILE, "stock profile differs")
    _require(amendment["simulator_seed"] == SIMULATOR_SEED, "stock seed differs")
    _require(
        amendment["episodes_per_cell"] == EPISODES_PER_CELL,
        "stock episode count differs",
    )

    waves: list[dict[str, Any]] = []
    cell_index = 0
    for checkpoint_index, row in enumerate(amendment["checkpoints"]):
        checkpoint = _file_identity(row["path"])
        for field in ("path", "size_bytes", "sha256"):
            _require(checkpoint[field] == row[field], f"checkpoint row {field} differs")
        checkpoint_path = Path(row["path"]).resolve()
        dataset_stats = _file_identity(checkpoint_path.parent / "dataset_stats.json")
        short = "c1" if row["control"] == "c1_architecture_only" else "c3"
        cells: list[dict[str, Any]] = []
        for gpu_index, (task, task_config) in enumerate(
            (item for task in TASKS for item in ((task, TASK_CONFIGS[0]), (task, TASK_CONFIGS[1])))
        ):
            domain = TASK_CONFIG_TO_DOMAIN[task_config]
            cell_root = (
                output
                / f"cells/seed_{row['training_seed']}/{short}/{task}/{domain}"
            ).resolve()
            cells.append(
                {
                    "cell_index": cell_index,
                    "checkpoint_index": checkpoint_index,
                    "physical_gpu_index": physical_gpus[gpu_index],
                    "control": row["control"],
                    "short_control": short,
                    "training_seed": row["training_seed"],
                    "checkpoint": checkpoint,
                    "dataset_stats": dataset_stats,
                    "task": task,
                    "task_config": task_config,
                    "domain": domain,
                    "cell_root": str(cell_root),
                    "attempt_policy": (
                        "append-only attempts; exactly one completed manifest permitted"
                    ),
                }
            )
            cell_index += 1
        waves.append(
            {
                "wave_index": checkpoint_index,
                "control": row["control"],
                "short_control": short,
                "training_seed": row["training_seed"],
                "checkpoint": checkpoint,
                "parallel_task_domain_cells": cells,
            }
        )

    _require(len(waves) == 6 and cell_index == 36, "stock rollout matrix is not 6x6")
    plan = {
        "kind": "policy_release_author_stock_seed42_rollout_plan",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "evaluation_profile": PROFILE,
        "formal_root": str(formal),
        "rollout_root": str(output),
        "stock_protocol_amendment": amendment_identity,
        "stock_protocol_amendment_id": amendment["amendment_id"],
        "simulator_seed": SIMULATOR_SEED,
        "candidate_start_seed": amendment["runtime_seed_bank"][
            "candidate_start_seed"
        ],
        "simulator_seed_bank_id": amendment["runtime_seed_bank"][
            "simulator_seed_bank_id"
        ],
        "episodes_per_cell": EPISODES_PER_CELL,
        "tasks": list(TASKS),
        "task_configs": list(TASK_CONFIGS),
        "domains": list(DOMAINS),
        "expected_completed_manifest_count": 36,
        "episode_pairing": "not_claimed",
        "shared_starting_seed_only": True,
        "per_checkpoint_expert_filtering": True,
        "accepted_episode_sequence_recorded": False,
        "parallelism": {
            "checkpoint_waves": 6,
            "workers_per_wave": 6,
            "physical_gpu_ids": list(physical_gpus),
            "within_wave": "one checkpoint; six task/domain cells in parallel",
            "between_waves": "six C1/C3 checkpoint waves in immutable row order",
        },
        "waves": waves,
    }
    plan["plan_payload_sha256"] = _canonical_sha256(plan)
    return plan


def validate_stock_rollout_plan(
    path: str | Path,
    *,
    require_output_absent: bool,
) -> dict[str, Any]:
    plan, resolved = _load_json(path, "stock rollout plan")
    declared = plan.get("plan_payload_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_payload_sha256", None)
    _require(declared == _canonical_sha256(unhashed), "stock plan payload SHA differs")
    rebuilt = build_stock_rollout_plan(
        formal_root=plan.get("formal_root", ""),
        rollout_root=plan.get("rollout_root", ""),
        amendment_path=plan.get("stock_protocol_amendment", {}).get("path", ""),
        gpu_ids=plan.get("parallelism", {}).get("physical_gpu_ids", ()),
        require_output_absent=require_output_absent,
    )
    _require(plan == rebuilt, "stock rollout plan differs from immutable inputs")
    return {"status": "PASS", "plan": _file_identity(resolved), "payload": plan}


def _planned_cells(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        cell
        for wave in plan["waves"]
        for cell in wave["parallel_task_domain_cells"]
    ]


def audit_stock_completed_cell(
    plan_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    validated = validate_stock_rollout_plan(
        plan_path, require_output_absent=False
    )
    plan = validated["payload"]
    manifest = Path(manifest_path).expanduser().resolve()
    payload, _ = _load_json(manifest, "stock completed-rollouts manifest")
    matches: list[Mapping[str, Any]] = []
    for cell in _planned_cells(plan):
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
    _require(len(matches) == 1, "completed manifest is outside one planned cell")
    cell = matches[0]
    _require(
        payload.get("schema_version") == STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        "completed cell is not author-stock schema v7",
    )
    _require(
        payload.get("stock_protocol_amendment")
        == plan["stock_protocol_amendment"],
        "completed cell amendment raw SHA differs",
    )
    _require(
        Path(str(payload.get("checkpoint", ""))).resolve()
        == Path(cell["checkpoint"]["path"]).resolve(),
        "completed cell checkpoint differs",
    )
    _require(
        Path(str(payload.get("output_dir", ""))).resolve() == manifest.parent,
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
    records = _records_from_completed_manifest(payload)
    _require(len(records) == 1, "stock transport audit returned !=1 record")
    return {
        "status": "PASS",
        "cell_index": cell["cell_index"],
        "manifest": _file_identity(manifest),
        "record": records[0],
        "episode_pairing": "not_claimed",
    }


def aggregate_stock_rollouts(plan_path: str | Path) -> dict[str, Any]:
    validated = validate_stock_rollout_plan(
        plan_path, require_output_absent=False
    )
    plan = validated["payload"]
    rollout_root = Path(plan["rollout_root"]).resolve()
    _require(rollout_root.is_dir(), f"rollout root is missing: {rollout_root}")
    manifests: list[Path] = []
    for cell in _planned_cells(plan):
        root = Path(cell["cell_root"]).resolve()
        matches = sorted(root.glob("attempt_*/completed_rollouts.json"))
        _require(
            len(matches) == 1,
            f"cell {cell['cell_index']} must contain one completed attempt; found {len(matches)}",
        )
        audit_stock_completed_cell(plan_path, matches[0])
        manifests.append(matches[0].resolve())
    actual = {path.resolve() for path in rollout_root.rglob("completed_rollouts.json")}
    _require(actual == set(manifests), "unexpected completed manifest exists")
    evaluation = aggregate_completed_rollout_manifests(manifests)
    _require(len(evaluation["records"]) == 36, "stock aggregate record count differs")
    _require(evaluation.get("evaluation_profile") == PROFILE, "stock aggregate profile differs")
    _require(
        evaluation.get("stock_protocol_amendment_id")
        == plan["stock_protocol_amendment_id"],
        "stock aggregate amendment id differs",
    )
    _require(evaluation.get("episode_pairing") == "not_claimed", "stock aggregate pairing claim differs")
    summary = audit_and_summarize(
        evaluation,
        training_seeds=(1, 2, 3),
        episodes_per_cell=EPISODES_PER_CELL,
    )
    summary.update(
        {
            "evaluation_profile": PROFILE,
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
            "comparison_interpretation": (
                "C1/C3 are matched by Stage-2 training seed; online success-rate "
                "cells follow author-stock independent expert filtering and are "
                "not episode-paired."
            ),
        }
    )
    aggregate_root = rollout_root / "aggregate"
    _require(not aggregate_root.exists(), f"refusing to overwrite aggregate: {aggregate_root}")
    evaluation_path = _write_new_json(
        aggregate_root / "evaluation_records.json", evaluation
    )
    summary_path = _write_new_json(aggregate_root / "summary.json", summary)
    completion = {
        "kind": "policy_release_author_stock_seed42_completion_audit",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "evaluation_profile": PROFILE,
        "episode_pairing": "not_claimed",
        "stock_protocol_amendment": plan["stock_protocol_amendment"],
        "plan": validated["plan"],
        "completed_manifests": [_file_identity(path) for path in manifests],
        "evaluation_records": _file_identity(evaluation_path),
        "summary": _file_identity(summary_path),
        "record_count": 36,
        "online_rollout_complete": True,
    }
    completion_path = _write_new_json(
        aggregate_root / "completion_audit.json", completion
    )
    return {**completion, "completion_audit": _file_identity(completion_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    prepare.add_argument("--rollout-root")
    prepare.add_argument("--amendment")
    prepare.add_argument("--gpu-ids", default=",".join(map(str, DEFAULT_GPU_IDS)))
    prepare.add_argument("--output-plan")
    audit_plan = commands.add_parser("audit-plan")
    audit_plan.add_argument("--plan", required=True)
    audit_plan.add_argument("--allow-existing-rollout-root", action="store_true")
    audit_cell = commands.add_parser("audit-cell")
    audit_cell.add_argument("--plan", required=True)
    audit_cell.add_argument("--manifest", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--plan", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        formal = Path(args.formal_root).resolve()
        output = (
            Path(args.rollout_root).resolve()
            if args.rollout_root
            else (formal / DEFAULT_OUTPUT_SUBDIR).resolve()
        )
        plan_path = (
            Path(args.output_plan).resolve()
            if args.output_plan
            else (formal / "manifests" / DEFAULT_PLAN_NAME).resolve()
        )
        plan = build_stock_rollout_plan(
            formal_root=formal,
            rollout_root=output,
            amendment_path=args.amendment,
            gpu_ids=args.gpu_ids,
            require_output_absent=True,
        )
        written = _write_new_json(plan_path, plan)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "gpu_started": False,
                    "evaluation_profile": PROFILE,
                    "plan": _file_identity(written),
                    "checkpoint_waves": 6,
                    "cells": 36,
                    "episode_pairing": "not_claimed",
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "audit-plan":
        report = validate_stock_rollout_plan(
            args.plan,
            require_output_absent=not args.allow_existing_rollout_root,
        )
        print(json.dumps({"status": "PASS", "plan": report["plan"]}, indent=2))
    elif args.command == "audit-cell":
        print(json.dumps(audit_stock_completed_cell(args.plan, args.manifest), indent=2))
    else:
        print(json.dumps(aggregate_stock_rollouts(args.plan), indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_GPU_IDS",
    "DEFAULT_OUTPUT_SUBDIR",
    "DEFAULT_PLAN_NAME",
    "StockRolloutError",
    "aggregate_stock_rollouts",
    "audit_stock_completed_cell",
    "build_stock_rollout_plan",
    "normalize_gpu_ids",
    "validate_stock_rollout_plan",
]
