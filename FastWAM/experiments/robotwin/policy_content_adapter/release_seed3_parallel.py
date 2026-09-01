#!/usr/bin/env python3
"""Create and audit the no-stop, packed Seed-3 rollout acceleration.

The original author-stock runner remains untouched and continues through
Seed 1 and Seed 2.  This sidecar advances only planned cells 24..35.  Those
cells retain their checkpoint, task/domain, simulator seed, policy settings,
and append-only output roots; only wall-clock order and physical GPU placement
change.  The original runner later audits and skips every completed cell.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import release_seed23_packed as packed
from .release_formal_stock_rollout import audit_stock_completed_cell


KIND = "policy_release_author_stock_seed3_parallel_schedule"
SCHEMA_VERSION = 1
COMPLETION_KIND = "policy_release_author_stock_seed3_parallel_completion"
COMPLETION_SCHEMA_VERSION = 1
TARGET_INDICES = tuple(range(24, 36))

# Launch order is one new evaluator per GPU first, then the second slots.
# GPU 2/4 may already host Seed-1 Open; GPU 3 hosts the small SAM3 service.
ASSIGNMENTS = (
    (26, 0),  # seed3/C1 Open clean
    (27, 1),  # seed3/C1 Open random
    (28, 2),  # seed3/C1 Move clean
    (32, 3),  # seed3/C3 Open clean
    (29, 4),  # seed3/C1 Move random
    (33, 5),  # seed3/C3 Open random
    (34, 6),  # seed3/C3 Move clean
    (35, 7),  # seed3/C3 Move random
    (24, 0),  # seed3/C1 Place clean
    (25, 1),  # seed3/C1 Place random
    (30, 5),  # seed3/C3 Place clean
    (31, 6),  # seed3/C3 Place random
)


class Seed3ParallelError(RuntimeError):
    """The Seed-3 no-stop acceleration contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Seed3ParallelError(message)


def _core(
    plan_path: str | Path,
    continuation_path: str | Path,
    gpu_reports: Sequence[str | Path],
    *,
    require_roots_absent: bool,
) -> dict[str, Any]:
    plan, plan_identity, continuation, gates = packed.validate_inputs(
        plan_path, continuation_path
    )
    cells = packed._cells(plan)
    _require(
        {index for index, _ in ASSIGNMENTS} == set(TARGET_INDICES),
        "Seed-3 assignment does not cover cells 24..35 exactly once",
    )
    _require(
        len({index for index, _ in ASSIGNMENTS}) == len(ASSIGNMENTS),
        "Seed-3 assignment contains duplicate cells",
    )
    helper_counts = {gpu: 0 for gpu in range(8)}
    rows: list[dict[str, Any]] = []
    for order, (index, gpu) in enumerate(ASSIGNMENTS):
        row = cells[index]
        helper_counts[gpu] += 1
        _require(helper_counts[gpu] <= 2, f"GPU {gpu} exceeds two helper jobs")
        _require(
            row.get("training_seed") == 3
            and row.get("short_control") in {"c1", "c3"}
            and row.get("control")
            in {"c1_architecture_only", "c3_ours"},
            f"cell {index} is not a formal Seed-3 C1/C3 cell",
        )
        root = Path(str(row.get("cell_root", ""))).resolve()
        if require_roots_absent:
            _require(not root.exists(), f"Seed-3 cell root already exists: {root}")
        rows.append(
            {
                "launch_order": order,
                "cell_index": index,
                "actual_gpu": gpu,
                "planned_gpu": row["physical_gpu_index"],
                "training_seed": 3,
                "control": row["control"],
                "short_control": row["short_control"],
                "task": row["task"],
                "task_config": row["task_config"],
                "domain": row["domain"],
                "cell_root": str(root),
                "checkpoint": row["checkpoint"],
                "dataset_stats": row["dataset_stats"],
                "attempt_policy": row["attempt_policy"],
            }
        )
    reports = packed.validate_gpu_reports(gpu_reports)
    seed3_gates = [row for row in gates if row["training_seed"] == 3]
    _require(len(seed3_gates) == 2, "Seed-3 C1/C3 deployment gates differ")
    return {
        "evaluation_profile": plan["evaluation_profile"],
        "stock_rollout_plan": plan_identity,
        "asset_repair_continuation": continuation["continuation"],
        "stock_protocol_amendment": plan["stock_protocol_amendment"],
        "helper_sources": {
            "python": packed._identity(Path(__file__)),
            "shell": packed._identity(
                Path(__file__).with_name("run_release_seed3_parallel.sh")
            ),
            "shared_packed_validation": packed._identity(packed.__file__),
        },
        "deployment_gates": seed3_gates,
        "gpu_preflight_reports": reports,
        "assignments": rows,
        "execution_change": {
            "seed3_runs_while_stock_runner_finishes_seed1_and_seed2": True,
            "main_runner_signaled_or_stopped": False,
            "wall_clock_order_and_gpu_placement_only": True,
            "maximum_helper_evaluators_per_gpu": 2,
            "gpu3_maximum_helper_evaluators": 1,
            "failed_or_partial_cells_fall_back_to_original_runner": True,
        },
        "immutable_protocol": {
            "checkpoint_bytes_changed": False,
            "dataset_stats_changed": False,
            "simulator_seed_changed": False,
            "stock_evaluation_settings_changed": False,
            "cell_output_roots_changed": False,
            "episode_pairing": "not_claimed",
            "append_only_attempts": True,
            "audit_each_completed_cell": True,
        },
    }


def materialize(
    *,
    plan_path: str | Path,
    continuation_path: str | Path,
    gpu_reports: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    core = _core(
        plan_path, continuation_path, gpu_reports, require_roots_absent=True
    )
    payload = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **core,
    }
    payload["schedule_id"] = "seed3-parallel-v1:" + packed.common._canonical_sha256(
        core
    )
    packed._write_new_json(output, payload)
    return payload


def validate(path: str | Path) -> dict[str, Any]:
    payload, resolved = packed._load_json(path, "Seed-3 parallel schedule")
    _require(payload.get("kind") == KIND, "Seed-3 schedule kind differs")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema differs")
    _require(payload.get("status") == "PASS", "schedule is not PASS")
    plan = packed._verify_identity(payload["stock_rollout_plan"], "schedule plan")
    continuation = packed._verify_identity(
        payload["asset_repair_continuation"], "schedule continuation"
    )
    raw_reports = payload.get("gpu_preflight_reports")
    _require(isinstance(raw_reports, list), "GPU reports are missing")
    report_paths = [row["report"]["path"] for row in raw_reports]
    core = _core(
        plan["path"], continuation["path"], report_paths, require_roots_absent=False
    )
    for key, expected in core.items():
        _require(payload.get(key) == expected, f"schedule field differs: {key}")
    expected_id = "seed3-parallel-v1:" + packed.common._canonical_sha256(core)
    _require(payload.get("schedule_id") == expected_id, "schedule id differs")
    return {
        "status": "PASS",
        "schedule": packed._identity(resolved),
        "schedule_id": expected_id,
        "payload": payload,
    }


def emit(path: str | Path) -> None:
    rows = validate(path)["payload"]["assignments"]
    for row in rows:
        values = (
            row["cell_index"],
            row["actual_gpu"],
            row["checkpoint"]["path"],
            row["dataset_stats"]["path"],
            row["task"],
            row["task_config"],
            row["domain"],
            row["cell_root"],
        )
        text = "\t".join(str(value) for value in values)
        _require("\n" not in text and "\r" not in text, "invalid shell record")
        print(text)


def complete(
    *, schedule_path: str | Path, output: str | Path
) -> dict[str, Any]:
    schedule = validate(schedule_path)
    plan_path = schedule["payload"]["stock_rollout_plan"]["path"]
    completed: list[dict[str, Any]] = []
    for row in schedule["payload"]["assignments"]:
        root = Path(row["cell_root"])
        manifests = sorted(root.glob("attempt_*/completed_rollouts.json"))
        _require(len(manifests) == 1, f"cell {row['cell_index']} is not uniquely complete")
        audit = audit_stock_completed_cell(plan_path, manifests[0])
        completed.append(
            {
                "cell_index": row["cell_index"],
                "actual_gpu": row["actual_gpu"],
                "manifest": audit["manifest"],
                "success_rate": audit["record"]["success_rate"],
                "episodes": audit["record"]["episodes"],
            }
        )
    core = {
        "schedule": schedule["schedule"],
        "schedule_id": schedule["schedule_id"],
        "completed_cells": completed,
        "completed_cell_count": 12,
        "main_runner_will_audit_and_skip": True,
    }
    payload = {
        "kind": COMPLETION_KIND,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **core,
        "completion_id": "seed3-parallel-complete-v1:"
        + packed.common._canonical_sha256(core),
    }
    packed._write_new_json(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("materialize")
    create.add_argument("--plan", required=True)
    create.add_argument("--continuation", required=True)
    create.add_argument("--gpu-preflight-report", action="append", required=True)
    create.add_argument("--output", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--path", required=True)
    rows = commands.add_parser("emit")
    rows.add_argument("--path", required=True)
    done = commands.add_parser("complete")
    done.add_argument("--schedule", required=True)
    done.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize(
            plan_path=args.plan,
            continuation_path=args.continuation,
            gpu_reports=args.gpu_preflight_report,
            output=args.output,
        )
        printable = {"status": result["status"], "schedule_id": result["schedule_id"]}
    elif args.command == "validate":
        result = validate(args.path)
        printable = {key: value for key, value in result.items() if key != "payload"}
    elif args.command == "emit":
        emit(args.path)
        return 0
    else:
        result = complete(schedule_path=args.schedule, output=args.output)
        printable = {"status": result["status"], "completion_id": result["completion_id"]}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
