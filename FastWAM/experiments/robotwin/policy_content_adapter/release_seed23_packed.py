#!/usr/bin/env python3
"""Strict one-shot scheduler amendment for packed seed-2/3 stock rollout cells.

The module is deliberately signal-free.  It validates the immutable formal
experiment, binds the exact live seed-1/C3 Open wait window, writes a
create-only scheduling amendment only after the parent is proven stopped, and
emits the paired cell queue consumed by ``run_release_seed23_packed.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import release_c3_overlap as common
from .release_stock_eval_protocol import PROFILE
from .runtime_utils import PROJECT_ROOT


KIND = "policy_release_author_stock_seed23_packed_schedule"
SCHEMA_VERSION = 1
PREFLIGHT_KIND = "policy_release_seed23_packed_live_preflight"
PREFLIGHT_SCHEMA_VERSION = 1
TARGET_CELL_INDICES = tuple(range(12, 36))
BLOCKING_CELL_INDICES = (8, 9)
ALL_GPU_IDS = tuple(range(8))
DOUBLE_SLOT_GPUS = (0, 1, 5, 6, 7)
SINGLE_HELPER_SLOT_GPUS = (2, 3, 4)
MIN_FREE_GPU_MIB = {
    **{gpu: 60_000 for gpu in DOUBLE_SLOT_GPUS},
    **{gpu: 30_000 for gpu in SINGLE_HELPER_SLOT_GPUS},
}
PAIR_INDICES = tuple(
    [(12 + offset, 18 + offset) for offset in range(6)]
    + [(24 + offset, 30 + offset) for offset in range(6)]
)
# Four Open pairs first (eight cells, one per GPU), then Place/Move pairs.
PAIR_QUEUE_ORDER = (2, 3, 8, 9, 0, 1, 4, 5, 6, 7, 10, 11)


class Seed23PackedError(RuntimeError):
    """The packed seed-2/3 schedule cannot be proven safe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Seed23PackedError(message)


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    try:
        return common._load_json(path, label)
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    try:
        return common._write_new_json(path, value)
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc


def _identity(path: str | Path) -> dict[str, Any]:
    try:
        return common._file_identity(path)
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc


def _verify_identity(value: Any, label: str) -> dict[str, Any]:
    try:
        return common._verify_file_identity(value, label)
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc


def _cells(plan: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    try:
        return common._cells_by_index(plan)
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc


def _validate_gate(path: Path, checkpoint: str | Path, label: str) -> dict[str, Any]:
    gate, resolved = _load_json(path, label)
    _require(gate.get("status") == "PASS", f"{label} is not PASS")
    _require(
        Path(str(gate.get("checkpoint", ""))).resolve() == Path(checkpoint).resolve(),
        f"{label} checkpoint differs",
    )
    tasks = gate.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 3, f"{label} task count differs")
    _require(
        {row.get("task") for row in tasks if isinstance(row, Mapping)}
        == {"place_a2b_left", "open_microwave", "move_stapler_pad"},
        f"{label} tasks differ",
    )
    _require(
        all(
            isinstance(row, Mapping)
            and row.get("action_finite") is True
            and row.get("action_shape") == [14]
            for row in tasks
        ),
        f"{label} action contract differs",
    )
    return _identity(resolved)


def validate_inputs(
    plan_path: str | Path, continuation_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        plan, plan_identity, continuation = common._validate_experiment_inputs(
            plan_path, continuation_path
        )
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc
    _require(plan.get("evaluation_profile") == PROFILE, "stock profile differs")
    cells = _cells(plan)
    for index in TARGET_CELL_INDICES:
        row = cells[index]
        seed = 2 if index < 24 else 3
        short = "c1" if index in range(12, 18) or index in range(24, 30) else "c3"
        control = "c1_architecture_only" if short == "c1" else "c3_ours"
        _require(
            (row.get("training_seed"), row.get("short_control"), row.get("control"))
            == (seed, short, control),
            f"target cell {index} seed/control differs",
        )
        _require(
            row.get("attempt_policy")
            == "append-only attempts; exactly one completed manifest permitted",
            f"target cell {index} attempt policy differs",
        )
    for c1_index, c3_index in PAIR_INDICES:
        c1, c3 = cells[c1_index], cells[c3_index]
        _require(
            (c1["training_seed"], c1["task"], c1["task_config"], c1["domain"])
            == (c3["training_seed"], c3["task"], c3["task_config"], c3["domain"]),
            f"paired cells {c1_index}/{c3_index} differ",
        )
    rollout = Path(plan["rollout_root"]).resolve()
    gates: list[dict[str, Any]] = []
    for seed, c1_index, c3_index in ((2, 12, 18), (3, 24, 30)):
        for short, index in (("c1", c1_index), ("c3", c3_index)):
            gates.append(
                {
                    "training_seed": seed,
                    "short_control": short,
                    "gate": _validate_gate(
                        rollout / f"deployment_gates/seed_{seed}_{short}.json",
                        cells[index]["checkpoint"]["path"],
                        f"seed-{seed}/{short.upper()} deployment gate",
                    ),
                }
            )
    return plan, plan_identity, continuation, gates


def _expected_eval_argv(
    plan: Mapping[str, Any], cell: Mapping[str, Any], worker_pid: int, output: Path
) -> list[str]:
    mutable = dict(cell)
    mutable["_stock_protocol_amendment_path"] = plan["stock_protocol_amendment"]["path"]
    return common._expected_eval_argv(mutable, worker_pid=worker_pid, output_dir=output)


def _match_blocking_evaluator(
    plan: Mapping[str, Any], argv: Sequence[str], worker_pid: int
) -> tuple[int, Path]:
    cells = _cells(plan)
    output_args = [arg for arg in argv if arg.startswith("EVALUATION.output_dir=")]
    _require(len(output_args) == 1, f"runner child {worker_pid} output argument differs")
    output = Path(output_args[0].split("=", 1)[1]).resolve()
    matches = [
        index
        for index in BLOCKING_CELL_INDICES
        if output.parent == Path(cells[index]["cell_root"]).resolve()
    ]
    _require(len(matches) == 1, f"runner child {worker_pid} is outside seed-1/C3 Open")
    index = matches[0]
    _require(
        re.fullmatch(rf"attempt_\d{{8}}T\d{{6}}Z_pid{worker_pid}", output.name) is not None,
        f"runner child {worker_pid} attempt name differs",
    )
    _require(
        list(argv) == _expected_eval_argv(plan, cells[index], worker_pid, output),
        f"live evaluator argv differs for cell {index}",
    )
    return index, output


def _active_target_evaluators(plan: Mapping[str, Any], proc_root: Path) -> list[int]:
    cells = _cells(plan)
    checkpoints = {
        f"ckpt={Path(cells[index]['checkpoint']['path']).resolve()}"
        for index in TARGET_CELL_INDICES
    }
    active: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise Seed23PackedError(f"cannot enumerate {proc_root}: {exc}") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            argv = [
                os.fsdecode(part)
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (
            len(argv) >= 4
            and argv[1:3] == ["-m", common.EVAL_MODULE]
            and any(checkpoint in argv for checkpoint in checkpoints)
        ):
            active.append(int(entry.name))
    return sorted(active)


def _all_policy_evaluator_pids(proc_root: Path) -> list[int]:
    pids: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise Seed23PackedError(f"cannot enumerate {proc_root}: {exc}") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            argv = [
                os.fsdecode(part)
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if len(argv) >= 3 and argv[1:3] == ["-m", common.EVAL_MODULE]:
            pids.append(int(entry.name))
    return sorted(pids)


def capture_live_window(
    plan: Mapping[str, Any], *, parent_state: str, proc_root: str | Path = "/proc"
) -> dict[str, Any]:
    root = Path(proc_root)
    try:
        parent = common._read_process(common.MAIN_RUNNER_PID, root)
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc
    _require(parent["state"] == parent_state, f"main runner state is {parent['state']}, not {parent_state}")
    _require(parent["argv"] == list(common.MAIN_RUNNER_ARGV), "main runner argv differs")
    _require(len(parent["children"]) == 2, "main runner must have both seed-1/C3 Open children")
    workers: list[dict[str, Any]] = []
    seen: set[int] = set()
    for worker_pid in parent["children"]:
        try:
            worker = common._read_process(worker_pid, root)
        except common.C3OverlapError as exc:
            raise Seed23PackedError(str(exc)) from exc
        _require(worker["ppid"] == common.MAIN_RUNNER_PID, "runner child PPID differs")
        _require(worker["argv"] == list(common.MAIN_RUNNER_ARGV), "runner child argv differs")
        _require(len(worker["children"]) == 1, "runner child must own exactly one evaluator")
        try:
            evaluator = common._read_process(worker["children"][0], root)
        except common.C3OverlapError as exc:
            raise Seed23PackedError(str(exc)) from exc
        _require(evaluator["ppid"] == worker_pid, "evaluator is not a direct runner-child process")
        cell_index, output = _match_blocking_evaluator(plan, evaluator["argv"], worker_pid)
        _require(cell_index not in seen, "duplicate seed-1/C3 Open child")
        seen.add(cell_index)
        workers.append(
            {
                "cell_index": cell_index,
                "physical_gpu_index": _cells(plan)[cell_index]["physical_gpu_index"],
                "worker": worker,
                "evaluator": evaluator,
                "output_dir": str(output),
            }
        )
    _require(seen == set(BLOCKING_CELL_INDICES), "live children are not cells 8 and 9")
    workers.sort(key=lambda row: int(row["cell_index"]))
    expected_evaluators = sorted(int(row["evaluator"]["pid"]) for row in workers)
    _require(
        _all_policy_evaluator_pids(root) == expected_evaluators,
        "unexpected policy evaluator process exists outside seed-1/C3 Open",
    )
    active = _active_target_evaluators(plan, root)
    _require(not active, f"seed-2/3 evaluator is already active: {active}")
    cells = _cells(plan)
    for index in TARGET_CELL_INDICES:
        target = Path(cells[index]["cell_root"]).resolve()
        _require(not target.exists(), f"target cell root already exists: cell {index}: {target}")
    try:
        boot_id = common._read_boot_id(root)
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc
    return {
        "boot_id": boot_id,
        "parent": parent,
        "direct_seed1_c3_open_workers": workers,
        "active_seed23_evaluator_pids": [],
        "target_cell_roots_absent": True,
    }


def _process_identity(process: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: process[field]
        for field in ("pid", "ppid", "start_time_ticks", "argv", "children")
    }


def _window_identity(window: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "boot_id": window["boot_id"],
        "parent": _process_identity(window["parent"]),
        "direct_seed1_c3_open_workers": [
            {
                "cell_index": row["cell_index"],
                "physical_gpu_index": row["physical_gpu_index"],
                "worker": _process_identity(row["worker"]),
                "evaluator": _process_identity(row["evaluator"]),
                "output_dir": row["output_dir"],
            }
            for row in window["direct_seed1_c3_open_workers"]
        ],
    }


def validate_gpu_reports(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    _require(len(paths) == 8, "exactly eight GPU preflight reports are required")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        payload, resolved = _load_json(path, "packed GPU preflight")
        gpu = payload.get("physical_gpu_index")
        free = payload.get("memory_free_mib_at_preflight")
        _require(isinstance(gpu, int) and gpu in ALL_GPU_IDS, "GPU preflight index differs")
        _require(gpu not in seen, "duplicate GPU preflight report")
        required = MIN_FREE_GPU_MIB[gpu]
        _require(isinstance(free, int) and free >= required, f"GPU {gpu} free memory is below {required} MiB")
        seen.add(gpu)
        rows.append(
            {
                "physical_gpu_index": gpu,
                "memory_free_mib_at_preflight": free,
                "minimum_free_mib": required,
                "report": _identity(resolved),
            }
        )
    _require(seen == set(ALL_GPU_IDS), "GPU preflight set differs")
    rows.sort(key=lambda row: int(row["physical_gpu_index"]))
    return rows


def write_preflight(
    *, plan_path: str | Path, continuation_path: str | Path, output: str | Path
) -> dict[str, Any]:
    plan, plan_identity, continuation, gates = validate_inputs(plan_path, continuation_path)
    window = capture_live_window(plan, parent_state="S")
    core = {
        "stock_rollout_plan": plan_identity,
        "asset_repair_continuation": continuation["continuation"],
        "deployment_gates": gates,
        "main_runner_identity": _window_identity(window),
        "target_cell_indices": list(TARGET_CELL_INDICES),
    }
    payload = {
        "kind": PREFLIGHT_KIND,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PASS",
        **core,
        "preflight_payload_sha256": common._canonical_sha256(core),
    }
    _write_new_json(output, payload)
    return payload


def _target_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = _cells(plan)
    return [
        {
            "cell_index": index,
            "training_seed": cells[index]["training_seed"],
            "control": cells[index]["control"],
            "short_control": cells[index]["short_control"],
            "planned_physical_gpu_index": cells[index]["physical_gpu_index"],
            "task": cells[index]["task"],
            "task_config": cells[index]["task_config"],
            "domain": cells[index]["domain"],
            "cell_root": str(Path(cells[index]["cell_root"]).resolve()),
            "checkpoint": cells[index]["checkpoint"],
            "dataset_stats": cells[index]["dataset_stats"],
            "attempt_policy": cells[index]["attempt_policy"],
        }
        for index in TARGET_CELL_INDICES
    ]


def _schedule_core(
    *,
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
    continuation_identity: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    runner_identity: Mapping[str, Any],
    gpu_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_by_index = {row["cell_index"]: row for row in _target_rows(plan)}
    pairs_by_index = [
        {
            "pair_index": pair_index,
            "training_seed": target_by_index[c1]["training_seed"],
            "task": target_by_index[c1]["task"],
            "task_config": target_by_index[c1]["task_config"],
            "domain": target_by_index[c1]["domain"],
            "cell_indices": [c1, c3],
            "launch_policy": "launch C1/C3 together whenever two helper slots are available",
        }
        for pair_index, (c1, c3) in enumerate(PAIR_INDICES)
    ]
    pairs = [pairs_by_index[index] for index in PAIR_QUEUE_ORDER]
    reservations = [
        {
            "physical_gpu_index": row["physical_gpu_index"],
            "external_live_eval_processes": 1,
            "helper_eval_process_cap": 1,
            "total_eval_process_cap": 2,
            "blocking_cell_index": row["cell_index"],
        }
        for row in runner_identity["direct_seed1_c3_open_workers"]
    ]
    return {
        "evaluation_profile": PROFILE,
        "helper_sources": {
            "python": _identity(Path(__file__)),
            "shell": _identity(Path(__file__).with_name("run_release_seed23_packed.sh")),
            "shared_validation": _identity(common.__file__),
        },
        "stock_rollout_plan": dict(plan_identity),
        "asset_repair_continuation": dict(continuation_identity),
        "stock_protocol_amendment": plan["stock_protocol_amendment"],
        "deployment_gates": [dict(row) for row in gates],
        "main_runner_identity": dict(runner_identity),
        "gpu_preflight_reports": [dict(row) for row in gpu_reports],
        "external_gpu_reservations": reservations,
        "gpu_capacity_policy": {
            "maximum_total_eval_processes_per_gpu": 2,
            "helper_capacity_by_gpu": {str(gpu): (1 if gpu in (2, 3, 4) else 2) for gpu in ALL_GPU_IDS},
            "fixed_conservative_reservations_until_helper_exit": True,
            "gpu_3_helper_cap_is_one_because_sam3_is_present": True,
        },
        "advanced_cells": list(target_by_index.values()),
        "paired_queue": pairs,
        "execution_change": {
            "execution_order_and_physical_gpu_packing_only": True,
            "parent_suspend_signal_scope": "positive parent PID only; no process-group STOP",
            "seed1_c3_open_children_continue": True,
            "resume_parent_after_owned_workers_are_terminated_and_reaped": True,
            "partial_failure_recovery": "terminate helper workers; keep append-only attempts; resume stock runner",
        },
        "immutable_protocol": {
            "plan_payload_changed": False,
            "checkpoint_bytes_changed": False,
            "dataset_stats_changed": False,
            "stock_evaluation_settings_changed": False,
            "cell_output_roots_changed": False,
            "episode_pairing": "not_claimed",
            "append_only_attempts": True,
            "audit_each_completed_cell": True,
        },
    }


def write_schedule_after_stop(
    *,
    preflight_path: str | Path,
    plan_path: str | Path,
    continuation_path: str | Path,
    gpu_preflight_reports: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    preflight, _ = _load_json(preflight_path, "packed live preflight")
    _require(preflight.get("kind") == PREFLIGHT_KIND, "preflight kind differs")
    core = {
        key: value
        for key, value in preflight.items()
        if key not in {"kind", "schema_version", "status", "preflight_payload_sha256"}
    }
    _require(preflight.get("schema_version") == PREFLIGHT_SCHEMA_VERSION, "preflight schema differs")
    _require(preflight.get("status") == "PASS", "preflight is not PASS")
    _require(preflight.get("preflight_payload_sha256") == common._canonical_sha256(core), "preflight SHA differs")
    plan, plan_identity, continuation, gates = validate_inputs(plan_path, continuation_path)
    _require(preflight.get("stock_rollout_plan") == plan_identity, "preflight plan differs")
    _require(preflight.get("asset_repair_continuation") == continuation["continuation"], "preflight continuation differs")
    _require(preflight.get("deployment_gates") == gates, "preflight deployment gates differ")
    stopped = capture_live_window(plan, parent_state="T")
    stopped_identity = _window_identity(stopped)
    _require(stopped_identity == preflight.get("main_runner_identity"), "runner identity changed across STOP")
    gpu_reports = validate_gpu_reports(gpu_preflight_reports)
    schedule_core = _schedule_core(
        plan=plan,
        plan_identity=plan_identity,
        continuation_identity=continuation["continuation"],
        gates=gates,
        runner_identity=stopped_identity,
        gpu_reports=gpu_reports,
    )
    payload = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **schedule_core,
    }
    payload["scheduling_amendment_id"] = "seed23-packed-v1:" + common._canonical_sha256(schedule_core)
    _write_new_json(output, payload)
    return payload


def validate_schedule(
    path: str | Path,
    *,
    expected_plan: str | Path | None = None,
    expected_continuation: str | Path | None = None,
) -> dict[str, Any]:
    payload, resolved = _load_json(path, "packed scheduling amendment")
    _require(payload.get("kind") == KIND, "schedule kind differs")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schedule schema differs")
    _require(payload.get("status") == "PASS", "schedule is not PASS")
    plan_identity = _verify_identity(payload.get("stock_rollout_plan"), "schedule plan")
    continuation_identity = _verify_identity(payload.get("asset_repair_continuation"), "schedule continuation")
    if expected_plan is not None:
        _require(plan_identity["path"] == str(Path(expected_plan).resolve()), "runtime plan differs")
    if expected_continuation is not None:
        _require(continuation_identity["path"] == str(Path(expected_continuation).resolve()), "runtime continuation differs")
    plan, rebuilt_plan, continuation, gates = validate_inputs(
        plan_identity["path"], continuation_identity["path"]
    )
    raw_reports = payload.get("gpu_preflight_reports")
    _require(isinstance(raw_reports, list), "schedule GPU reports are missing")
    paths = []
    for row in raw_reports:
        _require(isinstance(row, Mapping) and isinstance(row.get("report"), Mapping), "GPU report row differs")
        paths.append(str(row["report"].get("path", "")))
    gpu_reports = validate_gpu_reports(paths)
    schedule_core = _schedule_core(
        plan=plan,
        plan_identity=rebuilt_plan,
        continuation_identity=continuation["continuation"],
        gates=gates,
        runner_identity=payload.get("main_runner_identity", {}),
        gpu_reports=gpu_reports,
    )
    for field, expected in schedule_core.items():
        _require(payload.get(field) == expected, f"schedule field differs: {field}")
    expected_id = "seed23-packed-v1:" + common._canonical_sha256(schedule_core)
    _require(payload.get("scheduling_amendment_id") == expected_id, "schedule id differs")
    return {
        "status": "PASS",
        "schedule": _identity(resolved),
        "scheduling_amendment_id": expected_id,
        "payload": payload,
    }


def validate_parent_identity(path: str | Path, *, require_stopped: bool) -> dict[str, Any]:
    payload, _ = _load_json(path, "packed process binding")
    if payload.get("kind") == KIND:
        expected = validate_schedule(path)["payload"]["main_runner_identity"]
    else:
        _require(payload.get("kind") == PREFLIGHT_KIND, "process binding kind differs")
        core = {
            key: value
            for key, value in payload.items()
            if key not in {"kind", "schema_version", "status", "preflight_payload_sha256"}
        }
        _require(payload.get("status") == "PASS", "process binding is not PASS")
        _require(payload.get("preflight_payload_sha256") == common._canonical_sha256(core), "process binding SHA differs")
        expected = payload.get("main_runner_identity")
        _require(isinstance(expected, Mapping), "runner identity is missing")
    try:
        actual = common._read_process(common.MAIN_RUNNER_PID, Path("/proc"))
        boot_id = common._read_boot_id(Path("/proc"))
    except common.C3OverlapError as exc:
        raise Seed23PackedError(str(exc)) from exc
    if require_stopped:
        _require(actual["state"] == "T", "main runner is not stopped")
    _require(boot_id == expected["boot_id"], "boot id changed")
    parent = expected["parent"]
    for field in ("pid", "ppid", "start_time_ticks", "argv"):
        _require(actual[field] == parent[field], f"main runner {field} changed")
    return {"status": "PASS", "parent_pid": common.MAIN_RUNNER_PID, "state": actual["state"]}


def emit_pairs(path: str | Path) -> None:
    payload = validate_schedule(path)["payload"]
    rows = {row["cell_index"]: row for row in payload["advanced_cells"]}
    amendment = payload["stock_protocol_amendment"]["path"]
    for pair_index in PAIR_QUEUE_ORDER:
        pair = PAIR_INDICES[pair_index]
        for index in pair:
            row = rows[index]
            values = [
                pair_index,
                index,
                row["planned_physical_gpu_index"],
                row["checkpoint"]["path"],
                row["dataset_stats"]["path"],
                row["task"],
                row["task_config"],
                row["domain"],
                row["cell_root"],
                amendment,
            ]
            text = "\t".join(str(value) for value in values)
            _require("\n" not in text and "\r" not in text, "pair shell record contains newline")
            print(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--plan", required=True)
    preflight.add_argument("--continuation", required=True)
    preflight.add_argument("--output", required=True)
    materialize = commands.add_parser("materialize-after-stop")
    materialize.add_argument("--preflight", required=True)
    materialize.add_argument("--plan", required=True)
    materialize.add_argument("--continuation", required=True)
    materialize.add_argument("--gpu-preflight-report", action="append", required=True)
    materialize.add_argument("--output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--path", required=True)
    validate.add_argument("--plan")
    validate.add_argument("--continuation")
    parent = commands.add_parser("validate-parent")
    parent.add_argument("--path", required=True)
    parent.add_argument("--require-stopped", action="store_true")
    emit = commands.add_parser("emit-pairs")
    emit.add_argument("--path", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preflight":
        result = write_preflight(plan_path=args.plan, continuation_path=args.continuation, output=args.output)
    elif args.command == "materialize-after-stop":
        result = write_schedule_after_stop(
            preflight_path=args.preflight,
            plan_path=args.plan,
            continuation_path=args.continuation,
            gpu_preflight_reports=args.gpu_preflight_report,
            output=args.output,
        )
    elif args.command == "validate":
        result = validate_schedule(args.path, expected_plan=args.plan, expected_continuation=args.continuation)
    elif args.command == "validate-parent":
        result = validate_parent_identity(args.path, require_stopped=args.require_stopped)
    else:
        emit_pairs(args.path)
        return
    printable = {key: value for key, value in result.items() if key != "payload"}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ALL_GPU_IDS",
    "BLOCKING_CELL_INDICES",
    "MIN_FREE_GPU_MIB",
    "PAIR_INDICES",
    "Seed23PackedError",
    "TARGET_CELL_INDICES",
    "capture_live_window",
    "validate_gpu_reports",
    "validate_schedule",
]
