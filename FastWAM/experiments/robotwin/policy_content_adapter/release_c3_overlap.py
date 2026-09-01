#!/usr/bin/env python3
"""Fail-closed, one-shot scheduling amendment for the seed-1 C3 overlap.

This module does not send signals or start GPU work.  It proves the exact
live-process window used by ``run_release_c3_overlap.sh``, writes the immutable
scheduling sidecar with create-only semantics after the parent is stopped, and
revalidates the experiment-owned inputs used by the shell wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asset_repair_selection_confirmation import validate_formal_continuation
from .release_formal_stock_rollout import validate_stock_rollout_plan
from .release_stock_eval_protocol import PROFILE
from .runtime_utils import PROJECT_ROOT


KIND = "policy_release_author_stock_seed1_c3_overlap_schedule"
SCHEMA_VERSION = 1
PREFLIGHT_KIND = "policy_release_seed1_c3_overlap_live_preflight"
PREFLIGHT_SCHEMA_VERSION = 1
MAIN_RUNNER_PID = 3_759_159
MAIN_RUNNER_ARGV = (
    "bash",
    "experiments/robotwin/policy_content_adapter/run_release_formal_stock_rollout.sh",
)
MAIN_RUNNER_SCRIPT = (PROJECT_ROOT / MAIN_RUNNER_ARGV[1]).resolve()
# Keep the literal argv spelling used by the already-running stock shell.  Do
# not resolve this symlink: /proc/<pid>/cmdline preserves ``.../bin/python``.
DEFAULT_PYTHON_BIN = "/root/anaconda3/envs/fastwam-robotwin-bw/bin/python"
DEFAULT_ROBOTWIN_ROOT = (PROJECT_ROOT / "third_party/RoboTwin").resolve()
TARGET_CELL_INDICES = (6, 7, 10, 11)
BLOCKING_CELL_INDICES = (2, 3)
TARGET_GPU_IDS = (0, 1, 5, 6)
MIN_FREE_GPU_MIB = 60_000
EVAL_MODULE = "experiments.robotwin.policy_content_adapter.eval_robotwin_single"


class C3OverlapError(RuntimeError):
    """The one-shot C3 overlap cannot be proven safe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C3OverlapError(message)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
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


def _verify_file_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} identity is missing")
    actual = _file_identity(str(value.get("path", "")))
    for field in ("kind", "path", "size_bytes", "sha256"):
        _require(actual[field] == value.get(field), f"{label} {field} differs")
    return actual


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise C3OverlapError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} root must be an object")
    return payload, resolved


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o644)
    except FileExistsError as exc:
        raise C3OverlapError(f"refusing to overwrite output: {destination}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def _cells_by_index(plan: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for wave in plan.get("waves", []):
        for raw in wave.get("parallel_task_domain_cells", []):
            _require(isinstance(raw, Mapping), "stock plan cell is invalid")
            cell = dict(raw)
            index = cell.get("cell_index")
            _require(isinstance(index, int) and index not in rows, "stock cell index differs")
            rows[index] = cell
    _require(set(rows) == set(range(36)), "stock rollout plan does not contain cells 0..35")
    return rows


def _validate_experiment_inputs(
    plan_path: str | Path, continuation_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = validate_stock_rollout_plan(plan_path, require_output_absent=False)
    plan = report["payload"]
    _require(plan.get("evaluation_profile") == PROFILE, "author-stock profile differs")
    continuation = validate_formal_continuation(
        continuation_path,
        expected_plan=plan_path,
        expected_amendment=plan["stock_protocol_amendment"]["path"],
        expected_rollout_root=plan["rollout_root"],
    )
    cells = _cells_by_index(plan)
    target_expected = (
        (6, 0, "place_a2b_left", "demo_clean", "clean"),
        (7, 1, "place_a2b_left", "demo_randomized", "official_random"),
        (10, 5, "move_stapler_pad", "demo_clean", "clean"),
        (11, 6, "move_stapler_pad", "demo_randomized", "official_random"),
    )
    for index, gpu, task, task_config, domain in target_expected:
        row = cells[index]
        _require(
            (
                row.get("training_seed"),
                row.get("short_control"),
                row.get("control"),
                row.get("physical_gpu_index"),
                row.get("task"),
                row.get("task_config"),
                row.get("domain"),
            )
            == (1, "c3", "c3_ours", gpu, task, task_config, domain),
            f"target stock cell {index} differs",
        )
        _require(
            row.get("attempt_policy")
            == "append-only attempts; exactly one completed manifest permitted",
            f"target stock cell {index} attempt policy differs",
        )
    for index, gpu, config, domain in (
        (2, 2, "demo_clean", "clean"),
        (3, 4, "demo_randomized", "official_random"),
    ):
        row = cells[index]
        _require(
            (
                row.get("training_seed"),
                row.get("short_control"),
                row.get("control"),
                row.get("physical_gpu_index"),
                row.get("task"),
                row.get("task_config"),
                row.get("domain"),
            )
            == (1, "c1", "c1_architecture_only", gpu, "open_microwave", config, domain),
            f"blocking stock cell {index} differs",
        )
    gate_path = (
        Path(plan["rollout_root"]).resolve()
        / "deployment_gates/seed_1_c3.json"
    )
    gate, _ = _load_json(gate_path, "seed-1/C3 deployment gate")
    expected_checkpoint = str(Path(cells[6]["checkpoint"]["path"]).resolve())
    _require(gate.get("status") == "PASS", "seed-1/C3 deployment gate is not PASS")
    _require(
        str(Path(str(gate.get("checkpoint", ""))).resolve()) == expected_checkpoint,
        "seed-1/C3 deployment gate checkpoint differs",
    )
    gate_tasks = gate.get("tasks")
    _require(isinstance(gate_tasks, list) and len(gate_tasks) == 3, "deployment gate task count differs")
    _require(
        {row.get("task") for row in gate_tasks if isinstance(row, Mapping)}
        == {"place_a2b_left", "open_microwave", "move_stapler_pad"},
        "deployment gate tasks differ",
    )
    _require(
        all(
            isinstance(row, Mapping)
            and row.get("action_finite") is True
            and row.get("action_shape") == [14]
            for row in gate_tasks
        ),
        "deployment gate action contract differs",
    )
    plan["_overlap_deployment_gate"] = _file_identity(gate_path)
    return plan, report["plan"], continuation


def _read_boot_id(proc_root: Path) -> str:
    path = proc_root / "sys/kernel/random/boot_id"
    value = path.read_text(encoding="ascii").strip()
    _require(re.fullmatch(r"[0-9a-fA-F-]{36}", value) is not None, "invalid boot id")
    return value.lower()


def _read_process(pid: int, proc_root: Path) -> dict[str, Any]:
    _require(isinstance(pid, int) and pid > 1, "process PID must be a positive non-system PID")
    base = proc_root / str(pid)
    try:
        stat_text = (base / "stat").read_text(encoding="utf-8")
        close = stat_text.rfind(")")
        _require(close > 0, f"invalid /proc stat for PID {pid}")
        fields = stat_text[close + 2 :].split()
        _require(len(fields) >= 20, f"short /proc stat for PID {pid}")
        argv_raw = (base / "cmdline").read_bytes()
        argv = [os.fsdecode(part) for part in argv_raw.split(b"\0") if part]
        children_text = (base / "task" / str(pid) / "children").read_text(
            encoding="ascii"
        )
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise C3OverlapError(f"required PID {pid} disappeared") from exc
    except PermissionError as exc:
        raise C3OverlapError(f"cannot inspect required PID {pid}") from exc
    children: list[int] = []
    for value in children_text.split():
        _require(value.isdigit() and int(value) > 1, f"invalid child PID under {pid}")
        children.append(int(value))
    _require(len(children) == len(set(children)), f"duplicate child PID under {pid}")
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "state": fields[0],
        "start_time_ticks": int(fields[19]),
        "argv": argv,
        "children": sorted(children),
    }


def _expected_eval_argv(
    cell: Mapping[str, Any], *, worker_pid: int, output_dir: Path
) -> list[str]:
    return [
        DEFAULT_PYTHON_BIN,
        "-m",
        EVAL_MODULE,
        f"ckpt={Path(cell['checkpoint']['path']).resolve()}",
        f"gpu_id={cell['physical_gpu_index']}",
        "seed=42",
        "mixed_precision=bf16",
        f"EVALUATION.robotwin_root={DEFAULT_ROBOTWIN_ROOT}",
        f"EVALUATION.task_name={cell['task']}",
        f"EVALUATION.task_config={cell['task_config']}",
        "EVALUATION.eval_num_episodes=100",
        f"EVALUATION.output_dir={output_dir}",
        f"EVALUATION.dataset_stats_path={Path(cell['dataset_stats']['path']).resolve()}",
        "+EVALUATION.stock_protocol_amendment="
        f"{Path(cell['_stock_protocol_amendment_path']).resolve()}",
        "EVALUATION.instruction_type=unseen",
        "EVALUATION.action_horizon=null",
        "EVALUATION.replan_steps=24",
        "EVALUATION.num_inference_steps=10",
        "EVALUATION.sigma_shift=null",
        "EVALUATION.text_cfg_scale=1.0",
        "EVALUATION.rand_device=cpu",
        "EVALUATION.tiled=false",
        "EVALUATION.timing_enabled=false",
        "EVALUATION.skip_get_obs_within_replan=true",
    ]


def _eval_cell_from_argv(
    argv: Sequence[str], cells: Mapping[int, dict[str, Any]], worker_pid: int
) -> tuple[int, Path]:
    output_args = [arg for arg in argv if arg.startswith("EVALUATION.output_dir=")]
    _require(len(output_args) == 1, f"C1 worker {worker_pid} output argument differs")
    output = Path(output_args[0].split("=", 1)[1]).resolve()
    matches = [
        index
        for index in BLOCKING_CELL_INDICES
        if output.parent == Path(cells[index]["cell_root"]).resolve()
    ]
    _require(len(matches) == 1, f"C1 worker {worker_pid} output is outside blocking cells")
    index = matches[0]
    _require(
        re.fullmatch(rf"attempt_\d{{8}}T\d{{6}}Z_pid{worker_pid}", output.name) is not None,
        f"C1 worker {worker_pid} attempt name differs",
    )
    expected = _expected_eval_argv(cells[index], worker_pid=worker_pid, output_dir=output)
    _require(list(argv) == expected, f"C1 Open evaluator argv differs for cell {index}")
    return index, output


def _find_active_seed1_c3(proc_root: Path, c3_checkpoint: str) -> list[int]:
    active: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise C3OverlapError(f"cannot enumerate {proc_root}: {exc}") from exc
    expected_ckpt = f"ckpt={Path(c3_checkpoint).resolve()}"
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
        if len(argv) >= 4 and argv[1:3] == ["-m", EVAL_MODULE] and expected_ckpt in argv:
            active.append(int(entry.name))
    return sorted(active)


def _target_roots_absent(cells: Mapping[int, dict[str, Any]]) -> None:
    for index in TARGET_CELL_INDICES:
        root = Path(cells[index]["cell_root"]).resolve()
        _require(not root.exists(), f"target cell root already exists: cell {index}: {root}")


def _validated_gpu_reports(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    _require(len(paths) == 4, "exactly four GPU preflight reports are required")
    reports: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        payload, resolved = _load_json(path, "C3 overlap GPU preflight")
        gpu = payload.get("physical_gpu_index")
        free = payload.get("memory_free_mib_at_preflight")
        _require(isinstance(gpu, int) and gpu in TARGET_GPU_IDS, "GPU preflight index differs")
        _require(gpu not in seen, "duplicate GPU preflight report")
        _require(isinstance(free, int) and free >= MIN_FREE_GPU_MIB, f"GPU {gpu} has <60000 MiB free")
        seen.add(gpu)
        reports.append(
            {
                "physical_gpu_index": gpu,
                "memory_free_mib_at_preflight": free,
                "report": _file_identity(resolved),
            }
        )
    _require(seen == set(TARGET_GPU_IDS), "GPU preflight set differs")
    reports.sort(key=lambda row: TARGET_GPU_IDS.index(int(row["physical_gpu_index"])))
    return reports


def capture_live_window(
    plan: Mapping[str, Any], *, proc_root: str | Path = "/proc", parent_state: str
) -> dict[str, Any]:
    root = Path(proc_root)
    cells = _cells_by_index(plan)
    amendment_path = str(Path(plan["stock_protocol_amendment"]["path"]).resolve())
    for cell in cells.values():
        cell["_stock_protocol_amendment_path"] = amendment_path
    parent = _read_process(MAIN_RUNNER_PID, root)
    _require(parent["state"] == parent_state, f"main runner state is {parent['state']}, not {parent_state}")
    _require(parent["argv"] == list(MAIN_RUNNER_ARGV), "main runner argv differs")
    _require(len(parent["children"]) == 2, "main runner must have exactly two direct children")
    direct_workers: list[dict[str, Any]] = []
    seen_cells: set[int] = set()
    for worker_pid in parent["children"]:
        worker = _read_process(worker_pid, root)
        _require(worker["ppid"] == MAIN_RUNNER_PID, "runner child PPID differs")
        _require(worker["argv"] == list(MAIN_RUNNER_ARGV), "runner child argv differs")
        _require(len(worker["children"]) == 1, "C1 runner child must own exactly one evaluator")
        evaluator = _read_process(worker["children"][0], root)
        _require(evaluator["ppid"] == worker_pid, "C1 evaluator is not a direct worker child")
        cell_index, output = _eval_cell_from_argv(evaluator["argv"], cells, worker_pid)
        _require(cell_index not in seen_cells, "duplicate live C1 Open cell")
        seen_cells.add(cell_index)
        direct_workers.append(
            {
                "cell_index": cell_index,
                "worker": worker,
                "evaluator": evaluator,
                "output_dir": str(output),
            }
        )
    _require(seen_cells == set(BLOCKING_CELL_INDICES), "live children are not both C1 Open cells")
    direct_workers.sort(key=lambda row: int(row["cell_index"]))
    c3_checkpoint = cells[TARGET_CELL_INDICES[0]]["checkpoint"]["path"]
    active_c3 = _find_active_seed1_c3(root, c3_checkpoint)
    _require(not active_c3, f"seed-1 C3 evaluator is already active: {active_c3}")
    _target_roots_absent(cells)
    return {
        "boot_id": _read_boot_id(root),
        "parent": parent,
        "direct_c1_open_workers": direct_workers,
        "active_seed1_c3_evaluator_pids": [],
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
        "direct_c1_open_workers": [
            {
                "cell_index": row["cell_index"],
                "worker": _process_identity(row["worker"]),
                "evaluator": _process_identity(row["evaluator"]),
                "output_dir": row["output_dir"],
            }
            for row in window["direct_c1_open_workers"]
        ],
    }


def write_preflight(
    *, plan_path: str | Path, continuation_path: str | Path, output: str | Path
) -> dict[str, Any]:
    plan, plan_identity, continuation = _validate_experiment_inputs(
        plan_path, continuation_path
    )
    window = capture_live_window(plan, parent_state="S")
    core = {
        "stock_rollout_plan": plan_identity,
        "asset_repair_continuation": continuation["continuation"],
        "seed1_c3_deployment_gate": plan["_overlap_deployment_gate"],
        "main_runner_identity": _window_identity(window),
        "target_cell_indices": list(TARGET_CELL_INDICES),
    }
    payload = {
        "kind": PREFLIGHT_KIND,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PASS",
        **core,
        "preflight_payload_sha256": _canonical_sha256(core),
    }
    _write_new_json(output, payload)
    return payload


def _static_schedule_core(
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
    continuation_identity: Mapping[str, Any],
    runner_identity: Mapping[str, Any],
    gpu_preflight_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = _cells_by_index(plan)
    selected = []
    for index in TARGET_CELL_INDICES:
        cell = cells[index]
        selected.append(
            {
                "cell_index": index,
                "training_seed": cell["training_seed"],
                "control": cell["control"],
                "short_control": cell["short_control"],
                "physical_gpu_index": cell["physical_gpu_index"],
                "task": cell["task"],
                "task_config": cell["task_config"],
                "domain": cell["domain"],
                "cell_root": str(Path(cell["cell_root"]).resolve()),
                "checkpoint": cell["checkpoint"],
                "dataset_stats": cell["dataset_stats"],
                "attempt_policy": cell["attempt_policy"],
            }
        )
    return {
        "evaluation_profile": PROFILE,
        "overlap_helper_sources": {
            "python": _file_identity(Path(__file__)),
            "shell": _file_identity(Path(__file__).with_name("run_release_c3_overlap.sh")),
        },
        "stock_rollout_plan": dict(plan_identity),
        "asset_repair_continuation": dict(continuation_identity),
        "stock_protocol_amendment": plan["stock_protocol_amendment"],
        "seed1_c3_deployment_gate": plan["_overlap_deployment_gate"],
        "gpu_preflight_reports": [dict(row) for row in gpu_preflight_reports],
        "minimum_free_gpu_mib": MIN_FREE_GPU_MIB,
        "main_runner_identity": dict(runner_identity),
        "blocked_wave_cells": list(BLOCKING_CELL_INDICES),
        "advanced_cells": selected,
        "execution_change": {
            "execution_order_only": True,
            "advance_seed1_c3_place_and_move_while_seed1_c1_open_finishes": True,
            "parent_suspend_signal_scope": "positive parent PID only; no process-group STOP",
            "parent_children_continue_while_parent_is_stopped": True,
            "resume_parent_after_owned_workers_are_terminated_and_reaped": True,
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
    preflight, _ = _load_json(preflight_path, "C3 overlap preflight")
    _require(preflight.get("kind") == PREFLIGHT_KIND, "preflight kind differs")
    _require(preflight.get("schema_version") == PREFLIGHT_SCHEMA_VERSION, "preflight schema differs")
    unhashed = {
        key: value
        for key, value in preflight.items()
        if key not in {"kind", "schema_version", "status", "preflight_payload_sha256"}
    }
    _require(preflight.get("status") == "PASS", "preflight is not PASS")
    _require(
        preflight.get("preflight_payload_sha256") == _canonical_sha256(unhashed),
        "preflight payload SHA differs",
    )
    plan, plan_identity, continuation = _validate_experiment_inputs(
        plan_path, continuation_path
    )
    _require(preflight.get("stock_rollout_plan") == plan_identity, "preflight plan differs")
    _require(
        preflight.get("asset_repair_continuation") == continuation["continuation"],
        "preflight continuation differs",
    )
    _require(
        preflight.get("seed1_c3_deployment_gate") == plan["_overlap_deployment_gate"],
        "preflight deployment gate differs",
    )
    stopped = capture_live_window(plan, parent_state="T")
    stopped_identity = _window_identity(stopped)
    _require(
        stopped_identity == preflight.get("main_runner_identity"),
        "live runner identity changed between preflight and STOP",
    )
    gpu_reports = _validated_gpu_reports(gpu_preflight_reports)
    core = _static_schedule_core(
        plan, plan_identity, continuation["continuation"], stopped_identity, gpu_reports
    )
    payload = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **core,
    }
    payload["scheduling_amendment_id"] = "seed1-c3-overlap-v1:" + _canonical_sha256(core)
    _write_new_json(output, payload)
    return payload


def validate_schedule(
    path: str | Path,
    *,
    expected_plan: str | Path | None = None,
    expected_continuation: str | Path | None = None,
) -> dict[str, Any]:
    payload, resolved = _load_json(path, "C3 overlap scheduling amendment")
    _require(payload.get("kind") == KIND, "scheduling amendment kind differs")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "scheduling amendment schema differs")
    _require(payload.get("status") == "PASS", "scheduling amendment is not PASS")
    plan_identity = _verify_file_identity(payload.get("stock_rollout_plan"), "schedule plan")
    continuation_identity = _verify_file_identity(
        payload.get("asset_repair_continuation"), "schedule continuation"
    )
    if expected_plan is not None:
        _require(plan_identity["path"] == str(Path(expected_plan).resolve()), "runtime plan differs")
    if expected_continuation is not None:
        _require(
            continuation_identity["path"] == str(Path(expected_continuation).resolve()),
            "runtime continuation differs",
        )
    plan, rebuilt_plan_identity, continuation = _validate_experiment_inputs(
        plan_identity["path"], continuation_identity["path"]
    )
    raw_gpu_reports = payload.get("gpu_preflight_reports")
    _require(isinstance(raw_gpu_reports, list), "schedule GPU reports are missing")
    gpu_paths = []
    for row in raw_gpu_reports:
        _require(isinstance(row, Mapping), "schedule GPU report row is invalid")
        report = row.get("report")
        _require(isinstance(report, Mapping), "schedule GPU report identity is missing")
        gpu_paths.append(str(report.get("path", "")))
    gpu_reports = _validated_gpu_reports(gpu_paths)
    core = _static_schedule_core(
        plan,
        rebuilt_plan_identity,
        continuation["continuation"],
        payload.get("main_runner_identity", {}),
        gpu_reports,
    )
    for field, expected in core.items():
        _require(payload.get(field) == expected, f"scheduling amendment field differs: {field}")
    expected_id = "seed1-c3-overlap-v1:" + _canonical_sha256(core)
    _require(payload.get("scheduling_amendment_id") == expected_id, "schedule id differs")
    return {
        "status": "PASS",
        "scheduling_amendment": _file_identity(resolved),
        "scheduling_amendment_id": expected_id,
        "payload": payload,
    }


def validate_parent_identity(
    path: str | Path, *, require_stopped: bool
) -> dict[str, Any]:
    payload, _ = _load_json(path, "C3 overlap process binding")
    if payload.get("kind") == KIND:
        expected = validate_schedule(path)["payload"]["main_runner_identity"]
    else:
        _require(payload.get("kind") == PREFLIGHT_KIND, "process binding kind differs")
        _require(
            payload.get("schema_version") == PREFLIGHT_SCHEMA_VERSION,
            "process binding schema differs",
        )
        core = {
            key: value
            for key, value in payload.items()
            if key not in {"kind", "schema_version", "status", "preflight_payload_sha256"}
        }
        _require(payload.get("status") == "PASS", "process binding is not PASS")
        _require(
            payload.get("preflight_payload_sha256") == _canonical_sha256(core),
            "process binding payload SHA differs",
        )
        expected = payload.get("main_runner_identity")
        _require(isinstance(expected, Mapping), "preflight runner identity is missing")
    root = Path("/proc")
    actual = _read_process(MAIN_RUNNER_PID, root)
    if require_stopped:
        _require(actual["state"] == "T", "main runner is not stopped")
    _require(_read_boot_id(root) == expected["boot_id"], "boot id changed before CONT")
    expected_parent = expected["parent"]
    for field in ("pid", "ppid", "start_time_ticks", "argv"):
        _require(actual[field] == expected_parent[field], f"main runner {field} changed before CONT")
    return {"status": "PASS", "parent_pid": MAIN_RUNNER_PID, "state": actual["state"]}


def emit_cells(path: str | Path) -> None:
    report = validate_schedule(path)
    payload = report["payload"]
    rows = payload["advanced_cells"]
    stock_amendment_path = payload["stock_protocol_amendment"]["path"]
    for row in rows:
        values = [
            row["cell_index"],
            row["physical_gpu_index"],
            row["checkpoint"]["path"],
            row["dataset_stats"]["path"],
            row["task"],
            row["task_config"],
            row["domain"],
            row["cell_root"],
            stock_amendment_path,
        ]
        text = "\t".join(str(value) for value in values)
        _require("\n" not in text and "\r" not in text, "cell shell record contains newline")
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
    emit = commands.add_parser("emit-cells")
    emit.add_argument("--path", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preflight":
        result = write_preflight(
            plan_path=args.plan, continuation_path=args.continuation, output=args.output
        )
    elif args.command == "materialize-after-stop":
        result = write_schedule_after_stop(
            preflight_path=args.preflight,
            plan_path=args.plan,
            continuation_path=args.continuation,
            gpu_preflight_reports=args.gpu_preflight_report,
            output=args.output,
        )
    elif args.command == "validate":
        result = validate_schedule(
            args.path, expected_plan=args.plan, expected_continuation=args.continuation
        )
    elif args.command == "validate-parent":
        result = validate_parent_identity(args.path, require_stopped=args.require_stopped)
    else:
        emit_cells(args.path)
        return
    printable = {key: value for key, value in result.items() if key != "payload"}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BLOCKING_CELL_INDICES",
    "C3OverlapError",
    "KIND",
    "MAIN_RUNNER_PID",
    "TARGET_CELL_INDICES",
    "capture_live_window",
    "emit_cells",
    "validate_schedule",
    "write_preflight",
    "write_schedule_after_stop",
]
