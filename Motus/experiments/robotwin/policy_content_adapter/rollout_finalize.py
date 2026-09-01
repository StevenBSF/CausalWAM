"""Finalize and re-audit a completed 100-episode RoboTwin cell."""

from pathlib import Path
from .evaluation import CELL_SCHEMA, validate_cell
from .rollout_common import identity, need, read_json, verify_identity, write_json
from .rollout_plan import validate_plan


def finalize_cell(plan_path, result_path, log_path, output_path):
    plan, plan_file = read_json(plan_path)
    validate_plan(plan)
    result = Path(result_path).resolve()
    log = Path(log_path).resolve()
    need(result.is_file() and log.is_file(), "rollout result/log missing")
    values = [line.strip() for line in result.read_text().splitlines() if line.strip()]
    need(bool(values), "rollout result is empty")
    try:
        rate = float(values[-1])
    except ValueError as exc:
        raise ValueError("rollout result is not numeric") from exc
    successes = round(rate * 100)
    need(
        0 <= successes <= 100 and abs(rate - successes / 100) < 1e-9,
        "result is not a 100-episode rate",
    )
    cell = {
        "schema": CELL_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "control": plan["control"],
        "training_seed": plan["training_seed"],
        "task": plan["task"],
        "domain": plan["domain"],
        "episode_count": 100,
        "success_count": successes,
        "success_rate": successes / 100,
        "checkpoint_sha256": plan["deployment"]["checkpoint"]["sha256"],
        "evaluation_settings_sha256": plan["evaluation_settings_sha256"],
        "episode_pairing": "shared_start_seed_not_exact_pairing",
        "simulator_seed": plan["simulator_seed"],
        "cell_plan": identity(plan_file),
        "result_file": identity(result),
        "worker_log": identity(log),
    }
    validate_cell(cell)
    write_json(output_path, cell)
    return cell


def audit_cell(path):
    cell, _ = read_json(path)
    validate_cell(cell)
    for name in ("cell_plan", "result_file", "worker_log"):
        verify_identity(cell[name])
    plan, _ = read_json(cell["cell_plan"]["path"])
    validate_plan(plan)
    need(
        cell["checkpoint_sha256"] == plan["deployment"]["checkpoint"]["sha256"],
        "checkpoint ancestry changed",
    )
    need(
        cell["evaluation_settings_sha256"] == plan["evaluation_settings_sha256"],
        "settings ancestry changed",
    )
    return cell
