"""Bind one Motus deployment checkpoint to one RoboTwin evaluation cell."""

from pathlib import Path
import torch
from .evaluation import DOMAINS
from .protocol import CONTROLS, TASKS
from .rollout_common import identity, need, read_json
from .rollout_settings import validate_settings

SCHEMA = "motus_policy_content_adapter_rollout_cell_plan"


def validate_deployment(checkpoint_path, summary_path, control, training_seed):
    need(control in CONTROLS, "invalid control")
    summary, summary_file = read_json(summary_path)
    checkpoint = Path(checkpoint_path).resolve()
    need(summary.get("status") == "COMPLETE", "training is incomplete")
    need(
        summary.get("control") == control
        and summary.get("training_seed") == training_seed,
        "training identity differs",
    )
    claimed = summary.get("deployment_checkpoint", {})
    need(
        Path(claimed.get("path", "")).resolve() == checkpoint, "deployment path differs"
    )
    checkpoint_id = identity(checkpoint)
    need(
        checkpoint_id["size_bytes"] == claimed.get("size_bytes")
        and checkpoint_id["sha256"] == claimed.get("sha256"),
        "checkpoint changed",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    need(
        payload.get("schema") == "motus_policy_content_adapter_deployment_checkpoint",
        "deployment schema changed",
    )
    need(
        payload.get("control") == control
        and payload.get("training_seed") == training_seed,
        "deployment identity differs",
    )
    return {
        "checkpoint": checkpoint_id,
        "training_summary": identity(summary_file),
        "regime": payload["regime"],
        "optimizer_steps": payload["optimizer_steps"],
    }


def build_plan(
    settings_path,
    checkpoint_path,
    summary_path,
    control,
    training_seed,
    task,
    domain,
    output_dir,
):
    need(task in TASKS and domain in DOMAINS, "invalid task/domain")
    settings, settings_file = read_json(settings_path)
    validate_settings(settings)
    deployment = validate_deployment(
        checkpoint_path, summary_path, control, training_seed
    )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "control": control,
        "training_seed": int(training_seed),
        "task": task,
        "domain": domain,
        "episode_count": 100,
        "simulator_seed": settings["contract"]["simulator_seed"],
        "task_config": settings["contract"]["task_configs"][domain],
        "episode_pairing": settings["contract"]["episode_pairing"],
        "settings": identity(settings_file),
        "evaluation_settings_sha256": settings["contract_sha256"],
        "deployment": deployment,
        "base_checkpoint": settings["base_checkpoint"],
        "output_dir": str(Path(output_dir).resolve()),
    }


def validate_plan(value):
    need(
        value.get("schema") == SCHEMA and value.get("status") == "PASS", "invalid plan"
    )
    need(
        value.get("episode_count") == 100
        and value.get("task") in TASKS
        and value.get("domain") in DOMAINS,
        "cell contract changed",
    )
    need(
        value.get("control") in CONTROLS and int(value.get("training_seed", -1)) >= 0,
        "cell identity changed",
    )
    return dict(value)
