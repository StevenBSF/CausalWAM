import json
from pathlib import Path
import torch
import pytest
from experiments.robotwin.policy_content_adapter.paired_data import (
    canonical_json_sha256,
    sha256_file,
)
from experiments.robotwin.policy_content_adapter.rollout_common import (
    RolloutError,
    write_json,
)
from experiments.robotwin.policy_content_adapter.rollout_finalize import (
    audit_cell,
    finalize_cell,
)
from experiments.robotwin.policy_content_adapter.rollout_plan import build_plan
from experiments.robotwin.policy_content_adapter.rollout_settings import SCHEMA


def _settings(path):
    contract = {
        "simulator_seed": 42,
        "episodes_per_cell": 100,
        "instruction_type": "unseen",
        "task_configs": {"clean": "demo_clean", "official_random": "demo_randomized"},
        "inference_steps": 10,
        "episode_selection": "author_stock_expert_filter",
        "episode_pairing": "shared_start_seed_not_exact_pairing",
        "tasks": ["place_a2b_left", "open_microwave", "move_stapler_pad"],
        "domains": ["clean", "official_random"],
    }
    value = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
        "source_files": [],
        "source_inventory_sha256": canonical_json_sha256([]),
        "base_checkpoint": {"path": "base", "size_bytes": 1, "sha256": "b" * 64},
    }
    write_json(path, value)


def test_cell_plan_finalize_and_tamper_detection(tmp_path: Path):
    settings = tmp_path / "settings.json"
    _settings(settings)
    checkpoint = tmp_path / "deployment.pt"
    torch.save(
        {
            "schema": "motus_policy_content_adapter_deployment_checkpoint",
            "control": "m3_ours",
            "training_seed": 1,
            "regime": "m_p1",
            "optimizer_steps": 5,
        },
        checkpoint,
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "control": "m3_ours",
                "training_seed": 1,
                "deployment_checkpoint": {
                    "path": str(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": sha256_file(checkpoint),
                },
            }
        )
    )
    plan = build_plan(
        settings,
        checkpoint,
        summary,
        "m3_ours",
        1,
        "place_a2b_left",
        "clean",
        tmp_path / "cell",
    )
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    result = tmp_path / "_result_clean.txt"
    result.write_text("0.73\n")
    log = tmp_path / "worker.log"
    log.write_text("done\n")
    receipt = tmp_path / "completed.json"
    cell = finalize_cell(plan_path, result, log, receipt)
    assert cell["success_count"] == 73 and audit_cell(receipt)["success_rate"] == 0.73
    result.write_text("0.74\n")
    with pytest.raises(RolloutError, match="identity changed"):
        audit_cell(receipt)
