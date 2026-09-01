from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.robotwin.policy_content_adapter.training_audit import (
    audit_training_pair,
)


def _run(root: Path, control: str) -> None:
    root.mkdir(parents=True)
    summary = {
        "status": "COMPLETE",
        "control": control,
        "regime": "m_p1",
        "training_seed": 1,
        "lambda_contrastive": 0.0 if control.startswith("m1") else 0.1,
        "temperature": 0.07,
        "optimizer_steps": 2,
        "micro_steps": 2,
        "world_size": 1,
        "global_batch": 1,
        "gradient_audit_backend": "deepspeed_zero_partition_pre_step_v1",
        "initial_hash_stage": "post_accelerator_prepare_pre_optimizer_step",
        "artifact_shas": {"base": "a" * 64},
        "initial_conditioner_sha256": "b" * 64,
        "final_conditioner_sha256": ("c" if control.startswith("m1") else "d") * 64,
        "initial_action_expert_sha256": "e" * 64,
        "final_action_expert_sha256": "e" * 64,
    }
    identity = {
        "artifact_shas": summary["artifact_shas"],
        "initial_conditioner_sha256": summary["initial_conditioner_sha256"],
        "initial_action_expert_sha256": summary["initial_action_expert_sha256"],
        "official_sequence_sha256": "f" * 64,
        "paired_sequence_sha256": "1" * 64,
    }
    (root / "training_summary.json").write_text(json.dumps(summary))
    (root / "run_identity.json").write_text(json.dumps(identity))
    rows = []
    for step in (1, 2):
        action = 0.2 + step
        total = action
        if not control.startswith("m1"):
            total = float(
                torch.tensor(action, dtype=torch.float32)
                + torch.tensor(1.0, dtype=torch.bfloat16) * 0.1
            )
        rows.append(
            {
                "optimizer_step": step,
                "micro_step": step,
                "rank": 0,
                "action_rng_seed": step,
                "loss_action": action,
                "loss_contrastive": 1.0,
                "loss_total": total,
                "official_task": ["place_a2b_left"],
                "official_domain": ["clean"],
                "official_episode_index": [11000],
                "official_condition_frame_index": [step],
                "paired_physical_state_ids": ["s0", "s1"],
                "paired_task_ids": ["place_a2b_left", "place_a2b_left"],
                "gradient_audit": {
                    "status": "PASS",
                    "adapter_grad_norm": 1.0,
                    "adapter_gate_grad": 0.1,
                    "content_head_grad_norm": 1.0,
                    "action_expert_grad_norm": 0.0,
                    "backend": "deepspeed_zero_partition_pre_step_v1",
                },
            }
        )
    (root / "train_rank0.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_training_pair_audit_compares_actual_sequences(tmp_path: Path) -> None:
    m1, m3 = tmp_path / "m1", tmp_path / "m3"
    _run(m1, "m1_architecture_action_control")
    _run(m3, "m3_ours")
    result = audit_training_pair(m1, m3)
    assert result["status"] == "PASS" and result["compared_rows"] == 2
    rows = (m3 / "train_rank0.jsonl").read_text().splitlines()
    changed = json.loads(rows[1])
    changed["official_condition_frame_index"] = [99]
    rows[1] = json.dumps(changed)
    (m3 / "train_rank0.jsonl").write_text("\n".join(rows) + "\n")
    try:
        audit_training_pair(m1, m3)
    except Exception as exc:
        assert "sequence field" in str(exc)
    else:
        raise AssertionError("mismatched training sequences were accepted")
