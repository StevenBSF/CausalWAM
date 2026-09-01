"""Strict post-training M1/M3 fairness and update audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .paired_data import sha256_file


AUDIT_SCHEMA = "motus_policy_content_adapter_m1_m3_training_audit"


class TrainingAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingAuditError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                _require(isinstance(value, dict), f"{path} has a non-object row")
                rows.append(value)
    return rows


def audit_training_pair(
    m1_dir: str | Path,
    m3_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    m1_root, m3_root = Path(m1_dir).resolve(), Path(m3_dir).resolve()
    summaries = [_load(root / "training_summary.json") for root in (m1_root, m3_root)]
    identities = [_load(root / "run_identity.json") for root in (m1_root, m3_root)]
    _require(summaries[0].get("status") == summaries[1].get("status") == "COMPLETE", "both runs must be complete")
    _require(summaries[0].get("control") == "m1_architecture_action_control", "first run is not M1")
    _require(summaries[1].get("control") == "m3_ours", "second run is not M3")
    _require(float(summaries[0].get("lambda_contrastive", -1)) == 0.0, "M1 contrastive weight is not zero")
    _require(float(summaries[1].get("lambda_contrastive", 0)) > 0.0, "M3 contrastive weight is not positive")
    _require(summaries[0].get("temperature") == summaries[1].get("temperature"), "M1/M3 temperature differs")
    for field in (
        "regime",
        "training_seed",
        "optimizer_steps",
        "micro_steps",
        "world_size",
        "global_batch",
        "training_profile",
        "epochs",
        "steps_per_epoch",
        "scheduler_contract",
        "gradient_audit_backend",
        "initial_hash_stage",
        "artifact_shas",
        "initial_conditioner_sha256",
        "initial_action_expert_sha256",
    ):
        _require(summaries[0].get(field) == summaries[1].get(field), f"M1/M3 {field} differs")
    _require(
        summaries[0].get("gradient_audit_backend")
        == "deepspeed_zero_partition_pre_step_v1",
        "formal gradient audit did not run before the DeepSpeed optimizer step",
    )
    _require(
        summaries[0].get("initial_hash_stage")
        == "post_accelerator_prepare_pre_optimizer_step",
        "frozen/update hashes do not use the DeepSpeed runtime boundary",
    )
    for field in (
        "artifact_shas",
        "initial_conditioner_sha256",
        "initial_action_expert_sha256",
        "official_sequence_sha256",
        "paired_sequence_sha256",
    ):
        _require(identities[0].get(field) == identities[1].get(field), f"run identity {field} differs")
    _require(summaries[0]["final_conditioner_sha256"] != summaries[0]["initial_conditioner_sha256"], "M1 conditioner did not update")
    _require(summaries[1]["final_conditioner_sha256"] != summaries[1]["initial_conditioner_sha256"], "M3 conditioner did not update")
    if summaries[0]["regime"] == "m_p1":
        _require(summaries[0]["final_action_expert_sha256"] == summaries[0]["initial_action_expert_sha256"], "M1 M-P1 Action Expert changed")
        _require(summaries[1]["final_action_expert_sha256"] == summaries[1]["initial_action_expert_sha256"], "M3 M-P1 Action Expert changed")
    else:
        _require(summaries[0]["final_action_expert_sha256"] != summaries[0]["initial_action_expert_sha256"], "M1 M-P2 Action Expert did not update")
        _require(summaries[1]["final_action_expert_sha256"] != summaries[1]["initial_action_expert_sha256"], "M3 M-P2 Action Expert did not update")

    compared_rows = 0
    gradient_rows = [0, 0]
    positive_head_rows = [0, 0]
    sequence_fields = (
        "optimizer_step",
        "micro_step",
        "rank",
        "action_rng_seed",
        "official_task",
        "official_domain",
        "official_episode_index",
        "official_condition_frame_index",
        "paired_physical_state_ids",
        "paired_task_ids",
    )
    world_size = int(summaries[0]["world_size"])
    log_identities = []
    for rank in range(world_size):
        paths = [root / f"train_rank{rank}.jsonl" for root in (m1_root, m3_root)]
        rows = [_load_jsonl(path) for path in paths]
        _require(len(rows[0]) == len(rows[1]) == int(summaries[0]["micro_steps"]), f"rank {rank} log length changed")
        for row_index, (m1_row, m3_row) in enumerate(zip(*rows, strict=True)):
            for field in sequence_fields:
                _require(m1_row.get(field) == m3_row.get(field), f"rank {rank} row {row_index} sequence field {field} differs")
            _require(float(m1_row["loss_total"]) == float(m1_row["loss_action"]), "M1 total is not action-only")
            # The action MSE is FP32 while the cached-token contrastive branch
            # is BF16.  Reconstruct the actual PyTorch promotion/rounding path
            # rather than re-evaluating the three logged scalars in FP64.
            expected_m3 = float(
                torch.tensor(
                    float(m3_row["loss_action"]), dtype=torch.float32
                )
                + torch.tensor(
                    float(m3_row["loss_contrastive"]), dtype=torch.bfloat16
                )
                * float(summaries[1]["lambda_contrastive"])
            )
            _require(
                math.isclose(
                    float(m3_row["loss_total"]),
                    expected_m3,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ),
                "M3 total-loss formula changed",
            )
            for run_index, row in enumerate((m1_row, m3_row)):
                for loss_name in (
                    "loss_total",
                    "loss_action",
                    "loss_contrastive",
                ):
                    _require(
                        math.isfinite(float(row[loss_name])),
                        f"rank {rank} row {row_index} {loss_name} is non-finite",
                    )
                gradient = row.get("gradient_audit")
                if gradient is None:
                    continue
                _require(
                    gradient.get("status") == "PASS",
                    f"rank {rank} row {row_index} gradient audit failed",
                )
                _require(
                    gradient.get("backend")
                    == summaries[run_index]["gradient_audit_backend"],
                    f"rank {rank} row {row_index} gradient audit backend changed",
                )
                _require(
                    float(gradient.get("adapter_grad_norm", 0)) > 0
                    and float(gradient.get("adapter_gate_grad", 0)) != 0,
                    f"rank {rank} row {row_index} adapter gradient path is closed",
                )
                if summaries[run_index]["regime"] == "m_p1":
                    _require(
                        float(gradient.get("action_expert_grad_norm", -1)) == 0.0,
                        "M-P1 Action Expert received gradients",
                    )
                else:
                    _require(
                        float(gradient.get("action_expert_grad_norm", 0)) > 0.0,
                        "M-P2 Action Expert received no gradients",
                    )
                gradient_rows[run_index] += 1
                if float(gradient.get("content_head_grad_norm", 0)) > 0:
                    positive_head_rows[run_index] += 1
            compared_rows += 1
        log_identities.append(
            {
                "rank": rank,
                "m1": {"path": str(paths[0]), "size_bytes": paths[0].stat().st_size, "sha256": sha256_file(paths[0])},
                "m3": {"path": str(paths[1]), "size_bytes": paths[1].stat().st_size, "sha256": sha256_file(paths[1])},
            }
        )
    expected_gradient_rows = int(summaries[0]["optimizer_steps"]) * world_size
    _require(
        gradient_rows == [expected_gradient_rows, expected_gradient_rows],
        "optimizer-step gradient audit count changed",
    )
    _require(
        all(value > 0 for value in positive_head_rows),
        "M1 or M3 never opened a Content Head gradient path",
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "training_seed": summaries[0]["training_seed"],
        "regime": summaries[0]["regime"],
        "optimizer_steps": summaries[0]["optimizer_steps"],
        "micro_steps_per_rank": summaries[0]["micro_steps"],
        "world_size": world_size,
        "compared_rows": compared_rows,
        "gradient_rows": {
            "m1": gradient_rows[0],
            "m3": gradient_rows[1],
        },
        "positive_content_head_gradient_rows": {
            "m1": positive_head_rows[0],
            "m3": positive_head_rows[1],
        },
        "gradient_audit_backend": summaries[0]["gradient_audit_backend"],
        "initial_hash_stage": summaries[0]["initial_hash_stage"],
        "sequence_fields": list(sequence_fields),
        "only_objective_difference": "lambda_contrastive_0_vs_positive",
        "total_loss_reconstruction": (
            "fp32_action_plus_bfloat16_weighted_contrastive"
        ),
        "logs": log_identities,
    }
    if output_path is not None:
        output = Path(output_path).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", required=True)
    parser.add_argument("--m3-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_training_pair(args.m1_dir, args.m3_dir, output_path=args.output)
    print(json.dumps({"status": result["status"], "compared_rows": result["compared_rows"], "output": str(Path(args.output).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
