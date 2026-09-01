from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.config_audit import TASKS
from experiments.robotwin.policy_content_adapter.model import (
    EXPECTED_ACTION_DIT_PARAMETER_COUNT,
    EXPECTED_ADAPTER_PARAMETER_COUNT,
    EXPECTED_HEAD_PARAMETER_COUNT,
)
from experiments.robotwin.policy_content_adapter.smoke_audit import (
    SmokeAuditError,
    audit_smoke_run,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _valid_smoke_run(root: Path, regime: str) -> Path:
    root.mkdir(parents=True)
    source = {
        "status": "PASS",
        "scope": "all_python_files_under_src_fastwam",
        "file_count": 47,
        "source_root": "/workspace/src",
    }
    state_bank_sha = "a" * 64
    text_cache_sha = "b" * 64
    inventory_sha = "c" * 64
    loader_rng = {
        "status": "PASS",
        "training_dataloader_generator_seed": 100,
        "identity_dataloader_generator_seed": 200,
        "identity_loader_is_separate": True,
    }
    config = {
        "schema_version": 3,
        "formal": False,
        "formal_training_auto_started": False,
        "control": regime,
        "selection_role": "engineering_method_smoke",
        "tasks": list(TASKS),
        "policy": {
            "head_init_mode": "random",
            "head_init": None,
            "head_init_seed": 17,
        },
        "paired": {
            "protocol_id": "policy_native50hz_four_scene_v1",
            "supervision_mode": "contrastive",
            "state_bank": "/paired/state_bank.json",
            "text_cache_dir": "/paired/text_cache",
        },
        "loss": {"lambda_contrastive": 0.1, "lambda_paired_action": 0.0},
        "training": {
            "max_steps": 3,
            "world_size": 1,
            "effective_official_global_batch": 1,
            "effective_paired_groups_per_step": 2,
        },
        "runtime_provenance": {"fastwam_source": source},
    }
    action_count = 0 if regime == "p_v1" else EXPECTED_ACTION_DIT_PARAMETER_COUNT
    summary = {
        "status": "SMOKE_COMPLETE",
        "regime": regime,
        "paired_supervision_mode": "contrastive",
        "lambda_contrastive": 0.1,
        "paired_contrastive_gradient_enabled": True,
        "base_lineage_manifest_sha256": "f" * 64,
        "steps": 3,
        "runtime_batch_contract": {
            "status": "PASS",
            "accelerator_num_processes": 1,
            "gradient_accumulation_steps": 1,
            "effective_official_global_batch": 1,
            "effective_paired_groups_per_step": 2,
        },
        "official_loader_rng_contract": loader_rng,
        "formal_training_auto_started": False,
        "head_init": {"mode": "random", "seed": 17, "identity": None},
        "parameter_counts": {
            "content_head": EXPECTED_HEAD_PARAMETER_COUNT,
            "adapter": EXPECTED_ADAPTER_PARAMETER_COUNT,
            "action_dit": action_count,
            "total": EXPECTED_HEAD_PARAMETER_COUNT
            + EXPECTED_ADAPTER_PARAMETER_COUNT
            + action_count,
        },
        "final_gate_raw": -0.01,
        "final_gate_tanh": -0.009999,
        "last_metrics": {
            "loss_total": 1.2,
            "loss_action": 1.0,
            "loss_paired_action": 0.0,
            "loss_contrastive": 2.0,
            "positive_similarity": 0.7,
            "negative_similarity": 0.2,
            "gate_raw": -0.01,
            "gate_tanh": -0.009999,
            "content_head_grad_norm": 1.0,
            "adapter_grad_norm": 1.0,
            "loss_finite": True,
            "gradients_finite": True,
            "layer16_shape": "[1, 120, 3072]",
            "zc_shape": "[1, 8, 384]",
        },
        "official_task_sequence": list(TASKS),
        "paired_task_sequence": list(TASKS),
        "official_sample_sequence_sha256": "d" * 64,
        "paired_physical_state_sequence_sha256": "e" * 64,
    }
    identity = {
        "status": "PASS",
        "native_prefill_kv_bit_exact": True,
        "action_output_bit_exact": True,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "gate_raw": 0.0,
        "finite": True,
        "layer16_shape": [1, 120, 3072],
        "content_token_shape": [1, 8, 384],
    }
    gradient_steps = []
    for step in range(1, 4):
        action_norm = 0.0 if regime == "p_v1" else 2.0
        gradient_steps.append(
            {
                "step": step,
                "gate_raw_after_step": -0.001 * step,
                "combined": {
                    "content_head": {"all_finite": True, "gradient_norm": 1.0},
                    "adapter": {"all_finite": True, "gradient_norm": 1.0},
                    "action_dit": {
                        "all_finite": True,
                        "gradient_norm": action_norm,
                        "gradient_tensors": 0 if regime == "p_v1" else 2,
                    },
                    "video_backbone": {"gradient_tensors": 0, "gradient_norm": 0.0},
                    "vae": {"gradient_tensors": 0, "gradient_norm": 0.0},
                },
                "action_only_probe": {
                    "all_finite": True,
                    "gate_grad_norm": 1.0,
                    "head_grad_norm": 0.0 if step == 1 else 1.0,
                    "adapter_attention_grad_norm": 0.0 if step == 1 else 1.0,
                },
                "action_only_official_content_token_grad_norm": 0.0
                if step == 1
                else 1.0,
            }
        )
    gradient = {"status": "PASS", "regime": regime, "steps": gradient_steps}
    action_update: dict[str, object]
    if regime == "p_v1":
        action_update = {"changed": False}
    else:
        action_update = {
            "changed": True,
            "changed_fraction": 0.75,
            "required_changed_strata": 7,
            "bf16_deployment_category_visibility": {
                "early": True,
                "mid": True,
                "late": True,
                "head": True,
            },
            "optimizer_exp_avg": {
                "all_finite": True,
                "nonzero_fraction": 0.75,
            },
        }
    update = {
        "head_and_adapter": {
            "max_abs_delta_by_module": {"content_head": 0.01, "adapter": 0.02},
            "all_finite": True,
        },
        "action_dit": action_update,
    }
    official = {
        "task_order": list(TASKS),
        "loader_rng_contract": loader_rng,
        "task_histogram": {
            task: {"episodes": 1, "samples": 1} for task in TASKS
        },
    }
    provenance = {
        "stream_contract": {
            "concatenated": False,
            "official_role": "policy_action_supervision",
            "paired_role": "content_invariance_supervision",
            "paired_supervision_mode": "contrastive",
        },
        "paired": {
            "protocol_id": "policy_native50hz_four_scene_v1",
            "r3_role": "training_positive",
            "r3_training_positive": True,
            "variant_names": [
                "clean",
                "style_00_seed_0",
                "style_01_seed_1",
                "style_02_seed_2",
            ],
            "view_count": 4,
            "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "camera_count": 3,
            "native_fps": 50,
            "action_steps": 32,
            "action_dim": 14,
            "temporal_resampling": "none",
            "paired_state_bank_sha256": state_bank_sha,
            "physical_state_inventory_sha256": inventory_sha,
            "shared_state_bank_contract": {
                "status": "PASS",
                "kind": "policy_shared_paired_state_bank",
                "physical_state_count": 720,
                "sampling": {
                    "algorithm": "sha256_rank_endpoint_safe_v1",
                    "seed": 42,
                    "states_per_trajectory": 8,
                },
            },
            "extraction_contract": {
                "schema": "policy_cache_extraction_contract_v1",
                "runtime_artifacts": {
                    "text_cache": {
                        "kind": "directory",
                        "size_bytes": 1,
                        "file_count": 1,
                        "sha256": text_cache_sha,
                    }
                },
            },
            "native_prefill_identity_audit": {
                "status": "PASS",
                "checked_states": 1,
                "rtol": 0.0,
                "atol": 0.0,
            },
        },
    }
    distribution = {
        "status": "DIAGNOSTIC_ONLY_POLICY_NATIVE50HZ",
        "official_clean_claim_supported": True,
        "official_domain_partition_verified": True,
        "intrinsic_metadata_domain_field": False,
        "automatic_data_substitution": False,
    }
    rollout = {
        "status": "PASS",
        "kind": "no_sapien_one_action_rollout_smoke",
        "sapien_imported": False,
        "tasks": [
            {
                "task": task,
                "executed_actions": 1,
                "action_shape": [14],
                "action_finite": True,
                "zc_shape": [1, 8, 384],
                "zc_finite": True,
            }
            for task in TASKS
        ],
        "checkpoint_audit": {"fastwam_source": dict(source)},
    }

    artifacts = {
        "requested_config.json": {},
        "run_config.json": config,
        "artifact_identities.json": {
            "base_checkpoint": {"status": "PASS"},
            "base_lineage_manifest": {"sha256": "f" * 64},
            "paired_state_bank": {"sha256": state_bank_sha},
            "paired_text_cache": {"sha256": text_cache_sha},
        },
        "official_subset_audit.json": official,
        "base_lineage_audit.json": {
            "status": "PASS",
            "kind": "policy_author_release_base_lineage",
            "base_kind": "author_release",
        },
        "release_paired_binding_audit.json": {"status": "PASS"},
        "release_paired_binding_crosscheck.json": {"status": "PASS"},
        "identity_audit.json": identity,
        "data_provenance_audit.json": provenance,
        "data_distribution_audit.json": distribution,
        "gradient_audit.json": gradient,
        "parameter_update_audit.json": update,
        "training_summary.json": summary,
        "rollout_load_execute.json": rollout,
    }
    for filename, value in artifacts.items():
        _write_json(root / filename, value)
    (root / "checkpoint.pt").write_bytes(b"compact-checkpoint")
    with (root / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "loss_total"))
        writer.writeheader()
        for step in range(1, 4):
            writer.writerow({"step": step, "loss_total": 1.0})
    return root


@pytest.mark.parametrize("regime", ("p_v1", "p_v2"))
def test_strict_smoke_audit_accepts_all_ten_goals(
    tmp_path: Path, regime: str
) -> None:
    report = audit_smoke_run(_valid_smoke_run(tmp_path / regime, regime), regime)
    assert report["status"] == "PASS"
    assert report["tasks"] == list(TASKS)
    assert report["ten_smoke_goals"]["rollout_load_execute"] == "PASS"


def test_strict_smoke_audit_rejects_nonidentity(tmp_path: Path) -> None:
    root = _valid_smoke_run(tmp_path / "bad-identity", "p_v1")
    identity_path = root / "identity_audit.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["max_abs_error"] = 1e-5
    _write_json(identity_path, identity)
    with pytest.raises(SmokeAuditError, match="max_abs_error is nonzero"):
        audit_smoke_run(root, "p_v1")


def test_strict_smoke_audit_requires_r3_as_training_positive(tmp_path: Path) -> None:
    root = _valid_smoke_run(tmp_path / "bad-r3-role", "p_v1")
    provenance_path = root / "data_provenance_audit.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["paired"]["r3_role"] = "holdout"
    provenance["paired"]["r3_training_positive"] = False
    _write_json(provenance_path, provenance)
    with pytest.raises(SmokeAuditError, match="R3 is not a training positive"):
        audit_smoke_run(root, "p_v1")


def test_strict_smoke_audit_rejects_incomplete_rollout_source_scope(
    tmp_path: Path,
) -> None:
    root = _valid_smoke_run(tmp_path / "bad-source", "p_v2")
    rollout_path = root / "rollout_load_execute.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout["checkpoint_audit"]["fastwam_source"]["scope"] = "selected_files_only"
    _write_json(rollout_path, rollout)
    with pytest.raises(SmokeAuditError, match="source scopes differ"):
        audit_smoke_run(root, "p_v2")
