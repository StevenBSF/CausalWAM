from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from experiments.robotwin.policy_content_adapter import config_audit
from experiments.robotwin.policy_content_adapter import model as policy_model
from experiments.robotwin.policy_content_adapter import (
    release_engineering_smoke_resume as resume_module,
)
from experiments.robotwin.policy_content_adapter.config_audit import (
    ConfigAuditError,
    load_config,
    validate_c1_c3_pair,
    validate_config_structure,
)
from experiments.robotwin.policy_content_adapter.materialize_release_engineering_smoke import (
    build_resolved_pair,
)
from experiments.robotwin.policy_content_adapter.release_official_text_cache_binding import (
    ReleaseOfficialTextCacheBindingError,
    validate_binding_payload,
)
from experiments.robotwin.policy_content_adapter.release_engineering_smoke_resume import (
    ReleaseEngineeringSmokeResumeError,
    audit_resume_after_c1_train,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _resolved_pair(tmp_path: Path) -> tuple[dict, dict]:
    return build_resolved_pair(
        c1_template=load_config(CONFIG_DIR / "c1_architecture_only.yaml"),
        c3_template=load_config(CONFIG_DIR / "c3_ours.yaml"),
        output_root=tmp_path,
        regime="p_v1",
        seed=42,
        steps=3,
        release_paired_binding_manifest=tmp_path / "paired_binding.json",
        release_paired_binding_sha256="a" * 64,
        paired_text_cache=tmp_path / "paired_text",
        paired_text_cache_sha256="b" * 64,
        paired_cache=tmp_path / "layer16.pt",
        paired_cache_sha256="c" * 64,
        official_text_cache=tmp_path / "official_text",
        official_text_cache_binding_manifest=tmp_path / "official_text_binding.json",
        official_text_cache_binding_manifest_sha256="d" * 64,
        seed_bank_manifest=tmp_path / "smoke_seeds.json",
        seed_bank_manifest_sha256="e" * 64,
        seed_bank_id="robotwin-seed-bank-test",
    )


def test_c1_c3_engineering_smoke_is_three_step_matched_pair(tmp_path: Path) -> None:
    c1, c3 = _resolved_pair(tmp_path)
    validate_config_structure(c1)
    validate_config_structure(c3)
    assert validate_c1_c3_pair(c1, c3)["fairness"] == "PASS"
    assert c1["stage"] == c3["stage"] == "smoke"
    assert c1["formal"] is c3["formal"] is False
    assert c1["p_mode_selection_manifest"] is c3["p_mode_selection_manifest"] is None
    assert c1["training"]["max_steps"] == c3["training"]["max_steps"] == 3
    assert c1["official"]["on_the_fly_text_smoke"] is False
    assert c1["official"]["text_cache_binding_manifest"]
    assert c1["loss"]["lambda_contrastive"] == 0.0
    assert c3["loss"]["lambda_contrastive"] == 0.1


def test_control_stage_still_requires_selected_p_mode(tmp_path: Path) -> None:
    c1, _ = _resolved_pair(tmp_path)
    c1["stage"] = "control"
    with pytest.raises(ConfigAuditError, match="requires p_mode_selection_manifest"):
        validate_config_structure(c1)


def test_engineering_pair_rejects_a_second_treatment_difference(tmp_path: Path) -> None:
    c1, c3 = _resolved_pair(tmp_path)
    changed = copy.deepcopy(c3)
    changed["training"]["seed"] = 43
    with pytest.raises(ConfigAuditError, match="unfair common Stage-2 mismatch"):
        validate_c1_c3_pair(c1, changed)


def test_official_text_binding_rejects_strict_aggregate_mismatch() -> None:
    counts = {
        task: {"clean": 50, "official_random": 500}
        for task in ("place_a2b_left", "open_microwave", "move_stapler_pad")
    }
    value = {
        "schema_version": 1,
        "kind": "policy_release_official_text_cache_binding",
        "status": "PASS",
        "base_lineage": {
            "sha256": "a" * 64,
            "official_manifest_sha256": "b" * 64,
        },
        "cache": {
            "directory": "/cache",
            "file_count": 68704,
            "total_size_bytes": 1,
            "aggregate_payload_sha256": "c" * 64,
        },
        "completion_audit": {"path": "/audit", "sha256": "d" * 64},
        "inventory": {"path": "/inventory", "sha256": "e" * 64},
        "strict_payload_revalidation": {
            "status": "PASS",
            "evidence": {"path": "/evidence", "sha256": "f" * 64},
            "all_payload_shapes_valid": True,
            "all_required_cache_files_present": True,
            "aggregate_payload_sha256": "0" * 64,
        },
        "official_dataset": {
            "manifest_sha256": "b" * 64,
            "selected_episode_counts_by_domain": counts,
        },
    }
    with pytest.raises(
        ReleaseOfficialTextCacheBindingError, match="strict payload aggregate differs"
    ):
        validate_binding_payload(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _resume_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "resume"
    c1, c3 = _resolved_pair(root)
    base = root / "release.pt"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_bytes(b"test release")
    base_sha = _file_sha256(base)
    monkeypatch.setattr(
        config_audit, "AUTHOR_RELEASE_CHECKPOINT_SHA256", base_sha
    )
    # Execution-readiness, including the real 12 GB author-release lineage,
    # is covered by the config/materializer tests.  This synthetic fixture is
    # deliberately tiny and isolates the resume boundary itself.
    monkeypatch.setattr(resume_module, "validate_execution_ready", lambda _cfg: None)
    for config in (c1, c3):
        config["base_checkpoint"] = str(base.resolve())
        config["artifacts"]["base_checkpoint_sha256"] = base_sha

    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    c1_path = config_dir / "c1_engineering_smoke.yaml"
    c3_path = config_dir / "c3_engineering_smoke.yaml"
    OmegaConf.save(OmegaConf.create(c1), c1_path)
    OmegaConf.save(OmegaConf.create(c3), c3_path)
    _write_json(
        root / "materialization_manifest.json",
        {
            "schema_version": 1,
            "kind": "policy_release_c1_c3_engineering_smoke_materialization",
            "status": "PASS",
            "scientific_result": False,
            "p_mode_selection_evidence": False,
            "formal_training_auto_started": False,
            "regime": "p_v1",
            "optimizer_steps_per_control": 3,
            "configs": {
                "c1": {"path": str(c1_path.resolve()), "sha256": _file_sha256(c1_path)},
                "c3": {"path": str(c3_path.resolve()), "sha256": _file_sha256(c3_path)},
            },
        },
    )

    c1_root = root / "runs/c1"
    c1_root.mkdir(parents=True)
    run_config = copy.deepcopy(c1)
    _write_json(c1_root / "run_config.json", run_config)
    checkpoint = c1_root / "checkpoint.pt"
    torch.save(
        {
            "schema": policy_model.POLICY_CHECKPOINT_SCHEMA,
            "schema_version": policy_model.POLICY_CHECKPOINT_VERSION,
            "regime": "p_v1",
            "step": 3,
            "base_checkpoint": {
                "kind": "file",
                "path": str(base.resolve()),
                "size_bytes": base.stat().st_size,
                "mtime_ns": base.stat().st_mtime_ns,
                "sha256": base_sha,
            },
            "artifact_identities": {},
            "head_config": {},
            "adapter_config": {},
            "run_config": run_config,
            "content_head": {"placeholder": torch.ones(1)},
            "content_adapter": {"placeholder": torch.ones(1)},
        },
        checkpoint,
    )
    contract_sha = "1" * 64
    official_sha = "2" * 64
    paired_sha = "3" * 64
    _write_json(
        c1_root / "training_summary.json",
        {
            "status": "SMOKE_COMPLETE",
            "formal_training_auto_started": False,
            "control": "c1_architecture_only",
            "regime": "p_v1",
            "steps": 3,
            "lambda_contrastive": 0.0,
            "paired_contrastive_gradient_enabled": False,
            "checkpoint": str(checkpoint.resolve()),
            "matched_stream_contract_sha256": contract_sha,
            "official_sample_sequence_sha256": official_sha,
            "paired_physical_state_sequence_sha256": paired_sha,
            "deliverable_status": {
                "implementation": "PASS",
                "short_update": "PASS",
                "gradient_audit": "PASS",
                "rollout_load_execute": "PENDING_SEPARATE_SMOKE",
            },
        },
    )
    _write_json(
        c1_root / "training_sequence_audit.json",
        {
            "status": "PASS",
            "official_sample_count": 3,
            "paired_physical_state_count": 6,
            "matched_stream_contract_sha256": contract_sha,
            "official_sample_sequence_sha256": official_sha,
            "paired_physical_state_sequence_sha256": paired_sha,
        },
    )
    _write_json(
        c1_root / "matched_stream_contract.json",
        {"status": "PASS", "sha256": contract_sha},
    )
    _write_json(
        c1_root / "gradient_audit.json",
        {
            "status": "PASS",
            "steps": [
                {
                    "step": step,
                    "action_only_probe": {
                        "all_finite": True,
                        "head_grad_norm": 0.0 if step == 1 else 0.1,
                        "adapter_attention_grad_norm": 0.0 if step == 1 else 0.1,
                        "gate_grad_norm": 0.1,
                    },
                }
                for step in (1, 2, 3)
            ],
        },
    )
    _write_json(
        c1_root / "parameter_update_audit.json",
        {
            "final_content_head_sha256": "4" * 64,
            "final_adapter_sha256": "5" * 64,
            "head_and_adapter": {
                "all_finite": True,
                "changed_parameter_tensors": 2,
                "max_abs_delta_by_module": {
                    "content_head": 0.1,
                    "adapter": 0.1,
                },
            },
        },
    )
    return root


def test_resume_after_c1_train_requires_and_accepts_exact_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _resume_fixture(tmp_path, monkeypatch)
    report = audit_resume_after_c1_train(output_root=root)
    assert report["status"] == "PASS"
    assert report["resume_boundary"] == "after_c1_train_before_c1_deploy"
    assert report["c3_run_directory_absent"] is True


def test_resume_after_c1_train_rejects_failed_summary_and_partial_c3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _resume_fixture(tmp_path, monkeypatch)
    summary_path = root / "runs/c1/training_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "FAILED"
    _write_json(summary_path, summary)
    with pytest.raises(ReleaseEngineeringSmokeResumeError, match="not complete"):
        audit_resume_after_c1_train(output_root=root)

    summary["status"] = "SMOKE_COMPLETE"
    _write_json(summary_path, summary)
    (root / "runs/c3").mkdir()
    with pytest.raises(ReleaseEngineeringSmokeResumeError, match="C3 run directory"):
        audit_resume_after_c1_train(output_root=root)
