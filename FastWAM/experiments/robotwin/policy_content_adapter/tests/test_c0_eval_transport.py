from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.policy_content_adapter import eval_robotwin_single
from experiments.robotwin.policy_content_adapter.c0_eval_transport import (
    C0TransportError,
    _c0_dataset_initialization_work_dir,
    _new_transport_conditioner,
    build_c0_eval_transport,
)
from experiments.robotwin.policy_content_adapter.c0_dev_gate_audit import (
    C0DevGateAuditError,
    audit_c0_dev_gate,
)
from experiments.robotwin.policy_content_adapter.model import module_state_sha256
from experiments.robotwin.policy_content_adapter.rollout_policy import (
    _read_checkpoint_provenance,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "author_release.pt"
    stats = tmp_path / "dataset_stats.json"
    official = tmp_path / "official.json"
    base.write_bytes(b"fixed-author-release")
    stats.write_text('{"stats":true}')
    official.write_text('{"manifest":true}')
    evidence = {}
    for name in ("readme", "task_config", "data_config", "evaluation_config"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name)
        evidence[name] = _identity(path)
    lineage = tmp_path / "author_release_lineage.json"
    lineage.write_text(json.dumps({
        "schema_version": 1,
        "kind": "policy_author_release_base_lineage",
        "status": "PASS",
        "lineage_id": "test-author-release",
        "base_kind": "author_release",
        "checkpoint": _identity(base),
        "dataset_stats": _identity(stats),
        "official_partition": {
            "task_count": 50, "episodes_per_task": 550, "clean_per_task": 50,
            "random_per_task": 500, "total_episodes": 27500,
            "partition_rule": "first_50_clean_next_500_official_random",
            "domain_rule_scope": "hash_bound_protocol_partition_not_checkpoint_payload",
            "manifest": _identity(official),
        },
        "source": {
            "repository": "test", "revision": "a" * 40,
            "release_model_id": "test", "checkpoint_url": "https://example/checkpoint",
            "dataset_url": "https://example/data", "evidence_files": evidence,
        },
        "model_contract": {
            "task_config": "robotwin_uncond_3cam_384_1e-4",
            "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "raw_camera_shape_chw": [3, 480, 640],
            "transformed_camera_shape_chw": [3, 240, 320],
            "final_video_size_hw": [384, 320], "action_steps": 32,
            "action_dim": 14, "state_dim": 14, "normalization": "z-score",
            "stepwise_action_normalization": False,
            "checkpoint_load_contract": {
                "loader": "strict_load_release_checkpoint", "mot_strict": True,
                "proprio_encoder_strict": True, "expected_missing_keys": 0,
                "expected_unexpected_keys": 0,
            },
        },
    }))
    evaluator = tmp_path / "eval.py"
    evaluator.write_text("# evaluator")
    bank = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=4, episodes_per_task=100, evaluator_source=evaluator,
        purpose="development_analysis",
    )
    seed_bank = tmp_path / "dev_bank.json"
    seed_bank.write_text(json.dumps(bank))
    model_base = tmp_path / "models"
    vae = model_base / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
    text = model_base / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
    tokenizer = model_base / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    vae.parent.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    vae.write_bytes(b"vae")
    text.write_bytes(b"text")
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    conditioner = _new_transport_conditioner(0)
    identity_audit = tmp_path / "identity.json"
    identity_audit.write_text(json.dumps({
        "status": "PASS", "kind": "policy_c0_zero_gate_runtime_identity",
        "native_prefill_kv_bit_exact": True, "action_output_bit_exact": True,
        "max_abs_error": 0.0, "max_rel_error": 0.0, "gate_raw": 0.0,
        "base_checkpoint_sha256": _sha(base), "dataset_stats_sha256": _sha(stats),
        "official_manifest_sha256": _sha(official),
        "base_lineage_manifest_sha256": _sha(lineage), "transport_seed": 0,
        "transport_head_sha256": module_state_sha256(conditioner.head),
        "transport_adapter_sha256": module_state_sha256(conditioner.adapter),
        "installed_transport_head_sha256": "1" * 64,
        "installed_transport_adapter_sha256": "2" * 64,
    }))
    return locals()


def _source_audit(tmp_path: Path) -> dict:
    source_root = tmp_path / "src" / "fastwam"
    source_root.mkdir(parents=True)
    package = source_root / "__init__.py"
    package.write_text("")
    return {
        "status": "PASS", "scope": "all_python_files_under_src_fastwam",
        "file_count": 1, "source_root": str(source_root), "package_file": str(package),
        "files": {"fastwam/__init__.py": {"path": str(package), "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}},
    }


def _bind_source(paths: dict, source: dict) -> None:
    payload = json.loads(paths["identity_audit"].read_text())
    payload["fastwam_source_sha256"] = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    paths["identity_audit"].write_text(json.dumps(payload))


def _kwargs(paths: dict, source: dict, output: Path) -> dict:
    bank = json.loads(paths["seed_bank"].read_text())
    return {
        "base_checkpoint": paths["base"], "dataset_stats": paths["stats"],
        "model_base_path": paths["model_base"], "official_manifest": paths["official"],
        "base_lineage_manifest": paths["lineage"], "identity_audit": paths["identity_audit"],
        "output": output, "rollout_protocol_id": "three_task_policy_online_v2",
        "simulator_seed_bank_id": bank["simulator_seed_bank_id"],
        "simulator_seed_bank_manifest": paths["seed_bank"], "source_audit": source,
    }


def test_c0_transport_is_fixed_release_without_training_seed(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    source = _source_audit(tmp_path)
    _bind_source(paths, source)
    output = tmp_path / "c0.pt"
    report = build_c0_eval_transport(**_kwargs(paths, source, output))
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert report["status"] == "PASS"
    assert report["training_seed"] is None
    assert payload["run_config"]["training"] == {"seed": None, "stage2_steps": 0}
    assert payload["run_config"]["formal"] is False
    assert payload["run_config"]["c0_semantics"]["base_lineage_manifest_sha256"] == _sha(paths["lineage"])
    assert payload["content_adapter"]["gate"].item() == 0.0
    assert "action_expert" not in payload


def test_c0_transport_rejects_stale_release_lineage_and_overwrite(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path / "stale")
    source = _source_audit(tmp_path / "stale")
    _bind_source(paths, source)
    lineage = json.loads(paths["lineage"].read_text())
    lineage["checkpoint"]["sha256"] = "0" * 64
    paths["lineage"].write_text(json.dumps(lineage))
    with pytest.raises(C0TransportError, match="author release lineage"):
        build_c0_eval_transport(**_kwargs(paths, source, tmp_path / "bad.pt"))

    paths = _artifacts(tmp_path / "fresh")
    source = _source_audit(tmp_path / "fresh")
    _bind_source(paths, source)
    output = tmp_path / "exists.pt"
    output.write_bytes(b"keep")
    with pytest.raises(C0TransportError, match="refusing to overwrite"):
        build_c0_eval_transport(**_kwargs(paths, source, output))


def test_c0_dataset_initialization_work_dir_is_scoped(tmp_path: Path) -> None:
    from fastwam.utils import misc

    sentinel = object()
    previous = getattr(misc, "_WORK_DIR", None)
    misc._WORK_DIR = sentinel
    try:
        destination = tmp_path / "audit.json"
        with _c0_dataset_initialization_work_dir(destination) as staging:
            assert staging.is_dir()
            assert staging.parent == tmp_path.resolve()
            assert Path(getattr(misc, "_WORK_DIR")) == staging
        assert getattr(misc, "_WORK_DIR") is sentinel
        assert not staging.exists()
    finally:
        misc._WORK_DIR = previous


def _completed_c0_dev_gate(tmp_path: Path) -> tuple[dict, dict[str, Path], Path, Path]:
    paths = _artifacts(tmp_path / "gate")
    bank = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=9,
        episodes_per_task=1,
        evaluator_source=paths["evaluator"],
        purpose="development_analysis",
    )
    paths["seed_bank"].write_text(json.dumps(bank))
    source = _source_audit(tmp_path / "gate")
    _bind_source(paths, source)
    identity = json.loads(paths["identity_audit"].read_text())
    identity["official_selection"] = {
        "selected_episode_counts_by_domain": {
            task: {"clean": 50, "official_random": 500}
            for task in ("place_a2b_left", "open_microwave", "move_stapler_pad")
        }
    }
    identity["runtime_artifacts"] = {
        "text_cache": {
            "kind": "audited_directory_binding",
            "directory_bytes_rehashed_for_c0": False,
        }
    }
    paths["identity_audit"].write_text(json.dumps(identity))
    checkpoint = tmp_path / "c0_dev_transport.pt"
    kwargs = _kwargs(paths, source, checkpoint)
    kwargs.update(
        simulator_seed_bank_id=bank["simulator_seed_bank_id"],
        episodes_per_task=1,
    )
    build_c0_eval_transport(**kwargs)
    metadata = _read_checkpoint_provenance(checkpoint)["metadata"]
    settings = {
        "schema": "robotwin.policy_content_adapter.rollout_settings",
        "schema_version": 1,
        "episodes_per_task": 1,
        "rollout_protocol_id": "three_task_policy_online_v2",
    }
    settings_sha = eval_robotwin_single._canonical_sha256(settings)
    contract = {
        "control": "c0_original",
        "stage": "control",
        "training_seed": None,
        "policy_regime": None,
        "head_init_sha256": None,
        "gca_init_sha256": None,
        "stage2_recipe_sha256": None,
        "p_mode_selection_manifest_sha256": None,
        "base_checkpoint_sha256": metadata["base_checkpoint"]["sha256"],
        "dataset_stats_sha256": metadata["artifact_identities"]["dataset_stats"]["sha256"],
        "base_lineage_manifest_sha256": metadata["artifact_identities"]["base_lineage_manifest"]["sha256"],
        "simulator_seed_bank_id": bank["simulator_seed_bank_id"],
        "simulator_seed_bank_purpose": "development_analysis",
        "declared_tasks": ["place_a2b_left", "open_microwave", "move_stapler_pad"],
        "declared_domains": ["clean", "official_random"],
        "declared_episodes_per_task": 1,
        "formal_evaluation_eligible": False,
    }
    runs = []
    for task in contract["declared_tasks"]:
        for task_config, domain, phase in (
            ("demo_clean", "clean", "clean"),
            ("demo_randomized", "official_random", "random"),
        ):
            log = tmp_path / f"{task}_{phase}.log"
            result = tmp_path / f"{task}_{phase}.txt"
            log.write_text("SAPIEN render device: explicit\n")
            result.write_text("1.0\n")
            runs.append(
                {
                    "task": task,
                    "task_config": task_config,
                    "domain": domain,
                    "phase": phase,
                    "episodes": 1,
                    "simulator_seed": 9,
                    "simulator_seed_bank_id": bank["simulator_seed_bank_id"],
                    "simulator_seed_bank_purpose": "development_analysis",
                    "rollout_settings_sha256": settings_sha,
                    "physical_gpu_index": 0,
                    "render_device_alias": "pci:0000:00:01.0",
                    "success_rate": 1.0,
                    "log": str(log),
                    "result": str(result),
                }
            )
    completed = {
        "schema": eval_robotwin_single.COMPLETED_ROLLOUTS_SCHEMA,
        "schema_version": eval_robotwin_single.COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_contract": contract,
        "checkpoint_fairness_identity": None,
        "episodes_per_task": 1,
        "simulator_seed": 9,
        "simulator_seed_bank": bank,
        "simulator_seed_bank_id": bank["simulator_seed_bank_id"],
        "simulator_seed_bank_purpose": "development_analysis",
        "rollout_protocol_id": "three_task_policy_online_v2",
        "rollout_settings": settings,
        "rollout_settings_sha256": settings_sha,
        "evaluation_protocol": {"eligible": False, "control": None},
        "evaluation_records": [],
        "gpu_runtime_binding": {
            "status": "PASS",
            "physical_gpu_index": 0,
            "pci_bus_id": "0000:00:01.0",
            "render_device_alias": "pci:0000:00:01.0",
            "sapien": {"can_render": True, "pci_bus_id": "0000:00:01.0"},
        },
        "runs": runs,
    }
    completed_path = tmp_path / "completed_rollouts.json"
    completed_path.write_text(json.dumps(completed))
    return completed, paths, checkpoint, completed_path


def test_c0_dev_gate_is_six_cell_nonformal_deployment_evidence(tmp_path: Path) -> None:
    completed, paths, checkpoint, completed_path = _completed_c0_dev_gate(tmp_path)
    report = audit_c0_dev_gate(
        identity_audit=paths["identity_audit"],
        transport_checkpoint=checkpoint,
        simulator_seed_bank_manifest=paths["seed_bank"],
        completed_rollouts=completed_path,
        output=tmp_path / "audit.json",
    )
    assert report["status"] == "PASS"
    assert report["scientific_result"] is False
    assert report["total_rollout_episodes"] == 6

    completed["evaluation_records"] = [{"not": "allowed"}]
    completed_path.write_text(json.dumps(completed))
    with pytest.raises(C0DevGateAuditError, match="formal records"):
        audit_c0_dev_gate(
            identity_audit=paths["identity_audit"],
            transport_checkpoint=checkpoint,
            simulator_seed_bank_manifest=paths["seed_bank"],
            completed_rollouts=completed_path,
            output=tmp_path / "bad-audit.json",
        )
