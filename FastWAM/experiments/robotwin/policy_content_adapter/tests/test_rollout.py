"""Lightweight fail-closed rollout bridge tests (no model weights or SAPIEN)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from experiments.robotwin.policy_content_adapter import eval_robotwin_single
from experiments.robotwin.policy_content_adapter import evaluation_protocol
from experiments.robotwin.policy_content_adapter import model as policy_model
from experiments.robotwin.policy_content_adapter import p_mode_selection
from experiments.robotwin.policy_content_adapter import rollout_policy


def test_default_rollout_task_order_matches_training_contract() -> None:
    assert rollout_policy.DEFAULT_TASKS == (
        "place_a2b_left",
        "open_microwave",
        "move_stapler_pad",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_checkpoint(
    tmp_path: Path,
    *,
    include_model_base: bool = True,
    stats_identity: bool = False,
    control: str = "c3_ours",
    training_seed: int = 2,
    rollout_protocol_id: str = "robotwin_policy_online_v2",
    simulator_seed_bank_id: str | None = None,
    episodes_per_task: int = 100,
    mechanism_followup: bool = False,
) -> tuple[Path, Path, Path]:
    model_base = tmp_path / "components"
    release_dir = model_base / "fastwam_release"
    release_dir.mkdir(parents=True)
    base = release_dir / "release.pt"
    base.write_bytes(b"immutable release checkpoint")
    stats = release_dir / "dataset_stats.json"
    stats.write_bytes(b'{"action": {"mean": [0.0]}}\n')
    base_sha = _sha256(base)
    stats_sha = _sha256(stats)
    lineage_path = release_dir / "author_release_lineage.json"
    lineage_path.write_text('{"kind":"policy_author_release_base_lineage","status":"PASS"}')
    lineage_sha = _sha256(lineage_path)
    official_sha = hashlib.sha256(b"official-manifest").hexdigest()
    component_shas = {
        "vae": hashlib.sha256(b"vae").hexdigest(),
        "text_encoder": hashlib.sha256(b"text-encoder").hexdigest(),
        "tokenizer": hashlib.sha256(b"tokenizer").hexdigest(),
    }
    evaluator_source = tmp_path / "eval_policy.py"
    evaluator_source.write_text("# deterministic evaluator\n", encoding="utf-8")
    dev_seed_bank = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=2,
        episodes_per_task=p_mode_selection.DEV_EPISODES_PER_CELL,
        evaluator_source=evaluator_source,
        purpose="dev_selection",
    )
    selection_shared = {
        field: (
            2 if field == "training_seed"
            else "c1_lambda0" if field == "selection_role"
            else 0.0 if field == "lambda_contrastive"
            else hashlib.sha256(field.encode()).hexdigest()
        )
        for field in p_mode_selection.SHARED_IDENTITY_FIELDS
    }
    selection_shared["rollout_protocol_id"] = rollout_protocol_id
    selection_shared.update(
        {
            "base_checkpoint_sha256": base_sha,
            "dataset_stats_sha256": stats_sha,
            "base_lineage_manifest_sha256": lineage_sha,
            "runtime_source_sha256": eval_robotwin_single._runtime_source_sha256(
                {"runtime_provenance": {"fastwam_source": rollout_policy.audit_local_fastwam_source()}}
            ),
            "official_manifest_sha256": official_sha,
            "paired_action_manifest_sha256": hashlib.sha256(b"paired-action").hexdigest(),
            "paired_state_bank_sha256": hashlib.sha256(b"paired-state").hexdigest(),
            "paired_text_cache_sha256": hashlib.sha256(b"paired-text").hexdigest(),
            "paired_cache_sha256": hashlib.sha256(b"paired-cache").hexdigest(),
        }
    )
    candidate_cells = {
        task: {"clean": 0.75, "official_random": 0.50}
        for task in rollout_policy.DEFAULT_TASKS
    }
    selection_candidates = {
        regime: {
            "regime": regime,
            "checkpoint": {"path": f"/{regime}.pt", "size_bytes": 1, "sha256": hashlib.sha256(regime.encode()).hexdigest()},
            "result_manifest": {"path": f"/{regime}.json", "size_bytes": 1, "sha256": hashlib.sha256((regime + "-result").encode()).hexdigest()},
            "result_files": [],
            "identity": selection_shared,
            "dev_seed_bank_id": dev_seed_bank["simulator_seed_bank_id"],
            "dev_seed_bank_sha256": p_mode_selection.canonical_sha256(
                p_mode_selection.seed_bank_identity_payload(dev_seed_bank)
            ),
            "episodes_per_cell": p_mode_selection.DEV_EPISODES_PER_CELL,
            "cells": candidate_cells,
            "three_task_macro": {"clean": 0.75, "official_random": 0.50},
        }
        for regime in ("p_v1", "p_v2")
    }
    selection_payload = {
        "kind": p_mode_selection.P_MODE_SELECTION_KIND,
        "schema_version": p_mode_selection.P_MODE_SELECTION_SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "selector_source": {"path": "/selector.py", "size_bytes": 1, "sha256": "a" * 64},
        "rule": p_mode_selection.SELECTION_RULE,
        "shared_candidate_identity": selection_shared,
        "dev_seed_bank": dev_seed_bank,
        "dev_seed_bank_sha256": p_mode_selection.canonical_sha256(
            p_mode_selection.seed_bank_identity_payload(dev_seed_bank)
        ),
        "candidates": selection_candidates,
        "best_clean_macro": 0.75,
        "eligible_regimes": ["p_v1", "p_v2"],
        "winner": "p_v1",
        "winner_reason": "test",
    }
    p_mode_selection.validate_selection_manifest_payload(selection_payload)
    selection_path = release_dir / "p_mode_selection.json"
    selection_path.write_text(json.dumps(selection_payload), encoding="utf-8")

    matrix_path = release_dir / "formal_matrix.json"
    matrix_path.write_text('{"status":"PASS"}')
    def small_identity(path: Path) -> dict[str, object]:
        return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    locked_configs = {}
    for locked_control in ("c1_architecture_only", "c3_ours"):
        locked_configs[locked_control] = [
            {
                "control": locked_control,
                "training_seed": seed,
                "lambda_contrastive": 0.0 if locked_control.startswith("c1") else 0.1,
                "source_config": {"path": f"/{locked_control}_{seed}.json", "size_bytes": 1, "sha256": hashlib.sha256(f"{locked_control}-{seed}".encode()).hexdigest()},
                "protocol_projection_sha256": hashlib.sha256(f"projection-{locked_control}-{seed}".encode()).hexdigest(),
            }
            for seed in (1, 2, 3)
        ]
    formal_lock_payload = {
        "kind": p_mode_selection.FORMAL_PROTOCOL_LOCK_KIND,
        "schema_version": p_mode_selection.FORMAL_PROTOCOL_LOCK_SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "base_lineage_manifest": small_identity(lineage_path),
        "p_mode_selection_manifest": small_identity(selection_path),
        "formal_matrix_audit": small_identity(matrix_path),
        "selected_policy_regime": "p_v1",
        "stage2_training_seeds": [1, 2, 3],
        "resolved_configs": locked_configs,
    }
    p_mode_selection.validate_formal_protocol_lock_manifest_payload(formal_lock_payload)
    formal_lock_path = release_dir / "formal_protocol_lock.json"
    formal_lock_path.write_text(json.dumps(formal_lock_payload))
    dev_summary = {
        "purpose": "dev_selection",
        "simulator_seed_bank_id": dev_seed_bank["simulator_seed_bank_id"],
        "member_count": dev_seed_bank["member_count"],
        "members_sha256": dev_seed_bank["members_sha256"],
        "members": dev_seed_bank["members"],
    }
    seed_bank = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=4,
        episodes_per_task=episodes_per_task,
        evaluator_source=evaluator_source,
        purpose="final_test",
        disjoint_from=[dev_summary],
        lock_ancestry={
            "p_mode_selection_manifest": small_identity(selection_path),
            "formal_protocol_lock_manifest": small_identity(formal_lock_path),
        },
    )
    seed_bank_path = release_dir / "seed_bank.json"
    seed_bank_path.write_text(json.dumps(seed_bank), encoding="utf-8")
    if simulator_seed_bank_id is None:
        simulator_seed_bank_id = seed_bank["simulator_seed_bank_id"]
    def identity(name: str, sha256: str, *, required: bool = False) -> dict[str, object]:
        return {
            "kind": "file",
            "path": str((release_dir / f"{name}.artifact").resolve()),
            "size_bytes": 1,
            "mtime_ns": 1,
            "sha256": sha256,
            "required_for_rollout": required,
            "verification_status": "PASS",
        }

    dataset_stats_identity = {
        "kind": "file",
        "path": str(stats.resolve()),
        "size_bytes": stats.stat().st_size,
        "mtime_ns": stats.stat().st_mtime_ns,
        "sha256": stats_sha,
        "required_for_rollout": True,
        "verification_status": "PASS",
    }
    artifact_identities: dict[str, dict[str, object]] = {
        "dataset_stats": dataset_stats_identity,
        "base_lineage_manifest": {
            **identity("lineage", lineage_sha),
            "path": str(lineage_path.resolve()),
            "size_bytes": lineage_path.stat().st_size,
            "mtime_ns": lineage_path.stat().st_mtime_ns,
        },
        "official_manifest": identity("official", official_sha),
        "paired_action_manifest": identity("paired_action", selection_shared["paired_action_manifest_sha256"]),
        "paired_state_bank": identity("paired_state", selection_shared["paired_state_bank_sha256"]),
        "paired_text_cache": identity("paired_text", selection_shared["paired_text_cache_sha256"]),
        "paired_train_cache": identity("paired_cache", selection_shared["paired_cache_sha256"]),
        "simulator_seed_bank_manifest": {
            **identity("seed_bank", _sha256(seed_bank_path), required=True),
            "path": str(seed_bank_path.resolve()),
            "size_bytes": seed_bank_path.stat().st_size,
            "mtime_ns": seed_bank_path.stat().st_mtime_ns,
        },
        "p_mode_selection_manifest": {
            **identity("selection", _sha256(selection_path), required=True),
            "path": str(selection_path.resolve()),
            "size_bytes": selection_path.stat().st_size,
            "mtime_ns": selection_path.stat().st_mtime_ns,
        },
        "formal_protocol_lock_manifest": {
            **identity("formal_lock", _sha256(formal_lock_path), required=True),
            "path": str(formal_lock_path.resolve()),
            "size_bytes": formal_lock_path.stat().st_size,
            "mtime_ns": formal_lock_path.stat().st_mtime_ns,
        },
        **{
            name: identity(name, sha256)
            for name, sha256 in component_shas.items()
        },
    }
    official: dict[str, object] = {"dataset_stats": str(stats.resolve())}
    if stats_identity:
        official["dataset_stats_identity"] = {
            "path": str(stats.resolve()),
            "size_bytes": stats.stat().st_size,
            "sha256": _sha256(stats),
        }
    run_config: dict[str, object] = {
        "kind": "policy_content_adapter_run",
        "stage": "formal",
        "formal": True,
        "control": control,
        "base_lineage_manifest": str(lineage_path.resolve()),
        "formal_protocol_lock_manifest": str(formal_lock_path.resolve()),
        "p_mode_selection_manifest": str(selection_path.resolve()),
        "tasks": list(rollout_policy.DEFAULT_TASKS),
        "training": {
            "seed": training_seed,
            "max_steps": 2,
            "official_batch_size": 1,
            "paired_groups_per_batch": 2,
            "world_size": 1,
            "gradient_accumulation_steps": 1,
            "effective_official_global_batch": 1,
            "effective_paired_groups_per_step": 2,
            "num_workers": 0,
            "mixed_precision": "bf16",
            "model_dtype": "bf16",
            "max_grad_norm": 1.0,
            "require_cuda": True,
            "separate_stream_rng": True,
            "preserve_official_sequence_across_controls": True,
        },
        "optimizer": {
            "name": "adamw",
            "lr_scheduler": "constant",
            "trainable_parameter_dtype": "fp32",
            "head_adapter_lr": 1.0e-4,
            "action_dit_lr": 1.0e-5,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
        },
        "loss": {
            "lambda_contrastive": 0.1 if control == "c3_ours" else 0.0,
        },
        "evaluation": {
            "tasks": list(rollout_policy.DEFAULT_TASKS),
            "required_domains": ["clean", "official_random"],
            "rollout_protocol_id": rollout_protocol_id,
            "simulator_seed_bank_id": simulator_seed_bank_id,
            "simulator_seed_bank_manifest": str(seed_bank_path.resolve()),
            "simulator_seed_bank_purpose": "final_test",
            "episodes_per_task": episodes_per_task,
        },
        "official": {
            **official,
            "selection_mode": "full_550_per_task",
            "expected_clean_per_task": 50,
            "expected_random_per_task": 500,
            "expected_total_per_task": 550,
            "sampling_mode": "all_frames",
            "task_balanced": True,
            "balanced_tasks": True,
            "domain_label": "official_clean_plus_random",
        },
        "policy": {
            "regime": "p_v1",
            "content_layer": 16,
            "input_token_count": 120,
            "input_dim": 3072,
            "queries": 8,
            "content_dim": 384,
            "attention_heads": 8,
            "adapter_count": 1,
            "action_hidden_dim": 1024,
            "gate_init_exact": 0.0,
        },
        "architecture": {
            "content_head": True,
            "gated_action_adapter": True,
            "adapter_injection": "action_encoder_output",
            "adapter_residual": "Xa_plus_tanh_gate_cross_attention",
            "mean_pool_on_policy_path": False,
        },
        "artifacts": {
            "base_checkpoint_sha256": base_sha,
            "dataset_stats_sha256": stats_sha,
            "base_lineage_manifest_sha256": lineage_sha,
            "simulator_seed_bank_manifest_sha256": _sha256(seed_bank_path),
            "p_mode_selection_manifest_sha256": _sha256(selection_path),
            "formal_protocol_lock_manifest_sha256": _sha256(formal_lock_path),
        },
        "resolved_base_checkpoint_identity": {
            "path": str(base.resolve()),
            "size_bytes": base.stat().st_size,
            "sha256": base_sha,
        },
        "resolved_artifact_identities": artifact_identities,
        "resolved_training_sequence_audit": {
            "status": "PASS",
            "official_sample_sequence_sha256": hashlib.sha256(b"official-sequence-2").hexdigest(),
            "paired_physical_state_sequence_sha256": hashlib.sha256(b"paired-sequence-2").hexdigest(),
            "matched_stream_contract_sha256": hashlib.sha256(b"matched-stream-2").hexdigest(),
        },
        "resolved_initialization": {
            "source_fp32_content_head_sha256": hashlib.sha256(b"head-init").hexdigest(),
            "training_fp32_content_head_sha256": hashlib.sha256(b"head-init").hexdigest(),
            "source_fp32_adapter_sha256": hashlib.sha256(b"gca-init").hexdigest(),
            "training_fp32_adapter_sha256": hashlib.sha256(b"gca-init").hexdigest(),
        },
        "runtime_provenance": {
            "fastwam_source": rollout_policy.audit_local_fastwam_source()
        },
    }
    if include_model_base:
        run_config["model_base_path"] = str(model_base.resolve())
    if mechanism_followup:
        pilot_bank = eval_robotwin_single._build_simulator_seed_bank(
            simulator_seed=53,
            episodes_per_task=p_mode_selection.DEV_EPISODES_PER_CELL,
            evaluator_source=evaluator_source,
            purpose="dev_selection",
        )
        seed_bank_path.write_text(json.dumps(pilot_bank), encoding="utf-8")
        seed_identity = artifact_identities["simulator_seed_bank_manifest"]
        seed_identity.update(
            {
                "path": str(seed_bank_path.resolve()),
                "size_bytes": seed_bank_path.stat().st_size,
                "mtime_ns": seed_bank_path.stat().st_mtime_ns,
                "sha256": _sha256(seed_bank_path),
            }
        )
        protocol = {
            "kind": "policy_pv2_actiondit_followup_protocol",
            "schema_version": 1,
            "status": "PASS",
            "study_classification": {
                "role": "post_hoc_actiondit_mechanism",
                "post_hoc_after_primary_results": True,
                "primary_experiment_remains_unchanged": True,
            },
            "locked_training": {
                "policy_regime": "p_v2",
                "action_dit_trainable": True,
                "pilot_training_seed": 1,
                "max_steps": 1800,
            },
            "pilot_gate": {
                "simulator_seed": 53,
                "episodes_per_task_domain": 20,
                "seed_bank_manifest_sha256": _sha256(seed_bank_path),
                "seed_bank_id": pilot_bank["simulator_seed_bank_id"],
            },
            "historical_p_mode_selection": {
                "winner": "p_v1",
                "use": "historical_context_not_treatment_selection",
                "sha256": _sha256(selection_path),
            },
        }
        mechanism_path = release_dir / "mechanism_protocol.json"
        mechanism_path.write_text(json.dumps(protocol), encoding="utf-8")
        run_config.update(
            {
                "stage": "mechanism_followup",
                "study_role": "post_hoc_actiondit_mechanism",
                "formal": False,
                "mechanism_protocol_manifest": str(mechanism_path.resolve()),
            }
        )
        run_config["training"].update({"seed": 1, "max_steps": 1800})
        run_config["policy"]["regime"] = "p_v2"
        run_config["evaluation"].update(
            {
                "simulator_seed_bank_id": pilot_bank["simulator_seed_bank_id"],
                "simulator_seed_bank_manifest": str(seed_bank_path.resolve()),
                "simulator_seed_bank_purpose": "dev_selection",
                "episodes_per_task": 20,
            }
        )
        run_config["artifacts"].update(
            {
                "simulator_seed_bank_manifest_sha256": _sha256(seed_bank_path),
                "mechanism_protocol_manifest_sha256": _sha256(mechanism_path),
            }
        )
    if control in {"c1_architecture_only", "c3_ours"}:
        locked_row = locked_configs[control][training_seed - 1]
        run_config["resolved_formal_protocol_lock"] = {
            "status": "PASS",
            "formal_protocol_lock_manifest_sha256": _sha256(formal_lock_path),
            "control": control,
            "training_seed": training_seed,
            "selected_policy_regime": "p_v1",
            "lambda_contrastive": run_config["loss"]["lambda_contrastive"],
            "protocol_projection_sha256": locked_row["protocol_projection_sha256"],
            "source_config": locked_row["source_config"],
        }
    is_c0 = control in {"c0_base", "c0_original"}
    if is_c0:
        run_config["kind"] = "policy_c0_eval_transport"
        run_config["stage"] = "control"
        run_config["p_mode_selection_manifest"] = None
        run_config["artifacts"]["p_mode_selection_manifest_sha256"] = None
        run_config["training"] = {"seed": None, "stage2_steps": 0}
        run_config["policy"] = {
            "method_modules_active": False,
            "transport_modules_installed": True,
        }
        run_config["c0_semantics"] = {
            "stage2_training": False,
            "action_expert_overlay": False,
            "head_gca_effect_on_action": "none_exact_zero_gate",
            "base_lineage_manifest_sha256": lineage_sha,
        }
    payload = {
        "schema": policy_model.POLICY_CHECKPOINT_SCHEMA,
        "schema_version": policy_model.POLICY_CHECKPOINT_VERSION,
        "regime": "p_v2" if mechanism_followup else "p_v1",
        "step": 1800 if mechanism_followup else (0 if is_c0 else 2),
        "base_checkpoint": {
            "path": str(base.resolve()),
            "size_bytes": base.stat().st_size,
            "mtime_ns": base.stat().st_mtime_ns,
            "sha256": base_sha,
        },
        "artifact_identities": artifact_identities,
        "head_config": {
            "backbone_dim": 3072,
            "embed_dim": 384,
            "num_queries": 8,
            "num_heads": 8,
        },
        "adapter_config": {
            "action_dim": 1024,
            "content_dim": 384,
            "num_heads": 8,
            "content_layer": 16,
        },
        "run_config": run_config,
        # Ensure mmap-backed metadata loading is also exercised with a storage.
        "content_head": {"placeholder": torch.ones(1)},
        "content_adapter": {"placeholder": torch.ones(1)},
    }
    checkpoint = tmp_path / "policy.pt"
    torch.save(payload, checkpoint)
    return checkpoint, model_base, stats


def _valid_load_audit(metadata: dict[str, object]) -> dict[str, object]:
    base = metadata["base_checkpoint"]
    assert isinstance(base, dict)
    artifacts = metadata["artifact_identities"]
    assert isinstance(artifacts, dict)
    required = {
        name: identity
        for name, identity in artifacts.items()
        if isinstance(identity, dict) and identity.get("required_for_rollout") is True
    }
    return {
        "base_checkpoint": dict(base),
        "release_load": {
            "path": base["path"],
            "size_bytes": base["size_bytes"],
        },
        "head_parameter_count": policy_model.EXPECTED_HEAD_PARAMETER_COUNT,
        "adapter_parameter_count": policy_model.EXPECTED_ADAPTER_PARAMETER_COUNT,
        "action_expert_overlaid": metadata["regime"] == "p_v2",
        "verified_runtime_artifacts": {
            name: {
                "kind": identity["kind"],
                "path": identity["path"],
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
            for name, identity in required.items()
        },
    }


def test_package_import_is_lazy_from_robotwin_style_pythonpath(tmp_path: Path) -> None:
    package_parent = Path(rollout_policy.__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_parent)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import policy_content_adapter; "
                "assert 'policy_content_adapter.model' not in sys.modules; "
                "import policy_content_adapter.rollout_policy; "
                "assert 'policy_content_adapter.model' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_model_base_resolution_uses_checkpoint_then_audited_layout(
    tmp_path: Path,
) -> None:
    checkpoint, model_base, _ = _compact_checkpoint(tmp_path / "declared")
    resolved, audit = rollout_policy._resolve_model_base_path(checkpoint, None)
    assert resolved == model_base.resolve()
    assert audit["resolution_source"] == "checkpoint_run_config"

    fallback_checkpoint, fallback_base, _ = _compact_checkpoint(
        tmp_path / "layout", include_model_base=False
    )
    resolved, audit = rollout_policy._resolve_model_base_path(
        fallback_checkpoint, None
    )
    assert resolved == fallback_base.resolve()
    assert audit["resolution_source"] == "base_checkpoint_layout"


def test_model_base_resolution_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    checkpoint, _, _ = _compact_checkpoint(tmp_path / "run")
    relocated = tmp_path / "relocated-components"
    relocated.mkdir()
    resolved, audit = rollout_policy._resolve_model_base_path(
        checkpoint, relocated
    )
    assert resolved == relocated.resolve()
    assert audit["resolution_source"] == "explicit_parameter"

    payload = torch.load(checkpoint, weights_only=True)
    payload["base_checkpoint"]["sha256"] = "not-a-digest"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="sha256"):
        rollout_policy._resolve_model_base_path(checkpoint, relocated)


def test_dataset_stats_binding_allows_relocation_only_with_sha256(
    tmp_path: Path,
) -> None:
    checkpoint, _, stats = _compact_checkpoint(
        tmp_path / "run", stats_identity=True
    )
    metadata = rollout_policy._read_checkpoint_provenance(checkpoint)["metadata"]
    relocated = tmp_path / "copied_stats.json"
    relocated.write_bytes(stats.read_bytes())
    audit = rollout_policy._validate_dataset_stats_binding(relocated, metadata)
    assert audit["verification"] == "sha256"
    assert audit["sha256"] == _sha256(stats)

    relocated.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        rollout_policy._validate_dataset_stats_binding(relocated, metadata)


def test_fastwam_source_binding_matches_exact_training_provenance(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _compact_checkpoint(tmp_path / "valid")
    metadata = rollout_policy._read_checkpoint_provenance(checkpoint)["metadata"]
    audit = rollout_policy._validate_fastwam_source_binding(metadata)
    expected = rollout_policy.audit_local_fastwam_source()
    assert audit["status"] == "PASS"
    assert audit["source_root"] == expected["source_root"]
    assert set(audit["files"]) == set(expected["files"])


@pytest.mark.parametrize("mutation", ["missing", "scope", "file_set", "sha256"])
def test_fastwam_source_binding_fails_closed_on_provenance_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    checkpoint, _, _ = _compact_checkpoint(tmp_path / mutation)
    metadata = rollout_policy._read_checkpoint_provenance(checkpoint)["metadata"]
    run_config = metadata["run_config"]
    assert isinstance(run_config, dict)
    provenance = run_config["runtime_provenance"]
    assert isinstance(provenance, dict)
    source = provenance["fastwam_source"]
    assert isinstance(source, dict)
    files = source["files"]
    assert isinstance(files, dict)

    if mutation == "missing":
        provenance.pop("fastwam_source")
        pattern = "lacks runtime_provenance.fastwam_source"
    elif mutation == "scope":
        source["scope"] = "selected_files_only"
        pattern = "audit scope is incomplete"
    elif mutation == "file_set":
        files.pop(next(iter(files)))
        pattern = "audited file set differs"
    else:
        identity = files[next(iter(files))]
        assert isinstance(identity, dict)
        identity["sha256"] = "0" * 64
        pattern = "SHA-256 differs"

    with pytest.raises(ValueError, match=pattern):
        rollout_policy._validate_fastwam_source_binding(metadata)


def test_compact_loader_always_verifies_base_and_validates_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _, stats = _compact_checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    metadata = rollout_policy._extract_checkpoint_metadata(payload)
    calls: list[bool] = []
    runtime = object()

    def fake_loader(
        model: nn.Module,
        path: str,
        *,
        verify_base: bool,
        verify_runtime_artifacts: bool,
        runtime_artifacts: dict[str, str],
    ):
        del model, path
        calls.append(verify_base)
        assert verify_runtime_artifacts is True
        assert runtime_artifacts == {"dataset_stats": str(stats.resolve())}
        return runtime, payload, _valid_load_audit(metadata)

    monkeypatch.setattr(
        policy_model,
        "load_policy_checkpoint_into_model",
        fake_loader,
    )
    loaded_runtime, loaded_metadata, _ = rollout_policy._load_policy_checkpoint(
        nn.Identity(), str(checkpoint), dataset_stats_path=stats
    )
    assert loaded_runtime is runtime
    assert loaded_metadata == metadata
    assert calls == [True]

    def bad_loader(
        model: nn.Module,
        path: str,
        *,
        verify_base: bool,
        verify_runtime_artifacts: bool,
        runtime_artifacts: dict[str, str],
    ):
        del model, path, verify_base, verify_runtime_artifacts, runtime_artifacts
        audit = _valid_load_audit(metadata)
        audit["head_parameter_count"] = 1
        return runtime, payload, audit

    monkeypatch.setattr(
        policy_model,
        "load_policy_checkpoint_into_model",
        bad_loader,
    )
    with pytest.raises(ValueError, match="head parameter count"):
        rollout_policy._load_policy_checkpoint(
            nn.Identity(), str(checkpoint), dataset_stats_path=stats
        )


def test_constructor_binds_component_base_before_model_instantiation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, model_base, stats = _compact_checkpoint(tmp_path)
    metadata = rollout_policy._read_checkpoint_provenance(checkpoint)["metadata"]
    observed: dict[str, object] = {}

    class FakeModel(nn.Module):
        pass

    class FakeProcessor:
        def eval(self):
            return self

        def set_normalizer_from_stats(self, value: object) -> None:
            observed["stats"] = value

    call_count = 0

    def fake_instantiate(cfg, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs:
            observed["env_during_model_init"] = os.environ.get(
                "DIFFSYNTH_MODEL_BASE_PATH"
            )
            observed["skip_pretrain"] = bool(cfg.skip_dit_load_from_pretrain)
            observed["action_pretrain"] = cfg.action_dit_pretrained_path
            return FakeModel()
        return FakeProcessor()

    runtime = SimpleNamespace(conditioner=nn.Identity())
    monkeypatch.setattr(rollout_policy, "instantiate", fake_instantiate)
    monkeypatch.setattr(
        rollout_policy,
        "_load_policy_checkpoint",
        lambda model, path, dataset_stats_path: (runtime, metadata, {}),
    )
    monkeypatch.setattr(
        rollout_policy,
        "load_dataset_stats_from_json",
        lambda path: {"loaded": path},
    )
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", "/ambient/wrong")

    policy = rollout_policy.PolicyContentAdapterRobotWinPolicy(
        model_cfg=OmegaConf.create({"load_text_encoder": False}),
        processor_cfg=OmegaConf.create({}),
        checkpoint_path=str(checkpoint),
        model_base_path=None,
        dataset_stats_path=stats,
        device="cpu",
        model_dtype=torch.float32,
        action_horizon=4,
        replan_steps=2,
        num_inference_steps=1,
        sigma_shift=None,
        seed=0,
        text_cfg_scale=1.0,
        negative_prompt="",
        rand_device="cpu",
        tiled=False,
        timing_enabled=False,
        num_video_frames=2,
    )
    assert isinstance(policy.model, FakeModel)
    assert call_count == 2
    assert observed["env_during_model_init"] == str(model_base.resolve())
    assert observed["skip_pretrain"] is True
    assert observed["action_pretrain"] is None
    assert os.environ["DIFFSYNTH_MODEL_BASE_PATH"] == "/ambient/wrong"


def test_cuda_request_never_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing an implicit CPU fallback"):
        rollout_policy._validate_rollout_device("cuda")
    assert rollout_policy._validate_rollout_device("cpu") == "cpu"


def test_eval_child_environment_and_output_path_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_base = tmp_path / "models"
    model_base.mkdir()
    monkeypatch.setenv("PYTHONPATH", "/legacy/one:/legacy/two")
    env = eval_robotwin_single._build_subprocess_environment(
        gpu_id=2,
        model_base_path=model_base,
    )
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[:2] == [
        str(eval_robotwin_single.PROJECT_ROOT),
        str(eval_robotwin_single.SRC_ROOT),
    ]
    assert entries[-2:] == ["/legacy/one", "/legacy/two"]
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["DIFFSYNTH_MODEL_BASE_PATH"] == str(model_base.resolve())

    nested = tmp_path / "do-not-drop" / "parent" / "run"
    assert eval_robotwin_single._resolve_output_dir(nested) == nested.resolve()


def test_eval_child_environment_accepts_only_matching_gpu_binding(
    tmp_path: Path,
) -> None:
    model_base = tmp_path / "models"
    model_base.mkdir()
    vulkan = tmp_path / "nvidia_icd.json"
    egl = tmp_path / "10_nvidia.json"
    vulkan.write_text("{}\n", encoding="utf-8")
    egl.write_text("{}\n", encoding="utf-8")
    binding = {
        "physical_gpu_index": 1,
        "pci_bus_id": "0000:00:02.0",
        "render_device_alias": "pci:0000:00:02.0",
        "vulkan_icd": str(vulkan),
        "egl_vendor": str(egl),
    }
    env = eval_robotwin_single._build_subprocess_environment(
        gpu_id=1,
        model_base_path=model_base,
        gpu_runtime_binding=binding,
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["ROBOTWIN_EXPECTED_GPU_PCI"] == "0000:00:02.0"
    assert env["ROBOTWIN_RENDER_DEVICE_ALIAS"] == "pci:0000:00:02.0"
    with pytest.raises(ValueError, match="differs from requested physical GPU"):
        eval_robotwin_single._build_subprocess_environment(
            gpu_id=0,
            model_base_path=model_base,
            gpu_runtime_binding=binding,
        )


def test_eval_domains_are_exactly_clean_and_official_random() -> None:
    assert eval_robotwin_single._resolve_task_configs("both") == [
        "demo_clean",
        "demo_randomized",
    ]
    assert eval_robotwin_single._resolve_task_configs("official_random") == [
        "demo_randomized"
    ]
    assert eval_robotwin_single._phase_name("demo_clean") == "clean"
    assert eval_robotwin_single._phase_name("demo_randomized") == "random"
    with pytest.raises(ValueError, match="clean/demo_clean"):
        eval_robotwin_single._resolve_task_configs("r3")
    with pytest.raises(ValueError, match="only demo_clean/demo_randomized"):
        eval_robotwin_single._phase_name("r3")


def test_checkpoint_evaluation_contract_is_metadata_bound(tmp_path: Path) -> None:
    checkpoint, _, _ = _compact_checkpoint(
        tmp_path,
        control="c2_naive_aug",
        training_seed=7,
        rollout_protocol_id="robotwin_policy_online_v2",
        episodes_per_task=100,
    )
    provenance = rollout_policy._read_checkpoint_provenance(checkpoint)
    contract = eval_robotwin_single._checkpoint_evaluation_contract(
        provenance,
        requested_tasks=["place_a2b_left", "open_microwave"],
        requested_domains=["clean", "official_random"],
        episodes_per_task=100,
    )
    assert contract["control"] == "c2_naive_aug"
    assert contract["training_seed"] == 7
    assert contract["base_checkpoint_sha256"] == provenance["metadata"][
        "base_checkpoint"
    ]["sha256"]
    assert contract["dataset_stats_sha256"] == provenance["metadata"][
        "artifact_identities"
    ]["dataset_stats"]["sha256"]
    assert contract["base_lineage_manifest_sha256"] == provenance["metadata"][
        "artifact_identities"
    ]["base_lineage_manifest"]["sha256"]
    assert contract["policy_regime"] == "p_v1"
    assert len(contract["head_init_sha256"]) == 64
    assert len(contract["gca_init_sha256"]) == 64
    assert contract["stage2_recipe_sha256"] == eval_robotwin_single._canonical_sha256(
        contract["stage2_recipe"]
    )
    assert len(contract["runtime_source_sha256"]) == 64
    assert contract["rollout_protocol_id"] == "robotwin_policy_online_v2"
    assert contract["simulator_seed_bank_id"].startswith("robotwin-seed-bank-v3:")
    assert contract["simulator_seed_bank_purpose"] == "final_test"
    assert len(contract["p_mode_selection_manifest_sha256"]) == 64
    assert (
        contract["source"]
        == "compact_checkpoint.audited_run_config_and_artifact_identities"
    )
    assert contract["safe_load"] == "torch.load(weights_only=True,mmap=True)"


def test_checkpoint_contract_accepts_disclosed_pv2_mechanism_pilot(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _compact_checkpoint(
        tmp_path,
        control="c1_architecture_only",
        training_seed=1,
        rollout_protocol_id="three_task_policy_online_v2",
        episodes_per_task=20,
        mechanism_followup=True,
    )
    provenance = rollout_policy._read_checkpoint_provenance(checkpoint)
    contract = eval_robotwin_single._checkpoint_evaluation_contract(
        provenance,
        requested_tasks=list(rollout_policy.DEFAULT_TASKS),
        requested_domains=["clean", "official_random"],
        episodes_per_task=20,
    )
    assert contract["stage"] == "mechanism_followup"
    assert contract["policy_regime"] == "p_v2"
    assert contract["training_seed"] == 1
    assert contract["checkpoint_step"] == 1800
    assert contract["simulator_seed_bank_purpose"] == "dev_selection"
    assert len(contract["mechanism_protocol_manifest_sha256"]) == 64
    assert contract["formal_evaluation_eligible"] is False


def test_c0_checkpoint_contract_has_ancestry_but_no_stage2_identity(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _compact_checkpoint(
        tmp_path,
        control="c0_original",
        training_seed=1,
    )
    provenance = rollout_policy._read_checkpoint_provenance(checkpoint)
    contract = eval_robotwin_single._checkpoint_evaluation_contract(
        provenance,
        requested_tasks=["place_a2b_left"],
        requested_domains=["clean"],
        episodes_per_task=100,
    )
    assert contract["training_seed"] is None
    assert all(
        isinstance(contract[field], str) and len(contract[field]) == 64
        for field in (
            "base_checkpoint_sha256",
            "dataset_stats_sha256",
            "base_lineage_manifest_sha256",
            "runtime_source_sha256",
        )
    )
    assert contract["policy_regime"] is None
    assert contract["head_init_sha256"] is None
    assert contract["gca_init_sha256"] is None
    assert contract["stage2_recipe"] is None
    assert contract["stage2_recipe_sha256"] is None
    assert contract["p_mode_selection_manifest_sha256"] is None


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        ("control", "control"),
        ("training_seed", "training.seed"),
        ("protocol", "rollout_protocol_id"),
        ("placeholder_protocol", "unresolved placeholder"),
        ("seed_bank", "simulator_seed_bank_id"),
        ("placeholder_seed_bank", "unresolved placeholder"),
        ("dataset_artifact", "dataset_stats"),
        ("lineage_artifact", "base_lineage_manifest"),
        ("unverified_artifact", "not audited PASS"),
        ("initialization", "resolved initialization audit"),
        ("runtime_source", "runtime_provenance"),
        ("effective_batch", "effective official global batch"),
    ],
)
def test_checkpoint_evaluation_contract_fails_closed_without_identity(
    tmp_path: Path,
    mutation: str,
    pattern: str,
) -> None:
    checkpoint, _, _ = _compact_checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    run_config = payload["run_config"]
    if mutation == "control":
        run_config.pop("control")
    elif mutation == "training_seed":
        run_config["training"].pop("seed")
    elif mutation == "protocol":
        run_config["evaluation"].pop("rollout_protocol_id")
    elif mutation == "placeholder_protocol":
        run_config["evaluation"]["rollout_protocol_id"] = "__REQUIRED__"
    elif mutation == "seed_bank":
        run_config["evaluation"].pop("simulator_seed_bank_id")
    elif mutation == "placeholder_seed_bank":
        run_config["evaluation"]["simulator_seed_bank_id"] = "__REQUIRED__"
    elif mutation == "dataset_artifact":
        payload["artifact_identities"].pop("dataset_stats")
    elif mutation == "lineage_artifact":
        payload["artifact_identities"].pop("base_lineage_manifest")
    elif mutation == "unverified_artifact":
        payload["artifact_identities"]["dataset_stats"]["verification_status"] = "UNKNOWN"
    elif mutation == "initialization":
        run_config.pop("resolved_initialization")
    elif mutation == "runtime_source":
        run_config.pop("runtime_provenance")
    else:
        run_config["training"]["effective_official_global_batch"] = 2
    torch.save(payload, checkpoint)
    provenance = rollout_policy._read_checkpoint_provenance(checkpoint)
    with pytest.raises(ValueError, match=pattern):
        eval_robotwin_single._checkpoint_evaluation_contract(
            provenance,
            requested_tasks=["place_a2b_left"],
            requested_domains=["clean"],
            episodes_per_task=100,
        )


def test_checkpoint_evaluation_contract_rejects_runtime_protocol_drift(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _compact_checkpoint(tmp_path, episodes_per_task=100)
    provenance = rollout_policy._read_checkpoint_provenance(checkpoint)
    with pytest.raises(ValueError, match="runtime episodes differ"):
        eval_robotwin_single._checkpoint_evaluation_contract(
            provenance,
            requested_tasks=["place_a2b_left"],
            requested_domains=["clean"],
            episodes_per_task=99,
        )
    with pytest.raises(ValueError, match="runtime domains are absent"):
        eval_robotwin_single._checkpoint_evaluation_contract(
            provenance,
            requested_tasks=["place_a2b_left"],
            requested_domains=["r3"],
            episodes_per_task=100,
        )


def test_simulator_seed_bank_identity_binds_seed_episodes_and_evaluator(
    tmp_path: Path,
) -> None:
    evaluator = tmp_path / "eval_policy.py"
    evaluator.write_text("# deterministic evaluator v1\n", encoding="utf-8")
    first = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=2,
        episodes_per_task=100,
        evaluator_source=evaluator,
        purpose="development_analysis",
    )
    repeated = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=2,
        episodes_per_task=100,
        evaluator_source=evaluator,
        purpose="development_analysis",
    )
    assert first == repeated
    assert first["candidate_start_seed"] == 300000
    assert first["simulator_seed_bank_id"].startswith("robotwin-seed-bank-v3:")
    assert first["purpose"] == "development_analysis"
    assert len(first["members"]) == p_mode_selection.SEED_BANK_MEMBER_COUNT
    assert eval_robotwin_single._verify_checkpoint_seed_bank(
        {
            "simulator_seed_bank_id": first["simulator_seed_bank_id"],
            "simulator_seed_bank_purpose": "development_analysis",
            "simulator_seed_bank_descriptor": first,
        },
        first,
    ) == first["simulator_seed_bank_id"]
    with pytest.raises(ValueError, match="differs from checkpoint protocol"):
        eval_robotwin_single._verify_checkpoint_seed_bank(
            {
                "simulator_seed_bank_id": "robotwin-seed-bank-v3:wrong",
                "simulator_seed_bank_purpose": "development_analysis",
                "simulator_seed_bank_descriptor": first,
            },
            first,
        )

    changed_seed = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=3,
        episodes_per_task=100,
        evaluator_source=evaluator,
        purpose="development_analysis",
    )
    changed_episodes = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=2,
        episodes_per_task=10,
        evaluator_source=evaluator,
        purpose="development_analysis",
    )
    assert changed_seed["simulator_seed_bank_id"] != first["simulator_seed_bank_id"]
    assert (
        changed_episodes["simulator_seed_bank_id"]
        != first["simulator_seed_bank_id"]
    )


def _completed_manifest_payload(
    *,
    task: str = "place_a2b_left",
    task_config: str = "demo_clean",
    settings_variant: str = "a",
) -> dict[str, object]:
    domain = eval_robotwin_single.TASK_CONFIG_TO_DOMAIN[task_config]
    phase = eval_robotwin_single._phase_name(task_config)
    rollout_settings = {
        "schema": "test.rollout_settings",
        "schema_version": 1,
        "episodes_per_task": 100,
        "rollout_protocol_id": "robotwin_policy_online_v2",
        "variant": settings_variant,
    }
    settings_sha = eval_robotwin_single._canonical_sha256(rollout_settings)
    members = list(range(500000, 500000 + p_mode_selection.SEED_BANK_MEMBER_COUNT))
    dev_members = list(range(300000, 300000 + p_mode_selection.SEED_BANK_MEMBER_COUNT))
    selection_lock_sha = hashlib.sha256(b"p-selection").hexdigest()
    formal_lock_sha = hashlib.sha256(b"formal-lock").hexdigest()
    seed_bank_payload = {
        "schema": "robotwin.sequential_expert_valid_seed_bank",
        "schema_version": 3,
        "purpose": "final_test",
        "simulator_seed": 4,
        "candidate_start_seed": 500000,
        "episodes_per_cell": 100,
        "selection": "test-selection",
        "evaluator_source_size_bytes": 1,
        "evaluator_source_sha256": "c" * 64,
        "member_count": len(members),
        "members_sha256": p_mode_selection.canonical_sha256(members),
        "members": members,
        "disjoint_from": [{
            "purpose": "dev_selection",
            "simulator_seed_bank_id": "robotwin-seed-bank-v3:" + "d" * 64,
            "member_count": len(dev_members),
            "members_sha256": p_mode_selection.canonical_sha256(dev_members),
            "members": dev_members,
        }],
        "lock_ancestry": {
            "p_mode_selection_manifest": {"path": "/locks/selection.json", "size_bytes": 1, "sha256": selection_lock_sha},
            "formal_protocol_lock_manifest": {"path": "/locks/formal.json", "size_bytes": 1, "sha256": formal_lock_sha},
        },
    }
    seed_bank_id = (
        "robotwin-seed-bank-v3:"
        + eval_robotwin_single._canonical_sha256(seed_bank_payload)
    )
    seed_bank = {
        **seed_bank_payload,
        "simulator_seed_bank_id": seed_bank_id,
        "evaluator_source_path": "/test/eval_policy.py",
        "identity_scope": "purpose_explicit_candidate_members_and_acceptance_algorithm",
    }
    stage2_recipe = {
        "schema": "policy_stage2_common_recipe_v1",
        "training": {"max_steps": 100},
    }
    fairness_identity = {
        "base_checkpoint_sha256": hashlib.sha256(b"base-2").hexdigest(),
        "dataset_stats_sha256": hashlib.sha256(b"stats-2").hexdigest(),
        "base_lineage_manifest_sha256": hashlib.sha256(b"release-lineage").hexdigest(),
        "policy_regime": "p_v1",
        "head_init_sha256": hashlib.sha256(b"head-2").hexdigest(),
        "gca_init_sha256": hashlib.sha256(b"gca-2").hexdigest(),
        "stage2_recipe_sha256": eval_robotwin_single._canonical_sha256(stage2_recipe),
        "p_mode_selection_manifest_sha256": selection_lock_sha,
        "official_sample_sequence_sha256": hashlib.sha256(b"official-sequence").hexdigest(),
        "paired_physical_state_sequence_sha256": hashlib.sha256(b"paired-sequence").hexdigest(),
        "matched_stream_contract_sha256": hashlib.sha256(b"stream-contract").hexdigest(),
        "runtime_source_sha256": hashlib.sha256(b"runtime-source").hexdigest(),
    }
    record = {
        "control": "c3_ours",
        "training_seed": 2,
        **fairness_identity,
        "lambda_contrastive": 0.1,
        "paired_contrastive_gradient_enabled": True,
        "task": task,
        "domain": domain,
        "episodes": 100,
        "success_rate": 0.75,
        "rollout_protocol_id": "robotwin_policy_online_v2",
        "simulator_seed_bank_id": seed_bank_id,
    }
    return {
        "schema": eval_robotwin_single.COMPLETED_ROLLOUTS_SCHEMA,
        "schema_version": eval_robotwin_single.COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        "checkpoint_contract": {
            "control": "c3_ours",
            "training_seed": 2,
            **fairness_identity,
            "lambda_contrastive": 0.1,
            "stage2_recipe": stage2_recipe,
            "simulator_seed_bank_purpose": "final_test",
            "formal_protocol_lock_manifest_sha256": formal_lock_sha,
            "formal_evaluation_eligible": True,
        },
        "checkpoint_fairness_identity": fairness_identity,
        "simulator_seed": 4,
        "episodes_per_task": 100,
        "rollout_protocol_id": "robotwin_policy_online_v2",
        "rollout_settings": rollout_settings,
        "rollout_settings_sha256": settings_sha,
        "simulator_seed_bank": seed_bank,
        "simulator_seed_bank_id": seed_bank_id,
        "simulator_seed_bank_purpose": "final_test",
        "evaluation_records": [record],
        "runs": [
            {
                "task": task,
                "phase": phase,
                "task_config": task_config,
                "domain": domain,
                "episodes": 100,
                "simulator_seed": 4,
                "rollout_protocol_id": "robotwin_policy_online_v2",
                "rollout_settings_sha256": settings_sha,
                "simulator_seed_bank_id": seed_bank_id,
                "simulator_seed_bank_purpose": "final_test",
                "success_rate": 0.75,
            }
        ],
    }


def test_completed_manifest_converts_to_evaluation_protocol_records(
    tmp_path: Path,
) -> None:
    clean_payload = _completed_manifest_payload()
    random_payload = _completed_manifest_payload(
        task="open_microwave",
        task_config="demo_randomized",
    )
    clean_path = tmp_path / "clean.json"
    random_path = tmp_path / "random.json"
    clean_path.write_text(json.dumps(clean_payload), encoding="utf-8")
    random_path.write_text(json.dumps(random_payload), encoding="utf-8")

    converted = eval_robotwin_single.aggregate_completed_rollout_manifests(
        [clean_path, random_path]
    )
    assert (
        converted["schema_version"]
        == eval_robotwin_single.EVALUATION_PROTOCOL_SCHEMA_VERSION
        == evaluation_protocol.SCHEMA_VERSION
    )
    assert (
        converted["profile"]
        == eval_robotwin_single.EVALUATION_PROTOCOL_PROFILE
        == evaluation_protocol.PROFILE
        == "c1_c3_primary"
    )
    assert converted["rollout_protocol_id"] == "robotwin_policy_online_v2"
    assert (
        converted["simulator_seed_bank_id"]
        == clean_payload["simulator_seed_bank_id"]
    )
    assert converted["records"] == [
        clean_payload["evaluation_records"][0],
        random_payload["evaluation_records"][0],
    ]
    assert all(
        set(record)
        == {
            "control",
            "training_seed",
            *eval_robotwin_single.FAIRNESS_RECORD_FIELDS,
            "lambda_contrastive",
            "paired_contrastive_gradient_enabled",
            "task",
            "domain",
            "episodes",
            "success_rate",
            "rollout_protocol_id",
            "simulator_seed_bank_id",
        }
        for record in converted["records"]
    )


def test_completed_manifest_conversion_rejects_tampering_and_mismatch(
    tmp_path: Path,
) -> None:
    tampered = _completed_manifest_payload()
    tampered["runs"][0]["domain"] = "r3"
    with pytest.raises(ValueError, match="task_config/domain/phase disagree"):
        eval_robotwin_single._records_from_completed_manifest(tampered)

    tampered_identity = _completed_manifest_payload()
    tampered_identity["checkpoint_contract"]["base_checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="checkpoint_fairness_identity differs"):
        eval_robotwin_single._records_from_completed_manifest(tampered_identity)

    tampered_recipe = _completed_manifest_payload()
    tampered_recipe["checkpoint_contract"]["stage2_recipe"]["training"][
        "max_steps"
    ] = 101
    with pytest.raises(ValueError, match="Stage-2 recipe SHA-256 differs"):
        eval_robotwin_single._records_from_completed_manifest(tampered_recipe)

    first = _completed_manifest_payload(task="place_a2b_left")
    second = _completed_manifest_payload(
        task="open_microwave",
        settings_variant="b",
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="rollout settings mismatch"):
        eval_robotwin_single.aggregate_completed_rollout_manifests(
            [first_path, second_path]
        )

    different_checkpoint = _completed_manifest_payload(task="open_microwave")
    replacement = "e" * 64
    different_checkpoint["checkpoint_contract"]["base_checkpoint_sha256"] = replacement
    different_checkpoint["checkpoint_fairness_identity"][
        "base_checkpoint_sha256"
    ] = replacement
    different_checkpoint["evaluation_records"][0][
        "base_checkpoint_sha256"
    ] = replacement
    third_path = tmp_path / "different_checkpoint.json"
    third_path.write_text(json.dumps(different_checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint fairness identity mismatch"):
        eval_robotwin_single.aggregate_completed_rollout_manifests(
            [first_path, third_path]
        )
