from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.policy_content_adapter.config_audit import load_config
from experiments.robotwin.policy_content_adapter import eval_robotwin_single
from experiments.robotwin.policy_content_adapter.materialize_pv2_actiondit_followup import (
    PILOT_MAX_STEPS,
    action_dit_release_payload_audit,
    build_followup_pair,
    build_smoke_pair_from_pilot,
    validate_followup_pair,
)
from experiments.robotwin.policy_content_adapter.p_mode_selection import (
    build_seed_bank_descriptor,
)
from experiments.robotwin.policy_content_adapter import pv2_actiondit_followup_audit
from experiments.robotwin.policy_content_adapter import pv2_actiondit_followup_report
from experiments.robotwin.policy_content_adapter import pv2_followup_eval100_amendment


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "kind": "file",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_eval100_amendment(tmp_path: Path) -> tuple[dict, Path]:
    project = tmp_path / "project"
    module_root = project / "experiments/robotwin/policy_content_adapter"
    robotwin_root = project / "third_party/RoboTwin"
    evaluator = robotwin_root / "script/eval_policy.py"
    for path, content in (
        (evaluator, "# stock evaluator\n"),
        (module_root / "pinned_eval_policy.py", "# pinned\n"),
        (module_root / "eval_robotwin_single.py", "# wrapper\n"),
        (project / "configs/sim_robotwin.yaml", "EVALUATION: {}\n"),
        (project / "configs/train.yaml", "seed: 42\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    root = tmp_path / "experiment"
    protocol = {
        "kind": "policy_pv2_actiondit_followup_protocol",
        "schema_version": 1,
        "status": "PASS",
        "pilot_gate": {
            "simulator_seed": 53,
            "episodes_per_task_domain": 20,
            "official_random_macro_delta_min": 0.03,
            "clean_macro_delta_min": -0.03,
        },
    }
    protocol_path = _write_json(root / "manifests/mechanism_protocol.json", protocol)
    old_bank = build_seed_bank_descriptor(
        simulator_seed=53,
        episodes_per_cell=20,
        evaluator_source=evaluator,
        purpose="dev_selection",
    )
    old_bank_path = _write_json(root / "manifests/dev_seed53_bank.json", old_bank)
    materialization = {
        "kind": "policy_pv2_actiondit_followup_materialization",
        "schema_version": 1,
        "status": "PASS",
        "protocol": _file_identity(protocol_path),
        "pilot_seed_bank": _file_identity(old_bank_path),
    }
    _write_json(root / "materialization_manifest.json", materialization)

    runs = {}
    for short in ("c1", "c3"):
        checkpoint = root / f"runs/seed_1/{short}/checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes((short + "-checkpoint").encode())
        identity = _file_identity(checkpoint)
        identity.pop("kind")
        runs[short] = {"checkpoint": identity}
    _write_json(
        root / "pilot_posttrain_audit.json",
        {
            "status": "PASS",
            "stage": "pilot_posttrain",
            "steps_per_control": 1800,
            "runs": runs,
        },
    )
    for relative in pv2_followup_eval100_amendment.EXPECTED_PARTIAL_RESULTS:
        result = root / "pilot_rollouts" / relative
        result.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately not a parseable success rate: amendment creation must
        # hash this evidence without reading it as a scientific result.
        result.write_text("ABORTED_VALUE_MUST_NOT_BE_PARSED\n", encoding="utf-8")

    payload, amendment_path = (
        pv2_followup_eval100_amendment.materialize_eval100_amendment(
            experiment_root=root,
            project_root=project,
        )
    )
    return payload, amendment_path


def _build_pair(tmp_path: Path) -> tuple[dict, dict, dict]:
    evaluator = tmp_path / "eval_policy.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    bank = build_seed_bank_descriptor(
        simulator_seed=53,
        episodes_per_cell=20,
        evaluator_source=evaluator,
        purpose="dev_selection",
    )
    bank_path = tmp_path / "dev53.json"
    bank_bytes = (json.dumps(bank, sort_keys=True) + "\n").encode()
    bank_path.write_bytes(bank_bytes)
    c1, c3 = build_followup_pair(
        template=load_config(CONFIG_DIR / "p_v2_dev_pilot.yaml"),
        output_root=tmp_path / "out",
        mechanism_protocol_manifest=tmp_path / "protocol.json",
        mechanism_protocol_manifest_sha256=_sha("protocol"),
        action_dit_initialization_audit_sha256=_sha("action-init"),
        historical_selection_manifest=tmp_path / "selection.json",
        historical_selection_sha256=_sha("selection"),
        release_paired_binding_manifest=tmp_path / "binding.json",
        release_paired_binding_sha256=_sha("binding"),
        paired_text_cache=tmp_path / "paired-text",
        paired_text_cache_sha256=_sha("paired-text"),
        paired_cache=tmp_path / "layer16.pt",
        paired_cache_sha256=_sha("layer16"),
        official_text_cache=tmp_path / "official-text",
        official_text_cache_binding_manifest=tmp_path / "official-binding.json",
        official_text_cache_binding_manifest_sha256=_sha("official-binding"),
        seed_bank_manifest=bank_path,
        seed_bank_manifest_sha256=hashlib.sha256(bank_bytes).hexdigest(),
        seed_bank_id=bank["simulator_seed_bank_id"],
    )
    return c1, c3, bank


def test_followup_pair_locks_pv2_and_only_contrastive_treatment(tmp_path: Path) -> None:
    c1, c3, bank = _build_pair(tmp_path)
    result = validate_followup_pair(c1, c3)
    assert result["status"] == "PASS"
    assert result["only_permitted_difference"] == "contrastive_coefficient_and_gradient"
    assert c1["policy"]["regime"] == c3["policy"]["regime"] == "p_v2"
    assert c1["policy"]["freeze"]["action_dit"] is False
    assert c3["policy"]["freeze"]["action_dit"] is False
    assert c1["training"]["max_steps"] == c3["training"]["max_steps"] == PILOT_MAX_STEPS
    assert c1["loss"]["lambda_contrastive"] == 0.0
    assert c3["loss"]["lambda_contrastive"] == 0.1
    assert c1["evaluation"]["simulator_seed_bank_id"] == bank["simulator_seed_bank_id"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("optimizer", "action_dit_lr", 2e-5),
        ("training", "max_steps", 1801),
        ("training", "official_batch_size", 2),
        ("policy", "head_init_seed", 2),
    ),
)
def test_followup_pair_rejects_second_treatment_difference(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    c1, c3, _ = _build_pair(tmp_path)
    changed = copy.deepcopy(c3)
    changed[section][key] = value
    with pytest.raises(Exception):
        validate_followup_pair(c1, changed)


def test_followup_pair_rejects_frozen_actiondit(tmp_path: Path) -> None:
    c1, c3, _ = _build_pair(tmp_path)
    changed = copy.deepcopy(c3)
    changed["policy"]["freeze"]["action_dit"] = True
    with pytest.raises(Exception, match="P-v2 must train ActionDiT"):
        validate_followup_pair(c1, changed)


def test_smoke_pair_keeps_pv2_but_uses_three_steps(tmp_path: Path) -> None:
    c1, c3, _ = _build_pair(tmp_path)
    evaluator = tmp_path / "smoke_eval.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    bank = build_seed_bank_descriptor(
        simulator_seed=54,
        episodes_per_cell=1,
        evaluator_source=evaluator,
        purpose="engineering_smoke",
    )
    bank_path = tmp_path / "smoke_bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    smoke_c1, smoke_c3 = build_smoke_pair_from_pilot(
        c1,
        c3,
        output_root=tmp_path / "out",
        seed_bank_manifest=bank_path,
        seed_bank_manifest_sha256=_sha("smoke-bank"),
        seed_bank_id=bank["simulator_seed_bank_id"],
    )
    assert smoke_c1["training"]["max_steps"] == 3
    assert smoke_c3["training"]["max_steps"] == 3
    assert smoke_c1["policy"]["freeze"]["action_dit"] is False
    assert smoke_c3["policy"]["freeze"]["action_dit"] is False
    assert smoke_c1["p_mode_selection_manifest"] is None


def test_actiondit_payload_hash_is_deterministic_and_covers_824_tensors(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "release.pt"
    state = {
        f"mixtures.action.tensor_{index:03d}": torch.tensor(
            [index, index + 1], dtype=torch.float32
        )
        for index in range(824)
    }
    state["mixtures.video.unrelated"] = torch.ones(1)
    torch.save({"mot": state, "step": 7, "torch_dtype": "torch.float32"}, checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    first = action_dit_release_payload_audit(
        checkpoint, expected_checkpoint_sha256=checkpoint_sha
    )
    second = action_dit_release_payload_audit(
        checkpoint, expected_checkpoint_sha256=checkpoint_sha
    )
    assert first == second
    assert first["tensor_count"] == 824
    assert len(first["action_dit_tensor_sha256"]) == 64


def _rollout_manifest(
    path: Path,
    *,
    control: str,
    rates: dict[str, tuple[float, float]],
    amendment: dict,
    amendment_path: Path,
) -> Path:
    runs = []
    for task, (clean, random) in rates.items():
        for domain, rate in (("clean", clean), ("official_random", random)):
            runs.append(
                {
                    "task": task,
                    "domain": domain,
                    "episodes": 100,
                    "success_rate": rate,
                    "pv2_followup_eval_amendment_id": amendment["amendment_id"],
                    "superseded_partial_20_episode_results_used": False,
                }
            )
    payload = {
        "schema": "policy_content_adapter.completed_rollouts",
        "schema_version": 8,
        "checkpoint_contract": {
            "control": control,
            "stage": "mechanism_followup",
            "policy_regime": "p_v2",
            "training_seed": 1,
            "checkpoint_step": 1800,
            "mechanism_protocol_manifest_sha256": amendment["mechanism_protocol"]["sha256"],
            "official_sample_sequence_sha256": "c" * 64,
            "paired_physical_state_sequence_sha256": "d" * 64,
            "matched_stream_contract_sha256": "e" * 64,
        },
        "simulator_seed": 53,
        "episodes_per_task": 100,
        "simulator_seed_bank_purpose": "dev_selection",
        "simulator_seed_bank_id": amendment["runtime_evaluation"]["seed_bank_id"],
        "evaluation_profile": amendment["profile"],
        "pv2_followup_eval_amendment": _file_identity(amendment_path),
        "pv2_followup_eval_amendment_id": amendment["amendment_id"],
        "episode_pairing": "not_claimed",
        "runs": runs,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_eval_loader_accepts_only_the_locked_posthoc_protocol(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
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
            "seed_bank_manifest_sha256": "a" * 64,
            "seed_bank_id": "robotwin-seed-bank-v3:" + "b" * 64,
        },
        "historical_p_mode_selection": {
            "winner": "p_v1",
            "use": "historical_context_not_treatment_selection",
            "sha256": "c" * 64,
        },
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    run_config = {
        "stage": "mechanism_followup",
        "study_role": "post_hoc_actiondit_mechanism",
        "formal": False,
        "mechanism_protocol_manifest": str(protocol_path.resolve()),
    }
    artifacts = {
        "mechanism_protocol_manifest_sha256": protocol_sha,
        "p_mode_selection_manifest_sha256": "c" * 64,
    }
    loaded, digest = eval_robotwin_single._load_pv2_followup_protocol(
        run_config,
        declared_artifacts=artifacts,
        simulator_seed_bank_manifest_sha256="a" * 64,
        simulator_seed_bank_id="robotwin-seed-bank-v3:" + "b" * 64,
    )
    assert loaded == protocol
    assert digest == protocol_sha
    protocol["pilot_gate"]["simulator_seed"] = 54
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    artifacts["mechanism_protocol_manifest_sha256"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="pilot seed bank/gate"):
        eval_robotwin_single._load_pv2_followup_protocol(
            run_config,
            declared_artifacts=artifacts,
            simulator_seed_bank_manifest_sha256="a" * 64,
            simulator_seed_bank_id="robotwin-seed-bank-v3:" + "b" * 64,
        )


def test_eval100_amendment_is_create_only_and_does_not_parse_partial_values(
    tmp_path: Path,
) -> None:
    amendment, path = _build_eval100_amendment(tmp_path)
    assert amendment["runtime_evaluation"]["episodes_per_task_domain"] == 100
    assert amendment["runtime_evaluation"]["episodes_per_checkpoint"] == 600
    assert amendment["invalid_aborted_20_episode_artifacts"]["status"] == "INVALID_ABORTED_NOT_USED"
    assert amendment["invalid_aborted_20_episode_artifacts"]["result_values_parsed"] is False
    validated, resolved = pv2_followup_eval100_amendment.validate_eval100_amendment(path)
    assert resolved == path
    assert validated == amendment
    with pytest.raises(
        pv2_followup_eval100_amendment.Pv2Eval100AmendmentError,
        match="already exists",
    ):
        pv2_followup_eval100_amendment.materialize_eval100_amendment(
            experiment_root=tmp_path / "experiment",
            project_root=tmp_path / "project",
        )


def test_eval100_amendment_fails_if_aborted_partial_is_tampered(tmp_path: Path) -> None:
    _, path = _build_eval100_amendment(tmp_path)
    partial = (
        tmp_path
        / "experiment/pilot_rollouts"
        / pv2_followup_eval100_amendment.EXPECTED_PARTIAL_RESULTS[0]
    )
    partial.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(
        pv2_followup_eval100_amendment.Pv2Eval100AmendmentError,
        match="partial evidence changed",
    ):
        pv2_followup_eval100_amendment.validate_eval100_amendment(path)


def test_eval100_checkpoint_contract_requires_exact_seed_episode_and_checkpoint(
    tmp_path: Path,
) -> None:
    amendment, _ = _build_eval100_amendment(tmp_path)
    row = amendment["checkpoints"][0]
    contract = {
        "control": row["control"],
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "training_seed": 1,
        "checkpoint_step": 1800,
        "formal_evaluation_eligible": False,
        "mechanism_protocol_manifest_sha256": amendment["mechanism_protocol"]["sha256"],
        "simulator_seed_bank_id": amendment["original_evaluation"]["seed_bank_id"],
        "simulator_seed_bank_manifest_sha256": amendment["original_evaluation"]["seed_bank"]["sha256"],
    }
    accepted = eval_robotwin_single._validate_pv2_eval100_checkpoint_contract(
        amendment,
        checkpoint_contract=contract,
        checkpoint_path=row["path"],
        requested_tasks=list(pv2_followup_eval100_amendment.TASKS),
        requested_domains=list(pv2_followup_eval100_amendment.DOMAINS),
        simulator_seed=53,
        episodes_per_task=100,
    )
    assert accepted["sha256"] == row["sha256"]
    with pytest.raises(ValueError, match="requires 100 episodes"):
        eval_robotwin_single._validate_pv2_eval100_checkpoint_contract(
            amendment,
            checkpoint_contract=contract,
            checkpoint_path=row["path"],
            requested_tasks=list(pv2_followup_eval100_amendment.TASKS),
            requested_domains=list(pv2_followup_eval100_amendment.DOMAINS),
            simulator_seed=53,
            episodes_per_task=20,
        )


def test_pilot_gate_requires_random_gain_and_clean_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pv2_actiondit_followup_audit,
        "audit_trained_pair",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    amendment, amendment_path = _build_eval100_amendment(tmp_path)
    c1 = _rollout_manifest(
        tmp_path / "c1.json",
        control="c1_architecture_only",
        rates={task: (0.70, 0.60) for task in ("place_a2b_left", "open_microwave", "move_stapler_pad")},
        amendment=amendment,
        amendment_path=amendment_path,
    )
    c3 = _rollout_manifest(
        tmp_path / "c3.json",
        control="c3_ours",
        rates={task: (0.68, 0.64) for task in ("place_a2b_left", "open_microwave", "move_stapler_pad")},
        amendment=amendment,
        amendment_path=amendment_path,
    )
    result = pv2_actiondit_followup_audit.evaluate_pilot_gate(
        tmp_path / "unused.json",
        c1_rollout_manifest=c1,
        c3_rollout_manifest=c3,
        evaluation_amendment=amendment_path,
    )
    assert result["pilot_gate_passed"] is True
    assert result["next_action"].startswith("EXPAND")


def test_pilot_gate_fails_without_three_point_random_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pv2_actiondit_followup_audit,
        "audit_trained_pair",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    tasks = ("place_a2b_left", "open_microwave", "move_stapler_pad")
    amendment, amendment_path = _build_eval100_amendment(tmp_path)
    c1 = _rollout_manifest(
        tmp_path / "c1.json",
        control="c1_architecture_only",
        rates={task: (0.70, 0.60) for task in tasks},
        amendment=amendment,
        amendment_path=amendment_path,
    )
    c3 = _rollout_manifest(
        tmp_path / "c3.json",
        control="c3_ours",
        rates={task: (0.70, 0.62) for task in tasks},
        amendment=amendment,
        amendment_path=amendment_path,
    )
    result = pv2_actiondit_followup_audit.evaluate_pilot_gate(
        tmp_path / "unused.json",
        c1_rollout_manifest=c1,
        c3_rollout_manifest=c3,
        evaluation_amendment=amendment_path,
    )
    assert result["pilot_gate_passed"] is False
    assert result["next_action"].startswith("STOP")


def test_report_window_uses_first_and_last_locked_windows() -> None:
    result = pv2_actiondit_followup_report._window([float(i) for i in range(250)])
    assert result == {"first": 49.5, "last": 199.5, "n": 100}


def test_pilot_report_is_create_only_and_preserves_gate_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c1_rollout = tmp_path / "c1_rollout.json"
    c3_rollout = tmp_path / "c3_rollout.json"
    c1_rollout.write_text("{}", encoding="utf-8")
    c3_rollout.write_text("{}", encoding="utf-8")
    materialization_path = tmp_path / "materialization.json"
    materialization_path.write_text(
        json.dumps(
            {
                "configs": {
                    "pilot": {
                        "c1": {"path": str(tmp_path / "c1.yaml")},
                        "c3": {"path": str(tmp_path / "c3.yaml")},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cells = {
        short: {
            task: {"clean": 0.7, "official_random": 0.6}
            for task in ("place_a2b_left", "open_microwave", "move_stapler_pad")
        }
        for short in ("c1", "c3")
    }
    decision = {
        "schema_version": 1,
        "kind": "policy_pv2_actiondit_followup_pilot_decision",
        "status": "PASS",
        "pilot_gate_passed": False,
        "next_action": "STOP_EXPANSION_AND_REPORT_FAILURE_MECHANISM",
        "rollout_manifests": {
            "c1": {"path": str(c1_rollout)},
            "c3": {"path": str(c3_rollout)},
        },
        "evaluation_amendment": {
            "path": str((tmp_path / "eval100_amendment.json").resolve()),
            "size_bytes": 2,
            "sha256": "f" * 64,
        },
        "cells": cells,
        "macro": {
            short: {"clean": 0.7, "official_random": 0.6}
            for short in ("c1", "c3")
        },
        "delta": {"clean": 0.0, "official_random": 0.0},
        "locked_thresholds": {
            "official_random_macro_delta_min": 0.03,
            "clean_macro_delta_min": -0.03,
            "both_required": True,
        },
        "conditions": {"official_random": False, "clean": True},
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(
        pv2_actiondit_followup_report,
        "audit_materialization",
        lambda path: {"status": "PASS"},
    )
    monkeypatch.setattr(
        pv2_actiondit_followup_report,
        "evaluate_pilot_gate",
        lambda *args, **kwargs: decision,
    )
    fake_mechanism = {
        "action_loss": {"first": 0.1, "last": 0.1, "n": 100},
        "contrastive_loss_diagnostic": {"first": 1.3, "last": 1.1, "n": 100},
        "positive_minus_negative_similarity": {"first": 0.2, "last": 0.8, "n": 100},
        "action_dit_gradient_norm": {"first": 1.0, "last": 1.0, "n": 100},
        "action_dit_update": {
            "changed_fraction": 1.0,
            "deployment_visible_changed_fraction": 0.6,
            "max_abs_delta": 0.01,
            "mean_abs_delta": 0.001,
            "required_changed_strata": 8,
            "bf16_deployment_category_visibility": {
                "early": True,
                "mid": True,
                "late": True,
                "head": True,
            },
        },
        "head_gca_update": {},
        "final_gate_raw": 0.001,
        "checkpoint": {"path": "/checkpoint.pt", "size_bytes": 1, "sha256": "a" * 64},
    }
    monkeypatch.setattr(
        pv2_actiondit_followup_report,
        "_training_mechanism",
        lambda path: fake_mechanism,
    )
    output = tmp_path / "report"
    result = pv2_actiondit_followup_report.build_pilot_report(
        materialization_manifest=materialization_path,
        pilot_decision=decision_path,
        output_dir=output,
    )
    assert result["pilot"]["gate_passed"] is False
    assert (output / "pilot_summary.json").is_file()
    assert (output / "pilot_summary.md").is_file()
    assert (output / "pilot_report_audit.json").is_file()
    with pytest.raises(pv2_actiondit_followup_report.Pv2FollowupReportError, match="reuse"):
        pv2_actiondit_followup_report.build_pilot_report(
            materialization_manifest=materialization_path,
            pilot_decision=decision_path,
            output_dir=output,
        )
