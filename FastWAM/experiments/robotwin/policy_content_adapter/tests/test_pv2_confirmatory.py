from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import eval_robotwin_pv2_confirmatory
from experiments.robotwin.policy_content_adapter import eval_robotwin_single
from experiments.robotwin.policy_content_adapter import p_mode_selection
from experiments.robotwin.policy_content_adapter import pv2_actiondit_followup_confirmatory
from experiments.robotwin.policy_content_adapter import pv2_actiondit_followup_final
from experiments.robotwin.policy_content_adapter import pv2_followup_eval100_amendment


def _dev_summary(bank: dict) -> dict:
    return {
        "purpose": "dev_selection",
        "simulator_seed_bank_id": bank["simulator_seed_bank_id"],
        "member_count": bank["member_count"],
        "members_sha256": bank["members_sha256"],
        "members": bank["members"],
    }


def test_confirmatory_seed_bank_requires_and_proves_dev_disjointness(
    tmp_path: Path,
) -> None:
    evaluator = tmp_path / "eval_policy.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    dev = p_mode_selection.build_seed_bank_descriptor(
        simulator_seed=53,
        episodes_per_cell=100,
        evaluator_source=evaluator,
        purpose="dev_selection",
    )
    confirmatory = p_mode_selection.build_seed_bank_descriptor(
        simulator_seed=59,
        episodes_per_cell=100,
        evaluator_source=evaluator,
        purpose="confirmatory_test",
        disjoint_from=[_dev_summary(dev)],
    )
    validated = p_mode_selection.validate_seed_bank_descriptor(
        confirmatory, expected_purpose="confirmatory_test"
    )
    assert validated["simulator_seed"] == 59
    assert validated["episodes_per_cell"] == 100
    assert set(validated["members"]).isdisjoint(dev["members"])
    without_exclusion = dict(confirmatory)
    without_exclusion["disjoint_from"] = []
    payload = p_mode_selection.seed_bank_identity_payload(without_exclusion)
    without_exclusion["simulator_seed_bank_id"] = (
        p_mode_selection.SEED_BANK_ID_PREFIX
        + p_mode_selection.canonical_sha256(payload)
    )
    with pytest.raises(ValueError, match="must exclude"):
        p_mode_selection.validate_seed_bank_descriptor(without_exclusion)


def test_confirmatory_matching_checkpoint_is_exact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    amendment = {
        "checkpoints": [
            {
                "path": str(checkpoint.resolve()),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": digest,
                "control": "c1_architecture_only",
                "training_seed": 2,
                "checkpoint_step": 1800,
            }
        ]
    }
    row = pv2_actiondit_followup_confirmatory.matching_checkpoint_row(
        amendment,
        checkpoint_path=checkpoint,
        control="c1_architecture_only",
        training_seed=2,
        checkpoint_step=1800,
    )
    assert row["sha256"] == digest
    with pytest.raises(
        pv2_actiondit_followup_confirmatory.Pv2ConfirmatoryError,
        match="lacks one checkpoint",
    ):
        pv2_actiondit_followup_confirmatory.matching_checkpoint_row(
            amendment,
            checkpoint_path=checkpoint,
            control="c3_ours",
            training_seed=2,
            checkpoint_step=1800,
        )


def test_confirmatory_launcher_installs_process_local_seed59_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = {"simulator_seed": 59, "episodes_per_cell": 100, "marker": "seed59"}
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    amendment = {
        "runtime_evaluation": {
            "seed_bank": {"path": str(bank_path.resolve())}
        }
    }
    monkeypatch.setattr(
        eval_robotwin_pv2_confirmatory,
        "validate_confirmatory_amendment",
        lambda path: (amendment, Path(path)),
    )
    old_builder = eval_robotwin_single._build_simulator_seed_bank
    old_values = {
        name: getattr(pv2_followup_eval100_amendment, name)
        for name in (
            "PROFILE",
            "SIMULATOR_SEED",
            "RUNTIME_EPISODES_PER_CELL",
            "TASKS",
            "DOMAINS",
            "validate_eval100_amendment",
            "matching_checkpoint_row",
        )
    }
    try:
        result = eval_robotwin_pv2_confirmatory.install_confirmatory_profile(
            tmp_path / "amendment.json"
        )
        assert result is amendment
        assert pv2_followup_eval100_amendment.SIMULATOR_SEED == 59
        assert pv2_followup_eval100_amendment.RUNTIME_EPISODES_PER_CELL == 100
        assert eval_robotwin_single._build_simulator_seed_bank()["marker"] == "seed59"
    finally:
        eval_robotwin_single._build_simulator_seed_bank = old_builder
        for name, value in old_values.items():
            setattr(pv2_followup_eval100_amendment, name, value)


def test_confirmatory_cli_requires_one_amendment_override(tmp_path: Path) -> None:
    path = tmp_path / "amendment.json"
    path.write_text("{}", encoding="utf-8")
    assert (
        eval_robotwin_pv2_confirmatory._amendment_from_argv(
            [f"+EVALUATION.pv2_followup_eval_amendment={path}"]
        )
        == path.resolve()
    )
    with pytest.raises(
        eval_robotwin_pv2_confirmatory.ConfirmatoryLauncherError,
        match="exactly one",
    ):
        eval_robotwin_pv2_confirmatory._amendment_from_argv([])


def test_three_seed_confirmatory_aggregate_and_terminal_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "experiment"
    amendment_path = root / "manifests/confirmatory_seed59_amendment_v1.json"
    amendment_path.parent.mkdir(parents=True, exist_ok=True)
    amendment_path.write_text("{}\n", encoding="utf-8")
    amendment_identity = {
        "path": str(amendment_path.resolve()),
        "size_bytes": amendment_path.stat().st_size,
        "sha256": hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
    }
    primary_path = tmp_path / "primary.json"
    primary_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    primary_identity = {
        "path": str(primary_path.resolve()),
        "size_bytes": primary_path.stat().st_size,
        "sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps({"primary_pv1_result": {"summary": primary_identity}}),
        encoding="utf-8",
    )
    protocol_identity = {
        "path": str(protocol_path.resolve()),
        "size_bytes": protocol_path.stat().st_size,
        "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    }
    pilot_path = root / "pilot_decision.json"
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_path.write_text(
        json.dumps(
            {
                "pilot_gate_passed": True,
                "macro": {"c1": {}, "c3": {}},
                "delta": {"clean": 0.02, "official_random": 0.04},
            }
        ),
        encoding="utf-8",
    )

    checkpoint_rows = []
    controls = {"c1": "c1_architecture_only", "c3": "c3_ours"}
    for seed in (1, 2, 3):
        for short, control in controls.items():
            checkpoint = root / f"runs/seed_{seed}/{short}/checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"seed{seed}-{short}".encode())
            checkpoint_rows.append(
                {
                    "training_seed": seed,
                    "short": short,
                    "control": control,
                    "checkpoint_step": 1800,
                    "path": str(checkpoint.resolve()),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
            )
    amendment = {
        "amendment_id": "confirmatory-id",
        "mechanism_protocol": protocol_identity,
        "checkpoints": checkpoint_rows,
        "runtime_evaluation": {"seed_bank_id": "seed59-bank"},
    }
    monkeypatch.setattr(
        pv2_actiondit_followup_final,
        "validate_confirmatory_amendment",
        lambda path: (amendment, amendment_path),
    )
    (root / "confirmatory_cpu_tests.log").write_text(
        "................................ [100%]\n387 passed in 30.00s\n",
        encoding="utf-8",
    )
    cpu_audit = pv2_actiondit_followup_final.record_cpu_test_audit(
        experiment_root=root
    )
    assert cpu_audit["passed"] == 387
    fairness_by_seed = {
        seed: {
            "head_init_sha256": hashlib.sha256(f"head{seed}".encode()).hexdigest(),
            "gca_init_sha256": hashlib.sha256(f"gca{seed}".encode()).hexdigest(),
            "official_sample_sequence_sha256": hashlib.sha256(f"official{seed}".encode()).hexdigest(),
            "paired_physical_state_sequence_sha256": hashlib.sha256(f"paired{seed}".encode()).hexdigest(),
            "matched_stream_contract_sha256": hashlib.sha256(f"matched{seed}".encode()).hexdigest(),
            "stage2_recipe_sha256": hashlib.sha256(f"recipe{seed}".encode()).hexdigest(),
        }
        for seed in (1, 2, 3)
    }
    for row in checkpoint_rows:
        seed = row["training_seed"]
        short = row["short"]
        control = row["control"]
        base = 0.50 + 0.01 * seed
        rate = base + (0.05 if short == "c3" else 0.0)
        runs = [
            {
                "task": task,
                "domain": domain,
                "episodes": 100,
                "success_rate": rate,
            }
            for task in pv2_actiondit_followup_confirmatory.TASKS
            for domain in pv2_actiondit_followup_confirmatory.DOMAINS
        ]
        payload = {
            "schema": "policy_content_adapter.completed_rollouts",
            "schema_version": 8,
            "pv2_followup_eval_amendment": amendment_identity,
            "pv2_followup_eval_amendment_id": "confirmatory-id",
            "evaluation_profile": pv2_actiondit_followup_confirmatory.PROFILE,
            "episode_pairing": "not_claimed",
            "simulator_seed": 59,
            "episodes_per_task": 100,
            "simulator_seed_bank_purpose": "confirmatory_test",
            "simulator_seed_bank_id": "seed59-bank",
            "rollout_settings_sha256": "a" * 64,
            "rollout_protocol_id": "policy-protocol",
            "checkpoint": row["path"],
            "checkpoint_contract": {
                "control": control,
                "training_seed": seed,
                "stage": "mechanism_followup",
                "policy_regime": "p_v2",
                "checkpoint_step": 1800,
                "mechanism_protocol_manifest_sha256": protocol_identity["sha256"],
                **fairness_by_seed[seed],
            },
            "runs": runs,
        }
        manifest = (
            root
            / f"confirmatory_rollouts_seed59_v1/seed_{seed}/{short}/completed_rollouts.json"
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    summary = pv2_actiondit_followup_final.aggregate_confirmatory(
        experiment_root=root
    )
    assert summary["record_count"] == 36
    assert summary["total_policy_episodes"] == 3600
    assert summary["c3_minus_c1"]["macro"]["clean"]["mean"] == pytest.approx(0.05)
    terminal = pv2_actiondit_followup_final.write_terminal_deliverables(
        experiment_root=root
    )
    assert terminal["completion_audit"]["sha256"]
    assert (root / "summary.json").is_file()
    assert (root / "summary.md").is_file()
    assert (root / "completion_audit.json").is_file()
