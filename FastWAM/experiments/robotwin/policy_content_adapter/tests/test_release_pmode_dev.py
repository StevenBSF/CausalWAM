from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.config_audit import load_config
from experiments.robotwin.policy_content_adapter.materialize_release_pmode_dev import (
    PModeDevMaterializationError,
    build_resolved_dev_pair,
    validate_pmode_dev_pair,
)
from experiments.robotwin.policy_content_adapter.p_mode_selection import (
    DEV_EPISODES_PER_CELL,
    build_seed_bank_descriptor,
)
from experiments.robotwin.policy_content_adapter import release_pmode_dev_audit
from experiments.robotwin.policy_content_adapter.train import (
    build_matched_c1_c3_stream_contract,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _resolved_pair(tmp_path: Path) -> tuple[dict, dict, dict]:
    evaluator = tmp_path / "eval_policy.py"
    evaluator.parent.mkdir(parents=True, exist_ok=True)
    evaluator.write_text("# locked evaluator\n", encoding="utf-8")
    bank = build_seed_bank_descriptor(
        simulator_seed=23,
        episodes_per_cell=DEV_EPISODES_PER_CELL,
        evaluator_source=evaluator,
        purpose="dev_selection",
    )
    bank_path = tmp_path / "dev_bank.json"
    bank_bytes = (json.dumps(bank, indent=2, sort_keys=True) + "\n").encode()
    bank_path.write_bytes(bank_bytes)
    p_v1, p_v2 = build_resolved_dev_pair(
        p_v1_template=load_config(CONFIG_DIR / "p_v1_dev_pilot.yaml"),
        p_v2_template=load_config(CONFIG_DIR / "p_v2_dev_pilot.yaml"),
        output_root=tmp_path / "output",
        training_seed=42,
        max_steps=100,
        official_batch_size=1,
        paired_groups_per_batch=2,
        world_size=1,
        gradient_accumulation_steps=1,
        release_paired_binding_manifest=tmp_path / "paired_binding.json",
        release_paired_binding_sha256=_sha("paired-binding"),
        paired_text_cache=tmp_path / "paired_text",
        paired_text_cache_sha256=_sha("paired-text"),
        paired_cache=tmp_path / "layer16.pt",
        paired_cache_sha256=_sha("paired-cache"),
        official_text_cache=tmp_path / "official_text",
        official_text_cache_binding_manifest=tmp_path / "official-binding.json",
        official_text_cache_binding_manifest_sha256=_sha("official-binding"),
        seed_bank_manifest=bank_path,
        seed_bank_manifest_sha256=hashlib.sha256(bank_bytes).hexdigest(),
        seed_bank_id=bank["simulator_seed_bank_id"],
    )
    return p_v1, p_v2, bank


def test_dev_pair_locks_c1_lambda0_recipe_and_one_seed_bank(tmp_path: Path) -> None:
    p_v1, p_v2, bank = _resolved_pair(tmp_path)
    audit = validate_pmode_dev_pair(p_v1, p_v2)
    assert audit["status"] == "PASS"
    assert audit["shared"]["training_seed"] == 42
    assert audit["shared"]["max_steps"] == 100
    assert audit["shared"]["official_batch_size"] == 1
    assert audit["shared"]["paired_groups_per_batch"] == 2
    assert audit["shared"]["world_size"] == 1
    assert audit["shared"]["episodes_per_task_domain"] == 20
    assert p_v1["evaluation"]["simulator_seed_bank_id"] == p_v2["evaluation"][
        "simulator_seed_bank_id"
    ] == bank["simulator_seed_bank_id"]
    assert p_v1["policy"]["freeze"]["action_dit"] is True
    assert p_v2["policy"]["freeze"]["action_dit"] is False
    assert p_v1["loss"]["lambda_contrastive"] == 0.0
    assert p_v2["loss"]["lambda_contrastive"] == 0.0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("training", "seed"), 43, "outside regime"),
        (("training", "max_steps"), 101, "outside regime"),
        (("evaluation", "simulator_seed_bank_id"), "different", "outside regime"),
        (("loss", "lambda_contrastive"), 0.1, "lambda_contrastive=0"),
    ),
)
def test_dev_pair_rejects_unfair_or_method_conditioned_changes(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    p_v1, p_v2, _ = _resolved_pair(tmp_path)
    changed = copy.deepcopy(p_v2)
    changed[path[0]][path[1]] = value
    with pytest.raises(PModeDevMaterializationError, match=message):
        validate_pmode_dev_pair(p_v1, changed)


def test_dev_seed_bank_binds_real_evaluator_bytes(tmp_path: Path) -> None:
    _, _, bank = _resolved_pair(tmp_path)
    evaluator = tmp_path / "eval_policy.py"
    assert bank["purpose"] == "dev_selection"
    assert bank["episodes_per_cell"] == DEV_EPISODES_PER_CELL
    assert bank["evaluator_source_sha256"] == hashlib.sha256(
        evaluator.read_bytes()
    ).hexdigest()
    assert bank["lock_ancestry"] == {}
    assert bank["disjoint_from"] == []


def test_dev_pair_builds_one_non_null_preselection_stream_contract(
    tmp_path: Path,
) -> None:
    p_v1, p_v2, _ = _resolved_pair(tmp_path)
    names = {
        "base_lineage_manifest",
        "release_paired_binding_manifest",
        "dataset_stats",
        "official_manifest",
        "paired_action_manifest",
        "paired_action_audit",
        "paired_state_bank",
        "paired_text_cache",
        "paired_train_cache",
        "official_text_cache_binding_manifest",
    }
    identities = {name: {"sha256": _sha(name)} for name in names}
    base = {"sha256": _sha("base")}
    first = build_matched_c1_c3_stream_contract(
        p_v1, base_identity=base, identities=identities
    )
    second = build_matched_c1_c3_stream_contract(
        p_v2, base_identity=base, identities=identities
    )
    assert first == second
    assert len(first["sha256"]) == 64
    assert first["contract"]["schema"] == (
        "policy_release_pmode_preselection_matched_stream_v1"
    )
    assert "regime" not in first["contract"]["initialization"]
    assert "p_mode_selection_manifest" not in first["contract"]["artifact_sha256"]


def _posttrain_fixture(tmp_path: Path) -> Path:
    p_v1, p_v2, _ = _resolved_pair(tmp_path)
    output = tmp_path / "output"
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "configs": {
            "p_v1": {"path": str((config_dir / "p_v1.yaml").resolve())},
            "p_v2": {"path": str((config_dir / "p_v2.yaml").resolve())},
        }
    }
    manifest_path = output / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    initialization = {
        "source_fp32_content_head_sha256": _sha("head"),
        "source_fp32_adapter_sha256": _sha("adapter"),
        "training_fp32_content_head_sha256": _sha("head"),
        "training_fp32_adapter_sha256": _sha("adapter"),
    }
    contract_body = {
        "schema": "policy_release_pmode_preselection_matched_stream_v1",
        "fixture": "shared",
    }
    contract_sha = hashlib.sha256(
        json.dumps(contract_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract = {
        "status": "PASS",
        "only_permitted_cross_control_difference": (
            "policy_regime_and_action_dit_freeze"
        ),
        "contract": contract_body,
        "sha256": contract_sha,
    }
    sequence = {
        "status": "PASS",
        "official_sample_sequence_sha256": _sha("official-sequence"),
        "paired_physical_state_sequence_sha256": _sha("paired-sequence"),
        "matched_stream_contract_sha256": contract_sha,
        "official_sample_count": 100,
        "paired_physical_state_count": 200,
    }
    for regime, config in (("p_v1", p_v1), ("p_v2", p_v2)):
        root = output / "runs" / regime
        root.mkdir(parents=True)
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(regime.encode())
        config["resolved_matched_stream_contract"] = contract
        config["resolved_training_sequence_audit"] = sequence
        (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "matched_stream_contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        (root / "training_sequence_audit.json").write_text(
            json.dumps(sequence), encoding="utf-8"
        )
        summary = {
            "regime": regime,
            "control": regime,
            "lambda_contrastive": 0.0,
            "steps": 100,
            "checkpoint": str(checkpoint.resolve()),
            "official_sample_sequence_sha256": _sha("official-sequence"),
            "paired_physical_state_sequence_sha256": _sha("paired-sequence"),
            "matched_stream_contract_sha256": contract_sha,
            "training_sequence_audit": sequence,
            "initialization": initialization,
        }
        (root / "training_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    return manifest_path


def test_posttrain_audit_requires_matched_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _posttrain_fixture(tmp_path)
    monkeypatch.setattr(
        release_pmode_dev_audit,
        "audit_materialization",
        lambda _path: {"status": "PASS"},
    )
    audit = release_pmode_dev_audit.audit_posttrain(manifest)
    assert audit["status"] == "PASS"
    assert audit["shared_sequences"]["official_sample_sequence_sha256"] == _sha(
        "official-sequence"
    )

    p_v2_summary = tmp_path / "output/runs/p_v2/training_summary.json"
    changed = json.loads(p_v2_summary.read_text())
    changed["paired_physical_state_sequence_sha256"] = _sha("different")
    p_v2_summary.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(release_pmode_dev_audit.PModeDevAuditError, match="paired_physical"):
        release_pmode_dev_audit.audit_posttrain(manifest)


def test_posttrain_audit_rejects_null_stream_contract_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _posttrain_fixture(tmp_path)
    monkeypatch.setattr(
        release_pmode_dev_audit,
        "audit_materialization",
        lambda _path: {"status": "PASS"},
    )
    summary_path = tmp_path / "output/runs/p_v1/training_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["matched_stream_contract_sha256"] = None
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        release_pmode_dev_audit.PModeDevAuditError, match="non-null SHA-256"
    ):
        release_pmode_dev_audit.audit_posttrain(manifest)
