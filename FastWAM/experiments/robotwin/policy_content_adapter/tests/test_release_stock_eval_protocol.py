from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import release_stock_eval_protocol as stock


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "kind": "file",
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": _sha(data),
    }


def _write(path: Path, data: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def _install_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    robotwin = project / "third_party/RoboTwin"
    formal = project / "formal"
    formal.mkdir(parents=True)

    _write(
        project / "configs/sim_robotwin.yaml",
        "EVALUATION:\n"
        "  eval_num_episodes: 100\n"
        "  instruction_type: unseen\n"
        "  replan_steps: 24\n"
        "  num_inference_steps: ${eval_num_inference_steps}\n"
        "  skip_get_obs_within_replan: true\n",
    )
    _write(project / "configs/train.yaml", "eval_num_inference_steps: 10\n")
    _write(robotwin / "script/eval_policy.py", "# stock evaluator\n")
    _write(
        project / "experiments/robotwin/policy_content_adapter/pinned_eval_policy.py",
        "# pinned launcher\n",
    )
    _write(robotwin / "envs/_base_task.py", "# local base task\n")
    _write(robotwin / "task_config/_camera_config.yml", "D435: {h: 480, w: 640}\n")
    _write(
        robotwin / "task_config/_embodiment_config.yml",
        "aloha-agilex: {file_path: ./assets/embodiments/aloha-agilex/}\n",
    )
    _write(
        robotwin / "task_config/_eval_step_limit.yml",
        "\n".join(f"{task}: 400" for task in stock.TASKS) + "\n",
    )
    _write(robotwin / "assets/embodiments/aloha-agilex/config.yml", "arm: aloha\n")
    task_config = (
        "embodiment: [aloha-agilex]\n"
        "camera:\n"
        "  head_camera_type: D435\n"
        "  wrist_camera_type: D435\n"
        "  collect_head_camera: true\n"
        "  collect_wrist_camera: true\n"
    )
    for name in stock.TASK_CONFIGS:
        _write(robotwin / f"task_config/{name}.yml", task_config)
    for task in stock.TASKS:
        _write(robotwin / f"envs/{task}.py", f"# {task}\n")

    checkpoints: dict[str, dict[str, dict]] = {}
    for seed in stock.FORMAL_SEEDS:
        checkpoints[str(seed)] = {}
        for short in stock.CONTROLS:
            checkpoint = _write(
                formal / f"runs/seed_{seed}/{short}/checkpoint.pt",
                f"checkpoint {seed} {short}\n",
            )
            checkpoints[str(seed)][short] = _identity(checkpoint)

    lock = _write(formal / "manifests/formal_protocol_lock.json", '{"status":"PASS"}\n')
    original_bank = _write(
        formal / "manifests/final_test_seed_bank.json", '{"simulator_seed":47}\n'
    )
    original_id = "robotwin-seed-bank-v3:" + "a" * 64
    materialization = {
        "status": "PASS",
        "artifacts": {
            "final_test_seed_bank": _identity(original_bank),
            "final_test_seed_bank_id": original_id,
            "formal_protocol_lock": _identity(lock),
        },
    }
    materialization_path = formal / "materialization_manifest.json"
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
    posttrain = {
        "status": "PASS",
        "formal_training_complete": True,
        "online_rollout_started": False,
        "checkpoints": checkpoints,
        "prelaunch": {
            "materialization_manifest": str(materialization_path.resolve()),
            "final_test_seed_bank_id": original_id,
            "final_test_seed_bank_sha256": _identity(original_bank)["sha256"],
            "formal_protocol_lock_sha256": _identity(lock)["sha256"],
        },
    }
    (formal / "strict_posttrain_pair_audit.json").write_text(
        json.dumps(posttrain), encoding="utf-8"
    )

    original_normalized = {
        "simulator_seed": 47,
        "simulator_seed_bank_id": original_id,
        "disjoint_from": [{"simulator_seed_bank_id": "dev", "members": [1]}],
        "lock_ancestry": {"formal_protocol_lock_manifest": _identity(lock)},
    }
    runtime_bank = {
        "simulator_seed": 42,
        "candidate_start_seed": 4_300_000,
        "episodes_per_cell": 100,
        "simulator_seed_bank_id": "robotwin-seed-bank-v3:" + "b" * 64,
        "evaluator_source_sha256": _identity(robotwin / "script/eval_policy.py")["sha256"],
        "lock_ancestry": original_normalized["lock_ancestry"],
    }

    def validate_bank(value, *, expected_purpose=None):
        del expected_purpose
        if isinstance(value, dict) and value.get("simulator_seed") == 47:
            return original_normalized
        if isinstance(value, dict) and value.get("simulator_seed") == 42:
            return runtime_bank
        raise ValueError("bad bank")

    monkeypatch.setattr(stock, "validate_seed_bank_descriptor", validate_bank)
    monkeypatch.setattr(
        stock, "validate_formal_protocol_lock_manifest_payload", lambda value: value
    )
    monkeypatch.setattr(
        stock,
        "build_seed_bank_descriptor",
        lambda **kwargs: runtime_bank,
    )
    return project, formal


def test_materialize_and_validate_stock_seed42_unpaired_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, formal = _install_inputs(monkeypatch, tmp_path)
    payload, path = stock.materialize_stock_eval_amendment(
        formal_root=formal, project_root=project
    )
    assert path == formal / stock.DEFAULT_AMENDMENT_RELATIVE
    assert payload["profile"] == stock.PROFILE
    assert payload["simulator_seed"] == 42
    assert payload["runtime_seed_bank"]["candidate_start_seed"] == 4_300_000
    assert payload["episodes_per_cell"] == 100
    assert payload["episode_pairing"] == "not_claimed"
    assert payload["scope"]["total_accepted_rollouts"] == 3600
    assert len(payload["checkpoints"]) == 6
    assert [
        (row["control"], row["training_seed"]) for row in payload["checkpoints"]
    ] == [
        ("c1_architecture_only", 1),
        ("c3_ours", 1),
        ("c1_architecture_only", 2),
        ("c3_ours", 2),
        ("c1_architecture_only", 3),
        ("c3_ours", 3),
    ]
    assert payload["amendment"]["weights_mutated"] is False
    assert payload["amendment"]["accepted_members_locked_before_evaluation"] is False
    assert "NOT paired" in payload["amendment"]["transparent_statement"]
    validated, validated_path = stock.validate_stock_eval_amendment(path)
    assert validated["amendment_id"] == payload["amendment_id"]
    assert validated_path == path


def test_materialization_is_create_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, formal = _install_inputs(monkeypatch, tmp_path)
    stock.materialize_stock_eval_amendment(formal_root=formal, project_root=project)
    with pytest.raises(stock.StockEvalProtocolError, match="refusing to overwrite"):
        stock.materialize_stock_eval_amendment(formal_root=formal, project_root=project)


def test_validation_detects_checkpoint_byte_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, formal = _install_inputs(monkeypatch, tmp_path)
    _, path = stock.materialize_stock_eval_amendment(
        formal_root=formal, project_root=project
    )
    checkpoint = formal / "runs/seed_2/c3/checkpoint.pt"
    checkpoint.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(stock.StockEvalProtocolError, match="checkpoint row 3"):
        stock.validate_stock_eval_amendment(path)


def test_validation_rejects_paired_episode_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, formal = _install_inputs(monkeypatch, tmp_path)
    payload, _ = stock.materialize_stock_eval_amendment(
        formal_root=formal, project_root=project
    )
    payload["episode_pairing"] = "paired"
    tampered = formal / "manifests/tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(stock.StockEvalProtocolError, match="pairing claim"):
        stock.validate_stock_eval_amendment(tampered)
