from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import release_formal_stock_rollout as stock
from experiments.robotwin.policy_content_adapter import eval_robotwin_single
from experiments.robotwin.policy_content_adapter.tests.test_rollout import (
    _completed_manifest_payload,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "kind": "file",
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _fake_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, dict]:
    formal = tmp_path / "formal"
    formal.mkdir()
    (formal / "formal_c1_c3.status").write_text(
        "DONE formal_training=true online_rollout_started=false utc=x\n",
        encoding="utf-8",
    )
    amendment_path = formal / "amendment.json"
    amendment_path.write_text("{}\n", encoding="utf-8")
    rows = []
    for seed in (1, 2, 3):
        for short, control in (
            ("c1", "c1_architecture_only"),
            ("c3", "c3_ours"),
        ):
            run = formal / f"runs/seed_{seed}/{short}"
            run.mkdir(parents=True)
            checkpoint = run / "checkpoint.pt"
            checkpoint.write_bytes(f"checkpoint-{seed}-{short}".encode())
            (run / "dataset_stats.json").write_text("{}\n", encoding="utf-8")
            identity = _identity(checkpoint)
            rows.append(
                {
                    "control": control,
                    "training_seed": seed,
                    "path": identity["path"],
                    "size_bytes": identity["size_bytes"],
                    "sha256": identity["sha256"],
                }
            )
    amendment = {
        "profile": stock.PROFILE,
        "amendment_id": "amendment-id",
        "simulator_seed": 42,
        "episodes_per_cell": 100,
        "checkpoints": rows,
        "runtime_seed_bank": {
            "candidate_start_seed": 4_300_000,
            "simulator_seed_bank_id": "runtime-bank-id",
        },
    }
    monkeypatch.setattr(
        stock,
        "validate_stock_eval_amendment",
        lambda path: (amendment, amendment_path.resolve()),
    )
    return formal, amendment_path, amendment


def test_stock_plan_is_six_checkpoint_waves_each_with_six_cells(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    formal, amendment, _ = _fake_inputs(monkeypatch, tmp_path)
    plan = stock.build_stock_rollout_plan(
        formal_root=formal,
        rollout_root=tmp_path / "rollouts",
        amendment_path=amendment,
        gpu_ids="0,1,2,4,5,6",
    )
    assert plan["evaluation_profile"] == stock.PROFILE
    assert plan["simulator_seed"] == 42
    assert plan["candidate_start_seed"] == 4_300_000
    assert plan["episode_pairing"] == "not_claimed"
    assert plan["parallelism"]["checkpoint_waves"] == 6
    assert len(plan["waves"]) == 6
    assert all(len(wave["parallel_task_domain_cells"]) == 6 for wave in plan["waves"])
    assert [
        cell["physical_gpu_index"]
        for cell in plan["waves"][0]["parallel_task_domain_cells"]
    ] == [0, 1, 2, 4, 5, 6]
    assert [
        (wave["training_seed"], wave["short_control"])
        for wave in plan["waves"]
    ] == [(1, "c1"), (1, "c3"), (2, "c1"), (2, "c3"), (3, "c1"), (3, "c3")]


def test_stock_plan_requires_six_unique_gpus() -> None:
    assert stock.normalize_gpu_ids("0,1,2,4,5,6") == (0, 1, 2, 4, 5, 6)
    with pytest.raises(stock.StockRolloutError, match="exactly six"):
        stock.normalize_gpu_ids("0,1")
    with pytest.raises(stock.StockRolloutError, match="unique"):
        stock.normalize_gpu_ids("0,1,2,2,4,5")


def test_stock_aggregate_preserves_unpaired_disclaimer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout"
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}\n", encoding="utf-8")
    amendment_identity = _identity(amendment)
    waves = []
    cell_index = 0
    for seed in (1, 2, 3):
        for short in ("c1", "c3"):
            checkpoint = tmp_path / f"{seed}_{short}.pt"
            checkpoint.write_bytes(b"checkpoint")
            cells = []
            for task in stock.TASKS:
                for task_config in stock.TASK_CONFIGS:
                    domain = stock.TASK_CONFIG_TO_DOMAIN[task_config]
                    root = rollout / f"cells/seed_{seed}/{short}/{task}/{domain}"
                    attempt = root / "attempt_1"
                    attempt.mkdir(parents=True)
                    manifest = attempt / "completed_rollouts.json"
                    manifest.write_text(
                        json.dumps(
                            {
                                "schema_version": 7,
                                "stock_protocol_amendment": amendment_identity,
                                "checkpoint": str(checkpoint.resolve()),
                                "output_dir": str(attempt.resolve()),
                                "runs": [
                                    {
                                        "task": task,
                                        "task_config": task_config,
                                        "domain": domain,
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    cells.append(
                        {
                            "cell_index": cell_index,
                            "cell_root": str(root.resolve()),
                            "checkpoint": {"path": str(checkpoint.resolve())},
                            "task": task,
                            "task_config": task_config,
                            "domain": domain,
                        }
                    )
                    cell_index += 1
            waves.append({"parallel_task_domain_cells": cells})
    plan = {
        "rollout_root": str(rollout.resolve()),
        "stock_protocol_amendment": amendment_identity,
        "stock_protocol_amendment_id": "amendment-id",
        "waves": waves,
    }
    monkeypatch.setattr(
        stock,
        "validate_stock_rollout_plan",
        lambda path, require_output_absent: {
            "status": "PASS",
            "plan": amendment_identity,
            "payload": plan,
        },
    )
    monkeypatch.setattr(stock, "_records_from_completed_manifest", lambda payload: [{}])
    monkeypatch.setattr(
        stock,
        "aggregate_completed_rollout_manifests",
        lambda paths: {
            "records": [{} for _ in range(36)],
            "evaluation_profile": stock.PROFILE,
            "stock_protocol_amendment_id": "amendment-id",
            "episode_pairing": "not_claimed",
        },
    )
    monkeypatch.setattr(
        stock,
        "audit_and_summarize",
        lambda payload, **kwargs: {
            "status": "PASS",
            "record_count": 36,
            "primary_comparison": "c3_ours_minus_c1_architecture_only",
        },
    )
    result = stock.aggregate_stock_rollouts(tmp_path / "plan.json")
    assert result["status"] == "PASS"
    summary = json.loads((rollout / "aggregate/summary.json").read_text())
    assert summary["episode_pairing"] == "not_claimed"
    assert "not episode-paired" in summary["comparison_interpretation"]


def test_main_runner_defaults_to_author_stock_and_exact_is_optional() -> None:
    root = Path(stock.__file__).resolve().parent
    dispatch = (root / "run_release_formal_rollout.sh").read_text()
    runner = (root / "run_release_formal_stock_rollout.sh").read_text()
    assert 'PROFILE="${PROFILE:-author_stock}"' in dispatch
    assert 'PROFILE}" == "author_stock"' in dispatch
    assert 'seed=42' in runner
    assert "episode_pairing=not_claimed" in runner
    assert "run_checkpoint_wave" in runner
    assert "+EVALUATION.stock_protocol_amendment" in runner


def test_stock_completed_transport_disclaims_episode_pairing_and_revalidates_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _completed_manifest_payload()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_identity = _identity(checkpoint)
    payload["checkpoint_contract"]["checkpoint_identity"] = {
        "path": checkpoint_identity["path"],
        "size_bytes": checkpoint_identity["size_bytes"],
        "mtime_ns": checkpoint.stat().st_mtime_ns,
    }
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text("{}\n", encoding="utf-8")
    amendment_identity = _identity(amendment_path)
    amendment = {
        "profile": stock.PROFILE,
        "amendment_id": "amendment-id",
        "checkpoints": [
            {
                "control": "c3_ours",
                "training_seed": 2,
                "path": checkpoint_identity["path"],
                "size_bytes": checkpoint_identity["size_bytes"],
                "sha256": checkpoint_identity["sha256"],
            }
        ],
        "runtime_seed_bank": payload["simulator_seed_bank"],
    }
    monkeypatch.setattr(
        "experiments.robotwin.policy_content_adapter.release_stock_eval_protocol.validate_stock_eval_amendment",
        lambda path: (amendment, amendment_path.resolve()),
    )
    payload.update(
        {
            "schema_version": 7,
            "evaluation_profile": stock.PROFILE,
            "stock_protocol_amendment": amendment_identity,
            "stock_protocol_amendment_id": "amendment-id",
            "episode_pairing": "not_claimed",
        }
    )
    payload["rollout_settings"].update(
        {
            "evaluation_profile": stock.PROFILE,
            "stock_protocol_amendment_id": "amendment-id",
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
        }
    )
    payload["rollout_settings_sha256"] = eval_robotwin_single._canonical_sha256(
        payload["rollout_settings"]
    )
    payload["runs"][0].update(
        {
            "rollout_settings_sha256": payload["rollout_settings_sha256"],
            "evaluation_profile": stock.PROFILE,
            "stock_protocol_amendment_id": "amendment-id",
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
            "accepted_episode_sequence_recorded": False,
        }
    )
    records = eval_robotwin_single._records_from_completed_manifest(payload)
    assert len(records) == 1
    tampered = json.loads(json.dumps(payload))
    tampered["runs"][0]["episode_pairing"] = "paired"
    with pytest.raises(ValueError, match="disclaimer"):
        eval_robotwin_single._records_from_completed_manifest(tampered)
