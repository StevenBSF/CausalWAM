from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import pair280_eval100 as protocol
from experiments.robotwin.policy_content_adapter.eval_robotwin_pair280 import (
    _selection_validator_with_pair280_projection,
    _validate_runtime_contract,
)


def _runtime_contract(checkpoint: Path) -> tuple[dict, dict]:
    expected = {
        "mechanism_protocol_manifest_sha256": "a" * 64,
        "official_sample_sequence_sha256": "b" * 64,
        "paired_physical_state_sequence_sha256": "c" * 64,
        "matched_stream_contract_sha256": "d" * 64,
        "simulator_seed_bank_id": "bank-id",
        "simulator_seed_bank_manifest_sha256": "e" * 64,
    }
    amendment = {
        "profile": protocol.PROFILE,
        "checkpoint_contract": expected,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size,
        },
    }
    runtime = {
        **expected,
        "control": "c3_ours",
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "training_seed": 1,
        "checkpoint_step": 18_215,
        "formal_evaluation_eligible": False,
        "lambda_contrastive": 0.1,
    }
    return amendment, runtime


def test_pair280_runtime_contract_accepts_one_exact_cell(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    amendment, runtime = _runtime_contract(checkpoint)
    row = _validate_runtime_contract(
        amendment,
        checkpoint_contract=runtime,
        checkpoint_path=checkpoint,
        requested_tasks=("open_microwave",),
        requested_domains=("official_random",),
        simulator_seed=53,
        episodes_per_task=100,
    )
    assert row["path"] == str(checkpoint.resolve())


def test_pair280_runtime_contract_rejects_sequence_or_episode_drift(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    amendment, runtime = _runtime_contract(checkpoint)
    runtime["paired_physical_state_sequence_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="paired_physical_state_sequence_sha256"):
        _validate_runtime_contract(
            amendment,
            checkpoint_contract=runtime,
            checkpoint_path=checkpoint,
            requested_tasks=("place_a2b_left",),
            requested_domains=("clean",),
            simulator_seed=53,
            episodes_per_task=100,
        )
    with pytest.raises(ValueError, match="seed53 and 100 episodes"):
        _validate_runtime_contract(
            amendment,
            checkpoint_contract={**runtime, "paired_physical_state_sequence_sha256": "c" * 64},
            checkpoint_path=checkpoint,
            requested_tasks=("place_a2b_left",),
            requested_domains=("clean",),
            simulator_seed=53,
            episodes_per_task=20,
        )


def test_pair280_six_gpu_runner_locks_six_cells_and_100_episodes() -> None:
    runner = Path(protocol.__file__).with_name(
        "run_pair280_eval100_multigpu.sh"
    ).read_text(encoding="utf-8")
    assert 'GPU_IDS="${GPU_IDS:-1,2,4,5,6,7}"' in runner
    assert "EVALUATION.eval_num_episodes=100" in runner
    assert "for index in 0 1 2 3 4 5" in runner
    assert "eval_robotwin_pair280" in runner


def test_pair280_selection_projection_changes_only_three_paired_identities() -> None:
    fields = (
        "paired_state_bank_sha256",
        "paired_text_cache_sha256",
        "paired_cache_sha256",
    )
    historical = {field: chr(97 + index) * 64 for index, field in enumerate(fields)}
    effective = {field: chr(100 + index) * 64 for index, field in enumerate(fields)}
    amendment = {
        "selection_ancestry_projection": {
            "status": "PASS",
            "allowed_fields": list(fields),
            "historical_p_mode_values": historical,
            "effective_pair280_values": effective,
        }
    }

    def original(_: dict) -> dict:
        return {
            "winner": "p_v1",
            "shared_candidate_identity": {
                **historical,
                "base_checkpoint_sha256": "z" * 64,
            },
        }

    projected = _selection_validator_with_pair280_projection(
        amendment, original
    )({})
    assert projected["winner"] == "p_v1"
    assert projected["shared_candidate_identity"]["base_checkpoint_sha256"] == "z" * 64
    for field in fields:
        assert projected["shared_candidate_identity"][field] == effective[field]

    bad = json.loads(json.dumps(amendment))
    bad["selection_ancestry_projection"]["historical_p_mode_values"][fields[0]] = (
        "x" * 64
    )
    with pytest.raises(ValueError, match="historical P-mode selection ancestry"):
        _selection_validator_with_pair280_projection(bad, original)({})


def test_pair280_shard_summary_requires_all_six_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    payload = {
        "checkpoint": {"path": str(checkpoint)},
        "amendment_id": "pair280-amendment",
        "claim_boundary": {},
    }
    monkeypatch.setattr(protocol, "validate", lambda _: (payload, amendment_path))
    rollout = tmp_path / "rollout"
    for task in protocol.TASKS:
        for domain in protocol.DOMAINS:
            path = rollout / "cells" / task / domain / "completed_rollouts.json"
            path.parent.mkdir(parents=True)
            value = {
                "checkpoint": str(checkpoint),
                "evaluation_profile": protocol.PROFILE,
                "pv2_followup_eval_amendment_id": "pair280-amendment",
                "rollout_settings_sha256": "a" * 64,
                "runs": [
                    {
                        "task": task,
                        "domain": domain,
                        "episodes": 100,
                        "success_rate": 0.5,
                        "physical_gpu_index": 1,
                    }
                ],
            }
            path.write_text(json.dumps(value), encoding="utf-8")
    result = protocol.summarize_shards(
        amendment=amendment_path, rollout_root=rollout
    )
    assert result["status"] == "PASS"
    assert result["total_episodes"] == 600
    assert result["macro"] == {"clean": 0.5, "official_random": 0.5}
