from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import release_formal_rollout as formal
from experiments.robotwin.policy_content_adapter import eval_robotwin_single


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(path: Path, label: str = "artifact") -> dict:
    return {
        "kind": "file",
        "path": str(path.resolve()),
        "size_bytes": 1,
        "sha256": _sha(label),
    }


def _install_fake_formal_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    c3_head_mismatch_seed: int | None = None,
) -> Path:
    root = tmp_path / "formal"
    root.mkdir()
    (tmp_path / "model_base").mkdir()
    posttrain_path = root / "strict_posttrain_pair_audit.json"
    posttrain_path.write_text("{}", encoding="utf-8")
    checkpoint_rows = {
        str(seed): {
            short: _identity(
                root / f"runs/seed_{seed}/{short}/checkpoint.pt",
                f"checkpoint-{seed}-{short}",
            )
            for short in ("c1", "c3")
        }
        for seed in formal.FORMAL_SEEDS
    }
    posttrain = {
        "status": "PASS",
        "formal_training_complete": True,
        "online_rollout_started": False,
        "checkpoints": checkpoint_rows,
    }
    bank_id = "robotwin-seed-bank-v3:" + _sha("bank")
    ancestry = {
        "materialization_identity": _identity(root / "materialization_manifest.json", "materialization"),
        "final_test_seed_bank": {
            "simulator_seed": 47,
            "simulator_seed_bank_id": bank_id,
        },
        "final_test_seed_bank_identity": _identity(root / "manifests/final_test_seed_bank.json", "bank-file"),
        "formal_protocol_lock_identity": _identity(root / "manifests/formal_protocol_lock.json", "lock-file"),
    }
    realization_path = root / formal.DEFAULT_REALIZATION_BANK_RELATIVE
    realization_cells = [
        {
            "task": task,
            "task_config": task_config,
            "domain": formal.FORMAL_TASK_CONFIG_TO_DOMAIN[task_config],
            "cell_id": f"cell:{task}:{task_config}",
            "ordered_seed_instruction_sha256": _sha(
                f"sequence-{task}-{task_config}"
            ),
        }
        for task in formal.TASKS
        for task_config in formal.TASK_CONFIGS
    ]
    exact_bank = {
        "candidate_seed_bank_id": bank_id,
        "candidate_seed_bank": {"sha256": _sha("bank-file")},
        "formal_protocol_lock": {"sha256": _sha("lock-file")},
        "realization_bank_id": "realization:" + _sha("realization"),
        "cells": realization_cells,
    }

    monkeypatch.setattr(formal, "_assert_status_done", lambda _: None)
    monkeypatch.setattr(formal, "_load_formal_ancestry", lambda _: ancestry)
    monkeypatch.setattr(
        formal,
        "validate_realization_bank",
        lambda path: (exact_bank, realization_path.resolve()),
    )
    monkeypatch.setattr(
        formal,
        "_load_json",
        lambda path, label: (posttrain, posttrain_path.resolve()),
    )

    def validate_identity(declared, *, label, expected_path=None):
        assert expected_path is not None
        return {
            **dict(declared),
            "path": str(expected_path.resolve()),
        }

    monkeypatch.setattr(formal, "_validate_bound_identity", validate_identity)
    monkeypatch.setattr(
        formal,
        "_stable_file_identity",
        lambda path: _identity(Path(path), Path(path).name),
    )

    def resolve_model_base(checkpoint, explicit):
        return (tmp_path / "model_base").resolve(), {
            "policy_checkpoint": {"path": str(Path(checkpoint).resolve())}
        }

    monkeypatch.setattr(formal, "_resolve_model_base_path", resolve_model_base)

    def checkpoint_contract(provenance, **kwargs):
        path = Path(provenance["policy_checkpoint"]["path"])
        seed = int(path.parts[path.parts.index("runs") + 1].split("_")[1])
        short = path.parts[path.parts.index("runs") + 2]
        control = "c1_architecture_only" if short == "c1" else "c3_ours"
        return {
            "control": control,
            "stage": "formal",
            "training_seed": seed,
            "policy_regime": "p_v1",
            "lambda_contrastive": 0.0 if short == "c1" else 0.1,
            "checkpoint_step": 1800,
            "formal_evaluation_eligible": True,
            "declared_tasks": list(formal.TASKS),
            "declared_domains": list(formal.DOMAINS),
            "simulator_seed_bank_id": bank_id,
            "simulator_seed_bank_manifest_sha256": _sha("bank-file"),
            "formal_protocol_lock_manifest_sha256": _sha("lock-file"),
            "dataset_stats_sha256": _sha("dataset_stats.json"),
            "rollout_protocol_id": "three_task_policy_online_v2",
        }

    monkeypatch.setattr(formal, "_checkpoint_evaluation_contract", checkpoint_contract)

    def fairness(contract, *, evaluation_control):
        seed = int(contract["training_seed"])
        short = "c1" if evaluation_control == "c1_architecture_only" else "c3"
        head = _sha(f"head-{seed}")
        if short == "c3" and c3_head_mismatch_seed == seed:
            head = _sha("tampered-head")
        values = {
            "base_checkpoint_sha256": _sha("base"),
            "dataset_stats_sha256": _sha("stats"),
            "base_lineage_manifest_sha256": _sha("lineage"),
            "policy_regime": "p_v1",
            "head_init_sha256": head,
            "gca_init_sha256": _sha(f"gca-{seed}"),
            "stage2_recipe_sha256": _sha("recipe"),
            "p_mode_selection_manifest_sha256": _sha("selection"),
            "official_sample_sequence_sha256": _sha(f"official-{seed}"),
            "paired_physical_state_sequence_sha256": _sha(f"paired-{seed}"),
            "matched_stream_contract_sha256": _sha(f"stream-{seed}"),
            "runtime_source_sha256": _sha("runtime"),
        }
        assert set(formal.FAIRNESS_RECORD_FIELDS) == set(values)
        return values

    monkeypatch.setattr(formal, "_fairness_identity_from_checkpoint_contract", fairness)
    return root


def test_gpu_assignment_requires_exactly_six_unique_physical_devices() -> None:
    assert formal.normalize_gpu_ids("0,1,2,4,5,6") == (0, 1, 2, 4, 5, 6)
    with pytest.raises(formal.FormalRolloutError, match="exactly six"):
        formal.normalize_gpu_ids("0,1,2")
    with pytest.raises(formal.FormalRolloutError, match="unique"):
        formal.normalize_gpu_ids("0,1,2,2,4,5")
    with pytest.raises(formal.FormalRolloutError, match="non-negative integers"):
        formal.normalize_gpu_ids("0,1,2,3,4,x")


def test_cpu_plan_is_six_exact_cell_waves_with_six_parallel_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _install_fake_formal_inputs(monkeypatch, tmp_path)
    rollout_root = tmp_path / "rollouts"
    plan = formal.build_rollout_plan(
        formal_root=root,
        rollout_root=rollout_root,
        gpu_ids="0,1,2,4,5,6",
    )
    assert plan["status"] == "PASS"
    assert plan["stage2_training_seeds"] == [1, 2, 3]
    assert plan["episodes_per_cell"] == 100
    assert plan["expected_record_count"] == 36
    assert plan["parallelism"]["waves"] == 6
    assert plan["parallelism"]["workers_per_wave"] == 6
    assert [job["physical_gpu_index"] for job in plan["candidates"]] == [0, 1, 2, 4, 5, 6]
    assert [
        (job["training_seed"], job["short_control"])
        for job in plan["candidates"]
    ] == [
        (1, "c1"),
        (1, "c3"),
        (2, "c1"),
        (2, "c3"),
        (3, "c1"),
        (3, "c3"),
    ]
    assert len(plan["waves"]) == 6
    assert all(len(wave["parallel_cells"]) == 6 for wave in plan["waves"])
    assert all(job["checkpoint"]["sha256"] for job in plan["candidates"])
    assert plan["formal_episode_realization_bank_id"].startswith("realization:")


def test_cpu_plan_refuses_existing_rollout_root_before_other_work(tmp_path: Path) -> None:
    formal_root = tmp_path / "formal"
    rollout_root = tmp_path / "rollout"
    formal_root.mkdir()
    rollout_root.mkdir()
    with pytest.raises(formal.FormalRolloutError, match="refusing to reuse"):
        formal.build_rollout_plan(
            formal_root=formal_root,
            rollout_root=rollout_root,
            gpu_ids="0,1,2,4,5,6",
        )


def test_cpu_plan_rejects_c1_c3_fairness_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _install_fake_formal_inputs(
        monkeypatch,
        tmp_path,
        c3_head_mismatch_seed=2,
    )
    with pytest.raises(formal.FormalRolloutError, match="fairness identity differs"):
        formal.build_rollout_plan(
            formal_root=root,
            rollout_root=tmp_path / "rollouts",
            gpu_ids="0,1,2,4,5,6",
        )


def test_aggregate_requires_exact_36_cell_manifests_and_writes_create_only_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rollout_root = tmp_path / "online"
    realization_identity = _identity(tmp_path / "realization.json", "realization")
    waves = []
    cell_index = 0
    for task in formal.TASKS:
        for task_config in formal.TASK_CONFIGS:
            domain = formal.FORMAL_TASK_CONFIG_TO_DOMAIN[task_config]
            sequence_sha = _sha(f"sequence-{task}-{domain}")
            parallel_cells = []
            for seed in formal.FORMAL_SEEDS:
                for short in ("c1", "c3"):
                    cell_root = rollout_root / f"cells/{task}/{domain}/seed_{seed}/{short}"
                    output = cell_root / "attempt_1"
                    output.mkdir(parents=True)
                    checkpoint = tmp_path / f"seed_{seed}_{short}.pt"
                    checkpoint.write_bytes(b"checkpoint")
                    manifest = output / "completed_rollouts.json"
                    manifest.write_text(
                        json.dumps(
                            {
                                "schema_version": 6,
                                "formal_exact_episode_replay": True,
                                "formal_episode_realization_bank_id": "realization-bank",
                                "formal_episode_realization_bank": realization_identity,
                                "checkpoint": str(checkpoint.resolve()),
                                "output_dir": str(output.resolve()),
                                "episodes_per_task": 100,
                                "simulator_seed_bank_id": "bank",
                                "runs": [
                                    {
                                        "task": task,
                                        "task_config": task_config,
                                        "domain": domain,
                                        "formal_episode_realization_cell_id": f"cell-{task}-{domain}",
                                        "ordered_seed_instruction_sha256": sequence_sha,
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    parallel_cells.append(
                        {
                            "cell_index": cell_index,
                            "checkpoint": {"path": str(checkpoint.resolve())},
                            "cell_root": str(cell_root.resolve()),
                            "task": task,
                            "task_config": task_config,
                            "domain": domain,
                            "realization_cell_id": f"cell-{task}-{domain}",
                            "ordered_seed_instruction_sha256": sequence_sha,
                        }
                    )
                    cell_index += 1
            waves.append({"parallel_cells": parallel_cells})
    plan = {
        "rollout_root": str(rollout_root.resolve()),
        "simulator_seed_bank_id": "bank",
        "formal_episode_realization_bank_id": "realization-bank",
        "formal_episode_realization_bank": realization_identity,
        "expected_record_count": 36,
        "waves": waves,
    }
    monkeypatch.setattr(
        formal,
        "validate_rollout_plan",
        lambda path, require_output_absent: {
            "status": "PASS",
            "plan": _identity(Path(path), "plan"),
            "payload": plan,
        },
    )
    monkeypatch.setattr(
        formal,
        "aggregate_completed_rollout_manifests",
        lambda paths: {
            "records": [{} for _ in range(36)],
            "formal_exact_episode_replay": True,
            "formal_episode_realization_bank_id": "realization-bank",
        },
    )
    monkeypatch.setattr(
        eval_robotwin_single,
        "_records_from_completed_manifest",
        lambda payload: [{}],
    )
    monkeypatch.setattr(
        formal,
        "audit_and_summarize",
        lambda payload, **kwargs: {
            "status": "PASS",
            "record_count": 36,
            "primary_comparison": "c3_ours_minus_c1_architecture_only",
        },
    )
    result = formal.aggregate_formal_rollouts(tmp_path / "plan.json")
    assert result["status"] == "PASS"
    assert result["record_count"] == 36
    assert (rollout_root / "aggregate/evaluation_records.json").is_file()
    assert (rollout_root / "aggregate/summary.json").is_file()
    assert (rollout_root / "aggregate/completion_audit.json").is_file()
    with pytest.raises(formal.FormalRolloutError, match="overwrite formal aggregate"):
        formal.aggregate_formal_rollouts(tmp_path / "plan.json")


def test_shell_runner_is_cpu_safe_by_default_and_candidate_parallel() -> None:
    runner = (
        Path(formal.__file__).resolve().parent / "run_release_formal_rollout.sh"
    ).read_text(encoding="utf-8")
    assert 'PHASE="${PHASE:-prepare}"' in runner
    assert "CONFIRM_FORMAL_ROLLOUT=YES" in runner
    assert "eval_robotwin_single" in runner
    assert "preflight_all_gpus" in runner
    assert "place_a2b_left demo_clean clean" in runner
    assert "move_stapler_pad demo_randomized official_random" in runner
    assert "run_one_cell" in runner
    assert "EVALUATION.eval_num_episodes=100" in runner
    assert "EVALUATION.formal_episode_realization_bank" in runner
    assert "attempt_" in runner
    assert "aggregate --plan" in runner
