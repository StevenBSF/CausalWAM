from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from experiments.robotwin.policy_content_adapter import stage1
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS


def test_stage1_config_is_independent_audit_only_original_joint_training() -> None:
    report = stage1.validate_stage1_config()
    assert report["status"] == "PASS"
    assert report["kind"] == "policy_stage1_native_run"
    assert report["schema_version"] == 1
    assert report["default_launches_training"] is False
    assert report["artifacts_status"] == "NOT_REQUESTED"
    assert report["selection"] == {
        "mode": "full_550_per_task",
        "tasks": list(OFFICIAL_TASKS),
        "counts_per_task": {"clean": 50, "official_random": 500, "total": 550},
    }
    original = report["original_fastwam"]
    assert original["model_factory"] == "fastwam.runtime.create_fastwam"
    assert original["num_epochs"] == 5
    assert original["loss"] == {
        "formula": "lambda_video * L_video + lambda_action * L_action",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
    }
    assert original["native_sequence"]["future_actions"] == 32
    assert report["formal_replication_seeds"] == [1, 2, 3]
    assert report["execution"] == {
        "local_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 2,
        "world_size": 8,
        "effective_global_batch_size": 128,
        "author_reference_local_batch_size": 16,
        "author_reference_gradient_accumulation_steps": 1,
    }
    assert report["formal_memory_amendment"]["new_execution"] == {
        "local_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 2,
        "world_size": 8,
        "effective_global_batch_size": 128,
    }


def test_batch8_accum2_preserves_formal_optimizer_step_budget() -> None:
    assert stage1._expected_optimizer_steps(  # noqa: SLF001
        dataset_size=466_240,
        local_batch_size=8,
        world_size=8,
        gradient_accumulation_steps=2,
        epochs=5,
    ) == 18_215


def test_stage1_default_main_never_calls_long_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("audit-only default must never launch training")

    monkeypatch.setattr(stage1, "launch_stage1", forbidden)
    assert stage1.main([]) == 0


@pytest.mark.parametrize("seed_args", ((), ("--seed", "0"), ("--seed", "4"), ("--seed", "42")))
def test_formal_stage1_rejects_non_protocol_seed_before_validation(
    seed_args: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid formal seed must fail before artifact validation")

    monkeypatch.setattr(stage1, "validate_stage1_config", forbidden)
    assert stage1.main(["--launch-stage1-long-training", *seed_args]) == 2


def test_seed_plan_only_cache_binding_accepts_legacy_and_rejects_other_changes(
    tmp_path: Path,
) -> None:
    legacy_sha = stage1._sha256(stage1.LEGACY_SEEDS012_STAGE1_CONFIG)  # noqa: SLF001
    cache_audit = {
        "stage1_config": {
            "path": str(stage1.DEFAULT_STAGE1_CONFIG.resolve()),
            "sha256": legacy_sha,
        }
    }
    binding = stage1._verify_text_cache_stage1_config_binding(  # noqa: SLF001
        cache_audit,
        current_config_path=stage1.DEFAULT_STAGE1_CONFIG,
    )
    assert binding["status"] == "PASS"
    assert binding["mode"] == "seed_plan_only_amendment_012_to_123"
    assert binding["old_formal_replication_seeds"] == [0, 1, 2]
    assert binding["new_formal_replication_seeds"] == [1, 2, 3]
    assert binding["cache_content_changed"] is False

    changed = yaml.safe_load(stage1.DEFAULT_STAGE1_CONFIG.read_text(encoding="utf-8"))
    changed["original_fastwam"]["training"]["learning_rate"] = 0.0002
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(stage1.Stage1ProtocolError, match="beyond the formal seed plan"):
        stage1._verify_text_cache_stage1_config_binding(  # noqa: SLF001
            cache_audit,
            current_config_path=changed_path,
        )


def test_common_model_initialization_seed_is_deterministic_and_seed_specific() -> None:
    def sample(seed: int):
        contract = stage1._seed_model_initialization(seed)  # noqa: SLF001
        return (
            contract,
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )

    first = sample(1)
    second = sample(1)
    third = sample(2)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert torch.equal(first[3], second[3])
    assert not torch.equal(first[3], third[3])
    assert first[0]["rank_offset_during_model_construction"] is False


def test_stage1_launch_rejects_unresolved_full550_stats_sha() -> None:
    with pytest.raises(stage1.Stage1ProtocolError, match="stats"):
        stage1.validate_stage1_config(
            require_artifacts=True,
            text_cache_override=Path("/tmp/does-not-matter"),
            training_seed_override=1,
        )


def test_stats_preparation_proves_native_loader_was_narrowed_to_exact_1650(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_counts = {
        task: {"clean": 50, "official_random": 500} for task in OFFICIAL_TASKS
    }

    class FakeSelection:
        manifest_sha256 = "a" * 64

        def as_provenance(self):
            return {
                "selection_mode": "full_550_per_task",
                "loaded_episode_count": 1_650,
                "loaded_episode_counts_by_task_domain": expected_counts,
            }

    captured = {}

    def fake_instantiate(cfg, **kwargs):
        del cfg
        captured.update(kwargs)
        from fastwam.utils import misc

        generated = Path(misc.get_work_dir()) / "dataset_stats.json"
        generated.write_text(json.dumps({"num_episodes": 1650}) + "\n", encoding="utf-8")
        return SimpleNamespace(_official_explicit_episode_selection=FakeSelection())

    monkeypatch.setattr(stage1, "instantiate_official_dataset", fake_instantiate)
    monkeypatch.setattr(stage1, "_compose_original_training_config", lambda: object())
    stats_path = tmp_path / "full550_stats.json"
    audit = {
        "artifacts": {
            "dataset_root": str(tmp_path / "official"),
            "official_manifest": str(tmp_path / "manifest.json"),
        }
    }
    report = stage1.prepare_stage1_dataset_stats(audit, stats_output=stats_path)
    assert report["status"] == "PASS"
    assert report["episode_count"] == 1_650
    assert report["counts_per_task_domain"] == expected_counts
    assert report["training_launched"] is False
    assert len(report["stats"]["sha256"]) == 64
    assert stats_path.is_file()
    assert stats_path.with_suffix(".json.audit.json").is_file()
    assert captured["dataset_stats_path"] is None
    assert captured["episode_selection_mode"] == "full_550_per_task"
    assert captured["allow_compute_dataset_stats"] is True


def test_stage1_dataset_init_uses_temporary_work_dir_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the native stats mirror into missing ``./runs``."""

    output_dir = tmp_path / "formal_seed1"
    captured: dict[str, Path] = {}

    class ExpectedStop(RuntimeError):
        pass

    def fake_instantiate(cfg, **kwargs):
        assert int(cfg.batch_size) == 8
        assert int(cfg.gradient_accumulation_steps) == 2
        del kwargs
        from fastwam.utils import misc

        work_dir = Path(misc.get_work_dir()).resolve()
        assert work_dir.is_dir()
        assert work_dir.name.startswith("fastwam-stage1-dataset-init-rank000-")
        assert not output_dir.exists()
        (work_dir / "dataset_stats.json").write_text("{}\n", encoding="utf-8")
        captured["work_dir"] = work_dir
        raise ExpectedStop

    cfg = SimpleNamespace(
        output_dir=None,
        seed=42,
        resume=None,
        model=SimpleNamespace(action_dit_pretrained_path=None),
    )
    monkeypatch.setattr(stage1, "_compose_original_training_config", lambda: cfg)
    monkeypatch.setattr(stage1, "_audit_composed_original", lambda _cfg: {})
    monkeypatch.setattr(stage1, "instantiate_official_dataset", fake_instantiate)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    audit = {
        "training_seed": 1,
        "artifacts": {
            "output_dir": str(output_dir),
            "action_dit_pretrained": {"path": str(tmp_path / "action.pt")},
            "text_embedding_cache_dir": str(tmp_path / "text-cache"),
            "dataset_root": str(tmp_path / "official"),
            "dataset_stats": {"path": str(tmp_path / "stats.json")},
            "official_manifest": str(tmp_path / "manifest.json"),
        },
    }

    with pytest.raises(ExpectedStop):
        stage1.launch_stage1(audit, config_path=stage1.DEFAULT_STAGE1_CONFIG)

    assert not output_dir.exists()
    assert not captured["work_dir"].exists()
