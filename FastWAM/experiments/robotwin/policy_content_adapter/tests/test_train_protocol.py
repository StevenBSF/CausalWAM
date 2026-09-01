from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.robotwin.policy_content_adapter import train as train_module
from experiments.robotwin.policy_content_adapter.protocol import (
    POLICY_CAMERA_NAMES,
    POLICY_PROTOCOL_ID,
    POLICY_VARIANTS,
)
from experiments.robotwin.policy_content_adapter.pair280_protocol import (
    PAIR280_ACTIVE_STEPS,
    PAIR280_CACHE_STORAGE,
    PAIR280_GROUPS,
    PAIR280_PROFILE_ID,
    paired_active_count,
)
from experiments.robotwin.policy_content_adapter.pair280_sampler import (
    PAIR280_SAMPLER_ID,
    ExactPair280GlobalBatchSampler,
    audit_global_distinct_sampler,
)
from experiments.robotwin.policy_content_adapter.train import (
    STAGE2_STEP_RNG_POLICY_ID,
    TASKS,
    PolicyTrainingModule,
    PolicyTrainingProgress,
    _distributed_dataloader_config,
    _dual_stream_cycle_audit,
    _is_formal_training_config,
    _new_cpu_generator,
    _official_loader_rng_contract,
    _pair280_paired_dataset_audit,
    _positive_action_path_coverage,
    _resolve_stage2_resume_state,
    _seed_dataloader_worker_from_torch,
    _stage2_step_rng_contract,
    _training_deliverable_status,
    build_matched_c1_c3_stream_contract,
    stage2_step_rng_seed,
    validate_run_config,
)


def test_pair280_dataset_audit_satisfies_common_four_scene_provenance() -> None:
    paired = _pair280_paired_dataset_audit(
        cache_manifest_identity={
            "path": "/immutable/pair280/cache_manifest.json",
            "size_bytes": 1,
            "sha256": "a" * 64,
        },
        physical_state_groups=PAIR280_GROUPS,
    )
    provenance = train_module.build_dual_stream_provenance(
        official={"status": "PASS"},
        paired=paired,
    )
    assert provenance["paired"]["protocol_id"] == POLICY_PROTOCOL_ID
    assert provenance["paired"]["variant_names"] == list(POLICY_VARIANTS)
    assert provenance["paired"]["r3_training_positive"] is True
    assert provenance["paired"]["supervision_mode"] == "contrastive"
    assert provenance["paired"]["physical_state_groups"] == 25_200
    assert provenance["paired"]["scene_views"] == 100_800


def test_dual_stream_cycle_audit_uses_the_live_official_iterator() -> None:
    iterator = train_module._CyclingIterator(["only-batch"])
    assert next(iterator) == "only-batch"
    assert next(iterator) == "only-batch"
    assert _dual_stream_cycle_audit(iterator) == {"official": 1, "paired": 0}


class _DistributedSamplerFixture(Dataset[int]):
    indices_by_task = {"task": tuple(range(32))}

    def __len__(self) -> int:
        return 32

    def __getitem__(self, index: int) -> int:
        return int(index)

    @staticmethod
    def physical_state_id_for_index(index: int) -> str:
        return f"state-{index:03d}"


def _config(control: str) -> dict[str, object]:
    mode = {
        "c1_architecture_only": "contrastive",
        "c2_naive_aug": "action",
        "c3_ours": "contrastive",
    }[control]
    paired = {
        "protocol_id": POLICY_PROTOCOL_ID,
        "variants": list(POLICY_VARIANTS),
        "view_count": 4,
        "r3_role": "training_positive",
        "camera_names": list(POLICY_CAMERA_NAMES),
        "camera_count": 3,
        "native_fps": 50,
        "action_steps": 32,
        "action_dim": 14,
        "temporal_resampling": "none",
        "native_action_targets": True,
        "supervision_mode": mode,
        "split": "train",
        "layer": 16,
        "cache": None,
        "action_root": None,
        "action_manifest": None,
        "action_audit": None,
        "state_bank": None,
        "text_cache_dir": None,
    }
    if mode == "action":
        paired.update(
            {
                "action_root": "/paired/root",
                "action_manifest": "/paired/manifest.json",
                "action_audit": "/paired/audit.json",
                "state_bank": "/paired/state_bank.json",
                "text_cache_dir": "/paired/text_cache",
            }
        )
    elif mode == "contrastive":
        paired["cache"] = "/paired/cache.pt"
        paired.update(
            {
                "action_root": "/paired/root",
                "action_manifest": "/paired/manifest.json",
                "action_audit": "/paired/audit.json",
                "state_bank": "/paired/state_bank.json",
                "text_cache_dir": "/paired/text_cache",
            }
        )
    return {
        "schema_version": 3,
        "formal": False,
        "control": control,
        "base_lineage_manifest": "/release/base_lineage.json",
        "release_paired_binding_manifest": "/release/paired_binding.json",
        "tasks": list(TASKS),
        "policy": {
            "regime": "p_v1",
            "content_layer": 16,
            "queries": 8,
            "content_dim": 384,
            "head_init_mode": "random",
            "head_init": None,
            "head_init_seed": 13,
            "adapter_init_seed": 13,
        },
        "loss": {
            "action": "native_flow_matching_mse",
            "video": None,
            "temperature": 0.07,
            "lambda_contrastive": 0.1 if control == "c3_ours" else 0.0,
            "lambda_paired_action": 1.0 if mode == "action" else 0.0,
        },
        "official": {
            "dataset_root": "/official/root",
            "dataset_stats": "/official/stats.json",
            "canonical_task_manifest": "/official/manifest.json",
            "selection_mode": "full_550_per_task",
            "expected_clean_per_task": 50,
            "expected_random_per_task": 500,
            "expected_total_per_task": 550,
            "sampling_mode": "all_frames",
            "balanced_tasks": True,
            "on_the_fly_text_smoke": True,
            "text_cache_dir": None,
        },
        "paired": paired,
        "optimizer": {
            "name": "adamw",
            "trainable_parameter_dtype": "fp32",
            "head_adapter_lr": 1e-4,
            "action_dit_lr": 1e-5,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "lr_scheduler": "constant",
        },
        "training": {
            "seed": 13,
            "max_steps": 3,
            "official_batch_size": 1,
            "paired_groups_per_batch": 2,
            "world_size": 1,
            "gradient_accumulation_steps": 1,
            "effective_official_global_batch": 1,
            "effective_paired_groups_per_step": 2,
            "num_workers": 0,
            "separate_stream_rng": True,
            "preserve_official_sequence_across_controls": True,
        },
    }


@pytest.mark.parametrize(
    "control", ("c1_architecture_only", "c2_naive_aug", "c3_ours")
)
def test_validate_run_config_accepts_distinct_c1_c2_c3_protocols(control: str) -> None:
    validate_run_config(_config(control))


def test_completed_formal_run_is_not_marked_not_started() -> None:
    assert _training_deliverable_status(formal=True)["formal_long_training"] == "PASS"
    assert (
        _training_deliverable_status(formal=False)["formal_long_training"]
        == "NOT_STARTED"
    )


def test_formal_training_flag_accepts_pair280_execution_contract() -> None:
    assert _is_formal_training_config(
        {"execution": {"long_formal_training": True}}
    )
    assert not _is_formal_training_config(
        {"execution": {"long_formal_training": False}}
    )
    assert _is_formal_training_config(
        {"formal": True, "execution": {"long_formal_training": True}}
    )
    with pytest.raises(RuntimeError, match="declarations disagree"):
        _is_formal_training_config(
            {"formal": False, "execution": {"long_formal_training": True}}
        )


def test_validate_run_config_rejects_legacy_r3_holdout_and_30hz() -> None:
    config = _config("c3_ours")
    config["paired"]["r3_excluded"] = True
    with pytest.raises(ValueError, match="r3_excluded"):
        validate_run_config(config)
    config = _config("c2_naive_aug")
    config["paired"]["native_fps"] = 30
    with pytest.raises(ValueError, match="native_fps"):
        validate_run_config(config)


def test_formal_training_refuses_legacy_pretrained_head() -> None:
    config = _config("c1_architecture_only")
    config["formal"] = True
    config["formal_protocol_lock_manifest"] = "/formal/lock.json"
    config["policy"].update(
        {"head_init_mode": "pretrained", "head_init": "/legacy/e2.pt"}
    )
    config["official"].update(
        {
            "domain_verified": True,
            "text_cache_dir": "/official/text",
            "on_the_fly_text_smoke": False,
        }
    )
    with pytest.raises(ValueError, match="forbids legacy"):
        validate_run_config(config)


def test_c1_requires_same_paired_cache_stream_but_zero_gradient_weight() -> None:
    config = copy.deepcopy(_config("c1_architecture_only"))
    assert config["paired"]["supervision_mode"] == "contrastive"
    assert config["loss"]["lambda_contrastive"] == 0.0
    validate_run_config(config)
    config["paired"]["cache"] = None
    with pytest.raises(ValueError, match="Layer-16"):
        validate_run_config(config)


def test_validate_run_config_locks_effective_batch_and_no_accumulation() -> None:
    config = _config("c3_ours")
    config["training"]["gradient_accumulation_steps"] = 2
    with pytest.raises(ValueError, match="exactly one"):
        validate_run_config(config)


@pytest.mark.parametrize("control", ("p_v1", "p_v2"))
def test_p_mode_selection_uses_c1_lambda_zero_objective(control: str) -> None:
    config = _config("c1_architecture_only")
    config["control"] = control
    config["selection_role"] = "c1_lambda0"
    config["policy"]["regime"] = control
    validate_run_config(config)
    config["loss"]["lambda_contrastive"] = 0.1
    with pytest.raises(ValueError, match="C1 lambda=0"):
        validate_run_config(config)


def test_engineering_method_smoke_may_exercise_contrastive_gradient() -> None:
    config = _config("c3_ours")
    config["control"] = "p_v1"
    config["selection_role"] = "engineering_method_smoke"
    config["policy"]["regime"] = "p_v1"
    validate_run_config(config)


def test_official_loader_rng_contract_is_independent_of_global_rng_progress() -> None:
    contract = _official_loader_rng_contract(17)
    training_seed = contract["training_dataloader_generator_seed"]
    first = torch.rand(8, generator=_new_cpu_generator(training_seed))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    _ = [random.random() for _ in range(100)]
    _ = np.random.rand(100)
    _ = torch.rand(100)
    second = torch.rand(8, generator=_new_cpu_generator(training_seed))
    assert torch.equal(first, second)
    assert contract["identity_dataloader_generator_seed"] != training_seed

    def worker_draws() -> tuple[float, float, float]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123456)
            _seed_dataloader_worker_from_torch(0)
            return random.random(), float(np.random.rand()), float(torch.rand(()))

    assert worker_draws() == worker_draws()
    config = _config("c3_ours")
    config["training"]["effective_paired_groups_per_step"] = 3
    with pytest.raises(ValueError, match="effective paired groups"):
        validate_run_config(config)


def test_custom_paired_batch_sampler_is_shardable_across_eight_processes() -> None:
    dataset = _DistributedSamplerFixture()
    sampler = train_module.SameTaskPhysicalStateBatchSampler(
        dataset,
        groups_per_batch=2,
        seed=17,
        batches_per_epoch=16,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    dataloader_config = _distributed_dataloader_config()
    assert dataloader_config.split_batches is False
    assert dataloader_config.even_batches is False
    assert dataloader_config.use_seedable_sampler is False

    prepared = prepare_data_loader(
        loader,
        num_processes=8,
        process_index=0,
        split_batches=dataloader_config.split_batches,
        even_batches=dataloader_config.even_batches,
        use_seedable_sampler=dataloader_config.use_seedable_sampler,
    )
    assert len(prepared) == 2
    assert len(list(prepared)) == 2

    with pytest.raises(ValueError, match="even_batches=False"):
        prepare_data_loader(
            loader,
            num_processes=8,
            process_index=0,
            split_batches=False,
            even_batches=True,
        )


class _Pair280DistributedFixture(Dataset[int]):
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []
        self.indices_by_task = {task: [] for task in TASKS}
        for task in TASKS:
            for content_id in range(30):
                trajectory = f"{task}/content_{content_id:06d}"
                for _state in range(280):
                    index = len(self.rows)
                    self.rows.append((task, trajectory))
                    self.indices_by_task[task].append(index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def trajectory_id_for_index(self, index: int) -> str:
        return self.rows[index][1]

    def physical_state_id_for_index(self, index: int) -> str:
        return f"state-{index:06d}"


def test_pair280_sampler_accelerate_shards_to_exact_active_steps() -> None:
    dataset = _Pair280DistributedFixture()
    sampler = ExactPair280GlobalBatchSampler(dataset, seed=1)
    loader = DataLoader(dataset, batch_sampler=sampler)
    config = _distributed_dataloader_config()
    assert len(sampler) == PAIR280_ACTIVE_STEPS * 8
    first_global_batches = []
    for process_index in range(8):
        prepared = prepare_data_loader(
            loader,
            num_processes=8,
            process_index=process_index,
            split_batches=config.split_batches,
            even_batches=config.even_batches,
            use_seedable_sampler=config.use_seedable_sampler,
        )
        assert len(prepared) == PAIR280_ACTIVE_STEPS
        batch = next(iter(prepared)).tolist()
        assert len(batch) == 2
        assert dataset.trajectory_id_for_index(batch[0]) != dataset.trajectory_id_for_index(batch[1])
        first_global_batches.extend(batch)
    assert len(first_global_batches) == 16
    assert len(set(first_global_batches)) == 16


def test_pair280_config_locks_exact_full_and_smoke_recipes() -> None:
    config = _config("c3_ours")
    config["policy"]["regime"] = "p_v2"
    config["paired"].update(
        {
            "sampling_profile": PAIR280_PROFILE_ID,
            "cache_format": PAIR280_CACHE_STORAGE,
            "engineering_smoke": True,
            "schedule": {
                "states_per_trajectory": 280,
                "physical_state_groups": PAIR280_GROUPS,
                "paired_epochs": 10,
                "active_steps": PAIR280_ACTIVE_STEPS,
                "total_steps": 18_215,
                "global_groups_per_active_step": 16,
                "sampler": PAIR280_SAMPLER_ID,
                "active_step_distribution": "floor_difference_v1",
            },
        }
    )
    config["training"].update(
        {
            "official_batch_size": 16,
            "paired_groups_per_batch": 2,
            "world_size": 8,
            "effective_official_global_batch": 128,
            "effective_paired_groups_per_step": 16,
            "save_every": 3,
            "save_optimizer": True,
        }
    )
    validate_run_config(config)
    formal_profile = copy.deepcopy(config)
    formal_profile["paired"]["engineering_smoke"] = False
    formal_profile["training"]["max_steps"] = 18_215
    formal_profile["training"]["save_every"] = 2_000
    validate_run_config(formal_profile)
    formal_profile["paired"]["schedule"]["active_steps"] -= 1
    with pytest.raises(ValueError, match="paired.schedule"):
        validate_run_config(formal_profile)


def test_pair280_progress_resume_counts_only_consumed_active_batches() -> None:
    progress = PolicyTrainingProgress(
        max_steps=18_215,
        requested_config_sha256="b" * 64,
        world_size=8,
        effective_official_global_batch=128,
        effective_paired_groups_per_step=16,
        paired_supervision_mode="contrastive",
        paired_schedule_profile=PAIR280_PROFILE_ID,
    )
    summary = {
        "shape": [1, 1, 1],
        "element_count": 1,
        "sum": 1.0,
        "sum_squares": 1.0,
        "token_count": 1,
        "token_l2_sum": 1.0,
        "token_l2_sum_squares": 1.0,
        "minimum": 1.0,
        "maximum": 1.0,
    }
    for step in (1, 2):
        official_ids = [f"official-{step}-{index}" for index in range(128)]
        official_tasks = [TASKS[index % len(TASKS)] for index in range(128)]
        paired_ids = [] if step == 1 else [f"paired-{index}" for index in range(16)]
        paired_tasks = [] if step == 1 else [TASKS[index % len(TASKS)] for index in range(16)]
        progress.rows.append(
            {
                "step": step,
                "official_sample_ids": ";".join(official_ids),
                "paired_physical_state_ids": ";".join(paired_ids),
            }
        )
        progress.gradient_steps.append({"step": step})
        progress.seen_official_sample_ids.extend(official_ids)
        progress.seen_official_tasks.extend(official_tasks)
        progress.seen_paired_state_ids.extend(paired_ids)
        progress.seen_paired_tasks.extend(paired_tasks)
        progress.official_distribution.add(summary, tasks=official_tasks)
        if paired_tasks:
            progress.paired_distribution.add(summary, tasks=paired_tasks)
    progress.completed_step = 2
    progress.positive_action_signal_steps = 2
    state = progress.state_dict()
    assert state["paired_schedule_profile"] == PAIR280_PROFILE_ID
    restored = PolicyTrainingProgress(
        max_steps=18_215,
        requested_config_sha256="b" * 64,
        world_size=8,
        effective_official_global_batch=128,
        effective_paired_groups_per_step=16,
        paired_supervision_mode="contrastive",
        paired_schedule_profile=PAIR280_PROFILE_ID,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def _progress_with_two_steps() -> PolicyTrainingProgress:
    progress = PolicyTrainingProgress(
        max_steps=10,
        requested_config_sha256="a" * 64,
        world_size=1,
        effective_official_global_batch=2,
        effective_paired_groups_per_step=2,
        paired_supervision_mode="contrastive",
    )
    for step in (1, 2):
        official_ids = [f"official-{step}-0", f"official-{step}-1"]
        paired_ids = [f"paired-{step}-0", f"paired-{step}-1"]
        tasks = ["open_microwave", "place_a2b_left"]
        progress.rows.append(
            {
                "step": step,
                "official_sample_ids": ";".join(official_ids),
                "paired_physical_state_ids": ";".join(paired_ids),
            }
        )
        progress.gradient_steps.append({"step": step})
        progress.seen_official_sample_ids.extend(official_ids)
        progress.seen_paired_state_ids.extend(paired_ids)
        progress.seen_official_tasks.extend(tasks)
        progress.seen_paired_tasks.extend(tasks)
        summary = {
            "shape": [2, 1, 1],
            "element_count": 2,
            "sum": 1.0,
            "sum_squares": 1.0,
            "token_count": 2,
            "token_l2_sum": 1.0,
            "token_l2_sum_squares": 1.0,
            "minimum": 0.0,
            "maximum": 1.0,
        }
        progress.official_distribution.add(summary, tasks=tasks)
        progress.paired_distribution.add(summary, tasks=tasks)
    progress.completed_step = 2
    progress.positive_action_signal_steps = 2
    return progress


def test_stage2_progress_roundtrip_restores_exact_dual_stream_history() -> None:
    source = _progress_with_two_steps()
    restored = PolicyTrainingProgress(
        max_steps=10,
        requested_config_sha256="a" * 64,
        world_size=1,
        effective_official_global_batch=2,
        effective_paired_groups_per_step=2,
        paired_supervision_mode="contrastive",
    )
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()

    tampered = source.state_dict()
    tampered["seen_official_sample_ids"][0] = "wrong"
    with pytest.raises(ValueError, match="sequence histories"):
        restored.load_state_dict(tampered)


def test_positive_action_path_coverage_allows_isolated_ddp_cancellation() -> None:
    rows = [
        {
            "action_supervision_signal_positive": True,
            "gate_gradient_norm": 0.5,
            "action_only_official_content_token_grad_norm": 0.25,
            "combined": {
                "adapter": {"gradient_norm": 0.5},
                "adapter_attention_action_only_by_construction": {
                    "gradient_norm": 0.2
                },
                "action_dit": {"gradient_norm": 0.8},
            },
        },
        {
            # A synchronized batch may cancel exactly. It still counts as a
            # positive-weight batch, but not as path-connectivity evidence.
            "action_supervision_signal_positive": True,
            "gate_gradient_norm": 0.0,
            "action_only_official_content_token_grad_norm": 0.0,
            "combined": {
                "adapter": {"gradient_norm": 0.0},
                "adapter_attention_action_only_by_construction": {
                    "gradient_norm": 0.0
                },
                "action_dit": {"gradient_norm": 0.0},
            },
        },
        {
            "action_supervision_signal_positive": False,
            "gate_gradient_norm": 0.0,
            "action_only_official_content_token_grad_norm": 0.0,
            "combined": {},
        },
    ]
    assert _positive_action_path_coverage(rows) == {
        "positive_weight_steps": 2,
        "gate_positive_steps": 1,
        "adapter_attention_positive_steps": 1,
        "official_content_token_positive_steps": 1,
        "action_dit_positive_steps": 1,
    }


def test_latest_stage2_resume_state_is_strict_and_step_bound(tmp_path: Path) -> None:
    output = tmp_path / "run"
    state_root = output / "checkpoints" / "state"
    for step in (2_000, 4_000):
        state_dir = state_root / f"step_{step:08d}"
        state_dir.mkdir(parents=True)
        (state_dir / "policy_overlay.pt").write_bytes(b"overlay")
        payload = {
            "schema": "policy_stage2_native_accelerate_state_v1",
            "status": "PASS",
            "global_step": step,
            "next_step": step + 1,
            "max_steps": 10_000,
            "world_size": 8,
            "checkpoint_interval_steps": 2_000,
            "requested_config_sha256": "b" * 64,
            "official_batches_consumed_per_rank": step,
            "paired_batches_consumed_per_rank": step,
            "accelerate_state": "model_optimizer_rng_and_registered_progress",
            "policy_overlay": "policy_overlay.pt",
        }
        (state_dir / "trainer_state.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    selected, step, _payload = _resolve_stage2_resume_state(
        output,
        "latest",
        requested_config_sha256="b" * 64,
        max_steps=10_000,
        world_size=8,
        checkpoint_interval_steps=2_000,
    )
    assert selected.name == "step_00004000"
    assert step == 4_000

    state = json.loads((selected / "trainer_state.json").read_text())
    state["world_size"] = 1
    (selected / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="world_size"):
        _resolve_stage2_resume_state(
            output,
            "latest",
            requested_config_sha256="b" * 64,
            max_steps=10_000,
            world_size=8,
            checkpoint_interval_steps=2_000,
        )


def test_native_accelerate_state_restores_model_optimizer_rng_and_progress(
    tmp_path: Path,
) -> None:
    accelerator = Accelerator(cpu=True)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    progress = _progress_with_two_steps()
    accelerator.register_for_checkpointing(progress)

    loss = model(torch.ones(2, 3)).square().mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved_weight = model.weight.detach().clone()
    torch.manual_seed(12345)
    state_dir = tmp_path / "native_state"
    accelerator.save_state(str(state_dir), safe_serialization=False)
    expected_next_random = torch.rand(4)

    with torch.no_grad():
        model.weight.zero_()
    progress.completed_step = 0
    progress.rows.clear()
    torch.manual_seed(999)
    accelerator.load_state(str(state_dir))

    assert torch.equal(model.weight.detach(), saved_weight)
    assert progress.completed_step == 2
    assert len(progress.rows) == 2
    assert torch.equal(torch.rand(4), expected_next_random)
    assert optimizer.state


def test_pair280_resume_state_binds_active_paired_batch_offset(tmp_path: Path) -> None:
    output = tmp_path / "run"
    state_dir = output / "checkpoints/state/step_00002000"
    state_dir.mkdir(parents=True)
    (state_dir / "policy_overlay.pt").write_bytes(b"overlay")
    active_batches = paired_active_count(2_000)
    assert 0 < active_batches < 2_000
    payload = {
        "schema": "policy_stage2_native_accelerate_state_v1",
        "status": "PASS",
        "global_step": 2_000,
        "next_step": 2_001,
        "max_steps": 18_215,
        "world_size": 8,
        "checkpoint_interval_steps": 2_000,
        "requested_config_sha256": "c" * 64,
        "official_batches_consumed_per_rank": 2_000,
        "paired_batches_consumed_per_rank": active_batches,
        "accelerate_state": "model_optimizer_rng_and_registered_progress",
        "policy_overlay": "policy_overlay.pt",
    }
    (state_dir / "trainer_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    selected, step, verified = _resolve_stage2_resume_state(
        output,
        "latest",
        requested_config_sha256="c" * 64,
        max_steps=18_215,
        world_size=8,
        checkpoint_interval_steps=2_000,
        paired_schedule_profile=PAIR280_PROFILE_ID,
    )
    assert selected == state_dir.resolve()
    assert step == 2_000
    assert verified["paired_batches_consumed_per_rank"] == active_batches
    payload["paired_batches_consumed_per_rank"] = 2_000
    (state_dir / "trainer_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="paired_batches_consumed_per_rank"):
        _resolve_stage2_resume_state(
            output,
            "latest",
            requested_config_sha256="c" * 64,
            max_steps=18_215,
            world_size=8,
            checkpoint_interval_steps=2_000,
            paired_schedule_profile=PAIR280_PROFILE_ID,
        )


def test_stage2_step_rng_contract_is_deterministic_and_collision_free() -> None:
    first = _stage2_step_rng_contract(1, max_steps=1800, world_size=2)
    second = _stage2_step_rng_contract(1, max_steps=1800, world_size=2)
    assert first == second
    assert first["policy_id"] == STAGE2_STEP_RNG_POLICY_ID
    assert first["actual_seed_log"] == (
        "train_log.csv:official_rng_seed/paired_rng_seed"
    )

    seeds = {
        stage2_step_rng_seed(seed, step, process_index=rank, stream=stream)
        for seed in (1, 2, 3)
        for rank in (0, 1)
        for stream in ("official", "paired")
        for step in range(1800)
    }
    assert len(seeds) == 3 * 2 * 2 * 1800


def test_stage2_step_rng_contract_rejects_out_of_range_keys() -> None:
    with pytest.raises(ValueError, match="uint32"):
        stage2_step_rng_seed(-1, 0)
    with pytest.raises(ValueError, match="step_index"):
        stage2_step_rng_seed(1, 1_000_000)
    with pytest.raises(ValueError, match="stream"):
        stage2_step_rng_seed(1, 0, stream="unknown")

    config = _config("c3_ours")
    config["training"]["max_steps"] = 1_000_001
    with pytest.raises(ValueError, match="stream capacity"):
        validate_run_config(config)


def test_matched_stream_contract_is_identical_for_c1_c3() -> None:
    c1 = _config("c1_architecture_only")
    c3 = _config("c3_ours")
    for config in (c1, c3):
        config["policy"]["head_init_seed"] = 7
        config["policy"]["adapter_init_seed"] = 7
        config["training"]["seed"] = 7
        config["paired"]["split"] = "train"
        config["p_mode_selection_manifest"] = "/selection.json"
    names = (
        "base_lineage_manifest",
        "release_paired_binding_manifest",
        "dataset_stats",
        "official_manifest",
        "paired_action_manifest",
        "paired_action_audit",
        "paired_state_bank",
        "paired_text_cache",
        "paired_train_cache",
        "p_mode_selection_manifest",
    )
    identities = {
        name: {"sha256": f"{index:064x}"}
        for index, name in enumerate(names, start=1)
    }
    base_identity = {"sha256": "f" * 64}
    c1_contract = build_matched_c1_c3_stream_contract(
        c1, base_identity=base_identity, identities=identities
    )
    c3_contract = build_matched_c1_c3_stream_contract(
        c3, base_identity=base_identity, identities=identities
    )
    assert c1_contract["sha256"] == c3_contract["sha256"]
    c3["training"]["seed"] = 8
    changed = build_matched_c1_c3_stream_contract(
        c3, base_identity=base_identity, identities=identities
    )
    assert changed["sha256"] != c1_contract["sha256"]


def test_zero_lambda_computes_paired_diagnostics_without_paired_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_weight = nn.Parameter(torch.tensor(2.0))

    class DummyConditioner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.content_weight = nn.Parameter(torch.tensor(3.0))

    model = DummyModel()
    conditioner = DummyConditioner()
    runtime = type("Runtime", (), {"conditioner": conditioner})()

    def fake_action(model, _runtime, _batch):
        return model.action_weight.square(), {
            "loss_action": 4.0,
            "official_layer16_distribution": {"count": 1},
        }

    def fake_contrastive(conditioner, _batch, *, temperature):
        assert temperature == 0.07
        return conditioner.content_weight.square(), {
            "loss_contrastive": 9.0,
            "positive_similarity": 1.0,
            "negative_similarity": 0.0,
            "positives_per_anchor": 3,
            "r3_training_positive": True,
            "paired_clean_layer16_distribution": {"count": 1},
        }

    monkeypatch.setattr(train_module, "official_action_loss", fake_action)
    monkeypatch.setattr(train_module, "paired_contrastive_loss", fake_contrastive)
    training = PolicyTrainingModule(
        model,
        runtime,
        paired_supervision_mode="contrastive",
        lambda_contrastive=0.0,
        lambda_paired_action=0.0,
        temperature=0.07,
        training_seed=1,
        process_index=0,
    )
    total, _action, contrastive, diagnostics = training({}, {})
    total.backward()
    assert contrastive.item() == 9.0
    assert diagnostics["paired_contrastive_gradient_enabled"] is False
    assert model.action_weight.grad is not None and model.action_weight.grad.item() != 0.0
    assert conditioner.content_weight.grad is not None
    assert conditioner.content_weight.grad.item() == 0.0


def test_formal_protocol_lock_matches_cycle_free_config_projection(
    tmp_path: Path,
) -> None:
    config = _config("c1_architecture_only")
    config.update(
        {
            "formal": True,
            "stage": "formal",
            "formal_protocol_lock_manifest": str(tmp_path / "lock.json"),
            "p_mode_selection_manifest": str(tmp_path / "selection.json"),
        }
    )
    config["training"]["seed"] = 1
    config["policy"]["regime"] = "p_v1"
    projection_sha = train_module.p_mode_canonical_sha256(
        train_module.formal_config_protocol_projection(config)
    )

    def identity(name: str, digest: str) -> dict[str, object]:
        return {
            "path": str((tmp_path / name).resolve()),
            "size_bytes": 1,
            "sha256": digest,
        }

    lineage_sha = "a" * 64
    selection_sha = "b" * 64
    row_source = identity("source.yaml", "c" * 64)
    rows: dict[str, list[dict[str, object]]] = {}
    for control, coefficient in (("c1_architecture_only", 0.0), ("c3_ours", 0.1)):
        rows[control] = [
            {
                "control": control,
                "training_seed": seed,
                "lambda_contrastive": coefficient,
                "source_config": row_source,
                "protocol_projection_sha256": (
                    projection_sha if control == "c1_architecture_only" and seed == 1
                    else f"{seed + (10 if control == 'c3_ours' else 0):064x}"
                ),
            }
            for seed in (1, 2, 3)
        ]
    lock = {
        "kind": "policy_release_formal_protocol_lock",
        "schema_version": 1,
        "status": "PASS",
        "stage2_training_seeds": [1, 2, 3],
        "base_lineage_manifest": identity("lineage.json", lineage_sha),
        "p_mode_selection_manifest": identity("selection.json", selection_sha),
        "formal_matrix_audit": identity("matrix.json", "d" * 64),
        "selected_policy_regime": "p_v1",
        "resolved_configs": rows,
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    report = train_module._verify_formal_protocol_lock(  # noqa: SLF001
        config,
        lock_path=lock_path,
        lock_identity={"sha256": "e" * 64},
        projection_sha256=projection_sha,
        base_lineage_identity={"sha256": lineage_sha},
        p_mode_selection_identity={"sha256": selection_sha},
    )
    assert report["status"] == "PASS"
    assert report["protocol_projection_sha256"] == projection_sha
    with pytest.raises(ValueError, match="projection differs"):
        train_module._verify_formal_protocol_lock(  # noqa: SLF001
            config,
            lock_path=lock_path,
            lock_identity={"sha256": "e" * 64},
            projection_sha256="f" * 64,
            base_lineage_identity={"sha256": lineage_sha},
            p_mode_selection_identity={"sha256": selection_sha},
        )
