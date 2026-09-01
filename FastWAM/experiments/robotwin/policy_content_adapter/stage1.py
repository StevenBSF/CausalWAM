#!/usr/bin/env python3
"""Fail-closed Stage1 B_CR wrapper around original FastWAM joint training.

The default command performs configuration/provenance audit only.  Long
training is possible only with the explicit ``--launch-stage1-long-training``
flag and complete dataset, stats, text-cache, initialization, and output
gates.  The model, loss implementation, and trainer remain the original
FastWAM implementations.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.robotwin.policy_content_adapter.native50hz_paired import atomic_write_json
from experiments.robotwin.policy_content_adapter.official_data import (
    OFFICIAL_TASKS,
    OfficialThreeTaskDataset,
)
from experiments.robotwin.policy_content_adapter.runtime_utils import (
    DEFAULT_OFFICIAL_MANIFEST,
    PROJECT_ROOT,
    instantiate_official_dataset,
    temporary_environment,
)


DEFAULT_STAGE1_CONFIG = Path(__file__).resolve().parent / "configs" / "stage1_clean_random_base.yaml"
LEGACY_SEEDS012_STAGE1_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "stage1_clean_random_base_seeds012_legacy.yaml"
)
FORMAL_SEED_PLAN_AMENDMENT = (
    Path(__file__).resolve().parent
    / "configs"
    / "formal_seed_plan_amendment_20260818.json"
)
FORMAL_MEMORY_AMENDMENT = (
    Path(__file__).resolve().parent
    / "configs"
    / "formal_stage1_memory_amendment_20260819.json"
)
EXPECTED_KIND = "policy_stage1_native_run"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_TASK_CONFIG = "robotwin_uncond_3cam_384_1e-4"
EXPECTED_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
EXPECTED_ACTION_INIT = "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
FORMAL_REPLICATION_SEEDS = (1, 2, 3)
LEGACY_FORMAL_REPLICATION_SEEDS = (0, 1, 2)
UINT32_MAX = int(np.iinfo(np.uint32).max)
FORMAL_WORLD_SIZE = 8
FORMAL_LOCAL_BATCH_SIZE = 8
FORMAL_GRADIENT_ACCUMULATION_STEPS = 2
FORMAL_EFFECTIVE_GLOBAL_BATCH_SIZE = 128


class Stage1ProtocolError(RuntimeError):
    """The Stage1 launch cannot prove the approved protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1ProtocolError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_stage1_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    target = Path(path).expanduser().resolve()
    _require(target.is_file(), f"Stage1 config not found: {target}")
    try:
        value = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage1ProtocolError(f"cannot parse Stage1 config {target}: {exc}") from exc
    _require(isinstance(value, dict), "Stage1 config root must be a mapping")
    return target, value


def _without_formal_seed_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(value))
    try:
        normalized["original_fastwam"]["training"]["formal_replication_seeds"] = (
            "__FORMAL_SEED_PLAN__"
        )
    except (KeyError, TypeError) as exc:
        raise Stage1ProtocolError("Stage1 config lacks a formal seed plan") from exc
    return normalized


def _verify_formal_seed_plan_amendment(current_config_path: Path) -> dict[str, Any]:
    path = FORMAL_SEED_PLAN_AMENDMENT.resolve()
    _require(path.is_file(), f"formal seed-plan amendment not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage1ProtocolError(f"cannot parse formal seed-plan amendment: {exc}") from exc
    _require(value.get("status") == "LOCKED", "formal seed-plan amendment is not LOCKED")
    _require(value.get("result_dependent_selection") is False, "seed amendment is result-dependent")
    _require(
        tuple(value.get("old_formal_replication_seeds", ())) == LEGACY_FORMAL_REPLICATION_SEEDS
        and tuple(value.get("new_formal_replication_seeds", ())) == FORMAL_REPLICATION_SEEDS,
        "formal seed-plan amendment values changed",
    )
    _require(
        value.get("old_stage1_config", {}).get("sha256")
        == _sha256(LEGACY_SEEDS012_STAGE1_CONFIG),
        "seed amendment legacy config SHA mismatch",
    )
    _require(
        value.get("new_stage1_config", {}).get("sha256") == _sha256(current_config_path),
        "seed amendment current config SHA mismatch",
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "old_formal_replication_seeds": list(LEGACY_FORMAL_REPLICATION_SEEDS),
        "new_formal_replication_seeds": list(FORMAL_REPLICATION_SEEDS),
        "result_dependent_selection": False,
    }


def _verify_formal_memory_amendment(current_config_path: Path) -> dict[str, Any]:
    path = FORMAL_MEMORY_AMENDMENT.resolve()
    _require(path.is_file(), f"formal memory amendment not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage1ProtocolError(f"cannot parse formal memory amendment: {exc}") from exc
    _require(value.get("status") == "LOCKED", "formal memory amendment is not LOCKED")
    _require(value.get("result_dependent_selection") is False, "memory amendment is result-dependent")
    _require(value.get("text_cache_content_changed") is False, "memory amendment changes text cache")
    _require(
        value.get("stage1_config", {}).get("sha256") == _sha256(current_config_path),
        "memory amendment Stage1 config SHA mismatch",
    )
    expected_old = {
        "local_batch_size_per_gpu": 16,
        "gradient_accumulation_steps": 1,
        "world_size": FORMAL_WORLD_SIZE,
        "effective_global_batch_size": FORMAL_EFFECTIVE_GLOBAL_BATCH_SIZE,
    }
    expected_new = {
        "local_batch_size_per_gpu": FORMAL_LOCAL_BATCH_SIZE,
        "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
        "world_size": FORMAL_WORLD_SIZE,
        "effective_global_batch_size": FORMAL_EFFECTIVE_GLOBAL_BATCH_SIZE,
    }
    _require(value.get("old_execution") == expected_old, "memory amendment old execution changed")
    _require(value.get("new_execution") == expected_new, "memory amendment new execution changed")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "old_execution": expected_old,
        "new_execution": expected_new,
        "result_dependent_selection": False,
    }


def _verify_text_cache_stage1_config_binding(
    completed_cache_audit: Mapping[str, Any],
    *,
    current_config_path: str | Path,
) -> dict[str, Any]:
    """Accept an exact config binding or the audited 012 -> 123 amendment.

    Text embeddings depend on the selected episodes, prompt template, text
    encoder and tokenizer, not on the later training RNG labels.  The existing
    67-GiB cache was completed immediately before the formal seed bank was
    amended.  Preserve the exact old YAML bytes and prove that the seed list is
    the *only* semantic difference before reusing its payload audit.
    """

    cache_config = completed_cache_audit.get("stage1_config")
    _require(isinstance(cache_config, Mapping), "text cache lacks Stage1 config identity")
    cache_config_sha = str(cache_config.get("sha256", ""))
    current_path, current_value = _read_stage1_yaml(current_config_path)
    current_sha = _sha256(current_path)
    if cache_config_sha == current_sha:
        return {
            "status": "PASS",
            "mode": "exact_config_sha256",
            "cache_config_sha256": cache_config_sha,
            "launch_config_sha256": current_sha,
            "cache_content_changed": False,
        }

    legacy_path, legacy_value = _read_stage1_yaml(LEGACY_SEEDS012_STAGE1_CONFIG)
    legacy_sha = _sha256(legacy_path)
    _require(
        cache_config_sha == legacy_sha,
        "text cache was prepared from neither the current nor preserved legacy Stage1 config",
    )
    legacy_seeds = tuple(
        legacy_value["original_fastwam"]["training"]["formal_replication_seeds"]
    )
    current_seeds = tuple(
        current_value["original_fastwam"]["training"]["formal_replication_seeds"]
    )
    _require(
        legacy_seeds == LEGACY_FORMAL_REPLICATION_SEEDS,
        "preserved legacy Stage1 seed plan changed",
    )
    _require(
        current_seeds == FORMAL_REPLICATION_SEEDS,
        "current Stage1 formal seed plan is not (1, 2, 3)",
    )
    _require(
        _without_formal_seed_plan(legacy_value)
        == _without_formal_seed_plan(current_value),
        "Stage1 config differs from the cache-bound legacy config beyond the formal seed plan",
    )
    return {
        "status": "PASS",
        "mode": "seed_plan_only_amendment_012_to_123",
        "cache_config": {"path": str(legacy_path), "sha256": legacy_sha},
        "launch_config": {"path": str(current_path), "sha256": current_sha},
        "old_formal_replication_seeds": list(legacy_seeds),
        "new_formal_replication_seeds": list(current_seeds),
        "changed_field": "original_fastwam.training.formal_replication_seeds",
        "cache_content_changed": False,
        "result_dependent_selection": False,
        "formal_seed_plan_amendment": _verify_formal_seed_plan_amendment(current_path),
    }


def _model_initialization_seed_contract(seed: int) -> dict[str, Any]:
    _require(seed in FORMAL_REPLICATION_SEEDS, f"formal Stage1 seed must be one of {FORMAL_REPLICATION_SEEDS}")
    return {
        "base_seed": int(seed),
        "model_construction_seed_policy": "same base seed on every rank before model construction",
        "trainer_seed_policy": "original FastWAM base_seed + global_rank after model construction",
        "rank_offset_during_model_construction": False,
        "random_libraries": ["python", "numpy", "torch_cpu", "torch_cuda_all"],
    }


def _seed_model_initialization(seed: int) -> dict[str, Any]:
    """Seed random-kept FastWAM modules identically before construction."""

    contract = _model_initialization_seed_contract(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return contract


def _expected_optimizer_steps(
    *,
    dataset_size: int,
    local_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:
    values = (
        dataset_size,
        local_batch_size,
        world_size,
        gradient_accumulation_steps,
        epochs,
    )
    _require(all(int(value) > 0 for value in values), "optimizer-step inputs must be positive")
    micro_steps_per_epoch = math.ceil(dataset_size / (local_batch_size * world_size))
    return math.ceil(micro_steps_per_epoch / gradient_accumulation_steps) * epochs


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _compose_original_training_config() -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={EXPECTED_TASK_CONFIG}"])
    # Loss defaults are 1.0 in the original factory; make both coefficients
    # explicit in the resolved launch artifact without changing the formula.
    with open_dict(cfg.model.loss):
        cfg.model.loss.lambda_video = 1.0
        cfg.model.loss.lambda_action = 1.0
    return cfg


def _audit_composed_original(cfg: DictConfig) -> dict[str, Any]:
    _require(cfg.model._target_ == "fastwam.runtime.create_fastwam", "Stage1 model factory changed")
    _require(cfg.model.model_id == EXPECTED_MODEL_ID, "Stage1 video initialization changed")
    _require(
        cfg.model.action_dit_pretrained_path == EXPECTED_ACTION_INIT,
        "Stage1 ActionDiT initialization changed",
    )
    _require(cfg.model.skip_dit_load_from_pretrain is False, "Stage1 must load pre-RoboTwin DiT init")
    _require(cfg.resume is None, "Stage1 cannot resume a RoboTwin-trained checkpoint")
    expected_scalars = {
        "num_epochs": 5,
        "batch_size": 16,
        "num_workers": 8,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "weight_decay": 1e-2,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
    }
    for key, expected in expected_scalars.items():
        actual = cfg.get(key)
        if isinstance(expected, float):
            _require(math.isclose(float(actual), expected, abs_tol=1e-12), f"Stage1 {key} changed")
        else:
            _require(actual == expected, f"Stage1 {key} changed")
    _require(float(cfg.model.loss.lambda_video) == 1.0, "lambda_video must be 1")
    _require(float(cfg.model.loss.lambda_action) == 1.0, "lambda_action must be 1")
    _require(int(cfg.data.train.num_frames) == 33, "Stage1 native sequence must contain 33 states")
    _require(int(cfg.data.train.action_video_freq_ratio) == 4, "action/video ratio changed")
    _require(int(cfg.data.train.global_sample_stride) == 1, "native sample stride changed")
    return {
        "status": "PASS",
        "model_factory": str(cfg.model._target_),
        "model_id": str(cfg.model.model_id),
        "action_dit_pretrained_path": str(cfg.model.action_dit_pretrained_path),
        "skip_dit_load_from_pretrain": bool(cfg.model.skip_dit_load_from_pretrain),
        "resume": None,
        "num_epochs": int(cfg.num_epochs),
        "loss": {
            "formula": "lambda_video * L_video + lambda_action * L_action",
            "lambda_video": float(cfg.model.loss.lambda_video),
            "lambda_action": float(cfg.model.loss.lambda_action),
        },
        "native_sequence": {
            "states": int(cfg.data.train.num_frames),
            "future_actions": int(cfg.data.train.num_frames) - 1,
            "action_video_freq_ratio": int(cfg.data.train.action_video_freq_ratio),
            "global_sample_stride": int(cfg.data.train.global_sample_stride),
        },
    }


def validate_stage1_config(
    config_path: str | Path = DEFAULT_STAGE1_CONFIG,
    *,
    require_artifacts: bool = False,
    dataset_root_override: str | Path | None = None,
    dataset_stats_override: str | Path | None = None,
    dataset_stats_sha256_override: str | None = None,
    text_cache_override: str | Path | None = None,
    text_cache_audit_override: str | Path | None = None,
    text_cache_audit_sha256_override: str | None = None,
    output_dir_override: str | Path | None = None,
    model_base_path_override: str | Path | None = None,
    action_dit_init_override: str | Path | None = None,
    training_seed_override: int | None = None,
) -> dict[str, Any]:
    """Validate schema/invariants and optionally all launch artifacts.

    This callable is intentionally independent from the Stage2 config schema;
    ``config_audit.audit_config_directory`` routes ``kind``
    ``policy_stage1_native_run`` here.
    """

    path, value = _read_stage1_yaml(config_path)
    _require(value.get("kind") == EXPECTED_KIND, f"Stage1 kind must be {EXPECTED_KIND}")
    _require(
        value.get("schema_version") == EXPECTED_SCHEMA_VERSION,
        f"Stage1 schema_version must be {EXPECTED_SCHEMA_VERSION}",
    )
    _require(value.get("protocol_id") == "policy_protocol_v2_stage1_b_cr", "protocol_id changed")
    original = value.get("original_fastwam")
    _require(isinstance(original, Mapping), "original_fastwam mapping is missing")
    _require(original.get("hydra_config") == "train", "Stage1 must compose configs/train.yaml")
    _require(original.get("task_config") == EXPECTED_TASK_CONFIG, "original task config changed")
    _require(original.get("trainer") == "fastwam.trainer.Wan22Trainer", "trainer changed")
    _require(original.get("model_factory") == "fastwam.runtime.create_fastwam", "model factory changed")
    initialization = original.get("initialization")
    _require(isinstance(initialization, Mapping), "initialization mapping is missing")
    _require(initialization.get("video_model_id") == EXPECTED_MODEL_ID, "video model init changed")
    _require(initialization.get("action_dit_pretrained_path") == EXPECTED_ACTION_INIT, "ActionDiT init changed")
    _require(initialization.get("skip_dit_load_from_pretrain") is False, "DiT pretrain load disabled")
    _require(initialization.get("resume") is None, "Stage1 resume must be null")
    training = original.get("training")
    _require(isinstance(training, Mapping), "training mapping is missing")
    exact_training = {
        "num_epochs": 5,
        "batch_size": 16,
        "num_workers": 8,
        "learning_rate": 0.0001,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
    }
    for key, expected in exact_training.items():
        _require(training.get(key) == expected, f"Stage1 training.{key} changed")
    _require(training.get("seed") == 42, "Stage1 declared default seed changed")
    _require(
        tuple(training.get("formal_replication_seeds", ())) == FORMAL_REPLICATION_SEEDS,
        "Stage1 formal replication seed plan must be [1,2,3]",
    )
    seed_plan_amendment = _verify_formal_seed_plan_amendment(path)
    memory_amendment = _verify_formal_memory_amendment(path)
    loss = original.get("loss")
    _require(isinstance(loss, Mapping), "loss mapping is missing")
    _require(loss.get("lambda_video") == loss.get("lambda_action") == 1.0, "joint loss weights changed")

    initialization_artifacts = value.get("initialization_artifacts")
    _require(isinstance(initialization_artifacts, Mapping), "initialization_artifacts mapping is missing")
    model_base_path = Path(
        model_base_path_override
        if model_base_path_override is not None
        else initialization_artifacts.get("model_base_path", "")
    ).expanduser().resolve()
    action_init = Path(
        action_dit_init_override
        if action_dit_init_override is not None
        else initialization_artifacts.get("action_dit_pretrained_path", "")
    ).expanduser().resolve()
    _require(action_init.name == Path(EXPECTED_ACTION_INIT).name, "unexpected ActionDiT artifact name")

    data = value.get("data")
    _require(isinstance(data, Mapping), "Stage1 data mapping is missing")
    _require(data.get("episode_selection_mode") == "full_550_per_task", "full selection is required")
    _require(tuple(data.get("task_order", ())) == OFFICIAL_TASKS, "Stage1 task order changed")
    _require(
        data.get("exact_counts_per_task")
        == {"clean": 50, "official_random": 500, "total": 550},
        "Stage1 exact domain counts changed",
    )
    _require(data.get("sampling_mode") == "all_frames", "Stage1 must use all native frames")
    _require(data.get("native_fps") == 50, "Stage1 fps must be 50")
    _require(data.get("interpolation") == "forbidden", "Stage1 interpolation must be forbidden")
    stats_decl = data.get("dataset_stats")
    _require(isinstance(stats_decl, Mapping), "dataset_stats mapping is missing")
    declared_stats_sha = (
        dataset_stats_sha256_override
        if dataset_stats_sha256_override is not None
        else stats_decl.get("sha256")
    )
    stats_sha_unresolved = (
        isinstance(declared_stats_sha, str)
        and declared_stats_sha.startswith("__REQUIRED_STAGE1_FULL550_DATASET_STATS_SHA256")
    )
    if not stats_sha_unresolved:
        _require(
            isinstance(declared_stats_sha, str)
            and len(declared_stats_sha) == 64
            and all(char in "0123456789abcdef" for char in declared_stats_sha.lower()),
            "dataset stats SHA-256 must be explicit",
        )
    launch = value.get("launch")
    _require(isinstance(launch, Mapping), "launch mapping is missing")
    _require(launch.get("default_mode") == "audit_only", "Stage1 default must be audit_only")
    _require(
        launch.get("long_training_requires_flag") == "--launch-stage1-long-training",
        "Stage1 launch confirmation flag changed",
    )
    _require(launch.get("recommended_world_size") == 8, "Stage1 recommended world size changed")

    composed = _compose_original_training_config()
    composed_audit = _audit_composed_original(composed)
    dataset_root = _resolve_project_path(
        dataset_root_override if dataset_root_override is not None else data["dataset_root"]
    )
    manifest = _resolve_project_path(data.get("official_manifest", DEFAULT_OFFICIAL_MANIFEST))
    stats_path = _resolve_project_path(
        dataset_stats_override if dataset_stats_override is not None else stats_decl["path"]
    )
    text_cache_raw = (
        text_cache_override if text_cache_override is not None else data["text_embedding_cache_dir"]
    )
    text_cache = _resolve_project_path(text_cache_raw)
    text_cache_audit = (
        Path(text_cache_audit_override).expanduser().resolve()
        if text_cache_audit_override is not None
        else Path(f"{text_cache}.audit.json").resolve()
    )
    declared_text_cache_audit_sha = text_cache_audit_sha256_override
    if declared_text_cache_audit_sha is not None:
        _require(
            len(declared_text_cache_audit_sha) == 64
            and all(char in "0123456789abcdef" for char in declared_text_cache_audit_sha.lower()),
            "text-cache audit SHA-256 must be explicit",
        )
    output_value = output_dir_override if output_dir_override is not None else value["output"]["directory"]
    output_dir = _resolve_project_path(output_value)
    training_seed = int(
        training_seed_override if training_seed_override is not None else training["seed"]
    )
    _require(0 <= training_seed <= UINT32_MAX, "Stage1 training seed must fit uint32")
    declared_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _require(declared_world_size >= 1, "WORLD_SIZE must be positive")
    _require(
        training_seed + declared_world_size - 1 <= UINT32_MAX,
        "Stage1 seed + global rank exceeds uint32",
    )
    if require_artifacts:
        _require(
            training_seed in FORMAL_REPLICATION_SEEDS,
            f"formal Stage1 seed must be one of {FORMAL_REPLICATION_SEEDS}",
        )

    artifact_status = "NOT_REQUESTED"
    artifacts: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "official_manifest": str(manifest),
        "dataset_stats": {"path": str(stats_path), "declared_sha256": declared_stats_sha},
        "text_embedding_cache_dir": str(text_cache),
        "text_embedding_cache_audit": {
            "path": str(text_cache_audit),
            "declared_sha256": declared_text_cache_audit_sha,
        },
        "output_dir": str(output_dir),
        "model_base_path": str(model_base_path),
        "action_dit_pretrained": {"path": str(action_init)},
        "training_seed": training_seed,
    }
    if require_artifacts:
        _require(dataset_root.is_dir(), f"official dataset root not found: {dataset_root}")
        _require(manifest.is_file(), f"official manifest not found: {manifest}")
        _require(stats_path.is_file(), f"dataset stats not found: {stats_path}")
        _require(
            not stats_sha_unresolved,
            "three-task full550 dataset stats SHA placeholder is unresolved; generate stats first",
        )
        actual_stats_sha = _sha256(stats_path)
        _require(actual_stats_sha == declared_stats_sha, "dataset stats SHA-256 mismatch")
        _require("__REQUIRED" not in str(text_cache_raw), "text cache placeholder is unresolved")
        _require(text_cache.is_dir(), f"text embedding cache not found: {text_cache}")
        _require(text_cache_audit.is_file(), f"text cache audit not found: {text_cache_audit}")
        _require(
            declared_text_cache_audit_sha is not None,
            "text-cache audit SHA-256 is required for a long Stage1 launch",
        )
        actual_text_cache_audit_sha = _sha256(text_cache_audit)
        _require(
            actual_text_cache_audit_sha == declared_text_cache_audit_sha,
            "text-cache audit SHA-256 mismatch",
        )
        from experiments.robotwin.policy_content_adapter.stage1_text_cache import (
            verify_text_cache_audit,
        )

        verified_cache_audit, _ = verify_text_cache_audit(
            text_cache_audit,
            cache_dir=text_cache,
            stage1_config_sha256=None,
        )
        cache_config_binding = _verify_text_cache_stage1_config_binding(
            verified_cache_audit,
            current_config_path=path,
        )
        artifacts["text_embedding_cache_audit"] = {
            "path": str(text_cache_audit),
            "sha256": actual_text_cache_audit_sha,
            "aggregate_payload_sha256": verified_cache_audit["cache"][
                "aggregate_payload_sha256"
            ],
            "inventory": verified_cache_audit["inventory"],
            "stage1_config_binding": cache_config_binding,
        }
        _require(model_base_path.is_dir(), f"Wan model component base not found: {model_base_path}")
        required_wan_assets = (
            model_base_path
            / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
            *sorted(
                (
                    model_base_path / "Wan-AI/Wan2.2-TI2V-5B"
                ).glob("diffusion_pytorch_model-*-of-*.safetensors")
            ),
        )
        _require(len(required_wan_assets) == 4, "Wan2.2 initialization must contain one VAE and three DiT shards")
        _require(all(path.is_file() for path in required_wan_assets), "Wan2.2 initialization assets are incomplete")
        process_rank = int(os.environ.get("RANK", "0"))
        if process_rank == 0:
            _require(not output_dir.exists(), f"refusing to overwrite Stage1 output: {output_dir}")
        _require(action_init.is_file(), f"ActionDiT initialization not found: {action_init}")
        artifacts["dataset_stats"]["actual_sha256"] = actual_stats_sha
        if process_rank == 0:
            # Avoid eight concurrent 2-GiB CPFS reads under torchrun.  All
            # ranks prove the same shared path exists; rank 0 content-addresses
            # it for the persisted audit/completion manifest.
            artifacts["action_dit_pretrained"] = {
                "path": str(action_init),
                "sha256": _sha256(action_init),
            }
            artifacts["wan_initialization"] = {
                "model_base_path": str(model_base_path),
                "model_id": EXPECTED_MODEL_ID,
                "vae_path": str(required_wan_assets[0]),
                "video_dit_shards": [str(path) for path in required_wan_assets[1:]],
            }
        artifact_status = "PASS"

    sources = {}
    for relative in (
        "src/fastwam/runtime.py",
        "src/fastwam/trainer.py",
        "src/fastwam/models/wan22/fastwam.py",
        "src/fastwam/utils/pytorch_utils.py",
        "src/fastwam/utils/samplers.py",
    ):
        source = PROJECT_ROOT / relative
        _require(source.is_file(), f"original FastWAM source not found: {source}")
        sources[relative] = _sha256(source)
    return {
        "status": "PASS",
        "kind": EXPECTED_KIND,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "config": {"path": str(path), "sha256": _sha256(path)},
        "default_launches_training": False,
        "selection": {
            "mode": "full_550_per_task",
            "tasks": list(OFFICIAL_TASKS),
            "counts_per_task": {"clean": 50, "official_random": 500, "total": 550},
        },
        "training_seed": training_seed,
        "formal_replication_seeds": list(FORMAL_REPLICATION_SEEDS),
        "formal_seed_plan_amendment": seed_plan_amendment,
        "formal_memory_amendment": memory_amendment,
        "execution": {
            "local_batch_size_per_gpu": FORMAL_LOCAL_BATCH_SIZE,
            "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
            "world_size": FORMAL_WORLD_SIZE,
            "effective_global_batch_size": FORMAL_EFFECTIVE_GLOBAL_BATCH_SIZE,
            "author_reference_local_batch_size": int(composed.batch_size),
            "author_reference_gradient_accumulation_steps": int(
                composed.gradient_accumulation_steps
            ),
        },
        "model_initialization_seed_contract": (
            _model_initialization_seed_contract(training_seed)
            if training_seed in FORMAL_REPLICATION_SEEDS
            else None
        ),
        "original_fastwam": composed_audit,
        "artifacts_status": artifact_status,
        "artifacts": artifacts,
        "original_source_sha256": sources,
    }


def _audit_text_cache_for_selected_dataset(
    native_dataset: Any,
    cache_dir: Path,
    *,
    completion_audit_path: Path,
    stage1_config_path: str | Path,
) -> dict[str, Any]:
    from experiments.robotwin.policy_content_adapter.stage1_text_cache import (
        CACHE_SUFFIX,
        EXPECTED_UNIQUE_TASK_INDICES,
        merge_shard_reports,
        validate_cache_payload,
        verify_text_cache_audit,
    )

    completed_audit, inventory_entries = verify_text_cache_audit(
        completion_audit_path,
        cache_dir=cache_dir,
        stage1_config_sha256=None,
    )
    cache_config_binding = _verify_text_cache_stage1_config_binding(
        completed_audit,
        current_config_path=stage1_config_path,
    )
    inventory_by_digest = {entry["sha256"]: entry for entry in inventory_entries}
    inner = native_dataset.lerobot_dataset.multi_dataset._datasets[0]
    try:
        raw_indices = inner.hf_dataset.unique("task_index")
    except Exception as exc:
        raise Stage1ProtocolError(f"cannot enumerate selected task prompts: {exc}") from exc
    task_indices = sorted({int(value) for value in raw_indices})
    _require(task_indices, "selected official data has no task prompts")
    runtime_entries: list[dict[str, str]] = []
    for task_index in task_indices:
        instruction = inner.meta.tasks[task_index]
        prompt = (
            "A video recorded from a robot's point of view executing the following instruction: "
            f"{instruction}"
        )
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        _require(digest in inventory_by_digest, f"runtime prompt absent from inventory: {digest}")
        runtime_entries.append(
            {
                "sha256": digest,
                "prompt": prompt,
                "task": inventory_by_digest[digest]["task"],
                "task_index": str(task_index),
            }
        )
    runtime_entries.sort(key=lambda entry: entry["sha256"])
    _require(
        runtime_entries == inventory_entries,
        "runtime selected task_index inventory differs from completed text-cache inventory",
    )
    required_paths = [cache_dir / f"{entry['sha256']}{CACHE_SUFFIX}" for entry in runtime_entries]
    missing = [str(path) for path in required_paths if not path.is_file()]
    _require(
        not missing,
        f"text cache is incomplete for {len(missing)}/{len(required_paths)} selected prompts; "
        f"first missing: {missing[:3]}",
    )
    identities = []
    for position, path in enumerate(required_paths, start=1):
        identities.append(validate_cache_payload(path))
        if position % 1_000 == 0 or position == len(required_paths):
            print(
                f"Stage1 rank-0 text-cache audit: {position}/{len(required_paths)} payloads",
                file=sys.stderr,
                flush=True,
            )
    runtime_cache = merge_shard_reports(
        runtime_entries,
        [
            {
                "rank": 0,
                "world_size": 1,
                "assigned_count": len(runtime_entries),
                "created_count": 0,
                "skipped_valid_count": len(runtime_entries),
                "concurrent_valid_count": 0,
                "over_length_prompt_count": 0,
                "files": identities,
            }
        ],
        cache_dir=cache_dir,
    )
    _require(
        runtime_cache["aggregate_payload_sha256"]
        == completed_audit["cache"]["aggregate_payload_sha256"],
        "runtime text-cache payload aggregate differs from completed cache audit",
    )
    _require(
        len(runtime_entries) == EXPECTED_UNIQUE_TASK_INDICES,
        "runtime text-cache prompt count changed",
    )
    return {
        "status": "PASS",
        "unique_prompt_count": len(runtime_entries),
        "all_required_cache_files_present": True,
        "all_payload_shapes_valid": True,
        "over_length_prompt_count": 0,
        "aggregate_payload_sha256": runtime_cache["aggregate_payload_sha256"],
        "completion_audit": {
            "path": str(completion_audit_path.resolve()),
            "sha256": _sha256(completion_audit_path),
        },
        "inventory": completed_audit["inventory"],
        "stage1_config_binding": cache_config_binding,
    }


def prepare_stage1_dataset_stats(
    audit: Mapping[str, Any],
    *,
    stats_output: str | Path,
) -> dict[str, Any]:
    """Compute normalization stats on exactly the selected 1,650 episodes.

    This is an explicit CPU/read-only-data preparation phase; it never loads a
    model and never launches training.  The native RobotVideoDataset stats
    implementation is reused only after the scoped explicit loader has
    replaced the 27,500-release episode list with ``full_550_per_task``.
    """

    artifacts = audit["artifacts"]
    output = Path(stats_output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    audit_output = output.with_suffix(output.suffix + ".audit.json")
    _require(not output.exists(), f"refusing to overwrite dataset stats: {output}")
    _require(not audit_output.exists(), f"refusing to overwrite stats audit: {audit_output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    _require(not staging.exists(), f"stats staging path already exists: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        from fastwam.utils import misc

        misc.register_work_dir(staging)
        cfg = _compose_original_training_config()
        dataset = instantiate_official_dataset(
            cfg,
            dataset_root=artifacts["dataset_root"],
            dataset_stats_path=None,
            text_cache_dir=staging / "unused_text_cache",
            manifest_path=artifacts["official_manifest"],
            episode_selection_mode="full_550_per_task",
            allow_compute_dataset_stats=True,
        )
        selection = getattr(dataset, "_official_explicit_episode_selection", None)
        _require(selection is not None, "stats dataset lacks explicit full550 selection provenance")
        provenance = selection.as_provenance()
        _require(provenance.get("selection_mode") == "full_550_per_task", "stats used wrong mode")
        _require(provenance.get("loaded_episode_count") == 1_650, "stats did not use 1,650 episodes")
        expected_domain_counts = {
            task: {"clean": 50, "official_random": 500} for task in OFFICIAL_TASKS
        }
        _require(
            provenance.get("loaded_episode_counts_by_task_domain") == expected_domain_counts,
            "stats selection is not 50 Clean + 500 Official Random per task",
        )
        generated = staging / "dataset_stats.json"
        _require(generated.is_file(), "native stats computation did not produce dataset_stats.json")
        stats_sha = _sha256(generated)
        os.replace(generated, output)
        report = {
            "status": "PASS",
            "kind": "policy_stage1_full550_dataset_stats",
            "stats": {
                "path": str(output),
                "sha256": stats_sha,
                "size_bytes": output.stat().st_size,
            },
            "dataset_root": artifacts["dataset_root"],
            "official_manifest": {
                "path": artifacts["official_manifest"],
                "sha256": selection.manifest_sha256,
            },
            "selection": provenance,
            "episode_count": 1_650,
            "counts_per_task_domain": expected_domain_counts,
            "native_stats_implementation": (
                "RobotVideoDataset -> BaseLerobotDataset.get_dataset_stats after explicit episode narrowing"
            ),
            "training_launched": False,
        }
        atomic_write_json(audit_output, report)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@contextlib.contextmanager
def _temporary_original_dataset_factory(train_dataset: Any) -> Iterator[None]:
    import fastwam.runtime as runtime

    original = runtime.build_datasets
    runtime.build_datasets = lambda _cfg: (train_dataset, None)
    try:
        yield
    finally:
        runtime.build_datasets = original


@contextlib.contextmanager
def _temporary_stage1_dataset_init_work_dir(
    *,
    rank: int,
) -> Iterator[Path]:
    """Give the native dataset a real work dir before formal authorization.

    ``RobotVideoDataset`` mirrors supplied normalization statistics into the
    currently registered FastWAM work directory during construction.  The
    formal output directory deliberately does not exist until every dataset
    and text-cache gate has passed, so using FastWAM's default ``./runs`` here
    is both ambiguous and, on a clean checkout, a ``FileNotFoundError``.

    Keep this constructor-only side effect in a per-rank temporary directory.
    ``runtime.run_training`` later registers the authorized formal output as
    the real work directory.  Restoring the private registration value is
    necessary because ``misc.register_work_dir(None)`` itself attempts to
    create a directory named ``None`` in the author implementation.
    """

    from fastwam.utils import misc

    previous_work_dir = getattr(misc, "_WORK_DIR", None)
    prefix = f"fastwam-stage1-dataset-init-rank{rank:03d}-"
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        work_dir = Path(temporary).resolve()
        misc.register_work_dir(work_dir)
        try:
            yield work_dir
        finally:
            setattr(misc, "_WORK_DIR", previous_work_dir)


def launch_stage1(
    audit: dict[str, Any],
    *,
    config_path: str | Path,
) -> None:
    _, value = _read_stage1_yaml(config_path)
    artifacts = audit["artifacts"]
    cfg = _compose_original_training_config()
    _audit_composed_original(cfg)
    cfg.output_dir = artifacts["output_dir"]
    cfg.seed = int(audit["training_seed"])
    cfg.resume = None
    cfg.batch_size = FORMAL_LOCAL_BATCH_SIZE
    cfg.gradient_accumulation_steps = FORMAL_GRADIENT_ACCUMULATION_STEPS
    cfg.model.action_dit_pretrained_path = artifacts["action_dit_pretrained"]["path"]
    output_dir = Path(artifacts["output_dir"])
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _require(
        world_size == FORMAL_WORLD_SIZE,
        f"formal Stage1 requires WORLD_SIZE={FORMAL_WORLD_SIZE}, got {world_size}",
    )
    _require(
        int(cfg.batch_size) * world_size * int(cfg.gradient_accumulation_steps)
        == FORMAL_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "formal Stage1 effective global batch changed",
    )
    authorization = output_dir / ".stage1_launch_authorized.json"
    text_cache = Path(artifacts["text_embedding_cache_dir"])
    with _temporary_stage1_dataset_init_work_dir(rank=rank):
        native_dataset = instantiate_official_dataset(
            cfg,
            dataset_root=artifacts["dataset_root"],
            dataset_stats_path=artifacts["dataset_stats"]["path"],
            text_cache_dir=text_cache,
            manifest_path=artifacts["official_manifest"],
            episode_selection_mode="full_550_per_task",
        )
    strict_dataset = OfficialThreeTaskDataset(
        native_dataset,
        dataset_root=artifacts["dataset_root"],
        manifest_path=artifacts["official_manifest"],
        sampling_mode=value["data"]["sampling_mode"],
    )
    provenance = strict_dataset.provenance
    counts = provenance["selected_episode_counts_by_domain"]
    _require(
        counts
        == {
            task: {"clean": 50, "official_random": 500}
            for task in OFFICIAL_TASKS
        },
        "runtime Stage1 selection is not exactly 50 Clean + 500 Official Random per task",
    )
    cache_completion_audit_path = Path(artifacts["text_embedding_cache_audit"]["path"])
    cache_audit: dict[str, Any] | None = None
    if rank == 0:
        # Validate all 68,704 persisted payloads once, before creating the
        # formal output directory.  Other ranks wait for this immutable gate
        # instead of multiplying ~70 GiB of cache reads by WORLD_SIZE.
        cache_audit = _audit_text_cache_for_selected_dataset(
            native_dataset,
            text_cache,
            completion_audit_path=cache_completion_audit_path,
            stage1_config_path=audit["config"]["path"],
        )
        run_audit = dict(audit)
        run_audit["runtime_dataset"] = provenance
        run_audit["text_cache_audit"] = cache_audit
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output_dir / "stage1_protocol_audit.json", run_audit)
        atomic_write_json(
            authorization,
            {
                "status": "AUTHORIZED",
                "config_sha256": audit["config"]["sha256"],
                "training_seed": int(cfg.seed),
                "local_batch_size_per_gpu": int(cfg.batch_size),
                "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                "effective_global_batch_size": FORMAL_EFFECTIVE_GLOBAL_BATCH_SIZE,
                "text_cache_audit_sha256": _sha256(cache_completion_audit_path),
            },
        )
    else:
        # Rank 0 may need several minutes to validate the complete cache.
        for _ in range(3_600):
            if authorization.is_file():
                break
            time.sleep(1)
        _require(authorization.is_file(), "rank 0 did not authorize the Stage1 output")

    expected_steps = _expected_optimizer_steps(
        dataset_size=len(strict_dataset),
        local_batch_size=int(cfg.batch_size),
        world_size=world_size,
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        epochs=int(cfg.num_epochs),
    )

    import fastwam.runtime as runtime

    model_initialization_seed = _seed_model_initialization(int(cfg.seed))
    model_initialization_seed["world_size"] = world_size
    model_initialization_seed["trainer_process_seeds"] = list(
        range(int(cfg.seed), int(cfg.seed) + world_size)
    )
    with _temporary_original_dataset_factory(strict_dataset), temporary_environment(
        "DIFFSYNTH_MODEL_BASE_PATH", artifacts["model_base_path"]
    ):
        runtime.run_training(cfg)

    if rank == 0:
        _require(cache_audit is not None, "rank 0 cache audit was not retained")
        weights = sorted((output_dir / "checkpoints" / "weights").glob("step_*.pt"))
        _require(bool(weights), "Stage1 returned without a weights checkpoint")
        def checkpoint_step(path: Path) -> int:
            try:
                return int(path.stem.removeprefix("step_"))
            except ValueError as exc:
                raise Stage1ProtocolError(f"invalid Stage1 checkpoint name: {path}") from exc

        final_checkpoint = max(weights, key=checkpoint_step)
        final_step = checkpoint_step(final_checkpoint)
        _require(
            final_step == expected_steps,
            f"final Stage1 step {final_step} != five-epoch expected step {expected_steps}",
        )
        state_file = (
            output_dir
            / "checkpoints"
            / "state"
            / f"step_{final_step:06d}"
            / "trainer_state.json"
        )
        _require(state_file.is_file(), "final Stage1 trainer_state.json is missing")
        state = json.loads(state_file.read_text(encoding="utf-8"))
        _require(int(state.get("global_step", -1)) == final_step, "trainer state step mismatch")
        stats_path = Path(artifacts["dataset_stats"]["path"])
        stats_sha = _sha256(stats_path)
        _require(
            stats_sha == artifacts["dataset_stats"]["actual_sha256"],
            "dataset stats changed during Stage1",
        )
        completion = {
            "status": "PASS",
            "kind": "policy_stage1_b_cr_completion",
            "protocol_id": "policy_protocol_v2_stage1_b_cr",
            "training_seed": int(cfg.seed),
            "formal_replication_seeds": list(FORMAL_REPLICATION_SEEDS),
            "model_initialization_seed": model_initialization_seed,
            "world_size": world_size,
            "execution": audit["execution"],
            "formal_memory_amendment": audit["formal_memory_amendment"],
            "configured_epochs": int(cfg.num_epochs),
            "expected_optimizer_steps_for_five_epochs": expected_steps,
            "completed_optimizer_steps": final_step,
            "trainer_state": state,
            "base_checkpoint": {
                "path": str(final_checkpoint.resolve()),
                "size_bytes": final_checkpoint.stat().st_size,
                "sha256": _sha256(final_checkpoint),
            },
            "dataset_stats": {
                "path": str(stats_path.resolve()),
                "size_bytes": stats_path.stat().st_size,
                "sha256": stats_sha,
            },
            "text_embedding_cache": {
                "directory": str(text_cache.resolve()),
                "audit": cache_audit["completion_audit"],
                "inventory": cache_audit["inventory"],
                "unique_prompt_count": cache_audit["unique_prompt_count"],
                "aggregate_payload_sha256": cache_audit["aggregate_payload_sha256"],
                "over_length_prompt_count": cache_audit["over_length_prompt_count"],
                "stage1_config_binding": cache_audit["stage1_config_binding"],
            },
            "official_manifest": {
                "path": artifacts["official_manifest"],
                "sha256": native_dataset._official_explicit_episode_selection.manifest_sha256,  # noqa: SLF001
            },
            "initialization": {
                "video_model_id": EXPECTED_MODEL_ID,
                "model_base_path": artifacts["model_base_path"],
                "action_dit_pretrained": artifacts["action_dit_pretrained"],
                "wan_initialization": artifacts["wan_initialization"],
                "resume": None,
            },
            "selection": {
                "mode": "full_550_per_task",
                "episode_count": 1_650,
                "counts_per_task_domain": counts,
            },
            "stage1_config": audit["config"],
            "formal_seed_plan_amendment": audit["formal_seed_plan_amendment"],
            "original_source_sha256": audit["original_source_sha256"],
            "protocol_audit": {
                "path": str((output_dir / "stage1_protocol_audit.json").resolve()),
                "sha256": _sha256(output_dir / "stage1_protocol_audit.json"),
            },
        }
        atomic_write_json(output_dir / "stage1_completion_manifest.json", completion)
        authorization.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-stats", type=Path)
    parser.add_argument("--dataset-stats-sha256")
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--text-cache-audit", type=Path)
    parser.add_argument("--text-cache-audit-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-base-path", type=Path)
    parser.add_argument("--action-dit-init", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--prepare-dataset-stats",
        "--generate-dataset-stats",
        dest="prepare_dataset_stats",
        action="store_true",
        help="compute and SHA-lock stats from the exact full550 three-task selection; no training",
    )
    parser.add_argument("--launch-stage1-long-training", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.prepare_dataset_stats and args.launch_stage1_long_training:
        print("Stage1 failed closed: stats preparation and long training are mutually exclusive", file=sys.stderr)
        return 2
    if args.launch_stage1_long_training and args.seed not in FORMAL_REPLICATION_SEEDS:
        print(
            "Stage1 failed closed: formal long training requires an explicit "
            f"--seed in {FORMAL_REPLICATION_SEEDS}",
            file=sys.stderr,
        )
        return 2
    try:
        audit = validate_stage1_config(
            args.config,
            require_artifacts=args.launch_stage1_long_training,
            dataset_root_override=args.dataset_root,
            dataset_stats_override=args.dataset_stats,
            dataset_stats_sha256_override=args.dataset_stats_sha256,
            text_cache_override=args.text_cache_dir,
            text_cache_audit_override=args.text_cache_audit,
            text_cache_audit_sha256_override=args.text_cache_audit_sha256,
            output_dir_override=args.output_dir,
            model_base_path_override=args.model_base_path,
            action_dit_init_override=args.action_dit_init,
            training_seed_override=args.seed,
        )
        if args.audit_output is not None:
            atomic_write_json(args.audit_output, audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        if args.prepare_dataset_stats:
            stats_path = (
                args.dataset_stats
                if args.dataset_stats is not None
                else Path(audit["artifacts"]["dataset_stats"]["path"])
            )
            stats_report = prepare_stage1_dataset_stats(audit, stats_output=stats_path)
            print(json.dumps(stats_report, indent=2, sort_keys=True))
            print(
                "Dataset stats preparation PASS; use --dataset-stats-sha256 "
                f"{stats_report['stats']['sha256']} for the launch audit.",
                file=sys.stderr,
            )
            return 0
        if not args.launch_stage1_long_training:
            print(
                "Stage1 audit-only PASS; no model or dataset was instantiated and no training was launched.",
                file=sys.stderr,
            )
            return 0
        launch_stage1(audit, config_path=args.config)
    except Exception as exc:
        print(f"Stage1 failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_STAGE1_CONFIG",
    "EXPECTED_KIND",
    "EXPECTED_SCHEMA_VERSION",
    "Stage1ProtocolError",
    "launch_stage1",
    "prepare_stage1_dataset_stats",
    "validate_stage1_config",
]
