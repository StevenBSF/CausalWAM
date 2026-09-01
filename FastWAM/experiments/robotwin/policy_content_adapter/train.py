"""Strict release-base dual-stream P-v1/P-v2 policy-content-adapter trainer.

The entrypoint runs exactly the step count in the selected config.  It never
starts the formal matrix implicitly and fails closed on unresolved data,
checkpoint, task, domain, or architecture provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

_BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BOOTSTRAP_SRC_ROOT = _BOOTSTRAP_PROJECT_ROOT / "src"
if str(_BOOTSTRAP_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_SRC_ROOT))

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, skip_first_batches
from accelerate.utils import (
    DataLoaderConfiguration,
    broadcast_object_list,
    gather_object,
)
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from fastwam.utils import misc

from .config_audit import validate_execution_ready
from .full5ep_resume_amendment import verify_amendment as verify_full5ep_resume_amendment
from .data import (
    DualStreamIterator,
    FrozenPairedTokenDataset,
    NativePairedActionDataset,
    SameTaskPhysicalStateBatchSampler,
    audit_frozen_token_cache,
    audit_native_paired_action_contract,
    audit_native_paired_action_dataset,
    audit_policy_state_bank,
    build_policy_cache_extraction_contract,
    policy_cache_extractor_config,
    canonical_json_sha256,
    build_dual_stream_provenance,
    collate_paired_action_groups,
    collate_paired_token_groups,
    selected_episode_artifact_aggregate,
    verify_native_paired_action_manifest,
    verify_policy_state_bank,
)
from .losses import (
    official_action_loss,
    paired_action_loss,
    paired_contrastive_loss,
    zero_init_policy_identity_audit,
)
from .model import (
    GatedCrossAttentionAdapter,
    PolicyContentHead,
    artifact_identity,
    build_optimizer_param_groups,
    configure_trainable_modules,
    install_policy_content_adapter,
    load_e1_e3_head_checkpoint,
    module_state_sha256,
    save_policy_checkpoint,
)
from .official_data import (
    OFFICIAL_DOMAINS,
    OFFICIAL_TASKS,
    OfficialThreeTaskDataset,
    ThreeTaskRoundRobinSampler,
)
from .p_mode_selection import (
    canonical_sha256 as p_mode_canonical_sha256,
    formal_config_protocol_projection,
    validate_formal_protocol_lock_manifest_payload,
)
from .pair280_protocol import (
    PAIR280_ACTIVE_STEPS,
    PAIR280_CACHE_STORAGE,
    PAIR280_GROUPS,
    PAIR280_PROFILE_ID,
    PAIR280_STATE_ALGORITHM,
    PAIR280_STATE_ALGORITHM_VERSION,
    PAIR280_STATE_SEED,
    PAIR280_STATES_PER_TRAJECTORY,
    ShardedPair280TokenDataset,
    paired_active_count,
    paired_is_active,
    verify_pair280_state_bank,
)
from .pair280_sampler import (
    PAIR280_SAMPLER_ID,
    ExactPair280GlobalBatchSampler,
)
from .prepare_release_paired_text_cache import verify_release_paired_text_cache
from .release_lineage import verify_author_release_lineage
from .release_official_text_cache_binding import verify_binding as verify_official_text_cache_binding
from .release_paired_binding import verify_release_paired_binding
from .runtime_utils import (
    PROJECT_ROOT,
    audit_local_fastwam_source,
    dtype_from_name,
    instantiate_official_dataset,
    instantiate_release_model,
)
from . import runtime_utils as policy_runtime_utils
from .protocol import (
    POLICY_ACTION_DIM,
    POLICY_ACTION_STEPS,
    POLICY_CAMERA_COUNT,
    POLICY_CAMERA_NAMES,
    POLICY_NATIVE_FPS,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_VARIANTS,
    POLICY_VIEW_COUNT,
)
from .training_audit import (
    DistributionAccumulator,
    ParameterSnapshot,
    SampledParameterSnapshot,
    action_path_gradient_probe,
    assert_no_parameter_gradients,
    compare_distributions,
    module_gradient_report,
)


TASKS = OFFICIAL_TASKS
EXACT_LAYER = 16
EXACT_QUERIES = 8


def _pair280_paired_dataset_audit(
    *,
    cache_manifest_identity: Mapping[str, Any],
    physical_state_groups: int,
) -> dict[str, Any]:
    """Describe Pair-280 with the common four-scene provenance contract.

    Pair-280 replaces the monolithic frozen-token cache with trajectory shards,
    but it does not replace the Policy four-scene protocol.  Keep the common
    protocol fields explicit so ``build_dual_stream_provenance`` can apply the
    same fail-closed checks used by the original cache reader.
    """

    if int(physical_state_groups) != PAIR280_GROUPS:
        raise ValueError("Pair-280 provenance physical-state count changed")
    return {
        "status": "PASS",
        "kind": "policy_pair280_sharded_token_dataset",
        "profile_id": PAIR280_PROFILE_ID,
        "protocol_id": POLICY_PROTOCOL_ID,
        "variant_names": list(POLICY_VARIANTS),
        "view_count": POLICY_VIEW_COUNT,
        "r3_role": POLICY_R3_ROLE,
        "r3_training_positive": True,
        "supervision_mode": "contrastive",
        "layer": EXACT_LAYER,
        "cache_manifest": dict(cache_manifest_identity),
        "physical_state_groups": int(physical_state_groups),
        "scene_views": int(physical_state_groups) * POLICY_VIEW_COUNT,
        "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
        "paired_epochs": 10,
        "paired_active_steps": PAIR280_ACTIVE_STEPS,
        "exact_exposures_per_state": 10,
        "replacement_within_epoch": False,
    }


def _distributed_dataloader_config() -> DataLoaderConfiguration:
    """Keep custom batch samplers shardable without synthetic batch padding.

    The paired stream uses ``SameTaskPhysicalStateBatchSampler``.  It emits
    fixed-size batches but, like a generic PyTorch batch sampler, does not
    expose a ``batch_size`` attribute.  Accelerate therefore requires
    ``even_batches=False`` when it constructs ``BatchSamplerShard`` in a
    multi-process run.  Both Stage-2 streams are already materialized with a
    number of batches exactly divisible by the declared world size, so this
    disables only Accelerate's duplicate-padding behavior; it does not drop or
    add any training batch.
    """

    return DataLoaderConfiguration(
        split_batches=False,
        even_batches=False,
        use_seedable_sampler=False,
    )
EXACT_CONTENT_DIM = 384
EXACT_TEMPERATURE = 0.07
EXACT_LAMBDA = 0.1
ACTION_UPDATE_REQUIRED_PARAMETERS = (
    "action_expert.action_encoder.weight",
    "action_expert.text_embedding.2.weight",
    "action_expert.time_projection.1.weight",
    "action_expert.blocks.0.cross_attn.q.weight",
    "action_expert.blocks.10.self_attn.o.weight",
    "action_expert.blocks.20.ffn.2.weight",
    "action_expert.blocks.29.cross_attn.o.weight",
    "action_expert.head.weight",
)
OFFICIAL_DATALOADER_SEED_OFFSET = 31_415_926
PAIRED_DATALOADER_SEED_OFFSET = 14_142_135
IDENTITY_DATALOADER_SEED_OFFSET = 27_182_818
OFFICIAL_MAIN_PROCESS_DATA_SEED_OFFSET = 16_180_339
STAGE2_STEP_RNG_POLICY_ID = "stage2_mixed_radix_no_collision_v1"
STAGE2_STEP_RNG_STEPS_PER_STREAM = 1_000_000
STAGE2_STEP_RNG_RANK_CAPACITY = 1_024
STAGE2_STEP_RNG_TRAINING_SEED_MAX = 2**32 - 1
STAGE2_STEP_RNG_STREAM_IDS = {"official": 0, "paired": 1}


def stage2_step_rng_seed(
    training_seed: int,
    step_index: int,
    *,
    process_index: int = 0,
    stream: str = "official",
) -> int:
    """Map a Stage-2 run/stream/step key to one auditable Torch RNG seed.

    The mixed-radix layout is injective over the declared bounds.  In
    particular, adjacent formal training seeds do not produce shifted,
    overlapping noise/timestep streams.  C1 and C3 still receive identical
    official seeds when their training seed, process index, and step match.
    """

    if (
        not isinstance(training_seed, int)
        or isinstance(training_seed, bool)
        or not 0 <= training_seed <= STAGE2_STEP_RNG_TRAINING_SEED_MAX
    ):
        raise ValueError(
            "training_seed must be an integer in the uint32 range for the "
            "Stage-2 step RNG policy"
        )
    if (
        not isinstance(process_index, int)
        or isinstance(process_index, bool)
        or not 0 <= process_index < STAGE2_STEP_RNG_RANK_CAPACITY
    ):
        raise ValueError(
            f"process_index must be in [0, {STAGE2_STEP_RNG_RANK_CAPACITY})"
        )
    if (
        not isinstance(step_index, int)
        or isinstance(step_index, bool)
        or not 0 <= step_index < STAGE2_STEP_RNG_STEPS_PER_STREAM
    ):
        raise ValueError(
            f"step_index must be in [0, {STAGE2_STEP_RNG_STEPS_PER_STREAM})"
        )
    if stream not in STAGE2_STEP_RNG_STREAM_IDS:
        raise ValueError(
            f"stream must be one of {tuple(STAGE2_STEP_RNG_STREAM_IDS)}, got {stream!r}"
        )
    run_key = training_seed * STAGE2_STEP_RNG_RANK_CAPACITY + process_index
    stream_key = run_key * len(STAGE2_STEP_RNG_STREAM_IDS) + int(
        STAGE2_STEP_RNG_STREAM_IDS[stream]
    )
    seed = stream_key * STAGE2_STEP_RNG_STEPS_PER_STREAM + step_index
    if not 0 <= seed < 2**63:
        raise AssertionError("Stage-2 step RNG mapping exceeded signed int64")
    return seed


def _stage2_step_rng_contract(
    training_seed: int,
    *,
    max_steps: int,
    world_size: int,
) -> dict[str, Any]:
    """Describe the configured mapping and its concrete per-rank seed ranges."""

    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if max_steps > STAGE2_STEP_RNG_STEPS_PER_STREAM:
        raise ValueError(
            "max_steps exceeds the collision-free Stage-2 RNG stream capacity"
        )
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or not 0 < world_size <= STAGE2_STEP_RNG_RANK_CAPACITY
    ):
        raise ValueError(
            "world_size exceeds the collision-free Stage-2 RNG rank capacity"
        )
    # Validate the training seed even when no seed range has yet been requested.
    stage2_step_rng_seed(training_seed, 0)
    ranges = []
    for process_index in range(world_size):
        for stream in STAGE2_STEP_RNG_STREAM_IDS:
            ranges.append(
                {
                    "process_index": process_index,
                    "stream": stream,
                    "stream_id": STAGE2_STEP_RNG_STREAM_IDS[stream],
                    "first_step_index": 0,
                    "last_step_index": max_steps - 1,
                    "first_seed": stage2_step_rng_seed(
                        training_seed,
                        0,
                        process_index=process_index,
                        stream=stream,
                    ),
                    "last_seed": stage2_step_rng_seed(
                        training_seed,
                        max_steps - 1,
                        process_index=process_index,
                        stream=stream,
                    ),
                }
            )
    return {
        "status": "PASS",
        "policy_id": STAGE2_STEP_RNG_POLICY_ID,
        "mapping": (
            "(((training_seed * 1024 + process_index) * 2 + stream_id) "
            "* 1000000 + zero_based_step_index)"
        ),
        "collision_guarantee": (
            "injective for uint32 training_seed, process_index<1024, "
            "two declared streams, and step_index<1000000"
        ),
        "control_pairing": (
            "same training_seed/process_index/step gives identical C1/C3 "
            "official noise and timestep RNG"
        ),
        "training_seed": training_seed,
        "world_size": world_size,
        "max_steps": max_steps,
        "stream_ids": dict(STAGE2_STEP_RNG_STREAM_IDS),
        "bounds": {
            "training_seed": [0, STAGE2_STEP_RNG_TRAINING_SEED_MAX],
            "process_index": [0, STAGE2_STEP_RNG_RANK_CAPACITY - 1],
            "step_index": [0, STAGE2_STEP_RNG_STEPS_PER_STREAM - 1],
        },
        "configured_seed_ranges": ranges,
        "actual_seed_log": "train_log.csv:official_rng_seed/paired_rng_seed",
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty training log")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


_STAGE2_STATE_DIRECTORY_PATTERN = re.compile(r"^step_(\d{8})$")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object: {path}")
    return value


def _resolve_stage2_resume_state(
    output_dir: Path,
    resume: str | Path,
    *,
    requested_config_sha256: str,
    max_steps: int,
    world_size: int,
    checkpoint_interval_steps: int,
    paired_schedule_profile: str | None = None,
) -> tuple[Path, int, dict[str, Any]]:
    state_root = output_dir / "checkpoints" / "state"
    requested = str(resume).strip()
    if requested == "latest":
        candidates: list[tuple[int, Path]] = []
        if state_root.is_dir():
            for child in state_root.iterdir():
                match = _STAGE2_STATE_DIRECTORY_PATTERN.fullmatch(child.name)
                if match is not None and child.is_dir():
                    candidates.append((int(match.group(1)), child.resolve()))
        if not candidates:
            raise RuntimeError(f"no finalized Stage-2 resume state exists in {state_root}")
        _step_from_name, state_dir = max(candidates)
    else:
        state_dir = Path(requested).expanduser().resolve()
    if not state_dir.is_dir():
        raise RuntimeError(f"Stage-2 resume state is not a directory: {state_dir}")
    match = _STAGE2_STATE_DIRECTORY_PATTERN.fullmatch(state_dir.name)
    if match is None:
        raise RuntimeError("Stage-2 resume directory must be named step_XXXXXXXX")
    if state_dir.parent.resolve() != state_root.resolve():
        raise RuntimeError("Stage-2 resume directory is outside this run output")
    step_from_name = int(match.group(1))
    payload = _read_json_object(
        state_dir / "trainer_state.json", label="Stage-2 trainer state"
    )
    required = {
        "schema",
        "status",
        "global_step",
        "next_step",
        "max_steps",
        "world_size",
        "checkpoint_interval_steps",
        "requested_config_sha256",
        "official_batches_consumed_per_rank",
        "paired_batches_consumed_per_rank",
        "accelerate_state",
        "policy_overlay",
    }
    if set(payload) != required:
        raise RuntimeError("Stage-2 trainer-state schema differs")
    if (
        payload.get("schema") != "policy_stage2_native_accelerate_state_v1"
        or payload.get("status") != "PASS"
    ):
        raise RuntimeError("Stage-2 trainer state is not PASS schema v1")
    step = int(payload["global_step"])
    if step != step_from_name or int(payload["next_step"]) != step + 1:
        raise RuntimeError("Stage-2 trainer state step/path differs")
    if not 0 < step < int(max_steps):
        raise RuntimeError("Stage-2 resume step is outside the active run")
    if step % int(checkpoint_interval_steps) != 0:
        raise RuntimeError("Stage-2 resume step is not a configured save boundary")
    fixed = {
        "max_steps": int(max_steps),
        "world_size": int(world_size),
        "checkpoint_interval_steps": int(checkpoint_interval_steps),
        "requested_config_sha256": str(requested_config_sha256),
        "official_batches_consumed_per_rank": step,
        "paired_batches_consumed_per_rank": (
            paired_active_count(step)
            if paired_schedule_profile == PAIR280_PROFILE_ID
            else step
        ),
        "accelerate_state": "model_optimizer_rng_and_registered_progress",
        "policy_overlay": "policy_overlay.pt",
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Stage-2 trainer state changed {key}")
    if not (state_dir / "policy_overlay.pt").is_file():
        raise RuntimeError("Stage-2 checkpoint lacks its policy overlay")
    return state_dir, step, payload


def _save_stage2_native_checkpoint(
    *,
    accelerator: Accelerator,
    output_dir: Path,
    progress: PolicyTrainingProgress,
    raw_training_module: PolicyTrainingModule,
    base_checkpoint: Path,
    regime: str,
    config: Mapping[str, Any],
    base_identity: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    checkpoint_interval_steps: int,
) -> Path:
    """Mirror native FastWAM's weights + ``Accelerator.save_state`` bundle."""

    step = int(progress.completed_step)
    if step <= 0:
        raise RuntimeError("cannot checkpoint Stage-2 before an optimizer step")
    state_root = output_dir / "checkpoints" / "state"
    target = state_root / f"step_{step:08d}"
    nonce_box: list[Any] = [str(os.getpid()) if accelerator.is_main_process else None]
    broadcast_object_list(nonce_box, from_process=0)
    nonce = str(nonce_box[0])
    staging = state_root / f".step_{step:08d}.tmp-{nonce}"
    accelerator.wait_for_everyone()
    preflight_error: list[Any] = [None]
    if accelerator.is_main_process:
        try:
            if target.exists():
                raise RuntimeError(
                    f"refusing to overwrite finalized checkpoint: {target}"
                )
            if staging.exists():
                raise RuntimeError(
                    f"refusing to reuse checkpoint staging path: {staging}"
                )
            staging.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            preflight_error[0] = f"{type(exc).__name__}: {exc}"
    broadcast_object_list(preflight_error, from_process=0)
    if preflight_error[0] is not None:
        raise RuntimeError(f"Stage-2 checkpoint preflight failed: {preflight_error[0]}")
    accelerator.wait_for_everyone()

    accelerator.save_state(output_dir=str(staging), safe_serialization=False)
    accelerator.wait_for_everyone()
    finalize_error: list[Any] = [None]
    if accelerator.is_main_process:
        try:
            save_policy_checkpoint(
                staging / "policy_overlay.pt",
                model=raw_training_module.model,
                conditioner=raw_training_module.conditioner,
                base_checkpoint=base_checkpoint,
                regime=regime,
                step=step,
                run_config=config,
                optimizer=None,
                include_base_sha256=True,
                verified_base_identity=base_identity,
                artifact_identities=identities,
            )
            _write_json(
                staging / "trainer_state.json",
                {
                    "schema": "policy_stage2_native_accelerate_state_v1",
                    "status": "PASS",
                    "global_step": step,
                    "next_step": step + 1,
                    "max_steps": int(progress.max_steps),
                    "world_size": int(progress.world_size),
                    "checkpoint_interval_steps": int(checkpoint_interval_steps),
                    "requested_config_sha256": progress.requested_config_sha256,
                    "official_batches_consumed_per_rank": step,
                    "paired_batches_consumed_per_rank": (
                        paired_active_count(step)
                        if progress.paired_schedule_profile == PAIR280_PROFILE_ID
                        else step
                    ),
                    "accelerate_state": "model_optimizer_rng_and_registered_progress",
                    "policy_overlay": "policy_overlay.pt",
                },
            )
            staging.replace(target)
            _write_json(
                state_root / "latest.json",
                {
                    "schema": "policy_stage2_latest_state_v1",
                    "status": "PASS",
                    "global_step": step,
                    "state_dir": str(target),
                },
            )
        except Exception as exc:
            finalize_error[0] = f"{type(exc).__name__}: {exc}"
    broadcast_object_list(finalize_error, from_process=0)
    if finalize_error[0] is not None:
        raise RuntimeError(f"Stage-2 checkpoint finalization failed: {finalize_error[0]}")
    accelerator.wait_for_everyone()
    return target


def _training_deliverable_status(*, formal: bool) -> dict[str, str]:
    """Describe artifacts produced by this completed invocation.

    A formal invocation reaches this helper only after all configured optimizer
    steps, update/gradient audits, and checkpoint serialization have completed.
    Marking that same run as ``NOT_STARTED`` made a valid formal checkpoint
    look incomplete to downstream gates.
    """

    return {
        "implementation": "PASS",
        "gradient_audit": "PASS",
        "short_update": "PASS",
        "rollout_load_execute": "PENDING_SEPARATE_SMOKE",
        "formal_long_training": "PASS" if formal else "NOT_STARTED",
    }


def _is_formal_training_config(config: Mapping[str, Any]) -> bool:
    """Resolve the formal-run bit from either supported config contract.

    The original policy configs carry both the top-level ``formal`` flag and
    ``execution.long_formal_training``.  Pair-280 deliberately derives from a
    frozen source config and authorizes the long run through the execution
    contract only.  Treating only the top-level flag as authoritative caused
    a completed 18,215-step run to be labelled ``SMOKE_COMPLETE`` even though
    its immutable execution contract, train log, and strict audit were all
    formal.  Both declarations are fail-closed when present together.
    """

    explicit = config.get("formal")
    if explicit is not None and not isinstance(explicit, bool):
        raise RuntimeError("formal must be a boolean when present")
    execution = config.get("execution", {})
    execution_formal: Any = None
    if execution is not None:
        if not isinstance(execution, Mapping):
            raise RuntimeError("execution must be a mapping")
        execution_formal = execution.get("long_formal_training")
        if execution_formal is not None and not isinstance(execution_formal, bool):
            raise RuntimeError("execution.long_formal_training must be a boolean")
    if (
        explicit is not None
        and execution_formal is not None
        and explicit is not execution_formal
    ):
        raise RuntimeError(
            "formal and execution.long_formal_training declarations disagree"
        )
    if explicit is not None:
        return explicit
    return bool(execution_formal)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _set_seed(seed: int, rank: int) -> None:
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def _new_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _seed_dataloader_worker_from_torch(worker_id: int) -> None:
    """Seed every worker RNG from DataLoader's isolated generator base seed."""

    del worker_id
    worker_seed = int(torch.initial_seed())
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def _official_loader_rng_contract(training_seed: int) -> dict[str, Any]:
    seed = int(training_seed)
    return {
        "status": "PASS",
        "sampler_seed": seed,
        "training_dataloader_generator_seed": seed + OFFICIAL_DATALOADER_SEED_OFFSET,
        "paired_dataloader_generator_seed": seed + PAIRED_DATALOADER_SEED_OFFSET,
        "identity_dataloader_generator_seed": seed + IDENTITY_DATALOADER_SEED_OFFSET,
        "main_process_data_seed_base": seed + OFFICIAL_MAIN_PROCESS_DATA_SEED_OFFSET,
        "worker_seed_source": "torch.initial_seed_from_isolated_dataloader_generator",
        "worker_rngs": ["python.random", "numpy", "torch"],
        "identity_loader_is_separate": True,
    }


def build_matched_c1_c3_stream_contract(
    config: Mapping[str, Any],
    *,
    base_identity: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Canonicalize every input that must match in a paired policy comparison.

    ``control`` and ``lambda_contrastive`` are deliberately absent: C1 and C3
    are allowed to differ only in that gradient coefficient.  Consequently two
    correctly paired runs produce the same contract SHA without sharing mutable
    iterator state or a checkpoint generated by the other branch.  The same
    contract also covers the pre-selection P-v1/P-v2 dev pair.  For that pair,
    both the candidate regime and its derived ActionDiT freeze switch are
    deliberately absent, and no not-yet-created selection manifest is required.
    """

    control = str(config.get("control", ""))
    p_mode_preselection = (
        control in {"p_v1", "p_v2"}
        and str(config.get("stage", "")) == "dev_pilot"
        and str(config.get("selection_role", "")) == "c1_lambda0"
        and config.get("p_mode_selection_manifest") is None
    )

    required_identities = [
        "base_lineage_manifest",
        "release_paired_binding_manifest",
        "dataset_stats",
        "official_manifest",
        "paired_action_manifest",
        "paired_action_audit",
        "paired_state_bank",
        "paired_text_cache",
        "paired_train_cache",
    ]
    if config["official"].get("text_cache_dir") is not None:
        required_identities.append("official_text_cache_binding_manifest")
    engineering_smoke = (
        not bool(config.get("formal", False))
        and str(config.get("stage", "")) == "smoke"
        and control in {"c1_architecture_only", "c3_ours"}
    )
    if not engineering_smoke and not p_mode_preselection:
        required_identities.append("p_mode_selection_manifest")
    if bool(config.get("formal", False)):
        required_identities.append("formal_protocol_lock_manifest")
    missing = [name for name in required_identities if name not in identities]
    if missing:
        raise ValueError(f"matched C1/C3 stream contract lacks identities: {missing}")
    official = config["official"]
    paired = config["paired"]
    training = config["training"]
    policy = config["policy"]
    optimizer = config["optimizer"]
    initialization = {
        "head_init_mode": str(policy["head_init_mode"]),
        "head_init_seed": int(policy["head_init_seed"]),
        "adapter_init_seed": int(policy["adapter_init_seed"]),
        "p_mode_selection": (
            "NOT_APPLICABLE_PRESELECTION"
            if p_mode_preselection
            else (
                "NOT_APPLICABLE_ENGINEERING_SMOKE"
                if engineering_smoke
                else "HASH_BOUND_DEV_SELECTION"
            )
        ),
    }
    if not p_mode_preselection:
        initialization["regime"] = str(policy["regime"]).replace("-", "_")

    body = {
        "schema": (
            "policy_release_pmode_preselection_matched_stream_v1"
            if p_mode_preselection
            else "policy_release_c1_c3_matched_stream_v1"
        ),
        "base_checkpoint_sha256": str(base_identity["sha256"]),
        "artifact_sha256": {
            name: str(identities[name]["sha256"]) for name in required_identities
        },
        "training_seed": int(training["seed"]),
        "initialization": initialization,
        "official_stream": {
            "selection_mode": str(official["selection_mode"]),
            "sampling_mode": str(official["sampling_mode"]),
            "expected_counts_per_task": {
                "clean": int(official["expected_clean_per_task"]),
                "official_random": int(official["expected_random_per_task"]),
            },
            "sampler": "ThreeTaskRoundRobinSampler",
            "sampler_seed": int(training["seed"]),
            "balanced_tasks": bool(official["balanced_tasks"]),
            "text_conditioning": (
                "precomputed_hash_bound"
                if official.get("text_cache_dir") is not None
                else "on_the_fly_engineering_only"
            ),
        },
        "paired_stream": {
            "supervision_mode": str(paired["supervision_mode"]),
            "layer": int(paired["layer"]),
            "variants": list(paired["variants"]),
            "split": str(paired["split"]),
            "sampler": "SameTaskPhysicalStateBatchSampler",
            "sampler_seed": int(training["seed"]),
            "balanced_round_robin": True,
        },
        "optimizer": {
            "name": str(optimizer["name"]),
            "head_adapter_lr": float(optimizer["head_adapter_lr"]),
            "action_dit_lr": float(optimizer["action_dit_lr"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "betas": [float(value) for value in optimizer["betas"]],
            "lr_scheduler": str(optimizer["lr_scheduler"]),
        },
        "step_budget": {
            "max_steps": int(training["max_steps"]),
            "world_size": int(training["world_size"]),
            "official_batch_size_per_rank": int(training["official_batch_size"]),
            "paired_groups_per_batch_per_rank": int(
                training["paired_groups_per_batch"]
            ),
            "gradient_accumulation_steps": int(
                training["gradient_accumulation_steps"]
            ),
            "effective_official_global_batch": int(
                training["effective_official_global_batch"]
            ),
            "effective_paired_groups_per_step": int(
                training["effective_paired_groups_per_step"]
            ),
        },
        "checkpointing": {
            "engine": "accelerate_save_state_load_state",
            "save_every": int(training.get("save_every", 0)),
            "save_optimizer": bool(training.get("save_optimizer", False)),
            "dual_stream_resume_offset": "skip_first_prepared_batches",
        },
        "rng": {
            **_official_loader_rng_contract(int(training["seed"])),
            "stage2_per_step": _stage2_step_rng_contract(
                int(training["seed"]),
                max_steps=int(training["max_steps"]),
                world_size=int(training["world_size"]),
            ),
        },
    }
    return {
        "status": "PASS",
        "only_permitted_cross_control_difference": (
            "policy_regime_and_action_dit_freeze"
            if p_mode_preselection
            else "lambda_contrastive_0.0_vs_0.1"
        ),
        "contract": body,
        "sha256": canonical_json_sha256(body),
    }


@contextmanager
def _isolated_cpu_data_rng(seed: int):
    """Isolate main-process dataset transforms when num_workers is zero."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            torch.manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def _isolated_torch_rng(seed: int, device: torch.device):
    """Run one stream without advancing any other stream's RNG state."""

    devices: list[int] = []
    if device.type == "cuda":
        devices = [
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        ]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        yield


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        upper = value.upper()
        return (
            upper.startswith("__REQUIRED_")
            or upper.startswith("__SELECT_")
            or "PLACEHOLDER" in upper
        )
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return False


def validate_run_config(config: Mapping[str, Any]) -> None:
    """Validate the executable trainer contract, independent of model loading."""

    if int(config.get("schema_version", -1)) != 3:
        raise ValueError("Policy v2 run schema_version must be 3")
    tasks = tuple(str(value) for value in config.get("tasks", ()))
    if tasks != TASKS:
        raise ValueError(f"task order must be exactly {TASKS}, got {tasks}")
    formal = bool(config.get("formal", False))
    if formal and _contains_placeholder(config):
        raise ValueError("formal config still contains a required placeholder")
    if not config.get("base_lineage_manifest"):
        raise ValueError("author-release base_lineage_manifest is required")
    if not config.get("release_paired_binding_manifest"):
        raise ValueError("release_paired_binding_manifest is required")
    if formal and not config.get("formal_protocol_lock_manifest"):
        raise ValueError("formal runs require formal_protocol_lock_manifest")

    control = str(config.get("control", ""))
    execution = config.get("execution")
    if control == "c0_original":
        raise ValueError(
            "C0 is intentionally routed to the separate native action-only continuation "
            "runner; the adapter trainer must not relabel it"
        )
    if isinstance(execution, Mapping) and bool(execution.get("fail_closed", False)):
        raise ValueError(str(execution.get("reason", "config is explicitly fail-closed")))
    if control not in {
        "p_v1",
        "p_v2",
        "c1_architecture_only",
        "c2_naive_aug",
        "c3_ours",
    }:
        raise ValueError(f"unsupported adapter trainer control: {control!r}")

    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("config.policy must be a mapping")
    regime = str(policy.get("regime", "")).replace("-", "_")
    if regime not in {"p_v1", "p_v2"}:
        raise ValueError("policy.regime must be p_v1 or p_v2")
    if int(policy.get("content_layer", -1)) != EXACT_LAYER:
        raise ValueError("the prototype is locked to Layer-16")
    if int(policy.get("queries", -1)) != EXACT_QUERIES:
        raise ValueError("the prototype is locked to eight content queries")
    if int(policy.get("content_dim", -1)) != EXACT_CONTENT_DIM:
        raise ValueError("the prototype is locked to content_dim=384")
    head_init_mode = str(policy.get("head_init_mode", ""))
    if head_init_mode not in {"random", "pretrained"}:
        raise ValueError("policy.head_init_mode must be random or pretrained")
    if head_init_mode == "random":
        if policy.get("head_init") is not None:
            raise ValueError("random Head initialization requires policy.head_init=null")
        if not isinstance(policy.get("head_init_seed"), int):
            raise ValueError("random Head initialization requires integer policy.head_init_seed")
    else:
        if not policy.get("head_init"):
            raise ValueError("legacy_pretrained Head initialization requires policy.head_init")
        if formal:
            raise ValueError(
                "formal Policy v2 training forbids legacy representation Head checkpoints; "
                "legacy checkpoints remain inference/non-formal ablation only"
            )
    if not isinstance(policy.get("adapter_init_seed"), int):
        raise ValueError("policy.adapter_init_seed must be an integer")

    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        raise ValueError("config.loss must be a mapping")
    if str(loss.get("action")) != "native_flow_matching_mse":
        raise ValueError("official action objective must be native_flow_matching_mse")
    if loss.get("video") not in {None, "null", "disabled"}:
        raise ValueError("the frozen-video prototype must not optimize a video loss")
    temperature = float(loss.get("temperature", -1.0))
    if not math.isclose(temperature, EXACT_TEMPERATURE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("contrastive temperature must be exactly 0.07")
    coefficient = float(loss.get("lambda_contrastive", -1.0))
    paired_action_coefficient = float(loss.get("lambda_paired_action", -1.0))
    if control == "c1_architecture_only":
        if coefficient != 0.0 or paired_action_coefficient != 0.0:
            raise ValueError("C1 Architecture-only requires both paired loss weights to be zero")
    elif control == "c2_naive_aug":
        if coefficient != 0.0 or paired_action_coefficient != 1.0:
            raise ValueError("C2 requires lambda_contrastive=0 and lambda_paired_action=1")
    elif control == "c3_ours":
        if coefficient != EXACT_LAMBDA or paired_action_coefficient != 0.0:
            raise ValueError("C3 requires contrastive=0.1 and paired_action=0")
    else:
        selection_role = str(config.get("selection_role", ""))
        if paired_action_coefficient != 0.0:
            raise ValueError("P-v1/P-v2 must not use paired action supervision")
        if selection_role == "c1_lambda0":
            if coefficient != 0.0:
                raise ValueError(
                    "P-v1/P-v2 selection pilots must use the C1 lambda=0 objective"
                )
        elif selection_role == "engineering_method_smoke":
            if formal or coefficient != EXACT_LAMBDA:
                raise ValueError(
                    "engineering method smoke requires non-formal contrastive=0.1"
                )
        else:
            raise ValueError(
                "P-v1/P-v2 selection_role must be c1_lambda0 or "
                "engineering_method_smoke"
            )

    official = config.get("official")
    if not isinstance(official, Mapping):
        raise ValueError("config.official must be a mapping")
    for key in ("dataset_root", "dataset_stats", "canonical_task_manifest"):
        if not official.get(key):
            raise ValueError(f"official.{key} is required")
    if str(official.get("selection_mode", "")) != "full_550_per_task":
        raise ValueError("Stage-2 official stream must use full_550_per_task")
    if (
        int(official.get("expected_clean_per_task", -1)) != 50
        or int(official.get("expected_random_per_task", -1)) != 500
        or int(official.get("expected_total_per_task", -1)) != 550
    ):
        raise ValueError("Stage-2 official stream must declare 50 Clean + 500 Random per task")
    if str(official.get("sampling_mode")) not in {"all_frames", "episode_anchor"}:
        raise ValueError("official.sampling_mode must be all_frames or episode_anchor")
    if not bool(official.get("balanced_tasks", False)):
        raise ValueError("exact three-task runs require official.balanced_tasks=true")
    if formal:
        if not bool(official.get("domain_verified", False)):
            raise ValueError("formal runs require externally verified official domain provenance")
        if not official.get("text_cache_dir"):
            raise ValueError("formal runs require precomputed official prompt embeddings")
        if bool(official.get("on_the_fly_text_smoke", False)):
            raise ValueError("formal runs cannot use on-the-fly smoke text encoding")
    elif not bool(official.get("on_the_fly_text_smoke", False)) and not official.get(
        "text_cache_dir"
    ):
        raise ValueError("smoke needs either text_cache_dir or on_the_fly_text_smoke=true")
    if official.get("text_cache_dir") and not official.get(
        "text_cache_binding_manifest"
    ):
        raise ValueError(
            "precomputed official prompt embeddings require a release text-cache binding"
        )

    paired = config.get("paired")
    if not isinstance(paired, Mapping):
        raise ValueError("config.paired must be a mapping")
    expected_mode = {
        "c1_architecture_only": "contrastive",
        "c2_naive_aug": "action",
        "c3_ours": "contrastive",
        "p_v1": "contrastive",
        "p_v2": "contrastive",
    }[control]
    mode = str(paired.get("supervision_mode", ""))
    if mode != expected_mode:
        raise ValueError(f"{control} requires paired.supervision_mode={expected_mode!r}")
    if str(paired.get("split", "")) != "train":
        raise ValueError("Stage-2 paired.split must be train")
    if paired.get("protocol_id") != POLICY_PROTOCOL_ID:
        raise ValueError(f"paired.protocol_id must be {POLICY_PROTOCOL_ID!r}")
    if tuple(str(value) for value in paired.get("variants", ())) != POLICY_VARIANTS:
        raise ValueError(f"paired variants must be exact ordered {POLICY_VARIANTS}")
    if int(paired.get("view_count", -1)) != POLICY_VIEW_COUNT:
        raise ValueError("paired.view_count must be 4 scene versions")
    if paired.get("r3_role") != POLICY_R3_ROLE:
        raise ValueError("paired.r3_role must be training_positive")
    if int(paired.get("camera_count", -1)) != POLICY_CAMERA_COUNT:
        raise ValueError("paired.camera_count must be 3")
    if tuple(str(value) for value in paired.get("camera_names", ())) != POLICY_CAMERA_NAMES:
        raise ValueError(f"paired.camera_names must be {POLICY_CAMERA_NAMES}")
    if int(paired.get("native_fps", -1)) != POLICY_NATIVE_FPS:
        raise ValueError("paired.native_fps must be 50; interpolation is forbidden")
    if int(paired.get("action_steps", -1)) != POLICY_ACTION_STEPS:
        raise ValueError("paired.action_steps must be 32")
    if int(paired.get("action_dim", -1)) != POLICY_ACTION_DIM:
        raise ValueError("paired.action_dim must be 14")
    if str(paired.get("temporal_resampling", "")) != "none":
        raise ValueError("paired.temporal_resampling must be none")
    if paired.get("native_action_targets") is not True:
        raise ValueError("paired.native_action_targets must be true")
    if "r3_excluded" in paired:
        raise ValueError("legacy paired.r3_excluded is forbidden in Policy v2")
    if mode == "contrastive":
        if int(paired.get("layer", -1)) != EXACT_LAYER or not paired.get("cache"):
            raise ValueError("C3 paired cache must provide Layer-16")
        for key in (
            "action_root",
            "action_manifest",
            "action_audit",
            "state_bank",
            "text_cache_dir",
        ):
            if not paired.get(key):
                raise ValueError(
                    f"dual-stream contrastive-cache controls require paired.{key}"
                )
    elif mode == "action":
        for key in (
            "action_root",
            "action_manifest",
            "action_audit",
            "state_bank",
            "text_cache_dir",
        ):
            if not paired.get(key):
                raise ValueError(f"C2 requires paired.{key}")
        if paired.get("cache"):
            raise ValueError("C2 action mode must not consume a frozen token cache")
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("config.optimizer must be a mapping")
    if str(optimizer.get("name", "")).lower() != "adamw":
        raise ValueError("Stage-2 optimizer must be AdamW")
    if str(optimizer.get("trainable_parameter_dtype", "")).lower() != "fp32":
        raise ValueError(
            "trainable parameters must remain fp32 so AdamW has fp32 master/state precision"
        )
    if float(optimizer.get("head_adapter_lr", -1.0)) != 1e-4:
        raise ValueError("head/adapter LR must be 1e-4")
    if float(optimizer.get("action_dit_lr", -1.0)) != 1e-5:
        raise ValueError("ActionDiT LR must be 1e-5")
    if str(optimizer.get("lr_scheduler", "")) != "constant":
        raise ValueError("Stage-2 optimizer.lr_scheduler must be constant")
    if float(optimizer.get("weight_decay", -1.0)) != 0.0:
        raise ValueError("Stage-2 weight_decay must be exactly zero")
    if tuple(float(value) for value in optimizer.get("betas", ())) != (0.9, 0.95):
        raise ValueError("Stage-2 AdamW betas must be exactly [0.9, 0.95]")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("config.training must be a mapping")
    max_steps = int(training.get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError("training.max_steps must be positive")
    training_seed = training.get("seed")
    if not isinstance(training_seed, int) or isinstance(training_seed, bool):
        raise ValueError("training.seed must be an integer")
    if not formal and max_steps < len(TASKS):
        raise ValueError("three-task smoke needs at least three steps for exact task coverage")
    if int(training.get("official_batch_size", 0)) <= 0:
        raise ValueError("training.official_batch_size must be positive")
    if mode != "none" and int(training.get("paired_groups_per_batch", 0)) < 2:
        raise ValueError("paired batches require at least two physical states")
    for key in (
        "world_size",
        "gradient_accumulation_steps",
        "effective_official_global_batch",
        "effective_paired_groups_per_step",
    ):
        value = training.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if int(training["gradient_accumulation_steps"]) != 1:
        raise ValueError("Stage-2 gradient_accumulation_steps must be exactly one")
    if int(training["effective_official_global_batch"]) != (
        int(training["official_batch_size"]) * int(training["world_size"])
    ):
        raise ValueError("effective official global batch must equal local batch x world size")
    if int(training["effective_paired_groups_per_step"]) != (
        int(training["paired_groups_per_batch"]) * int(training["world_size"])
    ):
        raise ValueError("effective paired groups must equal local groups x world size")
    if official.get("text_cache_dir") is None and int(training.get("num_workers", -1)) != 0:
        raise ValueError("on-the-fly text smoke requires num_workers=0")
    if training.get("separate_stream_rng") is not True:
        raise ValueError("training.separate_stream_rng=true is required for matched controls")
    if training.get("preserve_official_sequence_across_controls") is not True:
        raise ValueError(
            "training.preserve_official_sequence_across_controls=true is required"
        )
    save_every = training.get("save_every", 0)
    if not isinstance(save_every, int) or isinstance(save_every, bool) or save_every < 0:
        raise ValueError("training.save_every must be a non-negative integer")
    if save_every > max_steps:
        raise ValueError("training.save_every cannot exceed training.max_steps")
    if "resume" in training:
        resume_value = training["resume"]
        if resume_value is not None and resume_value != "" and not isinstance(
            resume_value, str
        ):
            raise ValueError(
                "training.resume must be null or a state-directory string"
            )
    if save_every > 0 and training.get("save_optimizer") is not True:
        raise ValueError("periodic native checkpoints require save_optimizer=true")
    sampling_profile = paired.get("sampling_profile")
    if sampling_profile is not None:
        if sampling_profile != PAIR280_PROFILE_ID:
            raise ValueError("unknown paired sampling_profile")
        if control != "c3_ours" or regime != "p_v2":
            raise ValueError("Pair-280 first execution is locked to C3/P-v2")
        pair_schedule = paired.get("schedule")
        if not isinstance(pair_schedule, Mapping):
            raise ValueError("Pair-280 paired.schedule is required")
        expected_pair_schedule = {
            "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
            "physical_state_groups": PAIR280_GROUPS,
            "paired_epochs": 10,
            "active_steps": PAIR280_ACTIVE_STEPS,
            "total_steps": 18_215,
            "global_groups_per_active_step": 16,
            "sampler": PAIR280_SAMPLER_ID,
            "active_step_distribution": "floor_difference_v1",
        }
        if dict(pair_schedule) != expected_pair_schedule:
            raise ValueError("Pair-280 paired.schedule differs from the locked protocol")
        if paired.get("cache_format") != PAIR280_CACHE_STORAGE:
            raise ValueError("Pair-280 cache_format changed")
        pair280_smoke = bool(paired.get("engineering_smoke", False))
        if max_steps != (3 if pair280_smoke else 18_215) or int(training["world_size"]) != 8:
            raise ValueError(
                "Pair-280 requires three smoke steps or exactly 18,215 formal-profile steps on eight ranks"
            )
        if (
            int(training["official_batch_size"]) != 16
            or int(training["paired_groups_per_batch"]) != 2
            or int(training["effective_official_global_batch"]) != 128
            or int(training["effective_paired_groups_per_step"]) != 16
        ):
            raise ValueError("Pair-280 requires official global128 and paired global16")
        expected_save_every = 3 if pair280_smoke else 2_000
        if save_every != expected_save_every or training.get("save_optimizer") is not True:
            raise ValueError(
                f"Pair-280 requires save_every={expected_save_every} with optimizer state"
            )
    _stage2_step_rng_contract(
        training_seed,
        max_steps=max_steps,
        world_size=int(training["world_size"]),
    )


class PolicyTrainingModule(nn.Module):
    """Expose the exact custom objective through one DDP-visible forward."""

    def __init__(
        self,
        model: nn.Module,
        runtime,
        *,
        paired_supervision_mode: str,
        lambda_contrastive: float,
        lambda_paired_action: float,
        temperature: float,
        training_seed: int,
        process_index: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.conditioner = runtime.conditioner
        self.runtime = runtime
        self.paired_supervision_mode = str(paired_supervision_mode)
        self.lambda_contrastive = float(lambda_contrastive)
        self.lambda_paired_action = float(lambda_paired_action)
        self.temperature = float(temperature)
        # Fail during construction, rather than after a potentially expensive
        # model load, if this run key cannot be represented by the RNG policy.
        stage2_step_rng_seed(
            training_seed,
            0,
            process_index=process_index,
        )
        self.training_seed = int(training_seed)
        self.process_index = int(process_index)
        self._forward_index = 0
        if self.paired_supervision_mode not in {"none", "action", "contrastive"}:
            raise ValueError("invalid paired supervision mode")
        if self.paired_supervision_mode == "contrastive" and self.lambda_paired_action != 0.0:
            raise ValueError("contrastive-cache stream must not carry paired action weight")

    def forward(
        self,
        official_batch: Mapping[str, Any],
        paired_batch: Mapping[str, Any] | None,
        *,
        paired_active: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        step_index = self._forward_index
        self._forward_index += 1
        device = torch.device(
            getattr(self.model, "device", next(self.model.parameters()).device)
        )
        official_seed = stage2_step_rng_seed(
            self.training_seed,
            step_index,
            process_index=self.process_index,
            stream="official",
        )
        paired_seed = stage2_step_rng_seed(
            self.training_seed,
            step_index,
            process_index=self.process_index,
            stream="paired",
        )
        with _isolated_torch_rng(official_seed, device):
            action_loss, action_diagnostics = official_action_loss(
                self.model, self.runtime, official_batch
            )
        paired_action = action_loss.new_zeros(())
        if self.paired_supervision_mode == "contrastive":
            if paired_active:
                if paired_batch is None:
                    raise ValueError("active contrastive step received no paired batch")
                contrastive_loss, contrastive_diagnostics = paired_contrastive_loss(
                    self.conditioner,
                    paired_batch,
                    temperature=self.temperature,
                )
            else:
                if paired_batch is not None:
                    raise ValueError("inactive contrastive step unexpectedly received paired data")
                contrastive_loss = action_loss.new_zeros(())
                contrastive_diagnostics = {
                    "loss_contrastive": 0.0,
                    "positive_similarity": 0.0,
                    "negative_similarity": 0.0,
                    "positives_per_anchor": 0,
                    "r3_training_positive": True,
                    "paired_clean_layer16_distribution": None,
                }
            paired_action_diagnostics = {
                "loss_paired_action": 0.0,
                "paired_layer16_distribution": None,
            }
        elif self.paired_supervision_mode == "action":
            if paired_batch is None:
                raise ValueError("C2 action run received no paired batch")
            with _isolated_torch_rng(paired_seed, device):
                paired_action, paired_action_diagnostics = paired_action_loss(
                    self.model,
                    self.runtime,
                    paired_batch,
                )
            contrastive_loss = action_loss.new_zeros(())
            contrastive_diagnostics = {
                "loss_contrastive": 0.0,
                "positive_similarity": 0.0,
                "negative_similarity": 0.0,
                "positives_per_anchor": 0,
                "r3_training_positive": True,
                "paired_clean_layer16_distribution": None,
            }
        else:
            if paired_batch is not None:
                raise ValueError("C1 architecture control must not consume paired data")
            contrastive_loss = action_loss.new_zeros(())
            contrastive_diagnostics = {
                "loss_contrastive": 0.0,
                "positive_similarity": 0.0,
                "negative_similarity": 0.0,
                "positives_per_anchor": 0,
                "r3_training_positive": False,
                "paired_clean_layer16_distribution": None,
            }
            paired_action_diagnostics = {
                "loss_paired_action": 0.0,
                "paired_layer16_distribution": None,
            }
        total = (
            action_loss
            + self.lambda_contrastive * contrastive_loss
            + self.lambda_paired_action * paired_action
        )
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("policy total loss is non-finite")
        return total, action_loss, contrastive_loss, {
            **action_diagnostics,
            **contrastive_diagnostics,
            **paired_action_diagnostics,
            "paired_supervision_mode": self.paired_supervision_mode,
            "paired_contrastive_gradient_enabled": (
                self.paired_supervision_mode == "contrastive"
                and self.lambda_contrastive > 0.0
                and bool(paired_active)
            ),
            "paired_contrastive_active": bool(
                self.paired_supervision_mode == "contrastive" and paired_active
            ),
            "step_rng_policy_id": STAGE2_STEP_RNG_POLICY_ID,
            "step_rng_step_index": step_index,
            "step_rng_training_seed": self.training_seed,
            "step_rng_process_index": self.process_index,
            "official_rng_seed": official_seed,
            "paired_rng_seed": paired_seed if self.paired_supervision_mode == "action" else None,
            "loss_total": float(total.detach().item()),
        }

    def set_forward_index_for_resume(self, completed_step: int) -> None:
        completed = int(completed_step)
        if completed < 0:
            raise ValueError("completed_step must be non-negative")
        self._forward_index = completed


class PolicyTrainingProgress:
    """Accelerate-checkpointed Stage-2 step, stream, and audit progress.

    Model parameters, AdamW state and all process RNG states are handled by
    ``Accelerator.save_state/load_state`` exactly as in the native FastWAM
    trainer.  This registered object contains only state that is specific to
    our dual-stream loop and therefore has no counterpart in the native
    single-stream trainer.
    """

    SCHEMA = "policy_stage2_dual_stream_progress_v1"
    PAIR280_SCHEMA = "policy_stage2_dual_stream_progress_pair280_v2"

    def __init__(
        self,
        *,
        max_steps: int,
        requested_config_sha256: str,
        world_size: int,
        effective_official_global_batch: int,
        effective_paired_groups_per_step: int,
        paired_supervision_mode: str,
        paired_schedule_profile: str | None = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.requested_config_sha256 = str(requested_config_sha256)
        self.world_size = int(world_size)
        self.effective_official_global_batch = int(
            effective_official_global_batch
        )
        self.effective_paired_groups_per_step = int(
            effective_paired_groups_per_step
        )
        self.paired_supervision_mode = str(paired_supervision_mode)
        self.paired_schedule_profile = (
            None if paired_schedule_profile is None else str(paired_schedule_profile)
        )
        self.completed_step = 0
        self.rows: list[dict[str, Any]] = []
        self.gradient_steps: list[dict[str, Any]] = []
        self.official_distribution = DistributionAccumulator()
        self.paired_distribution = DistributionAccumulator()
        self.seen_official_tasks: list[str] = []
        self.seen_paired_tasks: list[str] = []
        self.seen_official_sample_ids: list[str] = []
        self.seen_paired_state_ids: list[str] = []
        self.positive_action_signal_steps = 0
        self.zero_action_signal_steps = 0

    def state_dict(self) -> dict[str, Any]:
        result = {
            "schema": (
                self.PAIR280_SCHEMA
                if self.paired_schedule_profile == PAIR280_PROFILE_ID
                else self.SCHEMA
            ),
            "completed_step": int(self.completed_step),
            "max_steps": self.max_steps,
            "requested_config_sha256": self.requested_config_sha256,
            "world_size": self.world_size,
            "effective_official_global_batch": self.effective_official_global_batch,
            "effective_paired_groups_per_step": self.effective_paired_groups_per_step,
            "paired_supervision_mode": self.paired_supervision_mode,
            "rows": list(self.rows),
            "gradient_steps": list(self.gradient_steps),
            "official_distribution": self.official_distribution.state_dict(),
            "paired_distribution": self.paired_distribution.state_dict(),
            "seen_official_tasks": list(self.seen_official_tasks),
            "seen_paired_tasks": list(self.seen_paired_tasks),
            "seen_official_sample_ids": list(self.seen_official_sample_ids),
            "seen_paired_state_ids": list(self.seen_paired_state_ids),
            "positive_action_signal_steps": int(
                self.positive_action_signal_steps
            ),
            "zero_action_signal_steps": int(self.zero_action_signal_steps),
        }
        if self.paired_schedule_profile is not None:
            result["paired_schedule_profile"] = self.paired_schedule_profile
        return result

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = set(self.state_dict())
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("Stage-2 progress checkpoint schema differs")
        expected_schema = (
            self.PAIR280_SCHEMA
            if self.paired_schedule_profile == PAIR280_PROFILE_ID
            else self.SCHEMA
        )
        if state.get("schema") != expected_schema:
            raise ValueError("Stage-2 progress checkpoint kind differs")
        fixed = {
            "max_steps": self.max_steps,
            "requested_config_sha256": self.requested_config_sha256,
            "world_size": self.world_size,
            "effective_official_global_batch": self.effective_official_global_batch,
            "effective_paired_groups_per_step": self.effective_paired_groups_per_step,
            "paired_supervision_mode": self.paired_supervision_mode,
        }
        if self.paired_schedule_profile is not None:
            fixed["paired_schedule_profile"] = self.paired_schedule_profile
        for key, expected in fixed.items():
            if state.get(key) != expected:
                raise ValueError(f"Stage-2 progress checkpoint changed {key}")
        completed = int(state["completed_step"])
        if not 0 < completed < self.max_steps:
            raise ValueError("resume step must be inside the configured training run")
        rows = [dict(row) for row in state["rows"]]
        gradient_steps = [dict(row) for row in state["gradient_steps"]]
        expected_steps = list(range(1, completed + 1))
        if [int(row.get("step", -1)) for row in rows] != expected_steps:
            raise ValueError("resumed train-log steps are not exact and contiguous")
        if [int(row.get("step", -1)) for row in gradient_steps] != expected_steps:
            raise ValueError("resumed gradient-audit steps are not exact and contiguous")

        seen_official_tasks = [str(value) for value in state["seen_official_tasks"]]
        seen_paired_tasks = [str(value) for value in state["seen_paired_tasks"]]
        seen_official_ids = [
            str(value) for value in state["seen_official_sample_ids"]
        ]
        seen_paired_ids = [str(value) for value in state["seen_paired_state_ids"]]
        if len(seen_official_ids) != completed * self.effective_official_global_batch:
            raise ValueError("resumed official sample count differs from completed steps")
        if len(seen_official_tasks) != len(seen_official_ids):
            raise ValueError("resumed official task/sample counts differ")
        expected_paired_steps = (
            paired_active_count(completed)
            if self.paired_schedule_profile == PAIR280_PROFILE_ID
            else completed
        )
        expected_paired = (
            0
            if self.paired_supervision_mode == "none"
            else expected_paired_steps * self.effective_paired_groups_per_step
        )
        if len(seen_paired_ids) != expected_paired:
            raise ValueError("resumed paired-state count differs from completed steps")
        if len(seen_paired_tasks) != len(seen_paired_ids):
            raise ValueError("resumed paired task/state counts differ")

        row_official_ids = [
            item
            for row in rows
            for item in str(row.get("official_sample_ids", "")).split(";")
            if item
        ]
        row_paired_ids = [
            item
            for row in rows
            for item in str(row.get("paired_physical_state_ids", "")).split(";")
            if item
        ]
        if row_official_ids != seen_official_ids or row_paired_ids != seen_paired_ids:
            raise ValueError("resumed sequence histories differ from train-log rows")

        positive = int(state["positive_action_signal_steps"])
        zero = int(state["zero_action_signal_steps"])
        if positive < 0 or zero < 0 or positive + zero != completed:
            raise ValueError("resumed action-signal counters differ from completed steps")
        official_distribution = DistributionAccumulator.from_state_dict(
            state["official_distribution"]
        )
        paired_distribution = DistributionAccumulator.from_state_dict(
            state["paired_distribution"]
        )
        if official_distribution.tasks != seen_official_tasks:
            raise ValueError("resumed official distribution task history differs")
        if paired_distribution.tasks != seen_paired_tasks:
            raise ValueError("resumed paired distribution task history differs")

        self.completed_step = completed
        self.rows[:] = rows
        self.gradient_steps[:] = gradient_steps
        self.official_distribution = official_distribution
        self.paired_distribution = paired_distribution
        self.seen_official_tasks[:] = seen_official_tasks
        self.seen_paired_tasks[:] = seen_paired_tasks
        self.seen_official_sample_ids[:] = seen_official_ids
        self.seen_paired_state_ids[:] = seen_paired_ids
        self.positive_action_signal_steps = positive
        self.zero_action_signal_steps = zero


class _CyclingIterator:
    def __init__(self, source) -> None:
        self.source = source
        self.iterator = iter(source)
        self.cycles = 0

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.cycles += 1
            self.iterator = iter(self.source)
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise RuntimeError("official stream is empty") from exc


def _dual_stream_cycle_audit(official_iterator: _CyclingIterator) -> dict[str, int]:
    """Report actual iterator restarts without referring to a retired wrapper.

    The official loader is the only cycling iterator in the current trainer.
    Paired loaders are materialized to their exact required length (and the
    Pair-280 loader is consumed only on its preregistered active steps), so a
    paired restart would be a protocol error rather than an allowed cycle.
    """

    cycles = int(official_iterator.cycles)
    if cycles < 0:
        raise ValueError("official iterator cycle count is negative")
    return {"official": cycles, "paired": 0}


def _task_values(batch: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = batch.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _integer_values(batch: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = batch.get(key)
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _expected_sha(config: Mapping[str, Any], name: str) -> str | None:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    value = artifacts.get(name)
    if isinstance(value, Mapping):
        value = value.get("sha256")
    if value is None:
        value = artifacts.get(f"{name}_sha256")
    if value is None and name == "official_manifest":
        value = artifacts.get("official_task_manifest_sha256")
    if value is None:
        return None
    digest = str(value)
    if _contains_placeholder(digest):
        return None
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"artifacts.{name} has an invalid SHA-256")
    return digest


def _audit_path(
    config: Mapping[str, Any],
    name: str,
    path: str | Path,
    *,
    required_for_rollout: bool,
) -> dict[str, Any]:
    identity = artifact_identity(_resolve_path(path))
    expected = _expected_sha(config, name)
    if expected is not None and identity["sha256"] != expected:
        raise ValueError(
            f"artifact {name} SHA-256 mismatch: {identity['sha256']} vs {expected}"
        )
    identity["required_for_rollout"] = bool(required_for_rollout)
    identity["verification_status"] = "PASS"
    return identity


def _resolve_component_path(model, config: Mapping[str, Any], name: str) -> Path:
    model_paths = dict(getattr(model, "model_paths", {}) or {})
    candidate = model_paths.get(name)
    if candidate not in {None, "SKIPPED_PRETRAIN"}:
        return _resolve_path(candidate)
    base = _resolve_path(config["model_base_path"])
    defaults = {
        "vae": base
        / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
        "text_encoder": base
        / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
        "tokenizer": base / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
    }
    return defaults[name].resolve()


def _scalar_mean(accelerator: Accelerator, value: float) -> float:
    tensor = torch.tensor(float(value), device=accelerator.device, dtype=torch.float64)
    reduced = accelerator.reduce(tensor, reduction="mean")
    return float(reduced.item())


def _resolve_prepared_raw_module(
    prepared: nn.Module, original: nn.Module
) -> nn.Module:
    """Verify the prepared wrapper chain without importing optional DeepSpeed.

    ``Accelerator.unwrap_model`` imports DeepSpeed merely to test its wrapper
    type.  Some otherwise valid FastWAM environments contain an optional
    DeepSpeed installation whose import requires ``CUDA_HOME``, even though
    this runner uses plain single-process/DDP training.  The pre-``prepare``
    module is already the authoritative underlying object, so retain it and
    prove that every prepared wrapper points back to it.
    """

    current = prepared
    visited: set[int] = set()
    while current is not original:
        identity = id(current)
        if identity in visited:
            raise RuntimeError("prepared module wrapper chain contains a cycle")
        visited.add(identity)
        child = getattr(current, "module", None)
        if not isinstance(child, nn.Module):
            raise RuntimeError(
                "prepared training module is neither the original module nor "
                "a plain module-wrapper chain"
            )
        current = child
    return original


def _distribution_report(
    official: DistributionAccumulator,
    paired: DistributionAccumulator,
    *,
    supervision_mode: str,
) -> dict[str, Any]:
    official_stats = official.finalize()
    paired_stats = paired.finalize()
    return {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ONLY_POLICY_NATIVE50HZ",
        "official_clean_claim_supported": True,
        "official_domain_partition_verified": True,
        "official_domain_label": "protocol_v2_hash_bound_range_partition",
        "official_domain_verification_scope": (
            "episode_index_ranges_in_hash_bound_release"
        ),
        "intrinsic_metadata_domain_field": False,
        "reason": (
            "Clean/Random membership is verified by protocol-v2 episode ranges bound to "
            "the release metadata SHA-256. It is not claimed to be an intrinsic metadata field."
        ),
        "layer": 16,
        "feature_point": "current-observation MoT video tokens after block 16",
        "official_protocol_v2_clean_plus_random": {
            "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "raw_camera_shape": [3, 480, 640],
            "model_composite_shape": [3, 384, 320],
            "fps": 50,
            "window": {"state_steps": 33, "action_steps": 32, "video_frames": 9},
            "proprio": "14-D normalized official state window",
            "action": "14-D normalized 32-step 50Hz official target",
            "prompt": "per-frame official task paraphrase",
            "episodes_per_task_by_domain": {"clean": 50, "random": 500},
            "feature_statistics": official_stats,
        },
        "our_paired_four_scene_versions": {
            "camera_keys": list(POLICY_CAMERA_NAMES),
            "on_disk_raw_camera_shape": [3, 480, 640],
            "processor_camera_shape": [3, 240, 320],
            "model_composite_shape": [3, 384, 320],
            "fps": POLICY_NATIVE_FPS,
            "window": {"state_steps": 33, "action_steps": 32, "video_frames": 9},
            "scene_versions": list(POLICY_VARIANTS),
            "r3_role": POLICY_R3_ROLE,
            "temporal_resampling": "none",
            "supervision_mode": supervision_mode,
            "feature_statistics": paired_stats,
        },
        "scalar_feature_shift": compare_distributions(official_stats, paired_stats),
        "automatic_data_substitution": False,
        "supervision_contract": {
            "official": "action_loss_only",
            "paired_C_R1_R2_R3": (
                "action_loss_only" if supervision_mode == "action" else "contrastive_loss_only"
            ),
        },
    }


def _positive_action_path_coverage(
    gradient_steps: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Count positive-weight steps proving each action-gradient path.

    A single synchronized DDP batch can legitimately have an exactly-zero
    scalar gate gradient after per-sample gradients cancel.  Connectivity is
    therefore a cumulative run property, not a valid every-batch invariant.
    Zero-weight scheduler endpoints remain subject to exact-zero checks.
    """

    counts = {
        "positive_weight_steps": 0,
        "gate_positive_steps": 0,
        "adapter_attention_positive_steps": 0,
        "official_content_token_positive_steps": 0,
        "action_dit_positive_steps": 0,
    }
    for row in gradient_steps:
        if row.get("action_supervision_signal_positive") is not True:
            continue
        counts["positive_weight_steps"] += 1
        combined = row.get("combined", {})
        adapter = combined.get("adapter", {})
        attention = combined.get(
            "adapter_attention_action_only_by_construction", {}
        )
        action_dit = combined.get("action_dit", {})
        if float(adapter.get("gradient_norm", 0.0)) > 0.0:
            counts["gate_positive_steps"] += int(
                float(row.get("gate_gradient_norm", 0.0)) > 0.0
            )
        counts["adapter_attention_positive_steps"] += int(
            float(attention.get("gradient_norm", 0.0)) > 0.0
        )
        counts["official_content_token_positive_steps"] += int(
            float(row.get("action_only_official_content_token_grad_norm", 0.0))
            > 0.0
        )
        counts["action_dit_positive_steps"] += int(
            float(action_dit.get("gradient_norm", 0.0)) > 0.0
        )
    return counts


def _assert_full_550_official_audit(report: Mapping[str, Any]) -> None:
    """Refuse any Stage-2 run that did not actually load 50C+500R/task."""

    if report.get("domain_verified") is not True:
        raise RuntimeError("official protocol-v2 domain partition was not verified")
    if report.get("domain_label") != "protocol_v2_hash_bound_range_partition":
        raise RuntimeError("official domain label is not the hash-bound v2 partition")
    explicit = report.get("explicit_episode_native_loader")
    if not isinstance(explicit, Mapping):
        raise RuntimeError("official full-550 explicit-loader audit is missing")
    if explicit.get("selection_mode") != "full_550_per_task":
        raise RuntimeError("official loader did not select full_550_per_task")
    if int(explicit.get("loaded_episode_count", -1)) != 1_650:
        raise RuntimeError("official loader did not load exactly 1,650 episodes")
    counts = report.get("selected_episode_counts_by_domain")
    if not isinstance(counts, Mapping):
        raise RuntimeError("official per-domain episode counts are missing")
    for task in TASKS:
        task_counts = counts.get(task)
        expected_counts = dict(zip(OFFICIAL_DOMAINS, (50, 500), strict=True))
        if not isinstance(task_counts, Mapping) or dict(task_counts) != expected_counts:
            raise RuntimeError(
                f"official domain counts for {task} are not 50 Clean + 500 Random: "
                f"{task_counts}"
            )


def _crosscheck_release_paired_binding(
    binding: Mapping[str, Any],
    *,
    base_lineage_identity: Mapping[str, Any],
    base_identity: Mapping[str, Any],
    dataset_stats_identity: Mapping[str, Any],
    paired_root: Path,
    native_manifest,
    state_bank,
    selected_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the Phase-C binding names the exact artifacts used by this run."""

    lineage = binding.get("base_lineage")
    paired = binding.get("paired_dataset")
    selected = binding.get("selected_train_artifacts")
    cache_protocol = binding.get("cache_protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (lineage, paired, selected, cache_protocol)
    ):
        raise ValueError("release/paired binding is missing required contract sections")
    assert isinstance(lineage, Mapping)
    assert isinstance(paired, Mapping)
    assert isinstance(selected, Mapping)
    assert isinstance(cache_protocol, Mapping)
    if lineage.get("sha256") != base_lineage_identity.get("sha256"):
        raise ValueError("release/paired binding names a different base lineage")
    if not isinstance(lineage.get("checkpoint"), Mapping) or lineage["checkpoint"].get(
        "sha256"
    ) != base_identity.get("sha256"):
        raise ValueError("release/paired binding names a different checkpoint")
    if not isinstance(lineage.get("dataset_stats"), Mapping) or lineage[
        "dataset_stats"
    ].get("sha256") != dataset_stats_identity.get("sha256"):
        raise ValueError("release/paired binding names different dataset stats")
    if _resolve_path(str(paired.get("root", ""))) != paired_root.resolve():
        raise ValueError("release/paired binding names a different paired root")
    expected_paired = {
        "native_action_manifest_sha256": native_manifest.sha256,
        "native_action_audit_sha256": native_manifest.audit_sha256,
        "state_bank_sha256": state_bank.sha256,
        "physical_state_inventory_sha256": (
            state_bank.physical_state_inventory_sha256
        ),
    }
    for key, expected in expected_paired.items():
        if paired.get(key) != expected:
            raise ValueError(f"release/paired binding {key} differs from runtime")
    for key in ("algorithm", "episode_count", "file_count", "size_bytes", "sha256"):
        if selected.get(key) != selected_artifacts.get(key):
            raise ValueError(
                f"release/paired binding selected_train_artifacts.{key} differs"
            )
    expected_cache_protocol = (
        {
            "capture_layer": EXACT_LAYER,
            "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
            "physical_state_groups": PAIR280_GROUPS,
            "scene_views": PAIR280_GROUPS * POLICY_VIEW_COUNT,
            "view_token_shape": [120, 3_072],
            "storage": PAIR280_CACHE_STORAGE,
        }
        if int(binding.get("schema_version", 1)) == 2
        else {
            "capture_layer": EXACT_LAYER,
            "states_per_trajectory": 8,
            "physical_state_groups": 720,
            "scene_views": 2_880,
            "view_token_shape": [120, 3_072],
        }
    )
    if dict(cache_protocol) != expected_cache_protocol:
        raise ValueError("release/paired binding cache protocol differs from Policy v2")
    return {
        "status": "PASS",
        "binding_manifest_sha256": binding["binding_manifest_identity"]["sha256"],
        "base_lineage_sha256": lineage["sha256"],
        "selected_train_artifacts_sha256": selected["sha256"],
        "state_bank_sha256": paired["state_bank_sha256"],
    }


def _verify_formal_protocol_lock(
    config: Mapping[str, Any],
    *,
    lock_path: Path,
    lock_identity: Mapping[str, Any],
    projection_sha256: str,
    base_lineage_identity: Mapping[str, Any],
    p_mode_selection_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Match this resolved formal config to its pre-final, cycle-free lock row."""

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read formal protocol lock {lock_path}: {exc}") from exc
    lock = validate_formal_protocol_lock_manifest_payload(payload)
    if lock["base_lineage_manifest"]["sha256"] != base_lineage_identity["sha256"]:
        raise ValueError("formal protocol lock binds a different base lineage")
    if (
        lock["p_mode_selection_manifest"]["sha256"]
        != p_mode_selection_identity["sha256"]
    ):
        raise ValueError("formal protocol lock binds a different P-mode selection")
    selected_regime = str(lock["selected_policy_regime"])
    configured_regime = str(config["policy"]["regime"]).replace("-", "_")
    if configured_regime != selected_regime:
        raise ValueError("formal config policy regime differs from protocol lock")
    control = str(config["control"])
    seed = int(config["training"]["seed"])
    matches = [
        row
        for row in lock["resolved_configs"][control]
        if int(row["training_seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError("formal protocol lock has no unique control/seed row")
    row = matches[0]
    if row["protocol_projection_sha256"] != projection_sha256:
        raise ValueError("formal config projection differs from the locked protocol")
    if float(row["lambda_contrastive"]) != float(
        config["loss"]["lambda_contrastive"]
    ):
        raise ValueError("formal config lambda_contrastive differs from protocol lock")
    return {
        "status": "PASS",
        "formal_protocol_lock_manifest_sha256": lock_identity["sha256"],
        "control": control,
        "training_seed": seed,
        "selected_policy_regime": selected_regime,
        "lambda_contrastive": float(row["lambda_contrastive"]),
        "protocol_projection_sha256": projection_sha256,
        "source_config": dict(row["source_config"]),
    }


def run(
    config_path: str | Path,
    *,
    output_override: str | None = None,
    resume_from: str | Path | None = None,
    resume_amendment: str | Path | None = None,
) -> Path:
    config_file = _resolve_path(config_path)
    raw_cfg = OmegaConf.load(config_file)
    config = OmegaConf.to_container(raw_cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("run config root must be a mapping")
    if output_override is not None:
        config["output_dir"] = output_override
    validate_execution_ready(config)
    validate_run_config(config)
    requested_config_sha256 = canonical_json_sha256(config)
    formal_protocol_projection_sha256 = (
        p_mode_canonical_sha256(formal_config_protocol_projection(config))
        if bool(config.get("formal", False))
        else None
    )
    amendment_audit = (
        verify_full5ep_resume_amendment(
            resume_amendment,
            # The amendment is bound to the full experiment root, while the
            # immutable run config points at <root>/runs/seed_N/<control>.
            expected_output_root=Path(config["output_dir"]).expanduser().resolve().parents[2],
            expected_config_sha256=requested_config_sha256,
        )
        if resume_amendment is not None
        else None
    )
    fastwam_source_audit = audit_local_fastwam_source()
    config["runtime_provenance"] = {
        "fastwam_source": fastwam_source_audit,
        "full5ep_resume_amendment": amendment_audit,
    }

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    dataloader_config = _distributed_dataloader_config()
    accelerator = Accelerator(
        mixed_precision=str(config["training"]["mixed_precision"]),
        gradient_accumulation_steps=int(
            config["training"]["gradient_accumulation_steps"]
        ),
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp],
    )
    if bool(config["training"].get("require_cuda", True)) and accelerator.device.type != "cuda":
        raise RuntimeError("policy smoke requires CUDA; refusing unsafe huge-model CPU fallback")
    declared_world_size = int(config["training"]["world_size"])
    actual_world_size = int(accelerator.num_processes)
    actual_gradient_accumulation = int(accelerator.gradient_accumulation_steps)
    actual_official_global_batch = (
        int(config["training"]["official_batch_size"]) * actual_world_size
    )
    actual_paired_groups_per_step = (
        int(config["training"]["paired_groups_per_batch"]) * actual_world_size
    )
    checkpoint_interval_steps = int(config["training"].get("save_every", 0))
    if actual_world_size != declared_world_size:
        raise RuntimeError(
            f"Accelerator world size {actual_world_size} differs from configured "
            f"training.world_size={declared_world_size}"
        )
    if actual_gradient_accumulation != 1:
        raise RuntimeError("Accelerator gradient accumulation must remain exactly one")
    if actual_official_global_batch != int(
        config["training"]["effective_official_global_batch"]
    ):
        raise RuntimeError("actual effective official global batch differs from config")
    if actual_paired_groups_per_step != int(
        config["training"]["effective_paired_groups_per_step"]
    ):
        raise RuntimeError("actual effective paired physical-state groups differ from config")
    runtime_batch_contract = {
        "status": "PASS",
        "accelerator_num_processes": actual_world_size,
        "accelerator_split_batches": bool(dataloader_config.split_batches),
        "accelerator_even_batches": bool(dataloader_config.even_batches),
        "accelerator_use_seedable_sampler": bool(
            dataloader_config.use_seedable_sampler
        ),
        "gradient_accumulation_steps": actual_gradient_accumulation,
        "local_official_batch_size": int(config["training"]["official_batch_size"]),
        "effective_official_global_batch": actual_official_global_batch,
        "local_paired_groups_per_batch": int(
            config["training"]["paired_groups_per_batch"]
        ),
        "effective_paired_groups_per_step": actual_paired_groups_per_step,
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "resume_engine": "accelerate_save_state_load_state",
    }
    config["resolved_runtime_batch_contract"] = runtime_batch_contract
    official_loader_rng_contract = _official_loader_rng_contract(
        int(config["training"]["seed"])
    )
    config["resolved_official_loader_rng_contract"] = official_loader_rng_contract
    stage2_step_rng_contract = _stage2_step_rng_contract(
        int(config["training"]["seed"]),
        max_steps=int(config["training"]["max_steps"]),
        world_size=actual_world_size,
    )
    config["resolved_stage2_step_rng_contract"] = stage2_step_rng_contract
    _set_seed(int(config["training"]["seed"]), accelerator.process_index)
    output_dir = _resolve_path(config["output_dir"])
    configured_resume = config["training"].get("resume")
    if resume_from is not None and configured_resume not in {None, "", False}:
        raise RuntimeError("resume was supplied by both config and runtime CLI")
    resume_request = resume_from if resume_from is not None else configured_resume
    if resume_request not in {None, "", False} and config.get(
        "full5ep_protocol_manifest"
    ) and amendment_audit is None:
        raise RuntimeError(
            "resuming an amended full5ep run requires --resume-amendment"
        )
    if resume_request in {None, "", False} and amendment_audit is not None:
        raise RuntimeError("--resume-amendment is only valid with --resume")
    resume_selection: list[Any] = [None]
    if resume_request not in {None, "", False}:
        if checkpoint_interval_steps <= 0:
            raise RuntimeError("resume requires a positive training.save_every")
        if accelerator.is_main_process:
            try:
                state_dir, resume_step, trainer_state = _resolve_stage2_resume_state(
                    output_dir,
                    resume_request,
                    requested_config_sha256=requested_config_sha256,
                    max_steps=int(config["training"]["max_steps"]),
                    world_size=actual_world_size,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                    paired_schedule_profile=config.get("paired", {}).get(
                        "sampling_profile"
                    ),
                )
                resume_selection[0] = {
                    "status": "PASS",
                    "state_dir": str(state_dir),
                    "completed_step": resume_step,
                    "trainer_state": trainer_state,
                }
            except Exception as exc:
                resume_selection[0] = {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        broadcast_object_list(resume_selection, from_process=0)
        if resume_selection[0].get("status") != "PASS":
            raise RuntimeError(
                "Stage-2 resume preflight failed: "
                f"{resume_selection[0].get('error')}"
            )
    resume_audit = resume_selection[0]
    resume_step = int(resume_audit["completed_step"]) if resume_audit else 0
    config["resolved_resume"] = (
        {
            "status": "PASS",
            "engine": "accelerate_save_state_load_state",
            "state_dir": str(resume_audit["state_dir"]),
            "completed_step": resume_step,
            "next_step": resume_step + 1,
        }
        if resume_audit is not None
        else {
            "status": "FRESH_START",
            "engine": "accelerate_save_state_load_state",
            "completed_step": 0,
            "next_step": 1,
        }
    )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "requested_config.json", config)
    accelerator.wait_for_everyone()
    # Native RobotVideoDataset mirrors pretrained normalization stats into the
    # registered work directory during construction.  Without this explicit
    # binding it writes to a relative ./runs path that need not exist.
    misc.register_work_dir(output_dir)

    base_checkpoint = _resolve_path(config["base_checkpoint"])
    head_init_mode = str(config["policy"]["head_init_mode"])
    head_init = (
        _resolve_path(config["policy"]["head_init"])
        if head_init_mode == "pretrained"
        else None
    )
    official_cfg = config["official"]
    dataset_stats = _resolve_path(official_cfg["dataset_stats"])
    manifest_path = _resolve_path(official_cfg["canonical_task_manifest"])
    official_text_cache_binding_path = (
        _resolve_path(official_cfg["text_cache_binding_manifest"])
        if official_cfg.get("text_cache_dir") is not None
        else None
    )
    base_lineage_path = _resolve_path(config["base_lineage_manifest"])
    release_paired_binding_path = _resolve_path(
        config["release_paired_binding_manifest"]
    )
    formal_protocol_lock_path = (
        _resolve_path(config["formal_protocol_lock_manifest"])
        if config.get("formal_protocol_lock_manifest") is not None
        else None
    )
    seed_bank_manifest_path = _resolve_path(
        config["evaluation"]["simulator_seed_bank_manifest"]
    )
    p_mode_selection_path = (
        _resolve_path(config["p_mode_selection_manifest"])
        if config.get("p_mode_selection_manifest") is not None
        else None
    )
    coefficient = float(config["loss"]["lambda_contrastive"])
    paired_action_coefficient = float(config["loss"]["lambda_paired_action"])
    paired_cfg = config["paired"]
    paired_mode = str(paired_cfg["supervision_mode"])
    pair280_enabled = paired_cfg.get("sampling_profile") == PAIR280_PROFILE_ID
    paired_path = (
        _resolve_path(paired_cfg["cache"]) if paired_mode == "contrastive" else None
    )
    paired_action_root = (
        _resolve_path(paired_cfg["action_root"]) if paired_mode != "none" else None
    )
    paired_action_manifest = (
        _resolve_path(paired_cfg["action_manifest"]) if paired_mode != "none" else None
    )
    paired_action_audit = (
        _resolve_path(paired_cfg["action_audit"]) if paired_mode != "none" else None
    )
    paired_state_bank_path = (
        _resolve_path(paired_cfg["state_bank"]) if paired_mode != "none" else None
    )
    paired_text_cache_path = (
        _resolve_path(paired_cfg["text_cache_dir"]) if paired_mode != "none" else None
    )

    # Bind immutable user-selected inputs before loading/training.
    base_identity = _audit_path(
        config, "base_checkpoint", base_checkpoint, required_for_rollout=False
    )
    identities: dict[str, dict[str, Any]] = {
        "dataset_stats": _audit_path(
            config, "dataset_stats", dataset_stats, required_for_rollout=True
        ),
        "official_manifest": _audit_path(
            config, "official_manifest", manifest_path, required_for_rollout=False
        ),
        "base_lineage_manifest": _audit_path(
            config,
            "base_lineage_manifest",
            base_lineage_path,
            required_for_rollout=False,
        ),
        "release_paired_binding_manifest": _audit_path(
            config,
            "release_paired_binding_manifest",
            release_paired_binding_path,
            required_for_rollout=False,
        ),
        "simulator_seed_bank_manifest": _audit_path(
            config,
            "simulator_seed_bank_manifest",
            seed_bank_manifest_path,
            required_for_rollout=True,
        ),
    }
    if official_text_cache_binding_path is not None:
        identities["official_text_cache_binding_manifest"] = _audit_path(
            config,
            "official_text_cache_binding_manifest",
            official_text_cache_binding_path,
            required_for_rollout=False,
        )
    if p_mode_selection_path is not None:
        identities["p_mode_selection_manifest"] = _audit_path(
            config,
            "p_mode_selection_manifest",
            p_mode_selection_path,
            required_for_rollout=True,
        )
    if formal_protocol_lock_path is not None:
        identities["formal_protocol_lock_manifest"] = _audit_path(
            config,
            "formal_protocol_lock_manifest",
            formal_protocol_lock_path,
            required_for_rollout=True,
        )
    base_lineage_audit = verify_author_release_lineage(
        base_lineage_path,
        checkpoint_path=base_checkpoint,
        dataset_stats_path=dataset_stats,
        official_manifest_path=manifest_path,
        expected_manifest_sha256=identities["base_lineage_manifest"]["sha256"],
    )
    official_text_cache_binding_audit = (
        verify_official_text_cache_binding(
            official_text_cache_binding_path,
            expected_sha256=identities[
                "official_text_cache_binding_manifest"
            ]["sha256"],
            expected_base_lineage_sha256=identities[
                "base_lineage_manifest"
            ]["sha256"],
            expected_cache_dir=_resolve_path(official_cfg["text_cache_dir"]),
        )
        if official_text_cache_binding_path is not None
        else None
    )
    if bool(config.get("formal", False)):
        assert formal_protocol_lock_path is not None
        assert formal_protocol_projection_sha256 is not None
        if "p_mode_selection_manifest" not in identities:
            raise ValueError("formal protocol lock requires P-mode selection identity")
        formal_protocol_lock_audit = _verify_formal_protocol_lock(
            config,
            lock_path=formal_protocol_lock_path,
            lock_identity=identities["formal_protocol_lock_manifest"],
            projection_sha256=formal_protocol_projection_sha256,
            base_lineage_identity=identities["base_lineage_manifest"],
            p_mode_selection_identity=identities["p_mode_selection_manifest"],
        )
    else:
        formal_protocol_lock_audit = None
    if head_init is not None:
        identities["head_init"] = _audit_path(
            config, "head_init", head_init, required_for_rollout=False
        )
    if paired_path is not None:
        identities["paired_train_cache"] = _audit_path(
            config, "paired_cache", paired_path, required_for_rollout=False
        )
    if paired_mode != "none":
        assert paired_action_manifest is not None and paired_action_audit is not None
        assert paired_state_bank_path is not None and paired_text_cache_path is not None
        identities["paired_action_manifest"] = _audit_path(
            config,
            "paired_action_manifest",
            paired_action_manifest,
            required_for_rollout=False,
        )
        identities["paired_action_audit"] = _audit_path(
            config,
            "paired_action_audit",
            paired_action_audit,
            required_for_rollout=False,
        )
        identities["paired_state_bank"] = _audit_path(
            config,
            "paired_state_bank",
            paired_state_bank_path,
            required_for_rollout=False,
        )
        identities["paired_text_cache"] = _audit_path(
            config,
            "paired_text_cache",
            paired_text_cache_path,
            required_for_rollout=False,
        )
        assert paired_action_root is not None
        native_paired_contract_audit = audit_native_paired_action_contract(
            dataset_root=paired_action_root,
            manifest_path=paired_action_manifest,
            audit_path=paired_action_audit,
            expected_tasks=TASKS,
            require_full_protocol_counts=True,
        )
        verified_native_manifest = verify_native_paired_action_manifest(
            paired_action_manifest,
            dataset_root=paired_action_root,
            audit_path=paired_action_audit,
        )
        verified_state_bank = (
            verify_pair280_state_bank(
                paired_state_bank_path,
                paired_root=paired_action_root,
                paired_manifest=paired_action_manifest,
                paired_audit=paired_action_audit,
                expected_sha256=identities["paired_state_bank"]["sha256"],
            )
            if pair280_enabled
            else verify_policy_state_bank(
                paired_state_bank_path,
                native_manifest=verified_native_manifest,
                expected_sha256=identities["paired_state_bank"]["sha256"],
                expected_tasks=TASKS,
            )
        )
        state_bank_audit = audit_policy_state_bank(verified_state_bank)
        selected_artifacts = selected_episode_artifact_aggregate(
            verified_state_bank.native_manifest,
            split="train",
        )
        release_paired_binding_audit = verify_release_paired_binding(
            release_paired_binding_path,
            expected_sha256=identities["release_paired_binding_manifest"][
                "sha256"
            ],
        )
        release_paired_binding_crosscheck = _crosscheck_release_paired_binding(
            release_paired_binding_audit,
            base_lineage_identity=identities["base_lineage_manifest"],
            base_identity=base_identity,
            dataset_stats_identity=identities["dataset_stats"],
            paired_root=paired_action_root,
            native_manifest=verified_native_manifest,
            state_bank=verified_state_bank,
            selected_artifacts=selected_artifacts,
        )
        paired_text_cache_audit = verify_release_paired_text_cache(
            paired_text_cache_path,
            expected_base_lineage_sha256=identities["base_lineage_manifest"][
                "sha256"
            ],
            expected_release_paired_binding_sha256=identities[
                "release_paired_binding_manifest"
            ]["sha256"],
        )
    else:
        native_paired_contract_audit = None
        verified_state_bank = None
        state_bank_audit = None
        selected_artifacts = None
        release_paired_binding_audit = None
        release_paired_binding_crosscheck = None
        paired_text_cache_audit = None

    model, native_cfg, release_audit = instantiate_release_model(
        base_checkpoint,
        device=str(accelerator.device),
        dtype=dtype_from_name(str(config["training"]["model_dtype"])),
        load_text_encoder=bool(official_cfg.get("on_the_fly_text_smoke", False)),
        model_base_path=config.get("model_base_path"),
        compute_checkpoint_sha256=False,
    )
    for component_name in ("vae", "text_encoder", "tokenizer"):
        identities[component_name] = _audit_path(
            config,
            component_name,
            _resolve_component_path(model, config, component_name),
            required_for_rollout=True,
        )
    expected_cache_extraction_contract: dict[str, Any] | None = None
    if paired_mode == "contrastive":
        assert paired_text_cache_path is not None
        assert verified_state_bank is not None
        extractor_source_path = (
            PROJECT_ROOT
            / "experiments/robotwin/policy_content_adapter/"
            / (
                "extract_pair280_cache.py"
                if pair280_enabled
                else "extract_policy_cache.py"
            )
        ).resolve()
        identities["policy_cache_extractor_source"] = artifact_identity(
            extractor_source_path
        )
        identities["policy_cache_extractor_source"].update(
            {"required_for_rollout": False, "verification_status": "PASS"}
        )
        support_source_paths = {
            "frozen_backbone": (
                PROJECT_ROOT / "experiments/robotwin/e0_e1/backbone.py"
            ).resolve(),
            "runtime_utils": (
                PROJECT_ROOT
                / "experiments/robotwin/policy_content_adapter/runtime_utils.py"
            ).resolve(),
            "policy_data": (
                PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/data.py"
            ).resolve(),
            "policy_protocol": (
                PROJECT_ROOT
                / "experiments/robotwin/policy_content_adapter/protocol.py"
            ).resolve(),
        }
        if pair280_enabled:
            support_source_paths["pair280_protocol"] = (
                PROJECT_ROOT
                / "experiments/robotwin/policy_content_adapter/pair280_protocol.py"
            ).resolve()
        support_source_identities = {
            name: artifact_identity(path) for name, path in support_source_paths.items()
        }
        for name, identity in support_source_identities.items():
            identities[f"policy_cache_{name}_source"] = {
                **identity,
                "required_for_rollout": False,
                "verification_status": "PASS",
            }
        assert selected_artifacts is not None
        expected_cache_extraction_contract = build_policy_cache_extraction_contract(
            base_lineage_identity=identities["base_lineage_manifest"],
            release_paired_binding_identity=identities[
                "release_paired_binding_manifest"
            ],
            dataset_stats_identity=identities["dataset_stats"],
            vae_identity=identities["vae"],
            text_encoder_identity=identities["text_encoder"],
            tokenizer_identity=identities["tokenizer"],
            text_cache_identity=identities["paired_text_cache"],
            fastwam_source_audit=fastwam_source_audit,
            extractor_source_identity=identities["policy_cache_extractor_source"],
            extractor_support_source_identities=support_source_identities,
            selected_episode_artifacts=selected_artifacts,
            extractor_config_override=(
                {
                    **policy_cache_extractor_config(
                        states_per_trajectory=PAIR280_STATES_PER_TRAJECTORY,
                        state_selection_algorithm=PAIR280_STATE_ALGORITHM,
                        state_selection_seed=PAIR280_STATE_SEED,
                        storage=PAIR280_CACHE_STORAGE,
                    ),
                    "parallel_workers": 8,
                    "shard_unit": "one_physical_trajectory",
                    "shard_count": 90,
                }
                if pair280_enabled
                else None
            ),
        )

    if head_init_mode == "random":
        head_seed = int(config["policy"]["head_init_seed"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            head = PolicyContentHead()
        head_payload: dict[str, Any] = {
            "experiment": "policy_v2_seeded_random",
            "layer": EXACT_LAYER,
            "checkpoint_kind": None,
        }
    else:
        head_seed = None
        head = PolicyContentHead()
        assert head_init is not None
        head_payload = load_e1_e3_head_checkpoint(head, head_init)
    adapter_seed = int(config["policy"]["adapter_init_seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(adapter_seed)
        adapter = GatedCrossAttentionAdapter()
    source_head_sha256 = module_state_sha256(head)
    source_adapter_sha256 = module_state_sha256(adapter)
    head_initial_state = {
        name: value.detach().to(device="cpu").clone()
        for name, value in head.state_dict().items()
    }
    adapter_initial_state = {
        name: value.detach().to(device="cpu").clone()
        for name, value in adapter.state_dict().items()
    }
    runtime = install_policy_content_adapter(
        model,
        head=head,
        adapter=adapter,
        enabled=True,
        content_layer=int(config["policy"]["content_layer"]),
        patch_video_prefill=True,
    )
    if float(runtime.conditioner.adapter.gate.detach().float().item()) != 0.0:
        raise RuntimeError("adapter gate did not initialize to exact zero")
    initialization = {
        "head_init_mode": head_init_mode,
        "head_seed": head_seed,
        "adapter_seed": adapter_seed,
        "source_fp32_content_head_sha256": source_head_sha256,
        "source_fp32_adapter_sha256": source_adapter_sha256,
        "identity_bf16_content_head_sha256": module_state_sha256(
            runtime.conditioner.head
        ),
        "identity_bf16_adapter_sha256": module_state_sha256(
            runtime.conditioner.adapter
        ),
        "gate_raw": 0.0,
    }

    native_dataset = instantiate_official_dataset(
        native_cfg,
        dataset_root=official_cfg["dataset_root"],
        dataset_stats_path=dataset_stats,
        text_cache_dir=official_cfg.get("text_cache_dir"),
        model_for_on_the_fly_text=(
            model if bool(official_cfg.get("on_the_fly_text_smoke", False)) else None
        ),
        manifest_path=manifest_path,
        episode_selection_mode=str(official_cfg["selection_mode"]),
    )
    official_dataset = OfficialThreeTaskDataset(
        native_dataset,
        dataset_root=official_cfg["dataset_root"],
        manifest_path=manifest_path,
        sampling_mode=str(official_cfg["sampling_mode"]),
    )
    _assert_full_550_official_audit(official_dataset.audit_report)
    official_dataset_audit = {
        **official_dataset.audit_report,
        "loader_rng_contract": official_loader_rng_contract,
    }
    workers = int(config["training"]["num_workers"])
    official_batch_size = int(config["training"]["official_batch_size"])
    max_steps = int(config["training"]["max_steps"])
    official_sampler = ThreeTaskRoundRobinSampler(
        official_dataset,
        seed=int(config["training"]["seed"]),
        num_samples=max_steps * official_batch_size * accelerator.num_processes,
        shuffle=True,
    )
    official_loader = DataLoader(
        official_dataset,
        batch_size=official_batch_size,
        sampler=official_sampler,
        num_workers=workers,
        pin_memory=accelerator.device.type == "cuda",
        drop_last=True,
        generator=_new_cpu_generator(
            official_loader_rng_contract["training_dataloader_generator_seed"]
        ),
        worker_init_fn=_seed_dataloader_worker_from_torch,
    )

    # Full native-vs-patched action comparison occurs before any trainability or
    # optimizer state changes.  Re-iterating the deterministic loader preserves
    # the first training sample.
    identity_sampler = ThreeTaskRoundRobinSampler(
        official_dataset,
        seed=int(config["training"]["seed"]),
        num_samples=official_batch_size,
        shuffle=True,
    )
    identity_loader = DataLoader(
        official_dataset,
        batch_size=official_batch_size,
        sampler=identity_sampler,
        num_workers=workers,
        pin_memory=accelerator.device.type == "cuda",
        drop_last=True,
        generator=_new_cpu_generator(
            official_loader_rng_contract["identity_dataloader_generator_seed"]
        ),
        worker_init_fn=_seed_dataloader_worker_from_torch,
    )
    with _isolated_cpu_data_rng(
        official_loader_rng_contract["identity_dataloader_generator_seed"]
    ):
        identity_iterator = iter(identity_loader)
        identity_batch = next(identity_iterator)
    identity_audit = zero_init_policy_identity_audit(model, runtime, identity_batch)
    identity_audit["official_task"] = list(_task_values(identity_batch, "official_task"))
    identity_audit["base_checkpoint_sha256"] = base_identity["sha256"]
    if accelerator.is_main_process:
        _write_json(output_dir / "identity_audit.json", identity_audit)
    del identity_batch
    del identity_iterator
    del identity_loader

    regime = str(config["policy"]["regime"])
    counts = configure_trainable_modules(model, runtime.conditioner, regime=regime)
    # The frozen release branch stays bf16.  Trainable leaves and AdamW state
    # must be fp32: lr=1e-4/1e-5 updates are frequently below a bf16 ULP and
    # would otherwise round to zero before accumulating.  Autocast still runs
    # the expensive matmuls in the configured mixed precision.
    runtime.conditioner.float()
    # install_policy_content_adapter intentionally cast the conditioner to the
    # release model dtype for the bit-exact identity check.  Restore the exact
    # FP32 seeded Head and adapter before optimization rather than
    # training a BF16-quantized copy promoted back to FP32.
    runtime.conditioner.head.load_state_dict(head_initial_state, strict=True)
    runtime.conditioner.adapter.load_state_dict(adapter_initial_state, strict=True)
    if module_state_sha256(runtime.conditioner.head) != source_head_sha256:
        raise RuntimeError("FP32 Content Head initialization was not restored exactly")
    if module_state_sha256(runtime.conditioner.adapter) != source_adapter_sha256:
        raise RuntimeError("FP32 adapter initialization was not restored exactly")
    initialization["training_fp32_content_head_sha256"] = source_head_sha256
    initialization["training_fp32_adapter_sha256"] = source_adapter_sha256
    if regime.replace("-", "_") == "p_v2":
        model.action_expert.float()
    trainable_parameters = [
        parameter
        for parameter in (*runtime.conditioner.parameters(), *model.action_expert.parameters())
        if parameter.requires_grad
    ]
    if not trainable_parameters or any(
        parameter.dtype is not torch.float32 for parameter in trainable_parameters
    ):
        raise RuntimeError("not all trainable parameters were promoted to fp32")
    precision_audit = {
        "frozen_model_dtype": str(model.torch_dtype),
        "trainable_parameter_dtype": "torch.float32",
        "trainable_parameter_tensors": len(trainable_parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "autocast_mixed_precision": str(config["training"]["mixed_precision"]),
    }
    groups = build_optimizer_param_groups(
        model,
        runtime.conditioner,
        regime=regime,
        head_adapter_lr=float(config["optimizer"]["head_adapter_lr"]),
        action_dit_lr=float(config["optimizer"]["action_dit_lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    lr_scheduler_mode = "constant"
    update_snapshot = ParameterSnapshot.capture(
        {
            "content_head": runtime.conditioner.head,
            "adapter": runtime.conditioner.adapter,
        }
    )
    action_snapshot = (
        SampledParameterSnapshot.capture(
            {"action_expert": model.action_expert},
            max_parameter_tensors=32,
            samples_per_tensor=1024,
            required_parameter_names=ACTION_UPDATE_REQUIRED_PARAMETERS,
        )
        if regime.replace("-", "_") == "p_v2"
        else None
    )

    paired_dataset: (
        FrozenPairedTokenDataset
        | ShardedPair280TokenDataset
        | NativePairedActionDataset
        | None
    ) = None
    paired_loader = None
    paired_collate = None
    if paired_mode == "contrastive":
        assert paired_path is not None
        assert verified_state_bank is not None
        assert expected_cache_extraction_contract is not None
        paired_dataset = (
            ShardedPair280TokenDataset(
                paired_path,
                state_bank=verified_state_bank,
                expected_manifest_sha256=identities["paired_train_cache"]["sha256"],
                expected_release_binding_sha256=identities[
                    "release_paired_binding_manifest"
                ]["sha256"],
                expected_extraction_contract=expected_cache_extraction_contract,
                # One complete byte audit is sufficient; every other rank still
                # verifies the manifest, file inventory, and sizes.
                verify_shard_hashes=accelerator.is_main_process,
            )
            if pair280_enabled
            else FrozenPairedTokenDataset(
                paired_path,
                state_bank=verified_state_bank,
                expected_extraction_contract=expected_cache_extraction_contract,
                layer=int(config["paired"]["layer"]),
                split="train",
                expected_backbone_sha256=base_identity["sha256"],
                expected_base_lineage_sha256=identities["base_lineage_manifest"]["sha256"],
                expected_release_paired_binding_sha256=identities[
                    "release_paired_binding_manifest"
                ]["sha256"],
                expected_action_manifest_sha256=identities["paired_action_manifest"]["sha256"],
                expected_action_audit_sha256=identities["paired_action_audit"]["sha256"],
                expected_state_bank_sha256=identities["paired_state_bank"]["sha256"],
            )
        )
        if paired_dataset.token_shape != (120, 3072):
            raise ValueError(
                f"paired Layer-16 tokens must be [120,3072], got {paired_dataset.token_shape}"
            )
        paired_collate = collate_paired_token_groups
    elif paired_mode == "action":
        assert paired_action_root is not None
        assert paired_action_manifest is not None
        assert paired_action_audit is not None
        assert paired_state_bank_path is not None
        assert paired_text_cache_path is not None
        factory = getattr(
            policy_runtime_utils,
            "instantiate_native_paired_action_dataset",
            None,
        )
        if not callable(factory):
            raise RuntimeError(
                "C2 requires runtime_utils.instantiate_native_paired_action_dataset; "
                "refusing to substitute a tensor cache or interpolated dataset"
            )
        paired_dataset = factory(
            native_cfg,
            dataset_root=paired_action_root,
            dataset_stats_path=dataset_stats,
            text_cache_dir=paired_text_cache_path,
            model_for_on_the_fly_text=None,
            manifest_path=paired_action_manifest,
            audit_path=paired_action_audit,
            state_bank_path=paired_state_bank_path,
            expected_state_bank_sha256=identities["paired_state_bank"]["sha256"],
            expected_tasks=TASKS,
            require_full_protocol_counts=True,
        )
        if not isinstance(paired_dataset, NativePairedActionDataset):
            raise TypeError("native paired action factory returned an unverified dataset")
        paired_collate = collate_paired_action_groups

    if paired_dataset is not None:
        if set(paired_dataset.indices_by_task) != set(TASKS):
            raise ValueError(
                "paired training dataset task set differs from the exact three tasks"
            )
        paired_sampler = (
            ExactPair280GlobalBatchSampler(
                paired_dataset,
                seed=int(config["training"]["seed"]),
            )
            if pair280_enabled
            else SameTaskPhysicalStateBatchSampler(
                paired_dataset,
                groups_per_batch=int(config["training"]["paired_groups_per_batch"]),
                # Accelerate shards one shared global batch sequence across ranks;
                # rank-dependent seeds here would double-shard unrelated streams.
                seed=int(config["training"]["seed"]),
                batches_per_epoch=max_steps * accelerator.num_processes,
                balanced_round_robin=True,
            )
        )
        paired_loader = DataLoader(
            paired_dataset,
            batch_sampler=paired_sampler,
            collate_fn=paired_collate,
            num_workers=0,
            pin_memory=accelerator.device.type == "cuda",
            generator=_new_cpu_generator(
                official_loader_rng_contract["paired_dataloader_generator_seed"]
            ),
        )

    progress = PolicyTrainingProgress(
        max_steps=max_steps,
        requested_config_sha256=requested_config_sha256,
        world_size=actual_world_size,
        effective_official_global_batch=actual_official_global_batch,
        effective_paired_groups_per_step=actual_paired_groups_per_step,
        paired_supervision_mode=paired_mode,
        paired_schedule_profile=(PAIR280_PROFILE_ID if pair280_enabled else None),
    )
    accelerator.register_for_checkpointing(progress)
    training_module = PolicyTrainingModule(
        model,
        runtime,
        paired_supervision_mode=paired_mode,
        lambda_contrastive=coefficient,
        lambda_paired_action=paired_action_coefficient,
        temperature=float(config["loss"]["temperature"]),
        training_seed=int(config["training"]["seed"]),
        process_index=accelerator.process_index,
    )
    raw_training_module = training_module
    if paired_loader is not None:
        training_module, optimizer, official_loader, paired_loader = accelerator.prepare(
            training_module, optimizer, official_loader, paired_loader
        )
    else:
        training_module, optimizer, official_loader = accelerator.prepare(
            training_module, optimizer, official_loader
        )
    raw_training_module = _resolve_prepared_raw_module(
        training_module, raw_training_module
    )
    if resume_audit is not None:
        accelerator.load_state(input_dir=str(resume_audit["state_dir"]))
        if progress.completed_step != resume_step:
            raise RuntimeError(
                "Accelerate custom progress step differs from trainer_state.json"
            )
        raw_training_module.set_forward_index_for_resume(resume_step)
        official_loader = skip_first_batches(official_loader, resume_step)
        if paired_loader is not None:
            paired_loader = skip_first_batches(
                paired_loader,
                paired_active_count(resume_step) if pair280_enabled else resume_step,
            )
    official_iterator = _CyclingIterator(official_loader)
    paired_iterator = iter(paired_loader) if paired_loader is not None else None
    optimizer.zero_grad(set_to_none=True)

    matched_stream_contract_enabled = config["control"] in {
        "c1_architecture_only",
        "c3_ours",
    } or (
        config["control"] in {"p_v1", "p_v2"}
        and str(config.get("stage", "")) == "dev_pilot"
        and str(config.get("selection_role", "")) == "c1_lambda0"
    )
    matched_stream_contract = (
        build_matched_c1_c3_stream_contract(
            config,
            base_identity=base_identity,
            identities=identities,
        )
        if matched_stream_contract_enabled
        else None
    )
    config["resolved_artifact_identities"] = identities
    config["resolved_base_checkpoint_identity"] = base_identity
    config["resolved_base_lineage"] = base_lineage_audit
    config["resolved_official_text_cache_binding"] = (
        official_text_cache_binding_audit
    )
    config["resolved_release_paired_binding"] = release_paired_binding_crosscheck
    config["resolved_paired_text_cache"] = (
        {
            "status": "PASS",
            "audit_identity": paired_text_cache_audit["audit_identity"],
            "directory_identity": paired_text_cache_audit["directory_identity"],
        }
        if paired_text_cache_audit is not None
        else None
    )
    config["resolved_formal_protocol_lock"] = formal_protocol_lock_audit
    config["resolved_matched_stream_contract"] = matched_stream_contract
    config["resolved_official_subset"] = official_dataset_audit
    config["resolved_initialization"] = initialization
    config["formal_training_auto_started"] = False
    if accelerator.is_main_process:
        _write_json(output_dir / "run_config.json", config)
        _write_json(output_dir / "official_subset_audit.json", official_dataset_audit)
        _write_json(output_dir / "base_lineage_audit.json", base_lineage_audit)
        if official_text_cache_binding_audit is not None:
            _write_json(
                output_dir / "official_text_cache_binding_audit.json",
                official_text_cache_binding_audit,
            )
        if release_paired_binding_audit is not None:
            _write_json(
                output_dir / "release_paired_binding_audit.json",
                release_paired_binding_audit,
            )
            _write_json(
                output_dir / "release_paired_binding_crosscheck.json",
                release_paired_binding_crosscheck,
            )
        if paired_text_cache_audit is not None:
            _write_json(
                output_dir / "paired_text_cache_audit.json",
                paired_text_cache_audit,
            )
        if formal_protocol_lock_audit is not None:
            _write_json(
                output_dir / "formal_protocol_lock_audit.json",
                formal_protocol_lock_audit,
            )
        if matched_stream_contract is not None:
            _write_json(
                output_dir / "matched_stream_contract.json", matched_stream_contract
            )
        _write_json(
            output_dir / "stage2_step_rng_contract.json", stage2_step_rng_contract
        )
        _write_json(output_dir / "artifact_identities.json", {
            "base_checkpoint": base_identity,
            **identities,
        })
    accelerator.wait_for_everyone()

    rows = progress.rows
    gradient_steps = progress.gradient_steps
    official_distribution = progress.official_distribution
    paired_distribution = progress.paired_distribution
    seen_official_tasks = progress.seen_official_tasks
    seen_paired_tasks = progress.seen_paired_tasks
    seen_official_sample_ids = progress.seen_official_sample_ids
    seen_paired_state_ids = progress.seen_paired_state_ids
    max_grad_norm = float(config["training"]["max_grad_norm"])
    allow_direct_probe = (
        accelerator.num_processes == 1
        and bool(config["training"].get("action_gradient_probe", True))
    )
    positive_action_signal_steps = progress.positive_action_signal_steps
    zero_action_signal_steps = progress.zero_action_signal_steps

    for step in range(resume_step + 1, max_steps + 1):
        data_seed = (
            int(official_loader_rng_contract["main_process_data_seed_base"])
            + accelerator.process_index * 1_000_000
            + step
        )
        with _isolated_cpu_data_rng(data_seed):
            official_batch = next(official_iterator)
            paired_active = bool(
                paired_iterator is not None
                and (not pair280_enabled or paired_is_active(step))
            )
            if paired_active:
                try:
                    paired_batch = next(paired_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "paired stream exhausted before its audited exposure budget"
                    ) from exc
            else:
                paired_batch = None
        official_tasks = _task_values(official_batch, "official_task")
        paired_tasks = _task_values(paired_batch or {}, "task")
        official_episodes = _integer_values(official_batch, "official_episode_index")
        official_base_indices = _integer_values(official_batch, "official_base_index")
        if len(official_episodes) != len(official_base_indices):
            raise RuntimeError("official episode/base identity batch lengths differ")
        official_sample_ids = tuple(
            f"episode_{episode:06d}/base_{base_index:09d}"
            for episode, base_index in zip(
                official_episodes, official_base_indices, strict=True
            )
        )
        paired_state_ids = _task_values(paired_batch or {}, "physical_state_id")

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            total_loss, action_loss, contrastive_loss, diagnostics = training_module(
                official_batch, paired_batch, paired_active=paired_active
            )
        direct_probe = (
            action_path_gradient_probe(
                action_loss,
                raw_training_module.conditioner,
            )
            if allow_direct_probe
            else {
                "head_grad_norm": float("nan"),
                "adapter_attention_grad_norm": float("nan"),
                "gate_grad_norm": float("nan"),
                "all_finite": True,
                "status": "disabled_for_distributed_run",
            }
        )
        accelerator.backward(total_loss)
        raw = raw_training_module

        head_report = module_gradient_report(raw.conditioner.head)
        adapter_report = module_gradient_report(raw.conditioner.adapter)
        adapter_attention_report = module_gradient_report(
            raw.conditioner.adapter.cross_attention
        )
        action_report = module_gradient_report(raw.model.action_expert)
        gate_grad = raw.conditioner.adapter.gate.grad
        gate_grad_norm = (
            0.0
            if gate_grad is None
            else float(gate_grad.detach().double().abs().item())
        )
        content_grad_norm = raw.conditioner.consume_action_content_gradient_audit()
        if not math.isfinite(content_grad_norm):
            raise FloatingPointError("official content-token gradient is non-finite")
        action_signal_positive = diagnostics.get(
            "action_supervision_signal_positive"
        )
        if not isinstance(action_signal_positive, bool):
            raise RuntimeError("official action loss omitted its supervision signal audit")
        if action_signal_positive:
            positive_action_signal_steps += 1
        else:
            zero_action_signal_steps += 1
            if diagnostics.get("zero_action_signal_reason") != "scheduler_zero_weight":
                raise RuntimeError("zero action signal lacks scheduler endpoint provenance")
            if float(diagnostics.get("action_weight_max", float("nan"))) != 0.0:
                raise RuntimeError("zero action signal has a nonzero scheduler weight")

        assert_no_parameter_gradients(raw.model.video_expert, label="Video Backbone")
        assert_no_parameter_gradients(raw.model.vae, label="VAE")
        if getattr(raw.model, "text_encoder", None) is not None:
            assert_no_parameter_gradients(raw.model.text_encoder, label="text encoder")
        if getattr(raw.model, "proprio_encoder", None) is not None:
            assert_no_parameter_gradients(raw.model.proprio_encoder, label="proprio encoder")
        if regime.replace("-", "_") == "p_v1":
            assert_no_parameter_gradients(raw.model.action_expert, label="P-v1 ActionDiT")
            if action_report["gradient_norm"] != 0.0:
                raise RuntimeError("P-v1 ActionDiT gradient norm is nonzero")
        elif not action_signal_positive and action_report["gradient_norm"] != 0.0:
            raise RuntimeError("zero-weight action loss unexpectedly reached P-v2 ActionDiT")

        if action_signal_positive and positive_action_signal_steps == 1 and (
            adapter_report["gradient_norm"] <= 0.0 or gate_grad_norm <= 0.0
        ):
            raise RuntimeError("first action batch did not reach the zero-init adapter gate")
        if (
            bool(diagnostics["paired_contrastive_active"])
            and coefficient > 0.0
            and head_report["gradient_norm"] <= 0.0
        ):
            raise RuntimeError("paired contrastive loss did not reach Content Head")
        # Gate=0 mathematically blocks action gradients into head/GCA weights on
        # the first *positive-weight* action batch.  From the next such batch
        # onward, the action-only probe must reach all three components.  The
        # native scheduler legitimately assigns exact zero weight at its
        # endpoint; those batches must have exactly zero action-path gradients.
        if (
            action_signal_positive
            and positive_action_signal_steps == 2
            and allow_direct_probe
        ):
            if (
                direct_probe["head_grad_norm"] <= 0.0
                or direct_probe["adapter_attention_grad_norm"] <= 0.0
                or direct_probe["gate_grad_norm"] <= 0.0
            ):
                raise RuntimeError(
                    f"action-only parameter gradient audit failed at step {step}: {direct_probe}"
                )
        elif not action_signal_positive:
            if float(action_loss.detach().item()) != 0.0:
                raise RuntimeError("zero-weight action batch produced nonzero weighted loss")
            zero_action_norms = {
                "adapter": float(adapter_report["gradient_norm"]),
                "adapter_attention": float(adapter_attention_report["gradient_norm"]),
                "gate": gate_grad_norm,
                "content_tokens": content_grad_norm,
            }
            if allow_direct_probe:
                zero_action_norms.update(
                    {
                        "probe_head": float(direct_probe["head_grad_norm"]),
                        "probe_adapter_attention": float(
                            direct_probe["adapter_attention_grad_norm"]
                        ),
                        "probe_gate": float(direct_probe["gate_grad_norm"]),
                    }
                )
            if any(value != 0.0 for value in zero_action_norms.values()):
                raise RuntimeError(
                    "zero-weight action batch produced an action-path gradient: "
                    f"{zero_action_norms}"
                )

        accelerator.clip_grad_norm_(training_module.parameters(), max_grad_norm)
        optimizer.step()
        raw = raw_training_module
        if not all(
            bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in raw.conditioner.parameters()
        ):
            raise FloatingPointError("head/adapter parameters became non-finite")

        official_summary = diagnostics.pop("official_layer16_distribution")
        contrastive_summary = diagnostics.pop("paired_clean_layer16_distribution")
        paired_action_summary = diagnostics.pop("paired_layer16_distribution")
        paired_summary = (
            contrastive_summary
            if paired_mode == "contrastive"
            else paired_action_summary
            if paired_mode == "action"
            else None
        )
        if not isinstance(official_summary, Mapping):
            raise RuntimeError("official Layer-16 distribution summary is missing")
        if paired_active:
            if not isinstance(paired_summary, Mapping):
                raise RuntimeError("paired Layer-16 distribution summary is missing")
        # One shared object gather gives the main-process artifact global task
        # coverage and global Layer-16 moments under DDP.  Passing a one-item
        # list is intentional: Accelerate flattens gathered object lists.
        gathered_stream_records = gather_object(
            [
                {
                    "official_summary": dict(official_summary),
                    "paired_summary": (
                        dict(paired_summary) if isinstance(paired_summary, Mapping) else None
                    ),
                    "official_tasks": list(official_tasks),
                    "paired_tasks": list(paired_tasks),
                    "official_sample_ids": list(official_sample_ids),
                    "paired_state_ids": list(paired_state_ids),
                }
            ]
        )
        step_official_tasks: list[str] = []
        step_paired_tasks: list[str] = []
        step_official_sample_ids: list[str] = []
        step_paired_state_ids: list[str] = []
        for stream_record in gathered_stream_records:
            record_official_tasks = tuple(
                str(value) for value in stream_record["official_tasks"]
            )
            record_paired_tasks = tuple(
                str(value) for value in stream_record["paired_tasks"]
            )
            record_official_sample_ids = tuple(
                str(value) for value in stream_record["official_sample_ids"]
            )
            record_paired_state_ids = tuple(
                str(value) for value in stream_record["paired_state_ids"]
            )
            official_distribution.add(
                stream_record["official_summary"], tasks=record_official_tasks
            )
            if isinstance(stream_record["paired_summary"], Mapping):
                paired_distribution.add(
                    stream_record["paired_summary"], tasks=record_paired_tasks
                )
            seen_official_tasks.extend(record_official_tasks)
            seen_paired_tasks.extend(record_paired_tasks)
            seen_official_sample_ids.extend(record_official_sample_ids)
            seen_paired_state_ids.extend(record_paired_state_ids)
            step_official_tasks.extend(record_official_tasks)
            step_paired_tasks.extend(record_paired_tasks)
            step_official_sample_ids.extend(record_official_sample_ids)
            step_paired_state_ids.extend(record_paired_state_ids)

        row = {
            "step": step,
            "official_tasks": ";".join(step_official_tasks),
            "paired_tasks": ";".join(step_paired_tasks),
            "official_sample_ids": ";".join(step_official_sample_ids),
            "paired_physical_state_ids": ";".join(step_paired_state_ids),
            "loss_total": _scalar_mean(accelerator, float(diagnostics["loss_total"])),
            "loss_action": _scalar_mean(accelerator, float(diagnostics["loss_action"])),
            "loss_paired_action": _scalar_mean(
                accelerator, float(diagnostics["loss_paired_action"])
            ),
            "loss_contrastive": _scalar_mean(
                accelerator, float(diagnostics["loss_contrastive"])
            ),
            "paired_contrastive_gradient_enabled": bool(
                diagnostics["paired_contrastive_gradient_enabled"]
            ),
            "paired_contrastive_active": bool(
                diagnostics["paired_contrastive_active"]
            ),
            "paired_active_index": (
                paired_active_count(step) - 1 if paired_active and pair280_enabled else ""
            ),
            "positive_similarity": _scalar_mean(
                accelerator, float(diagnostics["positive_similarity"])
            ),
            "negative_similarity": _scalar_mean(
                accelerator, float(diagnostics["negative_similarity"])
            ),
            "step_rng_policy_id": str(diagnostics["step_rng_policy_id"]),
            "step_rng_step_index": int(diagnostics["step_rng_step_index"]),
            "step_rng_training_seed": int(diagnostics["step_rng_training_seed"]),
            "step_rng_process_index": int(diagnostics["step_rng_process_index"]),
            "official_rng_seed": int(diagnostics["official_rng_seed"]),
            "paired_rng_seed": (
                int(diagnostics["paired_rng_seed"])
                if diagnostics["paired_rng_seed"] is not None
                else ""
            ),
            "official_data_seed": data_seed,
            "gate_raw": float(raw.conditioner.adapter.gate.detach().float().item()),
            "gate_tanh": raw.conditioner.adapter.gate_value,
            "content_head_grad_norm": float(head_report["gradient_norm"]),
            "adapter_grad_norm": float(adapter_report["gradient_norm"]),
            "adapter_attention_grad_norm": float(
                adapter_attention_report["gradient_norm"]
            ),
            "adapter_gate_grad_norm": gate_grad_norm,
            "action_only_content_token_grad_norm": content_grad_norm,
            "action_only_head_probe_grad_norm": float(direct_probe["head_grad_norm"]),
            "action_only_adapter_attention_probe_grad_norm": float(
                direct_probe["adapter_attention_grad_norm"]
            ),
            "action_only_gate_probe_grad_norm": float(direct_probe["gate_grad_norm"]),
            "action_dit_grad_norm": float(action_report["gradient_norm"]),
            "video_backbone_grad_norm": 0.0,
            "vae_grad_norm": 0.0,
            "trainable_parameters": counts["total"],
            "head_adapter_lr": float(optimizer.param_groups[0]["lr"]),
            "action_dit_lr": (
                float(optimizer.param_groups[1]["lr"])
                if len(optimizer.param_groups) > 1
                else 0.0
            ),
            "lr_scheduler": lr_scheduler_mode,
            "layer16_shape": json.dumps(diagnostics["video_token_shape"]),
            "zc_shape": json.dumps(diagnostics["content_token_shape"]),
            "action_token_shape": json.dumps(diagnostics["action_token_shape"]),
            "loss_finite": True,
            "gradients_finite": True,
            "action_timestep_min": float(diagnostics["action_timestep_min"]),
            "action_timestep_max": float(diagnostics["action_timestep_max"]),
            "action_weight_min": float(diagnostics["action_weight_min"]),
            "action_weight_max": float(diagnostics["action_weight_max"]),
            "action_effective_weight_sum": float(
                diagnostics["action_effective_weight_sum"]
            ),
            "action_supervision_signal_positive": action_signal_positive,
            "zero_action_signal_reason": str(
                diagnostics["zero_action_signal_reason"]
            ),
            "action_valid_steps_total": int(diagnostics["action_valid_steps_total"]),
            "action_unweighted_mse_mean": float(
                diagnostics["action_unweighted_mse_mean"]
            ),
        }
        rows.append(row)
        gradient_steps.append(
            {
                "step": step,
                "gate_raw_after_step": row["gate_raw"],
                "combined": {
                    "content_head": head_report,
                    "adapter": adapter_report,
                    "adapter_attention_action_only_by_construction": adapter_attention_report,
                    "action_dit": action_report,
                    "video_backbone": {"gradient_tensors": 0, "gradient_norm": 0.0},
                    "vae": {"gradient_tensors": 0, "gradient_norm": 0.0},
                },
                "action_only_probe": direct_probe,
                "action_only_official_content_token_grad_norm": content_grad_norm,
                "gate_gradient_norm": gate_grad_norm,
                "action_supervision_signal_positive": action_signal_positive,
                "zero_weight_action_step": not action_signal_positive,
                "zero_action_signal_reason": str(
                    diagnostics["zero_action_signal_reason"]
                ),
                "loss_action": float(action_loss.detach().item()),
                "action_weight_min": float(diagnostics["action_weight_min"]),
                "action_weight_max": float(diagnostics["action_weight_max"]),
                "action_effective_weight_sum": float(
                    diagnostics["action_effective_weight_sum"]
                ),
            }
        )
        if accelerator.is_main_process:
            print(json.dumps(row, sort_keys=True), flush=True)

        progress.completed_step = step
        progress.positive_action_signal_steps = positive_action_signal_steps
        progress.zero_action_signal_steps = zero_action_signal_steps
        if (
            checkpoint_interval_steps > 0
            and step % checkpoint_interval_steps == 0
        ):
            checkpoint_dir = _save_stage2_native_checkpoint(
                accelerator=accelerator,
                output_dir=output_dir,
                progress=progress,
                raw_training_module=raw_training_module,
                base_checkpoint=base_checkpoint,
                regime=regime,
                config=config,
                base_identity=base_identity,
                identities=identities,
                checkpoint_interval_steps=checkpoint_interval_steps,
            )
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "checkpoint_step": step,
                            "native_accelerate_state": str(checkpoint_dir),
                            "status": "PASS",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if positive_action_signal_steps < 2:
        raise RuntimeError(
            "training did not observe enough positive-weight action batches to "
            "audit gate opening and the downstream action path"
        )
    action_path_coverage = _positive_action_path_coverage(gradient_steps)
    required_coverage = (
        "gate_positive_steps",
        "adapter_attention_positive_steps",
        "official_content_token_positive_steps",
    )
    if regime.replace("-", "_") == "p_v2":
        required_coverage += ("action_dit_positive_steps",)
    missing_coverage = [
        name for name in required_coverage if action_path_coverage[name] <= 0
    ]
    if missing_coverage:
        raise RuntimeError(
            "training never proved cumulative action-path connectivity: "
            f"{missing_coverage}"
        )

    if progress.completed_step != max_steps:
        raise RuntimeError("Stage-2 progress did not reach configured max_steps")
    if pair280_enabled and not bool(paired_cfg.get("engineering_smoke", False)):
        expected_pair_exposures = PAIR280_GROUPS * 10
        if len(seen_paired_state_ids) != expected_pair_exposures:
            raise RuntimeError(
                "Pair-280 did not consume exactly ten exposures of every state: "
                f"{len(seen_paired_state_ids)} != {expected_pair_exposures}"
            )
        exposure_counts: dict[str, int] = {}
        for state_id in seen_paired_state_ids:
            exposure_counts[state_id] = exposure_counts.get(state_id, 0) + 1
        if len(exposure_counts) != PAIR280_GROUPS or set(exposure_counts.values()) != {10}:
            raise RuntimeError("Pair-280 state exposures are not exactly ten per state")
        assert paired_iterator is not None
        try:
            next(paired_iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Pair-280 paired loader contains unaudited extra batches")
    if checkpoint_interval_steps > 0 and max_steps % checkpoint_interval_steps != 0:
        _save_stage2_native_checkpoint(
            accelerator=accelerator,
            output_dir=output_dir,
            progress=progress,
            raw_training_module=raw_training_module,
            base_checkpoint=base_checkpoint,
            regime=regime,
            config=config,
            base_identity=base_identity,
            identities=identities,
            checkpoint_interval_steps=checkpoint_interval_steps,
        )

    accelerator.wait_for_everyone()
    raw = raw_training_module
    update_audit = update_snapshot.compare(
        {
            "content_head": raw.conditioner.head,
            "adapter": raw.conditioner.adapter,
        }
    )
    if update_audit["max_abs_delta_by_module"]["content_head"] <= 0.0:
        raise RuntimeError("short training did not update Content Head")
    if update_audit["max_abs_delta_by_module"]["adapter"] <= 0.0:
        raise RuntimeError("short training did not update action adapter")
    if float(raw.conditioner.adapter.gate.detach().float().item()) == 0.0:
        raise RuntimeError("short training did not open the adapter gate")
    action_update = (
        action_snapshot.compare({"action_expert": raw.model.action_expert})
        if action_snapshot is not None
        else {"changed": False, "reason": "P-v1 ActionDiT frozen"}
    )
    if regime.replace("-", "_") == "p_v2":
        action_update["changed"] = bool(action_update["changed_elements"] > 0)
        action_update["optimizer_exp_avg"] = action_snapshot.optimizer_state_report(
            {"action_expert": raw.model.action_expert}, optimizer
        )
        if float(action_update["changed_fraction"]) < 0.5:
            raise RuntimeError(
                "P-v2 changed fewer than half of sampled ActionDiT elements"
            )
        if float(action_update["optimizer_exp_avg"]["nonzero_fraction"]) < 0.5:
            raise RuntimeError(
                "P-v2 Adam exp_avg is nonzero for fewer than half of sampled elements"
            )
        required_changed = sum(
            int(action_update["by_parameter"][name]["changed_elements"] > 0)
            for name in ACTION_UPDATE_REQUIRED_PARAMETERS
        )
        if required_changed < 6:
            raise RuntimeError(
                "P-v2 sampled update did not reach at least six of eight required strata"
            )
        deployment_categories = {
            "early": (
                "action_expert.action_encoder.weight",
                "action_expert.blocks.0.cross_attn.q.weight",
            ),
            "mid": (
                "action_expert.blocks.10.self_attn.o.weight",
                "action_expert.blocks.20.ffn.2.weight",
            ),
            "late": ("action_expert.blocks.29.cross_attn.o.weight",),
            "head": ("action_expert.head.weight",),
        }
        category_visibility = {
            category: any(
                action_update["by_parameter"][name][
                    "deployment_visible_changed_elements"
                ]
                > 0
                for name in names
            )
            for category, names in deployment_categories.items()
        }
        action_update["required_changed_strata"] = required_changed
        action_update["bf16_deployment_category_visibility"] = category_visibility
        if not all(category_visibility.values()):
            raise RuntimeError(
                "P-v2 update would disappear after BF16 deployment in a required "
                f"ActionDiT category: {category_visibility}"
            )
    floating_optimizer_state_dtypes = sorted(
        {
            str(value.dtype)
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
        }
    )
    if floating_optimizer_state_dtypes != ["torch.float32"]:
        raise RuntimeError(
            "AdamW floating state must be exactly fp32, got "
            f"{floating_optimizer_state_dtypes}"
        )
    precision_audit["optimizer_floating_state_dtypes"] = floating_optimizer_state_dtypes

    if not bool(config.get("formal", False)):
        if set(seen_official_tasks) != set(TASKS):
            raise RuntimeError(
                f"official smoke did not cover exact three tasks: {seen_official_tasks}"
            )
        if paired_mode != "none" and set(seen_paired_tasks) != set(TASKS):
            raise RuntimeError(
                f"paired smoke did not cover exact three tasks: {seen_paired_tasks}"
            )

    if accelerator.is_main_process:
        official_audit = {
            **official_dataset_audit,
            "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "raw_camera_shape": [3, 480, 640],
            "model_composite_shape": [3, 384, 320],
            "fps": 50,
            "window": {"state_steps": 33, "action_steps": 32, "video_frames": 9},
            "action_dim": 14,
            "proprio_dim": 14,
            "prompt_source": "per-frame official task paraphrase",
        }
        if paired_dataset is not None:
            if paired_mode == "contrastive":
                if pair280_enabled:
                    assert isinstance(paired_dataset, ShardedPair280TokenDataset)
                    paired_audit = _pair280_paired_dataset_audit(
                        cache_manifest_identity=identities["paired_train_cache"],
                        physical_state_groups=len(paired_dataset),
                    )
                else:
                    assert isinstance(paired_dataset, FrozenPairedTokenDataset)
                    paired_audit = audit_frozen_token_cache(
                        paired_dataset,
                        expected_extraction_contract=expected_cache_extraction_contract,
                        layer=EXACT_LAYER,
                        split="train",
                        verified_cache_identity=identities["paired_train_cache"],
                        expected_backbone_sha256=base_identity["sha256"],
                        expected_base_lineage_sha256=identities[
                            "base_lineage_manifest"
                        ]["sha256"],
                        expected_release_paired_binding_sha256=identities[
                            "release_paired_binding_manifest"
                        ]["sha256"],
                        expected_action_manifest_sha256=identities["paired_action_manifest"]["sha256"],
                        expected_action_audit_sha256=identities["paired_action_audit"]["sha256"],
                        expected_state_bank_sha256=identities["paired_state_bank"]["sha256"],
                    )
                paired_audit["native_action_source_contract"] = native_paired_contract_audit
            else:
                assert isinstance(paired_dataset, NativePairedActionDataset)
                paired_audit = audit_native_paired_action_dataset(paired_dataset)
                paired_audit["native_action_source_contract"] = native_paired_contract_audit
            paired_audit["shared_state_bank_contract"] = state_bank_audit
            data_provenance = build_dual_stream_provenance(
                official=official_audit,
                paired=paired_audit,
            )
            distribution_audit = _distribution_report(
                official_distribution,
                paired_distribution,
                supervision_mode=paired_mode,
            )
        else:
            paired_audit = {
                "enabled": False,
                "reason": "no paired dataset was configured",
                "supervision_mode": "none",
                "r3_role": "not_consumed",
            }
            data_provenance = {
                "audit_schema_version": 1,
                "official": official_audit,
                "paired": paired_audit,
                "stream_contract": {
                    "concatenated": False,
                    "official_role": "policy_action_supervision",
                    "paired_role": "disabled",
                    "paired_supervision_mode": "none",
                },
            }
            distribution_audit = {
                "status": "NOT_RUN_FOR_ARCHITECTURE_ONLY_CONTROL",
                "official_clean_claim_supported": False,
            }
        _write_json(output_dir / "data_provenance_audit.json", data_provenance)
        _write_json(output_dir / "data_distribution_audit.json", distribution_audit)
        _write_json(
            output_dir / "gradient_audit.json",
            {
                "status": "PASS",
                "regime": regime,
                "zero_gate_first_step_semantics": (
                    "On the first positive-weight action batch, action loss reaches only "
                    "the exact-zero gate; paired contrastive reaches the Head directly "
                    "only when lambda_contrastive>0. From the next positive-weight batch, "
                    "action-only gradients reach head, GCA weights, and Zc. Native "
                    "scheduler endpoint batches are separately audited as exact-zero "
                    "action supervision."
                ),
                "positive_action_signal_steps": positive_action_signal_steps,
                "zero_action_signal_steps": zero_action_signal_steps,
                "positive_action_path_coverage": action_path_coverage,
                "steps": gradient_steps,
            },
        )
        _write_json(output_dir / "parameter_update_audit.json", {
            "head_and_adapter": update_audit,
            "action_dit": action_update,
            "final_content_head_sha256": module_state_sha256(raw.conditioner.head),
            "final_adapter_sha256": module_state_sha256(raw.conditioner.adapter),
        })
        _write_csv(output_dir / "train_log.csv", rows)

        training_sequence_audit = {
            "status": "PASS",
            "official_sample_sequence_sha256": canonical_json_sha256(
                seen_official_sample_ids
            ),
            "paired_physical_state_sequence_sha256": canonical_json_sha256(
                seen_paired_state_ids
            ),
            "matched_stream_contract_sha256": (
                matched_stream_contract["sha256"]
                if matched_stream_contract is not None
                else None
            ),
            "official_sample_count": len(seen_official_sample_ids),
            "paired_physical_state_count": len(seen_paired_state_ids),
            "paired_sampling_profile": (
                PAIR280_PROFILE_ID if pair280_enabled else None
            ),
            "paired_active_steps": (
                paired_active_count(max_steps) if pair280_enabled else max_steps
            ),
            "paired_exact_exposures_per_state": 10 if pair280_enabled else None,
        }
        config["resolved_training_sequence_audit"] = training_sequence_audit
        _write_json(
            output_dir / "training_sequence_audit.json", training_sequence_audit
        )
        _write_json(output_dir / "run_config.json", config)
        checkpoint_path = save_policy_checkpoint(
            output_dir / "checkpoint.pt",
            model=raw.model,
            conditioner=raw.conditioner,
            base_checkpoint=base_checkpoint,
            regime=regime,
            step=max_steps,
            run_config=config,
            optimizer=(optimizer if bool(config["training"].get("save_optimizer", False)) else None),
            include_base_sha256=True,
            verified_base_identity=base_identity,
            artifact_identities=identities,
        )
        optimizer_groups = [
            {
                "name": str(group.get("name", f"group_{index}")),
                "lr": float(group["lr"]),
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            }
            for index, group in enumerate(optimizer.param_groups)
        ]
        formal_training = _is_formal_training_config(config)
        summary = {
            "status": "COMPLETE" if formal_training else "SMOKE_COMPLETE",
            "formal_training_auto_started": False,
            "regime": regime,
            "control": config["control"],
            "paired_supervision_mode": paired_mode,
            "lambda_contrastive": coefficient,
            "paired_contrastive_gradient_enabled": coefficient > 0.0,
            "paired_sampling_profile": (
                PAIR280_PROFILE_ID if pair280_enabled else None
            ),
            "paired_schedule": (
                dict(config["paired"]["schedule"]) if pair280_enabled else None
            ),
            "steps": max_steps,
            "runtime_batch_contract": runtime_batch_contract,
            "official_loader_rng_contract": official_loader_rng_contract,
            "stage2_step_rng_contract": stage2_step_rng_contract,
            "checkpoint": str(checkpoint_path),
            "base_release_load": asdict(release_audit),
            "base_lineage_manifest_sha256": identities[
                "base_lineage_manifest"
            ]["sha256"],
            "release_paired_binding_manifest_sha256": identities[
                "release_paired_binding_manifest"
            ]["sha256"],
            "matched_stream_contract_sha256": (
                matched_stream_contract["sha256"]
                if matched_stream_contract is not None
                else None
            ),
            "head_init": {
                "mode": head_init_mode,
                "seed": head_seed,
                "identity": identities.get("head_init"),
                "experiment": head_payload.get("experiment"),
                "layer": head_payload.get("layer"),
                "checkpoint_kind": head_payload.get("checkpoint_kind"),
            },
            "parameter_counts": counts,
            "precision": precision_audit,
            "optimizer_groups": optimizer_groups,
            "lr_scheduler": {
                "name": lr_scheduler_mode,
                "step_calls": 0,
                "semantics": "fixed learning rates for every optimizer step",
            },
            "checkpointing": {
                "engine": "accelerate_save_state_load_state",
                "save_every": checkpoint_interval_steps,
                "state_root": str(output_dir / "checkpoints" / "state"),
                "model_optimizer_rng_saved": checkpoint_interval_steps > 0,
                "dual_stream_progress_registered": checkpoint_interval_steps > 0,
                "resumed_from_step": resume_step,
            },
            "initialization": initialization,
            "final_gate_raw": float(raw.conditioner.adapter.gate.detach().float().item()),
            "final_gate_tanh": raw.conditioner.adapter.gate_value,
            "parameter_updates": {
                "head_and_adapter": update_audit,
                "action_dit": action_update,
            },
            "official_task_sequence": seen_official_tasks,
            "paired_task_sequence": seen_paired_tasks,
            "official_sample_sequence_sha256": canonical_json_sha256(
                seen_official_sample_ids
            ),
            "paired_physical_state_sequence_sha256": (
                canonical_json_sha256(seen_paired_state_ids)
                if paired_mode != "none"
                else None
            ),
            "training_sequence_audit": training_sequence_audit,
            "dual_stream_cycles": _dual_stream_cycle_audit(official_iterator),
            "identity_audit": identity_audit,
            "last_metrics": rows[-1],
            "deliverable_status": _training_deliverable_status(
                formal=formal_training
            ),
        }
        _write_json(output_dir / "training_summary.json", summary)
    accelerator.wait_for_everyone()
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume a strict native Accelerate state directory (default: latest).",
    )
    parser.add_argument(
        "--resume-amendment",
        help="Strict post-incident amendment required by an amended full5ep resume.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    destination = run(
        args.config,
        output_override=args.output_dir,
        resume_from=args.resume,
        resume_amendment=args.resume_amendment,
    )
    print(f"policy content-adapter run finished: {destination}")


if __name__ == "__main__":
    main()
