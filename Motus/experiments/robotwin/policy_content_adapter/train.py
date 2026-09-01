"""Distributed M1/M3 Motus adapter training entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config_audit import load_yaml, validate_run_config
from .dual_stream import audit_dual_stream_gradients, compute_dual_stream_loss
from .model import (
    GatedCrossAttentionAdapter,
    MotusContentHead,
    MotusPolicyContentConditioner,
    configure_trainable_parameters,
    optimizer_parameter_groups,
)
from .official_data import MotusOfficialDataset
from .paired_data import sha256_file
from .runtime import instantiate_author_release, load_lineage
from .sampler import (
    DeterministicMotusEpochBatchSampler,
    DeterministicSameTaskBatchSampler,
    DeterministicStepBatchSampler,
    action_step_seed,
    sampler_sequence_sha256,
)
from .source_audit import validate_source_audit
from .task_text_cache import load_task_embeddings, validate_task_text_cache
from .token_cache import FrozenMotusTokenDataset, validate_token_cache


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TRAINING_SUMMARY_SCHEMA = "motus_policy_content_adapter_training_summary"
RESUME_SIDECAR_SCHEMA = "motus_policy_content_adapter_distributed_resume"


def _sha(path: str | Path) -> str:
    return sha256_file(Path(path))


def resolve_deepspeed_config(
    path: str | Path, training: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind DeepSpeed batch/clipping values to the audited run config."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    micro = int(training["per_device_batch"])
    accumulation = int(training["gradient_accumulation_steps"])
    world = int(training["world_size"])
    config.update(
        {
            "train_micro_batch_size_per_gpu": micro,
            "gradient_accumulation_steps": accumulation,
            "train_batch_size": micro * accumulation * world,
            "gradient_clipping": float(training["grad_clip_norm"]),
        }
    )
    return config


def build_training_scheduler(
    optimizer: torch.optim.Optimizer, training: Mapping[str, Any]
) -> Any:
    """Build either the legacy smoke scheduler or Motus's author scheduler."""

    scheduler_type = str(training.get("scheduler", "constant"))
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if scheduler_type == "motus_author_linear":
        from utils.scheduler import LambdaLinearScheduler

        return LambdaLinearScheduler(
            optimizer,
            warm_up_steps=int(training["warmup_steps"]),
            cycle_length=int(training["cycle_length"]),
            f_max=float(training["f_max"]),
            f_min=float(training["f_min"]),
            f_start=float(training["f_start"]),
        )
    raise ValueError(f"unsupported scheduler {scheduler_type!r}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def module_tensor_sha256(module: torch.nn.Module) -> str:
    """Hash tensor names, shapes, dtypes and exact bytes in state-dict order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(b"\0")
        if value.numel():
            digest.update(
                value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _deepspeed_gradient_snapshot(
    engine: Any,
    *,
    conditioner: MotusPolicyContentConditioner,
    optimizer_group_names: list[str],
) -> dict[str, float]:
    """Read reduced ZeRO-1 gradient partitions before ``engine.step``."""

    from deepspeed.utils import safe_get_full_grad

    zero_optimizer = getattr(engine, "optimizer", None)
    averaged = getattr(zero_optimizer, "averaged_gradients", None)
    if not isinstance(averaged, dict):
        raise RuntimeError("DeepSpeed averaged gradient partitions are unavailable")
    if sorted(averaged) != list(range(len(optimizer_group_names))):
        raise RuntimeError("DeepSpeed optimizer group layout changed")
    norms: dict[str, float] = {}
    for group_index, name in enumerate(optimizer_group_names):
        fragments = averaged.get(group_index)
        if not isinstance(fragments, list) or not fragments:
            raise RuntimeError(f"DeepSpeed gradient group {name!r} is unavailable")
        squared = torch.zeros(
            (), device=conditioner.adapter.gate.device, dtype=torch.float32
        )
        finite = torch.ones(
            (), device=conditioner.adapter.gate.device, dtype=torch.float32
        )
        for fragment in fragments:
            value = fragment.detach().float()
            is_finite = torch.isfinite(value)
            finite.mul_(is_finite.all().float())
            squared.add_(torch.where(is_finite, value, 0.0).square().sum())
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(squared, op=torch.distributed.ReduceOp.SUM)
        if finite.item() != 1.0 or not bool(torch.isfinite(squared).item()):
            raise FloatingPointError(f"DeepSpeed gradient group {name!r} is non-finite")
        norms[name] = float(torch.sqrt(squared).item())
    audited_parameters = {
        "content_queries": conditioner.head.content_queries,
        "adapter_gate": conditioner.adapter.gate,
        "adapter_out_bias": conditioner.adapter.cross_attention.out_proj.bias,
    }
    audited_gradients = {
        name: safe_get_full_grad(parameter)
        for name, parameter in audited_parameters.items()
    }
    if any(value is None for value in audited_gradients.values()):
        raise RuntimeError("DeepSpeed did not expose an audited adapter gradient")
    if any(
        not bool(torch.isfinite(value).all().item())
        for value in audited_gradients.values()
    ):
        raise FloatingPointError("DeepSpeed audited adapter gradient is non-finite")
    head_gradient = audited_gradients["content_queries"].detach().float()
    gate_gradient = audited_gradients["adapter_gate"].detach().float()
    adapter_bias_gradient = audited_gradients["adapter_out_bias"].detach().float()
    if gate_gradient.numel() != 1:
        raise RuntimeError("DeepSpeed adapter-gate gradient shape changed")
    gate_value = float(gate_gradient.item())
    adapter_subset_norm = float(
        torch.sqrt(
            gate_gradient.square().sum()
            + adapter_bias_gradient.square().sum()
        ).item()
    )
    return {
        "content_head_grad_norm": float(
            torch.linalg.vector_norm(head_gradient).item()
        ),
        "adapter_grad_norm": adapter_subset_norm,
        "adapter_gate_grad": gate_value,
        "action_expert_grad_norm": norms.get("action_expert", 0.0),
    }


def _official_collator(processor):
    from data.dataset import (
        _process_language_embeddings_batch,
        _process_vlm_inputs_batch,
    )
    from data.utils.image_utils import tensor_to_pil
    from utils.vlm_utils import preprocess_vlm_messages

    def collate(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
        first_frames = torch.stack([item["first_frame"] for item in batch])
        vlm = [
            preprocess_vlm_messages(
                item["text_instruction"],
                tensor_to_pil(item["first_frame"]),
                processor,
            )
            for item in batch
        ]
        return {
            "first_frame": first_frames,
            "video_frames": torch.stack([item["video_frames"] for item in batch]),
            "initial_state": torch.stack([item["initial_state"] for item in batch]),
            "action_sequence": torch.stack([item["action_sequence"] for item in batch]),
            "language_embedding": _process_language_embeddings_batch(
                [item["language_embedding"] for item in batch]
            ),
            "vlm_inputs": _process_vlm_inputs_batch(vlm),
            "task": [item["task"] for item in batch],
            "domain": [item["domain"] for item in batch],
            "episode_index": [int(item["episode_index"]) for item in batch],
            "condition_frame_index": [
                int(item["condition_frame_index"]) for item in batch
            ],
            "virtual_epoch": [int(item["virtual_epoch"]) for item in batch],
            "virtual_sample_index": [
                int(item["virtual_sample_index"]) for item in batch
            ],
        }

    return collate


def _paired_collate(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "visual_tokens": torch.stack(
            [item["visual_tokens"] for item in batch], dim=0
        ),
        "physical_state_ids": [item["physical_state_id"] for item in batch],
        "task_ids": [item["task"] for item in batch],
    }


def _artifact_path(config: Mapping[str, Any], name: str) -> Path:
    return Path(config["artifacts"][name]["path"]).resolve()


def _validate_strict_audit(path: Path, lineage_path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "motus_robotwin2_strict_load_audit" or value.get("status") != "PASS":
        raise RuntimeError("strict-load audit is not PASS")
    if value.get("lineage_manifest", {}).get("sha256") != _sha(lineage_path):
        raise RuntimeError("strict-load audit lineage mismatch")


def _validate_zero_gate_audit(
    path: Path, lineage_path: Path, official_manifest: Path
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "motus_policy_content_adapter_zero_gate_audit" or value.get("status") != "PASS":
        raise RuntimeError("zero-gate audit is not PASS")
    if value.get("lineage_manifest", {}).get("sha256") != _sha(lineage_path):
        raise RuntimeError("zero-gate audit lineage mismatch")
    if value.get("official_manifest", {}).get("sha256") != _sha(official_manifest):
        raise RuntimeError("zero-gate audit official-data mismatch")
    if value.get("action_bit_exact") is not True or value.get("video_bit_exact") is not True:
        raise RuntimeError("zero-gate audit is not bit-exact")


def _config_sha(config_path: Path) -> str:
    return _sha(config_path)


def _resume_sidecar(
    path: Path,
    *,
    config_sha256: str,
    optimizer_step: int,
    micro_step: int,
    artifact_shas: Mapping[str, str],
) -> dict[str, Any]:
    value = {
        "schema": RESUME_SIDECAR_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "config_sha256": config_sha256,
        "optimizer_step": int(optimizer_step),
        "next_micro_step": int(micro_step),
        "artifact_shas": dict(artifact_shas),
    }
    (path / "resume_sidecar.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _load_resume_sidecar(
    checkpoint: Path,
    *,
    config_sha256: str,
    artifact_shas: Mapping[str, str],
) -> dict[str, Any]:
    path = checkpoint / "resume_sidecar.json"
    if not path.is_file():
        raise RuntimeError("resume checkpoint has no policy sidecar")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != RESUME_SIDECAR_SCHEMA or value.get("status") != "PASS":
        raise RuntimeError("resume sidecar is invalid")
    if value.get("config_sha256") != config_sha256:
        raise RuntimeError("resume config SHA changed")
    if value.get("artifact_shas") != dict(artifact_shas):
        raise RuntimeError("resume artifact ancestry changed")
    return value


def _save_distributed_state(
    accelerator,
    checkpoint: Path,
    *,
    config_sha256: str,
    optimizer_step: int,
    next_micro_step: int,
    artifact_shas: Mapping[str, str],
) -> None:
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint}")
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _resume_sidecar(
            checkpoint,
            config_sha256=config_sha256,
            optimizer_step=optimizer_step,
            micro_step=next_micro_step,
            artifact_shas=artifact_shas,
        )
    accelerator.wait_for_everyone()


def run(config_path: str | Path, *, deepspeed_config: str | None, resume: str | None) -> Path:
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin

    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    validate_run_config(config, require_runnable=resume is None)
    training = config["training"]
    objective = config["objective"]
    model_config = config["model"]
    output = Path(config["output_dir"]).resolve()
    if resume is None and output.exists():
        raise FileExistsError(f"refusing to overwrite run {output}")

    resolved_deepspeed = (
        resolve_deepspeed_config(deepspeed_config, training)
        if deepspeed_config
        else None
    )
    accelerator = Accelerator(
        deepspeed_plugin=(
            DeepSpeedPlugin(hf_ds_config=resolved_deepspeed)
            if resolved_deepspeed is not None
            else None
        ),
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        mixed_precision="bf16",
    )
    if accelerator.num_processes != int(training["world_size"]):
        raise RuntimeError(
            f"world size {accelerator.num_processes} != config {training['world_size']}"
        )
    rank = accelerator.process_index
    local_rank = accelerator.local_process_index

    lineage_path = _artifact_path(config, "base_lineage_manifest")
    implementation_path = _artifact_path(config, "implementation_audit")
    strict_path = _artifact_path(config, "strict_load_audit")
    zero_gate_path = _artifact_path(config, "zero_gate_audit")
    official_manifest = _artifact_path(config, "official_manifest")
    token_cache = _artifact_path(config, "frozen_token_cache")
    text_cache = _artifact_path(config, "task_text_cache")
    artifact_shas = {
        name: str(config["artifacts"][name]["sha256"])
        for name in config["artifacts"]
    }
    validate_source_audit(
        json.loads(implementation_path.read_text(encoding="utf-8")),
        verify_files=True,
    )
    _validate_strict_audit(strict_path, lineage_path)
    _validate_zero_gate_audit(
        zero_gate_path, lineage_path, official_manifest
    )
    lineage = load_lineage(lineage_path, verify_files=True)
    validate_task_text_cache(text_cache, verify_encoder_assets=False)
    validate_token_cache(
        token_cache,
        expected_paired_manifest_sha256=artifact_shas["paired_manifest"],
        expected_base_lineage_sha256=artifact_shas["base_lineage_manifest"],
        verify_shards=True,
    )
    task_embeddings = load_task_embeddings(text_cache)
    seed = int(training["seed"])
    # Common initialization on all ranks.  The author checkpoint overwrites
    # Motus weights; this seed deterministically owns only new Head/GCA state.
    _seed_everything(seed)
    motus = instantiate_author_release(
        lineage,
        batch_size=int(training["per_device_batch"]),
        local_cuda_index=local_rank,
        strict=True,
    )
    adapter_cfg = model_config["adapter"]
    conditioner = MotusPolicyContentConditioner(
        MotusContentHead(
            backbone_dim=int(adapter_cfg["backbone_dim"]),
            content_dim=int(adapter_cfg["content_dim"]),
            num_queries=int(adapter_cfg["content_queries"]),
            num_heads=int(adapter_cfg["content_heads"]),
        ),
        GatedCrossAttentionAdapter(
            action_dim=int(adapter_cfg["action_dim"]),
            content_dim=int(adapter_cfg["content_dim"]),
            num_heads=int(adapter_cfg["action_heads"]),
        ),
        capture_layer=int(adapter_cfg["capture_layer"]),
    ).to(device=motus.device, dtype=motus.dtype)
    motus.set_policy_content_conditioner(conditioner)
    parameter_counts = configure_trainable_parameters(
        motus, conditioner, regime=model_config["regime"]
    )
    groups = optimizer_parameter_groups(
        motus,
        conditioner,
        head_adapter_lr=float(training["head_adapter_lr"]),
        action_expert_lr=(
            None
            if training["action_expert_lr"] is None
            else float(training["action_expert_lr"])
        ),
    )
    optimizer_group_names = [str(group["name"]) for group in groups]
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
    )
    scheduler = build_training_scheduler(optimizer, training)

    from accelerate.utils import DistributedType

    from .vlm_processor import load_qwen3_vl_processor

    processor = load_qwen3_vl_processor(lineage["vlm"]["root"])
    official_dataset = MotusOfficialDataset(
        official_manifest,
        task_embeddings=task_embeddings,
        training_seed=seed,
    )
    paired_dataset = FrozenMotusTokenDataset(token_cache, verify_shards=False)
    max_steps = int(training["max_steps"])
    accumulation = int(training["gradient_accumulation_steps"])
    total_micro_steps = max_steps * accumulation
    config_digest = _config_sha(config_path)
    start_optimizer_step = 0
    start_micro_step = 0
    if resume is not None:
        sidecar = _load_resume_sidecar(
            Path(resume).resolve(),
            config_sha256=config_digest,
            artifact_shas=artifact_shas,
        )
        start_optimizer_step = int(sidecar["optimizer_step"])
        start_micro_step = int(sidecar["next_micro_step"])
        if start_micro_step != start_optimizer_step * accumulation:
            raise RuntimeError("resume checkpoint is not on an optimizer boundary")

    if training.get("official_sampler") == "motus_distributed_drop_last_epoch_v1":
        official_sampler = DeterministicMotusEpochBatchSampler(
            dataset_size=len(official_dataset),
            local_batch_size=int(training["per_device_batch"]),
            world_size=accelerator.num_processes,
            rank=rank,
            training_seed=seed,
            total_micro_steps=total_micro_steps,
            start_micro_step=start_micro_step,
            include_epoch_in_index=True,
        )
        if int(training["steps_per_epoch"]) != official_sampler.steps_per_epoch:
            raise RuntimeError("formal steps_per_epoch disagrees with runtime sampler")
    else:
        official_sampler = DeterministicStepBatchSampler(
            dataset_size=len(official_dataset),
            local_batch_size=int(training["per_device_batch"]),
            world_size=accelerator.num_processes,
            rank=rank,
            training_seed=seed,
            stream="official",
            total_micro_steps=total_micro_steps,
            start_micro_step=start_micro_step,
            include_epoch_in_index=True,
        )
    paired_sampler = DeterministicSameTaskBatchSampler(
        task_labels=[str(item["task"]) for item in paired_dataset.index],
        local_batch_size=int(training.get("paired_groups_per_device", 2)),
        world_size=accelerator.num_processes,
        rank=rank,
        training_seed=seed,
        total_micro_steps=total_micro_steps,
        start_micro_step=start_micro_step,
    )
    official_loader = DataLoader(
        official_dataset,
        batch_sampler=official_sampler,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=_official_collator(processor),
        pin_memory=True,
    )
    paired_loader = DataLoader(
        paired_dataset,
        batch_sampler=paired_sampler,
        num_workers=0,
        collate_fn=_paired_collate,
        pin_memory=True,
    )
    # The batch samplers are already rank-local and step-addressed.  Preparing
    # them with Accelerate would shard a second time and destroy M1/M3 sequence
    # equivalence, so only model/optimizer/scheduler are wrapped here.
    motus, optimizer, scheduler = accelerator.prepare(
        motus, optimizer, scheduler
    )
    base = accelerator.unwrap_model(motus)
    conditioner = base.policy_content_conditioner
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    gradient_audit_backend = (
        "deepspeed_zero_partition_pre_step_v1"
        if is_deepspeed
        else "pytorch_parameter_grad"
    )
    # DeepSpeed canonicalizes all model tensors to its BF16 runtime dtype.
    # Freeze/update comparisons therefore start after prepare but before any
    # resume state is loaded or optimizer step is executed.
    initial_hash_stage = "post_accelerator_prepare_pre_optimizer_step"
    initial_conditioner_sha256 = (
        module_tensor_sha256(conditioner)
        if accelerator.is_main_process
        else None
    )
    initial_action_expert_sha256 = (
        module_tensor_sha256(base.action_expert)
        if accelerator.is_main_process
        else None
    )

    if resume is not None:
        accelerator.load_state(str(Path(resume).resolve()))
    if accelerator.is_main_process and resume is None:
        output.mkdir(parents=True)
        shutil.copy2(config_path, output / "requested_config.yaml")
        (output / "run_identity.json").write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "config_sha256": config_digest,
                    "artifact_shas": artifact_shas,
                    "parameter_counts": parameter_counts,
                    "initial_conditioner_sha256": initial_conditioner_sha256,
                    "initial_action_expert_sha256": initial_action_expert_sha256,
                    "initial_hash_stage": initial_hash_stage,
                    "official_sequence_sha256": sampler_sequence_sha256(
                        official_sampler
                    ),
                    "paired_sequence_sha256": sampler_sequence_sha256(
                        paired_sampler
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()

    log_path = output / f"train_rank{rank}.jsonl"
    log_handle = log_path.open("a", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = start_optimizer_step
    micro_step = start_micro_step
    started = time.monotonic()
    try:
        for official_batch, paired_batch in zip(
            official_loader, paired_loader, strict=True
        ):
            if optimizer_step >= max_steps:
                break
            step_seed = action_step_seed(seed, rank, micro_step)
            torch.manual_seed(step_seed)
            torch.cuda.manual_seed(step_seed)
            official_batch = _move_tensors(official_batch, accelerator.device)
            paired_batch = _move_tensors(paired_batch, accelerator.device)
            with accelerator.accumulate(motus):
                loss = compute_dual_stream_loss(
                    motus_model=base,
                    conditioner=conditioner,
                    official_batch=official_batch,
                    paired_visual_tokens=paired_batch["visual_tokens"],
                    paired_physical_state_ids=paired_batch["physical_state_ids"],
                    paired_task_ids=paired_batch["task_ids"],
                    control=config["control"],
                    lambda_contrastive=float(
                        objective["lambda_contrastive"]
                    ),
                    temperature=float(objective["temperature"]),
                )
                if is_deepspeed:
                    motus.set_gradient_accumulation_boundary(
                        is_boundary=accelerator.sync_gradients
                    )
                    motus.backward(loss.total)
                else:
                    accelerator.backward(loss.total)
                gradient_audit = None
                if accelerator.sync_gradients:
                    gradient_snapshot = (
                        _deepspeed_gradient_snapshot(
                            motus,
                            conditioner=conditioner,
                            optimizer_group_names=optimizer_group_names,
                        )
                        if is_deepspeed
                        else None
                    )
                    gradient_audit = audit_dual_stream_gradients(
                        motus_model=base,
                        conditioner=conditioner,
                        control=config["control"],
                        regime=model_config["regime"],
                        step=optimizer_step + 1,
                        gradient_snapshot=gradient_snapshot,
                    )
                    if not is_deepspeed:
                        accelerator.clip_grad_norm_(
                            [p for p in base.parameters() if p.requires_grad],
                            float(training["grad_clip_norm"]),
                        )
                if is_deepspeed:
                    if accelerator.sync_gradients:
                        motus.step()
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            micro_step += 1
            if accelerator.sync_gradients:
                optimizer_step += 1
            row = {
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "rank": rank,
                "action_rng_seed": step_seed,
                **loss.scalar_metrics(),
                "official_task": official_batch["task"],
                "official_domain": official_batch["domain"],
                "official_episode_index": official_batch["episode_index"],
                "official_condition_frame_index": official_batch[
                    "condition_frame_index"
                ],
                "paired_physical_state_ids": paired_batch[
                    "physical_state_ids"
                ],
                "paired_task_ids": paired_batch["task_ids"],
                "gradient_audit": gradient_audit,
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
            log_handle.write(json.dumps(row, sort_keys=True) + "\n")
            log_handle.flush()
            if (
                accelerator.sync_gradients
                and optimizer_step > 0
                and optimizer_step % int(training["checkpoint_interval"]) == 0
            ):
                _save_distributed_state(
                    accelerator,
                    output / "checkpoints" / f"step_{optimizer_step:08d}",
                    config_sha256=config_digest,
                    optimizer_step=optimizer_step,
                    next_micro_step=micro_step,
                    artifact_shas=artifact_shas,
                )
    finally:
        log_handle.close()

    accelerator.wait_for_everyone()
    if optimizer_step != max_steps or micro_step != total_micro_steps:
        raise RuntimeError(
            f"training stopped at optimizer/micro {optimizer_step}/{micro_step}, "
            f"expected {max_steps}/{total_micro_steps}"
        )
    final_checkpoint = output / "checkpoints" / f"step_{optimizer_step:08d}"
    if not final_checkpoint.exists():
        _save_distributed_state(
            accelerator,
            final_checkpoint,
            config_sha256=config_digest,
            optimizer_step=optimizer_step,
            next_micro_step=micro_step,
            artifact_shas=artifact_shas,
        )
    if accelerator.is_main_process:
        final_conditioner_sha256 = module_tensor_sha256(conditioner)
        final_action_expert_sha256 = module_tensor_sha256(base.action_expert)
        if final_conditioner_sha256 == initial_conditioner_sha256:
            raise RuntimeError("Content Head/GCA did not update")
        if model_config["regime"] == "m_p2":
            if final_action_expert_sha256 == initial_action_expert_sha256:
                raise RuntimeError("M-P2 Action Expert did not update")
        elif final_action_expert_sha256 != initial_action_expert_sha256:
            raise RuntimeError("M-P1 frozen Action Expert changed")
        deployment_path = output / "deployment_checkpoint.pt"
        torch.save(
            {
                "schema": "motus_policy_content_adapter_deployment_checkpoint",
                "schema_version": 1,
                "control": config["control"],
                "regime": model_config["regime"],
                "training_seed": seed,
                "optimizer_steps": optimizer_step,
                "artifact_shas": artifact_shas,
                "conditioner": conditioner.state_dict(),
                "action_expert": (
                    base.action_expert.state_dict()
                    if model_config["regime"] == "m_p2"
                    else None
                ),
                "initial_conditioner_sha256": initial_conditioner_sha256,
                "final_conditioner_sha256": final_conditioner_sha256,
                "initial_action_expert_sha256": initial_action_expert_sha256,
                "final_action_expert_sha256": final_action_expert_sha256,
                "initial_hash_stage": initial_hash_stage,
            },
            deployment_path,
        )
        summary = {
            "schema": TRAINING_SUMMARY_SCHEMA,
            "schema_version": 1,
            "status": "COMPLETE",
            "control": config["control"],
            "regime": model_config["regime"],
            "training_seed": seed,
            "lambda_contrastive": float(objective["lambda_contrastive"]),
            "temperature": float(objective["temperature"]),
            "optimizer_steps": optimizer_step,
            "micro_steps": micro_step,
            "world_size": accelerator.num_processes,
            "global_batch": int(training["global_batch"]),
            "training_profile": str(
                training.get("profile", "engineering_smoke")
            ),
            "epochs": training.get("epochs"),
            "steps_per_epoch": training.get("steps_per_epoch"),
            "scheduler_contract": {
                key: training.get(key)
                for key in (
                    "scheduler",
                    "warmup_steps",
                    "cycle_length",
                    "f_max",
                    "f_min",
                    "f_start",
                )
                if key in training
            },
            "gradient_audit_backend": gradient_audit_backend,
            "elapsed_seconds": time.monotonic() - started,
            "final_checkpoint": str(final_checkpoint),
            "deployment_checkpoint": {
                "path": str(deployment_path),
                "size_bytes": deployment_path.stat().st_size,
                "sha256": sha256_file(deployment_path),
            },
            "initial_conditioner_sha256": initial_conditioner_sha256,
            "final_conditioner_sha256": final_conditioner_sha256,
            "initial_action_expert_sha256": initial_action_expert_sha256,
            "final_action_expert_sha256": final_action_expert_sha256,
            "initial_hash_stage": initial_hash_stage,
            "artifact_shas": artifact_shas,
        }
        (output / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--deepspeed")
    parser.add_argument("--resume")
    args = parser.parse_args()
    destination = run(
        args.config, deepspeed_config=args.deepspeed, resume=args.resume
    )
    print(json.dumps({"status": "COMPLETE", "output": str(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
