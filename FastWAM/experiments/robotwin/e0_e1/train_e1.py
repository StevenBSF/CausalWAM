#!/usr/bin/env python3
"""Train only ContrastiveContentHead from frozen, cached FastWAM tokens."""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .cache import load_cache
from .data import VARIANTS
from .head import ContrastiveContentHead, multi_positive_supcon_loss
from .io_utils import (
    atomic_torch_save,
    file_identity,
    module_state_sha256,
    parameter_checksum,
    write_csv,
    write_json,
)
from .negatives import build_state_negative_mask


def _backbone_semantics(provenance: Any) -> dict[str, Any]:
    """Return extraction semantics, excluding audit-only verification status."""

    if not isinstance(provenance, dict):
        raise ValueError("cache provenance backbone must be a dictionary")
    semantics = dict(provenance)
    # Running the native-prefill equivalence assertion does not alter the
    # captured representation.  It is intentionally enabled only once in the
    # smoke recipe, so treat its outcome as audit metadata rather than a model
    # or preprocessing compatibility field.
    semantics.pop("native_prefill_verified", None)
    return semantics


def _groups_by_task(records: list[dict[str, Any]]) -> dict[str, list[list[int]]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(str(record["task"]), str(record["physical_state_id"]))].append(index)
    result: dict[str, list[list[int]]] = defaultdict(list)
    for (task, _), indices in grouped.items():
        if len(indices) != 4:
            raise ValueError(f"state group must contain four renderings, got {indices}")
        result[task].append(sorted(indices))
    if any(len(groups) < 2 for groups in result.values()):
        raise ValueError("every task needs at least two physical-state groups")
    return dict(result)


def _batch_indices(
    groups_by_task: dict[str, list[list[int]]],
    *,
    groups_per_batch: int,
    generator: random.Random,
) -> list[int]:
    task = generator.choice(sorted(groups_by_task))
    groups = groups_by_task[task]
    if groups_per_batch > len(groups):
        selected = [generator.choice(groups) for _ in range(groups_per_batch)]
    else:
        selected = generator.sample(groups, groups_per_batch)
    indices = [index for group in selected for index in group]
    if len(set(indices)) != len(indices):
        raise ValueError("batch repeated a physical state; lower --groups-per-batch")
    return indices


def _physical_state_negative_mask(
    cache: dict[str, Any],
    indices: list[int],
    *,
    min_temporal_gap: int,
    min_state_distance: float,
) -> torch.Tensor:
    """Build a representation-level mask from group-level cached states."""

    records = [cache["records"][index] for index in indices]
    group_state_by_id: dict[tuple[str, str], dict[str, float]] = {}
    if len(cache["records"]) != len(cache["physical_states"]) * 4:
        raise ValueError("cache records/physical_states do not form four-render groups")
    for group_index in range(len(cache["physical_states"])):
        start = group_index * len(VARIANTS)
        group = cache["records"][start : start + len(VARIANTS)]
        if tuple(str(record["variant"]) for record in group) != VARIANTS:
            raise ValueError(f"cache group {group_index} has non-canonical variant order")
        keys = {
            (str(record["task"]), str(record["physical_state_id"]))
            for record in group
        }
        if len(keys) != 1:
            raise ValueError(f"cache group {group_index} is not one physical state")
        key = next(iter(keys))
        if key in group_state_by_id:
            raise ValueError(f"duplicate physical-state group {key}")
        group_state_by_id[key] = cache["physical_states"][group_index]
    states = [
        group_state_by_id[(str(record["task"]), str(record["physical_state_id"]))]
        for record in records
    ]
    return torch.from_numpy(
        build_state_negative_mask(
            records,
            states,
            min_temporal_gap=min_temporal_gap,
            min_state_distance=min_state_distance,
        )
    )


@torch.no_grad()
def _similarities(
    embeddings: torch.Tensor,
    labels: list[str],
    *,
    negative_mask: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    normalized = F.normalize(embeddings.float(), dim=-1)
    similarity = normalized @ normalized.T
    count = len(labels)
    not_self = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    equal = torch.tensor(
        [[left == right for right in labels] for left in labels],
        dtype=torch.bool,
        device=similarity.device,
    )
    positive = similarity[equal & not_self]
    if negative_mask is None:
        selected_negatives = (~equal) & not_self
    else:
        if (
            negative_mask.shape != similarity.shape
            or negative_mask.dtype is not torch.bool
        ):
            raise ValueError("diagnostic negative mask must be bool [N,N]")
        selected_negatives = negative_mask.to(device=similarity.device)
        if bool((selected_negatives & (equal | ~not_self)).any().item()):
            raise ValueError("diagnostic negative mask contains a non-negative pair")
    negative = similarity[selected_negatives]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("diagnostic batch lacks positive or negative pairs")
    positive_mean = float(positive.mean().item())
    negative_mean = float(negative.mean().item())
    embedding_norm = float(embeddings.float().norm(dim=-1).mean().item())
    # Exact collapse makes every different-state cosine one.  Logging and
    # enforcing this spread turns collapse detection into an executable gate,
    # rather than relying on a later visual inspection of the loss curve.
    state_spread = 1.0 - negative_mean
    if not all(
        math.isfinite(value)
        for value in (positive_mean, negative_mean, embedding_norm, state_spread)
    ):
        raise FloatingPointError("non-finite representation diagnostics")
    if state_spread <= 1e-7:
        raise FloatingPointError(
            f"representation collapse detected: state spread={state_spread}"
        )
    return positive_mean, negative_mean, embedding_norm, state_spread


def train_head(
    *,
    train_cache_path: str | Path,
    val_cache_path: str | Path,
    layer: int,
    output_dir: str | Path,
    steps: int = 1000,
    groups_per_batch: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    temperature: float = 0.07,
    val_every: int = 50,
    seed: int = 0,
    device: str = "cuda",
    min_temporal_gap: int = 8,
    min_state_distance: float = 1e-5,
) -> Path:
    if steps <= 0 or groups_per_batch < 2 or val_every <= 0:
        raise ValueError("steps/val_every must be positive and groups_per_batch >= 2")
    if min_temporal_gap < 0 or min_state_distance < 0:
        raise ValueError("negative thresholds must be non-negative")
    torch.manual_seed(seed)
    generator = random.Random(seed)
    train_cache_identity = file_identity(train_cache_path)
    val_cache_identity = file_identity(val_cache_path)
    train_cache = load_cache(train_cache_path)
    val_cache = load_cache(val_cache_path)
    train_splits = {str(record["split"]) for record in train_cache["records"]}
    val_splits = {str(record["split"]) for record in val_cache["records"]}
    if train_splits != {"train"} or val_splits != {"val"}:
        raise ValueError(
            f"training requires train/val caches, got {train_splits}/{val_splits}"
        )
    train_tasks = {str(record["task"]) for record in train_cache["records"]}
    val_tasks = {str(record["task"]) for record in val_cache["records"]}
    if train_tasks != val_tasks:
        raise ValueError(
            f"train/validation task sets differ: {sorted(train_tasks)}/{sorted(val_tasks)}"
        )
    train_provenance = train_cache.get("provenance")
    val_provenance = val_cache.get("provenance")
    if not isinstance(train_provenance, dict) or not isinstance(
        val_provenance, dict
    ):
        raise ValueError("train/validation caches are missing provenance")
    for provenance_key in ("backbone", "task_prompt_sha256"):
        if provenance_key not in train_provenance or provenance_key not in val_provenance:
            raise ValueError(
                f"train/validation cache provenance is missing {provenance_key}"
            )
        train_value = train_provenance[provenance_key]
        val_value = val_provenance[provenance_key]
        if provenance_key == "backbone":
            train_value = _backbone_semantics(train_value)
            val_value = _backbone_semantics(val_value)
        if train_value != val_value:
            raise ValueError(
                f"train/validation cache {provenance_key} provenance differs"
            )
    train_trajectories = {
        (str(record["task"]), str(record["trajectory_id"]))
        for record in train_cache["records"]
    }
    val_trajectories = {
        (str(record["task"]), str(record["trajectory_id"]))
        for record in val_cache["records"]
    }
    leakage = train_trajectories & val_trajectories
    if leakage:
        raise ValueError(
            f"train/validation physical-trajectory leakage: {sorted(leakage)[:5]}"
        )
    layer_key = str(int(layer))
    if (
        layer_key not in train_cache["tokens_by_layer"]
        or layer_key not in val_cache["tokens_by_layer"]
    ):
        raise KeyError(f"layer {layer} is absent from train or validation cache")
    train_tokens = train_cache["tokens_by_layer"][layer_key]
    val_tokens = val_cache["tokens_by_layer"][layer_key]
    if train_tokens.shape[1:] != val_tokens.shape[1:]:
        raise ValueError("train/validation token shapes differ")
    execution_device = torch.device(device)
    head_config = {
        "backbone_dim": int(train_tokens.shape[-1]),
        "embed_dim": 384,
        "num_queries": 8,
        "num_heads": 8,
    }
    head = ContrastiveContentHead(**head_config).to(execution_device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_groups = _groups_by_task(train_cache["records"])
    val_groups = _groups_by_task(val_cache["records"])
    if any(groups_per_batch > len(groups) for groups in train_groups.values()):
        raise ValueError("--groups-per-batch exceeds a task's unique train states")
    if min(len(groups) for groups in val_groups.values()) < 2:
        raise ValueError("validation cache cannot form contrastive negatives")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []
    initial_checksum = parameter_checksum(head)
    initial_head_sha256 = module_state_sha256(head)

    @torch.no_grad()
    def validate_all_tasks() -> dict[str, float]:
        """Deterministically cover every validation physical state."""

        head.eval()
        task_losses: list[float] = []
        task_positives: list[float] = []
        task_negatives: list[float] = []
        task_norms: list[float] = []
        task_spreads: list[float] = []
        for task in sorted(val_groups):
            task_indices = [index for group in val_groups[task] for index in group]
            state_ids = [
                str(val_cache["records"][index]["physical_state_id"])
                for index in task_indices
            ]
            task_ids = [task] * len(task_indices)
            negative_mask = _physical_state_negative_mask(
                val_cache,
                task_indices,
                min_temporal_gap=min_temporal_gap,
                min_state_distance=min_state_distance,
            ).to(device=execution_device)
            embedding_chunks = [
                head(
                    val_tokens.index_select(
                        0, torch.tensor(chunk, dtype=torch.long)
                    ).to(device=execution_device, dtype=torch.float32)
                )
                for chunk in (
                    task_indices[start : start + groups_per_batch * 4]
                    for start in range(0, len(task_indices), groups_per_batch * 4)
                )
            ]
            task_embeddings = torch.cat(embedding_chunks, dim=0)
            task_loss = multi_positive_supcon_loss(
                task_embeddings,
                state_ids,
                task_ids,
                temperature=temperature,
                negative_mask=negative_mask,
            )
            task_losses.append(float(task_loss.item()))
            positive, negative, norm, spread = _similarities(
                task_embeddings, state_ids, negative_mask=negative_mask
            )
            task_positives.append(positive)
            task_negatives.append(negative)
            task_norms.append(norm)
            task_spreads.append(spread)
        return {
            "loss": sum(task_losses) / len(task_losses),
            "positive": sum(task_positives) / len(task_positives),
            "negative": sum(task_negatives) / len(task_negatives),
            "norm": sum(task_norms) / len(task_norms),
            # Worst task is the collapse gate; cross-task discrimination must
            # never hide within-task collapse.
            "spread": min(task_spreads),
        }

    for step in range(1, steps + 1):
        indices = _batch_indices(
            train_groups, groups_per_batch=groups_per_batch, generator=generator
        )
        index_tensor = torch.tensor(indices, dtype=torch.long)
        tokens = train_tokens.index_select(0, index_tensor).to(
            device=execution_device, dtype=torch.float32
        )
        state_ids = [
            str(train_cache["records"][index]["physical_state_id"])
            for index in indices
        ]
        task_ids = [str(train_cache["records"][index]["task"]) for index in indices]
        negative_mask = _physical_state_negative_mask(
            train_cache,
            indices,
            min_temporal_gap=min_temporal_gap,
            min_state_distance=min_state_distance,
        ).to(device=execution_device)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = head(tokens)
        loss = multi_positive_supcon_loss(
            embeddings,
            state_ids,
            task_ids,
            temperature=temperature,
            negative_mask=negative_mask,
        )
        loss.backward()
        gradients = [
            parameter.grad.detach().float().norm().square()
            for parameter in head.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("head received no gradients")
        grad_norm = float(torch.stack(gradients).sum().sqrt().item())
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise FloatingPointError(f"invalid head gradient norm {grad_norm}")
        optimizer.step()
        head.eval()
        with torch.no_grad():
            updated_embeddings = head(tokens)
        positive, negative, embedding_norm, state_spread = _similarities(
            updated_embeddings, state_ids, negative_mask=negative_mask
        )
        row: dict[str, Any] = {
            "step": step,
            "train_contrastive_loss": float(loss.detach().item()),
            "train_positive_similarity": positive,
            "train_negative_similarity": negative,
            "embedding_norm": embedding_norm,
            "embedding_state_spread": state_spread,
            "gradient_norm": grad_norm,
            "val_contrastive_loss": None,
            "val_positive_similarity": None,
            "val_negative_similarity": None,
            "val_embedding_state_spread": None,
        }

        if step == 1 or step % val_every == 0 or step == steps:
            val_diagnostics = validate_all_tasks()
            row.update(
                {
                    "val_contrastive_loss": val_diagnostics["loss"],
                    "val_positive_similarity": val_diagnostics["positive"],
                    "val_negative_similarity": val_diagnostics["negative"],
                    "val_embedding_state_spread": val_diagnostics["spread"],
                }
            )
            print(
                f"step={step}/{steps} "
                f"train_loss={row['train_contrastive_loss']:.6f} "
                f"train_pos={row['train_positive_similarity']:.6f} "
                f"train_neg={row['train_negative_similarity']:.6f} "
                f"grad={row['gradient_norm']:.6f} "
                f"val_loss={row['val_contrastive_loss']:.6f} "
                f"val_spread={row['val_embedding_state_spread']:.6f}",
                flush=True,
            )
        log_rows.append(row)

    final_checksum = parameter_checksum(head)
    if initial_checksum == final_checksum:
        raise RuntimeError("optimizer completed without changing the content head")
    if file_identity(train_cache_path) != train_cache_identity:
        raise RuntimeError("train cache changed while the content head was training")
    if file_identity(val_cache_path) != val_cache_identity:
        raise RuntimeError("validation cache changed while the content head was training")
    checkpoint = {
        "schema_version": 1,
        "step": int(steps),
        "layer": int(layer),
        "head_config": head_config,
        "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "trainable_parameter_count": head.trainable_parameter_count(),
        "temperature": float(temperature),
        "negative_filter": {
            "min_temporal_gap": int(min_temporal_gap),
            "min_state_distance": float(min_state_distance),
        },
        "seed": int(seed),
        "initial_head_sha256": initial_head_sha256,
        "train_cache": str(Path(train_cache_path).expanduser().resolve()),
        "val_cache": str(Path(val_cache_path).expanduser().resolve()),
        "train_cache_identity": train_cache_identity,
        "val_cache_identity": val_cache_identity,
    }
    checkpoint_path = atomic_torch_save(destination / "e1_content_head.pt", checkpoint)
    write_json(destination / "train_log.json", log_rows)
    write_csv(destination / "train_log.csv", log_rows)
    return checkpoint_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-temporal-gap", type=int, default=8)
    parser.add_argument("--min-state-distance", type=float, default=1e-5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint = train_head(
        train_cache_path=args.train_cache,
        val_cache_path=args.val_cache,
        layer=args.layer,
        output_dir=args.output_dir,
        steps=args.steps,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        val_every=args.val_every,
        seed=args.seed,
        device=args.device,
        min_temporal_gap=args.min_temporal_gap,
        min_state_distance=args.min_state_distance,
    )
    print(f"saved trained head: {checkpoint}")


if __name__ == "__main__":
    main()
