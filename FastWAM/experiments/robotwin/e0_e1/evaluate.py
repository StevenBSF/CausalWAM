#!/usr/bin/env python3
"""Evaluate raw backbone pooling or a content head on a val/test token cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .cache import load_cache, representation_records
from .head import ContrastiveContentHead
from .io_utils import file_identity, module_state_sha256, write_csv, write_json
from .metrics import compute_representation_metrics
from .negatives import select_state_negative_pairs


EXPERIMENTS = ("E0-RawBackbone", "E1-InitHead", "E1-TrainedHead")
EVALUATION_SPLITS = ("val", "test")


def _head_from_checkpoint(
    checkpoint_path: Path,
    *,
    expected_layer: int,
    expected_backbone_dim: int,
    device: torch.device,
) -> tuple[ContrastiveContentHead, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported head checkpoint: {checkpoint_path}")
    config = payload.get("head_config")
    if not isinstance(config, dict):
        raise ValueError("head checkpoint has no head_config")
    if int(payload.get("layer")) != expected_layer:
        raise ValueError("head checkpoint layer does not match the requested cache layer")
    if int(config.get("backbone_dim")) != expected_backbone_dim:
        raise ValueError("head checkpoint backbone dimension mismatch")
    head = ContrastiveContentHead(**config)
    incompatible = head.load_state_dict(payload["head"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict head load failed: {incompatible}")
    return head.to(device).eval(), payload


def evaluate_cache(
    *,
    cache_path: str | Path,
    layer: int,
    experiment: str,
    output_dir: str | Path,
    head_checkpoint: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
    min_temporal_gap: int = 8,
    min_state_distance: float = 1e-5,
    inference_batch_size: int = 64,
) -> list[dict[str, Any]]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    payload = load_cache(cache_path)
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    cache_splits = {str(record["split"]) for record in payload["records"]}
    if len(cache_splits) != 1 or not cache_splits <= set(EVALUATION_SPLITS):
        raise ValueError(
            "representation evaluation requires one validation or held-out test "
            f"cache, got {cache_splits}"
        )
    evaluation_split = next(iter(cache_splits))
    layer_key = str(int(layer))
    if layer_key not in payload["tokens_by_layer"]:
        raise KeyError(f"cache does not contain layer {layer}; choices={sorted(payload['tokens_by_layer'])}")
    records = representation_records(payload)
    negative_pairs = select_state_negative_pairs(
        payload["records"],
        payload["physical_states"],
        min_temporal_gap=min_temporal_gap,
        min_state_distance=min_state_distance,
    )
    execution_device = torch.device(device)
    head_info: dict[str, Any] | None = None
    if experiment == "E0-RawBackbone":
        if head_checkpoint is not None:
            raise ValueError("E0-RawBackbone does not accept --head-checkpoint")
        embeddings = payload["pooled_by_layer"][layer_key].float()
    else:
        tokens = payload["tokens_by_layer"][layer_key]
        backbone_dim = int(tokens.shape[-1])
        if experiment == "E1-InitHead":
            if head_checkpoint is not None:
                raise ValueError("E1-InitHead must be randomly initialized, not loaded")
            torch.manual_seed(seed)
            head = ContrastiveContentHead(backbone_dim=backbone_dim).to(execution_device).eval()
            head_info = {
                "initialization_seed": int(seed),
                "initial_head_sha256": module_state_sha256(head),
                "trainable_parameter_count": head.trainable_parameter_count(),
            }
        else:
            if head_checkpoint is None:
                raise ValueError("E1-TrainedHead requires --head-checkpoint")
            head, checkpoint_payload = _head_from_checkpoint(
                Path(head_checkpoint).expanduser().resolve(),
                expected_layer=int(layer),
                expected_backbone_dim=backbone_dim,
                device=execution_device,
            )
            head_info = {
                "checkpoint": str(Path(head_checkpoint).expanduser().resolve()),
                "checkpoint_step": int(checkpoint_payload["step"]),
                "training_seed": int(checkpoint_payload["seed"]),
                "initial_head_sha256": str(
                    checkpoint_payload["initial_head_sha256"]
                ),
                "trainable_parameter_count": head.trainable_parameter_count(),
            }
        with torch.inference_mode():
            embedding_batches = [
                head(
                    tokens[start : start + inference_batch_size].to(
                        device=execution_device, dtype=torch.float32
                    )
                ).cpu()
                for start in range(0, len(tokens), inference_batch_size)
            ]
            embeddings = torch.cat(embedding_batches, dim=0)

    rows = compute_representation_metrics(
        embeddings,
        records,
        layer=f"video_block_{int(layer):02d}",
        experiment=experiment,
        state_negative_pairs=negative_pairs,
        min_temporal_gap=min_temporal_gap,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    slug = experiment.lower().replace("-", "_")
    result = {
        "cache": str(Path(cache_path).expanduser().resolve()),
        "cache_identity": file_identity(cache_path),
        "cache_provenance": payload["provenance"],
        "evaluation_split": evaluation_split,
        "layer": int(layer),
        "experiment": experiment,
        "head": head_info,
        "negative_filter": {
            "min_temporal_gap": int(min_temporal_gap),
            "min_state_distance": float(min_state_distance),
            "num_pairs": len(negative_pairs),
        },
        "metrics": rows,
    }
    output_stem = f"{slug}_layer_{int(layer):02d}"
    write_json(destination / f"{output_stem}.json", result)
    write_csv(destination / f"{output_stem}.csv", rows)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-temporal-gap", type=int, default=8)
    parser.add_argument("--min-state-distance", type=float, default=1e-5)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = _parser().parse_args()
    rows = evaluate_cache(
        cache_path=args.cache,
        layer=args.layer,
        experiment=args.experiment,
        output_dir=args.output_dir,
        head_checkpoint=args.head_checkpoint,
        seed=args.seed,
        device=args.device,
        min_temporal_gap=args.min_temporal_gap,
        min_state_distance=args.min_state_distance,
        inference_batch_size=args.inference_batch_size,
    )
    for row in rows:
        print(
            f"{row['task']:>22}  style={row['style_distance']:.6f}  "
            f"state={row['state_distance']:.6f}  ratio={row['state_style_ratio']:.3f}  "
            f"R@1={row['retrieval_r1']:.3f}"
        )


if __name__ == "__main__":
    main()
