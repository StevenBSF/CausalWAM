#!/usr/bin/env python3
"""Strictly index paired data and cache frozen FastWAM video tokens."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from .backbone import (
    DEFAULT_LAYERS,
    FrozenFastWAMExtractor,
    format_deployment_prompt,
)
from .cache import build_cache_payload, save_cache
from .data import (
    E0_E1_PROTOCOL,
    R3_HOLDOUT_PROTOCOL,
    R3_VARIANT,
    PairedFrameDataset,
    TASKS,
)
from .prompts import TASK_INSTRUCTIONS


DEFAULT_DATA_ROOT = "third_party/RoboTwin/data"
DEFAULT_CHECKPOINT = (
    "/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/"
    "robotwin_uncond_3cam_384.pt"
)
DEFAULT_STATS = (
    "/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints/fastwam_release/"
    "robotwin_uncond_3cam_384_dataset_stats.json"
)
DEFAULT_MODEL_BASE = "/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints"


def extract_cache(
    *,
    data_root: str | Path,
    tasks: tuple[str, ...],
    split: str,
    states_per_trajectory: int,
    checkpoint: str | Path,
    dataset_stats: str | Path,
    model_base_path: str | Path,
    output_path: str | Path,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    device: str = "cuda",
    allow_incomplete: bool = False,
    max_trajectories_per_task: int | None = None,
    content_ids: tuple[int, ...] | None = None,
    verify_native_prefill: bool = False,
    protocol: str = E0_E1_PROTOCOL,
    proprio_mode: str = "observed",
    decision_lock: str | Path | None = None,
) -> Path:
    protocol = str(protocol)
    proprio_mode = str(proprio_mode)
    destination = Path(output_path).expanduser().resolve()
    lock_identity: dict[str, Any] | None = None
    if protocol == R3_HOLDOUT_PROTOCOL:
        if proprio_mode not in ("observed", "constant_zero_normalized"):
            raise ValueError("R3 holdout extraction requires an explicit proprio mode")
        if split == "test":
            if decision_lock is None:
                raise ValueError("R3 test extraction requires a pre-existing decision lock")
            from .decision_lock_e2e3 import load_decision_lock

            experiment = "E2" if proprio_mode == "observed" else "E3"
            lock_payload, lock_identity = load_decision_lock(
                decision_lock,
                experiment=experiment,
                expected_test_output=destination,
            )
            selected_layer = int(lock_payload["selected_layer"])
            if tuple(int(layer) for layer in layers) != (selected_layer,):
                raise ValueError(
                    "R3 test extraction layers must equal the decision-locked "
                    f"selected layer {(selected_layer,)!r}"
                )
        elif decision_lock is not None:
            raise ValueError("decision lock is accepted only for R3 test extraction")
    else:
        if proprio_mode != "observed":
            raise ValueError("E0/E1 extraction supports only observed proprio")
        if decision_lock is not None:
            raise ValueError("E0/E1 extraction does not accept a decision lock")
    dataset = PairedFrameDataset(
        data_root,
        tasks=tasks,
        split=split,
        states_per_trajectory=states_per_trajectory,
        allow_incomplete=allow_incomplete,
        max_trajectories_per_task=max_trajectories_per_task,
        content_ids=content_ids,
        protocol=protocol,
    )
    manifest_dir = destination.parent / "manifests"
    # Every cache needs immutable, cache-specific source manifests.  A shared
    # stem would let a later E3/test extraction overwrite E2/train provenance.
    manifest_paths = dataset.write_manifests(manifest_dir, stem=destination.stem)
    manifest_jsonl_sha256 = hashlib.sha256(
        manifest_paths["jsonl"].read_bytes()
    ).hexdigest()
    extractor = FrozenFastWAMExtractor.from_release_checkpoint(
        checkpoint,
        dataset_stats,
        model_base_path=model_base_path,
        device=device,
        capture_layers=layers,
        verify_native_prefill=verify_native_prefill,
        compute_checkpoint_sha256=True,
    )
    task_contexts = {
        task: extractor.encode_instruction(TASK_INSTRUCTIONS[task]) for task in tasks
    }
    task_prompts = {
        task: format_deployment_prompt(TASK_INSTRUCTIONS[task]) for task in tasks
    }
    layer_batches: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    samples: list[dict[str, Any]] = []
    backbone_provenance: dict[str, Any] | None = None
    condition_provenance: dict[str, dict[str, Any]] = {}
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        task = str(sample["task"])
        context, context_mask = task_contexts[task]
        # Active renderings share exactly one current physical state.  Feeding
        # them in one extraction keeps text/proprio condition identical.  The
        # task-level prompt context is encoded once and reused without changing
        # deployment semantics; current proprio is still appended per sample.
        output = extractor.extract_current_observations(
            sample["images"],
            sample["proprio_raw"],
            context=context,
            context_mask=context_mask,
            proprio_mode=proprio_mode,
        )
        for layer in layers:
            layer_batches[layer].append(output.tokens_by_layer[layer])
        current_provenance = dict(output.provenance)
        current_condition = dict(current_provenance.pop("condition"))
        current_normalized_proprio = str(
            current_provenance.pop("normalized_proprio_sha256")
        )
        current_image_source_range = list(
            current_provenance.pop("image_source_range")
        )
        current_image_input_range = list(
            current_provenance.pop("image_input_range")
        )
        if backbone_provenance is None:
            backbone_provenance = current_provenance
        elif current_provenance != backbone_provenance:
            raise RuntimeError(
                "frozen backbone provenance changed between aligned samples"
            )
        physical_key = str(sample["physical_key"])
        condition_provenance[physical_key] = {
            "task": task,
            "context": current_condition,
            "normalized_proprio_sha256": current_normalized_proprio,
            "image_source_range": current_image_source_range,
            "image_input_range": current_image_input_range,
            "visual_input_sha256": sample["visual_input_sha256"],
        }
        sample_for_cache = dict(sample)
        sample_for_cache.pop("images")
        samples.append(sample_for_cache)
        print(
            f"[{sample_index + 1}/{len(dataset)}] {sample['physical_key']} "
            f"tokens={tuple(output.tokens_by_layer[layers[0]].shape)}",
            flush=True,
        )
    tokens_by_layer = {
        layer: torch.cat(batches, dim=0) for layer, batches in layer_batches.items()
    }
    if backbone_provenance is None:
        raise RuntimeError("validated dataset unexpectedly produced no samples")
    payload = build_cache_payload(
        tokens_by_layer=tokens_by_layer,
        samples=samples,
        provenance={
            "protocol": protocol,
            "split": split,
            "active_variants": list(dataset.active_variants),
            "holdout_variant": (
                R3_VARIANT if protocol == R3_HOLDOUT_PROTOCOL else None
            ),
            "proprio_mode": proprio_mode,
            "decision_lock_identity": lock_identity,
            "decision_lock_created_before_test": (
                True if protocol == R3_HOLDOUT_PROTOCOL and split == "test" else None
            ),
            "tasks": list(tasks),
            "states_per_trajectory": int(states_per_trajectory),
            "allow_incomplete": bool(allow_incomplete),
            "max_trajectories_per_task": max_trajectories_per_task,
            "content_ids": None if content_ids is None else list(content_ids),
            "manifest_jsonl": str(manifest_paths["jsonl"]),
            "manifest_csv": str(manifest_paths["csv"]),
            # This canonical source-row digest is deliberately independent of
            # the E2/E3 token values.  The decision lock later requires the
            # E2/E3 train and validation digests to match, proving both
            # conditions were extracted from the exact same image/frame rows.
            "source_manifest_sha256": manifest_jsonl_sha256,
            "task_prompts": task_prompts,
            "task_prompt_sha256": {
                task: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for task, prompt in task_prompts.items()
            },
            "backbone": backbone_provenance,
            "conditions_by_physical_state": condition_provenance,
        },
    )
    return save_cache(destination, payload)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated layer indices"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated layer indices")
    return result


def _csv_content_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated content IDs"
        ) from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated non-negative content IDs"
        )
    return result


def _csv_tasks(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or not set(result) <= set(TASKS):
        raise argparse.ArgumentTypeError(f"tasks must be a subset of {TASKS}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tasks", type=_csv_tasks, default=TASKS)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--states-per-trajectory", type=int, default=8)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--model-base-path", default=DEFAULT_MODEL_BASE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=_csv_ints, default=DEFAULT_LAYERS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="smoke only: accept a contiguous published prefix",
    )
    parser.add_argument("--max-trajectories-per-task", type=int)
    parser.add_argument(
        "--content-ids",
        type=_csv_content_ids,
        help="smoke only: exact published physical trajectory IDs",
    )
    parser.add_argument("--verify-native-prefill", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=(E0_E1_PROTOCOL, R3_HOLDOUT_PROTOCOL),
        default=E0_E1_PROTOCOL,
    )
    parser.add_argument(
        "--proprio-mode",
        choices=("observed", "constant_zero_normalized"),
        default="observed",
    )
    parser.add_argument("--decision-lock")
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = extract_cache(
        data_root=args.data_root,
        tasks=args.tasks,
        split=args.split,
        states_per_trajectory=args.states_per_trajectory,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        model_base_path=args.model_base_path,
        output_path=args.output,
        layers=args.layers,
        device=args.device,
        allow_incomplete=args.allow_incomplete,
        max_trajectories_per_task=args.max_trajectories_per_task,
        content_ids=args.content_ids,
        verify_native_prefill=args.verify_native_prefill,
        protocol=args.protocol,
        proprio_mode=args.proprio_mode,
        decision_lock=args.decision_lock,
    )
    print(f"saved frozen token cache: {path}")


if __name__ == "__main__":
    main()
