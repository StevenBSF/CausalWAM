"""GPU extraction of frozen current-observation Motus WAN tokens."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .observation_content import extract_observation_visual_tokens
from .paired_data import (
    MotusPairedObservationDataset,
    sha256_file,
    validate_paired_observation_manifest,
)
from .runtime import instantiate_author_release, load_lineage
from .task_text_cache import load_task_embeddings, validate_task_text_cache
from .token_cache import FrozenTokenCacheWriter, validate_token_cache


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in batch], dim=0),
        "physical_state_ids": [item["physical_state_id"] for item in batch],
        "task_ids": [item["task"] for item in batch],
    }


def _validate_strict_load_audit(path: Path, lineage_identity: dict[str, Any]) -> None:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("schema") != "motus_robotwin2_strict_load_audit" or audit.get("status") != "PASS":
        raise RuntimeError("strict-load audit is not PASS")
    if audit.get("lineage_manifest", {}).get("sha256") != lineage_identity["sha256"]:
        raise RuntimeError("strict-load audit was produced for another lineage")
    contract = audit.get("load_contract", {})
    if contract.get("strict") is not True or contract.get("missing_keys") != 0 or contract.get("unexpected_keys") != 0:
        raise RuntimeError("strict-load audit contract changed")


def extract_cache(
    *,
    lineage_path: str | Path,
    strict_load_audit_path: str | Path,
    paired_manifest_path: str | Path,
    task_text_cache_dir: str | Path,
    output_dir: str | Path,
    local_cuda_index: int,
    groups_per_batch: int,
    capture_layer: int,
    heartbeat_groups: int,
) -> dict[str, Any]:
    lineage_path = Path(lineage_path).resolve()
    paired_manifest_path = Path(paired_manifest_path).resolve()
    strict_load_audit_path = Path(strict_load_audit_path).resolve()
    lineage_identity = _identity(lineage_path)
    paired_identity = _identity(paired_manifest_path)
    lineage = load_lineage(lineage_path, verify_files=True)
    _validate_strict_load_audit(strict_load_audit_path, lineage_identity)
    paired_manifest = json.loads(paired_manifest_path.read_text(encoding="utf-8"))
    validate_paired_observation_manifest(paired_manifest, verify_source_paths=True)
    validate_task_text_cache(task_text_cache_dir, verify_encoder_assets=True)
    task_embeddings = load_task_embeddings(task_text_cache_dir)
    if groups_per_batch <= 0 or heartbeat_groups <= 0:
        raise ValueError("batch and heartbeat counts must be positive")

    model = instantiate_author_release(
        lineage,
        batch_size=groups_per_batch * 4,
        local_cuda_index=local_cuda_index,
        strict=True,
    )
    model.eval()
    dataset = MotusPairedObservationDataset(
        paired_manifest_path, verify_source_paths=False
    )
    loader = DataLoader(
        dataset,
        batch_size=groups_per_batch,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    writer = FrozenTokenCacheWriter(
        output_dir,
        paired_manifest_identity=paired_identity,
        base_lineage_identity=lineage_identity,
        capture_layer=capture_layer,
    )
    completed = 0
    started = time.monotonic()
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch["images"]
                groups = images.shape[0]
                flattened = images.reshape(
                    groups * 4, *images.shape[2:]
                )
                language = [
                    task_embeddings[task]
                    for task in batch["task_ids"]
                    for _ in range(4)
                ]
                visual = extract_observation_visual_tokens(
                    model,
                    first_frame=flattened,
                    language_embeddings=language,
                    capture_layer=capture_layer,
                )
                visual = visual.reshape(
                    groups, 4, visual.shape[1], visual.shape[2]
                )
                writer.add(
                    visual,
                    batch["physical_state_ids"],
                    batch["task_ids"],
                )
                completed += groups
                if completed % heartbeat_groups == 0 or completed == len(dataset):
                    elapsed = time.monotonic() - started
                    print(
                        json.dumps(
                            {
                                "event": "heartbeat",
                                "completed_groups": completed,
                                "total_groups": len(dataset),
                                "elapsed_seconds": round(elapsed, 3),
                                "groups_per_second": round(completed / elapsed, 6),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        writer.finalize()
    except Exception:
        writer.abort()
        raise
    result = validate_token_cache(
        output_dir,
        expected_paired_manifest_sha256=paired_identity["sha256"],
        expected_base_lineage_sha256=lineage_identity["sha256"],
        verify_shards=True,
    )
    result["output"] = str(Path(output_dir).resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--strict-load-audit", required=True)
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--task-text-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--capture-layer", type=int, default=16)
    parser.add_argument("--heartbeat-groups", type=int, default=20)
    args = parser.parse_args()
    result = extract_cache(
        lineage_path=args.lineage,
        strict_load_audit_path=args.strict_load_audit,
        paired_manifest_path=args.paired_manifest,
        task_text_cache_dir=args.task_text_cache,
        output_dir=args.output,
        local_cuda_index=args.local_cuda_index,
        groups_per_batch=args.groups_per_batch,
        capture_layer=args.capture_layer,
        heartbeat_groups=args.heartbeat_groups,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

