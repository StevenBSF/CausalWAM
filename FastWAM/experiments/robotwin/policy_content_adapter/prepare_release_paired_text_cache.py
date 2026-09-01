#!/usr/bin/env python3
"""Prepare the three immutable Wan text embeddings used by Policy paired data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.robotwin.e0_e1.backbone import format_deployment_prompt
from experiments.robotwin.policy_content_adapter.model import artifact_identity
from experiments.robotwin.policy_content_adapter.native50hz_paired import (
    TASK_INSTRUCTIONS,
    atomic_write_json,
)
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS
from experiments.robotwin.policy_content_adapter.release_lineage import (
    verify_author_release_lineage,
)
from experiments.robotwin.policy_content_adapter.release_paired_binding import (
    verify_release_paired_binding,
)
from experiments.robotwin.policy_content_adapter.stage1_text_cache import (
    CACHE_SUFFIX,
    CONTEXT_DIM,
    CONTEXT_LEN,
    Wan22TextBatchEncoder,
    merge_shard_reports,
    prepare_prompt_shard,
    validate_cache_payload,
)


SCHEMA_VERSION = 1
AUDIT_KIND = "policy_release_paired_text_cache"
AUDIT_FILENAME = "release_paired_text_cache.audit.json"


class ReleasePairedTextCacheError(RuntimeError):
    """The paired prompt cache cannot prove the release-base contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleasePairedTextCacheError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_prompt_entries() -> list[dict[str, str]]:
    entries = []
    for task in OFFICIAL_TASKS:
        prompt = format_deployment_prompt(TASK_INSTRUCTIONS[task])
        entries.append(
            {
                "task": task,
                "prompt": prompt,
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return entries


def _payload_cache_audit(
    cache_dir: Path,
    entries: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    identities = []
    for entry in entries:
        path = cache_dir / f"{entry['sha256']}{CACHE_SUFFIX}"
        identities.append(validate_cache_payload(path))
    report = {
        "rank": 0,
        "world_size": 1,
        "assigned_count": len(entries),
        "created_count": 0,
        "skipped_valid_count": len(entries),
        "concurrent_valid_count": 0,
        "over_length_prompt_count": 0,
        "files": identities,
    }
    return merge_shard_reports(entries, [report], cache_dir=cache_dir)


BatchEncoder = Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor]]


def materialize_paired_prompt_payloads(
    cache_dir: str | Path,
    *,
    encode_batch: BatchEncoder,
) -> dict[str, Any]:
    """Create exactly three payloads; dependency injection keeps CPU tests tiny."""

    root = Path(cache_dir).expanduser().resolve()
    _require(
        not (root / AUDIT_FILENAME).exists(),
        "completed paired text cache already exists",
    )
    entries = paired_prompt_entries()
    report = prepare_prompt_shard(
        entries,
        cache_dir=root,
        rank=0,
        world_size=1,
        batch_size=len(entries),
        resume=False,
        audit_only=False,
        encode_batch=encode_batch,
    )
    _require(
        report["created_count"] == len(entries),
        "paired prompt payloads were not all created",
    )
    return merge_shard_reports(entries, [report], cache_dir=root)


def verify_release_paired_text_cache(
    cache_dir: str | Path,
    *,
    expected_base_lineage_sha256: str | None = None,
    expected_release_paired_binding_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    audit_path = root / AUDIT_FILENAME
    _require(root.is_dir(), f"paired text cache not found: {root}")
    _require(audit_path.is_file(), f"paired text cache audit not found: {audit_path}")
    try:
        value = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleasePairedTextCacheError(
            f"cannot read paired text cache audit: {exc}"
        ) from exc
    _require(isinstance(value, dict), "paired text cache audit must be an object")
    _require(value.get("status") == "PASS", "paired text cache audit is not PASS")
    _require(value.get("kind") == AUDIT_KIND, "paired text cache audit kind changed")
    _require(
        value.get("schema_version") == SCHEMA_VERSION,
        "paired text cache schema changed",
    )
    _require(
        value.get("context_len") == CONTEXT_LEN, "paired text context length changed"
    )
    _require(
        value.get("context_dim") == CONTEXT_DIM, "paired text context dimension changed"
    )
    _require(
        value.get("prompts") == paired_prompt_entries(),
        "paired text prompt inventory changed",
    )
    lineage = value.get("base_lineage_manifest")
    binding = value.get("release_paired_binding_manifest")
    _require(
        isinstance(lineage, Mapping), "paired text cache lacks base lineage identity"
    )
    _require(
        isinstance(binding, Mapping),
        "paired text cache lacks release/paired binding identity",
    )
    if expected_base_lineage_sha256 is not None:
        _require(
            lineage.get("sha256") == expected_base_lineage_sha256,
            "paired text cache base lineage differs",
        )
    if expected_release_paired_binding_sha256 is not None:
        _require(
            binding.get("sha256") == expected_release_paired_binding_sha256,
            "paired text cache release/paired binding differs",
        )
    actual_cache = _payload_cache_audit(root, paired_prompt_entries())
    declared_cache = value.get("cache")
    _require(
        isinstance(declared_cache, Mapping), "paired text cache payload audit missing"
    )
    for key in (
        "file_count",
        "total_size_bytes",
        "aggregate_payload_sha256",
        "all_payloads_valid",
        "extra_pt_files",
        "over_length_prompt_count",
    ):
        _require(
            actual_cache.get(key) == declared_cache.get(key),
            f"paired text cache {key} differs",
        )
    expected_names = {
        f"{entry['sha256']}{CACHE_SUFFIX}" for entry in paired_prompt_entries()
    }
    actual_names = {path.name for path in root.glob("*.pt") if path.is_file()}
    _require(
        actual_names == expected_names,
        "paired text cache contains missing/extra payloads",
    )
    return {
        **value,
        "audit_identity": artifact_identity(audit_path),
        "directory_identity": artifact_identity(root),
    }


def prepare_release_paired_text_cache(
    *,
    cache_dir: str | Path,
    base_lineage_manifest: str | Path,
    base_lineage_sha256: str,
    release_paired_binding: str | Path,
    release_paired_binding_sha256: str,
    checkpoint: str | Path,
    dataset_stats: str | Path,
    official_manifest: str | Path,
    model_base_path: str | Path,
    device: str,
) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    audit_path = root / AUDIT_FILENAME
    _require(
        not audit_path.exists(),
        f"refusing to overwrite paired text cache: {audit_path}",
    )
    lineage = verify_author_release_lineage(
        base_lineage_manifest,
        checkpoint_path=checkpoint,
        dataset_stats_path=dataset_stats,
        official_manifest_path=official_manifest,
        expected_manifest_sha256=base_lineage_sha256,
    )
    binding = verify_release_paired_binding(
        release_paired_binding,
        expected_sha256=release_paired_binding_sha256,
    )
    _require(
        binding["base_lineage"]["sha256"] == lineage["manifest_identity"]["sha256"],
        "release/paired binding names a different base lineage",
    )
    _require(
        torch.cuda.is_available(), "CUDA is required to encode paired text prompts"
    )
    encoder = Wan22TextBatchEncoder(model_base_path=model_base_path, device=device)
    cache_audit = materialize_paired_prompt_payloads(root, encode_batch=encoder)
    source = Path(__file__).resolve()
    result = {
        "status": "PASS",
        "kind": AUDIT_KIND,
        "schema_version": SCHEMA_VERSION,
        "base_lineage_manifest": dict(lineage["manifest_identity"]),
        "release_paired_binding_manifest": dict(binding["binding_manifest_identity"]),
        "prompts": paired_prompt_entries(),
        "context_len": CONTEXT_LEN,
        "context_dim": CONTEXT_DIM,
        "model": encoder.provenance(),
        "cache": cache_audit,
        "implementation": {"path": str(source), "sha256": _sha256(source)},
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(audit_path, result)
    return verify_release_paired_text_cache(
        root,
        expected_base_lineage_sha256=base_lineage_sha256,
        expected_release_paired_binding_sha256=release_paired_binding_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--base-lineage-manifest", required=True, type=Path)
    parser.add_argument("--base-lineage-sha256", required=True)
    parser.add_argument("--release-paired-binding", required=True, type=Path)
    parser.add_argument("--release-paired-binding-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-stats", required=True, type=Path)
    parser.add_argument("--official-manifest", required=True, type=Path)
    parser.add_argument("--model-base-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-release-paired-text-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _require(
            args.prepare_release_paired_text_cache,
            "refusing GPU encoding without --prepare-release-paired-text-cache",
        )
        result = prepare_release_paired_text_cache(
            cache_dir=args.cache_dir,
            base_lineage_manifest=args.base_lineage_manifest,
            base_lineage_sha256=args.base_lineage_sha256,
            release_paired_binding=args.release_paired_binding,
            release_paired_binding_sha256=args.release_paired_binding_sha256,
            checkpoint=args.checkpoint,
            dataset_stats=args.dataset_stats,
            official_manifest=args.official_manifest,
            model_base_path=args.model_base_path,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(
            f"Release paired text cache failed closed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_KIND",
    "ReleasePairedTextCacheError",
    "materialize_paired_prompt_payloads",
    "paired_prompt_entries",
    "prepare_release_paired_text_cache",
    "verify_release_paired_text_cache",
]
