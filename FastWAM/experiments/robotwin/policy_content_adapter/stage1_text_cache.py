#!/usr/bin/env python3
"""Prepare the exact Stage-1 full550 RoboTwin text-embedding cache.

This entry point is deliberately separate from model training.  It derives
the prompt inventory from the ``task_index`` values that actually occur in
the 1,650 hash-bound Parquet episodes selected by Policy Protocol v2, maps
those indices through the release ``tasks.jsonl``, and encodes only that
inventory with the original FastWAM Wan2.2 text encoder.

The command supports deterministic rank striding under ``torchrun``, atomic
create-only payload writes, fail-closed resume, inventory-only preparation,
and an audit-only pass that never loads the text encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

from experiments.robotwin.policy_content_adapter.native50hz_paired import atomic_write_json
from experiments.robotwin.policy_content_adapter.official_data import (
    OFFICIAL_DOMAINS,
    OFFICIAL_TASKS,
    select_official_full_550_per_task,
    verify_official_task_manifest,
)
from experiments.robotwin.policy_content_adapter.runtime_utils import (
    DEFAULT_OFFICIAL_MANIFEST,
    PROJECT_ROOT,
    temporary_environment,
)


DEFAULT_DATASET_ROOT = Path(
    "/mnt/cpfs-E/baoshifeng/FastWAM/data/robotwin2.0/robotwin2.0"
)
DEFAULT_MODEL_BASE_PATH = Path("/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints")
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/stage1_artifacts/full550_three_task_text_cache"
)
DEFAULT_STAGE1_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "stage1_clean_random_base.yaml"
)

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
ENCODER_ID = "wan22ti2v5b"
CONTEXT_LEN = 128
CONTEXT_DIM = 4096
DEFAULT_BATCH_SIZE = 16
CACHE_SUFFIX = f".t5_len{CONTEXT_LEN}.{ENCODER_ID}.pt"

# These counts were independently audited from the task_index column of the
# exact hash-bound Parquet files.  Compiling them into the preparation gate
# prevents an episodes.jsonl paraphrase list from being mistaken for the set
# of prompts that training can actually request.
EXPECTED_SELECTED_EPISODES = 1_650
EXPECTED_SELECTED_FRAMES = 466_240
EXPECTED_UNIQUE_TASK_INDICES = 68_704
EXPECTED_UNIQUE_TASK_INDICES_BY_TASK: Mapping[str, int] = {
    "place_a2b_left": 42_467,
    "open_microwave": 828,
    "move_stapler_pad": 25_409,
}

INVENTORY_KIND = "policy_stage1_full550_text_prompt_inventory"
AUDIT_KIND = "policy_stage1_full550_text_cache"
SCHEMA_VERSION = 1


class Stage1TextCacheError(RuntimeError):
    """The subset cache cannot prove the approved Stage-1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1TextCacheError(message)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(path_value: str | Path | Sequence[str | Path]) -> dict[str, Any]:
    """Content-address one model file, file list, or tokenizer directory."""

    raw_paths = (
        list(path_value)
        if isinstance(path_value, Sequence) and not isinstance(path_value, (str, bytes, Path))
        else [path_value]
    )
    files: list[Path] = []
    roots: list[str] = []
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        _require(path.exists(), f"model artifact is missing: {path}")
        roots.append(str(path))
        if path.is_file():
            files.append(path)
        else:
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
    _require(bool(files), f"model artifact has no files: {roots}")
    digest = hashlib.sha256()
    identities = []
    for path in sorted(set(files), key=lambda item: str(item)):
        file_sha = _sha256_file(path)
        size = path.stat().st_size
        identities.append({"path": str(path), "size_bytes": size, "sha256": file_sha})
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "roots": roots,
        "file_count": len(identities),
        "total_size_bytes": sum(item["size_bytes"] for item in identities),
        "aggregate_sha256": digest.hexdigest(),
        "files": identities,
    }


def _canonical_prompt_set_sha(entries: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["prompt"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _task_and_domain_by_episode(selection: Any) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    for task, domains in selection.episodes_by_task_domain:
        for domain, episode_ids in domains:
            for episode_index in episode_ids:
                _require(
                    int(episode_index) not in result,
                    f"duplicate selected episode {episode_index}",
                )
                result[int(episode_index)] = (str(task), str(domain))
    _require(
        len(result) == EXPECTED_SELECTED_EPISODES,
        f"full550 selection has {len(result)} episodes, expected {EXPECTED_SELECTED_EPISODES}",
    )
    return result


def _read_parquet_task_indices(
    dataset_root: Path,
    episode_to_partition: Mapping[int, tuple[str, str]],
    *,
    data_path_template: str,
    chunks_size: int,
) -> tuple[dict[str, set[int]], dict[int, set[int]], int]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency is required in runtime image
        raise Stage1TextCacheError("pyarrow is required for Stage1 text-cache inventory") from exc

    indices_by_task = {task: set() for task in OFFICIAL_TASKS}
    indices_by_episode: dict[int, set[int]] = {}
    frame_count = 0
    for episode_index in sorted(episode_to_partition):
        task, _ = episode_to_partition[episode_index]
        relative = data_path_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        parquet_path = dataset_root / relative
        _require(parquet_path.is_file(), f"selected Parquet is missing: {parquet_path}")
        try:
            table = pq.read_table(parquet_path, columns=["episode_index", "task_index"])
        except Exception as exc:
            raise Stage1TextCacheError(f"cannot read selected Parquet {parquet_path}: {exc}") from exc
        _require(table.num_rows > 0, f"selected Parquet is empty: {parquet_path}")
        episode_values = table.column("episode_index").to_pylist()
        _require(
            all(int(value) == episode_index for value in episode_values),
            f"Parquet episode_index mismatch: {parquet_path}",
        )
        task_values = table.column("task_index").to_pylist()
        try:
            episode_indices = {int(value) for value in task_values}
        except (TypeError, ValueError, OverflowError) as exc:
            raise Stage1TextCacheError(
                f"non-integer task_index in selected Parquet: {parquet_path}"
            ) from exc
        _require(episode_indices, f"selected Parquet has no task_index: {parquet_path}")
        _require(
            all(value >= 0 for value in episode_indices),
            f"negative task_index in selected Parquet: {parquet_path}",
        )
        indices_by_episode[episode_index] = episode_indices
        indices_by_task[task].update(episode_indices)
        frame_count += int(table.num_rows)

    _require(
        frame_count == EXPECTED_SELECTED_FRAMES,
        f"selected Parquet frame count {frame_count} != {EXPECTED_SELECTED_FRAMES}",
    )
    for task, expected in EXPECTED_UNIQUE_TASK_INDICES_BY_TASK.items():
        actual = len(indices_by_task[task])
        _require(actual == expected, f"{task} has {actual} unique task_index values, expected {expected}")
    all_indices = set().union(*(indices_by_task[task] for task in OFFICIAL_TASKS))
    _require(
        len(all_indices) == sum(len(indices_by_task[task]) for task in OFFICIAL_TASKS),
        "one task_index occurs in more than one protocol task",
    )
    _require(
        len(all_indices) == EXPECTED_UNIQUE_TASK_INDICES,
        f"selected Parquet has {len(all_indices)} unique task_index values, "
        f"expected {EXPECTED_UNIQUE_TASK_INDICES}",
    )
    return indices_by_task, indices_by_episode, frame_count


def _map_selected_task_indices(
    tasks_path: Path,
    selected_indices: set[int],
) -> dict[int, str]:
    remaining = set(selected_indices)
    result: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                raise Stage1TextCacheError(
                    f"cannot parse tasks metadata {tasks_path}:{line_number}: {exc}"
                ) from exc
            _require(isinstance(record, Mapping), f"tasks row is not an object: {line_number}")
            _require("task_index" in record and "task" in record, f"invalid tasks row: {line_number}")
            task_index = int(record["task_index"])
            if task_index not in remaining:
                continue
            instruction = record["task"]
            _require(
                isinstance(instruction, str) and bool(instruction.strip()),
                f"empty instruction for task_index {task_index}",
            )
            _require(task_index not in result, f"duplicate task_index row {task_index}")
            result[task_index] = instruction
            remaining.remove(task_index)
    _require(
        not remaining,
        f"tasks.jsonl lacks {len(remaining)} selected task indices; first={sorted(remaining)[:5]}",
    )
    return result


def _read_episode_task_declarations(
    episodes_path: Path,
    selected_episode_ids: set[int],
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                raise Stage1TextCacheError(
                    f"cannot parse episodes metadata {episodes_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, Mapping) or "episode_index" not in record:
                raise Stage1TextCacheError(f"invalid episodes row: {line_number}")
            episode_index = int(record["episode_index"])
            if episode_index not in selected_episode_ids:
                continue
            _require(episode_index not in result, f"duplicate episode metadata {episode_index}")
            tasks = record.get("tasks")
            _require(isinstance(tasks, list) and bool(tasks), f"episode {episode_index} has no tasks")
            _require(
                all(isinstance(value, str) and bool(value.strip()) for value in tasks),
                f"episode {episode_index} contains an invalid task declaration",
            )
            result[episode_index] = set(tasks)
    missing = selected_episode_ids - set(result)
    _require(not missing, f"episodes.jsonl lacks selected episodes: {sorted(missing)[:5]}")
    return result


def build_full550_prompt_inventory(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    stage1_config: str | Path = DEFAULT_STAGE1_CONFIG,
) -> dict[str, Any]:
    """Build the exact prompt inventory without loading any model."""

    root = Path(dataset_root).expanduser().resolve()
    config_path = Path(stage1_config).expanduser().resolve()
    _require(config_path.is_file(), f"Stage1 config not found: {config_path}")
    verified = verify_official_task_manifest(manifest_path, root)
    selection = select_official_full_550_per_task(verified)
    episode_to_partition = _task_and_domain_by_episode(selection)

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    _require(isinstance(info, Mapping), "official info.json root must be an object")
    data_path_template = info.get("data_path")
    chunks_size = info.get("chunks_size")
    _require(isinstance(data_path_template, str), "official data_path template is missing")
    _require(isinstance(chunks_size, int) and chunks_size > 0, "official chunks_size is invalid")

    indices_by_task, indices_by_episode, frame_count = _read_parquet_task_indices(
        root,
        episode_to_partition,
        data_path_template=data_path_template,
        chunks_size=chunks_size,
    )
    all_indices = set().union(*(indices_by_task[task] for task in OFFICIAL_TASKS))
    instructions = _map_selected_task_indices(root / "meta/tasks.jsonl", all_indices)
    episode_declarations = _read_episode_task_declarations(
        root / "meta/episodes.jsonl", set(episode_to_partition)
    )
    for episode_index, task_indices in indices_by_episode.items():
        undeclared = {
            instructions[task_index]
            for task_index in task_indices
            if instructions[task_index] not in episode_declarations[episode_index]
        }
        _require(
            not undeclared,
            f"episode {episode_index} Parquet task_index is absent from its metadata tasks; "
            f"first={sorted(undeclared)[:3]}",
        )

    entry_by_digest: dict[str, dict[str, str]] = {}
    for task_index in sorted(all_indices):
        prompt = DEFAULT_PROMPT.format(task=instructions[task_index])
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        existing = entry_by_digest.get(prompt_sha)
        _require(
            existing is None or existing["prompt"] == prompt,
            f"SHA-256 collision between selected prompts: {prompt_sha}",
        )
        _require(existing is None, f"duplicate prompt text has multiple selected task_index values: {prompt_sha}")
        owner = next(task for task in OFFICIAL_TASKS if task_index in indices_by_task[task])
        entry_by_digest[prompt_sha] = {
            "sha256": prompt_sha,
            "prompt": prompt,
            "task": owner,
            "task_index": str(task_index),
        }
    entries = sorted(entry_by_digest.values(), key=lambda item: item["sha256"])
    _require(len(entries) == EXPECTED_UNIQUE_TASK_INDICES, "prompt inventory count changed")

    counts_by_task_domain = {
        task: {domain: 0 for domain in OFFICIAL_DOMAINS} for task in OFFICIAL_TASKS
    }
    for _, (task, domain) in episode_to_partition.items():
        counts_by_task_domain[task][domain] += 1

    return {
        "status": "PASS",
        "kind": INVENTORY_KIND,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "policy_protocol_v2_stage1_b_cr",
        "training_launched": False,
        "stage1_config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        },
        "official_manifest": {
            "path": str(verified.manifest_path),
            "sha256": verified.manifest_sha256,
        },
        "dataset_root": str(root),
        "metadata": verified.as_provenance()["meta_files"],
        "selection": {
            "mode": "full_550_per_task",
            "episode_count": len(episode_to_partition),
            "frame_count": frame_count,
            "episode_counts_by_task_domain": counts_by_task_domain,
            "prompt_source": "selected_parquet_frame_task_index_union",
        },
        "prompts": {
            "unique_count": len(entries),
            "unique_counts_by_task": {
                task: len(indices_by_task[task]) for task in OFFICIAL_TASKS
            },
            "set_sha256": _canonical_prompt_set_sha(entries),
            "entries": entries,
        },
        "prompt_template": {
            "value": DEFAULT_PROMPT,
            "sha256": hashlib.sha256(DEFAULT_PROMPT.encode("utf-8")).hexdigest(),
        },
        "cache_contract": {
            "context_len": CONTEXT_LEN,
            "context_dim": CONTEXT_DIM,
            "context_dtype": "torch.bfloat16",
            "mask_dtype": "torch.bool",
            "encoder_id": ENCODER_ID,
            "filename_suffix": CACHE_SUFFIX,
            "over_length_prompt_count": 0,
        },
    }


def validate_prompt_inventory(value: Mapping[str, Any]) -> list[dict[str, str]]:
    _require(value.get("status") == "PASS", "prompt inventory status is not PASS")
    _require(value.get("kind") == INVENTORY_KIND, "unexpected prompt inventory kind")
    _require(value.get("schema_version") == SCHEMA_VERSION, "prompt inventory schema changed")
    selection = value.get("selection")
    _require(isinstance(selection, Mapping), "prompt inventory selection is missing")
    _require(selection.get("mode") == "full_550_per_task", "prompt inventory used wrong selection")
    _require(selection.get("episode_count") == EXPECTED_SELECTED_EPISODES, "episode count changed")
    _require(selection.get("frame_count") == EXPECTED_SELECTED_FRAMES, "frame count changed")
    _require(
        selection.get("prompt_source") == "selected_parquet_frame_task_index_union",
        "prompt inventory source is not selected Parquet task_index",
    )
    prompts = value.get("prompts")
    _require(isinstance(prompts, Mapping), "prompt inventory prompts are missing")
    raw_entries = prompts.get("entries")
    _require(isinstance(raw_entries, list), "prompt inventory entries are missing")
    entries: list[dict[str, str]] = []
    previous = ""
    for raw_entry in raw_entries:
        _require(isinstance(raw_entry, Mapping), "prompt inventory entry is not an object")
        prompt = raw_entry.get("prompt")
        digest = raw_entry.get("sha256")
        task = raw_entry.get("task")
        task_index = raw_entry.get("task_index")
        _require(isinstance(prompt, str) and bool(prompt), "empty prompt inventory entry")
        _require(isinstance(digest, str) and len(digest) == 64, "invalid prompt SHA-256")
        _require(hashlib.sha256(prompt.encode("utf-8")).hexdigest() == digest, "prompt SHA mismatch")
        _require(digest > previous, "prompt inventory is not uniquely SHA-sorted")
        _require(task in OFFICIAL_TASKS, "prompt inventory task is invalid")
        _require(isinstance(task_index, str) and task_index.isdigit(), "task_index is invalid")
        previous = digest
        entries.append(
            {"sha256": digest, "prompt": prompt, "task": str(task), "task_index": task_index}
        )
    _require(len(entries) == EXPECTED_UNIQUE_TASK_INDICES, "prompt entry count changed")
    _require(prompts.get("unique_count") == len(entries), "declared prompt count mismatch")
    counts = {task: sum(entry["task"] == task for entry in entries) for task in OFFICIAL_TASKS}
    _require(counts == dict(EXPECTED_UNIQUE_TASK_INDICES_BY_TASK), "per-task prompt counts changed")
    _require(prompts.get("unique_counts_by_task") == counts, "declared per-task counts mismatch")
    _require(
        prompts.get("set_sha256") == _canonical_prompt_set_sha(entries),
        "prompt-set SHA-256 mismatch",
    )
    template = value.get("prompt_template")
    _require(isinstance(template, Mapping), "prompt template identity is missing")
    _require(template.get("value") == DEFAULT_PROMPT, "prompt template bytes changed")
    _require(
        template.get("sha256") == hashlib.sha256(DEFAULT_PROMPT.encode("utf-8")).hexdigest(),
        "prompt template SHA-256 mismatch",
    )
    contract = value.get("cache_contract")
    _require(isinstance(contract, Mapping), "prompt cache contract is missing")
    _require(contract.get("context_len") == CONTEXT_LEN, "context_len changed")
    _require(contract.get("context_dim") == CONTEXT_DIM, "context_dim changed")
    _require(contract.get("over_length_prompt_count") == 0, "over-length contract changed")
    return entries


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _payload_sha256(context: torch.Tensor, mask: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(b"context\0torch.bfloat16\0" + b"128x4096\0")
    digest.update(_tensor_bytes(context))
    digest.update(b"mask\0torch.bool\0" + b"128\0")
    digest.update(_tensor_bytes(mask))
    return digest.hexdigest()


def validate_cache_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"text-cache payload is missing: {target}")
    try:
        try:
            payload = torch.load(target, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch
            payload = torch.load(target, map_location="cpu")
    except Exception as exc:
        raise Stage1TextCacheError(f"cannot load text-cache payload {target}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"text-cache payload is not a mapping: {target}")
    _require(set(payload) == {"context", "mask"}, f"unexpected payload keys: {target}")
    context = payload["context"]
    mask = payload["mask"]
    _require(isinstance(context, torch.Tensor), f"context tensor is missing: {target}")
    _require(isinstance(mask, torch.Tensor), f"mask tensor is missing: {target}")
    _require(tuple(context.shape) == (CONTEXT_LEN, CONTEXT_DIM), f"context shape invalid: {target}")
    _require(tuple(mask.shape) == (CONTEXT_LEN,), f"mask shape invalid: {target}")
    _require(context.dtype == torch.bfloat16, f"context dtype invalid: {target}")
    _require(mask.dtype == torch.bool, f"mask dtype invalid: {target}")
    _require(bool(torch.isfinite(context.float()).all()), f"context has non-finite values: {target}")
    _require(not bool(mask.all()), f"over-length/truncated prompt payload is forbidden: {target}")
    return {
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "payload_sha256": _payload_sha256(context, mask),
    }


def _atomic_torch_save_new(path: Path, payload: Mapping[str, torch.Tensor]) -> bool:
    """Atomically install one payload without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)


def _cache_path(cache_dir: Path, entry: Mapping[str, str]) -> Path:
    return cache_dir / f"{entry['sha256']}{CACHE_SUFFIX}"


BatchEncoder = Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor]]


def prepare_prompt_shard(
    entries: Sequence[Mapping[str, str]],
    *,
    cache_dir: str | Path,
    rank: int,
    world_size: int,
    batch_size: int,
    resume: bool,
    audit_only: bool,
    encode_batch: BatchEncoder | None,
    progress_every: int = 0,
) -> dict[str, Any]:
    _require(world_size > 0 and 0 <= rank < world_size, "invalid distributed rank/world size")
    _require(batch_size > 0, "batch_size must be positive")
    _require(progress_every >= 0, "progress_every must be nonnegative")
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    assigned = list(entries[rank::world_size])
    identities: dict[str, dict[str, Any]] = {}
    missing: list[Mapping[str, str]] = []
    skipped = 0
    for position, entry in enumerate(assigned, start=1):
        path = _cache_path(root, entry)
        if path.exists():
            _require(resume or audit_only, f"cache exists but --resume was not supplied: {path}")
            identities[path.name] = validate_cache_payload(path)
            skipped += 1
        else:
            missing.append(entry)
        if progress_every > 0 and (position % progress_every == 0 or position == len(assigned)):
            print(
                f"[stage1-text-cache rank={rank}/{world_size}] scanned "
                f"{position}/{len(assigned)} existing targets",
                file=sys.stderr,
                flush=True,
            )
    if audit_only:
        _require(not missing, f"audit-only cache is missing {len(missing)} rank-{rank} payloads")
    else:
        _require(encode_batch is not None or not missing, "an encoder is required for missing prompts")

    created = 0
    race_reused = 0
    over_length = 0
    if missing:
        assert encode_batch is not None
        with torch.no_grad():
            total_batches = (len(missing) + batch_size - 1) // batch_size
            for batch_index, start in enumerate(range(0, len(missing), batch_size), start=1):
                batch = missing[start : start + batch_size]
                contexts, masks = encode_batch([entry["prompt"] for entry in batch])
                _require(isinstance(contexts, torch.Tensor), "encoder contexts are not a tensor")
                _require(isinstance(masks, torch.Tensor), "encoder masks are not a tensor")
                _require(
                    tuple(contexts.shape) == (len(batch), CONTEXT_LEN, CONTEXT_DIM),
                    f"encoder context batch shape invalid: {tuple(contexts.shape)}",
                )
                _require(
                    tuple(masks.shape) == (len(batch), CONTEXT_LEN),
                    f"encoder mask batch shape invalid: {tuple(masks.shape)}",
                )
                masks = masks.to(dtype=torch.bool)
                batch_over_length = int(masks.all(dim=1).sum().item())
                over_length += batch_over_length
                _require(
                    batch_over_length == 0,
                    f"rank {rank} encoder produced {batch_over_length} over-length prompts "
                    f"in batch {batch_index}",
                )
                for offset, entry in enumerate(batch):
                    payload = {
                        "context": contexts[offset]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous(),
                        "mask": masks[offset].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                    }
                    _require(
                        bool(torch.isfinite(payload["context"].float()).all()),
                        f"rank {rank} encoder produced non-finite context for {entry['sha256']}",
                    )
                    path = _cache_path(root, entry)
                    installed = _atomic_torch_save_new(path, payload)
                    if installed:
                        created += 1
                    else:
                        _require(resume, f"cache appeared concurrently without --resume: {path}")
                        race_reused += 1
                    identities[path.name] = validate_cache_payload(path)
                if progress_every > 0 and (
                    batch_index % progress_every == 0 or batch_index == total_batches
                ):
                    print(
                        f"[stage1-text-cache rank={rank}/{world_size}] encoded batches "
                        f"{batch_index}/{total_batches}; created={created} reused={race_reused}",
                        file=sys.stderr,
                        flush=True,
                    )
    _require(over_length == 0, f"rank {rank} encoded {over_length} over-length prompts")
    _require(len(identities) == len(assigned), f"rank {rank} cache coverage is incomplete")
    return {
        "rank": rank,
        "world_size": world_size,
        "assigned_count": len(assigned),
        "created_count": created,
        "skipped_valid_count": skipped,
        "concurrent_valid_count": race_reused,
        "over_length_prompt_count": over_length,
        "files": [identities[name] for name in sorted(identities)],
    }


def merge_shard_reports(
    entries: Sequence[Mapping[str, str]],
    reports: Sequence[Mapping[str, Any]],
    *,
    cache_dir: str | Path,
) -> dict[str, Any]:
    _require(bool(reports), "no shard reports were produced")
    world_sizes = {int(report["world_size"]) for report in reports}
    _require(len(world_sizes) == 1, "shard reports disagree on world_size")
    world_size = next(iter(world_sizes))
    _require({int(report["rank"]) for report in reports} == set(range(world_size)), "rank reports incomplete")
    expected_names = {f"{entry['sha256']}{CACHE_SUFFIX}" for entry in entries}
    identities: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        _require(report.get("over_length_prompt_count") == 0, "a shard has over-length prompts")
        files = report.get("files")
        _require(isinstance(files, list), "shard file identities are missing")
        for identity in files:
            _require(isinstance(identity, Mapping), "invalid shard file identity")
            name = identity.get("filename")
            _require(isinstance(name, str), "shard filename is invalid")
            _require(name not in identities, f"duplicate cache identity across shards: {name}")
            identities[name] = identity
    _require(set(identities) == expected_names, "shard reports do not exactly cover prompt inventory")

    root = Path(cache_dir).expanduser().resolve()
    actual_pt_names = {path.name for path in root.glob("*.pt") if path.is_file()}
    _require(
        actual_pt_names == expected_names,
        f"cache directory .pt set mismatch: expected {len(expected_names)}, got {len(actual_pt_names)}",
    )
    digest = hashlib.sha256()
    total_size = 0
    for name in sorted(identities):
        identity = identities[name]
        size = int(identity["size_bytes"])
        payload_sha = str(identity["payload_sha256"])
        _require(size > 0, f"empty cache payload: {name}")
        _require(len(payload_sha) == 64, f"invalid payload SHA-256: {name}")
        total_size += size
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload_sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "directory": str(root),
        "file_count": len(identities),
        "total_size_bytes": total_size,
        "aggregate_payload_sha256": digest.hexdigest(),
        "all_payloads_valid": True,
        "extra_pt_files": 0,
        "over_length_prompt_count": 0,
        "world_size": world_size,
        "shards": [
            {
                key: report[key]
                for key in (
                    "rank",
                    "assigned_count",
                    "created_count",
                    "skipped_valid_count",
                    "concurrent_valid_count",
                )
            }
            for report in sorted(reports, key=lambda item: int(item["rank"]))
        ],
    }


def verify_text_cache_audit(
    audit_path: str | Path,
    *,
    cache_dir: str | Path,
    stage1_config_sha256: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Verify a completed cache sidecar without loading the text encoder.

    Payload tensors are validated during preparation and represented by the
    aggregate canonical payload digest.  Stage1 independently reloads all
    tensors and recomputes that aggregate before training.
    """

    target = Path(audit_path).expanduser().resolve()
    root = Path(cache_dir).expanduser().resolve()
    _require(target.is_file(), f"Stage1 text-cache audit is missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage1TextCacheError(f"cannot parse Stage1 text-cache audit {target}: {exc}") from exc
    _require(isinstance(value, dict), "Stage1 text-cache audit root is not an object")
    _require(value.get("status") == "PASS", "Stage1 text-cache audit status is not PASS")
    _require(value.get("kind") == AUDIT_KIND, "unexpected Stage1 text-cache audit kind")
    _require(value.get("schema_version") == SCHEMA_VERSION, "Stage1 text-cache audit schema changed")
    _require(value.get("training_launched") is False, "text-cache audit claims training")
    config = value.get("stage1_config")
    _require(isinstance(config, Mapping), "text-cache Stage1 config identity is missing")
    if stage1_config_sha256 is not None:
        _require(
            config.get("sha256") == stage1_config_sha256,
            "text cache was prepared for a different Stage1 config",
        )
    inventory_identity = value.get("inventory")
    _require(isinstance(inventory_identity, Mapping), "text-cache inventory identity is missing")
    inventory_path = Path(str(inventory_identity.get("path", ""))).expanduser().resolve()
    _require(inventory_path.is_file(), f"text-cache inventory is missing: {inventory_path}")
    _require(
        _sha256_file(inventory_path) == inventory_identity.get("sha256"),
        "text-cache inventory SHA-256 mismatch",
    )
    inventory, entries = _load_inventory(inventory_path)
    _require(
        inventory.get("stage1_config") == dict(config),
        "text-cache audit and prompt inventory bind different Stage1 configs",
    )
    _require(
        inventory["prompts"]["set_sha256"] == inventory_identity.get("prompt_set_sha256"),
        "text-cache prompt-set identity mismatch",
    )
    cache = value.get("cache")
    _require(isinstance(cache, Mapping), "text-cache completion section is missing")
    _require(Path(str(cache.get("directory", ""))).resolve() == root, "text-cache directory mismatch")
    _require(cache.get("file_count") == EXPECTED_UNIQUE_TASK_INDICES, "text-cache file count changed")
    _require(cache.get("all_payloads_valid") is True, "text-cache payload validation is absent")
    _require(cache.get("extra_pt_files") == 0, "text-cache audit permits extra .pt files")
    _require(cache.get("over_length_prompt_count") == 0, "text-cache has over-length prompts")
    aggregate = cache.get("aggregate_payload_sha256")
    _require(isinstance(aggregate, str) and len(aggregate) == 64, "cache aggregate SHA is invalid")
    model = value.get("model")
    _require(isinstance(model, Mapping), "text-cache model provenance is missing")
    _require(model.get("model_id") == MODEL_ID, "text-cache used a different text model")
    _require(
        model.get("tokenizer_model_id") == TOKENIZER_MODEL_ID,
        "text-cache used a different tokenizer",
    )
    for key in ("text_encoder_identity", "tokenizer_identity"):
        identity = model.get(key)
        _require(isinstance(identity, Mapping), f"text-cache {key} is missing")
        identity_sha = identity.get("aggregate_sha256")
        _require(
            isinstance(identity_sha, str) and len(identity_sha) == 64,
            f"text-cache {key} aggregate SHA-256 is invalid",
        )
        _require(int(identity.get("file_count", 0)) > 0, f"text-cache {key} has no files")
    expected_names = {f"{entry['sha256']}{CACHE_SUFFIX}" for entry in entries}
    actual_names = {path.name for path in root.glob("*.pt") if path.is_file()}
    _require(actual_names == expected_names, "current cache .pt set differs from completed audit")
    return value, entries


class Wan22TextBatchEncoder:
    """Thin wrapper around the author's precompute_text_embeds implementation."""

    def __init__(self, *, model_base_path: str | Path, device: str) -> None:
        from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
        from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

        self.device = device
        base = Path(model_base_path).expanduser().resolve()
        _require(base.is_dir(), f"model base path not found: {base}")
        with temporary_environment("DIFFSYNTH_MODEL_BASE_PATH", str(base)), temporary_environment(
            "DIFFSYNTH_SKIP_DOWNLOAD", "true"
        ):
            _, text_config, _, tokenizer_config = _resolve_configs(
                model_id=MODEL_ID,
                tokenizer_model_id=TOKENIZER_MODEL_ID,
                redirect_common_files=True,
            )
            text_config.download_if_necessary()
            tokenizer_config.download_if_necessary()
            _require(bool(text_config.path), "Wan2.2 text encoder asset is missing")
            _require(bool(tokenizer_config.path), "Wan tokenizer asset is missing")
            self.text_encoder = _load_registered_model(
                text_config.path,
                "wan_video_text_encoder",
                torch_dtype=torch.bfloat16,
                device=device,
            ).eval()
            self.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path,
                seq_len=CONTEXT_LEN,
                clean="whitespace",
            )
            self.text_encoder_path = str(text_config.path)
            self.tokenizer_path = str(tokenizer_config.path)
            self._raw_text_encoder_path = text_config.path
            self._raw_tokenizer_path = tokenizer_config.path
        self.model_base_path = str(base)

    def __call__(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.tokenizer(
            list(prompts), return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        context = self.text_encoder(ids, mask)
        return context, mask

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "tokenizer_model_id": TOKENIZER_MODEL_ID,
            "encoder_id": ENCODER_ID,
            "model_base_path": self.model_base_path,
            "text_encoder_path": self.text_encoder_path,
            "tokenizer_path": self.tokenizer_path,
            "text_encoder_identity": _artifact_identity(self._raw_text_encoder_path),
            "tokenizer_identity": _artifact_identity(self._raw_tokenizer_path),
            "dtype": "torch.bfloat16",
            "context_len": CONTEXT_LEN,
            "context_dim": CONTEXT_DIM,
            "implicit_downloads": False,
        }


def _distributed_context() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _default_sidecar(cache_dir: Path, suffix: str) -> Path:
    return cache_dir.with_name(f"{cache_dir.name}.{suffix}.json")


def _load_inventory(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage1TextCacheError(f"cannot load prompt inventory {path}: {exc}") from exc
    _require(isinstance(value, dict), "prompt inventory root is not an object")
    entries = validate_prompt_inventory(value)
    return value, entries


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    rank, world_size, local_rank, initialized_here = _distributed_context()
    try:
        if args.expected_world_size is not None:
            _require(
                world_size == args.expected_world_size,
                f"WORLD_SIZE={world_size}, expected {args.expected_world_size}",
            )
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        inventory_path = (
            Path(args.inventory_output).expanduser().resolve()
            if args.inventory_output
            else _default_sidecar(cache_dir, "inventory")
        )
        audit_path = (
            Path(args.audit_output).expanduser().resolve()
            if args.audit_output
            else _default_sidecar(cache_dir, "audit")
        )
        if rank == 0:
            cache_dir.mkdir(parents=True, exist_ok=True)
            fresh_inventory = build_full550_prompt_inventory(
                dataset_root=args.dataset_root,
                manifest_path=args.official_manifest,
                stage1_config=args.config,
            )
            if inventory_path.exists():
                existing, _ = _load_inventory(inventory_path)
                _require(existing == fresh_inventory, "existing prompt inventory differs from fresh audit")
            else:
                atomic_write_json(inventory_path, fresh_inventory)
        _barrier(world_size)
        inventory, entries = _load_inventory(inventory_path)
        if args.inventory_only:
            if rank == 0:
                return {
                    "status": "PASS",
                    "mode": "inventory_only",
                    "inventory": {
                        "path": str(inventory_path),
                        "sha256": _sha256_file(inventory_path),
                    },
                    "prompt_count": len(entries),
                    "training_launched": False,
                }
            return None

        if audit_path.exists() and not args.audit_only:
            raise Stage1TextCacheError(
                f"completed text-cache audit already exists; refusing to regenerate or overwrite: "
                f"{audit_path}. Use --audit-only to revalidate it."
            )
        if args.audit_only:
            _require(
                audit_path.is_file(),
                f"--audit-only requires an existing completed audit: {audit_path}",
            )

        encoder: Wan22TextBatchEncoder | None = None
        if not args.audit_only:
            _require(torch.cuda.is_available(), "CUDA is required to encode Stage1 text prompts")
            device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
            encoder = Wan22TextBatchEncoder(
                model_base_path=args.model_base_path,
                device=device,
            )
        report = prepare_prompt_shard(
            entries,
            cache_dir=cache_dir,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            resume=bool(args.resume),
            audit_only=bool(args.audit_only),
            encode_batch=encoder,
            progress_every=args.progress_every,
        )

        run_token_file = audit_path.with_name(f".{audit_path.name}.run-token")
        if rank == 0:
            atomic_write_json(run_token_file, {"token": uuid.uuid4().hex})
        _barrier(world_size)
        token_value = json.loads(run_token_file.read_text(encoding="utf-8"))
        run_token = str(token_value["token"])
        shard_path = audit_path.with_name(
            f".{audit_path.name}.{run_token}.rank-{rank:03d}.json"
        )
        atomic_write_json(shard_path, report)
        _barrier(world_size)

        result: dict[str, Any] | None = None
        if rank == 0:
            shard_paths = [
                audit_path.with_name(f".{audit_path.name}.{run_token}.rank-{value:03d}.json")
                for value in range(world_size)
            ]
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
            cache_audit = merge_shard_reports(entries, reports, cache_dir=cache_dir)
            source = Path(__file__).resolve()
            result = {
                "status": "PASS",
                "kind": AUDIT_KIND,
                "schema_version": SCHEMA_VERSION,
                "protocol_id": "policy_protocol_v2_stage1_b_cr",
                "mode": "audit_only" if args.audit_only else "prepare",
                "training_launched": False,
                "inventory": {
                    "path": str(inventory_path),
                    "sha256": _sha256_file(inventory_path),
                    "prompt_set_sha256": inventory["prompts"]["set_sha256"],
                },
                "stage1_config": inventory["stage1_config"],
                "official_manifest": inventory["official_manifest"],
                "metadata": inventory["metadata"],
                "selection": inventory["selection"],
                "model": (
                    encoder.provenance()
                    if encoder is not None
                    else {
                        "model_id": MODEL_ID,
                        "tokenizer_model_id": TOKENIZER_MODEL_ID,
                        "encoder_id": ENCODER_ID,
                        "context_len": CONTEXT_LEN,
                        "context_dim": CONTEXT_DIM,
                        "verified_from_existing_payloads": True,
                    }
                ),
                "cache": cache_audit,
                "implementation": {
                    "path": str(source),
                    "sha256": _sha256_file(source),
                    "author_reference": {
                        "path": str((PROJECT_ROOT / "scripts/precompute_text_embeds.py").resolve()),
                        "sha256": _sha256_file(PROJECT_ROOT / "scripts/precompute_text_embeds.py"),
                    },
                    "sharding": "sha256-sorted prompt entries sliced as entries[rank::world_size]",
                    "atomic_write": "same-directory temporary + create-only hard link",
                    "resume_policy": "validate existing payload then skip; never overwrite",
                },
            }
            if audit_path.exists():
                existing_audit, _ = verify_text_cache_audit(
                    audit_path,
                    cache_dir=cache_dir,
                    stage1_config_sha256=inventory["stage1_config"]["sha256"],
                )
                _require(
                    args.audit_only
                    and existing_audit["cache"]["aggregate_payload_sha256"]
                    == result["cache"]["aggregate_payload_sha256"],
                    f"refusing to overwrite completed text-cache audit: {audit_path}",
                )
                result = existing_audit
            else:
                atomic_write_json(audit_path, result)
            for path in shard_paths:
                path.unlink(missing_ok=True)
            run_token_file.unlink(missing_ok=True)
        _barrier(world_size)
        return result
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--official-manifest", type=Path, default=DEFAULT_OFFICIAL_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--inventory-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--model-base-path", type=Path, default=DEFAULT_MODEL_BASE_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="emit a per-rank heartbeat every N scan items / encoding batches; 0 disables",
    )
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--resume", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inventory-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        if result is not None:
            print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"Stage1 text-cache failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_KIND",
    "CACHE_SUFFIX",
    "EXPECTED_SELECTED_EPISODES",
    "EXPECTED_SELECTED_FRAMES",
    "EXPECTED_UNIQUE_TASK_INDICES",
    "EXPECTED_UNIQUE_TASK_INDICES_BY_TASK",
    "INVENTORY_KIND",
    "Stage1TextCacheError",
    "build_full550_prompt_inventory",
    "main",
    "merge_shard_reports",
    "prepare_prompt_shard",
    "validate_cache_payload",
    "validate_prompt_inventory",
    "verify_text_cache_audit",
]
