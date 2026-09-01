#!/usr/bin/env python3
"""Parallel, resumable extraction of the Pair-280 Layer-16 cache."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import secrets
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.robotwin.e0_e1.backbone import (
    FrozenFastWAMExtractor,
    format_deployment_prompt,
)

from .data import (
    build_policy_cache_extraction_contract,
    policy_cache_extractor_config,
    selected_episode_artifact_aggregate,
)
from .extract_policy_cache import (
    _cache_dataset_initialization_work_dir,
    _normalized_video_to_uint8_current,
    runtime_component_identities,
    select_indices_from_verified_state_bank,
    verify_release_base_lineage,
)
from .model import artifact_identity
from .native50hz_paired import TASK_INSTRUCTIONS
from .official_data import OFFICIAL_TASKS
from .protocol import POLICY_VARIANTS
from .pair280_protocol import (
    PAIR280_CACHE_SCHEMA,
    PAIR280_CACHE_SCHEMA_VERSION,
    PAIR280_CACHE_STORAGE,
    PAIR280_GROUPS,
    PAIR280_PROFILE_ID,
    PAIR280_STATE_ALGORITHM,
    PAIR280_STATE_SEED,
    PAIR280_STATES_PER_TRAJECTORY,
    PAIR280_TOKEN_SHAPE,
    PAIR280_TRAIN_TRAJECTORIES,
    PAIR280_VIEWS,
    Pair280ContractError,
    canonical_json_sha256,
    validate_pair280_cache_manifest,
    verify_pair280_state_bank,
)
from .prepare_release_paired_text_cache import verify_release_paired_text_cache
from .release_paired_binding import verify_release_paired_binding
from .runtime_utils import (
    PROJECT_ROOT,
    audit_local_fastwam_source,
    compose_robotwin_config,
    instantiate_native_paired_action_dataset,
)


CAPTURE_LAYER = 16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pair280ContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _relative_identity(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    _require(resolved.is_relative_to(root.resolve()), f"artifact escaped cache root: {resolved}")
    identity = artifact_identity(resolved)
    _require(identity.get("kind") == "file", f"cache artifact is not a file: {resolved}")
    return {
        "relative_path": resolved.relative_to(root.resolve()).as_posix(),
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def audit_inputs(
    *,
    base_lineage_manifest: str | Path,
    base_lineage_sha256: str,
    release_binding_path: str | Path,
    release_binding_sha256: str,
    checkpoint: str | Path,
    dataset_stats: str | Path,
    official_manifest: str | Path,
    paired_root: str | Path,
    paired_manifest: str | Path,
    paired_audit: str | Path,
    state_bank: str | Path,
    paired_text_cache: str | Path,
) -> dict[str, Any]:
    lineage = verify_release_base_lineage(
        base_lineage_manifest,
        checkpoint=checkpoint,
        dataset_stats=dataset_stats,
        official_manifest=official_manifest,
        expected_manifest_sha256=base_lineage_sha256,
    )
    binding = verify_release_paired_binding(
        release_binding_path, expected_sha256=release_binding_sha256
    )
    _require(int(binding["schema_version"]) == 2, "Pair-280 requires release binding schema v2")
    bank = verify_pair280_state_bank(
        state_bank,
        paired_root=paired_root,
        paired_manifest=paired_manifest,
        paired_audit=paired_audit,
    )
    paired_dataset = binding["paired_dataset"]
    _require(paired_dataset["state_bank_sha256"] == bank.sha256, "Pair-280 binding state-bank SHA differs")
    _require(
        paired_dataset["physical_state_inventory_sha256"]
        == bank.physical_state_inventory_sha256,
        "Pair-280 binding inventory differs",
    )
    selected = selected_episode_artifact_aggregate(bank.native_manifest, split="train")
    for key in ("algorithm", "episode_count", "file_count", "size_bytes", "sha256"):
        _require(binding["selected_train_artifacts"][key] == selected[key], f"Pair-280 selected artifact {key} differs")
    text_audit = verify_release_paired_text_cache(
        paired_text_cache,
        expected_base_lineage_sha256=lineage["manifest_identity"]["sha256"],
        expected_release_paired_binding_sha256=binding["binding_manifest_identity"]["sha256"],
    )
    return {
        "status": "PASS",
        "kind": "policy_pair280_cache_input_audit",
        "profile_id": PAIR280_PROFILE_ID,
        "base_lineage": lineage,
        "release_paired_binding": binding,
        "state_bank": {
            "path": str(bank.path),
            "size_bytes": bank.path.stat().st_size,
            "sha256": bank.sha256,
            "physical_state_inventory_sha256": bank.physical_state_inventory_sha256,
            "groups": len(bank.anchors),
        },
        "selected_episode_artifacts": selected,
        "paired_text_cache": artifact_identity(paired_text_cache),
        "paired_text_cache_audit": text_audit["audit_identity"],
        "source_artifacts": {
            "paired_action_manifest": artifact_identity(paired_manifest),
            "paired_action_audit": artifact_identity(paired_audit),
        },
    }


def _build_extraction_contract(
    *, input_audit: Mapping[str, Any], components: Mapping[str, Any]
) -> dict[str, Any]:
    extractor_source = artifact_identity(Path(__file__))
    support_paths = {
        "frozen_backbone": PROJECT_ROOT / "experiments/robotwin/e0_e1/backbone.py",
        "runtime_utils": PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/runtime_utils.py",
        "policy_data": PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/data.py",
        "policy_protocol": PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/protocol.py",
        "pair280_protocol": PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/pair280_protocol.py",
    }
    extractor_config = policy_cache_extractor_config(
        states_per_trajectory=PAIR280_STATES_PER_TRAJECTORY,
        state_selection_algorithm=PAIR280_STATE_ALGORITHM,
        state_selection_seed=PAIR280_STATE_SEED,
        storage=PAIR280_CACHE_STORAGE,
    )
    extractor_config.update(
        {
            "parallel_workers": 8,
            "shard_unit": "one_physical_trajectory",
            "shard_count": PAIR280_TRAIN_TRAJECTORIES,
        }
    )
    return build_policy_cache_extraction_contract(
        base_lineage_identity=input_audit["base_lineage"]["manifest_identity"],
        release_paired_binding_identity=input_audit["release_paired_binding"]["binding_manifest_identity"],
        dataset_stats_identity=artifact_identity(
            input_audit["base_lineage"]["dataset_stats"]["path"]
        ),
        vae_identity=components["vae"],
        text_encoder_identity=components["text_encoder"],
        tokenizer_identity=components["tokenizer"],
        text_cache_identity=input_audit["paired_text_cache"],
        fastwam_source_audit=audit_local_fastwam_source(),
        extractor_source_identity=extractor_source,
        extractor_support_source_identities={
            name: artifact_identity(path) for name, path in support_paths.items()
        },
        selected_episode_artifacts=input_audit["selected_episode_artifacts"],
        extractor_config_override=extractor_config,
    )


def _verify_existing_shard(root: Path, metadata_path: Path) -> dict[str, Any]:
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("status") == "PASS", f"invalid shard metadata: {metadata_path}")
    tensor_identity = value.get("tensor_file")
    _require(isinstance(tensor_identity, Mapping), "shard tensor identity missing")
    tensor_path = root / str(tensor_identity["relative_path"])
    _require(tensor_path.is_file(), f"shard tensor missing: {tensor_path}")
    _require(int(tensor_path.stat().st_size) == int(tensor_identity["size_bytes"]), "shard tensor size changed")
    _require(_sha256(tensor_path) == tensor_identity["sha256"], "shard tensor SHA changed")
    value["metadata_file"] = _relative_identity(root, metadata_path)
    return value


def extract_worker(
    *,
    input_audit_path: str | Path,
    output_root: str | Path,
    paired_root: str | Path,
    paired_manifest: str | Path,
    paired_audit: str | Path,
    state_bank: str | Path,
    paired_text_cache: str | Path,
    model_base_path: str | Path,
    worker_index: int,
    worker_count: int,
    device: str,
    trajectory_limit: int | None = None,
) -> dict[str, Any]:
    cpu_threads = int(os.environ.get("PAIR280_CPU_THREADS_PER_WORKER", "16"))
    _require(1 <= cpu_threads <= 32, "PAIR280_CPU_THREADS_PER_WORKER must be in [1,32]")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError as exc:
        raise Pair280ContractError(
            "Pair-280 inter-op thread cap was applied after parallel work started"
        ) from exc
    root = Path(output_root).expanduser().resolve()
    _require(root.is_dir(), f"Pair-280 output root missing: {root}")
    _require(int(worker_count) == 8, "Pair-280 extraction requires exactly eight workers")
    _require(0 <= int(worker_index) < int(worker_count), "invalid Pair-280 worker index")
    input_audit = json.loads(Path(input_audit_path).read_text(encoding="utf-8"))
    _require(input_audit.get("status") == "PASS", "Pair-280 input audit is not PASS")
    bank = verify_pair280_state_bank(
        state_bank,
        paired_root=paired_root,
        paired_manifest=paired_manifest,
        paired_audit=paired_audit,
        expected_sha256=input_audit["state_bank"]["sha256"],
    )
    stats = Path(input_audit["base_lineage"]["dataset_stats"]["path"])
    with _cache_dataset_initialization_work_dir(root / f"worker_{worker_index}"):
        dataset = instantiate_native_paired_action_dataset(
            compose_robotwin_config(),
            dataset_root=paired_root,
            dataset_stats_path=stats,
            text_cache_dir=paired_text_cache,
            model_for_on_the_fly_text=None,
            manifest_path=paired_manifest,
            audit_path=paired_audit,
            state_bank_path=state_bank,
            expected_state_bank_sha256=bank.sha256,
            split="train",
            expected_tasks=OFFICIAL_TASKS,
            require_full_protocol_counts=True,
            state_bank_states_per_trajectory=PAIR280_STATES_PER_TRAJECTORY,
            state_bank_sampling_algorithm=PAIR280_STATE_ALGORITHM,
            state_bank_sampling_version=1,
            state_bank_sampling_seed=PAIR280_STATE_SEED,
        )
    selected_indices, selection_plan = select_indices_from_verified_state_bank(dataset, bank)
    _require(len(selected_indices) == PAIR280_GROUPS, "Pair-280 native dataset resolution changed")
    trajectory_rows: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    current_trajectory: str | None = None
    for index, plan in zip(selected_indices, selection_plan, strict=True):
        trajectory = str(plan["trajectory_id"])
        if current_trajectory is not None and trajectory != current_trajectory:
            trajectory_rows.append((current_trajectory, current))
            current = []
        current_trajectory = trajectory
        current.append((index, plan))
    if current_trajectory is not None:
        trajectory_rows.append((current_trajectory, current))
    _require(len(trajectory_rows) == PAIR280_TRAIN_TRAJECTORIES, "Pair-280 trajectory partition changed")
    assigned = [row for position, row in enumerate(trajectory_rows) if position % worker_count == worker_index]
    _require(bool(assigned), "Pair-280 worker has no assigned trajectories")
    if trajectory_limit is not None:
        _require(int(trajectory_limit) > 0, "trajectory_limit must be positive")
        assigned = assigned[: int(trajectory_limit)]

    audit_path = root / "workers" / f"worker_{worker_index:02d}.json"
    if audit_path.exists():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        _require(existing.get("status") == "PASS", "existing Pair-280 worker audit is not PASS")
        _require(int(existing.get("worker_index", -1)) == worker_index, "existing worker audit index changed")
        _require(int(existing.get("completed_shards", -1)) == len(assigned), "existing worker audit shard count changed")
        for trajectory_id, rows in assigned:
            task = str(rows[0][1]["task"])
            content_id = int(rows[0][1]["content_id"])
            metadata_path = root / "shards" / task / f"content_{content_id:06d}.json"
            metadata = _verify_existing_shard(root, metadata_path)
            _require(metadata["trajectory_id"] == trajectory_id, "existing worker shard trajectory changed")
        print(f"worker={worker_index} SKIP complete verified worker audit", flush=True)
        return existing

    checkpoint = Path(input_audit["base_lineage"]["checkpoint"]["path"])
    extractor = FrozenFastWAMExtractor.from_release_checkpoint(
        checkpoint,
        stats,
        model_base_path=model_base_path,
        device=device,
        capture_layers=(CAPTURE_LAYER,),
        verify_native_prefill=worker_index == 0,
        compute_checkpoint_sha256=True,
    )
    components = runtime_component_identities(extractor)
    extraction_contract = _build_extraction_contract(
        input_audit=input_audit, components=components
    )
    shard_results: list[dict[str, Any]] = []
    verified_prefill_states = 0
    for trajectory_position, (trajectory_id, rows) in enumerate(assigned):
        _require(len(rows) == PAIR280_STATES_PER_TRAJECTORY, f"{trajectory_id} state count changed")
        task = str(rows[0][1]["task"])
        content_id = int(rows[0][1]["content_id"])
        relative_base = Path("shards") / task / f"content_{content_id:06d}"
        tensor_path = root / relative_base.with_suffix(".safetensors")
        metadata_path = root / relative_base.with_suffix(".json")
        if tensor_path.exists() or metadata_path.exists():
            _require(tensor_path.is_file() and metadata_path.is_file(), f"partial shard exists for {trajectory_id}")
            shard_results.append(_verify_existing_shard(root, metadata_path))
            print(f"worker={worker_index} SKIP verified {trajectory_id}", flush=True)
            continue
        token_rows: list[torch.Tensor] = []
        proprio_rows: list[torch.Tensor] = []
        state_rows: list[dict[str, Any]] = []
        started = time.monotonic()
        for local_index, (dataset_index, plan) in enumerate(rows):
            sample = dataset[dataset_index]
            _require(sample["physical_state_id"] == plan["physical_state_id"], "Pair-280 dataset/state plan differs")
            _require(not bool(sample["action_is_pad"].any()), "Pair-280 selected state contains padding")
            _require(
                sample["prompt"] == format_deployment_prompt(TASK_INSTRUCTIONS[sample["task"]]),
                "Pair-280 prompt changed",
            )
            state_window = sample["state_window"]
            _require(tuple(state_window.shape) == (4, 33, 14), "Pair-280 state window shape changed")
            proprio_views = state_window[:, 0].detach().cpu().float()
            _require(torch.equal(proprio_views, proprio_views[0:1].expand_as(proprio_views)), "Pair-280 view proprio differs")
            output = extractor.extract_current_observations(
                _normalized_video_to_uint8_current(sample["video"]),
                proprio_views,
                context=sample["context"].detach().cpu(),
                context_mask=sample["context_mask"].detach().cpu(),
                proprio_mode="observed",
            )
            native_verified = bool(output.provenance.get("native_prefill_verified"))
            if native_verified:
                verified_prefill_states += 1
                _require(worker_index == 0 and verified_prefill_states == 1, "native prefill verification ran unexpectedly")
                extractor.verify_native_prefill = False
            tokens = output.tokens_by_layer[CAPTURE_LAYER].detach().cpu().contiguous()
            _require(tuple(tokens.shape) == (4, *PAIR280_TOKEN_SHAPE), "Pair-280 Layer-16 shape changed")
            _require(tokens.dtype == torch.bfloat16, "Pair-280 Layer-16 dtype changed")
            _require(bool(torch.isfinite(tokens.float()).all()), "Pair-280 Layer-16 contains NaN/inf")
            token_rows.append(tokens)
            proprio_rows.append(proprio_views[0].clone())
            state_rows.append(
                {
                    "task": str(sample["task"]),
                    "trajectory_id": str(sample["trajectory_id"]),
                    "content_id": int(sample["content_id"]),
                    "frame_offset": int(sample["frame_offset"]),
                    "physical_state_id": str(sample["physical_state_id"]),
                    "episode_indices": [int(value) for value in sample["episode_indices"]],
                }
            )
            if (local_index + 1) % 20 == 0 or local_index + 1 == len(rows):
                print(
                    f"worker={worker_index} trajectory={trajectory_position + 1}/{len(assigned)} "
                    f"state={local_index + 1}/280 id={sample['physical_state_id']}",
                    flush=True,
                )
        tokens_tensor = torch.stack(token_rows).contiguous()
        proprio_tensor = torch.stack(proprio_rows).float().contiguous()
        _require(tuple(tokens_tensor.shape) == (280, 4, *PAIR280_TOKEN_SHAPE), "Pair-280 shard token shape changed")
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tensor_path.with_name(f".{tensor_path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
        try:
            save_file(
                {"tokens": tokens_tensor, "proprio_raw": proprio_tensor},
                temporary,
                metadata={
                    "profile_id": PAIR280_PROFILE_ID,
                    "trajectory_id": trajectory_id,
                    "state_inventory_sha256": canonical_json_sha256(
                        [row["physical_state_id"] for row in state_rows]
                    ),
                },
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.link(temporary, tensor_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        tensor_identity = _relative_identity(root, tensor_path)
        metadata = {
            "schema_version": 1,
            "kind": "policy_pair280_trajectory_cache_shard",
            "status": "PASS",
            "profile_id": PAIR280_PROFILE_ID,
            "worker_index": worker_index,
            "trajectory_id": trajectory_id,
            "task": task,
            "content_id": content_id,
            "state_count": len(state_rows),
            "physical_state_ids": [row["physical_state_id"] for row in state_rows],
            "state_rows": state_rows,
            "tensor_file": tensor_identity,
            "token_shape": [280, 4, *PAIR280_TOKEN_SHAPE],
            "token_dtype": str(tokens_tensor.dtype),
            "proprio_shape": [280, 14],
            "state_bank_sha256": bank.sha256,
            "release_paired_binding_sha256": input_audit["release_paired_binding"]["binding_manifest_identity"]["sha256"],
            "extraction_contract_sha256": canonical_json_sha256(extraction_contract),
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_new_json(metadata_path, metadata)
        metadata["metadata_file"] = _relative_identity(root, metadata_path)
        shard_results.append(metadata)
        del tokens_tensor, proprio_tensor, token_rows, proprio_rows
    worker_audit = {
        "schema_version": 1,
        "kind": "policy_pair280_extraction_worker_audit",
        "status": "PASS",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "assigned_trajectories": len(assigned),
        "completed_shards": len(shard_results),
        "engineering_trajectory_limit": trajectory_limit,
        "native_prefill_verified_states": verified_prefill_states,
        "components": components,
        "extraction_contract": extraction_contract,
        "shard_metadata_sha256": canonical_json_sha256(
            [row["metadata_file"]["sha256"] for row in shard_results]
        ),
    }
    _write_new_json(audit_path, worker_audit)
    return worker_audit


def merge_cache(*, input_audit_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    manifest_path = root / "cache_manifest.json"
    _require(not manifest_path.exists(), f"Pair-280 cache manifest already exists: {manifest_path}")
    input_audit = json.loads(Path(input_audit_path).read_text(encoding="utf-8"))
    worker_audits: list[dict[str, Any]] = []
    for worker_index in range(8):
        path = root / "workers" / f"worker_{worker_index:02d}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        _require(value.get("status") == "PASS", f"worker {worker_index} did not pass")
        worker_audits.append(value)
    contract_shas = {
        canonical_json_sha256(value["extraction_contract"]) for value in worker_audits
    }
    _require(len(contract_shas) == 1, "Pair-280 workers used different extraction contracts")
    component_shas = {
        canonical_json_sha256(value["components"]) for value in worker_audits
    }
    _require(len(component_shas) == 1, "Pair-280 workers used different model components")
    _require(
        sum(int(value["native_prefill_verified_states"]) for value in worker_audits) == 1,
        "Pair-280 native-prefill bit-exact audit must run exactly once",
    )
    metadata_paths = sorted((root / "shards").glob("*/content_*.json"))
    _require(len(metadata_paths) == PAIR280_TRAIN_TRAJECTORIES, "Pair-280 merge requires 90 metadata shards")
    shards: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        metadata = _verify_existing_shard(root, metadata_path)
        tensor_path = root / metadata["tensor_file"]["relative_path"]
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            _require(set(handle.keys()) == {"proprio_raw", "tokens"}, "Pair-280 shard tensor keys changed")
            tokens = handle.get_slice("tokens")
            proprio = handle.get_slice("proprio_raw")
            _require(tuple(tokens.get_shape()) == (280, 4, *PAIR280_TOKEN_SHAPE), "Pair-280 shard token shape changed")
            _require(tuple(proprio.get_shape()) == (280, 14), "Pair-280 shard proprio shape changed")
            for start in range(0, 280, 20):
                _require(bool(torch.isfinite(tokens[start : start + 20].float()).all()), "Pair-280 shard contains non-finite tokens")
            _require(bool(torch.isfinite(proprio[:].float()).all()), "Pair-280 shard contains non-finite proprio")
        shards.append(
            {
                key: metadata[key]
                for key in (
                    "trajectory_id",
                    "task",
                    "content_id",
                    "state_count",
                    "physical_state_ids",
                    "tensor_file",
                )
            }
            | {"metadata_file": _relative_identity(root, metadata_path)}
        )
    task_order = {task: position for position, task in enumerate(OFFICIAL_TASKS)}
    shards.sort(key=lambda row: (task_order[row["task"]], int(row["content_id"])))
    ordered_states = [state for shard in shards for state in shard["physical_state_ids"]]
    _require(len(ordered_states) == len(set(ordered_states)) == PAIR280_GROUPS, "Pair-280 merged states changed")
    bank_states = json.loads(Path(input_audit["state_bank"]["path"]).read_text(encoding="utf-8"))["states"]
    expected_states = [str(row["physical_state_id"]) for row in bank_states]
    _require(ordered_states == expected_states, "Pair-280 shard order differs from state bank")
    payload_size = sum(
        int(shard[key]["size_bytes"])
        for shard in shards
        for key in ("tensor_file", "metadata_file")
    )
    manifest = {
        "schema": PAIR280_CACHE_SCHEMA,
        "schema_version": PAIR280_CACHE_SCHEMA_VERSION,
        "kind": "policy_pair280_layer16_cache",
        "status": "PASS",
        "profile_id": PAIR280_PROFILE_ID,
        "storage": PAIR280_CACHE_STORAGE,
        "capture_layer": CAPTURE_LAYER,
        "physical_state_groups": PAIR280_GROUPS,
        "scene_views": PAIR280_VIEWS,
        "variant_names": list(POLICY_VARIANTS),
        "token_shape": [4, *PAIR280_TOKEN_SHAPE],
        "token_dtype": "torch.bfloat16",
        "payload_size_bytes": payload_size,
        "ordered_state_ids_sha256": canonical_json_sha256(ordered_states),
        "state_bank": dict(input_audit["state_bank"]),
        "release_paired_binding": dict(
            input_audit["release_paired_binding"]["binding_manifest_identity"]
        ),
        "base_lineage": dict(input_audit["base_lineage"]["manifest_identity"]),
        "backbone_checkpoint": dict(input_audit["base_lineage"]["checkpoint"]),
        "source_artifacts": dict(input_audit["source_artifacts"]),
        "selected_episode_artifacts": dict(input_audit["selected_episode_artifacts"]),
        "paired_text_cache": dict(input_audit["paired_text_cache"]),
        "components": dict(worker_audits[0]["components"]),
        "extraction_contract": dict(worker_audits[0]["extraction_contract"]),
        "native_prefill_identity_audit": {
            "status": "PASS",
            "checked_states": 1,
            "comparison": "bit_exact_K_and_V_for_every_layer",
            "rtol": 0.0,
            "atol": 0.0,
        },
        "worker_audits": [
            _relative_identity(root, root / "workers" / f"worker_{index:02d}.json")
            for index in range(8)
        ],
        "shards": shards,
    }
    _write_new_json(manifest_path, manifest)
    verified = validate_pair280_cache_manifest(
        manifest_path,
        expected_state_bank_sha256=input_audit["state_bank"]["sha256"],
        expected_release_binding_sha256=input_audit["release_paired_binding"]["binding_manifest_identity"]["sha256"],
        verify_shard_hashes=True,
    )
    audit = {
        "schema_version": 1,
        "kind": "policy_pair280_layer16_cache_audit",
        "status": "PASS",
        "profile_id": PAIR280_PROFILE_ID,
        "cache_manifest": verified["manifest_identity"],
        "physical_state_groups": PAIR280_GROUPS,
        "scene_views": PAIR280_VIEWS,
        "shard_count": len(shards),
        "payload_size_bytes": payload_size,
        "all_shard_hashes_verified": True,
        "all_tokens_finite": True,
        "ordered_state_ids_sha256": manifest["ordered_state_ids_sha256"],
        "extraction_contract_sha256": next(iter(contract_shas)),
    }
    _write_new_json(root / "cache_audit.json", audit)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-inputs")
    for command in (audit,):
        command.add_argument("--base-lineage-manifest", required=True, type=Path)
        command.add_argument("--base-lineage-sha256", required=True)
        command.add_argument("--release-binding", required=True, type=Path)
        command.add_argument("--release-binding-sha256", required=True)
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--dataset-stats", required=True, type=Path)
        command.add_argument("--official-manifest", required=True, type=Path)
        command.add_argument("--paired-root", required=True, type=Path)
        command.add_argument("--paired-manifest", required=True, type=Path)
        command.add_argument("--paired-audit", required=True, type=Path)
        command.add_argument("--state-bank", required=True, type=Path)
        command.add_argument("--paired-text-cache", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    worker = sub.add_parser("extract-worker")
    worker.add_argument("--input-audit", required=True, type=Path)
    worker.add_argument("--output-root", required=True, type=Path)
    worker.add_argument("--paired-root", required=True, type=Path)
    worker.add_argument("--paired-manifest", required=True, type=Path)
    worker.add_argument("--paired-audit", required=True, type=Path)
    worker.add_argument("--state-bank", required=True, type=Path)
    worker.add_argument("--paired-text-cache", required=True, type=Path)
    worker.add_argument("--model-base-path", required=True, type=Path)
    worker.add_argument("--worker-index", required=True, type=int)
    worker.add_argument("--worker-count", default=8, type=int)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--trajectory-limit", type=int)
    merge = sub.add_parser("merge")
    merge.add_argument("--input-audit", required=True, type=Path)
    merge.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "audit-inputs":
            result = audit_inputs(
                base_lineage_manifest=args.base_lineage_manifest,
                base_lineage_sha256=args.base_lineage_sha256,
                release_binding_path=args.release_binding,
                release_binding_sha256=args.release_binding_sha256,
                checkpoint=args.checkpoint,
                dataset_stats=args.dataset_stats,
                official_manifest=args.official_manifest,
                paired_root=args.paired_root,
                paired_manifest=args.paired_manifest,
                paired_audit=args.paired_audit,
                state_bank=args.state_bank,
                paired_text_cache=args.paired_text_cache,
            )
            _write_new_json(args.output.expanduser().resolve(), result)
        elif args.command == "extract-worker":
            result = extract_worker(
                input_audit_path=args.input_audit,
                output_root=args.output_root,
                paired_root=args.paired_root,
                paired_manifest=args.paired_manifest,
                paired_audit=args.paired_audit,
                state_bank=args.state_bank,
                paired_text_cache=args.paired_text_cache,
                model_base_path=args.model_base_path,
                worker_index=args.worker_index,
                worker_count=args.worker_count,
                device=args.device,
                trajectory_limit=args.trajectory_limit,
            )
        else:
            result = merge_cache(
                input_audit_path=args.input_audit,
                output_root=args.output_root,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"Pair-280 extraction failed closed: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
