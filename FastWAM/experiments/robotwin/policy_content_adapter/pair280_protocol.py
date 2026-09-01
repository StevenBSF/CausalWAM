#!/usr/bin/env python3
"""Pair-280 post-training data, cache, and exact exposure contracts.

This module is deliberately additive.  The historical 8-state/trajectory
Policy-v2 cache remains readable and immutable; Pair-280 uses a derived state
bank, a schema-v2 release binding, trajectory-sharded safetensors, and an exact
ten-pass sampler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import secrets
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.utils.data import Dataset, Sampler

from .data import (
    DataContractError,
    PolicyPhysicalStateAnchor,
    VerifiedPolicyStateBank,
    physical_state_inventory_sha256,
    policy_state_bank_offsets,
    verify_native_paired_action_manifest,
    verify_policy_state_bank,
)
from .model import artifact_identity
from .native50hz_paired import atomic_write_json
from .official_data import OFFICIAL_TASKS
from .protocol import (
    POLICY_ACTION_DIM,
    POLICY_ACTION_STEPS,
    POLICY_CAMERA_COUNT,
    POLICY_CAMERA_NAMES,
    POLICY_NATIVE_FPS,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_STATE_BANK_SCHEMA,
    POLICY_STATE_BANK_SCHEMA_VERSION,
    POLICY_TEMPORAL_RESAMPLING,
    POLICY_VARIANTS,
    POLICY_VIEW_COUNT,
)
from .release_paired_binding import (
    BINDING_KIND,
    PAIR280_BINDING_SCHEMA_VERSION,
    verify_release_paired_binding,
)


PAIR280_PROFILE_ID = "pair280_exact10_v1"
PAIR280_STATE_ALGORITHM = "sha256_rank_endpoint_safe_pair280_v1"
PAIR280_STATE_ALGORITHM_VERSION = 1
PAIR280_STATE_SEED = 42
PAIR280_STATES_PER_TRAJECTORY = 280
PAIR280_TRAIN_TRAJECTORIES = 90
PAIR280_GROUPS = 25_200
PAIR280_VIEWS = 100_800
PAIR280_EPOCHS = 10
PAIR280_GLOBAL_GROUPS = 16
PAIR280_LOCAL_GROUPS = 2
PAIR280_WORLD_SIZE = 8
PAIR280_ACTIVE_STEPS = 15_750
PAIR280_TOTAL_STEPS = 18_215
PAIR280_INACTIVE_STEPS = PAIR280_TOTAL_STEPS - PAIR280_ACTIVE_STEPS
PAIR280_CACHE_SCHEMA = "policy_pair280_sharded_cache_v1"
PAIR280_CACHE_SCHEMA_VERSION = 1
PAIR280_CACHE_STORAGE = "trajectory_sharded_safetensors_v1"
PAIR280_TOKEN_SHAPE = (120, 3072)


class Pair280ContractError(ValueError):
    """A Pair-280 artifact or execution schedule is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pair280ContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} missing/unsafe: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pair280ContractError(f"cannot parse {label}: {resolved}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, resolved


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def pair280_protocol_metadata() -> dict[str, Any]:
    return {
        "protocol_id": POLICY_PROTOCOL_ID,
        "variant_names": list(POLICY_VARIANTS),
        "view_count": POLICY_VIEW_COUNT,
        "r3_role": POLICY_R3_ROLE,
        "camera_count": POLICY_CAMERA_COUNT,
        "camera_names": list(POLICY_CAMERA_NAMES),
        "native_fps": POLICY_NATIVE_FPS,
        "action_steps": POLICY_ACTION_STEPS,
        "action_dim": POLICY_ACTION_DIM,
        "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
        "native_action_targets": True,
        "split": "train",
    }


def build_pair280_state_bank(
    *,
    paired_root: str | Path,
    paired_manifest: str | Path,
    paired_audit: str | Path,
) -> dict[str, Any]:
    verified = verify_native_paired_action_manifest(
        paired_manifest,
        dataset_root=paired_root,
        audit_path=paired_audit,
    )
    train_groups = verified.groups_for_split("train")
    _require(len(train_groups) == PAIR280_TRAIN_TRAJECTORIES, "Pair-280 requires 90 train trajectories")
    _require(
        tuple(dict.fromkeys(group.task for group in train_groups)) == tuple(OFFICIAL_TASKS),
        "Pair-280 task order changed",
    )
    anchors: list[PolicyPhysicalStateAnchor] = []
    minimum_valid = min(group.valid_action_anchor_count for group in train_groups)
    _require(
        minimum_valid >= PAIR280_STATES_PER_TRAJECTORY,
        "a train trajectory has fewer than 280 endpoint-safe action anchors",
    )
    for group in train_groups:
        offsets = policy_state_bank_offsets(
            task=group.task,
            content_id=group.content_id,
            episode_length=group.episode_length,
            seed=PAIR280_STATE_SEED,
            states_per_trajectory=PAIR280_STATES_PER_TRAJECTORY,
            sampling_algorithm=PAIR280_STATE_ALGORITHM,
        )
        _require(len(offsets) == PAIR280_STATES_PER_TRAJECTORY, "Pair-280 selector returned wrong count")
        anchors.extend(
            PolicyPhysicalStateAnchor(
                task=group.task,
                content_id=group.content_id,
                trajectory_id=group.trajectory_id,
                frame_offset=offset,
            )
            for offset in offsets
        )
    _require(len(anchors) == PAIR280_GROUPS, "Pair-280 state count is not 25,200")
    return {
        "schema": POLICY_STATE_BANK_SCHEMA,
        "schema_version": POLICY_STATE_BANK_SCHEMA_VERSION,
        **pair280_protocol_metadata(),
        "paired_action_manifest_sha256": verified.sha256,
        "paired_action_audit_sha256": verified.audit_sha256,
        "sampling": {
            "algorithm": PAIR280_STATE_ALGORITHM,
            "version": PAIR280_STATE_ALGORITHM_VERSION,
            "seed": PAIR280_STATE_SEED,
            "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
            "endpoint_rule": "33_state_frames_and_32_actions_without_padding",
            "short_trajectory_policy": "fail_closed",
        },
        "physical_state_inventory_sha256": physical_state_inventory_sha256(anchors),
        "states": [
            {**anchor.as_dict(), "physical_state_id": anchor.physical_state_id}
            for anchor in anchors
        ],
        "pair280_contract": {
            "profile_id": PAIR280_PROFILE_ID,
            "train_trajectory_count": PAIR280_TRAIN_TRAJECTORIES,
            "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
            "physical_state_groups": PAIR280_GROUPS,
            "scene_views": PAIR280_VIEWS,
            "minimum_valid_anchor_count": minimum_valid,
        },
    }


def verify_pair280_state_bank(
    state_bank_path: str | Path,
    *,
    paired_root: str | Path,
    paired_manifest: str | Path,
    paired_audit: str | Path,
    expected_sha256: str | None = None,
) -> VerifiedPolicyStateBank:
    verified_manifest = verify_native_paired_action_manifest(
        paired_manifest,
        dataset_root=paired_root,
        audit_path=paired_audit,
    )
    try:
        bank = verify_policy_state_bank(
            state_bank_path,
            native_manifest=verified_manifest,
            expected_sha256=expected_sha256,
            expected_tasks=OFFICIAL_TASKS,
            expected_states_per_trajectory=PAIR280_STATES_PER_TRAJECTORY,
            expected_sampling_algorithm=PAIR280_STATE_ALGORITHM,
            expected_sampling_version=PAIR280_STATE_ALGORITHM_VERSION,
            expected_sampling_seed=PAIR280_STATE_SEED,
        )
    except DataContractError as exc:
        raise Pair280ContractError(str(exc)) from exc
    _require(len(bank.anchors) == PAIR280_GROUPS, "Pair-280 verified state count changed")
    return bank


def build_pair280_release_binding(
    *,
    parent_binding_path: str | Path,
    state_bank_path: str | Path,
    paired_root: str | Path,
    paired_manifest: str | Path,
    paired_audit: str | Path,
) -> dict[str, Any]:
    parent = verify_release_paired_binding(parent_binding_path)
    _require(int(parent["schema_version"]) == 1, "Pair-280 parent must be the audited schema-v1 binding")
    bank = verify_pair280_state_bank(
        state_bank_path,
        paired_root=paired_root,
        paired_manifest=paired_manifest,
        paired_audit=paired_audit,
    )
    parent_identity = dict(parent["binding_manifest_identity"])
    result = deepcopy(parent)
    result.pop("binding_manifest_identity", None)
    result["schema_version"] = PAIR280_BINDING_SCHEMA_VERSION
    result["kind"] = BINDING_KIND
    result["parent_binding"] = parent_identity
    result["paired_dataset"]["state_bank_anchor_count"] = PAIR280_GROUPS
    result["paired_dataset"]["state_bank_sha256"] = bank.sha256
    result["paired_dataset"]["physical_state_inventory_sha256"] = (
        bank.physical_state_inventory_sha256
    )
    result["meta_artifacts"]["policy_paired_state_bank"] = artifact_identity(
        bank.path
    )
    result["cache_protocol"] = {
        "capture_layer": 16,
        "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
        "physical_state_groups": PAIR280_GROUPS,
        "scene_views": PAIR280_VIEWS,
        "view_token_shape": list(PAIR280_TOKEN_SHAPE),
        "storage": PAIR280_CACHE_STORAGE,
    }
    result["pair280_contract"] = {
        "profile_id": PAIR280_PROFILE_ID,
        "paired_epochs": PAIR280_EPOCHS,
        "paired_active_steps": PAIR280_ACTIVE_STEPS,
        "total_optimizer_steps": PAIR280_TOTAL_STEPS,
        "uniform_active_schedule": "floor_difference_v1",
    }
    return result


def paired_active_count(completed_steps: int) -> int:
    completed = int(completed_steps)
    _require(0 <= completed <= PAIR280_TOTAL_STEPS, "completed step outside Pair-280 run")
    return (completed * PAIR280_ACTIVE_STEPS) // PAIR280_TOTAL_STEPS


def paired_is_active(step: int) -> bool:
    one_based = int(step)
    _require(1 <= one_based <= PAIR280_TOTAL_STEPS, "step outside Pair-280 run")
    return paired_active_count(one_based) > paired_active_count(one_based - 1)


def paired_active_index(step: int) -> int | None:
    if not paired_is_active(step):
        return None
    return paired_active_count(int(step)) - 1


def audit_pair280_active_schedule() -> dict[str, Any]:
    active = [step for step in range(1, PAIR280_TOTAL_STEPS + 1) if paired_is_active(step)]
    _require(len(active) == PAIR280_ACTIVE_STEPS, "Pair-280 active-step count changed")
    gaps = [right - left for left, right in zip(active, active[1:])]
    _require(set(gaps).issubset({1, 2}), "Pair-280 active steps are not uniformly spread")
    _require(paired_active_count(PAIR280_TOTAL_STEPS) == PAIR280_ACTIVE_STEPS, "active prefix count changed")
    return {
        "status": "PASS",
        "profile_id": PAIR280_PROFILE_ID,
        "total_steps": PAIR280_TOTAL_STEPS,
        "active_steps": len(active),
        "inactive_steps": PAIR280_INACTIVE_STEPS,
        "first_active_step": active[0],
        "last_active_step": active[-1],
        "gap_counts": {str(gap): gaps.count(gap) for gap in sorted(set(gaps))},
        "schedule_sha256": canonical_json_sha256(active),
    }


class ExactPair280BatchSampler(Sampler[list[int]]):
    """Emit exactly ten no-replacement passes over all 25,200 states.

    Each emitted list is one rank-local batch of two states from the same task
    and from two distinct physical trajectories.  Accelerate consumes eight
    consecutive lists as one global optimizer step.
    """

    def __init__(self, dataset: Dataset, *, seed: int) -> None:
        self.seed = int(seed)
        tasks = getattr(dataset, "indices_by_task", None)
        trajectory_lookup = getattr(dataset, "trajectory_id_for_index", None)
        _require(isinstance(tasks, Mapping), "Pair-280 dataset lacks indices_by_task")
        _require(callable(trajectory_lookup), "Pair-280 dataset lacks trajectory lookup")
        by_task_trajectory: dict[str, dict[str, list[int]]] = {}
        for task, indices in tasks.items():
            grouped: dict[str, list[int]] = defaultdict(list)
            for index in indices:
                grouped[str(trajectory_lookup(int(index)))].append(int(index))
            _require(len(grouped) == 30, f"Pair-280 task {task} must contain 30 trajectories")
            _require(
                all(len(values) == PAIR280_STATES_PER_TRAJECTORY for values in grouped.values()),
                f"Pair-280 task {task} trajectory state counts changed",
            )
            by_task_trajectory[str(task)] = dict(sorted(grouped.items()))
        _require(tuple(sorted(by_task_trajectory)) == tuple(sorted(OFFICIAL_TASKS)), "Pair-280 task set changed")
        self._by_task_trajectory = by_task_trajectory

    def __len__(self) -> int:
        return PAIR280_GROUPS * PAIR280_EPOCHS // PAIR280_LOCAL_GROUPS

    def _task_pairs(self, task: str, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed * 10_000_019 + epoch * 1_000_003 + OFFICIAL_TASKS.index(task))
        trajectories = list(self._by_task_trajectory[task])
        state_lists = {
            trajectory: list(self._by_task_trajectory[task][trajectory])
            for trajectory in trajectories
        }
        for values in state_lists.values():
            rng.shuffle(values)
        pairs: list[list[int]] = []
        for offset_rank in range(PAIR280_STATES_PER_TRAJECTORY):
            order = list(trajectories)
            rng.shuffle(order)
            for start in range(0, len(order), 2):
                left, right = order[start : start + 2]
                _require(left != right, "Pair-280 local batch repeats a trajectory")
                pairs.append([state_lists[left][offset_rank], state_lists[right][offset_rank]])
        _require(len(pairs) == 4_200, f"Pair-280 task {task} local-batch count changed")
        return pairs

    def __iter__(self) -> Iterator[list[int]]:
        for epoch in range(PAIR280_EPOCHS):
            task_pairs = {task: self._task_pairs(task, epoch) for task in OFFICIAL_TASKS}
            positions = {task: 0 for task in OFFICIAL_TASKS}
            labels: list[str] = []
            label_rng = random.Random(self.seed * 97_409 + epoch)
            for _ in range(4_200):
                order = list(OFFICIAL_TASKS)
                label_rng.shuffle(order)
                labels.extend(order)
            _require(len(labels) == 12_600, "Pair-280 epoch local-batch count changed")
            for task in labels:
                position = positions[task]
                yield task_pairs[task][position]
                positions[task] += 1
            _require(set(positions.values()) == {4_200}, "Pair-280 task schedule is imbalanced")


def audit_exact_pair280_sampler(dataset: Dataset, *, seed: int) -> dict[str, Any]:
    sampler = ExactPair280BatchSampler(dataset, seed=seed)
    trajectory_lookup = getattr(dataset, "trajectory_id_for_index")
    state_lookup = getattr(dataset, "physical_state_id_for_index")
    epoch_batch_count = PAIR280_GROUPS // PAIR280_LOCAL_GROUPS
    counts: Counter[int] = Counter()
    epoch_inventory_digests: list[str] = []
    epoch_values: list[str] = []
    for batch_index, batch in enumerate(sampler):
        _require(len(batch) == 2 and batch[0] != batch[1], "Pair-280 batch is not two distinct states")
        _require(
            trajectory_lookup(batch[0]) != trajectory_lookup(batch[1]),
            "Pair-280 local negatives share one physical trajectory",
        )
        counts.update(batch)
        epoch_values.extend(str(state_lookup(index)) for index in batch)
        if (batch_index + 1) % epoch_batch_count == 0:
            _require(len(epoch_values) == PAIR280_GROUPS, "Pair-280 epoch exposure count changed")
            _require(len(set(epoch_values)) == PAIR280_GROUPS, "Pair-280 epoch is not no-replacement")
            epoch_inventory_digests.append(canonical_json_sha256(epoch_values))
            epoch_values = []
    _require(len(counts) == PAIR280_GROUPS, "Pair-280 sampler did not cover every state")
    _require(set(counts.values()) == {PAIR280_EPOCHS}, "Pair-280 state exposure is not exactly ten")
    return {
        "status": "PASS",
        "profile_id": PAIR280_PROFILE_ID,
        "seed": int(seed),
        "physical_state_groups": len(counts),
        "paired_epochs": PAIR280_EPOCHS,
        "local_batches": len(sampler),
        "global_active_steps": len(sampler) // PAIR280_WORLD_SIZE,
        "exposures_per_state": PAIR280_EPOCHS,
        "replacement_within_epoch": False,
        "same_trajectory_local_negatives": False,
        "epoch_order_sha256": epoch_inventory_digests,
    }


def validate_pair280_cache_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_state_bank_sha256: str | None = None,
    expected_release_binding_sha256: str | None = None,
    expected_extraction_contract: Mapping[str, Any] | None = None,
    verify_shard_hashes: bool = False,
) -> dict[str, Any]:
    value, path = _load_json(manifest_path, "Pair-280 cache manifest")
    identity = artifact_identity(path)
    if expected_manifest_sha256 is not None:
        _require(identity["sha256"] == expected_manifest_sha256, "Pair-280 cache manifest SHA changed")
    _require(value.get("schema") == PAIR280_CACHE_SCHEMA, "Pair-280 cache schema changed")
    _require(int(value.get("schema_version", -1)) == PAIR280_CACHE_SCHEMA_VERSION, "Pair-280 cache version changed")
    _require(value.get("status") == "PASS", "Pair-280 cache status is not PASS")
    _require(value.get("profile_id") == PAIR280_PROFILE_ID, "Pair-280 profile id changed")
    _require(int(value.get("physical_state_groups", -1)) == PAIR280_GROUPS, "Pair-280 cache group count changed")
    _require(int(value.get("scene_views", -1)) == PAIR280_VIEWS, "Pair-280 cache view count changed")
    _require(value.get("storage") == PAIR280_CACHE_STORAGE, "Pair-280 cache storage changed")
    _require(value.get("token_shape") == [4, *PAIR280_TOKEN_SHAPE], "Pair-280 token shape changed")
    _require(value.get("token_dtype") == "torch.bfloat16", "Pair-280 token dtype changed")
    if expected_state_bank_sha256 is not None:
        _require(value.get("state_bank", {}).get("sha256") == expected_state_bank_sha256, "Pair-280 cache state bank changed")
    if expected_release_binding_sha256 is not None:
        _require(value.get("release_paired_binding", {}).get("sha256") == expected_release_binding_sha256, "Pair-280 cache release binding changed")
    extraction_contract = value.get("extraction_contract")
    _require(isinstance(extraction_contract, Mapping), "Pair-280 extraction contract is missing")
    if expected_extraction_contract is not None:
        _require(
            dict(extraction_contract) == dict(expected_extraction_contract),
            "Pair-280 extraction dependencies differ from the current runtime",
        )
    shards = value.get("shards")
    _require(isinstance(shards, list) and len(shards) == PAIR280_TRAIN_TRAJECTORIES, "Pair-280 cache must contain 90 shards")
    seen_states: list[str] = []
    total_bytes = 0
    root = path.parent
    for shard_index, shard in enumerate(shards):
        _require(isinstance(shard, Mapping), f"Pair-280 shard {shard_index} metadata missing")
        _require(int(shard.get("state_count", -1)) == PAIR280_STATES_PER_TRAJECTORY, "Pair-280 shard state count changed")
        state_ids = shard.get("physical_state_ids")
        _require(isinstance(state_ids, list) and len(state_ids) == PAIR280_STATES_PER_TRAJECTORY, "Pair-280 shard state ids changed")
        seen_states.extend(str(value) for value in state_ids)
        for key in ("tensor_file", "metadata_file"):
            artifact = shard.get(key)
            _require(isinstance(artifact, Mapping), f"Pair-280 shard lacks {key}")
            relative = Path(str(artifact.get("relative_path", "")))
            _require(not relative.is_absolute() and ".." not in relative.parts, "Pair-280 shard path is unsafe")
            actual = root / relative
            _require(actual.is_file() and not actual.is_symlink(), f"Pair-280 shard file missing/unsafe: {actual}")
            stat_size = int(actual.stat().st_size)
            _require(stat_size == int(artifact.get("size_bytes", -1)), f"Pair-280 shard size changed: {actual}")
            _require(_valid_sha256(artifact.get("sha256")), "Pair-280 shard SHA is invalid")
            if verify_shard_hashes:
                _require(_sha256(actual) == artifact["sha256"], f"Pair-280 shard bytes changed: {actual}")
            total_bytes += stat_size
    _require(len(seen_states) == len(set(seen_states)) == PAIR280_GROUPS, "Pair-280 cache state inventory is not unique")
    _require(canonical_json_sha256(seen_states) == value.get("ordered_state_ids_sha256"), "Pair-280 ordered state inventory changed")
    _require(total_bytes == int(value.get("payload_size_bytes", -1)), "Pair-280 payload size changed")
    return {**value, "manifest_identity": identity, "manifest_path": str(path)}


class ShardedPair280TokenDataset(Dataset[dict[str, Any]]):
    """Lazy mmap-backed reader for trajectory-sharded Pair-280 tokens."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        state_bank: VerifiedPolicyStateBank,
        expected_manifest_sha256: str,
        expected_release_binding_sha256: str,
        expected_extraction_contract: Mapping[str, Any],
        verify_shard_hashes: bool,
    ) -> None:
        self._manifest = validate_pair280_cache_manifest(
            manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_state_bank_sha256=state_bank.sha256,
            expected_release_binding_sha256=expected_release_binding_sha256,
            expected_extraction_contract=expected_extraction_contract,
            verify_shard_hashes=verify_shard_hashes,
        )
        self.cache_path = Path(self._manifest["manifest_path"])
        self._state_bank = state_bank
        self._records: list[dict[str, Any]] = []
        self._indices_by_task: dict[str, list[int]] = defaultdict(list)
        shards = self._manifest["shards"]
        by_state: dict[str, tuple[int, int]] = {}
        for shard_index, shard in enumerate(shards):
            for local_index, state_id in enumerate(shard["physical_state_ids"]):
                by_state[str(state_id)] = (shard_index, local_index)
        for index, anchor in enumerate(state_bank.anchors):
            _require(anchor.physical_state_id in by_state, f"Pair-280 cache lacks {anchor.physical_state_id}")
            shard_index, local_index = by_state[anchor.physical_state_id]
            self._records.append(
                {
                    "shard_index": shard_index,
                    "local_index": local_index,
                    "task": anchor.task,
                    "physical_state_id": anchor.physical_state_id,
                    "trajectory_id": anchor.trajectory_id,
                    "content_id": anchor.content_id,
                    "frame_offset": anchor.frame_offset,
                }
            )
            self._indices_by_task[anchor.task].append(index)
        _require(len(self._records) == PAIR280_GROUPS, "Pair-280 runtime dataset count changed")
        self._handles: OrderedDict[int, Any] = OrderedDict()
        self._max_open_handles = 24

    @property
    def indices_by_task(self) -> dict[str, tuple[int, ...]]:
        return {task: tuple(values) for task, values in self._indices_by_task.items()}

    @property
    def token_shape(self) -> tuple[int, int]:
        return PAIR280_TOKEN_SHAPE

    @property
    def token_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def __len__(self) -> int:
        return len(self._records)

    def physical_state_id_for_index(self, index: int) -> str:
        return str(self._records[index]["physical_state_id"])

    def trajectory_id_for_index(self, index: int) -> str:
        return str(self._records[index]["trajectory_id"])

    def _handle(self, shard_index: int):
        if shard_index in self._handles:
            handle = self._handles.pop(shard_index)
            self._handles[shard_index] = handle
            return handle
        shard = self._manifest["shards"][shard_index]
        path = self.cache_path.parent / shard["tensor_file"]["relative_path"]
        handle = safe_open(path, framework="pt", device="cpu")
        self._handles[shard_index] = handle
        while len(self._handles) > self._max_open_handles:
            self._handles.popitem(last=False)
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        normalized = index if index >= 0 else len(self) + index
        row = self._records[normalized]
        handle = self._handle(int(row["shard_index"]))
        local = int(row["local_index"])
        tokens = handle.get_slice("tokens")[local]
        proprio = handle.get_slice("proprio_raw")[local]
        _require(tuple(tokens.shape) == (4, *PAIR280_TOKEN_SHAPE), "Pair-280 token slice shape changed")
        return {
            "tokens": tokens,
            "variant_names": POLICY_VARIANTS,
            "protocol_id": POLICY_PROTOCOL_ID,
            "r3_role": POLICY_R3_ROLE,
            "task": row["task"],
            "physical_state_id": row["physical_state_id"],
            "trajectory_id": row["trajectory_id"],
            "content_id": row["content_id"],
            "frame_offset": row["frame_offset"],
            "split": "train",
            "layer": 16,
            "dataset_index": normalized,
            "record_indices": (),
            "records": tuple(
                {
                    "task": row["task"],
                    "physical_state_id": row["physical_state_id"],
                    "trajectory_id": row["trajectory_id"],
                    "content_id": row["content_id"],
                    "frame_offset": row["frame_offset"],
                    "split": "train",
                    "variant": variant,
                    "view_index": view_index,
                }
                for view_index, variant in enumerate(POLICY_VARIANTS)
            ),
            "physical_state": deepcopy(row),
            "proprio_raw": proprio,
            "condition_provenance": None,
        }


def build_protocol_manifest() -> dict[str, Any]:
    _require(PAIR280_GROUPS * PAIR280_EPOCHS % PAIR280_GLOBAL_GROUPS == 0, "paired exposure budget is not integral")
    _require(PAIR280_ACTIVE_STEPS == PAIR280_GROUPS * PAIR280_EPOCHS // PAIR280_GLOBAL_GROUPS, "paired active-step formula changed")
    return {
        "schema_version": 1,
        "kind": "policy_pair280_posttraining_protocol",
        "status": "PASS",
        "profile_id": PAIR280_PROFILE_ID,
        "official": {
            "frame_anchor_samples": 466_240,
            "epochs": 5,
            "global_batch": 128,
            "steps_per_epoch": math.ceil(466_240 / 128),
            "optimizer_steps": PAIR280_TOTAL_STEPS,
            "action_loss_steps": PAIR280_TOTAL_STEPS,
        },
        "paired": {
            "train_physical_trajectories": PAIR280_TRAIN_TRAJECTORIES,
            "states_per_trajectory": PAIR280_STATES_PER_TRAJECTORY,
            "physical_state_groups": PAIR280_GROUPS,
            "scene_views": PAIR280_VIEWS,
            "epochs": PAIR280_EPOCHS,
            "global_groups_per_active_step": PAIR280_GLOBAL_GROUPS,
            "active_steps": PAIR280_ACTIVE_STEPS,
            "inactive_action_only_steps": PAIR280_INACTIVE_STEPS,
            "sampling": "no_replacement_within_epoch_exact10",
        },
        "active_schedule": audit_pair280_active_schedule(),
        "training": {
            "regime": "p_v2",
            "control": "c3_ours",
            "training_seed": 1,
            "world_size": 8,
            "official_local_batch": 16,
            "paired_local_groups": 2,
            "lambda_contrastive": 0.1,
            "save_every_steps": 2_000,
            "exact_resume": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    state = sub.add_parser("build-state-bank")
    state.add_argument("--paired-root", required=True, type=Path)
    state.add_argument("--paired-manifest", required=True, type=Path)
    state.add_argument("--paired-audit", required=True, type=Path)
    state.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify-state-bank")
    verify.add_argument("--paired-root", required=True, type=Path)
    verify.add_argument("--paired-manifest", required=True, type=Path)
    verify.add_argument("--paired-audit", required=True, type=Path)
    verify.add_argument("--state-bank", required=True, type=Path)
    binding = sub.add_parser("build-binding")
    binding.add_argument("--parent-binding", required=True, type=Path)
    binding.add_argument("--paired-root", required=True, type=Path)
    binding.add_argument("--paired-manifest", required=True, type=Path)
    binding.add_argument("--paired-audit", required=True, type=Path)
    binding.add_argument("--state-bank", required=True, type=Path)
    binding.add_argument("--output", required=True, type=Path)
    protocol = sub.add_parser("write-protocol")
    protocol.add_argument("--output", required=True, type=Path)
    cache = sub.add_parser("verify-cache")
    cache.add_argument("--manifest", required=True, type=Path)
    cache.add_argument("--verify-shard-hashes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build-state-bank":
            payload = build_pair280_state_bank(
                paired_root=args.paired_root,
                paired_manifest=args.paired_manifest,
                paired_audit=args.paired_audit,
            )
            _write_new_json(args.output, payload)
            verified = verify_pair280_state_bank(
                args.output,
                paired_root=args.paired_root,
                paired_manifest=args.paired_manifest,
                paired_audit=args.paired_audit,
            )
            result = {"status": "PASS", "path": str(verified.path), "sha256": verified.sha256, "groups": len(verified.anchors)}
        elif args.command == "verify-state-bank":
            verified = verify_pair280_state_bank(
                args.state_bank,
                paired_root=args.paired_root,
                paired_manifest=args.paired_manifest,
                paired_audit=args.paired_audit,
            )
            result = {"status": "PASS", "path": str(verified.path), "sha256": verified.sha256, "groups": len(verified.anchors)}
        elif args.command == "build-binding":
            payload = build_pair280_release_binding(
                parent_binding_path=args.parent_binding,
                state_bank_path=args.state_bank,
                paired_root=args.paired_root,
                paired_manifest=args.paired_manifest,
                paired_audit=args.paired_audit,
            )
            _write_new_json(args.output, payload)
            result = verify_release_paired_binding(args.output)
        elif args.command == "write-protocol":
            _write_new_json(args.output, build_protocol_manifest())
            result, _ = _load_json(args.output, "Pair-280 protocol")
        else:
            result = validate_pair280_cache_manifest(
                args.manifest, verify_shard_hashes=args.verify_shard_hashes
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"Pair-280 failed closed: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
