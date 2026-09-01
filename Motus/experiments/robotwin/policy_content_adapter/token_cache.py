"""Sharded frozen WAN-token cache for Motus paired observations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .paired_data import sha256_file
from .protocol import (
    DEFAULT_BACKBONE_DIM,
    PAIRED_STATE_COUNT,
    PAIRED_VIEW_COUNT,
    PROTOCOL_ID,
    TASKS,
)


CACHE_SCHEMA = "motus_policy_frozen_observation_token_cache"
CACHE_VERSION = 1


class TokenCacheError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenCacheError(message)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FrozenTokenCacheWriter:
    """Create an append-in-memory/sharded-on-disk cache, then atomically seal it."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        paired_manifest_identity: Mapping[str, Any],
        base_lineage_identity: Mapping[str, Any],
        capture_layer: int,
        shard_groups: int = 16,
        expected_groups: int = PAIRED_STATE_COUNT,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite token cache {self.output_dir}")
        if capture_layer <= 0 or shard_groups <= 0 or expected_groups <= 0:
            raise ValueError("cache layer/count values must be positive")
        for name, identity in (
            ("paired manifest", paired_manifest_identity),
            ("base lineage", base_lineage_identity),
        ):
            if not _valid_sha(identity.get("sha256")):
                raise ValueError(f"{name} identity has no valid SHA")
            if int(identity.get("size_bytes", -1)) < 0:
                raise ValueError(f"{name} identity has no valid size")
        self.paired_manifest_identity = dict(paired_manifest_identity)
        self.base_lineage_identity = dict(base_lineage_identity)
        self.capture_layer = int(capture_layer)
        self.shard_groups = int(shard_groups)
        self.expected_groups = int(expected_groups)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}.staging-",
                dir=self.output_dir.parent,
            )
        )
        self._tokens: list[torch.Tensor] = []
        self._state_ids: list[str] = []
        self._tasks: list[str] = []
        self._seen: set[str] = set()
        self._shards: list[dict[str, Any]] = []
        self._group_count = 0
        self._token_shape: tuple[int, int] | None = None
        self._closed = False

    def add(
        self,
        visual_tokens: torch.Tensor,
        physical_state_ids: Sequence[str],
        task_ids: Sequence[str],
    ) -> None:
        if self._closed:
            raise TokenCacheError("cache writer is already closed")
        if visual_tokens.ndim != 4 or visual_tokens.shape[1] != PAIRED_VIEW_COUNT:
            raise ValueError("visual_tokens must be [G,4,L,D]")
        groups, _, token_count, feature_dim = visual_tokens.shape
        if groups <= 0 or feature_dim != DEFAULT_BACKBONE_DIM:
            raise ValueError("visual token dimensions changed")
        if len(physical_state_ids) != groups or len(task_ids) != groups:
            raise ValueError("cache labels do not match group batch")
        if not torch.is_floating_point(visual_tokens):
            raise TypeError("visual tokens must be floating point")
        if not bool(torch.isfinite(visual_tokens).all().item()):
            raise ValueError("visual tokens are non-finite")
        shape = (int(token_count), int(feature_dim))
        if self._token_shape is None:
            self._token_shape = shape
        elif shape != self._token_shape:
            raise ValueError("visual token shape changed between batches")
        for index, (state_id, task) in enumerate(
            zip(physical_state_ids, task_ids, strict=True)
        ):
            state_id = str(state_id)
            task = str(task)
            if not state_id or state_id in self._seen:
                raise ValueError("physical state ids must be unique and non-empty")
            if task not in TASKS:
                raise ValueError(f"unexpected task {task!r}")
            self._seen.add(state_id)
            self._state_ids.append(state_id)
            self._tasks.append(task)
            self._tokens.append(visual_tokens[index].detach().cpu().to(torch.bfloat16))
            self._group_count += 1
            if len(self._tokens) == self.shard_groups:
                self._flush()

    def _flush(self) -> None:
        if not self._tokens:
            return
        shard_index = len(self._shards)
        path = self.staging / f"shard_{shard_index:05d}.pt"
        first_group = self._group_count - len(self._tokens)
        payload = {
            "schema": CACHE_SCHEMA,
            "schema_version": CACHE_VERSION,
            "first_group_index": first_group,
            "visual_tokens": torch.stack(self._tokens, dim=0),
            "physical_state_ids": self._state_ids[-len(self._tokens) :],
            "task_ids": self._tasks[-len(self._tokens) :],
        }
        torch.save(payload, path)
        self._shards.append(
            {
                "name": path.name,
                "first_group_index": first_group,
                "group_count": len(self._tokens),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self._tokens.clear()

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise TokenCacheError("cache writer is already closed")
        try:
            self._flush()
            _require(
                self._group_count == self.expected_groups,
                f"expected {self.expected_groups} groups, got {self._group_count}",
            )
            _require(self._token_shape is not None, "cache contains no tokens")
            per_task = {task: self._tasks.count(task) for task in TASKS}
            if self.expected_groups == PAIRED_STATE_COUNT:
                _require(
                    per_task == {task: 240 for task in TASKS},
                    "formal cache per-task counts changed",
                )
            index = [
                {"physical_state_id": state, "task": task, "group_index": index}
                for index, (state, task) in enumerate(
                    zip(self._state_ids, self._tasks, strict=True)
                )
            ]
            manifest = {
                "schema": CACHE_SCHEMA,
                "schema_version": CACHE_VERSION,
                "status": "PASS",
                "protocol_id": PROTOCOL_ID,
                "capture": {
                    "branch": "current_observation_frozen_video_only_wan_t0",
                    "layer": self.capture_layer,
                    "dtype": "torch.bfloat16",
                    "view_token_shape": list(self._token_shape),
                    "views_per_state": PAIRED_VIEW_COUNT,
                },
                "counts": {
                    "physical_states": self._group_count,
                    "scene_views": self._group_count * PAIRED_VIEW_COUNT,
                    "per_task_physical_states": per_task,
                    "shards": len(self._shards),
                },
                "paired_manifest_identity": self.paired_manifest_identity,
                "base_lineage_identity": self.base_lineage_identity,
                "shards": self._shards,
                "index_sha256": _canonical_sha(index),
                "index": index,
            }
            manifest_path = self.staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(self.staging, self.output_dir)
            self._closed = True
            return manifest
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if not self._closed:
            shutil.rmtree(self.staging, ignore_errors=True)
            self._closed = True


def validate_token_cache(
    cache_dir: str | Path,
    *,
    expected_paired_manifest_sha256: str | None = None,
    expected_base_lineage_sha256: str | None = None,
    verify_shards: bool = True,
) -> dict[str, Any]:
    root = Path(cache_dir).resolve()
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "token cache manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == CACHE_SCHEMA, "token cache schema changed")
    _require(manifest.get("schema_version") == CACHE_VERSION, "token cache version changed")
    _require(manifest.get("status") == "PASS", "token cache is not PASS")
    paired_sha = manifest.get("paired_manifest_identity", {}).get("sha256")
    lineage_sha = manifest.get("base_lineage_identity", {}).get("sha256")
    if expected_paired_manifest_sha256 is not None:
        _require(paired_sha == expected_paired_manifest_sha256, "paired manifest ancestry changed")
    if expected_base_lineage_sha256 is not None:
        _require(lineage_sha == expected_base_lineage_sha256, "base lineage ancestry changed")
    index = manifest.get("index")
    _require(isinstance(index, list), "token cache index is missing")
    _require(_canonical_sha(index) == manifest.get("index_sha256"), "token cache index SHA changed")
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and shards, "token cache has no shards")
    group_total = 0
    for shard in shards:
        path = root / str(shard.get("name", ""))
        _require(path.is_file(), f"token shard is missing: {path}")
        _require(path.stat().st_size == int(shard.get("size_bytes", -1)), "token shard size changed")
        if verify_shards:
            _require(sha256_file(path) == shard.get("sha256"), "token shard SHA changed")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            _require(payload.get("schema") == CACHE_SCHEMA, "token shard schema changed")
            tokens = payload.get("visual_tokens")
            _require(isinstance(tokens, torch.Tensor) and tokens.ndim == 4, "token shard payload changed")
            _require(tokens.shape[1] == PAIRED_VIEW_COUNT and tokens.shape[-1] == DEFAULT_BACKBONE_DIM, "token shard shape changed")
            _require(tokens.dtype == torch.bfloat16 and bool(torch.isfinite(tokens).all()), "token shard values changed")
        group_total += int(shard.get("group_count", -1))
    _require(group_total == len(index), "token shard/index counts differ")
    counts = manifest.get("counts", {})
    _require(group_total == int(counts.get("physical_states", -1)), "token cache group count changed")
    return {
        "status": "PASS",
        "physical_states": group_total,
        "scene_views": group_total * PAIRED_VIEW_COUNT,
        "shards": len(shards),
        "manifest_sha256": sha256_file(manifest_path),
    }


class FrozenMotusTokenDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_cached_shards: int = 2,
        verify_shards: bool = False,
    ) -> None:
        self.root = Path(cache_dir).resolve()
        validate_token_cache(self.root, verify_shards=verify_shards)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.index = self.manifest["index"]
        self.shards = self.manifest["shards"]
        self.max_cached_shards = int(max_cached_shards)
        if self.max_cached_shards <= 0:
            raise ValueError("max_cached_shards must be positive")
        self._cache: OrderedDict[int, Mapping[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _shard_for(self, index: int) -> tuple[int, int]:
        for shard_index, shard in enumerate(self.shards):
            first = int(shard["first_group_index"])
            count = int(shard["group_count"])
            if first <= index < first + count:
                return shard_index, index - first
        raise IndexError(index)

    def _load_shard(self, shard_index: int) -> Mapping[str, Any]:
        payload = self._cache.pop(shard_index, None)
        if payload is None:
            payload = torch.load(
                self.root / self.shards[shard_index]["name"],
                map_location="cpu",
                weights_only=False,
            )
        self._cache[shard_index] = payload
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, offset = self._shard_for(index)
        payload = self._load_shard(shard_index)
        meta = self.index[index]
        if payload["physical_state_ids"][offset] != meta["physical_state_id"]:
            raise TokenCacheError("token shard/index physical state mismatch")
        if payload["task_ids"][offset] != meta["task"]:
            raise TokenCacheError("token shard/index task mismatch")
        return {
            "visual_tokens": payload["visual_tokens"][offset],
            "physical_state_id": meta["physical_state_id"],
            "task": meta["task"],
        }

