"""Fail-closed Policy v2 paired-data plumbing.

Policy v2 keeps the official action stream and the paired stream independent.
The paired stream always represents one physical state rendered as the ordered
``(C, R1, R2, R3)`` scene versions.  Every scene version still contains the
same three cameras.  R3 is a training positive; it is never interpreted as a
Policy evaluation domain here.

Two paired supervision modes are supported:

* C2 ``action`` wraps a native 50 Hz LeRobot/FastWAM dataset and returns four
  processed samples with one exact 32x14 action target; and
* C3 ``contrastive`` reads a Policy-only frozen Layer-16 token cache.

No representation-level E0--E3 protocol is imported, and no temporal
interpolation or conversion path exists in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from .protocol import (
    POLICY_ACTION_DIM,
    POLICY_ACTION_MANIFEST_SCHEMA,
    POLICY_ACTION_MANIFEST_VERSION,
    POLICY_ACTION_STEPS,
    POLICY_CAMERA_COUNT,
    POLICY_CAMERA_NAMES,
    POLICY_CONTENTS_PER_TASK_BY_SPLIT,
    POLICY_DATA_SPLITS,
    POLICY_NATIVE_FPS,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_STATE_BANK_SAMPLING_ALGORITHM,
    POLICY_STATE_BANK_SAMPLING_VERSION,
    POLICY_STATE_BANK_SCHEMA,
    POLICY_STATE_BANK_SCHEMA_VERSION,
    POLICY_STATE_BANK_SEED,
    POLICY_STATE_STEPS,
    POLICY_STATES_PER_TRAJECTORY,
    POLICY_TEMPORAL_RESAMPLING,
    POLICY_TOKEN_CACHE_SCHEMA,
    POLICY_TOKEN_CACHE_SCHEMA_VERSION,
    POLICY_TRAIN_SPLITS,
    POLICY_VARIANTS,
    POLICY_VIEW_COUNT,
    PolicyProtocolError,
    policy_split_for_content_id,
    validate_policy_protocol_metadata,
)


class DataContractError(ValueError):
    """An input cannot prove the Policy v2 data contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataContractError(message)


def _canonical_layer(layer: int) -> tuple[int, str]:
    value = int(layer)
    _require(value > 0, f"layer must be positive, got {layer!r}")
    return value, str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_artifact_binding(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Project a file/directory identity onto immutable content fields."""

    _require(isinstance(identity, Mapping), "artifact identity must be a mapping")
    kind = str(identity.get("kind", ""))
    _require(kind in {"file", "directory"}, "artifact identity kind is invalid")
    digest = str(identity.get("sha256", ""))
    _require(_valid_sha256(digest), "artifact identity SHA-256 is invalid")
    result = {
        "kind": kind,
        "size_bytes": int(identity.get("size_bytes", -1)),
        "sha256": digest,
    }
    _require(result["size_bytes"] >= 0, "artifact identity size is invalid")
    if kind == "directory":
        result["file_count"] = int(identity.get("file_count", -1))
        _require(result["file_count"] > 0, "directory artifact file_count is invalid")
    return result


def policy_cache_extractor_config(
    *,
    states_per_trajectory: int = POLICY_STATES_PER_TRAJECTORY,
    state_selection_algorithm: str = POLICY_STATE_BANK_SAMPLING_ALGORITHM,
    state_selection_seed: int = POLICY_STATE_BANK_SEED,
    storage: str | None = None,
) -> dict[str, Any]:
    result = {
        "capture_layer": 16,
        "token_shape_per_scene": [120, 3072],
        "model_dtype": "torch.bfloat16",
        "representation_source": "current_observation_video_prefill",
        "proprio_mode": "observed",
        "context_source": "precomputed_text_embedding_cache",
        "verify_native_prefill_mode": "first_state_only",
        "native_prefill_checked_states": 1,
        "native_prefill_rtol": 0.0,
        "native_prefill_atol": 0.0,
        "state_selection_algorithm": str(state_selection_algorithm),
        "state_selection_seed": int(state_selection_seed),
        "states_per_trajectory": int(states_per_trajectory),
    }
    if storage is not None:
        result["storage"] = str(storage)
    return result


def policy_cache_preprocessing_contract() -> dict[str, Any]:
    return {
        "schema": "policy_cache_preprocessing_v1",
        "camera_names": list(POLICY_CAMERA_NAMES),
        "native_fps": POLICY_NATIVE_FPS,
        "raw_camera_window_shape": [POLICY_CAMERA_COUNT, POLICY_STATE_STEPS, 3, 240, 320],
        "raw_state_window_shape": [POLICY_STATE_STEPS, POLICY_ACTION_DIM],
        "raw_action_window_shape": [POLICY_ACTION_STEPS, POLICY_ACTION_DIM],
        "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
        "padding_allowed": False,
        "current_observation_index": 0,
        "normalized_video_to_uint8": "round((x+1)*127.5)_clamp_0_255",
        "deployment_image_layout": "head_256x320_over_left_right_128x160",
        "proprio_normalization": "selected_base_lineage_dataset_stats",
        "prompt_conditioning": "precomputed_per_task_text_embedding",
    }


def canonical_fastwam_source_binding(audit: Mapping[str, Any]) -> dict[str, Any]:
    _require(audit.get("status") == "PASS", "FastWAM source audit is not PASS")
    files = audit.get("files")
    _require(isinstance(files, Mapping) and files, "FastWAM source file inventory is missing")
    inventory: list[dict[str, Any]] = []
    for relative, identity in sorted(files.items()):
        _require(isinstance(identity, Mapping), f"FastWAM source identity missing for {relative}")
        inventory.append(
            {
                "relative_path": str(relative),
                "size_bytes": int(identity.get("size_bytes", -1)),
                "sha256": str(identity.get("sha256", "")),
            }
        )
    _require(
        all(item["size_bytes"] >= 0 and _valid_sha256(item["sha256"]) for item in inventory),
        "FastWAM source inventory contains an invalid file identity",
    )
    _require(int(audit.get("file_count", -1)) == len(inventory), "FastWAM source count mismatch")
    return {
        "scope": "all_python_files_under_src_fastwam",
        "file_count": len(inventory),
        "inventory_sha256": canonical_json_sha256(inventory),
    }


def build_policy_cache_extraction_contract(
    *,
    base_lineage_identity: Mapping[str, Any],
    release_paired_binding_identity: Mapping[str, Any],
    dataset_stats_identity: Mapping[str, Any],
    vae_identity: Mapping[str, Any],
    text_encoder_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    text_cache_identity: Mapping[str, Any],
    fastwam_source_audit: Mapping[str, Any],
    extractor_source_identity: Mapping[str, Any],
    extractor_support_source_identities: Mapping[str, Mapping[str, Any]],
    selected_episode_artifacts: Mapping[str, Any],
    extractor_config_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extractor_config = (
        policy_cache_extractor_config()
        if extractor_config_override is None
        else dict(extractor_config_override)
    )
    preprocessing = policy_cache_preprocessing_contract()
    selected = dict(selected_episode_artifacts)
    _require(
        selected.get("algorithm") == "relative_path_size_and_bytes_sha256_v1"
        and selected.get("split") == "train"
        and int(selected.get("episode_count", -1)) > 0
        and int(selected.get("file_count", -1)) == int(selected["episode_count"]) * 4
        and _valid_sha256(selected.get("sha256")),
        "selected paired episode artifact aggregate is invalid",
    )
    required_support = {
        "frozen_backbone",
        "runtime_utils",
        "policy_data",
        "policy_protocol",
    }
    _require(
        required_support.issubset(set(extractor_support_source_identities)),
        "extractor support sources must bind backbone, runtime, data, and protocol",
    )
    return {
        "schema": "policy_cache_extraction_contract_v2",
        "runtime_artifacts": {
            "base_lineage": canonical_artifact_binding(base_lineage_identity),
            "release_paired_binding": canonical_artifact_binding(
                release_paired_binding_identity
            ),
            "dataset_stats": canonical_artifact_binding(dataset_stats_identity),
            "vae": canonical_artifact_binding(vae_identity),
            "text_encoder": canonical_artifact_binding(text_encoder_identity),
            "tokenizer": canonical_artifact_binding(tokenizer_identity),
            "text_cache": canonical_artifact_binding(text_cache_identity),
        },
        "fastwam_source": canonical_fastwam_source_binding(fastwam_source_audit),
        "extractor_source": canonical_artifact_binding(extractor_source_identity),
        "extractor_support_sources": {
            name: canonical_artifact_binding(identity)
            for name, identity in sorted(extractor_support_source_identities.items())
        },
        "extractor_config": {
            "value": extractor_config,
            "sha256": canonical_json_sha256(extractor_config),
        },
        "preprocessing_contract": {
            "value": preprocessing,
            "sha256": canonical_json_sha256(preprocessing),
        },
        "selected_episode_artifacts": selected,
    }


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except DataContractError:
        raise
    except Exception as exc:
        raise DataContractError(f"cannot parse JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise DataContractError(f"cannot load Policy token cache {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Policy token cache root must be a mapping: {path}")
    return value


def _protocol_metadata(value: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    try:
        return validate_policy_protocol_metadata(value, split=split)
    except PolicyProtocolError as exc:
        raise DataContractError(str(exc)) from exc


class FrozenPairedTokenDataset(Dataset[dict[str, Any]]):
    """Read a Policy-only frozen cache as four-scene physical-state groups."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        state_bank: VerifiedPolicyStateBank,
        expected_extraction_contract: Mapping[str, Any],
        layer: int = 16,
        split: str = "train",
        expected_backbone_sha256: str | None = None,
        expected_base_lineage_sha256: str | None = None,
        expected_release_paired_binding_sha256: str | None = None,
        expected_action_manifest_sha256: str | None = None,
        expected_action_audit_sha256: str | None = None,
        expected_state_bank_sha256: str | None = None,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser().resolve()
        _require(self.cache_path.is_file(), f"frozen token cache not found: {self.cache_path}")
        _require(
            isinstance(state_bank, VerifiedPolicyStateBank),
            "Policy frozen cache requires a verified shared state bank",
        )
        _require(split in POLICY_TRAIN_SPLITS, f"paired split must be one of {POLICY_TRAIN_SPLITS}")
        self.split = str(split)
        _require(self.split == "train", "formal Policy state bank currently defines train only")
        self.layer, layer_key = _canonical_layer(layer)

        payload = _load_torch_mapping(self.cache_path)
        _require(
            payload.get("schema") == POLICY_TOKEN_CACHE_SCHEMA,
            f"paired cache schema must be {POLICY_TOKEN_CACHE_SCHEMA!r}",
        )
        _require(
            int(payload.get("schema_version", -1)) == POLICY_TOKEN_CACHE_SCHEMA_VERSION,
            f"paired cache schema_version must be {POLICY_TOKEN_CACHE_SCHEMA_VERSION}",
        )
        provenance = payload.get("provenance")
        _require(isinstance(provenance, Mapping), "paired cache provenance is missing")
        protocol = _protocol_metadata(provenance, split=self.split)
        extraction_contract = provenance.get("extraction_contract")
        _require(
            isinstance(expected_extraction_contract, Mapping),
            "current runtime extraction contract is required",
        )
        _require(
            isinstance(extraction_contract, Mapping),
            "paired cache lacks its full extraction contract",
        )
        _require(
            dict(extraction_contract) == dict(expected_extraction_contract),
            "paired cache extraction dependencies differ from the current runtime/data artifacts",
        )
        native_prefill_audit = provenance.get("native_prefill_identity_audit")
        _require(
            isinstance(native_prefill_audit, Mapping)
            and native_prefill_audit.get("status") == "PASS"
            and int(native_prefill_audit.get("checked_states", -1)) == 1
            and native_prefill_audit.get("comparison")
            == "bit_exact_K_and_V_for_every_layer"
            and float(native_prefill_audit.get("rtol", -1.0)) == 0.0
            and float(native_prefill_audit.get("atol", -1.0)) == 0.0,
            "paired cache lacks the first-state bit-exact native prefill audit",
        )
        variants = tuple(str(value) for value in payload.get("variant_names", ()))
        _require(variants == POLICY_VARIANTS, "cache variant_names disagrees with Policy protocol")

        backbone = provenance.get("backbone_checkpoint")
        _require(isinstance(backbone, Mapping), "paired cache lacks backbone_checkpoint identity")
        backbone_sha256 = str(backbone.get("sha256", ""))
        _require(_valid_sha256(backbone_sha256), "paired cache backbone SHA-256 is invalid")
        if expected_backbone_sha256 is not None:
            _require(
                backbone_sha256 == str(expected_backbone_sha256),
                "paired cache was not extracted from the selected base-lineage checkpoint",
            )
        base_lineage = provenance.get("base_lineage_manifest")
        _require(
            isinstance(base_lineage, Mapping),
            "paired cache lacks its author-release base-lineage identity",
        )
        base_lineage_sha256 = str(base_lineage.get("sha256", ""))
        _require(
            _valid_sha256(base_lineage_sha256),
            "paired cache base-lineage SHA-256 is invalid",
        )
        if expected_base_lineage_sha256 is not None:
            _require(
                base_lineage_sha256 == str(expected_base_lineage_sha256),
                "paired cache was not extracted from the selected author-release lineage",
            )
        release_paired_binding = provenance.get("release_paired_binding_manifest")
        _require(
            isinstance(release_paired_binding, Mapping),
            "paired cache lacks its release/paired binding identity",
        )
        release_paired_binding_sha256 = str(
            release_paired_binding.get("sha256", "")
        )
        _require(
            _valid_sha256(release_paired_binding_sha256),
            "paired cache release/paired binding SHA-256 is invalid",
        )
        if expected_release_paired_binding_sha256 is not None:
            _require(
                release_paired_binding_sha256
                == str(expected_release_paired_binding_sha256),
                "paired cache was not extracted from the selected release/paired binding",
            )
        action_manifest_sha256 = str(provenance.get("paired_action_manifest_sha256", ""))
        action_audit_sha256 = str(provenance.get("paired_action_audit_sha256", ""))
        state_bank_sha256 = str(provenance.get("paired_state_bank_sha256", ""))
        inventory_sha256 = str(provenance.get("physical_state_inventory_sha256", ""))
        _require(
            _valid_sha256(action_manifest_sha256),
            "paired cache lacks its native action manifest SHA-256",
        )
        _require(
            _valid_sha256(action_audit_sha256),
            "paired cache lacks its native action audit SHA-256",
        )
        _require(_valid_sha256(state_bank_sha256), "paired cache lacks its state-bank SHA-256")
        _require(
            _valid_sha256(inventory_sha256),
            "paired cache lacks its physical-state inventory SHA-256",
        )
        if expected_action_manifest_sha256 is not None:
            _require(
                action_manifest_sha256 == expected_action_manifest_sha256,
                "paired cache source manifest differs from the selected native paired data",
            )
        if expected_action_audit_sha256 is not None:
            _require(
                action_audit_sha256 == expected_action_audit_sha256,
                "paired cache source audit differs from the selected native paired data",
            )
        _require(
            action_manifest_sha256 == state_bank.native_manifest.sha256,
            "paired cache and shared state bank reference different action manifests",
        )
        _require(
            action_audit_sha256 == state_bank.native_manifest.audit_sha256,
            "paired cache and shared state bank reference different action audits",
        )
        _require(
            state_bank_sha256 == state_bank.sha256,
            "paired cache was not extracted from the selected shared state bank",
        )
        if expected_state_bank_sha256 is not None:
            _require(
                state_bank_sha256 == expected_state_bank_sha256,
                "paired cache state-bank SHA-256 differs from run artifacts",
            )
        _require(
            inventory_sha256 == state_bank.physical_state_inventory_sha256,
            "paired cache physical-state inventory differs from the shared state bank",
        )

        tokens_by_layer = payload.get("tokens_by_layer")
        _require(isinstance(tokens_by_layer, Mapping), "paired cache tokens_by_layer is missing")
        _require(
            layer_key in tokens_by_layer,
            f"layer {self.layer} is absent; available layers are {tuple(sorted(tokens_by_layer))}",
        )
        tokens = tokens_by_layer[layer_key]
        _require(isinstance(tokens, torch.Tensor), f"layer {self.layer} tokens are not a tensor")
        _require(tokens.ndim == 3, f"layer {self.layer} tokens must have shape [N,S,D]")
        _require(tokens.device.type == "cpu", "frozen token cache must load on CPU")
        _require(not tokens.requires_grad, "frozen cache tensor unexpectedly requires gradients")
        _require(bool(torch.isfinite(tokens.float()).all()), "frozen cache contains NaN/inf")

        records = payload.get("records")
        _require(isinstance(records, Sequence), "paired cache records are missing")
        group_width = POLICY_VIEW_COUNT
        _require(len(records) == int(tokens.shape[0]), "record/token count mismatch")
        _require(len(records) % group_width == 0, "record count is not divisible by four scenes")
        group_count = len(records) // group_width
        _require(
            group_count == len(state_bank.anchors),
            "paired cache state count differs from the shared state bank",
        )
        physical_states = payload.get("physical_states")
        proprio_raw = payload.get("proprio_raw")
        _require(
            isinstance(physical_states, Sequence) and len(physical_states) == group_count,
            "physical-state metadata count mismatch",
        )
        _require(
            isinstance(proprio_raw, torch.Tensor) and proprio_raw.shape == (group_count, 14),
            "paired cache proprio_raw must have shape [physical_states,14]",
        )

        groups: list[dict[str, Any]] = []
        indices_by_task: dict[str, list[int]] = defaultdict(list)
        seen_state_ids: set[str] = set()
        invariant_fields = (
            "task",
            "physical_state_id",
            "content_id",
            "frame_offset",
            "split",
            "trajectory_id",
        )
        native_group_by_content = {
            (group.task, group.content_id): group
            for group in state_bank.native_manifest.groups_for_split("train")
        }
        for group_index, expected_anchor in enumerate(state_bank.anchors):
            start = group_index * group_width
            group_records = records[start : start + group_width]
            _require(
                all(isinstance(record, Mapping) for record in group_records),
                f"group {group_index} contains a non-mapping record",
            )
            ordered_variants = tuple(str(record.get("variant")) for record in group_records)
            _require(
                ordered_variants == POLICY_VARIANTS,
                f"group {group_index} is not ordered C/R1/R2/R3: {ordered_variants}",
            )
            reference = group_records[0]
            for record in group_records:
                _require(
                    all(record.get(field) == reference.get(field) for field in invariant_fields),
                    f"group {group_index} scenes do not share one physical-state identity",
                )
                _require(
                    str(record.get("split")) == self.split,
                    f"group {group_index} contains a non-{self.split} record",
                )
            state_id = str(reference.get("physical_state_id", ""))
            _require(bool(state_id), f"group {group_index} has an empty physical_state_id")
            _require(state_id not in seen_state_ids, f"duplicate physical_state_id {state_id!r}")
            seen_state_ids.add(state_id)
            task = str(reference.get("task", ""))
            _require(bool(task), f"group {group_index} has an empty task")
            actual_anchor = PolicyPhysicalStateAnchor(
                task=task,
                content_id=int(reference.get("content_id", -1)),
                trajectory_id=str(reference.get("trajectory_id", "")),
                frame_offset=int(reference.get("frame_offset", -1)),
            )
            _require(
                actual_anchor == expected_anchor,
                f"cache group {group_index} differs from shared state-bank inventory",
            )
            _require(
                state_id == expected_anchor.physical_state_id,
                f"cache group {group_index} physical_state_id differs from state bank",
            )
            native_group = native_group_by_content[(task, expected_anchor.content_id)]
            expected_episodes = native_group.episode_by_variant
            for record, variant in zip(group_records, POLICY_VARIANTS, strict=True):
                _require(
                    int(record.get("episode_index", -1)) == expected_episodes[variant],
                    f"cache group {group_index}/{variant} episode differs from action manifest",
                )
            dataset_index = len(groups)
            groups.append(
                {
                    "start": start,
                    "task": task,
                    "physical_state_id": state_id,
                    "trajectory_id": str(reference.get("trajectory_id", reference.get("content_id"))),
                    "content_id": int(reference["content_id"]),
                    "frame_offset": int(reference["frame_offset"]),
                }
            )
            indices_by_task[task].append(dataset_index)

        record_variants = {str(record.get("variant")) for record in records}
        _require(record_variants == set(POLICY_VARIANTS), "cache records are not exact C/R1/R2/R3")

        self._tokens = tokens.detach()
        self._proprio_raw = proprio_raw.detach()
        self._physical_states = tuple(deepcopy(value) for value in physical_states)
        self._records = tuple(deepcopy(value) for value in records)
        self._groups = tuple(groups)
        self._indices_by_task = {
            task: tuple(indices) for task, indices in sorted(indices_by_task.items())
        }
        self._provenance = deepcopy(dict(provenance))
        self._protocol = protocol
        self.backbone_sha256 = backbone_sha256
        self.base_lineage_sha256 = base_lineage_sha256
        self.release_paired_binding_sha256 = release_paired_binding_sha256
        self.action_manifest_sha256 = action_manifest_sha256
        self.action_audit_sha256 = action_audit_sha256
        self.state_bank_sha256 = state_bank_sha256
        self.physical_state_inventory_sha256 = inventory_sha256
        self._state_bank = state_bank
        self._extraction_contract = deepcopy(dict(extraction_contract))
        self._native_prefill_identity_audit = deepcopy(dict(native_prefill_audit))

    @property
    def variant_names(self) -> tuple[str, ...]:
        return POLICY_VARIANTS

    @property
    def indices_by_task(self) -> dict[str, tuple[int, ...]]:
        return dict(self._indices_by_task)

    @property
    def groups_by_task(self) -> dict[str, tuple[int, ...]]:
        return self.indices_by_task

    @property
    def provenance(self) -> dict[str, Any]:
        return deepcopy(self._provenance)

    @property
    def token_shape(self) -> tuple[int, int]:
        return int(self._tokens.shape[1]), int(self._tokens.shape[2])

    @property
    def token_dtype(self) -> torch.dtype:
        return self._tokens.dtype

    def __len__(self) -> int:
        return len(self._groups)

    def physical_state_id_for_index(self, index: int) -> str:
        return str(self._groups[index]["physical_state_id"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_index = index if index >= 0 else len(self._groups) + index
        group = self._groups[index]
        start = int(group["start"])
        stop = start + POLICY_VIEW_COUNT
        condition = self._provenance.get("conditions_by_physical_state", {}).get(
            group["physical_state_id"]
        )
        return {
            "tokens": self._tokens[start:stop],
            "variant_names": POLICY_VARIANTS,
            "protocol_id": POLICY_PROTOCOL_ID,
            "r3_role": POLICY_R3_ROLE,
            "task": group["task"],
            "physical_state_id": group["physical_state_id"],
            "trajectory_id": group["trajectory_id"],
            "content_id": group["content_id"],
            "frame_offset": group["frame_offset"],
            "split": self.split,
            "layer": self.layer,
            "dataset_index": group_index,
            "record_indices": tuple(range(start, stop)),
            "records": tuple(deepcopy(value) for value in self._records[start:stop]),
            "physical_state": deepcopy(self._physical_states[group_index]),
            "proprio_raw": self._proprio_raw[group_index],
            "condition_provenance": deepcopy(condition),
        }


def collate_paired_token_groups(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate physical states while preserving the four-scene axis."""

    _require(bool(samples), "cannot collate an empty paired batch")
    for sample in samples:
        _require(
            tuple(str(value) for value in sample.get("variant_names", ())) == POLICY_VARIANTS,
            "paired batch is not canonical ordered C/R1/R2/R3",
        )
        _require(sample.get("r3_role") == POLICY_R3_ROLE, "R3 is not marked training_positive")
        _require(
            isinstance(sample.get("tokens"), torch.Tensor)
            and sample["tokens"].ndim == 3
            and sample["tokens"].shape[0] == POLICY_VIEW_COUNT,
            "paired sample tokens must preserve a four-scene axis",
        )
    tokens = torch.stack([sample["tokens"] for sample in samples], dim=0)
    proprio = torch.stack([sample["proprio_raw"] for sample in samples], dim=0)
    tasks = tuple(str(sample["task"]) for sample in samples)
    return {
        "tokens": tokens,
        "variant_names": POLICY_VARIANTS,
        "protocol_id": POLICY_PROTOCOL_ID,
        "r3_role": POLICY_R3_ROLE,
        "supervision_mode": "contrastive",
        "task": tasks,
        "same_task": len(set(tasks)) == 1,
        "physical_state_id": tuple(str(sample["physical_state_id"]) for sample in samples),
        "trajectory_id": tuple(str(sample["trajectory_id"]) for sample in samples),
        "content_id": torch.tensor([int(sample["content_id"]) for sample in samples], dtype=torch.long),
        "frame_offset": torch.tensor([int(sample["frame_offset"]) for sample in samples], dtype=torch.long),
        "dataset_index": torch.tensor([int(sample["dataset_index"]) for sample in samples], dtype=torch.long),
        "proprio_raw": proprio,
        "records": tuple(sample["records"] for sample in samples),
        "physical_state": tuple(sample["physical_state"] for sample in samples),
        "condition_provenance": tuple(sample["condition_provenance"] for sample in samples),
        "split": str(samples[0]["split"]),
        "layer": int(samples[0]["layer"]),
    }


@dataclass(frozen=True)
class NativePairedEpisodeGroup:
    task: str
    content_id: int
    split: str
    trajectory_id: str
    episode_length: int
    valid_action_anchor_count: int
    episodes: tuple[tuple[str, int], ...]

    @property
    def episode_by_variant(self) -> dict[str, int]:
        return dict(self.episodes)


@dataclass(frozen=True)
class VerifiedNativeActionManifest:
    path: Path
    sha256: str
    dataset_root: Path
    groups: tuple[NativePairedEpisodeGroup, ...]
    audit_path: Path
    audit_sha256: str
    protocol: dict[str, Any]

    def groups_for_split(self, split: str) -> tuple[NativePairedEpisodeGroup, ...]:
        return tuple(group for group in self.groups if group.split == split)


@dataclass(frozen=True)
class PolicyPhysicalStateAnchor:
    task: str
    content_id: int
    trajectory_id: str
    frame_offset: int

    @property
    def physical_state_id(self) -> str:
        return f"{self.task}/content_{self.content_id:06d}/frame_{self.frame_offset:06d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "content_id": self.content_id,
            "trajectory_id": self.trajectory_id,
            "frame_offset": self.frame_offset,
        }


@dataclass(frozen=True)
class VerifiedPolicyStateBank:
    path: Path
    sha256: str
    native_manifest: VerifiedNativeActionManifest
    anchors: tuple[PolicyPhysicalStateAnchor, ...]
    physical_state_inventory_sha256: str
    protocol: dict[str, Any]
    sampling: dict[str, Any]

    @property
    def anchors_by_content(self) -> dict[tuple[str, int], tuple[PolicyPhysicalStateAnchor, ...]]:
        grouped: dict[tuple[str, int], list[PolicyPhysicalStateAnchor]] = defaultdict(list)
        for anchor in self.anchors:
            grouped[(anchor.task, anchor.content_id)].append(anchor)
        return {identity: tuple(values) for identity, values in grouped.items()}


def policy_state_bank_offsets(
    *,
    task: str,
    content_id: int,
    episode_length: int,
    seed: int = POLICY_STATE_BANK_SEED,
    states_per_trajectory: int = POLICY_STATES_PER_TRAJECTORY,
    sampling_algorithm: str = POLICY_STATE_BANK_SAMPLING_ALGORITHM,
) -> tuple[int, ...]:
    """Select stable, endpoint-safe state offsets without interpolation.

    The rank of each candidate is the SHA-256 digest of the protocol identity,
    global seed, physical trajectory identity, and candidate offset.  Selecting
    the eight smallest digests is independent of Python's RNG implementation.
    The returned offsets are sorted so the state-bank inventory is canonical.
    """

    length = int(episode_length)
    requested = int(states_per_trajectory)
    _require(requested > 0, "states_per_trajectory must be positive")
    algorithm = str(sampling_algorithm)
    _require(bool(algorithm), "state-bank sampling algorithm must be non-empty")
    valid_count = length - POLICY_ACTION_STEPS
    _require(
        valid_count >= requested,
        f"{task}/content_{int(content_id):06d} has only {valid_count} endpoint-safe "
        f"windows; need {requested}",
    )
    ranked: list[tuple[bytes, int]] = []
    for frame_offset in range(valid_count):
        identity = (
            f"{algorithm}|{int(seed)}|{task}|"
            f"{int(content_id)}|{frame_offset}"
        ).encode("utf-8")
        ranked.append((hashlib.sha256(identity).digest(), frame_offset))
    selected = sorted(
        frame_offset
        for _digest, frame_offset in sorted(ranked)[:requested]
    )
    return tuple(selected)


def physical_state_inventory_sha256(
    anchors: Sequence[PolicyPhysicalStateAnchor],
) -> str:
    body = [anchor.as_dict() for anchor in anchors]
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def selected_episode_artifact_aggregate(
    native_manifest: VerifiedNativeActionManifest,
    *,
    split: str = "train",
) -> dict[str, Any]:
    """Content-address every parquet/video artifact used by the state bank."""

    _require(split == "train", "Policy cache artifact aggregate currently supports train")
    root = native_manifest.dataset_root
    episode_ids = sorted(
        {
            episode
            for group in native_manifest.groups_for_split(split)
            for _variant, episode in group.episodes
        }
    )
    _require(bool(episode_ids), "selected paired episode inventory is empty")
    relative_paths: list[str] = []
    for episode in episode_ids:
        chunk = int(episode) // 1000
        relative_paths.append(f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet")
        for camera in POLICY_CAMERA_NAMES:
            relative_paths.append(
                f"videos/chunk-{chunk:03d}/observation.images.{camera}/"
                f"episode_{episode:06d}.mp4"
            )
    relative_paths.sort()
    digest = hashlib.sha256()
    total_size = 0
    for relative_text in relative_paths:
        path = root / relative_text
        _require(path.is_file() and not path.is_symlink(), f"selected artifact missing/unsafe: {path}")
        relative = relative_text.encode("utf-8")
        size = int(path.stat().st_size)
        digest.update(len(relative).to_bytes(8, byteorder="big", signed=False))
        digest.update(relative)
        digest.update(size.to_bytes(8, byteorder="big", signed=False))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        total_size += size
    return {
        "algorithm": "relative_path_size_and_bytes_sha256_v1",
        "dataset_root": str(root),
        "split": split,
        "episode_count": len(episode_ids),
        "file_count": len(relative_paths),
        "size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def verify_native_paired_action_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    audit_path: str | Path,
) -> VerifiedNativeActionManifest:
    """Verify the Policy manifest and the collector's independent strict audit."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    audit_file = Path(audit_path).expanduser().resolve()
    _require(root.is_dir(), f"paired native action root not found: {root}")
    _require(manifest_file.is_file(), f"paired action manifest not found: {manifest_file}")
    _require(audit_file.is_file(), f"paired action audit not found: {audit_file}")
    manifest = _load_json_object(manifest_file)
    _require(
        manifest.get("schema") == POLICY_ACTION_MANIFEST_SCHEMA,
        f"paired action manifest schema must be {POLICY_ACTION_MANIFEST_SCHEMA!r}",
    )
    _require(
        int(manifest.get("schema_version", -1)) == POLICY_ACTION_MANIFEST_VERSION,
        f"paired action manifest schema_version must be {POLICY_ACTION_MANIFEST_VERSION}",
    )
    protocol = _protocol_metadata(manifest, split=str(manifest.get("split", "train")))
    _require(
        Path(str(manifest.get("dataset_root", ""))).expanduser().resolve() == root,
        "paired manifest dataset_root differs from the selected root",
    )
    raw_groups = manifest.get("groups")
    _require(isinstance(raw_groups, list) and raw_groups, "paired manifest groups must be non-empty")
    groups: list[NativePairedEpisodeGroup] = []
    seen_content: set[tuple[str, int]] = set()
    seen_episodes: set[int] = set()
    for group_index, raw in enumerate(raw_groups):
        _require(isinstance(raw, Mapping), f"manifest group {group_index} is not a mapping")
        task = str(raw.get("task", ""))
        _require(bool(task), f"manifest group {group_index} has an empty task")
        content_id = int(raw.get("content_id", -1))
        _require(content_id >= 0, f"manifest group {group_index} has invalid content_id")
        split = str(raw.get("split", ""))
        _require(split in POLICY_DATA_SPLITS, f"manifest group {group_index} has invalid split")
        try:
            expected_split = policy_split_for_content_id(content_id)
        except PolicyProtocolError as exc:
            raise DataContractError(str(exc)) from exc
        _require(
            split == expected_split,
            f"manifest group {group_index} content_id {content_id} belongs to "
            f"{expected_split!r}, not {split!r}",
        )
        identity = (task, content_id)
        _require(identity not in seen_content, f"duplicate paired content group {identity}")
        seen_content.add(identity)
        episode_length = int(raw.get("episode_length", -1))
        valid_action_anchor_count = int(raw.get("valid_action_anchor_count", -1))
        _require(
            episode_length >= POLICY_STATE_STEPS + POLICY_STATES_PER_TRAJECTORY - 1,
            f"manifest group {group_index} is too short for eight endpoint-safe states: "
            f"episode_length={episode_length}",
        )
        _require(
            valid_action_anchor_count == episode_length - POLICY_ACTION_STEPS,
            f"manifest group {group_index} valid_action_anchor_count must equal "
            "episode_length-32 for a 33-state/32-action no-padding window",
        )
        episodes = raw.get("episodes")
        _require(isinstance(episodes, Mapping), f"manifest group {group_index} episodes missing")
        _require(
            set(str(value) for value in episodes) == set(POLICY_VARIANTS),
            "episode mapping must contain exactly C/R1/R2/R3",
        )
        canonical: list[tuple[str, int]] = []
        for variant in POLICY_VARIANTS:
            episode = int(episodes[variant])
            _require(episode >= 0, f"manifest group {group_index} has invalid episode")
            _require(episode not in seen_episodes, f"episode {episode} appears in multiple scene groups")
            seen_episodes.add(episode)
            canonical.append((variant, episode))
        groups.append(
            NativePairedEpisodeGroup(
                task=task,
                content_id=content_id,
                split=split,
                trajectory_id=str(raw.get("trajectory_id", f"{task}/content_{content_id:06d}")),
                episode_length=episode_length,
                valid_action_anchor_count=valid_action_anchor_count,
                episodes=tuple(canonical),
            )
        )

    manifest_sha256 = _sha256_file(manifest_file)
    audit = _load_json_object(audit_file)
    _require(audit.get("status") == "PASS", "paired native action audit status is not PASS")
    _protocol_metadata(audit, split=str(audit.get("split", protocol["split"])))
    _require(audit.get("manifest_sha256") == manifest_sha256, "paired audit manifest SHA-256 mismatch")
    _require(
        Path(str(audit.get("dataset_root", ""))).expanduser().resolve() == root,
        "paired audit dataset_root mismatch",
    )
    checks = audit.get("checks")
    _require(isinstance(checks, Mapping), "paired native action audit checks are missing")
    required_checks = (
        "three_camera_sync",
        "native_50hz",
        "action_window_32x14",
        "state_window_33x14",
        "cross_scene_state_exact",
        "cross_scene_action_exact",
        "temporal_resampling_absent",
        "endpoint_safe_state_bank_supported",
    )
    _require(
        all(checks.get(name) is True for name in required_checks),
        "paired native action audit did not pass every required check",
    )
    return VerifiedNativeActionManifest(
        path=manifest_file,
        sha256=manifest_sha256,
        dataset_root=root,
        groups=tuple(groups),
        audit_path=audit_file,
        audit_sha256=_sha256_file(audit_file),
        protocol=protocol,
    )


def verify_policy_state_bank(
    state_bank_path: str | Path,
    *,
    native_manifest: VerifiedNativeActionManifest,
    expected_sha256: str | None = None,
    expected_tasks: Sequence[str] | None = None,
    expected_states_per_trajectory: int = POLICY_STATES_PER_TRAJECTORY,
    expected_sampling_algorithm: str = POLICY_STATE_BANK_SAMPLING_ALGORITHM,
    expected_sampling_version: int = POLICY_STATE_BANK_SAMPLING_VERSION,
    expected_sampling_seed: int = POLICY_STATE_BANK_SEED,
) -> VerifiedPolicyStateBank:
    """Verify the one immutable train-state inventory shared by C2 and C3."""

    _require(
        isinstance(native_manifest, VerifiedNativeActionManifest),
        "state-bank verification requires a verified native action manifest",
    )
    path = Path(state_bank_path).expanduser().resolve()
    _require(path.is_file(), f"paired state bank not found: {path}")
    digest = _sha256_file(path)
    if expected_sha256 is not None:
        _require(_valid_sha256(expected_sha256), "expected paired state-bank SHA-256 is invalid")
        _require(digest == expected_sha256, "paired state-bank SHA-256 differs from run artifacts")
    value = _load_json_object(path)
    _require(
        value.get("schema") == POLICY_STATE_BANK_SCHEMA,
        f"state-bank schema must be {POLICY_STATE_BANK_SCHEMA!r}",
    )
    _require(
        int(value.get("schema_version", -1)) == POLICY_STATE_BANK_SCHEMA_VERSION,
        f"state-bank schema_version must be {POLICY_STATE_BANK_SCHEMA_VERSION}",
    )
    protocol = _protocol_metadata(value, split="train")
    _require(
        value.get("paired_action_manifest_sha256") == native_manifest.sha256,
        "state bank is not bound to the selected native action manifest",
    )
    _require(
        value.get("paired_action_audit_sha256") == native_manifest.audit_sha256,
        "state bank is not bound to the selected native action audit",
    )
    sampling = value.get("sampling")
    _require(isinstance(sampling, Mapping), "state-bank sampling metadata is missing")
    expected_sampling = {
        "algorithm": str(expected_sampling_algorithm),
        "version": int(expected_sampling_version),
        "seed": int(expected_sampling_seed),
        "states_per_trajectory": int(expected_states_per_trajectory),
        "endpoint_rule": "33_state_frames_and_32_actions_without_padding",
        "short_trajectory_policy": "fail_closed",
    }
    _require(
        dict(sampling) == expected_sampling,
        f"state-bank sampling metadata must be exactly {expected_sampling}",
    )

    train_groups = native_manifest.groups_for_split("train")
    _require(bool(train_groups), "native action manifest has no train trajectories")
    if expected_tasks is not None:
        expected = tuple(str(task) for task in expected_tasks)
        actual = tuple(dict.fromkeys(group.task for group in train_groups))
        _require(actual == expected, f"state-bank task order must be exactly {expected}")
    expected_anchors: list[PolicyPhysicalStateAnchor] = []
    for group in train_groups:
        for frame_offset in policy_state_bank_offsets(
            task=group.task,
            content_id=group.content_id,
            episode_length=group.episode_length,
            seed=int(expected_sampling_seed),
            states_per_trajectory=int(expected_states_per_trajectory),
            sampling_algorithm=str(expected_sampling_algorithm),
        ):
            expected_anchors.append(
                PolicyPhysicalStateAnchor(
                    task=group.task,
                    content_id=group.content_id,
                    trajectory_id=group.trajectory_id,
                    frame_offset=frame_offset,
                )
            )

    raw_states = value.get("states")
    _require(isinstance(raw_states, list), "state-bank states must be a list")
    _require(
        len(raw_states) == len(expected_anchors),
        f"state bank must contain exactly {len(expected_anchors)} train anchors",
    )
    parsed: list[PolicyPhysicalStateAnchor] = []
    seen: set[tuple[str, int, int]] = set()
    for index, (raw, expected_anchor) in enumerate(zip(raw_states, expected_anchors, strict=True)):
        _require(isinstance(raw, Mapping), f"state-bank entry {index} is not a mapping")
        anchor = PolicyPhysicalStateAnchor(
            task=str(raw.get("task", "")),
            content_id=int(raw.get("content_id", -1)),
            trajectory_id=str(raw.get("trajectory_id", "")),
            frame_offset=int(raw.get("frame_offset", -1)),
        )
        _require(
            anchor == expected_anchor,
            f"state-bank entry {index} differs from the canonical deterministic inventory: "
            f"expected {expected_anchor.as_dict()}, got {anchor.as_dict()}",
        )
        if "physical_state_id" in raw:
            _require(
                str(raw["physical_state_id"]) == anchor.physical_state_id,
                f"state-bank entry {index} physical_state_id is inconsistent",
            )
        identity = (anchor.task, anchor.content_id, anchor.frame_offset)
        _require(identity not in seen, f"duplicate state-bank anchor {identity}")
        seen.add(identity)
        parsed.append(anchor)
    inventory_sha = physical_state_inventory_sha256(parsed)
    _require(
        value.get("physical_state_inventory_sha256") == inventory_sha,
        "state-bank physical_state_inventory_sha256 mismatch",
    )
    return VerifiedPolicyStateBank(
        path=path,
        sha256=digest,
        native_manifest=native_manifest,
        anchors=tuple(parsed),
        physical_state_inventory_sha256=inventory_sha,
        protocol=protocol,
        sampling=dict(sampling),
    )


def audit_native_paired_action_contract(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    expected_tasks: Sequence[str] | None = None,
    require_full_protocol_counts: bool = False,
) -> dict[str, Any]:
    """Audit manifest/collector evidence without decoding video samples."""

    verified = verify_native_paired_action_manifest(
        manifest_path,
        dataset_root=dataset_root,
        audit_path=audit_path,
    )
    task_order = tuple(dict.fromkeys(group.task for group in verified.groups))
    if expected_tasks is not None:
        expected = tuple(str(task) for task in expected_tasks)
        _require(task_order == expected, f"paired manifest task order must be {expected}")
    counts = {
        task: {
            split: sum(
                group.task == task and group.split == split for group in verified.groups
            )
            for split in POLICY_DATA_SPLITS
        }
        for task in task_order
    }
    if require_full_protocol_counts:
        _require(expected_tasks is not None, "full paired count audit requires expected_tasks")
        for task in task_order:
            _require(
                counts[task] == POLICY_CONTENTS_PER_TASK_BY_SPLIT,
                f"full paired manifest counts for {task} must be "
                f"{POLICY_CONTENTS_PER_TASK_BY_SPLIT}, got {counts[task]}",
            )
    report = {
        "status": "PASS",
        "kind": "policy_native_paired_action_contract",
        "protocol_id": POLICY_PROTOCOL_ID,
        "variant_names": list(POLICY_VARIANTS),
        "view_count": POLICY_VIEW_COUNT,
        "r3_role": POLICY_R3_ROLE,
        "r3_training_positive": True,
        "camera_names": list(POLICY_CAMERA_NAMES),
        "camera_count": POLICY_CAMERA_COUNT,
        "native_fps": POLICY_NATIVE_FPS,
        "action_steps": POLICY_ACTION_STEPS,
        "action_dim": POLICY_ACTION_DIM,
        "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
        "dataset_root": str(verified.dataset_root),
        "manifest": {"path": str(verified.path), "sha256": verified.sha256},
        "collector_audit": {
            "path": str(verified.audit_path),
            "sha256": verified.audit_sha256,
        },
        "task_order": list(task_order),
        "content_groups_by_task_split": counts,
        "full_protocol_counts_required": bool(require_full_protocol_counts),
    }
    json.dumps(report, sort_keys=True)
    return report


def audit_policy_state_bank(state_bank: VerifiedPolicyStateBank) -> dict[str, Any]:
    _require(isinstance(state_bank, VerifiedPolicyStateBank), "expected verified state bank")
    counts: dict[str, dict[int, int]] = defaultdict(dict)
    for (task, content_id), anchors in state_bank.anchors_by_content.items():
        counts[task][content_id] = len(anchors)
    report = {
        "status": "PASS",
        "kind": "policy_shared_paired_state_bank",
        "schema": POLICY_STATE_BANK_SCHEMA,
        "schema_version": POLICY_STATE_BANK_SCHEMA_VERSION,
        "protocol_id": POLICY_PROTOCOL_ID,
        "split": "train",
        "state_bank": {"path": str(state_bank.path), "sha256": state_bank.sha256},
        "paired_action_manifest_sha256": state_bank.native_manifest.sha256,
        "paired_action_audit_sha256": state_bank.native_manifest.audit_sha256,
        "physical_state_inventory_sha256": state_bank.physical_state_inventory_sha256,
        "sampling": dict(state_bank.sampling),
        "physical_state_count": len(state_bank.anchors),
        "physical_states_by_task": {
            task: sum(values.values()) for task, values in sorted(counts.items())
        },
        "states_per_content_by_task": {
            task: {str(content_id): count for content_id, count in sorted(values.items())}
            for task, values in sorted(counts.items())
        },
    }
    json.dumps(report, sort_keys=True)
    return report


def _scalar_int(value: Any, *, label: str) -> int:
    if isinstance(value, torch.Tensor):
        _require(value.numel() == 1, f"{label} must contain one value")
        value = value.detach().cpu().item()
    _require(not isinstance(value, bool), f"{label} must be integer-like")
    integer = int(value)
    _require(integer == value, f"{label} is not exact integer-like")
    return integer


def _integer_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    _require(isinstance(value, Sequence), f"{label} must be a sequence")
    return tuple(_scalar_int(item, label=f"{label} item") for item in value)


class _ExactNativeIndexProxy:
    """Reject BaseLerobotDataset's random-index read recovery."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __len__(self) -> int:
        return len(self._target)

    def __getitem__(self, requested_index: int) -> Any:
        sample = self._target[requested_index]
        _require(isinstance(sample, Mapping), "native LeRobot loader returned a non-mapping")
        _require("idx" in sample, "native LeRobot sample lacks idx recovery guard")
        actual = _scalar_int(sample["idx"], label="native returned idx")
        _require(actual == int(requested_index), f"native loader replaced frame {requested_index} with {actual}")
        return sample


@dataclass(frozen=True)
class NativePairedFrameRecord:
    task: str
    content_id: int
    split: str
    trajectory_id: str
    frame_offset: int
    base_indices: tuple[int, ...]
    episode_indices: tuple[int, ...]

    @property
    def physical_state_id(self) -> str:
        return f"{self.task}/content_{self.content_id:06d}/frame_{self.frame_offset:06d}"


def _require_tensor(sample: Mapping[str, Any], key: str, *, label: str) -> torch.Tensor:
    value = sample.get(key)
    _require(isinstance(value, torch.Tensor), f"{label} {key!r} must be a tensor")
    _require(bool(torch.isfinite(value.float()).all()), f"{label} {key!r} contains NaN/inf")
    return value


def _assert_exact_views(values: Sequence[torch.Tensor], *, label: str) -> None:
    reference = values[0]
    _require(
        all(value.shape == reference.shape and value.dtype == reference.dtype for value in values),
        f"four scene versions have inconsistent {label} shape/dtype",
    )
    _require(
        all(torch.equal(value, reference) for value in values[1:]),
        f"four scene versions do not share exact {label}",
    )


class NativePairedActionDataset(Dataset[dict[str, Any]]):
    """Group one native FastWAM sample from each C/R1/R2/R3 episode.

    ``base_dataset`` must be the same processed ``RobotVideoDataset`` class used
    by the official stream, instantiated over the new LeRobot v2.1 root.  The
    wrapper changes no processor or normalization behavior.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        dataset_root: str | Path,
        manifest_path: str | Path,
        audit_path: str | Path,
        state_bank_path: str | Path,
        split: str = "train",
        expected_state_bank_sha256: str | None = None,
        expected_tasks: Sequence[str] | None = None,
        require_full_protocol_counts: bool = False,
        state_bank_states_per_trajectory: int = POLICY_STATES_PER_TRAJECTORY,
        state_bank_sampling_algorithm: str = POLICY_STATE_BANK_SAMPLING_ALGORITHM,
        state_bank_sampling_version: int = POLICY_STATE_BANK_SAMPLING_VERSION,
        state_bank_sampling_seed: int = POLICY_STATE_BANK_SEED,
    ) -> None:
        _require(split in POLICY_DATA_SPLITS, f"paired split must be one of {POLICY_DATA_SPLITS}")
        _require(split == "train", "shared Policy state bank currently supports train only")
        _require(callable(getattr(base_dataset, "_get", None)), "native paired dataset must expose _get")
        _require(
            not bool(getattr(base_dataset, "skip_padding_as_possible", False)),
            "skip_padding_as_possible can change the requested paired frame",
        )
        verified = verify_native_paired_action_manifest(
            manifest_path,
            dataset_root=dataset_root,
            audit_path=audit_path,
        )
        groups = verified.groups_for_split(split)
        _require(bool(groups), f"paired manifest contains no {split} groups")
        state_bank = verify_policy_state_bank(
            state_bank_path,
            native_manifest=verified,
            expected_sha256=expected_state_bank_sha256,
            expected_tasks=expected_tasks,
            expected_states_per_trajectory=state_bank_states_per_trajectory,
            expected_sampling_algorithm=state_bank_sampling_algorithm,
            expected_sampling_version=state_bank_sampling_version,
            expected_sampling_seed=state_bank_sampling_seed,
        )
        if expected_tasks is not None:
            expected = tuple(str(task) for task in expected_tasks)
            actual = tuple(dict.fromkeys(group.task for group in groups))
            _require(actual == expected, f"paired task order must be exactly {expected}, got {actual}")
            if require_full_protocol_counts:
                all_tasks = tuple(dict.fromkeys(group.task for group in verified.groups))
                _require(all_tasks == expected, "full paired manifest task order changed")
                for task in expected:
                    counts = {
                        name: sum(
                            group.task == task and group.split == name
                            for group in verified.groups
                        )
                        for name in POLICY_CONTENTS_PER_TASK_BY_SPLIT
                    }
                    _require(
                        counts == POLICY_CONTENTS_PER_TASK_BY_SPLIT,
                        f"full paired manifest counts for {task} must be "
                        f"{POLICY_CONTENTS_PER_TASK_BY_SPLIT}, got {counts}",
                    )

        lerobot = getattr(base_dataset, "lerobot_dataset", None)
        _require(lerobot is not None, "native paired dataset lacks lerobot_dataset")
        multi_dataset = getattr(lerobot, "multi_dataset", None)
        inner_datasets = getattr(multi_dataset, "_datasets", None)
        _require(
            isinstance(inner_datasets, Sequence) and len(inner_datasets) == 1,
            "paired wrapper requires exactly one underlying LeRobot dataset",
        )
        inner = inner_datasets[0]
        inner_root = Path(str(getattr(inner, "root", ""))).expanduser().resolve()
        _require(inner_root == verified.dataset_root, "native paired dataset root differs from manifest")
        episodes = getattr(inner, "episodes", None)
        _require(episodes is not None, "native paired dataset must expose explicit episode order")
        episode_ids = _integer_tuple(episodes, label="underlying paired episodes")
        _require(len(episode_ids) == len(set(episode_ids)), "underlying paired episode order has duplicates")
        local_by_episode = {episode: index for index, episode in enumerate(episode_ids)}

        episode_data_index = getattr(lerobot, "episode_data_index", None)
        _require(isinstance(episode_data_index, Mapping), "paired episode_data_index is missing")
        starts = _integer_tuple(episode_data_index.get("from"), label="paired episode starts")
        ends = _integer_tuple(episode_data_index.get("to"), label="paired episode ends")
        _require(len(starts) == len(episode_ids) == len(ends), "paired episode index lengths differ")

        native_group_runtime: dict[
            tuple[str, int],
            tuple[NativePairedEpisodeGroup, tuple[int, ...], tuple[int, ...]],
        ] = {}
        for group in groups:
            episode_by_variant = group.episode_by_variant
            _require(
                all(episode_by_variant[variant] in local_by_episode for variant in POLICY_VARIANTS),
                f"native paired dataset did not load every episode for {group.trajectory_id}",
            )
            local_positions = tuple(local_by_episode[episode_by_variant[variant]] for variant in POLICY_VARIANTS)
            lengths = tuple(ends[position] - starts[position] for position in local_positions)
            _require(len(set(lengths)) == 1, f"four episodes are not equal length for {group.trajectory_id}")
            _require(
                lengths[0] == group.episode_length,
                f"native episode length differs from manifest for {group.trajectory_id}",
            )
            native_group_runtime[(group.task, group.content_id)] = (
                group,
                local_positions,
                tuple(starts[position] for position in local_positions),
            )

        records: list[NativePairedFrameRecord] = []
        indices_by_task: dict[str, list[int]] = defaultdict(list)
        for anchor in state_bank.anchors:
            group, _local_positions, episode_starts = native_group_runtime[
                (anchor.task, anchor.content_id)
            ]
            _require(
                0 <= anchor.frame_offset < group.valid_action_anchor_count,
                f"state-bank offset needs native padding for {anchor.physical_state_id}",
            )
            episode_by_variant = group.episode_by_variant
            record = NativePairedFrameRecord(
                task=group.task,
                content_id=group.content_id,
                split=group.split,
                trajectory_id=group.trajectory_id,
                frame_offset=anchor.frame_offset,
                base_indices=tuple(start + anchor.frame_offset for start in episode_starts),
                episode_indices=tuple(
                    episode_by_variant[variant] for variant in POLICY_VARIANTS
                ),
            )
            dataset_index = len(records)
            records.append(record)
            indices_by_task[group.task].append(dataset_index)
        _require(bool(records), "native paired action dataset has no frame anchors")

        try:
            runner = copy.copy(base_dataset)
        except Exception as exc:
            raise DataContractError(f"cannot create guarded native paired dataset: {exc}") from exc
        proxy = _ExactNativeIndexProxy(lerobot)
        runner.lerobot_dataset = proxy

        self._base_dataset = base_dataset
        self._runner = runner
        self._proxy = proxy
        self._verified = verified
        self._state_bank = state_bank
        self._records = tuple(records)
        self._indices_by_task = {
            task: tuple(indices) for task, indices in sorted(indices_by_task.items())
        }
        self.split = str(split)
        self.sampling_mode = "shared_state_bank"
        self.require_full_protocol_counts = bool(require_full_protocol_counts)

    @property
    def variant_names(self) -> tuple[str, ...]:
        return POLICY_VARIANTS

    @property
    def indices_by_task(self) -> dict[str, tuple[int, ...]]:
        return dict(self._indices_by_task)

    @property
    def manifest(self) -> VerifiedNativeActionManifest:
        return self._verified

    @property
    def audit_report(self) -> dict[str, Any]:
        return {
            "kind": "policy_native_paired_action_dataset",
            "protocol_id": POLICY_PROTOCOL_ID,
            "variant_names": list(POLICY_VARIANTS),
            "view_count": POLICY_VIEW_COUNT,
            "r3_role": POLICY_R3_ROLE,
            "r3_training_positive": True,
            "camera_names": list(POLICY_CAMERA_NAMES),
            "camera_count": POLICY_CAMERA_COUNT,
            "native_fps": POLICY_NATIVE_FPS,
            "action_steps": POLICY_ACTION_STEPS,
            "action_dim": POLICY_ACTION_DIM,
            "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
            "native_action_targets": True,
            "split": self.split,
            "sampling_mode": self.sampling_mode,
            "state_bank": {"path": str(self._state_bank.path), "sha256": self._state_bank.sha256},
            "physical_state_inventory_sha256": self._state_bank.physical_state_inventory_sha256,
            "states_per_trajectory": POLICY_STATES_PER_TRAJECTORY,
            "full_protocol_counts_required": self.require_full_protocol_counts,
            "expected_contents_per_task_by_split": dict(
                POLICY_CONTENTS_PER_TASK_BY_SPLIT
            ),
            "manifest": {"path": str(self._verified.path), "sha256": self._verified.sha256},
            "collector_audit": {
                "path": str(self._verified.audit_path),
                "sha256": self._verified.audit_sha256,
            },
            "dataset_root": str(self._verified.dataset_root),
            "physical_state_count": len(self),
            "physical_states_by_task": {
                task: len(indices) for task, indices in self._indices_by_task.items()
            },
            "supervision_mode": "action",
        }

    def __len__(self) -> int:
        return len(self._records)

    def physical_state_id_for_index(self, index: int) -> str:
        return self._records[index].physical_state_id

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        processed: list[Mapping[str, Any]] = []
        raw: list[Mapping[str, Any]] = []
        for base_index in record.base_indices:
            native = self._proxy[base_index]
            sample = self._runner._get(base_index)  # noqa: SLF001
            _require(isinstance(sample, Mapping), "RobotVideoDataset._get returned non-mapping")
            raw.append(native)
            processed.append(sample)

        raw_actions = [_require_tensor(sample, "action", label="raw native") for sample in raw]
        raw_states = [_require_tensor(sample, "proprio", label="raw native") for sample in raw]
        raw_pixels = [sample.get("pixel_values") for sample in raw]
        _require(
            all(isinstance(value, torch.Tensor) for value in raw_pixels),
            "processed native pixel_values must be tensors",
        )
        _require(
            all(tuple(value.shape) == (POLICY_ACTION_STEPS, POLICY_ACTION_DIM) for value in raw_actions),
            "raw native action must be [32,14]",
        )
        _require(
            all(tuple(value.shape) == (POLICY_ACTION_STEPS + 1, POLICY_ACTION_DIM) for value in raw_states),
            "raw native state must be [33,14]",
        )
        _require(
            all(
                value.ndim == 5
                and value.shape[0] == POLICY_CAMERA_COUNT
                and value.shape[1] == POLICY_ACTION_STEPS + 1
                and value.shape[2] == 3
                and tuple(value.shape[-2:]) == (240, 320)
                for value in raw_pixels
            ),
            "processed native window must contain three synchronized [33,3,240,320] cameras",
        )
        _assert_exact_views(raw_actions, label="raw 32-step action target")
        _assert_exact_views(raw_states, label="raw 33-step robot state")

        keys = ("video", "action", "proprio", "context", "context_mask", "action_is_pad")
        values = {key: [_require_tensor(sample, key, label="processed") for sample in processed] for key in keys}
        _require(
            all(
                value.ndim == 4
                and value.shape[0] == 3
                and value.shape[1] > 1
                and value.shape[1] % 4 == 1
                for value in values["video"]
            ),
            "processed FastWAM video must be [3,T,H,W] with T>1 and T%4=1",
        )
        _require(
            all(tuple(value.shape) == (POLICY_ACTION_STEPS, POLICY_ACTION_DIM) for value in values["action"]),
            "processed action must be [32,14]",
        )
        _require(
            all(tuple(value.shape) == (POLICY_ACTION_STEPS, POLICY_ACTION_DIM) for value in values["proprio"]),
            "processed proprio must be [32,14]",
        )
        for key in ("action", "proprio", "context", "context_mask", "action_is_pad"):
            _assert_exact_views(values[key], label=f"processed {key}")
        prompts = tuple(str(sample.get("prompt", "")) for sample in processed)
        _require(bool(prompts[0]) and len(set(prompts)) == 1, "four scenes do not share one prompt")

        return {
            **{key: torch.stack(value, dim=0) for key, value in values.items()},
            "state_window": torch.stack(raw_states, dim=0),
            "raw_action_window": torch.stack(raw_actions, dim=0),
            "variant_names": POLICY_VARIANTS,
            "protocol_id": POLICY_PROTOCOL_ID,
            "r3_role": POLICY_R3_ROLE,
            "camera_names": POLICY_CAMERA_NAMES,
            "camera_count": POLICY_CAMERA_COUNT,
            "native_fps": POLICY_NATIVE_FPS,
            "action_steps": POLICY_ACTION_STEPS,
            "action_dim": POLICY_ACTION_DIM,
            "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
            "native_action_targets": True,
            "supervision_mode": "action",
            "task": record.task,
            "physical_state_id": record.physical_state_id,
            "trajectory_id": record.trajectory_id,
            "content_id": record.content_id,
            "frame_offset": record.frame_offset,
            "episode_indices": record.episode_indices,
            "base_indices": record.base_indices,
            "split": record.split,
            "prompt": prompts[0],
        }


def validate_native_paired_action_batch(batch: Mapping[str, Any]) -> None:
    """Validate a collated C2 batch before it reaches action supervision."""

    metadata = {
        key: batch.get(key)
        for key in (
            "protocol_id",
            "variant_names",
            "view_count",
            "r3_role",
            "camera_count",
            "camera_names",
            "native_fps",
            "action_steps",
            "action_dim",
            "temporal_resampling",
            "native_action_targets",
            "split",
        )
    }
    _protocol_metadata(metadata, split=str(batch.get("split", "train")))
    _require(batch.get("supervision_mode") == "action", "paired action batch mode changed")
    video = _require_tensor(batch, "video", label="paired action batch")
    action = _require_tensor(batch, "action", label="paired action batch")
    state = _require_tensor(batch, "state_window", label="paired action batch")
    proprio = _require_tensor(batch, "proprio", label="paired action batch")
    context = _require_tensor(batch, "context", label="paired action batch")
    context_mask = _require_tensor(batch, "context_mask", label="paired action batch")
    action_is_pad = _require_tensor(batch, "action_is_pad", label="paired action batch")
    _require(video.ndim == 6 and video.shape[1:3] == (POLICY_VIEW_COUNT, 3), "paired video must be [G,4,3,T,H,W]")
    groups = int(video.shape[0])
    _require(tuple(action.shape) == (groups, POLICY_VIEW_COUNT, POLICY_ACTION_STEPS, POLICY_ACTION_DIM), "paired action must be [G,4,32,14]")
    _require(tuple(state.shape) == (groups, POLICY_VIEW_COUNT, POLICY_ACTION_STEPS + 1, POLICY_ACTION_DIM), "paired state_window must be [G,4,33,14]")
    _require(tuple(proprio.shape) == (groups, POLICY_VIEW_COUNT, POLICY_ACTION_STEPS, POLICY_ACTION_DIM), "paired proprio must be [G,4,32,14]")
    _require(context.ndim == 4 and context.shape[:2] == (groups, POLICY_VIEW_COUNT), "paired context must be [G,4,L,D]")
    _require(context_mask.ndim == 3 and context_mask.shape[:2] == (groups, POLICY_VIEW_COUNT), "paired context_mask must be [G,4,L]")
    _require(tuple(action_is_pad.shape) == (groups, POLICY_VIEW_COUNT, POLICY_ACTION_STEPS), "paired action_is_pad must be [G,4,32]")
    for tensor, label in (
        (action, "action"),
        (state, "state"),
        (proprio, "proprio"),
        (context, "context"),
        (context_mask, "context_mask"),
        (action_is_pad, "action_is_pad"),
    ):
        reference = tensor[:, :1]
        _require(torch.equal(tensor, reference.expand_as(tensor)), f"C/R1/R2/R3 {label} is not exact")


def collate_paired_action_groups(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(samples), "cannot collate an empty paired action batch")
    tensor_keys = (
        "video",
        "action",
        "proprio",
        "context",
        "context_mask",
        "action_is_pad",
        "state_window",
        "raw_action_window",
    )
    result: dict[str, Any] = {
        key: torch.stack([_require_tensor(sample, key, label="paired sample") for sample in samples], dim=0)
        for key in tensor_keys
    }
    result.update(
        {
            "variant_names": POLICY_VARIANTS,
            "protocol_id": POLICY_PROTOCOL_ID,
            "view_count": POLICY_VIEW_COUNT,
            "r3_role": POLICY_R3_ROLE,
            "camera_names": POLICY_CAMERA_NAMES,
            "camera_count": POLICY_CAMERA_COUNT,
            "native_fps": POLICY_NATIVE_FPS,
            "action_steps": POLICY_ACTION_STEPS,
            "action_dim": POLICY_ACTION_DIM,
            "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
            "native_action_targets": True,
            "supervision_mode": "action",
            "split": str(samples[0]["split"]),
            "task": tuple(str(sample["task"]) for sample in samples),
            "physical_state_id": tuple(str(sample["physical_state_id"]) for sample in samples),
            "trajectory_id": tuple(str(sample["trajectory_id"]) for sample in samples),
            "content_id": torch.tensor([int(sample["content_id"]) for sample in samples]),
            "frame_offset": torch.tensor([int(sample["frame_offset"]) for sample in samples]),
            "episode_indices": tuple(sample["episode_indices"] for sample in samples),
            "base_indices": tuple(sample["base_indices"] for sample in samples),
            "prompt": tuple(str(sample["prompt"]) for sample in samples),
        }
    )
    validate_native_paired_action_batch(result)
    return result


def flatten_paired_action_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten ``[physical state, scene]`` into a native action batch."""

    validate_native_paired_action_batch(batch)
    video = batch["video"]
    groups, views = int(video.shape[0]), int(video.shape[1])
    result: dict[str, Any] = {}
    for key in ("video", "action", "proprio", "context", "context_mask", "action_is_pad"):
        value = batch[key]
        result[key] = value.reshape(groups * views, *value.shape[2:])
    for optional in ("image_is_pad", "proprio_is_pad"):
        value = batch.get(optional)
        if isinstance(value, torch.Tensor):
            result[optional] = value.reshape(groups * views, *value.shape[2:])
    return result


class SameTaskPhysicalStateBatchSampler(Sampler[list[int]]):
    """Emit same-task batches containing distinct physical states."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        groups_per_batch: int,
        seed: int = 0,
        drop_last: bool = True,
        batches_per_epoch: int | None = None,
        balanced_round_robin: bool = True,
    ) -> None:
        self.groups_per_batch = int(groups_per_batch)
        _require(self.groups_per_batch >= 2, "paired batches require at least two physical states")
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if batches_per_epoch is not None:
            _require(int(batches_per_epoch) > 0, "batches_per_epoch must be positive")
            batches_per_epoch = int(batches_per_epoch)
        self.batches_per_epoch = batches_per_epoch
        self.balanced_round_robin = bool(balanced_round_robin)
        self._epoch = 0
        groups = getattr(dataset, "indices_by_task", None)
        _require(isinstance(groups, Mapping), "paired dataset lacks indices_by_task")
        original_indices = {
            str(task): tuple(int(index) for index in indices)
            for task, indices in groups.items()
        }
        _require(bool(original_indices), "paired dataset contains no tasks")
        state_id_lookup = getattr(dataset, "physical_state_id_for_index", None)
        _require(
            callable(state_id_lookup),
            "paired dataset lacks metadata-only physical_state_id_for_index",
        )
        self._indices_by_task: dict[str, tuple[int, ...]] = {}
        for task, indices in original_indices.items():
            state_index_pairs = sorted(
                (str(state_id_lookup(index)), index) for index in indices
            )
            state_ids = {state_id for state_id, _ in state_index_pairs}
            _require(len(state_ids) == len(indices), f"task {task!r} contains duplicate physical states")
            self._indices_by_task[task] = tuple(index for _, index in state_index_pairs)
            _require(len(indices) >= self.groups_per_batch, f"task {task!r} has {len(indices)} states; need {self.groups_per_batch}")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        if self.drop_last:
            return sum(len(indices) // self.groups_per_batch for indices in self._indices_by_task.values())
        return sum(math.ceil(len(indices) / self.groups_per_batch) for indices in self._indices_by_task.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch * 1_000_003)
        tasks = sorted(self._indices_by_task)
        if self.batches_per_epoch is not None:
            task_order = list(tasks)
            rng.shuffle(task_order)
            for batch_index in range(self.batches_per_epoch):
                task = task_order[batch_index % len(task_order)] if self.balanced_round_robin else rng.choice(tasks)
                yield rng.sample(list(self._indices_by_task[task]), self.groups_per_batch)
            return
        batches: list[list[int]] = []
        for task in tasks:
            task_indices = list(self._indices_by_task[task])
            rng.shuffle(task_indices)
            full_stop = len(task_indices) - len(task_indices) % self.groups_per_batch
            for start in range(0, full_stop, self.groups_per_batch):
                batches.append(task_indices[start : start + self.groups_per_batch])
            remainder = task_indices[full_stop:]
            if remainder and not self.drop_last:
                candidates = [index for index in task_indices if index not in remainder]
                rng.shuffle(candidates)
                needed = self.groups_per_batch - len(remainder)
                _require(len(candidates) >= needed, f"cannot complete a distinct-state batch for {task}")
                batches.append(remainder + candidates[:needed])
        rng.shuffle(batches)
        yield from batches


class DualStreamIterator(Iterator[dict[str, Any]]):
    """Cycle official and paired loaders independently without concatenating."""

    def __init__(self, official: Iterable[Any], paired: Iterable[Any]) -> None:
        self._sources = {"official": official, "paired": paired}
        self._iterators = {name: iter(source) for name, source in self._sources.items()}
        self._cycles = {"official": 0, "paired": 0}

    @property
    def cycles(self) -> dict[str, int]:
        return dict(self._cycles)

    def __iter__(self) -> "DualStreamIterator":
        return self

    def _next_from(self, name: str) -> Any:
        iterator = self._iterators[name]
        try:
            return next(iterator)
        except StopIteration:
            self._cycles[name] += 1
            replacement = iter(self._sources[name])
            self._iterators[name] = replacement
            try:
                return next(replacement)
            except StopIteration as exc:
                raise DataContractError(f"{name} stream is empty or not re-iterable") from exc

    def __next__(self) -> dict[str, Any]:
        return {"official": self._next_from("official"), "paired": self._next_from("paired")}


def audit_file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    _require(source.is_file(), f"audit artifact is not a file: {source}")
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(source),
    }


def audit_artifacts(**artifacts: str | Path) -> dict[str, dict[str, Any]]:
    _require(bool(artifacts), "at least one artifact is required")
    return {str(name): audit_file_identity(path) for name, path in sorted(artifacts.items())}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return repr(value)


def audit_frozen_token_cache(
    cache: str | Path | FrozenPairedTokenDataset,
    *,
    state_bank: VerifiedPolicyStateBank | None = None,
    expected_extraction_contract: Mapping[str, Any] | None = None,
    layer: int = 16,
    split: str = "train",
    verified_cache_identity: Mapping[str, Any] | None = None,
    expected_backbone_sha256: str | None = None,
    expected_base_lineage_sha256: str | None = None,
    expected_release_paired_binding_sha256: str | None = None,
    expected_action_manifest_sha256: str | None = None,
    expected_action_audit_sha256: str | None = None,
    expected_state_bank_sha256: str | None = None,
) -> dict[str, Any]:
    dataset = (
        cache
        if isinstance(cache, FrozenPairedTokenDataset)
        else FrozenPairedTokenDataset(
            cache,
            state_bank=state_bank,
            expected_extraction_contract=expected_extraction_contract,
            layer=layer,
            split=split,
            expected_backbone_sha256=expected_backbone_sha256,
            expected_base_lineage_sha256=expected_base_lineage_sha256,
            expected_release_paired_binding_sha256=(
                expected_release_paired_binding_sha256
            ),
            expected_action_manifest_sha256=expected_action_manifest_sha256,
            expected_action_audit_sha256=expected_action_audit_sha256,
            expected_state_bank_sha256=expected_state_bank_sha256,
        )
    )
    if expected_backbone_sha256 is not None:
        _require(dataset.backbone_sha256 == expected_backbone_sha256, "cache/base SHA-256 mismatch")
    if expected_base_lineage_sha256 is not None:
        _require(
            dataset.base_lineage_sha256 == expected_base_lineage_sha256,
            "cache/base-lineage SHA-256 mismatch",
        )
    if expected_release_paired_binding_sha256 is not None:
        _require(
            dataset.release_paired_binding_sha256
            == expected_release_paired_binding_sha256,
            "cache/release-paired-binding SHA-256 mismatch",
        )
    if expected_action_manifest_sha256 is not None:
        _require(dataset.action_manifest_sha256 == expected_action_manifest_sha256, "cache/action-manifest SHA-256 mismatch")
    if expected_action_audit_sha256 is not None:
        _require(dataset.action_audit_sha256 == expected_action_audit_sha256, "cache/action-audit SHA-256 mismatch")
    if expected_state_bank_sha256 is not None:
        _require(dataset.state_bank_sha256 == expected_state_bank_sha256, "cache/state-bank SHA-256 mismatch")
    if expected_extraction_contract is not None:
        _require(
            dataset._extraction_contract == dict(expected_extraction_contract),  # noqa: SLF001
            "cache/current extraction contract mismatch",
        )
    task_counts = {task: len(indices) for task, indices in sorted(dataset.indices_by_task.items())}
    if verified_cache_identity is None:
        cache_identity = audit_file_identity(dataset.cache_path)
    else:
        cache_identity = dict(verified_cache_identity)
        _require(Path(str(cache_identity.get("path", ""))).expanduser().resolve() == dataset.cache_path, "verified paired-cache identity path differs")
        _require(int(cache_identity.get("size_bytes", -1)) == dataset.cache_path.stat().st_size, "verified paired-cache identity size is stale")
        _require(_valid_sha256(cache_identity.get("sha256")), "verified paired-cache identity lacks SHA-256")
    report = {
        "audit_schema_version": 2,
        "kind": "policy_frozen_paired_token_cache",
        "cache": cache_identity,
        "cache_schema": POLICY_TOKEN_CACHE_SCHEMA,
        "cache_schema_version": POLICY_TOKEN_CACHE_SCHEMA_VERSION,
        "protocol_id": POLICY_PROTOCOL_ID,
        "split": dataset.split,
        "layer": dataset.layer,
        "variant_names": list(POLICY_VARIANTS),
        "view_count": POLICY_VIEW_COUNT,
        "r3_role": POLICY_R3_ROLE,
        "r3_training_positive": True,
        "camera_names": list(POLICY_CAMERA_NAMES),
        "camera_count": POLICY_CAMERA_COUNT,
        "native_fps": POLICY_NATIVE_FPS,
        "action_steps": POLICY_ACTION_STEPS,
        "action_dim": POLICY_ACTION_DIM,
        "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
        "native_action_targets": True,
        "backbone_checkpoint_sha256": dataset.backbone_sha256,
        "base_lineage_manifest_sha256": dataset.base_lineage_sha256,
        "release_paired_binding_manifest_sha256": (
            dataset.release_paired_binding_sha256
        ),
        "paired_action_manifest_sha256": dataset.action_manifest_sha256,
        "paired_action_audit_sha256": dataset.action_audit_sha256,
        "paired_state_bank_sha256": dataset.state_bank_sha256,
        "physical_state_inventory_sha256": dataset.physical_state_inventory_sha256,
        "record_count": len(dataset) * POLICY_VIEW_COUNT,
        "physical_state_count": len(dataset),
        "physical_states_by_task": task_counts,
        "token_group_shape": [POLICY_VIEW_COUNT, *dataset.token_shape],
        "token_dtype": str(dataset.token_dtype),
        "supervision_mode": "contrastive",
        "cache_provenance": _json_safe(dataset.provenance),
        "extraction_contract": _json_safe(dataset._extraction_contract),  # noqa: SLF001
        "native_prefill_identity_audit": _json_safe(
            dataset._native_prefill_identity_audit  # noqa: SLF001
        ),
    }
    json.dumps(report, sort_keys=True)
    return report


def audit_native_paired_action_dataset(dataset: NativePairedActionDataset) -> dict[str, Any]:
    _require(isinstance(dataset, NativePairedActionDataset), "expected NativePairedActionDataset")
    report = dataset.audit_report
    _require(report["r3_training_positive"] is True, "action audit lost R3 training-positive role")
    json.dumps(report, sort_keys=True)
    return report


def build_dual_stream_provenance(
    *,
    official: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    _require(bool(official), "official-stream provenance is empty")
    _require(paired.get("protocol_id") == POLICY_PROTOCOL_ID, "paired provenance protocol changed")
    _require(tuple(str(value) for value in paired.get("variant_names", ())) == POLICY_VARIANTS, "paired provenance is not exact C/R1/R2/R3")
    _require(paired.get("r3_training_positive") is True, "paired provenance does not prove R3 training-positive")
    mode = str(paired.get("supervision_mode", ""))
    _require(mode in {"action", "contrastive"}, "paired provenance has invalid supervision mode")
    role = "policy_action_supervision" if mode == "action" else "content_invariance_supervision"
    body = {
        "audit_schema_version": 2,
        "official": _json_safe(official),
        "paired": _json_safe(paired),
        "stream_contract": {
            "concatenated": False,
            "official_role": "policy_action_supervision",
            "paired_role": role,
            "paired_supervision_mode": mode,
            "cycling": "independent",
        },
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**body, "audit_sha256": hashlib.sha256(canonical).hexdigest()}


FrozenTokenGroupDataset = FrozenPairedTokenDataset
SameTaskBatchSampler = SameTaskPhysicalStateBatchSampler


__all__ = [
    "DataContractError",
    "DualStreamIterator",
    "FrozenPairedTokenDataset",
    "FrozenTokenGroupDataset",
    "NativePairedActionDataset",
    "NativePairedEpisodeGroup",
    "PolicyPhysicalStateAnchor",
    "SameTaskBatchSampler",
    "SameTaskPhysicalStateBatchSampler",
    "VerifiedNativeActionManifest",
    "VerifiedPolicyStateBank",
    "audit_artifacts",
    "audit_file_identity",
    "audit_frozen_token_cache",
    "audit_native_paired_action_dataset",
    "audit_native_paired_action_contract",
    "audit_policy_state_bank",
    "build_dual_stream_provenance",
    "build_policy_cache_extraction_contract",
    "canonical_artifact_binding",
    "canonical_fastwam_source_binding",
    "canonical_json_sha256",
    "collate_paired_action_groups",
    "collate_paired_token_groups",
    "flatten_paired_action_batch",
    "physical_state_inventory_sha256",
    "policy_cache_extractor_config",
    "policy_cache_preprocessing_contract",
    "policy_state_bank_offsets",
    "selected_episode_artifact_aggregate",
    "validate_native_paired_action_batch",
    "verify_native_paired_action_manifest",
    "verify_policy_state_bank",
]
