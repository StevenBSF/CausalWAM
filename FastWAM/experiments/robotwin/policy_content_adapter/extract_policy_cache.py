#!/usr/bin/env python3
"""Extract the Policy-v2 four-scene Layer-16 cache from the author release.

Default execution audits identities only and never initializes a GPU model.
Actual extraction requires ``--extract-policy-cache``.  The train split uses
exactly eight deterministic, unpadded physical states per trajectory; each
state is emitted in ordered C/R1/R2/R3 form with Layer-16 shape [120,3072].
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.robotwin.e0_e1.backbone import (
    FrozenFastWAMExtractor,
    format_deployment_prompt,
)
from experiments.robotwin.policy_content_adapter.data import (
    NativePairedActionDataset,
    VerifiedPolicyStateBank,
    audit_native_paired_action_contract,
    build_policy_cache_extraction_contract,
    selected_episode_artifact_aggregate,
    verify_native_paired_action_manifest,
    verify_policy_state_bank,
)
from experiments.robotwin.policy_content_adapter.native50hz_paired import (
    TASK_INSTRUCTIONS,
    atomic_write_json,
)
from experiments.robotwin.policy_content_adapter.model import artifact_identity
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS
from experiments.robotwin.policy_content_adapter.release_lineage import (
    verify_author_release_lineage,
)
from experiments.robotwin.policy_content_adapter.release_paired_binding import (
    verify_release_paired_binding,
)
from experiments.robotwin.policy_content_adapter.protocol import (
    POLICY_ACTION_DIM,
    POLICY_ACTION_STEPS,
    POLICY_CAMERA_COUNT,
    POLICY_CAMERA_NAMES,
    POLICY_NATIVE_FPS,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_STATE_BANK_SAMPLING_ALGORITHM,
    POLICY_STATE_BANK_SEED,
    POLICY_TEMPORAL_RESAMPLING,
    POLICY_TOKEN_CACHE_SCHEMA,
    POLICY_TOKEN_CACHE_SCHEMA_VERSION,
    POLICY_VARIANTS,
    POLICY_VIEW_COUNT,
)
from experiments.robotwin.policy_content_adapter.prepare_release_paired_text_cache import (
    verify_release_paired_text_cache,
)
from experiments.robotwin.policy_content_adapter.runtime_utils import (
    PROJECT_ROOT,
    audit_local_fastwam_source,
    compose_robotwin_config,
    instantiate_native_paired_action_dataset,
)


CAPTURE_LAYER = 16
TOKEN_COUNT = 120
TOKEN_DIM = 3072
STATES_PER_TRAJECTORY = 8
SPLIT = "train"


class PolicyCacheContractError(RuntimeError):
    """The cache cannot be proven to come from the release/native pair contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyCacheContractError(message)


@contextlib.contextmanager
def _cache_dataset_initialization_work_dir(output_path: str | Path):
    """Provide the native dataset a transaction-local stats mirror path."""

    from fastwam.utils import misc

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_work_dir = getattr(misc, "_WORK_DIR", None)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.dataset-init-",
            dir=output.parent,
        ) as temporary:
            staging = Path(temporary).resolve()
            misc.register_work_dir(staging)
            _require(staging.is_dir(), "cache dataset-init work directory was not created")
            yield staging
    finally:
        # ``register_work_dir(None)`` is invalid in the upstream helper; restore
        # the exact prior sentinel without creating the generic ./runs side effect.
        misc._WORK_DIR = previous_work_dir  # noqa: SLF001 - scoped upstream compatibility


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def runtime_component_identities(extractor: FrozenFastWAMExtractor) -> dict[str, Any]:
    model_paths = getattr(extractor.model, "model_paths", None)
    _require(isinstance(model_paths, Mapping), "extractor model lacks runtime component paths")
    identities: dict[str, Any] = {}
    for component in ("vae", "text_encoder", "tokenizer"):
        raw_path = model_paths.get(component)
        _require(
            isinstance(raw_path, (str, Path)) and str(raw_path) not in {"", "None"},
            f"extractor runtime component path is missing: {component}",
        )
        identity = artifact_identity(raw_path)
        _require(_valid_sha256(identity.get("sha256")), f"invalid {component} SHA-256")
        identities[component] = identity
    return identities


def verify_release_base_lineage(
    base_lineage_manifest: str | Path,
    *,
    checkpoint: str | Path,
    dataset_stats: str | Path,
    official_manifest: str | Path,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Hash-bind cache extraction to the immutable author-release ancestry."""

    try:
        return verify_author_release_lineage(
            base_lineage_manifest,
            checkpoint_path=checkpoint,
            dataset_stats_path=dataset_stats,
            official_manifest_path=official_manifest,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except Exception as exc:
        raise PolicyCacheContractError(
            f"author-release base lineage verification failed: {exc}"
        ) from exc


def audit_policy_cache_inputs(
    *,
    base_lineage_manifest: str | Path,
    release_paired_binding: str | Path,
    checkpoint: str | Path,
    dataset_stats: str | Path,
    official_manifest: str | Path,
    dataset_root: str | Path,
    paired_manifest: str | Path,
    paired_audit: str | Path,
    paired_state_bank: str | Path,
    expected_base_lineage_sha256: str | None = None,
    expected_release_paired_binding_sha256: str | None = None,
) -> dict[str, Any]:
    base_lineage = verify_release_base_lineage(
        base_lineage_manifest,
        checkpoint=checkpoint,
        dataset_stats=dataset_stats,
        official_manifest=official_manifest,
        expected_manifest_sha256=expected_base_lineage_sha256,
    )
    paired = audit_native_paired_action_contract(
        dataset_root=dataset_root,
        manifest_path=paired_manifest,
        audit_path=paired_audit,
        expected_tasks=OFFICIAL_TASKS,
        require_full_protocol_counts=True,
    )
    verified = verify_native_paired_action_manifest(
        paired_manifest,
        dataset_root=dataset_root,
        audit_path=paired_audit,
    )
    state_bank = verify_policy_state_bank(
        paired_state_bank,
        native_manifest=verified,
        expected_tasks=OFFICIAL_TASKS,
    )
    train_counts = {
        task: sum(group.task == task and group.split == SPLIT for group in verified.groups)
        for task in OFFICIAL_TASKS
    }
    _require(train_counts == {task: 30 for task in OFFICIAL_TASKS}, "train split is not 30 trajectories/task")
    selected_artifacts = selected_episode_artifact_aggregate(verified, split=SPLIT)
    _require(int(selected_artifacts["episode_count"]) == 360, "train artifact inventory must bind 360 scenes")
    _require(int(selected_artifacts["file_count"]) == 1_440, "train artifact inventory must bind 1,440 files")
    try:
        release_binding = verify_release_paired_binding(
            release_paired_binding,
            expected_sha256=expected_release_paired_binding_sha256,
        )
    except Exception as exc:
        raise PolicyCacheContractError(
            f"release/paired binding verification failed: {exc}"
        ) from exc
    binding_lineage = release_binding.get("base_lineage")
    binding_dataset = release_binding.get("paired_dataset")
    binding_selected = release_binding.get("selected_train_artifacts")
    binding_cache = release_binding.get("cache_protocol")
    _require(isinstance(binding_lineage, Mapping), "binding base lineage is missing")
    _require(isinstance(binding_dataset, Mapping), "binding paired dataset is missing")
    _require(isinstance(binding_selected, Mapping), "binding train inventory is missing")
    _require(isinstance(binding_cache, Mapping), "binding cache protocol is missing")
    _require(
        binding_lineage.get("sha256")
        == base_lineage["manifest_identity"]["sha256"],
        "binding refers to a different author-release lineage",
    )
    _require(
        binding_lineage.get("checkpoint", {}).get("sha256")
        == base_lineage["checkpoint"]["sha256"],
        "binding refers to a different author-release checkpoint",
    )
    _require(
        binding_lineage.get("dataset_stats", {}).get("sha256")
        == base_lineage["dataset_stats"]["sha256"],
        "binding refers to different release dataset stats",
    )
    _require(
        Path(str(binding_dataset.get("root", ""))).expanduser().resolve()
        == Path(dataset_root).expanduser().resolve(),
        "binding refers to a different paired root",
    )
    expected_dataset_hashes = {
        "native_action_manifest_sha256": verified.sha256,
        "native_action_audit_sha256": verified.audit_sha256,
        "state_bank_sha256": state_bank.sha256,
        "physical_state_inventory_sha256": state_bank.physical_state_inventory_sha256,
    }
    for key, expected in expected_dataset_hashes.items():
        _require(binding_dataset.get(key) == expected, f"binding {key} differs")
    for key in ("algorithm", "episode_count", "file_count", "size_bytes", "sha256"):
        _require(
            binding_selected.get(key) == selected_artifacts.get(key),
            f"binding selected_train_artifacts.{key} differs",
        )
    _require(
        dict(binding_cache)
        == {
            "capture_layer": CAPTURE_LAYER,
            "states_per_trajectory": STATES_PER_TRAJECTORY,
            "physical_state_groups": 720,
            "scene_views": 2_880,
            "view_token_shape": [TOKEN_COUNT, TOKEN_DIM],
        },
        "binding cache protocol differs",
    )
    return {
        "status": "PASS",
        "kind": "policy_four_scene_cache_input_audit",
        "protocol_id": POLICY_PROTOCOL_ID,
        "split": SPLIT,
        "states_per_trajectory": STATES_PER_TRAJECTORY,
        "capture_layer": CAPTURE_LAYER,
        "token_shape": [TOKEN_COUNT, TOKEN_DIM],
        "old_e0_e3_cache_accepted": False,
        "temporal_resampling": POLICY_TEMPORAL_RESAMPLING,
        "base_lineage": base_lineage,
        "release_paired_binding": release_binding,
        "paired": paired,
        "paired_action_manifest_sha256": verified.sha256,
        "paired_action_audit_sha256": verified.audit_sha256,
        "paired_state_bank": {
            "path": str(state_bank.path),
            "sha256": state_bank.sha256,
            "physical_state_inventory_sha256": state_bank.physical_state_inventory_sha256,
            "anchor_count": len(state_bank.anchors),
        },
        "selected_episode_artifacts": selected_artifacts,
        "train_trajectories_by_task": train_counts,
    }


def select_eight_states_per_trajectory(
    dataset: NativePairedActionDataset,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Compatibility alias that resolves the already-verified shared bank.

    Policy-v2 never recomputes or independently samples cache offsets.  The
    dataset constructor has already verified and materialized the immutable
    state bank, so this helper delegates to that exact inventory.
    """

    _require(isinstance(dataset, NativePairedActionDataset), "expected native paired dataset")
    state_bank = getattr(dataset, "_state_bank", None)  # noqa: SLF001 - audited compatibility path
    _require(isinstance(state_bank, VerifiedPolicyStateBank), "paired dataset lacks verified state bank")
    return select_indices_from_verified_state_bank(dataset, state_bank)


def select_indices_from_verified_state_bank(
    dataset: NativePairedActionDataset,
    state_bank: VerifiedPolicyStateBank,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Resolve the one shared immutable state inventory into dataset indices."""

    _require(isinstance(dataset, NativePairedActionDataset), "expected native paired dataset")
    _require(isinstance(state_bank, VerifiedPolicyStateBank), "expected verified Policy state bank")
    by_id = {
        record.physical_state_id: (index, record)
        for index, record in enumerate(dataset._records)  # noqa: SLF001
    }
    _require(len(by_id) == len(dataset), "paired dataset contains duplicate physical states")
    selected: list[int] = []
    plan: list[dict[str, Any]] = []
    for anchor in state_bank.anchors:
        _require(anchor.physical_state_id in by_id, f"state-bank anchor absent: {anchor.physical_state_id}")
        dataset_index, record = by_id[anchor.physical_state_id]
        _require(record.task == anchor.task, "state-bank task differs from native record")
        _require(record.trajectory_id == anchor.trajectory_id, "state-bank trajectory differs")
        selected.append(dataset_index)
        plan.append(
            {
                **anchor.as_dict(),
                "physical_state_id": anchor.physical_state_id,
                "dataset_index": dataset_index,
                "episode_indices": list(record.episode_indices),
                "split": record.split,
            }
        )
    _require(
        len(selected) == len(state_bank.anchors),
        "resolved selection differs from the verified state-bank anchor count",
    )
    return selected, plan


def _policy_metadata() -> dict[str, Any]:
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
        "split": SPLIT,
    }


def validate_policy_cache_payload(
    payload: Mapping[str, Any],
    *,
    expected_extraction_contract: Mapping[str, Any],
    expected_backbone_sha256: str,
    expected_base_lineage_sha256: str,
    expected_release_paired_binding_sha256: str,
    expected_manifest_sha256: str,
    expected_audit_sha256: str,
    expected_state_bank_sha256: str,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    _require(payload.get("schema") == POLICY_TOKEN_CACHE_SCHEMA, "cache schema mismatch")
    _require(
        int(payload.get("schema_version", -1)) == POLICY_TOKEN_CACHE_SCHEMA_VERSION,
        "cache schema version mismatch",
    )
    _require(tuple(payload.get("variant_names", ())) == POLICY_VARIANTS, "variant order mismatch")
    provenance = payload.get("provenance")
    _require(isinstance(provenance, Mapping), "cache provenance missing")
    for key, expected in _policy_metadata().items():
        actual = tuple(provenance.get(key, ())) if key in ("variant_names", "camera_names") else provenance.get(key)
        normalized_expected = tuple(expected) if key in ("variant_names", "camera_names") else expected
        _require(actual == normalized_expected, f"cache policy metadata mismatch: {key}")
    backbone = provenance.get("backbone_checkpoint")
    _require(isinstance(backbone, Mapping), "backbone checkpoint provenance missing")
    _require(
        backbone.get("sha256") == expected_backbone_sha256,
        "cache backbone is not the selected author release",
    )
    base_lineage = provenance.get("base_lineage_manifest")
    _require(isinstance(base_lineage, Mapping), "cache base-lineage identity missing")
    _require(
        base_lineage.get("sha256") == expected_base_lineage_sha256,
        "cache author-release lineage identity mismatch",
    )
    release_binding = provenance.get("release_paired_binding_manifest")
    _require(
        isinstance(release_binding, Mapping),
        "cache release/paired binding identity missing",
    )
    _require(
        release_binding.get("sha256")
        == expected_release_paired_binding_sha256,
        "cache release/paired binding identity mismatch",
    )
    _require(
        provenance.get("paired_action_manifest_sha256") == expected_manifest_sha256,
        "cache paired manifest identity mismatch",
    )
    _require(
        provenance.get("paired_action_audit_sha256") == expected_audit_sha256,
        "cache paired audit identity mismatch",
    )
    _require(
        provenance.get("paired_state_bank_sha256") == expected_state_bank_sha256,
        "cache paired state-bank identity mismatch",
    )
    _require(
        provenance.get("physical_state_inventory_sha256") == expected_inventory_sha256,
        "cache physical-state inventory identity mismatch",
    )
    dataset_stats = provenance.get("dataset_stats")
    _require(isinstance(dataset_stats, Mapping), "cache dataset-stats identity missing")
    _require(_valid_sha256(dataset_stats.get("sha256")), "cache dataset-stats SHA-256 invalid")
    components = provenance.get("components")
    _require(isinstance(components, Mapping), "cache runtime component identities missing")
    _require(set(components) == {"vae", "text_encoder", "tokenizer"}, "cache components changed")
    for name, identity in components.items():
        _require(isinstance(identity, Mapping), f"cache {name} identity is not a mapping")
        _require(_valid_sha256(identity.get("sha256")), f"cache {name} SHA-256 invalid")
        _require(int(identity.get("size_bytes", -1)) > 0, f"cache {name} size is invalid")
    fastwam_source = provenance.get("fastwam_source")
    _require(
        isinstance(fastwam_source, Mapping)
        and fastwam_source.get("status") == "PASS"
        and int(fastwam_source.get("file_count", 0)) > 0
        and isinstance(fastwam_source.get("files"), Mapping),
        "cache FastWAM source audit is incomplete",
    )
    extractor_source = provenance.get("extractor_source")
    _require(isinstance(extractor_source, Mapping), "cache extractor source identity missing")
    _require(_valid_sha256(extractor_source.get("sha256")), "cache extractor source SHA-256 invalid")
    extraction_contract = provenance.get("extraction_contract")
    _require(
        isinstance(extraction_contract, Mapping)
        and extraction_contract.get("schema") == "policy_cache_extraction_contract_v2",
        "cache extraction contract missing",
    )
    _require(
        dict(extraction_contract) == dict(expected_extraction_contract),
        "cache extraction dependencies differ from the audited runtime/data contract",
    )
    runtime_artifacts = extraction_contract.get("runtime_artifacts")
    _require(
        isinstance(runtime_artifacts, Mapping)
        and set(runtime_artifacts)
        == {
            "base_lineage",
            "release_paired_binding",
            "dataset_stats",
            "vae",
            "text_encoder",
            "tokenizer",
            "text_cache",
        },
        "cache extraction runtime-artifact bindings changed",
    )
    for name, identity in runtime_artifacts.items():
        _require(isinstance(identity, Mapping), f"cache extraction identity missing: {name}")
        _require(_valid_sha256(identity.get("sha256")), f"cache extraction SHA invalid: {name}")
    for key in ("extractor_config", "preprocessing_contract"):
        contract = extraction_contract.get(key)
        _require(isinstance(contract, Mapping), f"cache extraction {key} missing")
        _require(_valid_sha256(contract.get("sha256")), f"cache extraction {key} SHA invalid")
        _require(
            contract.get("sha256") == _canonical_json_sha256(contract.get("value")),
            f"cache extraction {key} canonical SHA mismatch",
        )
    selected_artifacts = extraction_contract.get("selected_episode_artifacts")
    _require(isinstance(selected_artifacts, Mapping), "cache selected episode identity missing")
    _require(int(selected_artifacts.get("episode_count", -1)) == 360, "cache must bind 360 train scene episodes")
    _require(int(selected_artifacts.get("file_count", -1)) == 1_440, "cache must bind 1,440 episode files")
    _require(_valid_sha256(selected_artifacts.get("sha256")), "selected episode aggregate SHA invalid")
    text_cache = provenance.get("text_cache")
    _require(isinstance(text_cache, Mapping), "cache text-cache identity missing")
    _require(int(text_cache.get("file_count", -1)) > 0, "cache text directory is empty")
    _require(_valid_sha256(text_cache.get("sha256")), "cache text-cache aggregate SHA invalid")
    text_cache_audit = provenance.get("paired_text_cache_audit")
    _require(
        isinstance(text_cache_audit, Mapping),
        "cache paired text-cache audit identity missing",
    )
    _require(
        _valid_sha256(text_cache_audit.get("sha256")),
        "cache paired text-cache audit SHA invalid",
    )
    native_prefill_audit = provenance.get("native_prefill_identity_audit")
    _require(
        isinstance(native_prefill_audit, Mapping)
        and native_prefill_audit.get("status") == "PASS"
        and int(native_prefill_audit.get("checked_states", -1)) == 1
        and float(native_prefill_audit.get("rtol", -1)) == 0.0
        and float(native_prefill_audit.get("atol", -1)) == 0.0,
        "cache did not prove one first-state bit-exact native prefill audit",
    )
    tokens_by_layer = payload.get("tokens_by_layer")
    _require(isinstance(tokens_by_layer, Mapping) and set(tokens_by_layer) == {"16"}, "only Layer16 is allowed")
    tokens = tokens_by_layer["16"]
    _require(isinstance(tokens, torch.Tensor), "Layer16 cache is not a tensor")
    _require(tokens.device.type == "cpu" and not tokens.requires_grad, "cache tokens must be frozen CPU")
    _require(tuple(tokens.shape[1:]) == (TOKEN_COUNT, TOKEN_DIM), "Layer16 token shape mismatch")
    # Chunking avoids a second multi-gigabyte float32/bool allocation while
    # preserving a complete finite-value scan of the 2,880 token records.
    finite_scan_chunk = 32
    for start in range(0, int(tokens.shape[0]), finite_scan_chunk):
        _require(
            bool(torch.isfinite(tokens[start : start + finite_scan_chunk].float()).all()),
            f"cache contains non-finite tokens in records {start}:{start + finite_scan_chunk}",
        )
    records = payload.get("records")
    states = payload.get("physical_states")
    proprio = payload.get("proprio_raw")
    _require(isinstance(records, Sequence), "cache records missing")
    _require(isinstance(states, Sequence), "cache physical states missing")
    _require(len(records) == len(states) * POLICY_VIEW_COUNT == tokens.shape[0], "cache group counts mismatch")
    _require(
        isinstance(proprio, torch.Tensor) and tuple(proprio.shape) == (len(states), 14),
        "cache proprio_raw shape mismatch",
    )
    _require(len(states) == 720, "train cache must contain 90 trajectories x 8 states")
    for group_index in range(len(states)):
        group = records[group_index * 4 : group_index * 4 + 4]
        _require(tuple(record["variant"] for record in group) == POLICY_VARIANTS, "record views unordered")
        identity = {
            (record["task"], record["physical_state_id"], record["frame_offset"], record["split"])
            for record in group
        }
        _require(len(identity) == 1, "four records do not share one physical state")
    return {
        "status": "PASS",
        "physical_state_count": len(states),
        "record_count": len(records),
        "layer16_shape": list(tokens.shape),
        "backbone_checkpoint_sha256": expected_backbone_sha256,
        "base_lineage_manifest_sha256": expected_base_lineage_sha256,
        "release_paired_binding_manifest_sha256": (
            expected_release_paired_binding_sha256
        ),
        "paired_action_manifest_sha256": expected_manifest_sha256,
        "paired_action_audit_sha256": expected_audit_sha256,
        "paired_state_bank_sha256": expected_state_bank_sha256,
        "physical_state_inventory_sha256": expected_inventory_sha256,
    }


def _normalized_video_to_uint8_current(video: torch.Tensor) -> torch.Tensor:
    _require(video.ndim == 5 and tuple(video.shape[:3]) == (4, 3, 9), "paired video shape changed")
    current = video[:, :, 0].detach().float().cpu()
    _require(float(current.min()) >= -1.001 and float(current.max()) <= 1.001, "video range invalid")
    return ((current + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def _save_cache_transactionally(path: Path, payload: Mapping[str, Any]) -> str:
    _require(not path.exists(), f"refusing to overwrite Policy cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_policy_cache(
    *,
    input_audit: Mapping[str, Any],
    dataset_root: str | Path,
    paired_manifest: str | Path,
    paired_audit: str | Path,
    paired_state_bank: str | Path,
    text_cache_dir: str | Path,
    output_path: str | Path,
    model_base_path: str | Path | None,
    device: str = "cuda",
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite Policy cache: {output}")
    text_cache = Path(text_cache_dir).expanduser().resolve()
    _require(text_cache.is_dir(), f"paired text cache not found: {text_cache}")
    base_lineage = input_audit["base_lineage"]
    release_binding = input_audit["release_paired_binding"]
    checkpoint = Path(base_lineage["checkpoint"]["path"])
    stats = Path(base_lineage["dataset_stats"]["path"])
    cfg = compose_robotwin_config()
    dataset_stats_identity = artifact_identity(stats)
    _require(
        dataset_stats_identity["sha256"]
        == base_lineage["dataset_stats"]["sha256"],
        "dataset stats changed after the input audit",
    )
    extractor_source = artifact_identity(Path(__file__))
    capture_source = artifact_identity(PROJECT_ROOT / "experiments/robotwin/e0_e1/backbone.py")
    runtime_helper_source = artifact_identity(
        PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/runtime_utils.py"
    )
    policy_data_source = artifact_identity(
        PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/data.py"
    )
    policy_protocol_source = artifact_identity(
        PROJECT_ROOT / "experiments/robotwin/policy_content_adapter/protocol.py"
    )
    fastwam_source = audit_local_fastwam_source()
    text_cache_identity = artifact_identity(text_cache)
    paired_text_cache_audit = verify_release_paired_text_cache(
        text_cache,
        expected_base_lineage_sha256=base_lineage["manifest_identity"]["sha256"],
        expected_release_paired_binding_sha256=release_binding[
            "binding_manifest_identity"
        ]["sha256"],
    )
    with _cache_dataset_initialization_work_dir(output):
        dataset = instantiate_native_paired_action_dataset(
            cfg,
            dataset_root=dataset_root,
            dataset_stats_path=stats,
            text_cache_dir=text_cache,
            model_for_on_the_fly_text=None,
            manifest_path=paired_manifest,
            audit_path=paired_audit,
            state_bank_path=paired_state_bank,
            expected_state_bank_sha256=input_audit["paired_state_bank"]["sha256"],
            split=SPLIT,
            expected_tasks=OFFICIAL_TASKS,
            require_full_protocol_counts=True,
        )
    verified_manifest = verify_native_paired_action_manifest(
        paired_manifest,
        dataset_root=dataset_root,
        audit_path=paired_audit,
    )
    verified_state_bank = verify_policy_state_bank(
        paired_state_bank,
        native_manifest=verified_manifest,
        expected_sha256=input_audit["paired_state_bank"]["sha256"],
        expected_tasks=OFFICIAL_TASKS,
    )
    selected_indices, selection_plan = select_indices_from_verified_state_bank(
        dataset, verified_state_bank
    )
    extractor = FrozenFastWAMExtractor.from_release_checkpoint(
        checkpoint,
        stats,
        model_base_path=model_base_path,
        device=device,
        capture_layers=(CAPTURE_LAYER,),
        # The first physical state proves the mirrored capture loop is exactly
        # identical to native MoT prefill.  The flag is disabled immediately
        # after that state to avoid doubling all 720 extraction passes.
        verify_native_prefill=True,
        compute_checkpoint_sha256=True,
    )
    components = runtime_component_identities(extractor)
    extraction_contract = build_policy_cache_extraction_contract(
        base_lineage_identity=base_lineage["manifest_identity"],
        release_paired_binding_identity=release_binding[
            "binding_manifest_identity"
        ],
        dataset_stats_identity=dataset_stats_identity,
        vae_identity=components["vae"],
        text_encoder_identity=components["text_encoder"],
        tokenizer_identity=components["tokenizer"],
        text_cache_identity=text_cache_identity,
        fastwam_source_audit=fastwam_source,
        extractor_source_identity=extractor_source,
        extractor_support_source_identities={
            "frozen_backbone": capture_source,
            "runtime_utils": runtime_helper_source,
            "policy_data": policy_data_source,
            "policy_protocol": policy_protocol_source,
        },
        selected_episode_artifacts=input_audit["selected_episode_artifacts"],
    )

    token_batches: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    physical_states: list[dict[str, Any]] = []
    proprio_rows: list[torch.Tensor] = []
    condition_provenance: dict[str, Any] = {}
    backbone_extraction: dict[str, Any] | None = None
    native_prefill_identity_audit: dict[str, Any] | None = None
    for output_index, (dataset_index, plan_row) in enumerate(
        zip(selected_indices, selection_plan, strict=True)
    ):
        sample = dataset[dataset_index]
        for field in (
            "task",
            "trajectory_id",
            "content_id",
            "frame_offset",
            "physical_state_id",
            "split",
        ):
            _require(
                sample[field] == plan_row[field],
                f"dataset sample differs from state-bank plan at {field}: "
                f"{sample[field]!r} vs {plan_row[field]!r}",
            )
        _require(
            list(sample["episode_indices"]) == list(plan_row["episode_indices"]),
            "dataset episode order differs from the state-bank plan",
        )
        _require(
            sample["prompt"]
            == format_deployment_prompt(TASK_INSTRUCTIONS[sample["task"]]),
            "paired sample prompt differs from the exported task instruction",
        )
        _require(not bool(sample["action_is_pad"].any()), "selected Policy state contains padding")
        images = _normalized_video_to_uint8_current(sample["video"])
        state_window = sample["state_window"]
        _require(tuple(state_window.shape) == (4, 33, 14), "state window shape changed")
        proprio_views = state_window[:, 0].detach().cpu().float()
        _require(torch.equal(proprio_views, proprio_views[0:1].expand_as(proprio_views)), "view proprio differs")
        output_tokens = extractor.extract_current_observations(
            images,
            proprio_views,
            context=sample["context"].detach().cpu(),
            context_mask=sample["context_mask"].detach().cpu(),
            proprio_mode="observed",
        )
        if output_index == 0:
            _require(
                output_tokens.provenance.get("native_prefill_verified") is True,
                "first-state native prefill identity check was not executed",
            )
            native_prefill_identity_audit = {
                "status": "PASS",
                "checked_states": 1,
                "checked_physical_state_id": str(sample["physical_state_id"]),
                "comparison": "bit_exact_K_and_V_for_every_layer",
                "rtol": 0.0,
                "atol": 0.0,
            }
            extractor.verify_native_prefill = False
        else:
            _require(
                output_tokens.provenance.get("native_prefill_verified") is False,
                "native prefill identity check ran beyond the first state",
            )
        layer_tokens = output_tokens.tokens_by_layer[CAPTURE_LAYER]
        _require(tuple(layer_tokens.shape) == (4, TOKEN_COUNT, TOKEN_DIM), "Layer16 shape changed")
        token_batches.append(layer_tokens)
        provenance = dict(output_tokens.provenance)
        condition = provenance.pop("condition")
        normalized_proprio = provenance.pop("normalized_proprio_sha256")
        if backbone_extraction is None:
            backbone_extraction = provenance
            loaded_sha = provenance.get("checkpoint", {}).get("sha256")
            _require(
                loaded_sha == base_lineage["checkpoint"]["sha256"],
                "extractor loaded wrong author-release backbone",
            )
        else:
            # Per-sample pixel range/shape may vary, but architecture and
            # checkpoint fields below remain fixed and are audited separately.
            _require(
                provenance.get("checkpoint") == backbone_extraction.get("checkpoint"),
                "extractor checkpoint provenance changed",
            )
            _require(
                provenance.get("capture_layers") == [CAPTURE_LAYER],
                "extractor capture layer changed",
            )
        state_id = str(sample["physical_state_id"])
        condition_provenance[state_id] = {
            "condition": condition,
            "normalized_proprio_sha256": normalized_proprio,
        }
        proprio_rows.append(proprio_views[0].clone())
        physical_states.append(
            {
                "task": sample["task"],
                "trajectory_id": sample["trajectory_id"],
                "content_id": int(sample["content_id"]),
                "frame_offset": int(sample["frame_offset"]),
                "physical_state_id": state_id,
                "proprio_raw": proprio_views[0].tolist(),
            }
        )
        for view_index, variant in enumerate(POLICY_VARIANTS):
            records.append(
                {
                    "task": sample["task"],
                    "physical_state_id": state_id,
                    "trajectory_id": sample["trajectory_id"],
                    "content_id": int(sample["content_id"]),
                    "frame_offset": int(sample["frame_offset"]),
                    "split": SPLIT,
                    "variant": variant,
                    "episode_index": int(sample["episode_indices"][view_index]),
                    "view_index": view_index,
                }
            )
        print(
            f"[{output_index + 1}/{len(selected_indices)}] {state_id} tokens={tuple(layer_tokens.shape)}",
            flush=True,
        )

    _require(backbone_extraction is not None, "cache selection unexpectedly empty")
    _require(
        isinstance(native_prefill_identity_audit, Mapping)
        and native_prefill_identity_audit.get("status") == "PASS"
        and native_prefill_identity_audit.get("checked_states") == 1,
        "native prefill identity audit did not complete exactly once",
    )
    tokens = torch.cat(token_batches, dim=0).detach().cpu().contiguous()
    provenance = {
        **_policy_metadata(),
        "cache_kind": "author_release_native50hz_four_scene_layer16",
        "capture_layer": CAPTURE_LAYER,
        "states_per_trajectory": STATES_PER_TRAJECTORY,
        "state_selection_algorithm": POLICY_STATE_BANK_SAMPLING_ALGORITHM,
        "state_selection_seed": POLICY_STATE_BANK_SEED,
        "backbone_checkpoint": dict(base_lineage["checkpoint"]),
        "base_lineage_manifest": dict(base_lineage["manifest_identity"]),
        "release_paired_binding_manifest": dict(
            release_binding["binding_manifest_identity"]
        ),
        "dataset_stats": dataset_stats_identity,
        "components": components,
        "fastwam_source": fastwam_source,
        "extractor_source": extractor_source,
        "capture_source": capture_source,
        "runtime_helper_source": runtime_helper_source,
        "policy_data_source": policy_data_source,
        "policy_protocol_source": policy_protocol_source,
        "selected_episode_artifacts": input_audit["selected_episode_artifacts"],
        "text_cache": text_cache_identity,
        "paired_text_cache_audit": paired_text_cache_audit["audit_identity"],
        "extraction_contract": extraction_contract,
        "paired_action_manifest_sha256": input_audit["paired_action_manifest_sha256"],
        "paired_action_audit_sha256": input_audit["paired_action_audit_sha256"],
        "paired_state_bank_sha256": verified_state_bank.sha256,
        "physical_state_inventory_sha256": (
            verified_state_bank.physical_state_inventory_sha256
        ),
        "selection_plan": selection_plan,
        "conditions_by_physical_state": condition_provenance,
        "extractor": backbone_extraction,
        "native_prefill_identity_audit": native_prefill_identity_audit,
        "legacy_e0_e3_cache_reused": False,
        "temporal_interpolation_used": False,
    }
    payload = {
        "schema": POLICY_TOKEN_CACHE_SCHEMA,
        "schema_version": POLICY_TOKEN_CACHE_SCHEMA_VERSION,
        "variant_names": list(POLICY_VARIANTS),
        "tokens_by_layer": {str(CAPTURE_LAYER): tokens},
        "records": records,
        "physical_states": physical_states,
        "proprio_raw": torch.stack(proprio_rows).float().cpu(),
        "provenance": provenance,
    }
    payload_audit = validate_policy_cache_payload(
        payload,
        expected_extraction_contract=extraction_contract,
        expected_backbone_sha256=base_lineage["checkpoint"]["sha256"],
        expected_base_lineage_sha256=base_lineage["manifest_identity"]["sha256"],
        expected_release_paired_binding_sha256=release_binding[
            "binding_manifest_identity"
        ]["sha256"],
        expected_manifest_sha256=input_audit["paired_action_manifest_sha256"],
        expected_audit_sha256=input_audit["paired_action_audit_sha256"],
        expected_state_bank_sha256=verified_state_bank.sha256,
        expected_inventory_sha256=verified_state_bank.physical_state_inventory_sha256,
    )
    cache_sha = _save_cache_transactionally(output, payload)
    result = {
        **payload_audit,
        "cache": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": cache_sha,
        },
        "input_audit": dict(input_audit),
    }
    atomic_write_json(output.with_suffix(output.suffix + ".audit.json"), result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lineage-manifest", required=True, type=Path)
    parser.add_argument("--base-lineage-sha256")
    parser.add_argument("--release-paired-binding", required=True, type=Path)
    parser.add_argument("--release-paired-binding-sha256")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-stats", required=True, type=Path)
    parser.add_argument("--official-manifest", required=True, type=Path)
    parser.add_argument("--paired-root", required=True, type=Path)
    parser.add_argument("--paired-manifest", required=True, type=Path)
    parser.add_argument("--paired-audit", required=True, type=Path)
    parser.add_argument("--state-bank", required=True, type=Path)
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--model-base-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states-per-trajectory", type=int, default=STATES_PER_TRAJECTORY)
    parser.add_argument("--layer", type=int, default=CAPTURE_LAYER)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--extract-policy-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _require(args.states_per_trajectory == STATES_PER_TRAJECTORY, "states-per-trajectory is locked to 8")
        _require(args.layer == CAPTURE_LAYER, "Policy cache layer is locked to 16")
        audit = audit_policy_cache_inputs(
            base_lineage_manifest=args.base_lineage_manifest,
            release_paired_binding=args.release_paired_binding,
            checkpoint=args.checkpoint,
            dataset_stats=args.dataset_stats,
            official_manifest=args.official_manifest,
            dataset_root=args.paired_root,
            paired_manifest=args.paired_manifest,
            paired_audit=args.paired_audit,
            paired_state_bank=args.state_bank,
            expected_base_lineage_sha256=args.base_lineage_sha256,
            expected_release_paired_binding_sha256=(
                args.release_paired_binding_sha256
            ),
        )
        if args.audit_output is not None:
            atomic_write_json(args.audit_output, audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not args.extract_policy_cache:
            print("Policy cache input audit PASS; no GPU model was loaded and no cache was written.", file=sys.stderr)
            return 0
        _require(args.output is not None, "--output is required for extraction")
        _require(args.text_cache_dir is not None, "--text-cache-dir is required for extraction")
        result = extract_policy_cache(
            input_audit=audit,
            dataset_root=args.paired_root,
            paired_manifest=args.paired_manifest,
            paired_audit=args.paired_audit,
            paired_state_bank=args.state_bank,
            text_cache_dir=args.text_cache_dir,
            output_path=args.output,
            model_base_path=args.model_base_path,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"Policy cache extraction failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPTURE_LAYER",
    "STATES_PER_TRAJECTORY",
    "TOKEN_COUNT",
    "TOKEN_DIM",
    "PolicyCacheContractError",
    "_cache_dataset_initialization_work_dir",
    "audit_policy_cache_inputs",
    "extract_policy_cache",
    "select_eight_states_per_trajectory",
    "select_indices_from_verified_state_bank",
    "validate_policy_cache_payload",
    "verify_release_base_lineage",
]
