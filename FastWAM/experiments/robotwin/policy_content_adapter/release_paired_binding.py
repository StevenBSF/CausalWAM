#!/usr/bin/env python3
"""Bind the audited native-50Hz paired dataset to the fixed author release.

This is the Phase-C gate for the release-base protocol.  It revalidates all
600 exported scene episodes, verifies the immutable 720-state bank, hashes
every train-split parquet/video consumed by cache extraction, and records the
exact author-release lineage.  It never initializes a GPU model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.robotwin.policy_content_adapter.data import (
    audit_native_paired_action_contract,
    selected_episode_artifact_aggregate,
    verify_native_paired_action_manifest,
    verify_policy_state_bank,
)
from experiments.robotwin.policy_content_adapter.model import artifact_identity
from experiments.robotwin.policy_content_adapter.native50hz_paired import (
    atomic_write_json,
    validate_lerobot_v21_root,
)
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS
from experiments.robotwin.policy_content_adapter.protocol import POLICY_PROTOCOL_ID
from experiments.robotwin.policy_content_adapter.release_lineage import (
    AUTHOR_RELEASE_LINEAGE_KIND,
    verify_author_release_lineage,
)


BINDING_SCHEMA_VERSION = 1
PAIR280_BINDING_SCHEMA_VERSION = 2
BINDING_KIND = "policy_release_native50hz_paired_binding"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReleasePairedBindingError(RuntimeError):
    """The selected release/dataset pair cannot prove the formal contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleasePairedBindingError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing/unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleasePairedBindingError(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _identity(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing/unsafe: {path}")
    value = artifact_identity(path)
    _require(value.get("kind") == "file", f"{label} must be a file")
    return value


def validate_release_paired_binding_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate semantic invariants of an already written binding manifest."""

    _require(isinstance(payload, Mapping), "binding payload must be an object")
    value = dict(payload)
    schema_version = int(value.get("schema_version", -1))
    _require(
        schema_version in {BINDING_SCHEMA_VERSION, PAIR280_BINDING_SCHEMA_VERSION},
        "binding schema changed",
    )
    _require(value.get("kind") == BINDING_KIND, "binding kind changed")
    _require(value.get("status") == "PASS", "binding status must be PASS")
    _require(value.get("protocol_id") == POLICY_PROTOCOL_ID, "Policy protocol id changed")
    lineage = value.get("base_lineage")
    _require(isinstance(lineage, Mapping), "base_lineage is missing")
    _require(lineage.get("kind") == AUTHOR_RELEASE_LINEAGE_KIND, "base lineage is not author release")
    _require(
        isinstance(lineage.get("sha256"), str)
        and SHA256_PATTERN.fullmatch(str(lineage["sha256"])) is not None,
        "base lineage SHA-256 is invalid",
    )
    dataset = value.get("paired_dataset")
    _require(isinstance(dataset, Mapping), "paired_dataset is missing")
    expected = {
        "scene_episode_count": 600,
        "physical_trajectory_count": 150,
        "train_physical_trajectory_count": 90,
        "scene_variants_per_state": 4,
        "native_fps": 50,
        "action_steps": 32,
        "action_dim": 14,
        "interpolation_used": False,
        "all_pairs_exact": True,
    }
    for key, expected_value in expected.items():
        _require(dataset.get(key) == expected_value, f"paired_dataset.{key} changed")
    expected_anchors = 720 if schema_version == BINDING_SCHEMA_VERSION else 25_200
    _require(
        dataset.get("state_bank_anchor_count") == expected_anchors,
        "paired_dataset.state_bank_anchor_count changed",
    )
    selected = value.get("selected_train_artifacts")
    _require(isinstance(selected, Mapping), "selected_train_artifacts is missing")
    _require(selected.get("episode_count") == 360, "train scene inventory must contain 360 episodes")
    _require(selected.get("file_count") == 1_440, "train artifact inventory must contain 1,440 files")
    _require(
        isinstance(selected.get("sha256"), str)
        and SHA256_PATTERN.fullmatch(str(selected["sha256"])) is not None,
        "selected artifact SHA-256 is invalid",
    )
    cache_protocol = value.get("cache_protocol")
    if schema_version == PAIR280_BINDING_SCHEMA_VERSION:
        _require(isinstance(cache_protocol, Mapping), "cache_protocol is missing")
    expected_cache = (
        {
            "capture_layer": 16,
            "states_per_trajectory": 8,
            "physical_state_groups": 720,
            "scene_views": 2_880,
            "view_token_shape": [120, 3_072],
        }
        if schema_version == BINDING_SCHEMA_VERSION
        else {
            "capture_layer": 16,
            "states_per_trajectory": 280,
            "physical_state_groups": 25_200,
            "scene_views": 100_800,
            "view_token_shape": [120, 3_072],
            "storage": "trajectory_sharded_safetensors_v1",
        }
    )
    if cache_protocol is not None:
        _require(isinstance(cache_protocol, Mapping), "cache_protocol is invalid")
        _require(dict(cache_protocol) == expected_cache, "binding cache_protocol changed")
    if schema_version == PAIR280_BINDING_SCHEMA_VERSION:
        parent = value.get("parent_binding")
        _require(isinstance(parent, Mapping), "Pair-280 binding parent_binding is missing")
        _require(
            isinstance(parent.get("sha256"), str)
            and SHA256_PATTERN.fullmatch(str(parent["sha256"])) is not None,
            "Pair-280 parent binding SHA-256 is invalid",
        )
    return value


def verify_release_paired_binding(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a binding manifest and verify its own immutable identity."""

    manifest = Path(path).expanduser().resolve()
    value = validate_release_paired_binding_payload(_load_json(manifest, "release paired binding"))
    identity = _identity(manifest, "release paired binding")
    if expected_sha256 is not None:
        _require(
            SHA256_PATTERN.fullmatch(str(expected_sha256)) is not None,
            "expected release paired binding SHA-256 is invalid",
        )
        _require(identity["sha256"] == expected_sha256, "release paired binding SHA-256 mismatch")
    value["binding_manifest_identity"] = identity
    return value


def build_release_paired_binding(
    *,
    base_lineage_manifest: str | Path,
    base_lineage_sha256: str,
    checkpoint: str | Path,
    dataset_stats: str | Path,
    official_manifest: str | Path,
    paired_root: str | Path,
) -> dict[str, Any]:
    """Run the complete CPU/data audit and return a canonical binding payload."""

    root = Path(paired_root).expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), f"paired root missing/unsafe: {root}")
    lineage_path = Path(base_lineage_manifest).expanduser().resolve()
    lineage = verify_author_release_lineage(
        lineage_path,
        checkpoint_path=checkpoint,
        dataset_stats_path=dataset_stats,
        official_manifest_path=official_manifest,
        expected_manifest_sha256=base_lineage_sha256,
    )
    lineage_identity = lineage["manifest_identity"]

    fresh_export = validate_lerobot_v21_root(
        root,
        expected_contents=50,
        require_output_name=False,
        validate_policy_consumer_contract=True,
    )
    _require(fresh_export.get("status") == "PASS", "fresh LeRobot validation did not pass")
    _require(fresh_export.get("episode_count") == 600, "full export must contain 600 scenes")
    _require(fresh_export.get("content_count_per_task") == 50, "full export must contain 50 trajectories/task")
    _require(fresh_export.get("fps") == 50, "paired export must be native 50 Hz")
    _require(fresh_export.get("future_action_horizon") == 32, "paired export must use 32-step targets")
    _require(fresh_export.get("action_dim") == 14, "paired export action dimension must be 14")
    _require(fresh_export.get("all_pairs_exact") is True, "paired export equality changed")
    _require(fresh_export.get("interpolation_used") is False, "paired export used interpolation")

    meta = root / "meta"
    native_manifest_path = meta / "policy_native_action_manifest.json"
    native_audit_path = meta / "policy_native_action_audit.json"
    state_bank_path = meta / "policy_paired_state_bank.json"
    native_contract = audit_native_paired_action_contract(
        dataset_root=root,
        manifest_path=native_manifest_path,
        audit_path=native_audit_path,
        expected_tasks=OFFICIAL_TASKS,
        require_full_protocol_counts=True,
    )
    native_manifest = verify_native_paired_action_manifest(
        native_manifest_path,
        dataset_root=root,
        audit_path=native_audit_path,
    )
    state_bank = verify_policy_state_bank(
        state_bank_path,
        native_manifest=native_manifest,
        expected_tasks=OFFICIAL_TASKS,
    )
    _require(len(native_manifest.groups) == 150, "paired manifest must contain 150 trajectories")
    train_groups = native_manifest.groups_for_split("train")
    _require(len(train_groups) == 90, "paired train split must contain 90 trajectories")
    _require(len(state_bank.anchors) == 720, "state bank must contain 720 anchors")

    selected = selected_episode_artifact_aggregate(native_manifest, split="train")
    _require(selected.get("episode_count") == 360, "selected train scenes must equal 360")
    _require(selected.get("file_count") == 1_440, "selected train files must equal 1,440")

    export_audit_path = meta / "export_audit.json"
    export_audit = _load_json(export_audit_path, "original export audit")
    required_export = {
        "status": "PASS",
        "episode_count": 600,
        "content_count_per_task": 50,
        "fps": 50,
        "future_action_horizon": 32,
        "action_dim": 14,
        "all_pairs_exact": True,
        "interpolation_used": False,
    }
    for key, expected in required_export.items():
        _require(export_audit.get(key) == expected, f"original export audit {key} changed")

    meta_artifact_paths = {
        "export_audit": export_audit_path,
        "native50hz_export_contract": meta / "native50hz_export_contract.json",
        "paired_contents": meta / "paired_contents.jsonl",
        "policy_native_action_manifest": native_manifest_path,
        "policy_native_action_audit": native_audit_path,
        "policy_paired_state_bank": state_bank_path,
    }
    meta_identities = {
        name: _identity(path, name) for name, path in meta_artifact_paths.items()
    }

    result = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "kind": BINDING_KIND,
        "status": "PASS",
        "protocol_id": POLICY_PROTOCOL_ID,
        "base_lineage": {
            "kind": AUTHOR_RELEASE_LINEAGE_KIND,
            "path": str(lineage_path),
            "size_bytes": int(lineage_identity["size_bytes"]),
            "sha256": str(lineage_identity["sha256"]),
            "lineage_id": lineage["lineage_id"],
            "checkpoint": dict(lineage["checkpoint"]),
            "dataset_stats": dict(lineage["dataset_stats"]),
        },
        "paired_dataset": {
            "root": str(root),
            "scene_episode_count": 600,
            "physical_trajectory_count": 150,
            "train_physical_trajectory_count": 90,
            "state_bank_anchor_count": 720,
            "scene_variants_per_state": 4,
            "native_fps": 50,
            "action_steps": 32,
            "action_dim": 14,
            "interpolation_used": False,
            "all_pairs_exact": True,
            "task_order": list(OFFICIAL_TASKS),
            "paired_contents_sha256": fresh_export["paired_manifest_sha256"],
            "native_action_manifest_sha256": native_manifest.sha256,
            "native_action_audit_sha256": native_manifest.audit_sha256,
            "state_bank_sha256": state_bank.sha256,
            "physical_state_inventory_sha256": state_bank.physical_state_inventory_sha256,
        },
        "fresh_export_validation": fresh_export,
        "native_action_contract": native_contract,
        "meta_artifacts": meta_identities,
        "selected_train_artifacts": selected,
        "cache_protocol": {
            "capture_layer": 16,
            "states_per_trajectory": 8,
            "physical_state_groups": 720,
            "scene_views": 2_880,
            "view_token_shape": [120, 3_072],
        },
    }
    return validate_release_paired_binding_payload(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lineage-manifest", required=True, type=Path)
    parser.add_argument("--base-lineage-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-stats", required=True, type=Path)
    parser.add_argument("--official-manifest", required=True, type=Path)
    parser.add_argument("--paired-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bind-release-paired-data", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _require(args.bind_release_paired_data, "refusing expensive binding without --bind-release-paired-data")
        output = args.output.expanduser().resolve()
        _require(not output.exists(), f"refusing to overwrite release paired binding: {output}")
        result = build_release_paired_binding(
            base_lineage_manifest=args.base_lineage_manifest,
            base_lineage_sha256=args.base_lineage_sha256,
            checkpoint=args.checkpoint,
            dataset_stats=args.dataset_stats,
            official_manifest=args.official_manifest,
            paired_root=args.paired_root,
        )
        atomic_write_json(output, result)
        verified = verify_release_paired_binding(output)
        print(json.dumps(verified, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"Release paired binding failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BINDING_KIND",
    "BINDING_SCHEMA_VERSION",
    "PAIR280_BINDING_SCHEMA_VERSION",
    "ReleasePairedBindingError",
    "build_release_paired_binding",
    "validate_release_paired_binding_payload",
    "verify_release_paired_binding",
]
