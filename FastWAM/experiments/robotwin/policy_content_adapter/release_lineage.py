"""Fail-closed provenance checks for the fixed FastWAM author release base."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import POLICY_MAIN_BASE_KIND, POLICY_MAIN_BASE_LINEAGE_KIND


AUTHOR_RELEASE_LINEAGE_SCHEMA_VERSION = 1
AUTHOR_RELEASE_LINEAGE_KIND = POLICY_MAIN_BASE_LINEAGE_KIND
AUTHOR_RELEASE_BASE_KIND = POLICY_MAIN_BASE_KIND
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReleaseLineageError(ValueError):
    """The supplied artifact set cannot prove the author-release lineage."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseLineageError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    _require(path.is_file(), f"{label} does not exist: {path}")
    return path


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "kind": "file",
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _validate_declared_identity(
    declaration: Any,
    actual_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    value = _mapping(declaration, label)
    for key in ("path", "size_bytes", "sha256"):
        _require(key in value, f"{label}.{key} is required")
    declared_path = Path(str(value["path"])).expanduser().resolve()
    _require(declared_path == actual_path, f"{label} path differs from the lineage")
    _require(
        isinstance(value["size_bytes"], int)
        and not isinstance(value["size_bytes"], bool)
        and value["size_bytes"] >= 0,
        f"{label}.size_bytes must be a non-negative integer",
    )
    _require(
        isinstance(value["sha256"], str)
        and SHA256_PATTERN.fullmatch(value["sha256"]) is not None,
        f"{label}.sha256 must be a lowercase SHA-256",
    )
    actual = _file_identity(actual_path)
    _require(actual["size_bytes"] == value["size_bytes"], f"{label} size differs from the lineage")
    _require(actual["sha256"] == value["sha256"], f"{label} SHA-256 differs from the lineage")
    return actual


def validate_author_release_lineage_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable semantic fields without touching the referenced files."""

    value = _mapping(payload, "author release lineage")
    _require(
        value.get("schema_version") == AUTHOR_RELEASE_LINEAGE_SCHEMA_VERSION,
        f"author release lineage schema_version must be {AUTHOR_RELEASE_LINEAGE_SCHEMA_VERSION}",
    )
    _require(value.get("kind") == AUTHOR_RELEASE_LINEAGE_KIND, "author release lineage kind changed")
    _require(value.get("status") == "PASS", "author release lineage status must be PASS")
    _require(value.get("base_kind") == AUTHOR_RELEASE_BASE_KIND, "base_kind must be author_release")
    _require(bool(value.get("lineage_id")), "author release lineage_id is required")
    _require(
        "training_seed" not in value,
        "a fixed author release lineage must not declare a training_seed",
    )

    for label in ("checkpoint", "dataset_stats"):
        identity = _mapping(value.get(label), label)
        for key in ("path", "size_bytes", "sha256"):
            _require(key in identity, f"{label}.{key} is required")
        _require(bool(identity["path"]), f"{label}.path is required")
        _require(
            isinstance(identity["size_bytes"], int)
            and not isinstance(identity["size_bytes"], bool)
            and identity["size_bytes"] > 0,
            f"{label}.size_bytes must be positive",
        )
        _require(
            isinstance(identity["sha256"], str)
            and SHA256_PATTERN.fullmatch(identity["sha256"]) is not None,
            f"{label}.sha256 must be a lowercase SHA-256",
        )

    partition = _mapping(value.get("official_partition"), "official_partition")
    expected_partition = {
        "task_count": 50,
        "episodes_per_task": 550,
        "clean_per_task": 50,
        "random_per_task": 500,
        "total_episodes": 27_500,
        "partition_rule": "first_50_clean_next_500_official_random",
        "domain_rule_scope": "hash_bound_protocol_partition_not_checkpoint_payload",
    }
    for key, expected in expected_partition.items():
        _require(partition.get(key) == expected, f"official_partition.{key} must be {expected!r}")
    _require(
        partition["task_count"] * partition["episodes_per_task"]
        == partition["total_episodes"],
        "official release episode arithmetic is inconsistent",
    )
    _mapping(partition.get("manifest"), "official_partition.manifest")

    source = _mapping(value.get("source"), "source")
    for key in (
        "repository",
        "revision",
        "release_model_id",
        "checkpoint_url",
        "dataset_url",
        "evidence_files",
    ):
        _require(bool(source.get(key)), f"source.{key} is required")
    _require(
        isinstance(source["revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is not None,
        "source.revision must be a full Git commit",
    )
    evidence = _mapping(source["evidence_files"], "source.evidence_files")
    _require(
        set(evidence)
        == {"readme", "task_config", "data_config", "evaluation_config"},
        "source.evidence_files must bind README and the three author configs",
    )
    for label, identity in evidence.items():
        _mapping(identity, f"source.evidence_files.{label}")

    model = _mapping(value.get("model_contract"), "model_contract")
    expected_model = {
        "task_config": "robotwin_uncond_3cam_384_1e-4",
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "raw_camera_shape_chw": [3, 480, 640],
        "transformed_camera_shape_chw": [3, 240, 320],
        "final_video_size_hw": [384, 320],
        "action_steps": 32,
        "action_dim": 14,
        "state_dim": 14,
        "normalization": "z-score",
        "stepwise_action_normalization": False,
    }
    for key, expected in expected_model.items():
        _require(model.get(key) == expected, f"model_contract.{key} must be {expected!r}")
    load = _mapping(model.get("checkpoint_load_contract"), "model_contract.checkpoint_load_contract")
    _require(load.get("loader") == "strict_load_release_checkpoint", "release strict loader changed")
    _require(load.get("mot_strict") is True, "release mot load must be strict")
    _require(load.get("proprio_encoder_strict") is True, "release proprio load must be strict")
    _require(load.get("expected_missing_keys") == 0, "release load must allow zero missing keys")
    _require(load.get("expected_unexpected_keys") == 0, "release load must allow zero unexpected keys")
    return dict(value)


def verify_author_release_lineage(
    manifest_path: str | Path,
    *,
    checkpoint_path: str | Path,
    dataset_stats_path: str | Path,
    official_manifest_path: str | Path,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and verify the manifest and every local artifact it binds.

    This deliberately does not instantiate the 6B model.  The manifest locks
    the exact strict-load contract; train/cache/C0 gates must additionally run
    ``strict_load_release_checkpoint`` and persist its zero-incompatibility
    audit before GPU work is accepted.
    """

    manifest = _resolved_file(manifest_path, "author release lineage manifest")
    manifest_identity = _file_identity(manifest)
    if expected_manifest_sha256 is not None:
        _require(
            isinstance(expected_manifest_sha256, str)
            and SHA256_PATTERN.fullmatch(expected_manifest_sha256) is not None,
            "expected author release lineage SHA-256 is invalid",
        )
        _require(
            manifest_identity["sha256"] == expected_manifest_sha256,
            "author release lineage manifest SHA-256 mismatch",
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleaseLineageError(f"cannot read author release lineage {manifest}: {exc}") from exc
    normalized = validate_author_release_lineage_payload(payload)

    checkpoint = _resolved_file(checkpoint_path, "author release checkpoint")
    stats = _resolved_file(dataset_stats_path, "author release dataset stats")
    official = _resolved_file(official_manifest_path, "official three-task manifest")
    normalized["checkpoint"] = _validate_declared_identity(
        normalized["checkpoint"], checkpoint, label="checkpoint"
    )
    normalized["dataset_stats"] = _validate_declared_identity(
        normalized["dataset_stats"], stats, label="dataset_stats"
    )
    partition = dict(normalized["official_partition"])
    partition["manifest"] = _validate_declared_identity(
        partition["manifest"], official, label="official_partition.manifest"
    )
    normalized["official_partition"] = partition

    source = dict(normalized["source"])
    verified_evidence: dict[str, Any] = {}
    for label, declaration in source["evidence_files"].items():
        evidence_path = _resolved_file(declaration["path"], f"source evidence {label}")
        verified_evidence[label] = _validate_declared_identity(
            declaration,
            evidence_path,
            label=f"source.evidence_files.{label}",
        )
    source["evidence_files"] = verified_evidence
    normalized["source"] = source
    normalized["manifest_identity"] = manifest_identity
    return normalized


__all__ = [
    "AUTHOR_RELEASE_BASE_KIND",
    "AUTHOR_RELEASE_LINEAGE_KIND",
    "AUTHOR_RELEASE_LINEAGE_SCHEMA_VERSION",
    "ReleaseLineageError",
    "validate_author_release_lineage_payload",
    "verify_author_release_lineage",
]
