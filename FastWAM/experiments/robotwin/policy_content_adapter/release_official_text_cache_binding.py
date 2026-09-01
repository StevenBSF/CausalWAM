"""Bind the audited full-550 official prompt cache to the author release.

The 68,704 payloads (~72 GiB) were already tensor-reloaded and SHA-aggregated
by the strict Stage-1 launch gate.  This module turns that immutable evidence
into a release-base binding once.  Individual Stage-2 runs verify the small
binding/audit/inventory/evidence files and let RobotVideoDataset validate each
prompt payload when it is actually consumed; they never re-hash 72 GiB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .release_lineage import verify_author_release_lineage
from .stage1_text_cache import EXPECTED_UNIQUE_TASK_INDICES, verify_text_cache_audit


KIND = "policy_release_official_text_cache_binding"
SCHEMA_VERSION = 1


class ReleaseOfficialTextCacheBindingError(ValueError):
    """The official text cache lacks immutable release-base evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseOfficialTextCacheBindingError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"artifact does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256(resolved),
    }


def _json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleaseOfficialTextCacheBindingError(
            f"cannot parse {label} {resolved}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def validate_binding_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(value.get("kind") == KIND, "official text-cache binding kind changed")
    _require(value.get("schema_version") == SCHEMA_VERSION, "official text-cache binding schema changed")
    _require(value.get("status") == "PASS", "official text-cache binding is not PASS")
    lineage = value.get("base_lineage")
    cache = value.get("cache")
    completion = value.get("completion_audit")
    inventory = value.get("inventory")
    strict = value.get("strict_payload_revalidation")
    official = value.get("official_dataset")
    for section, label in (
        (lineage, "base_lineage"),
        (cache, "cache"),
        (completion, "completion_audit"),
        (inventory, "inventory"),
        (strict, "strict_payload_revalidation"),
        (official, "official_dataset"),
    ):
        _require(isinstance(section, Mapping), f"binding {label} is missing")
    assert isinstance(lineage, Mapping)
    assert isinstance(cache, Mapping)
    assert isinstance(completion, Mapping)
    assert isinstance(inventory, Mapping)
    assert isinstance(strict, Mapping)
    assert isinstance(official, Mapping)
    for identity, label in (
        (completion, "completion audit"),
        (inventory, "inventory"),
        (strict.get("evidence"), "strict payload evidence"),
    ):
        _require(isinstance(identity, Mapping), f"{label} identity is missing")
        assert isinstance(identity, Mapping)
        _require(len(str(identity.get("sha256", ""))) == 64, f"{label} SHA is invalid")
    _require(len(str(lineage.get("sha256", ""))) == 64, "base lineage SHA is invalid")
    _require(Path(str(cache.get("directory", ""))).is_absolute(), "cache directory must be absolute")
    _require(cache.get("file_count") == EXPECTED_UNIQUE_TASK_INDICES, "official prompt count changed")
    _require(int(cache.get("total_size_bytes", 0)) > 0, "official cache is empty")
    aggregate = str(cache.get("aggregate_payload_sha256", ""))
    _require(len(aggregate) == 64, "official payload aggregate SHA is invalid")
    _require(strict.get("status") == "PASS", "strict payload revalidation is not PASS")
    _require(strict.get("all_payload_shapes_valid") is True, "strict payload shape audit is absent")
    _require(strict.get("all_required_cache_files_present") is True, "strict cache completeness is absent")
    _require(strict.get("aggregate_payload_sha256") == aggregate, "strict payload aggregate differs")
    _require(official.get("manifest_sha256") == lineage.get("official_manifest_sha256"), "official manifest differs from lineage")
    counts = official.get("selected_episode_counts_by_domain")
    expected = {
        task: {"clean": 50, "official_random": 500}
        for task in ("place_a2b_left", "open_microwave", "move_stapler_pad")
    }
    _require(counts == expected, "official text cache was not audited on full 550/task")
    return dict(value)


def build_binding(
    *,
    cache_dir: str | Path,
    completion_audit: str | Path,
    strict_payload_evidence: str | Path,
    base_lineage_manifest: str | Path,
    base_checkpoint: str | Path,
    dataset_stats: str | Path,
    official_manifest: str | Path,
    expected_base_lineage_sha256: str,
) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    _require(root.is_dir(), f"official text cache does not exist: {root}")
    lineage = verify_author_release_lineage(
        base_lineage_manifest,
        checkpoint_path=base_checkpoint,
        dataset_stats_path=dataset_stats,
        official_manifest_path=official_manifest,
        expected_manifest_sha256=expected_base_lineage_sha256,
    )
    completion, _ = verify_text_cache_audit(completion_audit, cache_dir=root)
    completion_identity = _identity(completion_audit)
    inventory_identity = _identity(completion["inventory"]["path"])
    _require(
        inventory_identity["sha256"] == completion["inventory"]["sha256"],
        "completion audit binds a different inventory",
    )
    evidence = _json(strict_payload_evidence, "strict payload revalidation evidence")
    evidence_identity = _identity(strict_payload_evidence)
    _require(evidence.get("status") == "PASS", "strict payload evidence is not PASS")
    runtime = evidence.get("runtime_dataset")
    text = evidence.get("text_cache_audit")
    artifacts = evidence.get("artifacts")
    _require(isinstance(runtime, Mapping), "strict evidence runtime dataset is missing")
    _require(isinstance(text, Mapping), "strict evidence text-cache audit is missing")
    _require(isinstance(artifacts, Mapping), "strict evidence artifacts are missing")
    assert isinstance(runtime, Mapping)
    assert isinstance(text, Mapping)
    assert isinstance(artifacts, Mapping)
    _require(Path(str(artifacts.get("text_embedding_cache_dir", ""))).resolve() == root, "strict evidence names a different cache")
    _require(text.get("status") == "PASS", "strict evidence payload audit is not PASS")
    _require(text.get("all_payload_shapes_valid") is True, "payload shapes were not strictly checked")
    _require(text.get("all_required_cache_files_present") is True, "payload completeness was not strictly checked")
    _require(text.get("unique_prompt_count") == EXPECTED_UNIQUE_TASK_INDICES, "strict prompt count changed")
    _require(text.get("aggregate_payload_sha256") == completion["cache"]["aggregate_payload_sha256"], "strict aggregate differs from completion audit")
    _require(text.get("completion_audit", {}).get("sha256") == completion_identity["sha256"], "strict evidence binds another completion audit")
    _require(text.get("inventory", {}).get("sha256") == inventory_identity["sha256"], "strict evidence binds another inventory")
    explicit = runtime.get("explicit_episode_native_loader")
    _require(isinstance(explicit, Mapping), "strict evidence explicit loader is missing")
    assert isinstance(explicit, Mapping)
    _require(explicit.get("manifest_sha256") == lineage["official_partition"]["manifest"]["sha256"], "strict evidence official manifest differs")
    _require(explicit.get("loaded_episode_count") == 1650, "strict evidence did not load 1,650 episodes")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "PASS",
        "base_lineage": {
            "path": str(Path(base_lineage_manifest).expanduser().resolve()),
            "sha256": lineage["manifest_identity"]["sha256"],
            "checkpoint_sha256": lineage["checkpoint"]["sha256"],
            "dataset_stats_sha256": lineage["dataset_stats"]["sha256"],
            "official_manifest_sha256": lineage["official_partition"]["manifest"]["sha256"],
        },
        "cache": {
            "directory": str(root),
            "file_count": int(completion["cache"]["file_count"]),
            "total_size_bytes": int(completion["cache"]["total_size_bytes"]),
            "aggregate_payload_sha256": completion["cache"]["aggregate_payload_sha256"],
            "directory_bytes_rehashed_per_stage2_run": False,
            "runtime_validation": "RobotVideoDataset validates each consumed prompt payload",
        },
        "completion_audit": completion_identity,
        "inventory": {
            **inventory_identity,
            "prompt_set_sha256": completion["inventory"]["prompt_set_sha256"],
        },
        "strict_payload_revalidation": {
            "status": "PASS",
            "evidence": evidence_identity,
            "all_payload_shapes_valid": True,
            "all_required_cache_files_present": True,
            "aggregate_payload_sha256": text["aggregate_payload_sha256"],
            "unique_prompt_count": int(text["unique_prompt_count"]),
        },
        "official_dataset": {
            "manifest_sha256": explicit["manifest_sha256"],
            "selection_mode": explicit["selection_mode"],
            "loaded_episode_count": explicit["loaded_episode_count"],
            "selected_episode_counts_by_domain": runtime[
                "selected_episode_counts_by_domain"
            ],
            "metadata": runtime["meta_files"],
        },
    }
    return validate_binding_payload(payload)


def write_binding(output: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ReleaseOfficialTextCacheBindingError(
                f"refusing to overwrite official text-cache binding: {target}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def verify_binding(
    manifest: str | Path,
    *,
    expected_sha256: str,
    expected_base_lineage_sha256: str,
    expected_cache_dir: str | Path,
) -> dict[str, Any]:
    target = Path(manifest).expanduser().resolve()
    _require(_sha256(target) == expected_sha256, "official text-cache binding SHA differs")
    value = validate_binding_payload(_json(target, "official text-cache binding"))
    _require(value["base_lineage"]["sha256"] == expected_base_lineage_sha256, "official text cache descends from another base")
    _require(Path(value["cache"]["directory"]).resolve() == Path(expected_cache_dir).expanduser().resolve(), "official text cache directory differs from binding")
    _require(Path(expected_cache_dir).expanduser().resolve().is_dir(), "official text cache directory is missing")
    for section in ("completion_audit", "inventory"):
        identity = value[section]
        _require(_sha256(Path(identity["path"])) == identity["sha256"], f"official text-cache {section} SHA differs")
    evidence = value["strict_payload_revalidation"]["evidence"]
    _require(_sha256(Path(evidence["path"])) == evidence["sha256"], "strict payload evidence SHA differs")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--completion-audit", required=True)
    parser.add_argument("--strict-payload-evidence", required=True)
    parser.add_argument("--base-lineage-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--official-manifest", required=True)
    parser.add_argument("--expected-base-lineage-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_binding(
        cache_dir=args.cache_dir,
        completion_audit=args.completion_audit,
        strict_payload_evidence=args.strict_payload_evidence,
        base_lineage_manifest=args.base_lineage_manifest,
        base_checkpoint=args.base_checkpoint,
        dataset_stats=args.dataset_stats,
        official_manifest=args.official_manifest,
        expected_base_lineage_sha256=args.expected_base_lineage_sha256,
    )
    path = write_binding(args.output, payload)
    print(json.dumps({"status": "PASS", "path": str(path), "sha256": _sha256(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "KIND",
    "ReleaseOfficialTextCacheBindingError",
    "build_binding",
    "validate_binding_payload",
    "verify_binding",
    "write_binding",
]
