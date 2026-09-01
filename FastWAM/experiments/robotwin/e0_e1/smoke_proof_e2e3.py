#!/usr/bin/env python3
"""Create and verify the immutable canonical-smoke proof for an E2/E3 full run.

The full runner first re-audits the canonical smoke in read-only mode.  This
module then snapshots strong identities for both the smoke's scientific
configuration and its terminal audit/status artifacts.  The full run config
binds to the resulting proof file, so bypassing smoke execution cannot produce
an auditable full result unless a valid canonical smoke already exists.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_lock_e2e3 import strong_file_identity
from .io_utils import write_json


SCHEMA_VERSION = 1
PROTOCOL = "r3_holdout_v1"
CANONICAL_TASKS = ("place_a2b_left",)
CANONICAL_LAYERS = (8, 16, 24)
ARTIFACTS = {
    "run_config": "run_config.json",
    "state": "status/state.txt",
    "success_marker": "status/SUCCESS",
    "final_audit_marker": "status/final_audit.done",
}
SHARED_FULL_FIELDS = (
    "protocol",
    "data_root",
    "model_base",
    "layers",
    "seed",
    "temperature",
    "min_temporal_gap",
    "min_state_distance",
    "checkpoint",
    "dataset_stats",
    "experiment_code_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(value, Mapping), f"{label} root must be an object: {path}")
    return value


def _identity_matches(declared: Any, actual: Mapping[str, Any], *, label: str) -> None:
    _require(isinstance(declared, Mapping), f"{label} identity is missing")
    for field in ("path", "size_bytes", "mtime_ns", "sha256"):
        _require(
            declared.get(field) == actual.get(field),
            f"{label} identity {field} mismatch",
        )


def _iter_strong_identities(value: Any, *, label: str):
    if isinstance(value, Mapping):
        fields = {"path", "size_bytes", "mtime_ns", "sha256"}
        if fields.issubset(value):
            yield label, value
            return
        for key, child in value.items():
            yield from _iter_strong_identities(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_strong_identities(child, label=f"{label}[{index}]")


def _validate_scientific_artifact_chain(value: Any) -> None:
    seen: dict[Path, Mapping[str, Any]] = {}
    found = False
    for label, declared in _iter_strong_identities(
        value, label="audited_scientific_artifact_identities"
    ):
        found = True
        path = Path(str(declared["path"])).expanduser().resolve()
        if path in seen:
            _identity_matches(declared, seen[path], label=label)
            continue
        try:
            actual = strong_file_identity(path)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"cannot verify {label} at {path}: {error}") from error
        _identity_matches(declared, actual, label=label)
        seen[path] = actual
    _require(found, "smoke audit contains no strong scientific artifact identities")


def _validate_canonical_smoke_config(
    config: Mapping[str, Any], *, canonical_dir: Path
) -> None:
    _require(config.get("schema_version") == 1, "smoke run config schema mismatch")
    _require(config.get("protocol") == PROTOCOL, "smoke run config protocol mismatch")
    _require(config.get("mode") == "smoke", "proof source is not a smoke run")
    _require(tuple(config.get("tasks", ())) == CANONICAL_TASKS,
             f"canonical smoke tasks must be {CANONICAL_TASKS}")
    _require(tuple(config.get("layers", ())) == CANONICAL_LAYERS,
             f"canonical smoke layers must be {CANONICAL_LAYERS}")
    exact = {
        "states_per_trajectory": 2,
        "train_steps": 1,
        "groups_per_batch": 2,
        "val_every": 1,
        "seed": 0,
        "temperature": 0.07,
        "min_temporal_gap": 8,
        "min_state_distance": 1e-5,
    }
    for field, expected in exact.items():
        _require(config.get(field) == expected,
                 f"canonical smoke {field} must be {expected!r}")
    _require(canonical_dir.name == "smoke", "canonical smoke directory name mismatch")


def _validate_terminal_payloads(
    *, canonical_dir: Path, audit: Mapping[str, Any], deliverables: Mapping[str, Any]
) -> None:
    canonical_text = str(canonical_dir)
    _require(audit.get("schema_version") == 1, "smoke audit schema mismatch")
    _require(audit.get("protocol") == PROTOCOL, "smoke audit protocol mismatch")
    _require(audit.get("audit_status") == "PASS", "smoke audit is not PASS")
    _require(audit.get("run_mode") == "smoke", "smoke audit mode mismatch")
    _require(tuple(audit.get("tasks", ())) == CANONICAL_TASKS,
             "smoke audit task set mismatch")
    _require(str(Path(str(audit.get("run_dir", ""))).expanduser().resolve()) == canonical_text,
             "smoke audit run directory mismatch")
    assertions = audit.get("assertions")
    _require(isinstance(assertions, Mapping) and bool(assertions),
             "smoke audit assertions are missing")
    _require(all(value is True for value in assertions.values()),
             "one or more smoke audit assertions did not pass")

    _require(deliverables.get("schema_version") == 1,
             "smoke deliverables schema mismatch")
    _require(deliverables.get("protocol") == PROTOCOL,
             "smoke deliverables protocol mismatch")
    _require(deliverables.get("status") == "COMPLETE_AND_AUDITED",
             "smoke deliverables are not complete and audited")
    _require(
        str(Path(str(deliverables.get("run_dir", ""))).expanduser().resolve())
        == canonical_text,
        "smoke deliverables run directory mismatch",
    )


def build_smoke_proof(smoke_run_dir: str | Path) -> dict[str, Any]:
    """Build a deterministic proof from an already re-audited smoke run."""

    canonical_dir = Path(smoke_run_dir).expanduser().resolve()
    _require(canonical_dir.is_dir(), f"canonical smoke directory is missing: {canonical_dir}")
    paths = {name: canonical_dir / relative for name, relative in ARTIFACTS.items()}
    report_paths = {
        "protocol_audit": canonical_dir / "protocol_audit.json",
        "deliverables": canonical_dir / "deliverables.json",
    }
    config = _read_object(paths["run_config"], label="smoke run config")
    audit = _read_object(report_paths["protocol_audit"], label="smoke protocol audit")
    deliverables = _read_object(report_paths["deliverables"], label="smoke deliverables")
    _validate_canonical_smoke_config(config, canonical_dir=canonical_dir)
    _validate_terminal_payloads(
        canonical_dir=canonical_dir, audit=audit, deliverables=deliverables
    )
    try:
        state = paths["state"].read_text(encoding="utf-8").strip()
        success = paths["success_marker"].read_text(encoding="utf-8").strip()
        final_audit = paths["final_audit_marker"].read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"canonical smoke terminal marker is missing: {error}") from error
    _require(state == "SUCCESS", "canonical smoke state is not SUCCESS")
    _require(bool(success), "canonical smoke SUCCESS marker is empty")
    _require(bool(final_audit), "canonical smoke final-audit marker is empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "proof_kind": "canonical_smoke_strict_audit_v1",
        "canonical_smoke_run_dir": str(canonical_dir),
        "strict_audit_require_success_marker": True,
        "audited_scientific_artifact_identities": audit.get("artifact_identities"),
        "artifacts": {
            name: strong_file_identity(path) for name, path in paths.items()
        },
    }


def validate_smoke_proof(
    proof_path: str | Path,
    *,
    canonical_smoke_dir: str | Path,
    full_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate a proof and every smoke artifact to which it is bound."""

    source = Path(proof_path).expanduser().resolve()
    proof = _read_object(source, label="canonical smoke proof")
    canonical_dir = Path(canonical_smoke_dir).expanduser().resolve()
    _require(proof.get("schema_version") == SCHEMA_VERSION, "smoke proof schema mismatch")
    _require(proof.get("protocol") == PROTOCOL, "smoke proof protocol mismatch")
    _require(proof.get("proof_kind") == "canonical_smoke_strict_audit_v1",
             "smoke proof kind mismatch")
    _require(proof.get("strict_audit_require_success_marker") is True,
             "smoke proof does not record strict terminal audit")
    _require(
        str(Path(str(proof.get("canonical_smoke_run_dir", ""))).expanduser().resolve())
        == str(canonical_dir),
        "smoke proof does not reference the canonical smoke directory",
    )
    declared = proof.get("artifacts")
    _require(isinstance(declared, Mapping) and set(declared) == set(ARTIFACTS),
             "smoke proof artifact identity set is incomplete")
    paths = {name: canonical_dir / relative for name, relative in ARTIFACTS.items()}
    report_paths = {
        "protocol_audit": canonical_dir / "protocol_audit.json",
        "deliverables": canonical_dir / "deliverables.json",
    }
    for name, path in paths.items():
        try:
            actual = strong_file_identity(path)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"cannot verify smoke proof artifact {path}: {error}") from error
        _identity_matches(declared[name], actual, label=f"smoke proof {name}")

    config = _read_object(paths["run_config"], label="smoke run config")
    audit = _read_object(report_paths["protocol_audit"], label="smoke protocol audit")
    deliverables = _read_object(report_paths["deliverables"], label="smoke deliverables")
    _validate_canonical_smoke_config(config, canonical_dir=canonical_dir)
    _validate_terminal_payloads(
        canonical_dir=canonical_dir, audit=audit, deliverables=deliverables
    )
    _require(
        proof.get("audited_scientific_artifact_identities")
        == audit.get("artifact_identities"),
        "smoke proof scientific artifact identity chain mismatch",
    )
    _validate_scientific_artifact_chain(
        proof.get("audited_scientific_artifact_identities")
    )
    _require(paths["state"].read_text(encoding="utf-8").strip() == "SUCCESS",
             "canonical smoke state is not SUCCESS")
    _require(bool(paths["success_marker"].read_text(encoding="utf-8").strip()),
             "canonical smoke SUCCESS marker is empty")
    _require(bool(paths["final_audit_marker"].read_text(encoding="utf-8").strip()),
             "canonical smoke final-audit marker is empty")
    if full_config is not None:
        _require(full_config.get("mode") == "full", "smoke proof consumer is not a full run")
        for field in SHARED_FULL_FIELDS:
            _require(config.get(field) == full_config.get(field),
                     f"smoke/full shared provenance differs for {field}")
    return {
        "proof_identity": strong_file_identity(source),
        "canonical_smoke_run_dir": str(canonical_dir),
        "smoke_run_config_identity": dict(declared["run_config"]),
        "audited_scientific_artifact_identities": proof.get(
            "audited_scientific_artifact_identities"
        ),
        "smoke_success_marker_identity": dict(declared["success_marker"]),
    }


def write_smoke_proof(smoke_run_dir: str | Path, output: str | Path) -> Path:
    """Write once, or confirm that an existing immutable proof is identical."""

    value = build_smoke_proof(smoke_run_dir)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        existing = _read_object(destination, label="existing canonical smoke proof")
        _require(existing == value, "existing canonical smoke proof differs; refusing overwrite")
        return destination
    return write_json(destination, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = write_smoke_proof(args.smoke_run_dir, args.output)
    print(f"canonical smoke proof: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACTS",
    "CANONICAL_LAYERS",
    "CANONICAL_TASKS",
    "PROTOCOL",
    "build_smoke_proof",
    "validate_smoke_proof",
    "write_smoke_proof",
]
