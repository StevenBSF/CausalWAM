"""Strict prelaunch and post-training audits for the release P-mode dev pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_audit import load_config, validate_execution_ready
from .materialize_release_pmode_dev import (
    DEFAULT_EVALUATOR_SOURCE,
    validate_pmode_dev_pair,
)
from .p_mode_selection import validate_seed_bank_descriptor


class PModeDevAuditError(ValueError):
    """Materialized or trained P-mode candidates violate the locked pair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PModeDevAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _required_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a non-null SHA-256",
    )
    return value


def _json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PModeDevAuditError(f"cannot parse {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def audit_materialization(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _json(manifest_path, "materialization manifest")
    _require(
        manifest.get("kind") == "policy_release_pmode_dev_materialization"
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "PASS",
        "materialization manifest kind/version/status differs",
    )
    configs = manifest.get("configs")
    _require(
        isinstance(configs, Mapping) and set(configs) == {"p_v1", "p_v2"},
        "materialization must contain exactly P-v1/P-v2 configs",
    )
    loaded: dict[str, dict[str, Any]] = {}
    config_identities: dict[str, dict[str, Any]] = {}
    for regime in ("p_v1", "p_v2"):
        identity = configs[regime]
        _require(isinstance(identity, Mapping), f"{regime} config identity missing")
        config_path = Path(str(identity.get("path", ""))).expanduser().resolve()
        _require(config_path.is_file(), f"{regime} config missing")
        digest = _sha256(config_path)
        _require(digest == identity.get("sha256"), f"{regime} config SHA changed")
        value = load_config(config_path)
        validate_execution_ready(value)
        loaded[regime] = value
        config_identities[regime] = {
            "path": str(config_path),
            "size_bytes": config_path.stat().st_size,
            "sha256": digest,
        }
    fairness = validate_pmode_dev_pair(loaded["p_v1"], loaded["p_v2"])
    _require(fairness == manifest.get("fairness"), "materialized fairness proof changed")

    bank_path = Path(
        loaded["p_v1"]["evaluation"]["simulator_seed_bank_manifest"]
    ).expanduser().resolve()
    bank_sha = _sha256(bank_path)
    _require(
        bank_sha
        == manifest.get("artifacts", {}).get(
            "simulator_seed_bank_manifest_sha256"
        ),
        "dev seed-bank file SHA changed",
    )
    raw_bank = _json(bank_path, "dev seed bank")
    bank = validate_seed_bank_descriptor(
        raw_bank, expected_purpose="dev_selection"
    )
    _require(
        bank["simulator_seed_bank_id"]
        == loaded["p_v1"]["evaluation"]["simulator_seed_bank_id"]
        == loaded["p_v2"]["evaluation"]["simulator_seed_bank_id"],
        "dev seed-bank identity differs across candidates",
    )
    evaluator = DEFAULT_EVALUATOR_SOURCE
    _require(evaluator.is_file(), "seed-bank evaluator source is unavailable")
    _require(
        _sha256(evaluator) == bank["evaluator_source_sha256"],
        "RoboTwin evaluator source changed after dev bank lock",
    )
    return {
        "status": "PASS",
        "stage": "materialization",
        "manifest": str(manifest_path),
        "configs": config_identities,
        "fairness": fairness,
        "dev_seed_bank_id": bank["simulator_seed_bank_id"],
        "dev_seed_bank_manifest_sha256": bank_sha,
    }


def audit_posttrain(path: str | Path) -> dict[str, Any]:
    prelaunch = audit_materialization(path)
    manifest_path = Path(path).expanduser().resolve()
    manifest = _json(manifest_path, "materialization manifest")
    run_roots = {
        regime: Path(manifest["configs"][regime]["path"]).resolve().parents[1]
        / "runs"
        / regime
        for regime in ("p_v1", "p_v2")
    }
    summaries: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    sequences: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    checkpoint_identities: dict[str, dict[str, Any]] = {}
    for regime, root in run_roots.items():
        summary = _json(root / "training_summary.json", f"{regime} training summary")
        run_config = _json(root / "run_config.json", f"{regime} run config")
        sequence = _json(
            root / "training_sequence_audit.json", f"{regime} training sequence audit"
        )
        contract = _json(
            root / "matched_stream_contract.json", f"{regime} matched stream contract"
        )
        checkpoint = root / "checkpoint.pt"
        _require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"{regime} checkpoint missing")
        _require(summary.get("regime") == regime, f"{regime} summary regime differs")
        _require(summary.get("control") == regime, f"{regime} summary control differs")
        _require(float(summary.get("lambda_contrastive", -1.0)) == 0.0, f"{regime} summary is not lambda-zero")
        _require(summary.get("steps") == run_config["training"]["max_steps"], f"{regime} step count differs")
        _require(summary.get("checkpoint") == str(checkpoint), f"{regime} checkpoint path differs")
        _require(sequence.get("status") == "PASS", f"{regime} training sequence audit is not PASS")
        _require(contract.get("status") == "PASS", f"{regime} matched stream contract is not PASS")
        contract_body = contract.get("contract")
        _require(
            isinstance(contract_body, Mapping),
            f"{regime} matched stream contract body is missing",
        )
        contract_sha = _required_sha256(
            contract.get("sha256"), f"{regime} matched stream contract SHA"
        )
        _require(
            _canonical_sha256(contract_body) == contract_sha,
            f"{regime} matched stream contract body/SHA is inconsistent",
        )
        resolved_contract = run_config.get("resolved_matched_stream_contract")
        _require(
            isinstance(resolved_contract, Mapping),
            f"{regime} resolved matched stream contract is missing",
        )
        _require(
            dict(resolved_contract) == contract,
            f"{regime} resolved matched stream contract/file is inconsistent",
        )
        resolved_sequence = run_config.get("resolved_training_sequence_audit")
        _require(
            isinstance(resolved_sequence, Mapping),
            f"{regime} resolved training sequence audit is missing",
        )
        _require(
            dict(resolved_sequence) == sequence,
            f"{regime} resolved training sequence audit/file is inconsistent",
        )
        _require(
            summary.get("training_sequence_audit") == sequence,
            f"{regime} summary training sequence audit/file is inconsistent",
        )
        for field in (
            "official_sample_sequence_sha256",
            "paired_physical_state_sequence_sha256",
            "matched_stream_contract_sha256",
        ):
            summary_sha = _required_sha256(
                summary.get(field), f"{regime} summary {field}"
            )
            sequence_sha = _required_sha256(
                sequence.get(field), f"{regime} sequence {field}"
            )
            resolved_sha = _required_sha256(
                resolved_sequence.get(field), f"{regime} resolved sequence {field}"
            )
            _require(
                summary_sha == sequence_sha == resolved_sha,
                f"{regime} {field} bindings are inconsistent",
            )
        _require(
            summary["matched_stream_contract_sha256"] == contract_sha,
            f"{regime} matched stream contract SHA bindings are inconsistent",
        )
        summaries[regime] = summary
        configs[regime] = run_config
        sequences[regime] = sequence
        contracts[regime] = contract
        checkpoint_identities[regime] = {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "mtime_ns": checkpoint.stat().st_mtime_ns,
        }
    validate_pmode_dev_pair(configs["p_v1"], configs["p_v2"])
    for field in (
        "official_sample_sequence_sha256",
        "paired_physical_state_sequence_sha256",
        "matched_stream_contract_sha256",
    ):
        _require(
            summaries["p_v1"].get(field) == summaries["p_v2"].get(field),
            f"P-v1/P-v2 {field} differs",
        )
    _require(
        contracts["p_v1"] == contracts["p_v2"],
        "P-v1/P-v2 matched stream contract files differ",
    )
    _require(
        sequences["p_v1"] == sequences["p_v2"],
        "P-v1/P-v2 training sequence audit files differ",
    )
    for field in (
        "source_fp32_content_head_sha256",
        "source_fp32_adapter_sha256",
        "training_fp32_content_head_sha256",
        "training_fp32_adapter_sha256",
    ):
        _require(
            summaries["p_v1"].get("initialization", {}).get(field)
            == summaries["p_v2"].get("initialization", {}).get(field),
            f"P-v1/P-v2 initialization {field} differs",
        )
    return {
        "status": "PASS",
        "stage": "posttrain",
        "prelaunch": prelaunch,
        "checkpoints": checkpoint_identities,
        "shared_sequences": {
            field: summaries["p_v1"][field]
            for field in (
                "official_sample_sequence_sha256",
                "paired_physical_state_sequence_sha256",
                "matched_stream_contract_sha256",
            )
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-manifest", required=True)
    parser.add_argument(
        "--stage", choices=("materialization", "posttrain"), default="materialization"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        audit_materialization(args.materialization_manifest)
        if args.stage == "materialization"
        else audit_posttrain(args.materialization_manifest)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PModeDevAuditError",
    "audit_materialization",
    "audit_posttrain",
]
