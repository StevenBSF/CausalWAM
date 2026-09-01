"""Fail-closed resume gate after a completed C1 engineering-smoke train.

This is intentionally a single recovery point.  It may continue with C1
deployment and then a fresh C3 run, but it never resumes a partial C3 run or
reuses rollout/pair-audit evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_audit import (
    load_config,
    validate_c1_c3_pair,
    validate_execution_ready,
)
from .rollout_policy import _read_checkpoint_provenance


class ReleaseEngineeringSmokeResumeError(ValueError):
    """Existing artifacts cannot prove the exact C1-train resume point."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseEngineeringSmokeResumeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleaseEngineeringSmokeResumeError(
            f"cannot parse {label} {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _config_paths(
    materialization: Mapping[str, Any], output_root: Path
) -> tuple[Path, Path]:
    rows = materialization.get("configs")
    _require(isinstance(rows, Mapping), "materialization configs are missing")
    paths: list[Path] = []
    for name in ("c1", "c3"):
        row = rows.get(name) if isinstance(rows, Mapping) else None
        _require(isinstance(row, Mapping), f"materialized {name} config is missing")
        path = Path(str(row.get("path", ""))).expanduser().resolve()
        expected = (output_root / "configs" / f"{name}_engineering_smoke.yaml").resolve()
        _require(path == expected, f"materialized {name} config path is not canonical")
        _require(path.is_file(), f"materialized {name} config is missing: {path}")
        _require(_sha256(path) == row.get("sha256"), f"materialized {name} config SHA differs")
        paths.append(path)
    return paths[0], paths[1]


def _audit_c1_training(
    *,
    c1_root: Path,
    c1_config: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = (c1_root / "checkpoint.pt").resolve()
    summary = _json(c1_root / "training_summary.json", "C1 training summary")
    sequence = _json(c1_root / "training_sequence_audit.json", "C1 sequence audit")
    updates = _json(c1_root / "parameter_update_audit.json", "C1 parameter update audit")
    gradients = _json(c1_root / "gradient_audit.json", "C1 gradient audit")
    contract = _json(c1_root / "matched_stream_contract.json", "C1 matched stream contract")
    run_config = _json(c1_root / "run_config.json", "C1 resolved run config")
    _require(checkpoint.is_file(), f"C1 checkpoint is missing: {checkpoint}")
    _require(
        not (c1_root / "rollout_load_execute.json").exists(),
        "C1 rollout evidence already exists; this resume point is train-only",
    )

    expected_steps = int(materialization.get("optimizer_steps_per_control", -1))
    _require(expected_steps == 3, "engineering resume requires exactly three C1 steps")
    _require(summary.get("status") == "SMOKE_COMPLETE", "C1 training summary is not complete")
    _require(summary.get("formal_training_auto_started") is False, "C1 started formal training")
    _require(summary.get("control") == "c1_architecture_only", "C1 summary control differs")
    _require(summary.get("regime") == materialization.get("regime"), "C1 regime differs")
    _require(int(summary.get("steps", -1)) == expected_steps, "C1 step count differs")
    _require(float(summary.get("lambda_contrastive", -1.0)) == 0.0, "C1 lambda is not zero")
    _require(summary.get("paired_contrastive_gradient_enabled") is False, "C1 contrastive gradients were enabled")
    _require(Path(str(summary.get("checkpoint", ""))).resolve() == checkpoint, "C1 summary checkpoint path differs")
    deliverables = summary.get("deliverable_status")
    _require(isinstance(deliverables, Mapping), "C1 deliverable status is missing")
    _require(deliverables.get("implementation") == "PASS", "C1 implementation is not PASS")
    _require(deliverables.get("short_update") == "PASS", "C1 short update is not PASS")
    _require(deliverables.get("gradient_audit") == "PASS", "C1 gradient audit is not PASS")
    _require(deliverables.get("rollout_load_execute") == "PENDING_SEPARATE_SMOKE", "C1 is not at the pre-deploy boundary")

    for key, expected in c1_config.items():
        _require(run_config.get(key) == expected, f"C1 resolved run config changed input field {key!r}")
    provenance = _read_checkpoint_provenance(checkpoint)
    metadata = provenance["metadata"]
    _require(metadata.get("run_config") == run_config, "C1 checkpoint/run_config provenance differs")
    _require(metadata.get("regime") == materialization.get("regime"), "C1 checkpoint regime differs")
    _require(int(metadata.get("step", -1)) == expected_steps, "C1 checkpoint step differs")

    _require(sequence.get("status") == "PASS", "C1 sequence audit is not PASS")
    official_per_step = int(c1_config["training"]["effective_official_global_batch"])
    paired_per_step = int(c1_config["training"]["effective_paired_groups_per_step"])
    _require(
        int(sequence.get("official_sample_count", -1)) == expected_steps * official_per_step,
        "C1 official sequence count differs",
    )
    _require(
        int(sequence.get("paired_physical_state_count", -1)) == expected_steps * paired_per_step,
        "C1 paired sequence count differs",
    )
    for key in (
        "official_sample_sequence_sha256",
        "paired_physical_state_sequence_sha256",
        "matched_stream_contract_sha256",
    ):
        _require(_is_sha256(sequence.get(key)), f"C1 sequence {key} is invalid")
    _require(contract.get("status") == "PASS", "C1 matched stream contract is not PASS")
    _require(_is_sha256(contract.get("sha256")), "C1 matched stream contract SHA is invalid")
    _require(
        summary.get("matched_stream_contract_sha256")
        == sequence.get("matched_stream_contract_sha256")
        == contract.get("sha256"),
        "C1 matched stream contract bindings differ",
    )
    _require(
        summary.get("official_sample_sequence_sha256")
        == sequence.get("official_sample_sequence_sha256"),
        "C1 official sequence SHA differs from summary",
    )
    _require(
        summary.get("paired_physical_state_sequence_sha256")
        == sequence.get("paired_physical_state_sequence_sha256"),
        "C1 paired sequence SHA differs from summary",
    )

    _require(gradients.get("status") == "PASS", "C1 gradient audit is not PASS")
    gradient_steps = gradients.get("steps")
    _require(isinstance(gradient_steps, list) and len(gradient_steps) == expected_steps, "C1 gradient step count differs")
    for index, row in enumerate(gradient_steps, start=1):
        _require(isinstance(row, Mapping) and row.get("step") == index, "C1 gradient steps are not ordered")
        if index >= 2:
            probe = row.get("action_only_probe")
            _require(isinstance(probe, Mapping) and probe.get("all_finite") is True, "C1 action-only probe is not finite")
            for key in ("head_grad_norm", "adapter_attention_grad_norm", "gate_grad_norm"):
                _require(float(probe.get(key, 0.0)) > 0.0, f"C1 {key} did not receive action gradient")

    head_adapter = updates.get("head_and_adapter")
    _require(isinstance(head_adapter, Mapping), "C1 Head/GCA update audit is missing")
    _require(head_adapter.get("all_finite") is True, "C1 Head/GCA update is non-finite")
    _require(int(head_adapter.get("changed_parameter_tensors", 0)) > 0, "C1 changed no parameters")
    deltas = head_adapter.get("max_abs_delta_by_module")
    _require(isinstance(deltas, Mapping), "C1 module update deltas are missing")
    _require(float(deltas.get("content_head", 0.0)) > 0.0, "C1 Content Head did not update")
    _require(float(deltas.get("adapter", 0.0)) > 0.0, "C1 GCA did not update")
    for key in ("final_content_head_sha256", "final_adapter_sha256"):
        _require(_is_sha256(updates.get(key)), f"C1 {key} is invalid")

    return {
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "step": expected_steps,
        },
        "matched_stream_contract_sha256": contract["sha256"],
        "official_sample_sequence_sha256": sequence["official_sample_sequence_sha256"],
        "paired_physical_state_sequence_sha256": sequence[
            "paired_physical_state_sequence_sha256"
        ],
    }


def audit_resume_after_c1_train(*, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    _require(root.is_dir(), f"smoke output root is missing: {root}")
    manifest_path = (root / "materialization_manifest.json").resolve()
    materialization = _json(manifest_path, "materialization manifest")
    _require(materialization.get("schema_version") == 1, "materialization schema differs")
    _require(
        materialization.get("kind")
        == "policy_release_c1_c3_engineering_smoke_materialization",
        "materialization kind differs",
    )
    _require(materialization.get("status") == "PASS", "materialization is not PASS")
    _require(materialization.get("scientific_result") is False, "materialization claims a scientific result")
    _require(materialization.get("p_mode_selection_evidence") is False, "materialization claims P-mode selection")
    _require(materialization.get("formal_training_auto_started") is False, "materialization started formal training")

    c1_path, c3_path = _config_paths(materialization, root)
    c1_config = load_config(c1_path)
    c3_config = load_config(c3_path)
    validate_execution_ready(c1_config)
    validate_execution_ready(c3_config)
    fairness = validate_c1_c3_pair(c1_config, c3_config)
    _require(fairness.get("fairness") == "PASS", "materialized C1/C3 fairness is not PASS")

    c1_root = (root / "runs/c1").resolve()
    c3_root = (root / "runs/c3").resolve()
    _require(Path(c1_config["output_dir"]).resolve() == c1_root, "C1 output path differs")
    _require(Path(c3_config["output_dir"]).resolve() == c3_root, "C3 output path differs")
    _require(not c3_root.exists(), "C3 run directory already exists; partial C3 reuse is forbidden")
    _require(not (root / "strict_pair_audit.json").exists(), "strict pair audit already exists")
    c1 = _audit_c1_training(
        c1_root=c1_root,
        c1_config=c1_config,
        materialization=materialization,
    )
    return {
        "schema_version": 1,
        "kind": "policy_release_engineering_smoke_c1_train_resume_audit",
        "status": "PASS",
        "resume_boundary": "after_c1_train_before_c1_deploy",
        "formal_training_auto_started": False,
        "scientific_result": False,
        "output_root": str(root),
        "materialization_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "fairness": fairness,
        "c1": c1,
        "c3_run_directory_absent": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_resume_after_c1_train(output_root=args.output_root)
    target = Path(args.output_json).expanduser().resolve()
    _require(not target.exists(), f"refusing to overwrite resume audit: {target}")
    _require(target.parent == Path(args.output_root).expanduser().resolve(), "resume audit must be written inside output root")
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReleaseEngineeringSmokeResumeError",
    "audit_resume_after_c1_train",
]
