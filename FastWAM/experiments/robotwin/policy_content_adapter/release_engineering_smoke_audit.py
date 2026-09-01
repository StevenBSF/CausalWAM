"""Strict post-run audit for a matched release-base C1/C3 engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_audit import load_config, validate_c1_c3_pair, validate_execution_ready


class ReleaseEngineeringSmokeAuditError(ValueError):
    """The two short runs do not prove the matched engineering gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseEngineeringSmokeAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleaseEngineeringSmokeAuditError(
            f"cannot parse {label} {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _audit_rollout(path: Path) -> dict[str, Any]:
    value = _json(path, "rollout load/execute evidence")
    _require(value.get("status") == "PASS", "rollout load/execute is not PASS")
    _require(value.get("sapien_imported") is False, "engineering rollout imported SAPIEN")
    tasks = value.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 3, "rollout must cover three tasks")
    expected = ["place_a2b_left", "open_microwave", "move_stapler_pad"]
    _require([row.get("task") for row in tasks] == expected, "rollout task order changed")
    for row in tasks:
        _require(row.get("executed_actions") == 1, "rollout did not execute exactly one action")
        _require(row.get("action_shape") == [14], "rollout action shape changed")
        _require(row.get("action_finite") is True, "rollout action is non-finite")
        _require(row.get("zc_shape") == [1, 8, 384], "deployment content-token shape changed")
        _require(row.get("zc_finite") is True, "deployment content tokens are non-finite")
    return value


def _audit_run(root: Path, *, control: str, coefficient: float) -> dict[str, Any]:
    summary = _json(root / "training_summary.json", f"{control} training summary")
    sequence = _json(root / "training_sequence_audit.json", f"{control} sequence audit")
    contract = _json(root / "matched_stream_contract.json", f"{control} matched contract")
    identities = _json(root / "artifact_identities.json", f"{control} artifact identities")
    gradients = _json(root / "gradient_audit.json", f"{control} gradient audit")
    updates = _json(root / "parameter_update_audit.json", f"{control} update audit")
    _require((root / "checkpoint.pt").is_file(), f"{control} checkpoint is missing")
    rollout = _audit_rollout(root / "rollout_load_execute.json")
    _require(summary.get("status") == "SMOKE_COMPLETE", f"{control} smoke did not complete")
    _require(summary.get("formal_training_auto_started") is False, "formal training auto-started")
    _require(summary.get("control") == control, f"{control} summary is mislabeled")
    _require(summary.get("steps") == 3, f"{control} must run exactly three optimizer steps")
    _require(float(summary.get("lambda_contrastive", -1)) == coefficient, f"{control} lambda differs")
    _require(
        summary.get("paired_contrastive_gradient_enabled") is (coefficient > 0),
        f"{control} paired gradient switch differs",
    )
    _require(sequence.get("status") == "PASS", f"{control} sequence audit is not PASS")
    _require(contract.get("status") == "PASS", f"{control} matched contract is not PASS")
    _require(
        summary.get("matched_stream_contract_sha256") == contract.get("sha256")
        == sequence.get("matched_stream_contract_sha256"),
        f"{control} matched stream SHA is inconsistent",
    )
    required_identities = {
        "base_checkpoint",
        "base_lineage_manifest",
        "release_paired_binding_manifest",
        "official_text_cache_binding_manifest",
        "paired_action_manifest",
        "paired_action_audit",
        "paired_state_bank",
        "paired_text_cache",
        "paired_train_cache",
    }
    _require(required_identities <= set(identities), f"{control} artifact identities are incomplete")
    _require(gradients.get("status") == "PASS", f"{control} gradient audit is not PASS")
    steps = gradients.get("steps")
    _require(isinstance(steps, list) and len(steps) == 3, f"{control} gradient steps changed")
    for index, row in enumerate(steps, start=1):
        _require(row.get("step") == index, f"{control} gradient step order changed")
        if index >= 2:
            probe = row.get("action_only_probe")
            _require(isinstance(probe, Mapping), f"{control} action-only probe is missing")
            assert isinstance(probe, Mapping)
            for key in ("head_grad_norm", "adapter_attention_grad_norm", "gate_grad_norm"):
                _require(float(probe.get(key, 0.0)) > 0.0, f"{control} {key} did not receive action gradient")
    head_adapter = updates.get("head_and_adapter")
    _require(isinstance(head_adapter, Mapping), f"{control} update audit is missing")
    assert isinstance(head_adapter, Mapping)
    deltas = head_adapter.get("max_abs_delta_by_module")
    _require(isinstance(deltas, Mapping), f"{control} update deltas are missing")
    assert isinstance(deltas, Mapping)
    _require(float(deltas.get("content_head", 0.0)) > 0.0, f"{control} Content Head did not update")
    _require(float(deltas.get("adapter", 0.0)) > 0.0, f"{control} GCA did not update")
    return {
        "summary": summary,
        "sequence": sequence,
        "contract": contract,
        "identities": identities,
        "rollout": rollout,
    }


def audit_pair(
    *,
    materialization_manifest: str | Path,
    c1_run_dir: str | Path,
    c3_run_dir: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(materialization_manifest).expanduser().resolve()
    materialization = _json(manifest_path, "materialization manifest")
    _require(materialization.get("status") == "PASS", "materialization is not PASS")
    _require(materialization.get("scientific_result") is False, "engineering gate claims a scientific result")
    _require(materialization.get("p_mode_selection_evidence") is False, "engineering gate claims P-mode selection")
    config_rows = materialization.get("configs")
    _require(isinstance(config_rows, Mapping), "materialization configs are missing")
    assert isinstance(config_rows, Mapping)
    config_paths: dict[str, Path] = {}
    for name in ("c1", "c3"):
        row = config_rows.get(name)
        _require(isinstance(row, Mapping), f"materialized {name} config is missing")
        assert isinstance(row, Mapping)
        path = Path(str(row.get("path", ""))).expanduser().resolve()
        _require(_sha256(path) == row.get("sha256"), f"materialized {name} config SHA differs")
        config_paths[name] = path
    c1_config = load_config(config_paths["c1"])
    c3_config = load_config(config_paths["c3"])
    validate_execution_ready(c1_config)
    validate_execution_ready(c3_config)
    fairness = validate_c1_c3_pair(c1_config, c3_config)

    c1_root = Path(c1_run_dir).expanduser().resolve()
    c3_root = Path(c3_run_dir).expanduser().resolve()
    _require(Path(c1_config["output_dir"]).resolve() == c1_root, "C1 output differs from config")
    _require(Path(c3_config["output_dir"]).resolve() == c3_root, "C3 output differs from config")
    c1 = _audit_run(c1_root, control="c1_architecture_only", coefficient=0.0)
    c3 = _audit_run(c3_root, control="c3_ours", coefficient=0.1)
    _require(c1["contract"]["sha256"] == c3["contract"]["sha256"], "C1/C3 matched stream contracts differ")
    _require(c1["sequence"] == c3["sequence"], "C1/C3 consumed different sample/state sequences")
    for name in (
        "base_checkpoint",
        "base_lineage_manifest",
        "release_paired_binding_manifest",
        "official_text_cache_binding_manifest",
        "paired_action_manifest",
        "paired_action_audit",
        "paired_state_bank",
        "paired_text_cache",
        "paired_train_cache",
    ):
        _require(c1["identities"][name]["sha256"] == c3["identities"][name]["sha256"], f"C1/C3 {name} differs")
    for key in ("source_fp32_content_head_sha256", "source_fp32_adapter_sha256"):
        _require(c1["summary"]["initialization"][key] == c3["summary"]["initialization"][key], f"C1/C3 {key} differs")
    return {
        "schema_version": 1,
        "kind": "policy_release_c1_c3_engineering_smoke_audit",
        "status": "PASS",
        "scientific_result": False,
        "p_mode_selection_evidence": False,
        "formal_training_auto_started": False,
        "fairness": fairness,
        "matched_stream_contract_sha256": c1["contract"]["sha256"],
        "official_sample_sequence_sha256": c1["sequence"]["official_sample_sequence_sha256"],
        "paired_physical_state_sequence_sha256": c1["sequence"]["paired_physical_state_sequence_sha256"],
        "controls": {
            "c1": {"lambda_contrastive": 0.0, "run_dir": str(c1_root)},
            "c3": {"lambda_contrastive": 0.1, "run_dir": str(c3_root)},
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-manifest", required=True)
    parser.add_argument("--c1-run-dir", required=True)
    parser.add_argument("--c3-run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_pair(
        materialization_manifest=args.materialization_manifest,
        c1_run_dir=args.c1_run_dir,
        c3_run_dir=args.c3_run_dir,
    )
    target = Path(args.output_json).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(not target.exists(), f"refusing to overwrite pair audit: {target}")
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ReleaseEngineeringSmokeAuditError", "audit_pair"]
