"""Create audited pilot summaries for the P-v2 ActionDiT follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from .config_audit import load_config
from .pv2_actiondit_followup_audit import (
    audit_materialization,
    evaluate_pilot_gate,
)


class Pv2FollowupReportError(ValueError):
    """The pilot evidence is incomplete or differs from the locked decision."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2FollowupReportError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"artifact missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pv2FollowupReportError(f"cannot parse {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise Pv2FollowupReportError(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _window(values: list[float], count: int = 100) -> dict[str, float]:
    _require(bool(values), "training metric window is empty")
    n = min(count, len(values))
    return {"first": fmean(values[:n]), "last": fmean(values[-n:]), "n": n}


def _training_mechanism(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(config["output_dir"]).resolve()
    rows: list[dict[str, str]]
    with (root / "train_log.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == 1800, f"training log is not 1800 steps: {root}")

    def values(field: str) -> list[float]:
        parsed = [float(row[field]) for row in rows]
        _require(all(math.isfinite(value) for value in parsed), f"non-finite {field}")
        return parsed

    positive = values("positive_similarity")
    negative = values("negative_similarity")
    margins = [a - b for a, b in zip(positive, negative, strict=True)]
    summary = _json(root / "training_summary.json", "training summary")
    updates = _json(root / "parameter_update_audit.json", "parameter update audit")
    return {
        "control": config["control"],
        "lambda_contrastive": float(config["loss"]["lambda_contrastive"]),
        "action_loss": _window(values("loss_action")),
        "contrastive_loss_diagnostic": _window(values("loss_contrastive")),
        "positive_minus_negative_similarity": _window(margins),
        "action_dit_gradient_norm": _window(values("action_dit_grad_norm")),
        "action_dit_update": {
            key: updates["action_dit"][key]
            for key in (
                "changed_fraction",
                "deployment_visible_changed_fraction",
                "max_abs_delta",
                "mean_abs_delta",
                "required_changed_strata",
                "bf16_deployment_category_visibility",
            )
        },
        "head_gca_update": updates["head_and_adapter"],
        "final_gate_raw": float(summary["final_gate_raw"]),
        "checkpoint": _identity(root / "checkpoint.pt"),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    cells = summary["pilot"]["cells"]
    lines = [
        "# P-v2 ActionDiT Follow-up Pilot",
        "",
        "> Post-hoc mechanism study. P-v1 remains the primary experiment.",
        "",
        f"Pilot gate: **{'PASS' if summary['pilot']['gate_passed'] else 'FAIL'}**",
        "",
        "| Task | C1 Clean | C3 Clean | Delta | C1 Random | C3 Random | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in ("place_a2b_left", "open_microwave", "move_stapler_pad"):
        c1 = cells["c1"][task]
        c3 = cells["c3"][task]
        lines.append(
            f"| {task} | {100*c1['clean']:.1f}% | {100*c3['clean']:.1f}% | "
            f"{100*(c3['clean']-c1['clean']):+.1f}pp | "
            f"{100*c1['official_random']:.1f}% | {100*c3['official_random']:.1f}% | "
            f"{100*(c3['official_random']-c1['official_random']):+.1f}pp |"
        )
    macro = summary["pilot"]["macro"]
    delta = summary["pilot"]["delta"]
    lines.extend(
        [
            f"| Macro | {100*macro['c1']['clean']:.1f}% | {100*macro['c3']['clean']:.1f}% | "
            f"{100*delta['clean']:+.1f}pp | {100*macro['c1']['official_random']:.1f}% | "
            f"{100*macro['c3']['official_random']:.1f}% | {100*delta['official_random']:+.1f}pp |",
            "",
            "Locked gate requires Random >= +3pp and Clean >= -3pp.",
            "",
            "## Mechanism evidence",
            "",
        ]
    )
    for short in ("c1", "c3"):
        row = summary["training_mechanism"][short]
        lines.append(
            f"- {short.upper()}: action loss first/last window "
            f"{row['action_loss']['first']:.6g}/{row['action_loss']['last']:.6g}; "
            f"contrastive diagnostic {row['contrastive_loss_diagnostic']['first']:.4f}/"
            f"{row['contrastive_loss_diagnostic']['last']:.4f}; "
            f"ActionDiT sampled changed fraction {100*row['action_dit_update']['changed_fraction']:.2f}%; "
            f"final GCA gate {row['final_gate_raw']:+.6g}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["conclusion"],
            "",
            "No result-driven change to LR, lambda, steps, batch, or gate is permitted.",
            "",
        ]
    )
    return "\n".join(lines)


def build_pilot_report(
    *,
    materialization_manifest: str | Path,
    pilot_decision: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    materialization_path = Path(materialization_manifest).expanduser().resolve()
    decision_path = Path(pilot_decision).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    _require(not destination.exists(), f"refusing to reuse report directory: {destination}")
    prelaunch = audit_materialization(materialization_path)
    decision = _json(decision_path, "pilot decision")
    _require(
        decision.get("kind") == "policy_pv2_actiondit_followup_pilot_decision"
        and decision.get("schema_version") == 1
        and decision.get("status") == "PASS",
        "pilot decision kind/version/status differs",
    )
    rollout_identities = decision.get("rollout_manifests")
    _require(isinstance(rollout_identities, Mapping), "pilot rollout identities missing")
    evaluation_amendment = decision.get("evaluation_amendment")
    _require(
        isinstance(evaluation_amendment, Mapping)
        and Path(str(evaluation_amendment.get("path", ""))).is_absolute(),
        "pilot decision lacks the 100-episode evaluation amendment",
    )
    recomputed = evaluate_pilot_gate(
        materialization_path,
        c1_rollout_manifest=rollout_identities["c1"]["path"],
        c3_rollout_manifest=rollout_identities["c3"]["path"],
        evaluation_amendment=evaluation_amendment["path"],
    )
    for field in (
        "pilot_gate_passed",
        "next_action",
        "cells",
        "macro",
        "delta",
        "locked_thresholds",
        "conditions",
    ):
        _require(recomputed[field] == decision[field], f"pilot decision changed at {field}")

    materialization = _json(materialization_path, "materialization manifest")
    mechanism = {
        short: _training_mechanism(
            Path(materialization["configs"]["pilot"][short]["path"]).resolve()
        )
        for short in ("c1", "c3")
    }
    if decision["pilot_gate_passed"]:
        conclusion = (
            "Pilot passed both locked thresholds. Expansion to seeds 2/3 and the "
            "unopened seed59 confirmatory bank is authorized; this pilot alone is not "
            "the final three-seed result."
        )
    else:
        failed = [name for name, passed in decision["conditions"].items() if not passed]
        conclusion = (
            "Pilot failed the locked condition(s): "
            + ", ".join(failed)
            + ". Expansion stops; the result does not authorize tuning or seed59 access."
        )
    summary = {
        "schema_version": 1,
        "kind": "policy_pv2_actiondit_followup_pilot_summary",
        "status": "PASS",
        "study_classification": "post_hoc_mechanism_study",
        "primary_pv1_remains_authoritative": True,
        "materialization": _identity(materialization_path),
        "implementation_protocol_audit": prelaunch,
        "pilot_decision": _identity(decision_path),
        "pilot": {
            "training_seed": 1,
            "simulator_seed": 53,
            "episodes_per_task_domain": 100,
            "episodes_per_control": 600,
            "evaluation_amendment": evaluation_amendment,
            "gate_passed": bool(decision["pilot_gate_passed"]),
            "next_action": decision["next_action"],
            "cells": decision["cells"],
            "macro": decision["macro"],
            "delta": decision["delta"],
            "thresholds": decision["locked_thresholds"],
            "conditions": decision["conditions"],
        },
        "training_mechanism": mechanism,
        "conclusion": conclusion,
        "claim_boundary": {
            "may_replace_pv1_primary": False,
            "episode_pairing": "not_claimed_shared_starting_seed_only",
            "online_policy_metrics": ["clean_success_rate", "official_random_success_rate"],
            "r3_policy_success_reported": False,
            "result_driven_tuning_allowed": False,
        },
    }
    summary_json = destination / "pilot_summary.json"
    summary_md = destination / "pilot_summary.md"
    _write_new(summary_json, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    _write_new(summary_md, _markdown(summary).encode("utf-8"))
    audit = {
        "schema_version": 1,
        "kind": "policy_pv2_actiondit_followup_pilot_report_audit",
        "status": "PASS",
        "pilot_gate_passed": bool(decision["pilot_gate_passed"]),
        "terminal_if_failed": not bool(decision["pilot_gate_passed"]),
        "summary_json": _identity(summary_json),
        "summary_md": _identity(summary_md),
        "decision": _identity(decision_path),
        "materialization": _identity(materialization_path),
        "seeds_2_3_authorized": bool(decision["pilot_gate_passed"]),
        "confirmatory_seed59_authorized": bool(decision["pilot_gate_passed"]),
    }
    audit_path = destination / "pilot_report_audit.json"
    _write_new(audit_path, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    outputs = {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "audit": str(audit_path),
    }
    if not decision["pilot_gate_passed"]:
        experiment_root = destination.parent
        for relative in (
            "configs/seed_2",
            "configs/seed_3",
            "runs/seed_2",
            "runs/seed_3",
            "manifests/confirmatory_seed59_bank.json",
        ):
            _require(
                not (experiment_root / relative).exists(),
                f"failed pilot must not have expansion artifact: {relative}",
            )
        final_summary = {
            **summary,
            "kind": "policy_pv2_actiondit_followup_final_summary",
            "goal_terminal_condition": "PILOT_FAILED_EXPANSION_STOPPED",
            "seeds_2_3_started": False,
            "confirmatory_seed59_opened": False,
        }
        final_json = experiment_root / "summary.json"
        final_md = experiment_root / "summary.md"
        _write_new(
            final_json,
            (json.dumps(final_summary, indent=2, sort_keys=True) + "\n").encode(),
        )
        _write_new(final_md, _markdown(final_summary).encode("utf-8"))
        completion = {
            "schema_version": 1,
            "kind": "policy_pv2_actiondit_followup_completion_audit",
            "status": "PASS",
            "goal_terminal_condition": "PILOT_FAILED_EXPANSION_STOPPED",
            "pilot_gate_passed": False,
            "expansion_stopped": True,
            "result_driven_tuning_performed": False,
            "seeds_2_3_started": False,
            "confirmatory_seed59_opened": False,
            "summary_json": _identity(final_json),
            "summary_md": _identity(final_md),
            "pilot_report_audit": _identity(audit_path),
            "pilot_decision": _identity(decision_path),
            "primary_pv1_remains_authoritative": True,
        }
        completion_path = experiment_root / "completion_audit.json"
        _write_new(
            completion_path,
            (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode(),
        )
        outputs.update(
            {
                "final_summary_json": str(final_json),
                "final_summary_md": str(final_md),
                "completion_audit": str(completion_path),
            }
        )
    return {**summary, "outputs": outputs}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-manifest", required=True)
    parser.add_argument("--pilot-decision", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_pilot_report(
        materialization_manifest=args.materialization_manifest,
        pilot_decision=args.pilot_decision,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Pv2FollowupReportError", "build_pilot_report"]
