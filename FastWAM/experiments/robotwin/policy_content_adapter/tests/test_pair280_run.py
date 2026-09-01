from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.pair280_protocol import (
    PAIR280_PROFILE_ID,
)
from experiments.robotwin.policy_content_adapter.pair280_run import (
    Pair280RunError,
    _correct_formal_summary_labels,
    audit_run,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _smoke_fixture(root: Path) -> None:
    root.mkdir()
    (root / "checkpoint.pt").write_bytes(b"checkpoint")
    _write_json(
        root / "training_summary.json",
        {
            "steps": 3,
            "regime": "p_v2",
            "control": "c3_ours",
            "paired_sampling_profile": PAIR280_PROFILE_ID,
        },
    )
    _write_json(
        root / "training_sequence_audit.json",
        {"paired_physical_state_count": 32},
    )
    _write_json(root / "gradient_audit.json", {"status": "PASS"})
    fieldnames = (
        "step",
        "paired_contrastive_active",
        "paired_active_index",
        "paired_physical_state_ids",
        "loss_contrastive",
    )
    rows = [
        {
            "step": 1,
            "paired_contrastive_active": False,
            "paired_active_index": "",
            "paired_physical_state_ids": "",
            "loss_contrastive": 0.0,
        }
    ]
    for step, active_index in ((2, 0), (3, 1)):
        state_ids = [
            f"place_a2b_left/content_{index:06d}/frame_{step:06d}"
            for index in range(6)
        ]
        state_ids += [
            f"open_microwave/content_{index:06d}/frame_{step:06d}"
            for index in range(6)
        ]
        state_ids += [
            f"move_stapler_pad/content_{index:06d}/frame_{step:06d}"
            for index in range(4)
        ]
        rows.append(
            {
                "step": step,
                "paired_contrastive_active": True,
                "paired_active_index": active_index,
                "paired_physical_state_ids": ";".join(state_ids),
                "loss_contrastive": 1.0,
            }
        )
    with (root / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_pair280_smoke_audit_proves_inactive_and_active_rows(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    _smoke_fixture(root)
    audit = audit_run(root, smoke=True)
    assert audit["status"] == "PASS"
    assert audit["paired_active_steps"] == 2
    assert audit["inactive_steps_action_only_verified"] == 1
    assert audit["paired_exposures"] == 32


def test_pair280_smoke_audit_rejects_nonuniform_active_placement(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    _smoke_fixture(root)
    rows = list(csv.DictReader((root / "train_log.csv").open(encoding="utf-8")))
    rows[0]["paired_contrastive_active"] = "True"
    rows[0]["paired_active_index"] = "0"
    rows[0]["paired_physical_state_ids"] = rows[1]["paired_physical_state_ids"]
    rows[1]["paired_contrastive_active"] = "False"
    rows[1]["paired_active_index"] = ""
    rows[1]["paired_physical_state_ids"] = ""
    rows[1]["loss_contrastive"] = "0.0"
    with (root / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(Pair280RunError, match="uniform schedule"):
        audit_run(root, smoke=True)


def test_pair280_formal_label_reconciliation_is_exact_and_fail_closed() -> None:
    summary = {
        "status": "SMOKE_COMPLETE",
        "steps": 18_215,
        "deliverable_status": {
            "implementation": "PASS",
            "formal_long_training": "NOT_STARTED",
        },
        "scientific_payload": {"unchanged": True},
    }
    requested = {
        "execution": {
            "runner": "policy_pair280_posttraining",
            "long_formal_training": True,
        },
        "training": {"max_steps": 18_215},
        "paired": {"sampling_profile": PAIR280_PROFILE_ID},
    }
    strict = {
        "status": "PASS",
        "smoke": False,
        "steps": 18_215,
        "paired_active_steps": 15_750,
        "paired_unique_states": 25_200,
        "paired_exposures_per_state": 10,
    }
    corrected = _correct_formal_summary_labels(summary, requested, strict)
    assert corrected["status"] == "COMPLETE"
    assert corrected["deliverable_status"]["formal_long_training"] == "PASS"
    assert corrected["scientific_payload"] == summary["scientific_payload"]
    assert corrected["formal_label_reconciliation"]["scientific_payload_changed"] is False
    assert summary["status"] == "SMOKE_COMPLETE"

    bad = dict(strict)
    bad["paired_exposures_per_state"] = 9
    with pytest.raises(Pair280RunError, match="formal audit is incomplete"):
        _correct_formal_summary_labels(summary, requested, bad)
