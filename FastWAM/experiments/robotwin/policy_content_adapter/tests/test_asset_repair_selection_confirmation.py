from __future__ import annotations

from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import (
    asset_repair_selection_confirmation as confirmation,
)


def test_clean_guard_exact_proof_makes_p_v1_winner_random_invariant() -> None:
    proof = confirmation.prove_clean_guard_singleton(
        {"p_v1": 43, "p_v2": 36},
        clean_episodes=60,
        max_drop=0.05,
    )
    assert proof["clean_macro_exact"] == {"p_v1": "43/60", "p_v2": "3/5"}
    assert proof["eligibility_threshold_exact"] == "2/3"
    assert proof["eligible_regimes"] == ["p_v1"]
    assert proof["winner"] == "p_v1"
    assert proof["random_score_quantifier"]["p_v2"] == "arbitrary_in_[0,1]"


def test_clean_guard_refuses_confirmation_when_random_can_affect_winner() -> None:
    with pytest.raises(
        confirmation.AssetRepairSelectionError,
        match="Random revalidation is required",
    ):
        confirmation.prove_clean_guard_singleton(
            {"p_v1": 43, "p_v2": 41},
            clean_episodes=60,
            max_drop=0.05,
        )


def test_asset_audit_rejects_checkpoint_weight_change(tmp_path: Path) -> None:
    payload = {
        "kind": confirmation.ASSET_AUDIT_KIND,
        "schema_version": confirmation.ASSET_AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "operation": {
            "delete_files": False,
            "copy_missing_only": True,
            "missing_files_copied": 1,
        },
        "pre_repair": {"observed_missing_objaverse_model_directories": 1},
        "post_repair": {
            "missing_source_regular_files": 0,
            "other_regular_file_checksum_differences": 0,
        },
        "rollout_recovery": {
            "official_random_cells_started_before_repair_are_invalid": True,
            "checkpoint_weights_changed": True,
        },
    }
    with pytest.raises(
        confirmation.AssetRepairSelectionError,
        match="checkpoint weights",
    ):
        confirmation.validate_asset_repair_audit_payload(payload)


def test_stock_runner_requires_asset_repair_continuation() -> None:
    root = Path(confirmation.__file__).resolve().parent
    runner = (root / "run_release_formal_stock_rollout.sh").read_text(encoding="utf-8")
    assert "ASSET_REPAIR_CONTINUATION_PATH" in runner
    assert "validate-continuation" in runner
    assert "audit_asset_repair_continuation" in runner
