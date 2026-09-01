from __future__ import annotations

import copy
import hashlib

import pytest

from experiments.robotwin.policy_content_adapter.evaluation_protocol import (
    DOMAINS,
    PROFILE,
    SCHEMA_VERSION,
    TASKS,
    EvaluationProtocolError,
    audit_and_summarize,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _payload(*, include_c0: bool = False) -> dict:
    ancestry = {
        "base_checkpoint_sha256": _sha("fixed-release"),
        "dataset_stats_sha256": _sha("release-stats"),
        "base_lineage_manifest_sha256": _sha("author-release-lineage"),
        "runtime_source_sha256": _sha("runtime"),
    }
    records: list[dict] = []
    if include_c0:
        for task_index, task in enumerate(TASKS):
            for domain in DOMAINS:
                base = 0.70 if domain == "clean" else 0.40
                records.append(
                    {
                        "control": "c0_base",
                        "training_seed": None,
                        **ancestry,
                        **{field: None for field in (
                            "policy_regime", "head_init_sha256", "gca_init_sha256",
                            "stage2_recipe_sha256", "p_mode_selection_manifest_sha256",
                            "official_sample_sequence_sha256",
                            "paired_physical_state_sequence_sha256",
                            "matched_stream_contract_sha256",
                        )},
                        "lambda_contrastive": None,
                        "paired_contrastive_gradient_enabled": None,
                        "task": task,
                        "domain": domain,
                        "episodes": 100,
                        "success_rate": base - 0.05 + 0.01 * task_index,
                        "rollout_protocol_id": "robotwin_policy_online_v2",
                        "simulator_seed_bank_id": "locked-final-bank",
                    }
                )
    for control, offset in (("c1_architecture_only", 0.0), ("c3_ours", 0.10)):
        for seed in (1, 2, 3):
            for task_index, task in enumerate(TASKS):
                for domain in DOMAINS:
                    base = 0.70 if domain == "clean" else 0.40
                    records.append(
                        {
                            "control": control,
                            "training_seed": seed,
                            **ancestry,
                            "policy_regime": "p_v1",
                            "head_init_sha256": _sha(f"head-{seed}"),
                            "gca_init_sha256": _sha(f"gca-{seed}"),
                            "stage2_recipe_sha256": _sha(f"recipe-{seed}"),
                            "p_mode_selection_manifest_sha256": _sha("selection"),
                            "official_sample_sequence_sha256": _sha(f"official-sequence-{seed}"),
                            "paired_physical_state_sequence_sha256": _sha(f"paired-sequence-{seed}"),
                            "matched_stream_contract_sha256": _sha(f"matched-stream-{seed}"),
                            "lambda_contrastive": 0.0 if control.startswith("c1") else 0.1,
                            "paired_contrastive_gradient_enabled": control == "c3_ours",
                            "task": task,
                            "domain": domain,
                            "episodes": 100,
                            "success_rate": base + offset + 0.01 * task_index,
                            "rollout_protocol_id": "robotwin_policy_online_v2",
                            "simulator_seed_bank_id": "locked-final-bank",
                        }
                    )
    return {"schema_version": SCHEMA_VERSION, "profile": PROFILE, "records": records}


def test_primary_release_matrix_is_exactly_36_matched_c1_c3_records() -> None:
    report = audit_and_summarize(_payload())
    assert report["status"] == "PASS"
    assert report["schema_version"] == 4
    assert report["profile"] == "c1_c3_primary"
    assert report["record_count"] == report["required_primary_record_count"] == 36
    assert report["controls"]["c3_ours"]["macro_average"]["official_random"]["mean"] == pytest.approx(0.51)
    assert report["comparisons"]["c3_ours_minus_c1_architecture_only"]["macro_average"]["official_random"]["mean"] == pytest.approx(0.10)
    assert report["primary_comparison"] == "c3_ours_minus_c1_architecture_only"
    assert set(report["comparisons"]) == {"c3_ours_minus_c1_architecture_only"}
    assert report["optional_c0_reference"]["included"] is False
    assert report["c0_reference"] is None
    assert report["seed_pairing"]["c0_training_seed"] == "not_included"
    assert set(report["fairness_identity_audit"]["by_training_seed"]) == {"1", "2", "3"}


def test_complete_c0_may_be_attached_as_supplementary_reference() -> None:
    report = audit_and_summarize(_payload(include_c0=True))
    assert report["record_count"] == 42
    assert report["optional_c0_reference"]["included"] is True
    assert report["c0_reference"]["training_seed"] is None
    assert report["c0_reference"]["fixed_checkpoint"] is True
    assert report["c0_reference"]["macro_average"]["official_random"]["success_rate"] == pytest.approx(0.36)
    assert report["comparisons"]["c3_ours_minus_c0_base"]["macro_average"]["official_random"]["mean"] == pytest.approx(0.15)
    assert report["seed_pairing"]["c0_training_seed"] is None


def test_partial_c0_reference_is_rejected() -> None:
    payload = _payload(include_c0=True)
    payload["records"].pop(0)
    with pytest.raises(EvaluationProtocolError, match="all-or-none"):
        audit_and_summarize(payload)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("rollout_protocol_id", "rollout protocol mismatch"),
        ("simulator_seed_bank_id", "seed-bank mismatch"),
    ),
)
def test_optional_c0_must_share_primary_online_protocol(
    field: str, message: str
) -> None:
    payload = _payload(include_c0=True)
    payload["records"][0][field] = "different-c0-protocol"
    with pytest.raises(EvaluationProtocolError, match=message):
        audit_and_summarize(payload)


def test_r3_and_c2_are_not_release_main_table_controls() -> None:
    payload = _payload()
    payload["records"][0]["domain"] = "r3"
    with pytest.raises(EvaluationProtocolError, match="not Policy tests"):
        audit_and_summarize(payload)
    payload = _payload()
    payload["records"][0]["control"] = "c2_naive_aug"
    with pytest.raises(EvaluationProtocolError, match="unsupported control"):
        audit_and_summarize(payload)


def test_seed_bank_and_cells_are_exact() -> None:
    payload = _payload()
    payload["records"][0]["simulator_seed_bank_id"] = "different"
    with pytest.raises(EvaluationProtocolError, match="seed-bank mismatch"):
        audit_and_summarize(payload)
    payload = _payload()
    payload["records"].pop()
    with pytest.raises(EvaluationProtocolError, match="missing required evaluation cells"):
        audit_and_summarize(payload)
    payload = _payload()
    payload["records"].append(copy.deepcopy(payload["records"][0]))
    with pytest.raises(EvaluationProtocolError, match="duplicate evaluation cell"):
        audit_and_summarize(payload)


def test_c0_has_no_training_seed_and_is_not_duplicated() -> None:
    payload = _payload(include_c0=True)
    payload["records"][0]["training_seed"] = 1
    with pytest.raises(EvaluationProtocolError, match="fixed C0 training_seed must be null"):
        audit_and_summarize(payload)
    payload = _payload(include_c0=True)
    duplicate = copy.deepcopy(payload["records"][0])
    duplicate["training_seed"] = 2
    payload["records"].append(duplicate)
    with pytest.raises(EvaluationProtocolError, match="fixed C0 training_seed must be null"):
        audit_and_summarize(payload)


def test_episode_count_and_exact_success_count_are_locked() -> None:
    payload = _payload()
    payload["records"][0]["episodes"] = 99
    with pytest.raises(EvaluationProtocolError, match="exactly 100 episodes"):
        audit_and_summarize(payload)
    payload = _payload()
    payload["records"][0]["success_rate"] = 0.655
    with pytest.raises(EvaluationProtocolError, match="exact episode count"):
        audit_and_summarize(payload)


@pytest.mark.parametrize("field", (
    "base_checkpoint_sha256", "dataset_stats_sha256",
    "base_lineage_manifest_sha256", "runtime_source_sha256",
))
def test_all_controls_share_one_fixed_release_ancestry(field: str) -> None:
    payload = _payload()
    for record in payload["records"]:
        if record["control"] == "c3_ours" and record["training_seed"] == 1:
            record[field] = _sha("different-" + field)
    with pytest.raises(EvaluationProtocolError, match="fixed B_release ancestry mismatch"):
        audit_and_summarize(payload)


@pytest.mark.parametrize(("field", "replacement"), (
    ("policy_regime", "p_v2"),
    ("head_init_sha256", _sha("other-head")),
    ("gca_init_sha256", _sha("other-gca")),
    ("stage2_recipe_sha256", _sha("other-recipe")),
    ("p_mode_selection_manifest_sha256", _sha("other-selection")),
    ("official_sample_sequence_sha256", _sha("other-official-sequence")),
    ("paired_physical_state_sequence_sha256", _sha("other-paired-sequence")),
    ("matched_stream_contract_sha256", _sha("other-stream-contract")),
))
def test_c1_c3_share_stage2_identity_within_seed(field: str, replacement: str) -> None:
    payload = _payload()
    for record in payload["records"]:
        if record["control"] == "c3_ours" and record["training_seed"] == 2:
            record[field] = replacement
    with pytest.raises(EvaluationProtocolError, match="C1/C3 Stage-2 fairness identity mismatch"):
        audit_and_summarize(payload)


def test_c0_must_not_claim_stage2_identity() -> None:
    payload = _payload(include_c0=True)
    payload["records"][0]["policy_regime"] = "p_v1"
    with pytest.raises(EvaluationProtocolError, match="C0 policy_regime must be null"):
        audit_and_summarize(payload)


def test_c1_c3_treatment_coefficients_are_explicit() -> None:
    payload = _payload()
    target = next(record for record in payload["records"] if record["control"] == "c1_architecture_only")
    target["lambda_contrastive"] = 0.1
    target["paired_contrastive_gradient_enabled"] = True
    with pytest.raises(EvaluationProtocolError, match="C1 must have lambda=0"):
        audit_and_summarize(payload)


def test_all_stage2_seeds_share_global_selection() -> None:
    payload = _payload()
    for record in payload["records"]:
        if record["control"] != "c0_base" and record["training_seed"] == 2:
            record["p_mode_selection_manifest_sha256"] = _sha("seed2-selection")
    with pytest.raises(EvaluationProtocolError, match="global P-mode selection"):
        audit_and_summarize(payload)


def test_schema_v4_requires_c1_c3_primary_profile() -> None:
    payload = _payload()
    payload.pop("profile")
    with pytest.raises(EvaluationProtocolError, match="evaluation profile"):
        audit_and_summarize(payload)
