from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.config_audit import (
    ConfigAuditError,
    load_config,
    validate_c1_c3_pair,
)
from experiments.robotwin.policy_content_adapter.materialize_release_formal_c1c3 import (
    CONFIG_DIR,
    CONTROLS,
    DEFAULT_MAX_STEPS,
    DEFAULT_RECIPE_AMENDMENT,
    FORMAL_SEEDS,
    _finalize_config,
    _resolved_prelock_config,
    validate_formal_matrix_configs,
)
from experiments.robotwin.policy_content_adapter.p_mode_selection import (
    canonical_sha256,
    formal_config_protocol_projection,
)
from experiments.robotwin.policy_content_adapter.release_formal_c1c3_audit import (
    FormalC1C3AuditError,
    _audit_formal_gradient_audit,
)
from experiments.robotwin.policy_content_adapter.train import stage2_step_rng_seed


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _gradient_report(norm: float) -> dict:
    return {
        "gradient_tensors": int(norm > 0.0),
        "gradient_norm": norm,
        "all_finite": True,
    }


def _formal_gradient_fixture(*, zero_steps: set[int] | None = None) -> dict:
    zero_steps = {1, 18} if zero_steps is None else set(zero_steps)
    steps: list[dict] = []
    positive_ordinal = 0
    gate_open = False
    for step in range(1, DEFAULT_MAX_STEPS + 1):
        zero_weight = step in zero_steps
        if zero_weight:
            # A C3 contrastive gradient may reach the combined Head even though
            # every action-only path is exactly zero.
            head_norm = 0.25
            adapter_norm = attention_norm = 0.0
            probe_head = probe_attention = probe_gate = content_norm = 0.0
            loss_action = action_weight_min = action_weight_max = 0.0
            action_effective_weight_sum = 0.0
            zero_reason = "scheduler_zero_weight"
        else:
            positive_ordinal += 1
            gate_open = True
            first_positive = positive_ordinal == 1
            head_norm = 0.0 if first_positive else 0.25
            adapter_norm = 0.5
            attention_norm = 0.0 if first_positive else 0.2
            probe_head = 0.0 if first_positive else 0.15
            probe_attention = 0.0 if first_positive else 0.1
            probe_gate = 0.3
            content_norm = 0.0 if first_positive else 0.4
            loss_action = 0.2
            action_weight_min = 0.01
            action_weight_max = 0.5
            action_effective_weight_sum = 12.0
            zero_reason = "none"
        steps.append(
            {
                "step": step,
                "gate_raw_after_step": 1.0e-4 if gate_open else 0.0,
                "combined": {
                    "content_head": _gradient_report(head_norm),
                    "adapter": _gradient_report(adapter_norm),
                    "adapter_attention_action_only_by_construction": _gradient_report(
                        attention_norm
                    ),
                },
                "action_only_probe": {
                    "head_grad_norm": probe_head,
                    "adapter_attention_grad_norm": probe_attention,
                    "gate_grad_norm": probe_gate,
                    "all_finite": True,
                },
                "action_only_official_content_token_grad_norm": content_norm,
                "loss_action": loss_action,
                "action_weight_min": action_weight_min,
                "action_weight_max": action_weight_max,
                "action_effective_weight_sum": action_effective_weight_sum,
                "zero_action_signal_reason": zero_reason,
                "action_supervision_signal_positive": not zero_weight,
                "zero_weight_action_step": zero_weight,
            }
        )
    return {
        "status": "PASS",
        "regime": "p_v1",
        "positive_action_signal_steps": DEFAULT_MAX_STEPS - len(zero_steps),
        "zero_action_signal_steps": len(zero_steps),
        "steps": steps,
    }


def _prelock_matrix(tmp_path: Path) -> dict[int, dict[str, dict]]:
    templates = {
        "c1_architecture_only": load_config(
            CONFIG_DIR / "formal_c1_architecture_only.yaml"
        ),
        "c3_ours": load_config(CONFIG_DIR / "formal_c3_ours.yaml"),
    }
    matrix: dict[int, dict[str, dict]] = {}
    for seed in FORMAL_SEEDS:
        matrix[seed] = {}
        for control in CONTROLS:
            matrix[seed][control] = _resolved_prelock_config(
                template=templates[control],
                control=control,
                seed=seed,
                output_root=tmp_path,
                selection_path=(tmp_path / "selection.json").resolve(),
                selection_sha256=_sha("selection"),
                binding_path=(tmp_path / "binding.json").resolve(),
                binding_sha256=_sha("binding"),
                paired_text_cache=(tmp_path / "paired_text").resolve(),
                paired_text_cache_sha256=_sha("paired-text"),
                paired_cache=(tmp_path / "cache.pt").resolve(),
                paired_cache_sha256=_sha("paired-cache"),
                official_text_cache=(tmp_path / "official_text").resolve(),
                official_text_binding=(tmp_path / "official_binding.json").resolve(),
                official_text_binding_sha256=_sha("official-binding"),
                amendment_path=DEFAULT_RECIPE_AMENDMENT,
                lock_path=(tmp_path / "formal_lock.json").resolve(),
                final_seed_bank_path=(tmp_path / "final_bank.json").resolve(),
            )
    return matrix


def test_formal_recipe_is_explicit_and_transparently_amended() -> None:
    amendment = json.loads(DEFAULT_RECIPE_AMENDMENT.read_text())
    assert amendment["status"] == "PASS"
    assert "does not claim" in amendment["disclosure"]["non_retroactive_claim"]
    assert amendment["locked_recipe"] == {
        "selected_policy_regime": "p_v1",
        "stage2_training_seeds": [1, 2, 3],
        "max_steps": 1800,
        "official_batch_size_per_rank": 1,
        "paired_groups_per_batch_per_rank": 2,
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "effective_official_global_batch": 1,
        "effective_paired_groups_per_step": 2,
        "head_adapter_lr": 1.0e-4,
        "action_dit_lr_inactive_under_p_v1": 1.0e-5,
        "lr_scheduler": "constant",
        "lambda_contrastive": {
            "c1_architecture_only": 0.0,
            "c3_ours": 0.1,
        },
        "temperature": 0.07,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
        "mixed_precision": "bf16",
        "trainable_parameter_dtype": "fp32",
    }
    assert amendment["exposure_accounting"]["paired_exposure_equivalent_passes"] == 5.0
    assert "not five strict" in amendment["exposure_accounting"]["terminology_guard"]


def test_six_formal_prelock_configs_are_seed_matched_and_c1_c3_fair(
    tmp_path: Path,
) -> None:
    matrix = _prelock_matrix(tmp_path)
    audit = validate_formal_matrix_configs(matrix)
    assert audit["status"] == "PASS"
    assert [row["training_seed"] for row in audit["rows"]] == [1, 2, 3]
    assert len(
        {
            row["expected_initialization"]["source_fp32_content_head_sha256"]
            for row in audit["rows"]
        }
    ) == 3
    for seed in FORMAL_SEEDS:
        c1 = matrix[seed]["c1_architecture_only"]
        c3 = matrix[seed]["c3_ours"]
        assert c1["training"]["max_steps"] == c3["training"]["max_steps"] == DEFAULT_MAX_STEPS
        assert c1["loss"]["lambda_contrastive"] == 0.0
        assert c3["loss"]["lambda_contrastive"] == 0.1
        assert validate_c1_c3_pair(c1, c3)["fairness"] == "PASS"


def test_formal_seed_step_rng_is_c1_c3_matched_and_cross_seed_disjoint() -> None:
    by_seed = {
        seed: tuple(
            stage2_step_rng_seed(seed, step, stream="official")
            for step in range(DEFAULT_MAX_STEPS)
        )
        for seed in FORMAL_SEEDS
    }
    # C1 and C3 call the same pure mapping with the same formal seed.  Repeating
    # it here represents the two controls without introducing control as a key.
    for seed in FORMAL_SEEDS:
        c1 = by_seed[seed]
        c3 = tuple(
            stage2_step_rng_seed(seed, step, stream="official")
            for step in range(DEFAULT_MAX_STEPS)
        )
        assert c1 == c3

    flattened = [value for sequence in by_seed.values() for value in sequence]
    assert len(set(flattened)) == len(FORMAL_SEEDS) * DEFAULT_MAX_STEPS
    assert set(by_seed[1]).isdisjoint(by_seed[2])
    assert set(by_seed[1]).isdisjoint(by_seed[3])
    assert set(by_seed[2]).isdisjoint(by_seed[3])


def test_formal_pair_rejects_any_second_treatment_difference(tmp_path: Path) -> None:
    matrix = _prelock_matrix(tmp_path)
    matrix[2]["c3_ours"]["training"]["num_workers"] = 5
    with pytest.raises(ConfigAuditError, match="unfair common Stage-2 mismatch"):
        validate_formal_matrix_configs(matrix)


def test_executable_copy_preserves_cycle_free_locked_projection(tmp_path: Path) -> None:
    prelock = _prelock_matrix(tmp_path)[1]["c1_architecture_only"]
    final = _finalize_config(
        prelock,
        lock_sha256=_sha("formal-lock"),
        final_seed_bank_sha256=_sha("final-bank"),
        final_seed_bank_id="robotwin-seed-bank-v3:" + _sha("bank-id"),
    )
    assert final["execution"]["runnable"] is True
    assert canonical_sha256(formal_config_protocol_projection(prelock)) == canonical_sha256(
        formal_config_protocol_projection(final)
    )


def test_cross_seed_drift_outside_seed_fields_fails_closed(tmp_path: Path) -> None:
    matrix = _prelock_matrix(tmp_path)
    matrix[3]["c1_architecture_only"]["optimizer"]["head_adapter_lr"] = 2.0e-4
    # Make the same within-seed change so the C1/C3 pair passes; the cross-seed
    # projection must still catch the unreviewed seed-specific recipe.
    matrix[3]["c3_ours"]["optimizer"]["head_adapter_lr"] = 2.0e-4
    with pytest.raises(Exception, match="across seeds|Head/GCA LR"):
        validate_formal_matrix_configs(matrix)


def test_formal_runner_is_cpu_safe_by_default_and_has_no_rollout() -> None:
    runner = (CONFIG_DIR.parent / "run_release_formal_c1_c3.sh").read_text()
    assert 'PHASE="${PHASE:-prepare}"' in runner
    assert "CONFIRM_FORMAL_TRAINING=YES" in runner
    assert "rollout_policy" not in runner
    assert "eval_robotwin_single" not in runner
    assert "c0_eval_transport" not in runner


def test_formal_gradient_audit_checks_all_1800_rows_and_scheduler_zeros() -> None:
    result = _audit_formal_gradient_audit(
        _formal_gradient_fixture(), label="fixture gradient audit"
    )
    assert result == {
        "rows": 1800,
        "positive_action_signal_steps": 1798,
        "zero_action_signal_steps": 2,
    }


def test_formal_gradient_audit_rejects_row_reordering() -> None:
    gradient = _formal_gradient_fixture()
    gradient["steps"][99]["step"] = 99
    with pytest.raises(FormalC1C3AuditError, match="step 100 order changed"):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")


def test_formal_gradient_audit_rejects_declared_count_drift() -> None:
    gradient = _formal_gradient_fixture()
    gradient["positive_action_signal_steps"] -= 1
    gradient["zero_action_signal_steps"] += 1
    with pytest.raises(FormalC1C3AuditError, match="observed positive-row count"):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")


def test_formal_gradient_audit_rejects_inconsistent_zero_weight_flag() -> None:
    gradient = _formal_gradient_fixture()
    gradient["steps"][0]["zero_weight_action_step"] = False
    with pytest.raises(FormalC1C3AuditError, match="zero-weight flag contradicts"):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("probe", "zero-weight probe Head is nonzero"),
        ("adapter", "zero-weight combined adapter is nonzero"),
        ("content", "zero-weight official Zc is nonzero"),
    ),
)
def test_formal_gradient_audit_rejects_zero_weight_action_path_gradient(
    field: str, message: str
) -> None:
    gradient = _formal_gradient_fixture()
    zero_row = gradient["steps"][0]
    if field == "probe":
        zero_row["action_only_probe"]["head_grad_norm"] = 0.1
    elif field == "adapter":
        zero_row["combined"]["adapter"]["gradient_norm"] = 0.1
    else:
        zero_row["action_only_official_content_token_grad_norm"] = 0.1
    with pytest.raises(FormalC1C3AuditError, match=message):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("zero_action_signal_reason", "none", "reason is not scheduler_zero_weight"),
        ("loss_action", 0.1, "zero-weight weighted action loss is nonzero"),
        ("action_weight_min", 0.1, "zero-weight action weight minimum is nonzero"),
        ("action_weight_max", 0.1, "zero-weight action weight maximum is nonzero"),
        (
            "action_effective_weight_sum",
            0.1,
            "zero-weight effective action weight sum is nonzero",
        ),
    ),
)
def test_formal_gradient_audit_rejects_invalid_zero_weight_provenance(
    field: str, value: str | float, message: str
) -> None:
    gradient = _formal_gradient_fixture()
    gradient["steps"][0][field] = value
    with pytest.raises(FormalC1C3AuditError, match=message):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("zero_action_signal_reason", "scheduler_zero_weight", "reason is not none"),
        ("action_weight_max", 0.0, "action weight maximum is not positive"),
        (
            "action_effective_weight_sum",
            0.0,
            "effective action weight sum is not positive",
        ),
        ("loss_action", -0.1, "action loss is negative"),
    ),
)
def test_formal_gradient_audit_rejects_invalid_positive_weight_provenance(
    field: str, value: str | float, message: str
) -> None:
    gradient = _formal_gradient_fixture()
    gradient["steps"][1][field] = value
    with pytest.raises(FormalC1C3AuditError, match=message):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")


def test_formal_gradient_audit_requires_first_positive_row_to_open_gate() -> None:
    gradient = _formal_gradient_fixture()
    first_positive = gradient["steps"][1]
    first_positive["action_only_probe"]["gate_grad_norm"] = 0.0
    with pytest.raises(FormalC1C3AuditError, match="positive-weight gate probe is zero"):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("head", "post-gate positive-weight combined Head is zero"),
        ("gca", "post-gate positive-weight combined GCA attention is zero"),
        ("zc", "post-gate positive-weight official Zc is zero"),
        ("probe_head", "post-gate positive-weight probe Head is zero"),
        ("probe_gca", "post-gate positive-weight probe GCA attention is zero"),
    ),
)
def test_formal_gradient_audit_requires_complete_path_after_gate_opening(
    field: str, message: str
) -> None:
    gradient = _formal_gradient_fixture()
    second_positive = gradient["steps"][2]
    if field == "head":
        second_positive["combined"]["content_head"]["gradient_norm"] = 0.0
    elif field == "gca":
        second_positive["combined"][
            "adapter_attention_action_only_by_construction"
        ]["gradient_norm"] = 0.0
    elif field == "zc":
        second_positive["action_only_official_content_token_grad_norm"] = 0.0
    elif field == "probe_head":
        second_positive["action_only_probe"]["head_grad_norm"] = 0.0
    else:
        second_positive["action_only_probe"]["adapter_attention_grad_norm"] = 0.0
    with pytest.raises(FormalC1C3AuditError, match=message):
        _audit_formal_gradient_audit(gradient, label="fixture gradient audit")
