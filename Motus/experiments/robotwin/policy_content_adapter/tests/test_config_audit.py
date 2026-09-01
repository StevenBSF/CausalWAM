from __future__ import annotations

from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.config_audit import (
    ConfigAuditError,
    load_yaml,
    validate_m1_m3_pair,
    validate_run_config,
)


CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def test_prepared_smoke_pair_is_fair_but_not_runnable() -> None:
    m1 = load_yaml(CONFIG_ROOT / "m1_m_p1_smoke.yaml")
    m3 = load_yaml(CONFIG_ROOT / "m3_m_p1_smoke.yaml")
    result = validate_m1_m3_pair(m1, m3)
    assert result["status"] == "PASS"
    assert result["training_seed"] == 1
    with pytest.raises(ConfigAuditError, match="execution-ready"):
        validate_m1_m3_pair(m1, m3, require_runnable=True)


def test_pair_rejects_hidden_training_difference() -> None:
    m1 = load_yaml(CONFIG_ROOT / "m1_m_p1_smoke.yaml")
    m3 = load_yaml(CONFIG_ROOT / "m3_m_p1_smoke.yaml")
    m3["training"]["max_steps"] = 4
    with pytest.raises(ConfigAuditError, match="differ outside"):
        validate_m1_m3_pair(m1, m3)


def test_m1_cannot_enable_contrastive_gradient() -> None:
    m1 = load_yaml(CONFIG_ROOT / "m1_m_p1_smoke.yaml")
    m1["objective"]["lambda_contrastive"] = 0.1
    with pytest.raises(ValueError, match="M1"):
        validate_run_config(m1)


def test_paired_batch_requires_a_same_task_negative() -> None:
    m1 = load_yaml(CONFIG_ROOT / "m1_m_p1_smoke.yaml")
    m1["training"]["paired_groups_per_device"] = 1
    with pytest.raises(ConfigAuditError, match="same-task negative"):
        validate_run_config(m1)
