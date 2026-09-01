from __future__ import annotations

import pytest

from experiments.robotwin.policy_content_adapter.evaluation import (
    CELL_SCHEMA,
    EvaluationError,
    aggregate_cells,
)
from experiments.robotwin.policy_content_adapter.protocol import TASKS


def _cells():
    result = []
    for control in ("m1_architecture_action_control", "m3_ours"):
        for seed in (1, 2, 3):
            for task_index, task in enumerate(TASKS):
                for domain in ("clean", "official_random"):
                    successes = 40 + task_index + (5 if control == "m3_ours" else 0)
                    result.append(
                        {
                            "schema": CELL_SCHEMA,
                            "schema_version": 1,
                            "status": "PASS",
                            "control": control,
                            "training_seed": seed,
                            "task": task,
                            "domain": domain,
                            "episode_count": 100,
                            "success_count": successes,
                            "success_rate": successes / 100,
                            "checkpoint_sha256": str(seed) * 64,
                            "evaluation_settings_sha256": "a" * 64,
                            "episode_pairing": "shared_start_seed_not_exact_pairing",
                        }
                    )
    return result


def test_complete_100_episode_matrix_and_delta() -> None:
    summary = aggregate_cells(_cells(), formal_training_seeds=(1, 2, 3))
    assert summary["status"] == "PASS" and summary["cell_count"] == 36
    assert summary["m3_minus_m1_macro"]["clean"]["mean"] == pytest.approx(0.05)
    assert summary["m3_minus_m1_macro"]["official_random"]["mean"] == pytest.approx(0.05)


def test_missing_or_non100_cell_fails_closed() -> None:
    cells = _cells()
    with pytest.raises(EvaluationError, match="incomplete"):
        aggregate_cells(cells[:-1], formal_training_seeds=(1, 2, 3))
    cells[0]["episode_count"] = 99
    with pytest.raises(EvaluationError, match="100"):
        aggregate_cells(cells, formal_training_seeds=(1, 2, 3))

