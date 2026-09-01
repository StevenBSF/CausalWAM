from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import release_c3_overlap as overlap


def test_gpu_reports_require_exact_planned_set_and_memory_floor(tmp_path: Path) -> None:
    paths = []
    for gpu in overlap.TARGET_GPU_IDS:
        path = tmp_path / f"gpu_{gpu}.json"
        path.write_text(
            json.dumps(
                {
                    "physical_gpu_index": gpu,
                    "memory_free_mib_at_preflight": overlap.MIN_FREE_GPU_MIB,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    rows = overlap._validated_gpu_reports(paths)
    assert [row["physical_gpu_index"] for row in rows] == [0, 1, 5, 6]

    paths[0].write_text(
        json.dumps(
            {
                "physical_gpu_index": 0,
                "memory_free_mib_at_preflight": overlap.MIN_FREE_GPU_MIB - 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(overlap.C3OverlapError, match="<60000"):
        overlap._validated_gpu_reports(paths)


def test_create_only_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.json"
    overlap._write_new_json(path, {"status": "PASS"})
    with pytest.raises(overlap.C3OverlapError, match="overwrite"):
        overlap._write_new_json(path, {"status": "PASS"})


def test_shell_has_fail_closed_stop_cleanup_and_exact_cells() -> None:
    shell = Path(overlap.__file__).with_name("run_release_c3_overlap.sh").read_text(
        encoding="utf-8"
    )
    assert 'MAIN_RUNNER_PID=3759159' in shell
    assert 'kill -STOP -- "${MAIN_RUNNER_PID}"' in shell
    assert 'kill -CONT -- "${MAIN_RUNNER_PID}"' in shell
    assert 'setsid timeout --signal=TERM' in shell
    assert 'expected_indices=(6 7 10 11)' in shell
    assert 'expected_gpus=(0 1 5 6)' in shell
    assert shell.index("terminate_and_reap_owned_workers") < shell.index(
        'kill -CONT -- "${MAIN_RUNNER_PID}"'
    )
    assert shell.index("materialize-after-stop") > shell.index(
        'kill -STOP -- "${MAIN_RUNNER_PID}"'
    )
    assert "audit-cell" in shell
    assert "EVALUATION.eval_num_episodes=100" in shell
    assert "EVALUATION.replan_steps=24" in shell
    assert "EVALUATION.num_inference_steps=10" in shell
