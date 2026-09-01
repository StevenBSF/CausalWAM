from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import release_seed23_packed as packed


def test_queue_starts_with_all_four_open_pairs() -> None:
    assert packed.TARGET_CELL_INDICES == tuple(range(12, 36))
    assert packed.PAIR_QUEUE_ORDER[:4] == (2, 3, 8, 9)
    assert [packed.PAIR_INDICES[index] for index in packed.PAIR_QUEUE_ORDER[:4]] == [
        (14, 20),
        (15, 21),
        (26, 32),
        (27, 33),
    ]
    flattened = [cell for pair in packed.PAIR_INDICES for cell in pair]
    assert sorted(flattened) == list(range(12, 36))


def test_gpu_reports_enforce_double_and_single_slot_floors(tmp_path: Path) -> None:
    paths = []
    for gpu in packed.ALL_GPU_IDS:
        path = tmp_path / f"gpu_{gpu}.json"
        path.write_text(
            json.dumps(
                {
                    "physical_gpu_index": gpu,
                    "memory_free_mib_at_preflight": packed.MIN_FREE_GPU_MIB[gpu],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    rows = packed.validate_gpu_reports(paths)
    assert [row["physical_gpu_index"] for row in rows] == list(range(8))
    assert packed.MIN_FREE_GPU_MIB[3] == 30_000
    assert packed.MIN_FREE_GPU_MIB[0] == 60_000

    paths[3].write_text(
        json.dumps(
            {"physical_gpu_index": 3, "memory_free_mib_at_preflight": 29_999}
        ),
        encoding="utf-8",
    )
    with pytest.raises(packed.Seed23PackedError, match="GPU 3"):
        packed.validate_gpu_reports(paths)


def test_shell_has_packed_caps_open_spread_and_liveness_contract() -> None:
    shell = Path(packed.__file__).with_name("run_release_seed23_packed.sh").read_text(
        encoding="utf-8"
    )
    assert 'MAIN_RUNNER_PID=3759159' in shell
    assert '/bin/kill -STOP "${MAIN_RUNNER_PID}"' in shell
    assert '/bin/kill -CONT "${MAIN_RUNNER_PID}"' in shell
    assert shell.index('/bin/kill -STOP "${MAIN_RUNNER_PID}"') < shell.index(
        "materialize-after-stop"
    )
    assert shell.index("terminate_and_reap_owned_workers") < shell.index(
        '/bin/kill -CONT "${MAIN_RUNNER_PID}"'
    )
    assert 'GPU_CAP=([0]=2 [1]=2 [2]=1 [3]=1 [4]=1 [5]=2 [6]=2 [7]=2)' in shell
    assert "for row_index in 0 1 2 3 4 5 6 7" in shell
    assert 'launch_cell "${row_index}" "${row_index}"' in shell
    assert "wait -n -p finished" in shell
    assert "dynamic_free_mib" in shell
    assert "setsid timeout --signal=TERM" in shell
    assert "audit-cell" in shell
    assert "EVALUATION.eval_num_episodes=100" in shell
    assert "EVALUATION.replan_steps=24" in shell
    assert "EVALUATION.num_inference_steps=10" in shell
