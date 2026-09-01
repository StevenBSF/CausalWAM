from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.robotwin.policy_content_adapter import formal_episode_protocol as formal


START_SEED = 4_800_000


def _runtime(tmp_path: Path) -> dict:
    vulkan = tmp_path / "nvidia_icd.json"
    egl = tmp_path / "10_nvidia.json"
    vulkan.write_text("{}\n", encoding="utf-8")
    egl.write_text("{}\n", encoding="utf-8")
    return {
        "status": "PASS",
        "physical_gpu_index": 2,
        "pci_bus_id": "0000:0a:0b.0",
        "render_device_alias": "pci:0000:0a:0b.0",
        "gpu_name": "test gpu",
        "driver_version": "test driver",
        "vulkan_icd": str(vulkan),
        "egl_vendor": str(egl),
        "sapien": {
            "version": "3.0",
            "device_name": "test gpu",
            "logical_cuda_id": 0,
            "pci_bus_id": "0000:0a:0b.0",
            "can_render": True,
        },
    }


def _episodes() -> list[dict]:
    rows = []
    for index in range(100):
        instruction = f"instruction {index}"
        rows.append(
            {
                "episode_index": index,
                "simulator_seed": START_SEED + index * 2,
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
            }
        )
    return rows


class _Task:
    def __init__(
        self,
        *,
        setup_failure_seed: int | None = None,
        close_failure_index: int | None = None,
    ) -> None:
        self.setup_failure_seed = setup_failure_seed
        self.close_failure_index = close_failure_index
        self.setup_calls: list[int] = []
        self.close_calls = 0
        self.play_once_calls = 0
        self.eval_video_path = None
        self.render_freq = 0
        self.viewer = None
        self.test_num = 0
        self.suc = 0
        self.step_lim = 1
        self.take_action_cnt = 0
        self.eval_success = False
        self.instructions: list[str] = []

    def setup_demo(self, **kwargs) -> None:
        seed = int(kwargs["seed"])
        self.setup_calls.append(seed)
        if seed == self.setup_failure_seed:
            raise RuntimeError("exact setup failed")
        self.take_action_cnt = 0
        self.eval_success = False
        self.eval_video_path = None

    def play_once(self) -> None:
        self.play_once_calls += 1
        raise AssertionError("formal replay must not execute expert play_once")

    def set_instruction(self, *, instruction: str) -> None:
        self.instructions.append(instruction)

    def get_obs(self) -> dict:
        return {"observation": True}

    def close_env(self, **kwargs) -> None:
        del kwargs
        current = self.close_calls
        self.close_calls += 1
        if current == self.close_failure_index:
            raise RuntimeError("close failed")


def _module() -> SimpleNamespace:
    def evaluate(task_env, model, observation) -> None:
        del model, observation
        task_env.take_action_cnt += 1
        task_env.eval_success = True

    def reset(model) -> None:
        del model

    return SimpleNamespace(
        eval_function_decorator=lambda policy, name: (
            evaluate if name == "eval" else reset
        )
    )


def _install_exact_bank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, dict, dict]:
    bank_path = tmp_path / "realization_bank.json"
    bank_path.write_text("{}\n", encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps({"candidate_start_seed": START_SEED}),
        encoding="utf-8",
    )
    cell = {
        "cell_id": "cell-id",
        "candidate_seed_bank": {"path": str(candidate_path.resolve())},
        "ordered_seed_instruction_sha256": formal.canonical_sha256(
            formal.ordered_seed_instruction_payload(_episodes())
        ),
        "episodes": _episodes(),
    }
    bank = {"realization_bank_id": "realization-bank-id"}
    monkeypatch.setattr(
        formal,
        "select_realization_cell",
        lambda path, task, task_config: (bank, cell, bank_path.resolve()),
    )
    monkeypatch.setattr(
        formal,
        "validate_seed_bank_descriptor",
        lambda value, expected_purpose: value,
    )
    return bank_path, bank, cell


def _replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: _Task,
) -> tuple[Path, object]:
    bank_path, _, _ = _install_exact_bank(monkeypatch, tmp_path)
    trace_path = tmp_path / "trace.json"
    replay = formal.make_replay_eval_policy(
        _module(),
        {
            "formal_episode_realization_bank": str(bank_path),
            "formal_episode_trace_output": str(trace_path),
        },
        runtime_binding=_runtime(tmp_path),
    )
    args = {
        "task_config": "demo_clean",
        "policy_name": "policy_content_adapter.rollout_policy",
        "clear_cache_freq": 5,
        "ckpt_setting": "/checkpoint.pt",
    }
    return trace_path, lambda start=START_SEED: replay(
        "place_a2b_left",
        task,
        args,
        object(),
        start,
        test_num=100,
        video_size=None,
        instruction_type="unseen",
        skip_get_obs_within_replan=True,
    )


def test_replay_executes_exact_100_once_without_expert_or_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _Task()
    trace_path, replay = _replay(monkeypatch, tmp_path, task)
    _, successes = replay()
    assert successes == 100
    assert task.setup_calls == [row["simulator_seed"] for row in _episodes()]
    assert task.instructions == [row["instruction"] for row in _episodes()]
    assert task.play_once_calls == 0
    assert task.close_calls == 100
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["successes"] == 100
    assert trace["no_seed_replacement"] is True
    assert len(trace["episodes"]) == 100
    assert trace["pinned_runtime_binding"]["sapien"]["can_render"] is True


def test_replay_setup_failure_aborts_before_next_exact_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failure_seed = _episodes()[3]["simulator_seed"]
    task = _Task(setup_failure_seed=failure_seed)
    trace_path, replay = _replay(monkeypatch, tmp_path, task)
    with pytest.raises(formal.FormalEpisodeProtocolError, match="without replacement"):
        replay()
    assert task.setup_calls == [row["simulator_seed"] for row in _episodes()[:4]]
    assert task.play_once_calls == 0
    assert not trace_path.exists()


def test_replay_rejects_wrong_stock_start_seed_before_any_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _Task()
    trace_path, replay = _replay(monkeypatch, tmp_path, task)
    with pytest.raises(formal.FormalEpisodeProtocolError, match="start seed differs"):
        replay(START_SEED + 1)
    assert task.setup_calls == []
    assert not trace_path.exists()


def test_replay_normal_close_failure_aborts_and_does_not_write_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _Task(close_failure_index=2)
    trace_path, replay = _replay(monkeypatch, tmp_path, task)
    with pytest.raises(RuntimeError, match="close failed"):
        replay()
    assert task.setup_calls == [row["simulator_seed"] for row in _episodes()[:3]]
    assert not trace_path.exists()


def test_replay_trace_validator_rejects_success_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _Task()
    trace_path, replay = _replay(monkeypatch, tmp_path, task)
    replay()
    # select_realization_cell is already patched by _replay.  The live bank
    # and runtime artifact identities remain available, so the trace passes.
    formal.validate_replay_trace(
        trace_path,
        realization_bank_path=tmp_path / "realization_bank.json",
        task="place_a2b_left",
        task_config="demo_clean",
    )
    tampered = json.loads(trace_path.read_text(encoding="utf-8"))
    tampered["episodes"][0]["success"] = False
    tampered_path = tmp_path / "tampered_trace.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(formal.FormalEpisodeProtocolError):
        formal.validate_replay_trace(
            tampered_path,
            realization_bank_path=tmp_path / "realization_bank.json",
            task="place_a2b_left",
            task_config="demo_clean",
        )


def test_realization_and_rollout_shells_are_cpu_safe_by_default() -> None:
    root = Path(formal.__file__).resolve().parent
    realization = (root / "run_release_formal_episode_realization.sh").read_text()
    rollout = (root / "run_release_formal_rollout.sh").read_text()
    assert 'PHASE="${PHASE:-audit}"' in realization
    assert "CONFIRM_EXPERT_REALIZATION=YES" in realization
    assert 'PHASE="${PHASE:-prepare}"' in rollout
    assert "BLOCKED: exact policy-independent realization bank is missing" in rollout
    assert "CONFIRM_FORMAL_ROLLOUT=YES" in rollout
