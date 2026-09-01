from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.robotwin.policy_content_adapter import formal_episode_protocol as formal


class _UnStableError(Exception):
    pass


def _fake_descriptions(task: str, episodes: list[dict], limit: int) -> list[dict]:
    del task, episodes
    options = ["instruction alpha", "instruction beta", "instruction gamma"]
    random.shuffle(options)
    return [{"seen": [], "unseen": options[:limit]}]


def _module() -> SimpleNamespace:
    return SimpleNamespace(
        UnStableError=_UnStableError,
        generate_episode_descriptions=_fake_descriptions,
    )


class _Task:
    def __init__(
        self,
        *,
        unstable: set[int] = frozenset(),
        plan_fail: set[int] = frozenset(),
        check_fail: set[int] = frozenset(),
        generic_fail: set[int] = frozenset(),
    ) -> None:
        self.unstable = set(unstable)
        self.plan_fail = set(plan_fail)
        self.check_fail = set(check_fail)
        self.generic_fail = set(generic_fail)
        self.seed = -1
        self.plan_success = True
        self.closed: list[int] = []

    def setup_demo(self, **kwargs) -> None:
        self.seed = int(kwargs["seed"])
        if self.seed in self.unstable:
            raise _UnStableError("unstable")
        if self.seed in self.generic_fail:
            raise RuntimeError("renderer broke")

    def play_once(self) -> dict:
        self.plan_success = self.seed not in self.plan_fail
        return {"info": {"{A}": f"object-{self.seed}"}}

    def check_success(self) -> bool:
        return self.seed not in self.check_fail

    def close_env(self, **kwargs) -> None:
        del kwargs
        self.closed.append(self.seed)


def _candidate(members: list[int]) -> dict:
    return {
        "members": members,
        "candidate_start_seed": members[0],
        "simulator_seed_bank_id": "robotwin-seed-bank-v3:" + "a" * 64,
    }


def _runtime(tmp_path: Path) -> dict:
    vulkan = tmp_path / "nvidia_icd.json"
    egl = tmp_path / "10_nvidia.json"
    vulkan.write_text("{}\n", encoding="utf-8")
    egl.write_text("{}\n", encoding="utf-8")
    return {
        "status": "PASS",
        "physical_gpu_index": 2,
        "pci_bus_id": "00000000:0A:0B.0",
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


def test_instruction_choice_is_deterministic_restores_rng_and_proves_membership() -> None:
    module = _module()
    random.seed(98123)
    before = random.getstate()
    first, proof = formal._deterministic_instruction_choice(
        module,
        candidate_seed_bank_id="robotwin-seed-bank-v3:" + "a" * 64,
        task="place_a2b_left",
        task_config="demo_clean",
        simulator_seed=4800000,
        episode_info={"{A}": "object"},
        instruction_type="unseen",
        test_num=100,
    )
    assert random.getstate() == before
    random.seed(7)
    second, second_proof = formal._deterministic_instruction_choice(
        module,
        candidate_seed_bank_id="robotwin-seed-bank-v3:" + "a" * 64,
        task="place_a2b_left",
        task_config="demo_clean",
        simulator_seed=4800000,
        episode_info={"{A}": "object"},
        instruction_type="unseen",
        test_num=100,
    )
    assert (first, proof) == (second, second_proof)
    episode = {
        "simulator_seed": 4800000,
        "instruction": first,
        **proof,
    }
    formal._validate_instruction_proof(
        episode,
        candidate_seed_bank_id="robotwin-seed-bank-v3:" + "a" * 64,
        task="place_a2b_left",
        task_config="demo_clean",
        instruction_type="unseen",
        episode_index=0,
    )
    tampered = {**episode, "instruction": "not a legal generated description"}
    with pytest.raises(formal.FormalEpisodeProtocolError, match="deterministic legal"):
        formal._validate_instruction_proof(
            tampered,
            candidate_seed_bank_id="robotwin-seed-bank-v3:" + "a" * 64,
            task="place_a2b_left",
            task_config="demo_clean",
            instruction_type="unseen",
            episode_index=0,
        )


def test_expert_scan_is_exact_candidate_prefix_and_first_100() -> None:
    task = _Task(unstable={10}, plan_fail={11}, check_fail={12})
    members = list(range(10, 113))
    episodes, attempts = formal.scan_expert_candidates(
        _module(),
        task_name="open_microwave",
        task_config="demo_randomized",
        task_env=task,
        args={"render_freq": 9},
        candidate_bank=_candidate(members),
        instruction_type="unseen",
        test_num=100,
    )
    assert [row["simulator_seed"] for row in episodes] == list(range(13, 113))
    assert [row["simulator_seed"] for row in attempts] == members
    assert [row["candidate_index"] for row in attempts] == list(range(103))
    assert sum(row["accepted"] for row in attempts) == 100
    assert attempts[-1]["accepted"] is True
    assert [row["rejection"]["type"] for row in attempts[:3]] == [
        "UnStableError",
        "plan_success_false",
        "check_success_false",
    ]


def test_expert_scan_generic_error_and_short_pool_fail_closed() -> None:
    with pytest.raises(formal.FormalEpisodeProtocolError, match="non-UnStableError"):
        formal.scan_expert_candidates(
            _module(),
            task_name="move_stapler_pad",
            task_config="demo_clean",
            task_env=_Task(generic_fail={3}),
            args={"render_freq": 0},
            candidate_bank=_candidate(list(range(3, 110))),
            instruction_type="unseen",
            test_num=100,
        )
    with pytest.raises(formal.FormalEpisodeProtocolError, match="exhausted before 100"):
        formal.scan_expert_candidates(
            _module(),
            task_name="move_stapler_pad",
            task_config="demo_clean",
            task_env=_Task(),
            args={"render_freq": 0},
            candidate_bank=_candidate(list(range(99))),
            instruction_type="unseen",
            test_num=100,
        )


def test_attempt_audit_rejects_gap_reorder_and_accepted_mismatch() -> None:
    task = _Task(unstable={0})
    members = list(range(101))
    episodes, attempts = formal.scan_expert_candidates(
        _module(),
        task_name="place_a2b_left",
        task_config="demo_clean",
        task_env=task,
        args={"render_freq": 0},
        candidate_bank=_candidate(members),
        instruction_type="unseen",
        test_num=100,
    )
    seeds = [row["simulator_seed"] for row in episodes]
    formal._validate_attempt_prefix(
        attempts,
        candidate_members=members,
        accepted_episode_seeds=seeds,
    )
    reordered = copy.deepcopy(attempts)
    reordered[3], reordered[4] = reordered[4], reordered[3]
    with pytest.raises(formal.FormalEpisodeProtocolError, match="candidate index"):
        formal._validate_attempt_prefix(
            reordered,
            candidate_members=members,
            accepted_episode_seeds=seeds,
        )
    with pytest.raises(formal.FormalEpisodeProtocolError, match="one-to-one"):
        formal._validate_attempt_prefix(
            attempts,
            candidate_members=members,
            accepted_episode_seeds=list(reversed(seeds)),
        )


def test_runtime_binding_requires_exact_pci_and_sapien(tmp_path: Path) -> None:
    normalized = formal.validate_runtime_binding_payload(
        _runtime(tmp_path), verify_files=True
    )
    assert normalized["physical_gpu_index"] == 2
    assert normalized["pci_bus_id"] == "0000:0a:0b.0"
    assert normalized["sapien"]["logical_cuda_id"] == 0
    assert normalized["vulkan_icd"]["sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    bad = _runtime(tmp_path)
    bad["sapien"] = {**bad["sapien"], "pci_bus_id": "0000:0a:0c.0"}
    with pytest.raises(formal.FormalEpisodeProtocolError, match="different PCI"):
        formal.validate_runtime_binding_payload(bad, verify_files=True)


def test_audit_inputs_cross_checks_candidate_bound_stock_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "RoboTwin"
    evaluator = root / "script/eval_policy.py"
    generator = root / "description/utils/generate_episode_instructions.py"
    evaluator.parent.mkdir(parents=True)
    generator.parent.mkdir(parents=True)
    evaluator.write_text("stock evaluator\n", encoding="utf-8")
    generator.write_text("stock generator\n", encoding="utf-8")
    for name in (
        "demo_clean.yml",
        "demo_randomized.yml",
        "_camera_config.yml",
        "_embodiment_config.yml",
        "_eval_step_limit.yml",
    ):
        path = root / "task_config" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    candidate_file = tmp_path / "candidate.json"
    lock_file = tmp_path / "lock.json"
    candidate_file.write_text("{}\n", encoding="utf-8")
    lock_file.write_text("{}\n", encoding="utf-8")
    evaluator_identity = formal.stable_file_identity(evaluator)
    candidate = {
        "simulator_seed_bank_id": "robotwin-seed-bank-v3:" + "a" * 64,
        "members_sha256": "b" * 64,
        "members": list(range(100)),
        "evaluator_source_size_bytes": evaluator_identity["size_bytes"],
        "evaluator_source_sha256": evaluator_identity["sha256"],
    }
    monkeypatch.setattr(
        formal,
        "_parent_artifacts",
        lambda candidate_bank_path, formal_lock_path: (
            candidate,
            {"selected_policy_regime": "p_v1"},
            formal.stable_file_identity(candidate_file),
            formal.stable_file_identity(lock_file),
        ),
    )
    result = formal.audit_selector_inputs(
        robotwin_root=root,
        candidate_bank_path=candidate_file,
        formal_lock_path=lock_file,
    )
    assert result["status"] == "PASS"
    candidate["evaluator_source_sha256"] = "0" * 64
    with pytest.raises(formal.FormalEpisodeProtocolError, match="evaluator source SHA"):
        formal.audit_selector_inputs(
            robotwin_root=root,
            candidate_bank_path=candidate_file,
            formal_lock_path=lock_file,
        )


def test_gpu_cell_cli_is_create_only_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "exists.json"
    output.write_text("keep\n", encoding="utf-8")
    called = False

    def forbidden(**kwargs):
        nonlocal called
        called = True
        raise AssertionError(kwargs)

    monkeypatch.setattr(formal, "audit_selector_inputs", forbidden)
    with pytest.raises(formal.FormalEpisodeProtocolError, match="overwrite realization cell"):
        formal.realize_cell_on_gpu(
            robotwin_root=tmp_path,
            candidate_bank_path=tmp_path / "candidate.json",
            formal_lock_path=tmp_path / "lock.json",
            task="place_a2b_left",
            task_config="demo_clean",
            gpu_id=0,
            output=output,
        )
    assert called is False
    assert output.read_text(encoding="utf-8") == "keep\n"


def test_cell_command_plan_is_six_gpu_cells_plus_create_only_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    lock = tmp_path / "lock.json"
    candidate.write_text("{}\n", encoding="utf-8")
    lock.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        formal,
        "audit_selector_inputs",
        lambda **kwargs: {
            "status": "PASS",
            "candidate_seed_bank": formal.stable_file_identity(candidate),
            "candidate_seed_bank_id": "robotwin-seed-bank-v3:" + "a" * 64,
            "formal_protocol_lock": formal.stable_file_identity(lock),
        },
    )
    plan = formal.build_realization_cell_commands(
        robotwin_root=tmp_path,
        candidate_bank_path=candidate,
        formal_lock_path=lock,
        output_root=tmp_path / "realization",
        gpu_ids="0,1,2,4,5,6",
        python_executable="/usr/bin/python3",
    )
    assert plan["gpu_started"] is False
    assert len(plan["jobs"]) == 6
    assert [job["physical_gpu_index"] for job in plan["jobs"]] == [0, 1, 2, 4, 5, 6]
    assert [(job["task"], job["task_config"]) for job in plan["jobs"]] == [
        (task, config) for task in formal.TASKS for config in formal.TASK_CONFIGS
    ]
    assert all("realize-cell" in job["command"] for job in plan["jobs"])
    assert plan["merge_command"].count("--cell") == 6
    assert plan["merge_output"].endswith("/realization_bank.json")
    with pytest.raises(formal.FormalEpisodeProtocolError, match="six GPU"):
        formal.build_realization_cell_commands(
            robotwin_root=tmp_path,
            candidate_bank_path=candidate,
            formal_lock_path=lock,
            output_root=tmp_path / "other",
            gpu_ids="0,1",
        )


def test_six_cell_merge_binds_sources_configs_parents_and_sequence_shas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "RoboTwin"
    stock = root / "script/eval_policy.py"
    selector = tmp_path / "formal_episode_protocol.py"
    stock.parent.mkdir(parents=True)
    stock.write_text("stock\n", encoding="utf-8")
    selector.write_text("selector\n", encoding="utf-8")
    candidate_file = tmp_path / "candidate.json"
    lock_file = tmp_path / "lock.json"
    candidate_file.write_text("candidate\n", encoding="utf-8")
    lock_file.write_text("lock\n", encoding="utf-8")
    candidate_identity = formal.stable_file_identity(candidate_file)
    lock_identity = formal.stable_file_identity(lock_file)
    source_artifacts = {
        "stock_evaluator": formal.stable_file_identity(stock),
        "formal_episode_protocol": formal.stable_file_identity(selector),
    }
    config_files: dict[str, Path] = {}
    for config in formal.TASK_CONFIGS:
        path = root / "task_config" / f"{config}.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config + "\n", encoding="utf-8")
        config_files[config] = path
    config_artifacts = {
        config: {"domain_task_config": formal.stable_file_identity(path)}
        for config, path in config_files.items()
    }
    bank_id = "robotwin-seed-bank-v3:" + "a" * 64
    candidate = {
        "simulator_seed_bank_id": bank_id,
        "members_sha256": "b" * 64,
        "members": list(range(100)),
        "candidate_start_seed": 0,
        "evaluator_source_size_bytes": source_artifacts["stock_evaluator"][
            "size_bytes"
        ],
        "evaluator_source_sha256": source_artifacts["stock_evaluator"]["sha256"],
    }
    lock = {"selected_policy_regime": "p_v1"}
    monkeypatch.setattr(
        formal,
        "_parent_artifacts",
        lambda candidate_bank_path, formal_lock_path: (
            candidate,
            lock,
            candidate_identity,
            lock_identity,
        ),
    )
    monkeypatch.setattr(formal, "_source_artifacts", lambda robotwin_root: source_artifacts)
    monkeypatch.setattr(
        formal,
        "task_config_artifacts",
        lambda robotwin_root, task_config: config_artifacts[task_config],
    )
    runtime = _runtime(tmp_path)
    cells: list[Path] = []
    for task_name in formal.TASKS:
        for task_config in formal.TASK_CONFIGS:
            episodes, attempts = formal.scan_expert_candidates(
                _module(),
                task_name=task_name,
                task_config=task_config,
                task_env=_Task(),
                args={"render_freq": 0},
                candidate_bank=candidate,
                instruction_type="unseen",
                test_num=100,
            )
            payload = formal.build_realization_cell_manifest(
                robotwin_root=root,
                task=task_name,
                task_config=task_config,
                instruction_type="unseen",
                candidate_bank_path=candidate_file,
                formal_lock_path=lock_file,
                episodes=episodes,
                attempts=attempts,
                runtime_binding=runtime,
            )
            path = tmp_path / "cells" / task_name / f"{task_config}.json"
            formal._exclusive_json(path, payload)
            cells.append(path)
    merged = formal.finalize_realization_bank(
        candidate_bank_path=candidate_file,
        formal_lock_path=lock_file,
        cell_paths=cells,
    )
    assert merged["status"] == "PASS"
    assert merged["candidate_seed_bank"] == candidate_identity
    assert merged["formal_protocol_lock"] == lock_identity
    assert merged["source_artifacts"] == source_artifacts
    assert merged["task_config_artifacts"] == config_artifacts
    assert len(merged["cells"]) == 6
    assert merged["realization_bank_id"].startswith(formal.BANK_ID_PREFIX)
    assert all(row["ordered_seed_instruction_sha256"] for row in merged["cells"])
    with pytest.raises(formal.FormalEpisodeProtocolError, match="duplicate realization cell"):
        formal.finalize_realization_bank(
            candidate_bank_path=candidate_file,
            formal_lock_path=lock_file,
            cell_paths=[cells[0], cells[0], *cells[2:]],
        )


def test_one_click_realization_shell_is_cpu_safe_by_default_and_expert_only() -> None:
    script = (
        Path(formal.__file__).resolve().parent
        / "run_release_formal_episode_realization.sh"
    ).read_text(encoding="utf-8")
    assert 'PHASE="${PHASE:-audit}"' in script
    assert "CONFIRM_EXPERT_REALIZATION=YES" in script
    assert "realize-cell" in script
    assert "finalize" in script
    assert "eval_robotwin_single" not in script
    assert "policy checkpoint" in script.lower()
    assert "for index in 0 1 2 3 4 5" in script
