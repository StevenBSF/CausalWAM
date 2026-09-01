from experiments.robotwin.policy_content_adapter import eval_robotwin_pv2_full5ep
from experiments.robotwin.policy_content_adapter import pv2_full5ep_eval100


def test_full5ep_eval_constants_lock_development_bank() -> None:
    assert pv2_full5ep_eval100.SIMULATOR_SEED == 53
    assert pv2_full5ep_eval100.EPISODES_PER_CELL == 100
    assert pv2_full5ep_eval100.CHECKPOINT_STEP == 18_215
    assert pv2_full5ep_eval100.TASKS == (
        "place_a2b_left",
        "open_microwave",
        "move_stapler_pad",
    )
    assert pv2_full5ep_eval100.DOMAINS == ("clean", "official_random")


def test_full5ep_runtime_contract_accepts_exact_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_robotwin_pv2_full5ep,
        "matching_checkpoint_row",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    amendment = {
        "profile": pv2_full5ep_eval100.PROFILE,
        "checkpoint_contract": {
            "mechanism_protocol_manifest_sha256": "a" * 64,
            "simulator_seed_bank_id": "bank",
            "simulator_seed_bank_manifest_sha256": "b" * 64,
        },
    }
    contract = {
        "control": "c3_ours",
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "training_seed": 1,
        "checkpoint_step": 18_215,
        "formal_evaluation_eligible": False,
        "mechanism_protocol_manifest_sha256": "a" * 64,
        "simulator_seed_bank_id": "bank",
        "simulator_seed_bank_manifest_sha256": "b" * 64,
    }
    result = eval_robotwin_pv2_full5ep._validate_runtime_contract(
        amendment,
        checkpoint_contract=contract,
        checkpoint_path="checkpoint.pt",
        requested_tasks=pv2_full5ep_eval100.TASKS,
        requested_domains=pv2_full5ep_eval100.DOMAINS,
        simulator_seed=53,
        episodes_per_task=100,
    )
    assert result == {"status": "PASS"}


def test_full5ep_runtime_contract_rejects_pilot_step() -> None:
    amendment = {
        "profile": pv2_full5ep_eval100.PROFILE,
        "checkpoint_contract": {
            "mechanism_protocol_manifest_sha256": "a" * 64,
            "simulator_seed_bank_id": "bank",
            "simulator_seed_bank_manifest_sha256": "b" * 64,
        },
    }
    contract = {
        "control": "c3_ours",
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "training_seed": 1,
        "checkpoint_step": 1800,
        "formal_evaluation_eligible": False,
        "mechanism_protocol_manifest_sha256": "a" * 64,
        "simulator_seed_bank_id": "bank",
        "simulator_seed_bank_manifest_sha256": "b" * 64,
    }
    try:
        eval_robotwin_pv2_full5ep._validate_runtime_contract(
            amendment,
            checkpoint_contract=contract,
            checkpoint_path="checkpoint.pt",
            requested_tasks=pv2_full5ep_eval100.TASKS,
            requested_domains=pv2_full5ep_eval100.DOMAINS,
            simulator_seed=53,
            episodes_per_task=100,
        )
    except ValueError as exc:
        assert "checkpoint_step" in str(exc)
    else:
        raise AssertionError("pilot checkpoint step was accepted")


def test_full5ep_launcher_composes_absolute_project_config() -> None:
    cfg = eval_robotwin_pv2_full5ep._compose_eval_config(
        [
            "ckpt=/tmp/checkpoint.pt",
            "gpu_id=0",
            "seed=53",
            "EVALUATION.task_name=place_a2b_left",
            "EVALUATION.task_config=demo_clean",
            "EVALUATION.eval_num_episodes=100",
        ]
    )
    assert str(cfg.ckpt) == "/tmp/checkpoint.pt"
    assert int(cfg.gpu_id) == 0
    assert int(cfg.seed) == 53
    assert cfg.EVALUATION.task_name == "place_a2b_left"


def test_full5ep_runtime_contract_accepts_one_exact_cell(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_robotwin_pv2_full5ep,
        "matching_checkpoint_row",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    amendment = {
        "profile": pv2_full5ep_eval100.PROFILE,
        "checkpoint_contract": {
            "mechanism_protocol_manifest_sha256": "a" * 64,
            "simulator_seed_bank_id": "bank",
            "simulator_seed_bank_manifest_sha256": "b" * 64,
        },
    }
    contract = {
        "control": "c3_ours",
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "training_seed": 1,
        "checkpoint_step": 18_215,
        "formal_evaluation_eligible": False,
        "mechanism_protocol_manifest_sha256": "a" * 64,
        "simulator_seed_bank_id": "bank",
        "simulator_seed_bank_manifest_sha256": "b" * 64,
    }
    result = eval_robotwin_pv2_full5ep._validate_runtime_contract(
        amendment,
        checkpoint_contract=contract,
        checkpoint_path="checkpoint.pt",
        requested_tasks=("open_microwave",),
        requested_domains=("official_random",),
        simulator_seed=53,
        episodes_per_task=100,
    )
    assert result == {"status": "PASS"}
