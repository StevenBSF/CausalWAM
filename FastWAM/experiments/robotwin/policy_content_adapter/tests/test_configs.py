from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter.config_audit import (
    ConfigAuditError,
    FORMAL_FILENAMES,
    LEGACY_FORMAL_FILENAMES,
    TASKS,
    audit_config_directory,
    find_placeholders,
    load_config,
    validate_c1_c3_pair,
    validate_c2_c3_pair,
    validate_config_structure,
    validate_execution_ready,
    validate_formal_matrix,
    validate_formal_pair,
)
from experiments.robotwin.policy_content_adapter.protocol import (
    POLICY_CAMERA_NAMES,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_VARIANTS,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CURRENT_CONFIGS = (
    "p_v1_smoke.yaml",
    "p_v2_smoke.yaml",
    "p_v1_dev_pilot.yaml",
    "p_v2_dev_pilot.yaml",
    "c0_original.yaml",
    "c1_architecture_only.yaml",
    "c2_naive_aug.yaml",
    "c3_ours.yaml",
    *FORMAL_FILENAMES,
)


def _config(name: str) -> dict:
    return load_config(CONFIG_DIR / name)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("filename", CURRENT_CONFIGS)
def test_every_current_policy_v2_yaml_has_a_complete_structure(filename: str) -> None:
    validate_config_structure(_config(filename))


def test_old_formal_matrix_remains_legacy_and_fail_closed() -> None:
    for filename in LEGACY_FORMAL_FILENAMES:
        config = _config(filename)
        assert config["schema_version"] == 2
        assert config["execution"]["runnable"] is False
        assert config["execution"]["fail_closed"] is True
        with pytest.raises(ConfigAuditError, match="schema_version must be 3"):
            validate_config_structure(config)


@pytest.mark.parametrize("filename", CURRENT_CONFIGS)
def test_templates_cannot_run_before_required_artifacts_exist(filename: str) -> None:
    config = _config(filename)
    assert config["execution"]["runnable"] is False
    assert config["execution"]["fail_closed"] is True
    with pytest.raises(ConfigAuditError, match="unresolved config placeholders"):
        validate_execution_ready(config)


@pytest.mark.parametrize(
    "filename",
    ("p_v1_smoke.yaml", "p_v2_smoke.yaml", "c1_architecture_only.yaml", "c2_naive_aug.yaml", "c3_ours.yaml"),
)
def test_stage2_uses_locked_author_release_and_never_binds_legacy_e2_artifacts(filename: str) -> None:
    config = _config(filename)
    rendered = str(config)
    assert config["official"]["selection_mode"] == "full_550_per_task"
    assert config["official"]["expected_clean_per_task"] == 50
    assert config["official"]["expected_random_per_task"] == 500
    assert config["official"]["expected_total_per_task"] == 550
    assert "e2_train.pt" not in rendered
    assert "e2_best_content_head.pt" not in rendered
    assert config["base_checkpoint"].endswith(
        "/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
    )
    assert config["artifacts"]["base_checkpoint_sha256"] == (
        "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63"
    )
    assert config["base_lineage_manifest"].endswith("author_release_base_manifest.json")
    if filename != "c0_original.yaml":
        assert config["release_paired_binding_manifest"]


def test_all_stage2_templates_bind_the_same_completed_release_paired_audit() -> None:
    names = (
        "p_v1_smoke.yaml",
        "p_v2_smoke.yaml",
        "p_v1_dev_pilot.yaml",
        "p_v2_dev_pilot.yaml",
        "c1_architecture_only.yaml",
        "c2_naive_aug.yaml",
        "c3_ours.yaml",
        *FORMAL_FILENAMES,
    )
    configs = [_config(name) for name in names]
    paths = {config["release_paired_binding_manifest"] for config in configs}
    digests = {
        config["artifacts"]["release_paired_binding_manifest_sha256"]
        for config in configs
    }
    assert len(paths) == len(digests) == 1
    path = Path(paths.pop())
    digest = digests.pop()
    assert path.is_file()
    assert _sha256(path) == digest == (
        "ab2904a01636fdb6fd80798a65580cc58c07451fc30ebcd5a527161c56025835"
    )


def test_p_v1_and_p_v2_differ_in_action_dit_training_only() -> None:
    p_v1 = _config("p_v1_smoke.yaml")
    p_v2 = _config("p_v2_smoke.yaml")
    assert p_v1["policy"]["regime"] == "p_v1"
    assert p_v1["policy"]["freeze"]["action_dit"] is True
    assert p_v2["policy"]["regime"] == "p_v2"
    assert p_v2["policy"]["freeze"]["action_dit"] is False
    for section in ("tasks", "architecture", "official", "paired", "supervision", "loss", "optimizer", "training", "evaluation"):
        assert p_v1[section] == p_v2[section]
    assert p_v1["policy"]["head_init_mode"] == p_v2["policy"]["head_init_mode"] == "random"
    assert p_v1["policy"]["head_init"] is p_v2["policy"]["head_init"] is None


def test_p_mode_dev_pilots_are_matched_and_not_engineering_smokes() -> None:
    p_v1 = _config("p_v1_dev_pilot.yaml")
    p_v2 = _config("p_v2_dev_pilot.yaml")
    assert p_v1["stage"] == p_v2["stage"] == "dev_pilot"
    assert p_v1["evaluation"]["episodes_per_task"] == 20
    assert p_v1["evaluation"]["simulator_seed_bank_purpose"] == "dev_selection"
    assert p_v1["p_mode_selection_manifest"] is None
    assert p_v2["p_mode_selection_manifest"] is None
    assert p_v1["selection_role"] == p_v2["selection_role"] == "c1_lambda0"
    assert p_v1["loss"]["lambda_contrastive"] == 0.0
    assert p_v1["paired"]["contrastive_supervision"] is False
    for section in (
        "tasks",
        "architecture",
        "official",
        "paired",
        "supervision",
        "loss",
        "optimizer",
        "training",
        "evaluation",
    ):
        assert p_v1[section] == p_v2[section]
    assert p_v1["policy"]["regime"] == "p_v1"
    assert p_v2["policy"]["regime"] == "p_v2"
    assert p_v1["policy"]["freeze"]["action_dit"] is True
    assert p_v2["policy"]["freeze"]["action_dit"] is False


@pytest.mark.parametrize("filename", ("p_v1_smoke.yaml", "p_v2_smoke.yaml", "c1_architecture_only.yaml", "c3_ours.yaml"))
def test_contrastive_controls_require_four_scene_native50_protocol(filename: str) -> None:
    paired = _config(filename)["paired"]
    assert paired["protocol_id"] == POLICY_PROTOCOL_ID
    assert tuple(paired["variants"]) == POLICY_VARIANTS
    assert paired["view_count"] == 4
    assert paired["r3_role"] == POLICY_R3_ROLE
    assert tuple(paired["camera_names"]) == POLICY_CAMERA_NAMES
    assert paired["camera_count"] == 3
    assert paired["native_fps"] == 50
    assert paired["action_steps"] == 32
    assert paired["action_dim"] == 14
    assert paired["temporal_resampling"] == "none"
    assert paired["native_action_targets"] is True


def test_c1_c2_c3_have_identical_seeded_random_architecture_and_training_contract() -> None:
    c1 = _config("c1_architecture_only.yaml")
    c2 = _config("c2_naive_aug.yaml")
    c3 = _config("c3_ours.yaml")
    for candidate in (c2, c3):
        for field in ("tasks", "base_checkpoint", "base_lineage_manifest", "release_paired_binding_manifest", "model_base_path", "policy", "architecture", "official", "optimizer", "training", "evaluation"):
            assert candidate[field] == c1[field]
    assert c1["policy"]["head_init_mode"] == "random"
    assert c1["artifacts"]["head_init_sha256"] is None
    assert validate_c1_c3_pair(c1, c3)["fairness"] == "PASS"
    assert validate_c2_c3_pair(c2, c3)["fairness"] == "PASS"


def test_only_stage2_supervision_changes_across_c1_c2_c3() -> None:
    c1 = _config("c1_architecture_only.yaml")
    c2 = _config("c2_naive_aug.yaml")
    c3 = _config("c3_ours.yaml")
    assert c1["paired"]["supervision_mode"] == "contrastive"
    assert c2["paired"]["supervision_mode"] == "action"
    assert c3["paired"]["supervision_mode"] == "contrastive"
    assert c1["loss"]["lambda_paired_action"] == 0.0
    assert c2["loss"]["lambda_paired_action"] == 1.0
    assert c3["loss"]["lambda_contrastive"] == 0.1
    assert c1["loss"]["lambda_contrastive"] == 0.0
    assert c1["paired"]["cache"] == c3["paired"]["cache"]
    assert c1["paired"]["contrastive_supervision"] is False
    assert c3["paired"]["contrastive_supervision"] is True
    assert c2["paired"]["action_manifest"] == c3["paired"]["action_manifest"]
    assert c2["paired"]["action_audit"] == c3["paired"]["action_audit"]
    assert c1["optimizer"]["lr_scheduler"] == "constant"


def test_pair_audits_reject_matched_control_drift() -> None:
    c1 = _config("c1_architecture_only.yaml")
    c3 = copy.deepcopy(_config("c3_ours.yaml"))
    c3["training"]["max_grad_norm"] = 0.5
    with pytest.raises(ConfigAuditError, match="unfair common Stage-2 mismatch"):
        validate_c1_c3_pair(c1, c3)
    c2 = _config("c2_naive_aug.yaml")
    c3 = copy.deepcopy(_config("c3_ours.yaml"))
    c3["paired"]["action_manifest"] = "__REQUIRED_DIFFERENT_MANIFEST__"
    with pytest.raises(ConfigAuditError, match="paired dataset mismatch"):
        validate_c2_c3_pair(c2, c3)


def test_c0_is_fixed_author_release_reference_not_adapter_training() -> None:
    config = _config("c0_original.yaml")
    assert config["policy"]["enabled"] is False
    assert config["execution"]["runner"] == "author_release_reference"
    assert config["paired"]["enabled"] is False
    assert config["training"]["seed"] is None
    assert config["training"]["max_steps"] == 0
    assert config["release_paired_binding_manifest"] is None


@pytest.mark.parametrize("filename", FORMAL_FILENAMES)
def test_formal_templates_are_three_seed_full_online_evaluation(filename: str) -> None:
    config = _config(filename)
    placeholders = find_placeholders(config)
    assert "base_checkpoint" not in placeholders
    assert config["base_checkpoint"].endswith("robotwin_uncond_3cam_384.pt")
    assert "release_paired_binding_manifest" not in placeholders
    assert config["release_paired_binding_manifest"].endswith(
        "/release_base_v1/paired_binding_manifest.json"
    )
    assert "formal_protocol_lock_manifest" in placeholders
    assert "policy.regime" in placeholders
    assert "p_mode_selection_manifest" in placeholders
    assert "artifacts.p_mode_selection_manifest_sha256" in placeholders
    assert "evaluation.simulator_seed_bank_manifest" in placeholders
    assert "training.seed" in placeholders
    assert config["protocol"]["seeds"] == [1, 2, 3]
    assert config["evaluation"]["required_domains"] == ["clean", "official_random"]
    assert config["evaluation"]["episodes_per_task"] == 100
    assert "r3_background_only" not in config["evaluation"]


def test_formal_matrix_passes_primary_c1_c3_fairness() -> None:
    result = validate_formal_matrix(CONFIG_DIR)
    assert result["status"] == "PASS"
    assert result["matrix_id"] == "three_task_author_release_c1_c3_v3"
    assert [row["comparison"] for row in result["comparisons"]] == [
        "C3-C1 paired contrastive total gain",
    ]
    assert all(row["fairness"] == "PASS" for row in result["comparisons"])


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("official", "sampling_mode"), "episode_anchor", "all_frames"),
        (("protocol", "head_initialization"), "pretrained", "Head initialization"),
        (("protocol", "base_parent"), "strict_three_task_b_cr", "release parent"),
        (("protocol", "action_distribution"), "wrong", "action distribution"),
        (("evaluation", "episodes_per_task"), 10, "100 episodes"),
        (("evaluation", "required_domains"), ["clean", "r3"], "Clean/official Random"),
    ),
)
def test_formal_protocol_values_are_locked(path: tuple[str, ...], replacement: object, message: str) -> None:
    config = copy.deepcopy(_config("formal_c1_architecture_only.yaml"))
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ConfigAuditError, match=message):
        validate_config_structure(config)


def test_random_head_cannot_secretly_load_pretrained_e2_head() -> None:
    config = copy.deepcopy(_config("formal_c1_architecture_only.yaml"))
    config["policy"]["head_init"] = "outputs/e2_e3/full/e2/e2_best_content_head.pt"
    with pytest.raises(ConfigAuditError, match="must not load"):
        validate_config_structure(config)


def test_formal_declaration_cannot_hide_a_pretrained_head() -> None:
    config = copy.deepcopy(_config("formal_c1_architecture_only.yaml"))
    config["policy"]["head_init_mode"] = "pretrained"
    config["policy"]["head_init"] = "outputs/ablation/pretrained_head.pt"
    config["artifacts"]["head_init_sha256"] = "a" * 64
    with pytest.raises(ConfigAuditError, match="actually use seeded random"):
        validate_config_structure(config)


def test_wrong_r3_role_or_three_view_data_is_rejected() -> None:
    config = copy.deepcopy(_config("c3_ours.yaml"))
    config["paired"]["r3_role"] = "holdout"
    with pytest.raises(ConfigAuditError, match="training_positive"):
        validate_config_structure(config)
    config = copy.deepcopy(_config("c3_ours.yaml"))
    config["paired"]["variants"] = config["paired"]["variants"][:3]
    with pytest.raises(ConfigAuditError, match="C/R1/R2/R3"):
        validate_config_structure(config)


def test_formal_pair_alias_is_primary_c1_c3_comparison() -> None:
    result = validate_formal_pair(
        _config("formal_c1_architecture_only.yaml"),
        _config("formal_c3_ours.yaml"),
    )
    assert result["fairness"] == "PASS"


def test_directory_audit_classifies_current_and_legacy_without_marking_ready() -> None:
    result = audit_config_directory(CONFIG_DIR)
    assert result["status"] == "PASS"
    assert result["short_control_matrix"]["c1_c3"]["fairness"] == "PASS"
    assert result["short_control_matrix"]["c2_c3"]["fairness"] == "PASS"
    rows = {row["file"]: row for row in result["configs"]}
    assert all(rows[name]["execution_ready"] is False for name in CURRENT_CONFIGS)
    assert all(rows[name]["structure"] == "LEGACY_V1" for name in LEGACY_FORMAL_FILENAMES)
    assert tuple(_config("c1_architecture_only.yaml")["tasks"]) == TASKS
