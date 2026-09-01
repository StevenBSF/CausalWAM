from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import eval_robotwin_single
from experiments.robotwin.policy_content_adapter import p_mode_selection


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_candidate(
    root: Path,
    *,
    regime: str,
    seed_bank: dict,
    clean: float,
    random: float,
    stage: str = "dev_pilot",
    base_sha: str | None = None,
    paired_cache_sha: str | None = None,
    lambda_contrastive: float = 0.0,
    selection_role: str = "c1_lambda0",
) -> Path:
    candidate_root = root / regime
    candidate_root.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_root / "checkpoint.pt"
    checkpoint.write_bytes((regime + "-checkpoint").encode())
    settings = {
        "schema": "test.dev.rollout_settings",
        "episodes_per_task": p_mode_selection.DEV_EPISODES_PER_CELL,
        "rollout_protocol_id": "three_task_policy_online_v2",
    }
    settings_sha = p_mode_selection.canonical_sha256(settings)
    recipe = {"schema": "policy_stage2_common_recipe_v1", "max_steps": 100}
    identity = {
        "base_checkpoint_sha256": base_sha or _sha("base"),
        "dataset_stats_sha256": _sha("stats"),
        "base_lineage_manifest_sha256": _sha("release-lineage"),
        "policy_regime": regime,
        "head_init_sha256": _sha("head"),
        "gca_init_sha256": _sha("gca"),
        "stage2_recipe_sha256": p_mode_selection.canonical_sha256(recipe),
        "p_mode_selection_manifest_sha256": None,
        "runtime_source_sha256": _sha("runtime"),
        "official_sample_sequence_sha256": _sha("official-sequence"),
        "paired_physical_state_sequence_sha256": _sha("paired-sequence"),
        "matched_stream_contract_sha256": _sha("matched-stream"),
    }
    runs = []
    for task in p_mode_selection.TASKS:
        for task_config, domain, rate in (
            ("demo_clean", "clean", clean),
            ("demo_randomized", "official_random", random),
        ):
            result = candidate_root / f"{task}_{domain}.txt"
            result.write_text(f"{rate}\n")
            runs.append(
                {
                    "task": task,
                    "phase": "clean" if domain == "clean" else "random",
                    "task_config": task_config,
                    "domain": domain,
                    "episodes": p_mode_selection.DEV_EPISODES_PER_CELL,
                    "simulator_seed": seed_bank["simulator_seed"],
                    "rollout_protocol_id": "three_task_policy_online_v2",
                    "rollout_settings_sha256": settings_sha,
                    "simulator_seed_bank_id": seed_bank["simulator_seed_bank_id"],
                    "simulator_seed_bank_purpose": "dev_selection",
                    "result": str(result.resolve()),
                    "success_rate": rate,
                }
            )
    manifest = {
        "schema": p_mode_selection.COMPLETED_ROLLOUTS_SCHEMA,
        "schema_version": p_mode_selection.COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_contract": {
            "control": regime,
            "stage": stage,
            "training_seed": 42,
            "selection_role": selection_role,
            "lambda_contrastive": lambda_contrastive,
            **identity,
            "stage2_recipe": recipe,
            "checkpoint_identity": {
                "path": str(checkpoint.resolve()),
                "size_bytes": checkpoint.stat().st_size,
                "mtime_ns": checkpoint.stat().st_mtime_ns,
            },
            "rollout_protocol_id": "three_task_policy_online_v2",
            "simulator_seed_bank_id": seed_bank["simulator_seed_bank_id"],
            "simulator_seed_bank_purpose": "dev_selection",
            "simulator_seed_bank_manifest_sha256": _sha("dev-bank-file"),
            "dev_pilot_artifact_shas": {
                "official_manifest_sha256": _sha("official-manifest"),
                "paired_action_manifest_sha256": _sha("paired-action-manifest"),
                "paired_state_bank_sha256": _sha("paired-state-bank"),
                "paired_text_cache_sha256": _sha("paired-text-cache"),
                "paired_cache_sha256": paired_cache_sha or _sha("paired-cache"),
            },
            "declared_tasks": list(p_mode_selection.TASKS),
            "declared_domains": list(p_mode_selection.DOMAINS),
            "declared_episodes_per_task": p_mode_selection.DEV_EPISODES_PER_CELL,
            "formal_evaluation_eligible": False,
        },
        "checkpoint_fairness_identity": None,
        "simulator_seed": seed_bank["simulator_seed"],
        "episodes_per_task": p_mode_selection.DEV_EPISODES_PER_CELL,
        "rollout_protocol_id": "three_task_policy_online_v2",
        "rollout_settings": settings,
        "rollout_settings_sha256": settings_sha,
        "simulator_seed_bank": seed_bank,
        "simulator_seed_bank_id": seed_bank["simulator_seed_bank_id"],
        "simulator_seed_bank_purpose": "dev_selection",
        "evaluation_protocol": {"eligible": False, "control": None},
        "evaluation_records": [],
        "runs": runs,
    }
    destination = candidate_root / "completed_rollouts.json"
    destination.write_text(json.dumps(manifest))
    return destination


def _candidate_pair(tmp_path: Path, **p_v2_overrides: object) -> tuple[Path, Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    evaluator = tmp_path / "eval_policy.py"
    evaluator.write_text("# deterministic evaluator\n")
    bank = eval_robotwin_single._build_simulator_seed_bank(
        simulator_seed=2,
        episodes_per_task=p_mode_selection.DEV_EPISODES_PER_CELL,
        evaluator_source=evaluator,
        purpose="dev_selection",
    )
    p_v1 = _write_candidate(tmp_path, regime="p_v1", seed_bank=bank, clean=0.70, random=0.50)
    values = {"clean": 0.60, "random": 0.80, **p_v2_overrides}
    p_v2 = _write_candidate(
        tmp_path, regime="p_v2", seed_bank=bank,
        clean=float(values.pop("clean")), random=float(values.pop("random")), **values,
    )
    return p_v1, p_v2, bank


def test_selector_applies_clean_guard_and_refuses_overwrite(tmp_path: Path) -> None:
    p_v1, p_v2, _ = _candidate_pair(tmp_path)
    output = tmp_path / "selection.json"
    selected = p_mode_selection.select_p_mode(p_v1_manifest=p_v1, p_v2_manifest=p_v2, output=output)
    assert selected["winner"] == "p_v1"
    assert selected["shared_candidate_identity"]["selection_role"] == "c1_lambda0"
    assert selected["shared_candidate_identity"]["lambda_contrastive"] == 0.0
    with pytest.raises(p_mode_selection.PModeSelectionError, match="refusing to overwrite"):
        p_mode_selection.select_p_mode(p_v1_manifest=p_v1, p_v2_manifest=p_v2, output=output)


def test_selector_uses_random_macro_and_tie_prefers_p_v1(tmp_path: Path) -> None:
    p_v1, p_v2, _ = _candidate_pair(tmp_path / "winner", clean=0.70, random=0.65)
    assert p_mode_selection.select_p_mode(p_v1_manifest=p_v1, p_v2_manifest=p_v2, output=tmp_path / "winner.json")["winner"] == "p_v2"
    p_v1, p_v2, _ = _candidate_pair(tmp_path / "tie", clean=0.70, random=0.50)
    assert p_mode_selection.select_p_mode(p_v1_manifest=p_v1, p_v2_manifest=p_v2, output=tmp_path / "tie.json")["winner"] == "p_v1"


@pytest.mark.parametrize(("override", "message"), (
    ({"stage": "smoke"}, "only dev_pilot"),
    ({"base_sha": _sha("different")}, "candidate identity mismatch"),
    ({"paired_cache_sha": _sha("different-cache")}, "candidate identity mismatch"),
    ({"lambda_contrastive": 0.1}, "lambda_contrastive=0"),
    ({"selection_role": "c3_method"}, "C1 lambda=0"),
))
def test_selector_rejects_ineligible_or_unmatched_candidates(tmp_path: Path, override: dict, message: str) -> None:
    p_v1, p_v2, _ = _candidate_pair(tmp_path, **override)
    with pytest.raises(p_mode_selection.PModeSelectionError, match=message):
        p_mode_selection.select_p_mode(p_v1_manifest=p_v1, p_v2_manifest=p_v2, output=tmp_path / "selection.json")


def test_selector_reparses_result_files(tmp_path: Path) -> None:
    p_v1, p_v2, _ = _candidate_pair(tmp_path)
    payload = json.loads(p_v2.read_text())
    payload["runs"][0]["success_rate"] = 0.95
    p_v2.write_text(json.dumps(payload))
    with pytest.raises(p_mode_selection.PModeSelectionError, match="result file success rate differs"):
        p_mode_selection.select_p_mode(p_v1_manifest=p_v1, p_v2_manifest=p_v2, output=tmp_path / "selection.json")


def _formal_lock_inputs(root: Path, selection_path: Path) -> tuple[Path, Path, list[Path], list[Path]]:
    lineage = root / "lineage.json"
    lineage.write_text(json.dumps({
        "kind": "policy_author_release_base_lineage", "schema_version": 1,
        "status": "PASS", "base_kind": "author_release",
    }))
    matrix = root / "matrix.json"
    matrix.write_text('{"status":"PASS"}')
    lineage_sha = hashlib.sha256(lineage.read_bytes()).hexdigest()
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    winner = json.loads(selection_path.read_text())["winner"]
    configs: dict[str, list[Path]] = {"c1_architecture_only": [], "c3_ours": []}
    for control in configs:
        for seed in (1, 2, 3):
            path = root / f"{control}_{seed}.json"
            path.write_text(json.dumps({
                "formal": True, "stage": "formal", "control": control,
                "training": {"seed": seed}, "policy": {"regime": winner},
                "loss": {"lambda_contrastive": 0.0 if control.startswith("c1") else 0.1},
                "artifacts": {
                    "base_lineage_manifest_sha256": lineage_sha,
                    "p_mode_selection_manifest_sha256": selection_sha,
                },
            }))
            configs[control].append(path)
    return lineage, matrix, configs["c1_architecture_only"], configs["c3_ours"]


def test_final_bank_requires_disjoint_dev_and_post_selection_formal_lock(tmp_path: Path) -> None:
    p_v1, p_v2, dev = _candidate_pair(tmp_path / "pilot")
    selection = tmp_path / "selection.json"
    p_mode_selection.select_p_mode(p_v1_manifest=p_v1, p_v2_manifest=p_v2, output=selection)
    lineage, matrix, c1, c3 = _formal_lock_inputs(tmp_path, selection)
    lock = tmp_path / "formal_lock.json"
    p_mode_selection.write_formal_protocol_lock_manifest(
        base_lineage_manifest=lineage, p_mode_selection_manifest=selection,
        formal_matrix_audit=matrix, c1_configs=c1, c3_configs=c3, output=lock,
    )
    evaluator = tmp_path / "eval_policy.py"
    evaluator.write_text("# evaluator\n")
    dev_path = tmp_path / "dev_bank.json"
    dev_path.write_text(json.dumps(dev))
    final_path = tmp_path / "final_bank.json"
    final = p_mode_selection.write_seed_bank_manifest(
        simulator_seed=4, episodes_per_cell=100, evaluator_source=evaluator,
        purpose="final_test", output=final_path,
        disjoint_from_dev_manifest=dev_path,
        p_mode_selection_manifest=selection,
        formal_protocol_lock_manifest=lock,
    )
    assert set(final["members"]).isdisjoint(dev["members"])
    assert final["lock_ancestry"]["p_mode_selection_manifest"]["sha256"] == hashlib.sha256(selection.read_bytes()).hexdigest()
    assert p_mode_selection.validate_seed_bank_descriptor(final, expected_purpose="final_test")

    tampered = copy.deepcopy(final)
    tampered["disjoint_from"][0]["members"] = final["members"]
    tampered["disjoint_from"][0]["members_sha256"] = p_mode_selection.canonical_sha256(final["members"])
    with pytest.raises(p_mode_selection.PModeSelectionError, match="not disjoint"):
        p_mode_selection.validate_seed_bank_descriptor(tampered)


def test_final_bank_cannot_be_created_before_protocol_lock(tmp_path: Path) -> None:
    evaluator = tmp_path / "eval.py"
    evaluator.write_text("# eval\n")
    dev = p_mode_selection.write_seed_bank_manifest(
        simulator_seed=1, episodes_per_cell=20, evaluator_source=evaluator,
        purpose="dev_selection", output=tmp_path / "dev.json",
    )
    assert dev["lock_ancestry"] == {}
    with pytest.raises(p_mode_selection.PModeSelectionError, match="requires P-mode selection"):
        p_mode_selection.write_seed_bank_manifest(
            simulator_seed=4, episodes_per_cell=100, evaluator_source=evaluator,
            purpose="final_test", output=tmp_path / "final.json",
            disjoint_from_dev_manifest=tmp_path / "dev.json",
        )
