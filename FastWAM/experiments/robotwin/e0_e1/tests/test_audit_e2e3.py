from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.e0_e1.audit_e2e3 import (
    E2E3AuditError,
    _assert_e2_e3_cache_equivalence,
    _identity,
    _validate_cache,
    _validate_checkpoint_run_controls,
    _validate_comparison,
    _validate_metric,
    _validate_proprio,
    _validate_selection,
    audit_e2e3_run,
)
from experiments.robotwin.e0_e1.compare_e2e3 import ALL_CONTROLS, compare_e2e3
from experiments.robotwin.e0_e1.train_e2e3 import (
    _canonical_json_sha256,
    _controlled_training_config,
)


SEEN = ("clean", "style_00_seed_0", "style_01_seed_1")


def _run_controls() -> dict[str, int | float]:
    return {
        "train_steps": 2,
        "groups_per_batch": 2,
        "val_every": 1,
        "seed": 7,
        "temperature": 0.07,
        "min_temporal_gap": 8,
        "min_state_distance": 1e-5,
    }


def _cache(split: str, mode: str) -> dict:
    content = {"train": 0, "val": 30, "test": 40}[split]
    variants = SEEN if split != "test" else ("clean", "style_02_seed_2")
    state_id = f"toy/content_{content:06d}/frame_000000"
    visual = {
        state_id: {
            variant: {
                "deployment_composite": f"{index + 1:064x}",
                "encoded_rgb_by_camera": {
                    "head_camera": f"{index + 11:064x}",
                    "left_camera": f"{index + 21:064x}",
                    "right_camera": f"{index + 31:064x}",
                },
            }
            for index, variant in enumerate(variants)
        }
    }
    return {
        "records": [
            {
                "task": "toy",
                "physical_state_id": state_id,
                "trajectory_id": f"toy/content_{content:06d}",
                "timestep": 0,
                "trace_idx": 0,
                "content_id": content,
                "variant": variant,
                "split": split,
            }
            for variant in variants
        ],
        "physical_states": [{"robot.q": 0.0}],
        "proprio_raw": torch.zeros(1, 14),
        "visual_input_sha256_by_physical_state": visual,
        "provenance": {
            "source_manifest_sha256": "1" * 64,
            "task_prompt_sha256": {"toy": "2" * 64},
            "backbone": {
                "checkpoint": "frozen",
                "proprio_mode": mode,
                "native_prefill_verified": mode == "observed",
            },
        },
    }


def _contract(split: str) -> dict:
    content = {"train": 0, "val": 30, "test": 40}[split]
    return {
        "variant_names": list(SEEN if split != "test" else ("clean", "style_02_seed_2")),
        "variants_per_state": 3 if split != "test" else 2,
        "num_records": 3 if split != "test" else 2,
        "num_physical_states": 1,
        "physical_state_ids": [f"toy/content_{content:06d}/frame_000000"],
        "records_sha256": f"{content + 3:064x}",
        "physical_states_sha256": f"{content + 4:064x}",
        "proprio_raw_sha256": f"{content + 5:064x}",
        "proprio_raw_shape": [1, 14],
        "token_shapes_by_layer": {"8": [3 if split != "test" else 2, 4, 16]},
        "source_manifest_sha256": "1" * 64,
        "visual_inputs_sha256": f"{content + 6:064x}",
    }


def test_cache_equivalence_checks_exact_visual_inputs_and_contract() -> None:
    caches = {
        experiment: {
            split: _cache(split, "observed" if experiment == "E2" else "constant_zero_normalized")
            for split in ("train", "val", "test")
        }
        for experiment in ("E2", "E3")
    }
    evidence = {
        experiment: {
            split: {"scientific_contract": _contract(split)}
            for split in ("train", "val", "test")
        }
        for experiment in ("E2", "E3")
    }
    result = _assert_e2_e3_cache_equivalence(caches, evidence)
    assert result["test"]["visual_inputs_sha256"] == _contract("test")["visual_inputs_sha256"]

    tampered = copy.deepcopy(caches)
    state = next(iter(tampered["E3"]["test"]["visual_input_sha256_by_physical_state"]))
    tampered["E3"]["test"]["visual_input_sha256_by_physical_state"][state]["clean"][
        "deployment_composite"
    ] = "f" * 64
    with pytest.raises(E2E3AuditError, match="visual-input"):
        _assert_e2_e3_cache_equivalence(tampered, evidence)


def test_no_proprio_audit_requires_exact_state_keys_and_one_hex_hash() -> None:
    state = "toy/content_000000/frame_000000"
    zero_hash = "0" * 64
    cache = {
        "records": [{"physical_state_id": state}],
        "provenance": {
            "backbone": {
                "proprio_mode": "constant_zero_normalized",
                "uses_future_video": False,
                "uses_action_denoising": False,
                "uses_policy_rollout": False,
            },
            "conditions_by_physical_state": {
                state: {
                    "normalized_proprio_sha256": zero_hash,
                    "context": {"proprio": {
                        "mode": "constant_zero_normalized",
                        "intervention_point": "post_normalizer_pre_proprio_encoder",
                        "proprio_token_preserved": True,
                        "shape": [2, 14],
                        "observed_normalized_sha256": "1" * 64,
                        "effective_normalized_sha256": zero_hash,
                        "all_zero": True,
                    }},
                }
            },
        },
    }
    assert _validate_proprio(
        cache, mode="constant_zero_normalized", label="E3/train"
    ) == [zero_hash]

    cache["provenance"]["conditions_by_physical_state"]["extra"] = copy.deepcopy(
        cache["provenance"]["conditions_by_physical_state"][state]
    )
    with pytest.raises(E2E3AuditError, match="exactly cover"):
        _validate_proprio(cache, mode="constant_zero_normalized", label="E3/train")


def test_cache_audit_binds_loaded_backbone_to_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = {
        "path": "/models/release.pt",
        "size_bytes": 10,
        "mtime_ns": 20,
        "sha256": "a" * 64,
    }
    config = {
        "mode": "smoke",
        "tasks": ["place_a2b_left"],
        "states_per_trajectory": 2,
        "checkpoint": checkpoint,
        "dataset_stats": {
            "path": "/models/stats.json",
            "size_bytes": 30,
            "mtime_ns": 40,
            "sha256": "b" * 64,
        },
        "model_base": "/models",
    }
    cache = _cache("train", "observed")
    cache["variant_names"] = list(SEEN)
    cache["provenance"].update({
        "protocol": "r3_holdout_v1",
        "split": "train",
        "proprio_mode": "observed",
        "active_variants": list(SEEN),
        "holdout_variant": "style_02_seed_2",
        "allow_incomplete": True,
        "content_ids": [0],
        "tasks": ["toy"],
        "decision_lock_identity": None,
    })
    backbone = cache["provenance"]["backbone"]
    backbone.update({
        "checkpoint": checkpoint,
        "dataset_stats_path": "/models/stats.json",
        "dataset_stats_sha256": "b" * 64,
        "model_base_path": "/models",
    })
    cache_path = tmp_path / "cache.pt"
    cache_path.write_bytes(b"cache")
    monkeypatch.setattr(
        "experiments.robotwin.e0_e1.audit_e2e3._identity",
        lambda path, memo: {
            "path": str(Path(path).resolve()),
            "size_bytes": 5,
            "mtime_ns": 1,
            "sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        "experiments.robotwin.e0_e1.audit_e2e3.load_cache", lambda path: cache
    )
    with pytest.raises(E2E3AuditError, match="task order mismatch"):
        _validate_cache(
            cache_path,
            experiment="E2",
            split="train",
            config=config,
            memo={},
        )

    backbone["checkpoint"] = {**checkpoint, "sha256": "f" * 64}
    with pytest.raises(E2E3AuditError, match="backbone checkpoint identity sha256"):
        _validate_cache(
            cache_path,
            experiment="E2",
            split="train",
            config=config,
            memo={},
        )

    backbone["checkpoint"] = checkpoint
    backbone["dataset_stats_sha256"] = "f" * 64
    with pytest.raises(E2E3AuditError, match="dataset-stats SHA-256"):
        _validate_cache(
            cache_path,
            experiment="E2",
            split="train",
            config=config,
            memo={},
        )

    backbone["dataset_stats_sha256"] = "b" * 64
    backbone["model_base_path"] = "/wrong"
    with pytest.raises(E2E3AuditError, match="model-base path"):
        _validate_cache(
            cache_path,
            experiment="E2",
            split="train",
            config=config,
            memo={},
        )


def _candidate(path: Path, layer: int, retrieval: float, ratio: float) -> Path:
    rows = [{
        "task": "toy",
        "layer": f"video_block_{layer:02d}",
        "experiment": "E2-RawBackbone",
        "retrieval_r1": retrieval,
        "state_style_ratio": ratio,
    }]
    payload = {
        "protocol": "r3_holdout_v1",
        "evaluation_split": "val",
        "experiment": "E2-RawBackbone",
        "proprio_mode": "observed",
        "active_variants": list(SEEN),
        "record_variants": list(SEEN),
        "holdout_variant": "style_02_seed_2",
        "layer": layer,
        "cache_identity": {"path": "/cache", "size_bytes": 1, "mtime_ns": 2, "sha256": "a" * 64},
        "negative_filter": {
            "min_temporal_gap": 8,
            "min_state_distance": 1e-5,
            "num_pairs": 1,
        },
        "cache_provenance": {
            "protocol": "r3_holdout_v1",
            "split": "val",
            "active_variants": list(SEEN),
            "proprio_mode": "observed",
        },
        "metrics": rows + [{**rows[0], "task": "1-task-average"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_selection_audit_rejects_rank_values_forged_after_metrics(tmp_path: Path) -> None:
    layer_dir = tmp_path / "layer_selection"
    layer_dir.mkdir()
    paths = {
        layer: _candidate(tmp_path / f"layer_{layer}.json", layer, retrieval, ratio)
        for layer, retrieval, ratio in ((8, .2, 2.0), (16, .4, 1.5), (24, .3, 3.0))
    }
    rows = [
        {
            "layer": layer,
            "macro_retrieval_r1": {8: .2, 16: .4, 24: .3}[layer],
            "macro_state_style_ratio": {8: 2.0, 16: 1.5, 24: 3.0}[layer],
            "joint_rank_sum": {8: 5, 16: 4, 24: 3}[layer],
            "selected": layer == 24,
            "source": str(paths[layer].resolve()),
        }
        for layer in (8, 16, 24)
    ]
    selection = {
        "schema_version": 2,
        "protocol": "r3_holdout_v1",
        "evaluation_split": "val",
        "experiment": "E2-RawBackbone",
        "proprio_mode": "observed",
        "r3_used": False,
        "active_variants": list(SEEN),
        "task_set": ["toy"],
        "cache_identity": {"path": "/cache", "size_bytes": 1, "mtime_ns": 2, "sha256": "a" * 64},
        "negative_filter": {
            "min_temporal_gap": 8,
            "min_state_distance": 1e-5,
            "num_pairs": 1,
        },
        "selected_layer": 24,
        "candidates": rows,
    }
    (layer_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (layer_dir / "selected_layer.txt").write_text("24\n", encoding="utf-8")
    _validate_selection(
        tmp_path,
        e2_val_identity=selection["cache_identity"],
        tasks=("toy",),
        config=_run_controls(),
        memo={},
    )

    selection["negative_filter"]["min_temporal_gap"] = 9
    (layer_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(E2E3AuditError, match="differs from run config"):
        _validate_selection(
            tmp_path,
            e2_val_identity=selection["cache_identity"],
            tasks=("toy",),
            config=_run_controls(),
            memo={},
        )
    selection["negative_filter"]["min_temporal_gap"] = 8

    selection["candidates"][2]["macro_retrieval_r1"] = .99
    (layer_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(E2E3AuditError, match="retrieval is stale"):
        _validate_selection(
            tmp_path,
            e2_val_identity=selection["cache_identity"],
            tasks=("toy",),
            config=_run_controls(),
            memo={},
        )


def _checkpoint_for_run_controls(*, experiment: str, kind: str) -> dict:
    config = _run_controls()
    head_config = {
        "backbone_dim": 16,
        "embed_dim": 16,
        "num_queries": 2,
        "num_heads": 4,
    }
    controlled = _controlled_training_config(
        layer=8,
        head_config=head_config,
        steps=int(config["train_steps"]),
        groups_per_batch=int(config["groups_per_batch"]),
        learning_rate=1e-4,
        weight_decay=1e-2,
        temperature=float(config["temperature"]),
        val_every=int(config["val_every"]),
        seed=int(config["seed"]),
        min_temporal_gap=int(config["min_temporal_gap"]),
        min_state_distance=float(config["min_state_distance"]),
    )
    mode = "observed" if experiment == "E2" else "constant_zero_normalized"
    return {
        "checkpoint_kind": kind,
        "training_steps": config["train_steps"],
        "temperature": config["temperature"],
        "negative_filter": {
            "min_temporal_gap": config["min_temporal_gap"],
            "min_state_distance": config["min_state_distance"],
        },
        "seed": config["seed"],
        "controlled_training_config": controlled,
        "controlled_training_config_sha256": _canonical_json_sha256(controlled),
        "training_config": {
            **controlled,
            "experiment": experiment,
            "proprio_mode": mode,
            "device": "cpu",
        },
    }


@pytest.mark.parametrize("kind", ["best_val", "final"])
@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("steps", 3),
        ("groups_per_batch", 3),
        ("val_every", 2),
        ("seed", 8),
        ("min_temporal_gap", 9),
        ("min_state_distance", 2e-5),
    ],
)
def test_best_and_final_checkpoint_controls_cannot_be_consistently_forged(
    kind: str, field: str, tampered: int | float
) -> None:
    checkpoint = _checkpoint_for_run_controls(experiment="E2", kind=kind)
    controlled = checkpoint["controlled_training_config"]
    controlled[field] = tampered
    checkpoint["training_config"][field] = tampered
    checkpoint["controlled_training_config_sha256"] = _canonical_json_sha256(controlled)
    with pytest.raises(E2E3AuditError, match="differs from run config"):
        _validate_checkpoint_run_controls(
            checkpoint,
            experiment="E2",
            config=_run_controls(),
            label=f"E2/{kind}",
        )


@pytest.mark.parametrize("kind", ["best_val", "final"])
def test_best_and_final_checkpoint_temperature_and_training_config_are_bound(
    kind: str,
) -> None:
    checkpoint = _checkpoint_for_run_controls(experiment="E3", kind=kind)
    checkpoint["controlled_training_config"]["loss"]["temperature"] = 0.08
    checkpoint["training_config"]["loss"]["temperature"] = 0.08
    checkpoint["temperature"] = 0.08
    checkpoint["controlled_training_config_sha256"] = _canonical_json_sha256(
        checkpoint["controlled_training_config"]
    )
    with pytest.raises(E2E3AuditError, match="SupCon configuration"):
        _validate_checkpoint_run_controls(
            checkpoint,
            experiment="E3",
            config=_run_controls(),
            label=f"E3/{kind}",
        )

    checkpoint = _checkpoint_for_run_controls(experiment="E3", kind=kind)
    checkpoint["training_config"]["groups_per_batch"] = 99
    with pytest.raises(E2E3AuditError, match="training config groups_per_batch"):
        _validate_checkpoint_run_controls(
            checkpoint,
            experiment="E3",
            config=_run_controls(),
            label=f"E3/{kind}",
        )


def _metric_row(task: str) -> dict:
    return {
        "task": task,
        "style_distance": 0.1,
        "clean_r3_distance": 0.1,
        "style_distance_R3": 0.1,
        "state_distance": 0.5,
        "state_style_ratio": 5.0,
        "state_style_ratio_R3": 5.0,
        "retrieval_r1": 0.5,
        "retrieval_r5": 1.0,
        "r3_to_clean_retrieval_at1": 0.5,
        "r3_to_clean_retrieval_at5": 1.0,
        "R3_to_Clean_R@1": 0.5,
        "R3_to_Clean_R@5": 1.0,
        "positive_similarity": 0.9,
        "negative_similarity": 0.4,
    }


def _metric_for_control(
    path: Path,
    experiment: str,
    *,
    cache_identity: dict,
    lock_identity: dict,
    checkpoint_identity: dict,
) -> dict:
    if experiment.endswith("InitHead"):
        head = {
            "control": "initial_head",
            "checkpoint_kind": "best_val",
            "initialization_seed": 7,
            "paired_checkpoint_identity": checkpoint_identity,
        }
    elif experiment.endswith("TrainedHead"):
        head = {
            "control": "trained_best_validation_head",
            "checkpoint_kind": "best_val",
            "training_seed": 7,
            "best_step": 1,
            "checkpoint_identity": checkpoint_identity,
        }
    else:
        head = {
            "control": "raw_backbone",
            "checkpoint_kind": "best_val",
            "training_seed": 7,
            "paired_checkpoint_identity": checkpoint_identity,
        }
    payload = {
        "schema_version": 2,
        "protocol": "r3_holdout_v1",
        "evaluation_split": "test",
        "experiment": experiment,
        "proprio_mode": (
            "constant_zero_normalized" if experiment.startswith("E3-") else "observed"
        ),
        "active_variants": ["clean", "style_02_seed_2"],
        "record_variants": ["clean", "style_02_seed_2"],
        "holdout_variant": "style_02_seed_2",
        "r3_used_for_selection": False,
        "layer": 8,
        "cache_identity": cache_identity,
        "decision_lock_identity": lock_identity,
        "negative_filter": {
            "min_temporal_gap": 8,
            "min_state_distance": 1e-5,
            "num_pairs": 1,
        },
        "metric_protocol": {
            "style_order": ["r3"],
            "required_variants": ["clean", "r3"],
            "query": "R3",
            "gallery": "Clean",
        },
        "head": head,
        "metrics": [_metric_row("toy"), _metric_row("1-task-average")],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


@pytest.mark.parametrize("experiment", ALL_CONTROLS)
def test_all_six_test_metrics_bind_filter_and_control_seed(
    experiment: str, tmp_path: Path
) -> None:
    cache_path = tmp_path / "cache.pt"
    lock_path = tmp_path / "lock.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    for path, content in (
        (cache_path, b"cache"),
        (lock_path, b"lock"),
        (checkpoint_path, b"checkpoint"),
    ):
        path.write_bytes(content)
    memo: dict = {}
    cache_identity = _identity(cache_path, memo)
    lock_identity = _identity(lock_path, memo)
    checkpoint_identity = _identity(checkpoint_path, memo)
    metric_path = tmp_path / "metric.json"
    payload = _metric_for_control(
        metric_path,
        experiment,
        cache_identity=cache_identity,
        lock_identity=lock_identity,
        checkpoint_identity=checkpoint_identity,
    )
    keyword_args = {
        "experiment": experiment,
        "selected_layer": 8,
        "cache_identity": cache_identity,
        "lock_identity": lock_identity,
        "tasks": ("toy",),
        "checkpoint": {"best_step": 1},
        "checkpoint_identity": checkpoint_identity,
        "config": _run_controls(),
        "memo": memo,
    }
    _validate_metric(metric_path, **keyword_args)

    payload["negative_filter"]["min_state_distance"] = 2e-5
    metric_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(E2E3AuditError, match="differs from run config"):
        _validate_metric(metric_path, **keyword_args)

    payload["negative_filter"]["min_state_distance"] = 1e-5
    seed_field = "initialization_seed" if experiment.endswith("InitHead") else "training_seed"
    payload["head"][seed_field] = 8
    metric_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(E2E3AuditError, match="seed differs from run config"):
        _validate_metric(metric_path, **keyword_args)


def _comparison_metric(path: Path, experiment: str, *, value: float) -> Path:
    is_e3 = experiment.startswith("E3-")
    head: dict | None = None
    if experiment.endswith("InitHead"):
        head = {"initial_head_sha256": "a" * 64, "initialization_seed": 0}
    elif experiment.endswith("TrainedHead"):
        head = {
            "initial_head_sha256": "a" * 64,
            "training_seed": 0,
            "checkpoint_kind": "best_val",
        }
    style = 0.1
    state = 0.5
    row = {
        "task": "1-task-average",
        "style_distance": style,
        "clean_r3_distance": style,
        "style_distance_R3": style,
        "state_distance": state,
        "state_style_ratio": state / style + value,
        "state_style_ratio_R3": state / style + value,
        "retrieval_r1": value,
        "retrieval_r5": min(1.0, value + 0.2),
        "r3_to_clean_retrieval_at1": value,
        "r3_to_clean_retrieval_at5": min(1.0, value + 0.2),
        "R3_to_Clean_R@1": value,
        "R3_to_Clean_R@5": min(1.0, value + 0.2),
        "positive_similarity": 0.9,
        "negative_similarity": 0.4,
    }
    payload = {
        "schema_version": 2,
        "protocol": "r3_holdout_v1",
        "evaluation_split": "test",
        "experiment": experiment,
        "proprio_mode": "constant_zero_normalized" if is_e3 else "observed",
        "active_variants": ["clean", "style_02_seed_2"],
        "holdout_variant": "style_02_seed_2",
        "layer": 8,
        "cache_identity": {"path": "/e3" if is_e3 else "/e2"},
        "decision_lock_identity": {"path": "/lock", "sha256": "b" * 64},
        "head": head,
        "metrics": [row],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _comparison_fixture(tmp_path: Path) -> tuple[dict, dict, dict, dict]:
    values = {
        "E2-RawBackbone": 0.2,
        "E2-InitHead": 0.3,
        "E2-TrainedHead": 0.8,
        "E3-NoProprio-RawBackbone": 0.15,
        "E3-NoProprio-InitHead": 0.25,
        "E3-NoProprio-TrainedHead": 0.7,
    }
    paths = {
        experiment: _comparison_metric(
            tmp_path / f"{experiment}.json", experiment, value=values[experiment]
        )
        for experiment in ALL_CONTROLS
    }
    e1_payload = {
        "evaluation_split": "test",
        "experiment": "E1-TrainedHead",
        "layer": 16,
        "metrics": [{
            "task": "3-task-average",
            "clean_r3_distance": 0.12,
            "state_distance": 0.6,
            "r3_to_clean_retrieval_at1": 0.9,
            "r3_to_clean_retrieval_at5": 1.0,
            "negative_similarity": 0.28,
        }],
    }
    e1_path = tmp_path / "e1.json"
    e1_path.write_text(json.dumps(e1_payload), encoding="utf-8")
    comparison_dir = tmp_path / "comparison"
    compare_e2e3(
        list(paths.values()), e1_metric_path=e1_path, output_dir=comparison_dir
    )
    metrics = {
        experiment: json.loads(path.read_text(encoding="utf-8"))
        for experiment, path in paths.items()
    }
    evidence = {
        experiment: {
            "identity": {"path": str(path.resolve())},
            "macro": {},
        }
        for experiment, path in paths.items()
    }
    lock = {"path": "/lock", "sha256": "b" * 64}
    return metrics, evidence, lock, json.loads(
        (comparison_dir / "comparison.json").read_text(encoding="utf-8")
    )


def test_comparison_audit_rederives_all_outputs_from_metric_inputs(tmp_path: Path) -> None:
    metrics, evidence, lock, _ = _comparison_fixture(tmp_path)
    comparison, artifacts = _validate_comparison(
        tmp_path,
        metrics=metrics,
        metric_evidence=evidence,
        selected_layer=8,
        lock_identity=lock,
        memo={},
    )
    assert comparison["interpretation"]["e2_beats_raw_and_init_on_r1_and_ratio"] is True
    assert comparison["interpretation"]["e2_r1_retention_vs_e1"] == pytest.approx(
        0.8 / 0.9
    )
    assert comparison["interpretation"]["e2_ratio_retention_vs_e1"] > 0.5
    assert comparison["interpretation"]["large_e1_to_e2_drop"] is False
    assert set(artifacts) == {
        "comparison_json",
        "e1_metric",
        "controls_csv",
        "e1_e2_e3_csv",
        "summary_markdown",
    }


@pytest.mark.parametrize(
    "target",
    ["final_json", "interpretation", "e2_interpretation", "controls_csv", "summary"],
)
def test_comparison_audit_rejects_stale_derived_artifacts(
    tmp_path: Path, target: str
) -> None:
    metrics, evidence, lock, comparison = _comparison_fixture(tmp_path)
    comparison_dir = tmp_path / "comparison"
    if target == "final_json":
        comparison["final_e1_e2_e3"][0]["R3_to_Clean_R@1"] = 0.01
        (comparison_dir / "comparison.json").write_text(
            json.dumps(comparison), encoding="utf-8"
        )
    elif target == "interpretation":
        comparison["interpretation"]["large_no_proprio_drop"] = not comparison[
            "interpretation"
        ]["large_no_proprio_drop"]
        (comparison_dir / "comparison.json").write_text(
            json.dumps(comparison), encoding="utf-8"
        )
    elif target == "e2_interpretation":
        comparison["interpretation"]["e2_r1_retention_vs_e1"] = 1.0
        comparison["interpretation"]["large_e1_to_e2_drop"] = True
        (comparison_dir / "comparison.json").write_text(
            json.dumps(comparison), encoding="utf-8"
        )
    elif target == "controls_csv":
        path = comparison_dir / "controls.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("0.2", "0.1", 1), encoding="utf-8")
    else:
        path = comparison_dir / "summary.md"
        path.write_text(path.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")
    with pytest.raises(E2E3AuditError, match="comparison"):
        _validate_comparison(
            tmp_path,
            metrics=metrics,
            metric_evidence=evidence,
            selected_layer=8,
            lock_identity=lock,
            memo={},
        )


def test_top_level_audit_fails_closed_without_publishing_reports(tmp_path: Path) -> None:
    (tmp_path / "status").mkdir()
    (tmp_path / "status/state.txt").write_text("RUNNING\n", encoding="utf-8")
    with pytest.raises(E2E3AuditError, match="run config"):
        audit_e2e3_run(tmp_path)
    assert not (tmp_path / "protocol_audit.json").exists()
    assert not (tmp_path / "deliverables.json").exists()
