from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.e0_e1.cache import build_cache_payload, save_cache
from experiments.robotwin.e0_e1.compare import compare_results
from experiments.robotwin.e0_e1.evaluate import evaluate_cache
from experiments.robotwin.e0_e1.runner_checks import (
    RunnerArtifactError,
    init_config,
    validate_cache_artifact,
    validate_comparison_artifact,
    validate_e0_metrics,
    validate_metric_artifact,
    validate_selection_artifact,
    validate_test_metrics,
    validate_training_artifact,
)
from experiments.robotwin.e0_e1.select_layer import select_layer
from experiments.robotwin.e0_e1.train_e1 import train_head


TASKS = ("toy",)
LAYERS = (8, 16)


def _formal_cache(tmp_path: Path, split: str, states: int = 2) -> Path:
    ranges = {
        "train": range(0, 30),
        "val": range(30, 40),
        "test": range(40, 50),
    }
    samples: list[dict[str, object]] = []
    token_rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
    conditions: dict[str, dict[str, object]] = {}
    for content_id in ranges[split]:
        for state_index in range(states):
            physical_key = f"toy/content_{content_id:06d}/frame_{state_index:06d}"
            samples.append(
                {
                    "physical_key": physical_key,
                    "task": "toy",
                    "content_id": content_id,
                    "frame_idx": state_index * 10,
                    "trace_idx": state_index * 100,
                    "split": split,
                    "variant_names": (
                        "clean",
                        "style_00_seed_0",
                        "style_01_seed_1",
                        "style_02_seed_2",
                    ),
                    "proprio_raw": torch.zeros(14),
                    "physical_state_by_name": {
                        "robot.q": float(content_id * states + state_index)
                    },
                }
            )
            conditions[physical_key] = {"task": "toy"}
            for variant_index in range(4):
                tokens = torch.zeros(4, 16)
                tokens[:, (content_id * states + state_index) % 8] = 1.0
                tokens[:, 8 + variant_index] = 0.01 * (variant_index + 1)
                for layer in LAYERS:
                    token_rows[layer].append(tokens + layer * 1e-4)
    manifest_jsonl = tmp_path / f"{split}_manifest.jsonl"
    manifest_csv = tmp_path / f"{split}_manifest.csv"
    manifest_jsonl.write_text("{}\n", encoding="utf-8")
    manifest_csv.write_text("physical_key\n", encoding="utf-8")
    payload = build_cache_payload(
        tokens_by_layer={
            layer: torch.stack(rows) for layer, rows in token_rows.items()
        },
        samples=samples,
        provenance={
            "split": split,
            "tasks": ["toy"],
            "states_per_trajectory": states,
            "allow_incomplete": False,
            "max_trajectories_per_task": None,
            "content_ids": None,
            "manifest_jsonl": str(manifest_jsonl.resolve()),
            "manifest_csv": str(manifest_csv.resolve()),
            "task_prompt_sha256": {"toy": "0" * 64},
            "backbone": {
                "capture_layers": list(LAYERS),
                "uses_future_video": False,
                "uses_action_denoising": False,
                "uses_policy_rollout": False,
            },
            "conditions_by_physical_state": conditions,
        },
    )
    path = tmp_path / f"{split}.pt"
    save_cache(path, payload)
    return path


def test_init_config_is_atomic_and_resume_requires_exact_match(tmp_path: Path) -> None:
    path = tmp_path / "run_config.json"
    checkpoint = tmp_path / "checkpoint.pt"
    stats = tmp_path / "stats.json"
    checkpoint.write_bytes(b"checkpoint-v1")
    stats.write_text("{}", encoding="utf-8")
    config = {
        "tasks": ["toy"],
        "seed": 0,
        "checkpoint": str(checkpoint),
        "dataset_stats": str(stats),
    }
    assert init_config(path, config) == path.resolve()
    first_bytes = path.read_bytes()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_identity"]["size_bytes"] == len(b"checkpoint-v1")
    assert payload["dataset_stats_identity"]["path"] == str(stats.resolve())
    assert init_config(path, config) == path.resolve()
    assert path.read_bytes() == first_bytes
    with pytest.raises(RunnerArtifactError, match="configuration differs"):
        init_config(path, {**config, "seed": 1})
    checkpoint.write_bytes(b"checkpoint-v2-is-different")
    with pytest.raises(RunnerArtifactError, match="configuration differs"):
        init_config(path, config)


def test_formal_cache_validation_rejects_provenance_or_count_tampering(
    tmp_path: Path,
) -> None:
    cache = _formal_cache(tmp_path, "train")
    validated = validate_cache_artifact(
        cache,
        split="train",
        tasks=TASKS,
        layers=LAYERS,
        states_per_trajectory=2,
    )
    assert len(validated["records"]) == 30 * 2 * 4

    payload = torch.load(cache, map_location="cpu", weights_only=True)
    payload["provenance"]["allow_incomplete"] = True
    bad = tmp_path / "bad_train.pt"
    torch.save(payload, bad)
    with pytest.raises(RunnerArtifactError, match="allow_incomplete"):
        validate_cache_artifact(
            bad,
            split="train",
            tasks=TASKS,
            layers=LAYERS,
            states_per_trajectory=2,
        )


def test_unattended_artifact_chain_and_tamper_gates(tmp_path: Path) -> None:
    train = _formal_cache(tmp_path, "train")
    val = _formal_cache(tmp_path, "val")
    test = _formal_cache(tmp_path, "test")

    selection_metrics = tmp_path / "selection_metrics"
    metric_paths: list[Path] = []
    for layer in LAYERS:
        evaluate_cache(
            cache_path=val,
            layer=layer,
            experiment="E0-RawBackbone",
            output_dir=selection_metrics,
            device="cpu",
        )
        metric_paths.append(
            selection_metrics / f"e0_rawbackbone_layer_{layer:02d}.json"
        )
    assert len(
        validate_e0_metrics(
            metric_paths,
            cache_path=val,
            tasks=TASKS,
            layers=LAYERS,
            min_temporal_gap=8,
            min_state_distance=1e-5,
        )
    ) == 2

    selection_dir = tmp_path / "layer_selection"
    selected = select_layer(metric_paths, selection_dir)
    assert validate_selection_artifact(
        selection_dir / "selection.json",
        selected_layer_path=selection_dir / "selected_layer.txt",
        cache_path=val,
        tasks=TASKS,
        layers=LAYERS,
    ) == selected

    evaluate_cache(
        cache_path=val,
        layer=selected,
        experiment="E1-InitHead",
        output_dir=selection_metrics,
        seed=0,
        device="cpu",
    )
    validate_metric_artifact(
        selection_metrics / f"e1_inithead_layer_{selected:02d}.json",
        cache_path=val,
        split="val",
        tasks=TASKS,
        layer=selected,
        experiment="E1-InitHead",
        seed=0,
    )

    train_dir = tmp_path / "e1"
    checkpoint = train_head(
        train_cache_path=train,
        val_cache_path=val,
        layer=selected,
        output_dir=train_dir,
        steps=1,
        groups_per_batch=2,
        val_every=1,
        seed=0,
        device="cpu",
    )
    validate_training_artifact(
        checkpoint,
        log_path=train_dir / "train_log.json",
        train_cache_path=train,
        val_cache_path=val,
        layer=selected,
        steps=1,
        seed=0,
        min_temporal_gap=8,
        min_state_distance=1e-5,
    )

    test_metrics = tmp_path / "test_metrics"
    for experiment in ("E0-RawBackbone", "E1-InitHead"):
        evaluate_cache(
            cache_path=test,
            layer=selected,
            experiment=experiment,
            output_dir=test_metrics,
            seed=0,
            device="cpu",
        )
    evaluate_cache(
        cache_path=test,
        layer=selected,
        experiment="E1-TrainedHead",
        output_dir=test_metrics,
        head_checkpoint=checkpoint,
        seed=0,
        device="cpu",
    )
    e0 = test_metrics / f"e0_rawbackbone_layer_{selected:02d}.json"
    init = test_metrics / f"e1_inithead_layer_{selected:02d}.json"
    trained = test_metrics / f"e1_trainedhead_layer_{selected:02d}.json"
    validate_test_metrics(
        cache_path=test,
        tasks=TASKS,
        layer=selected,
        seed=0,
        e0_path=e0,
        init_path=init,
        trained_path=trained,
    )

    bad_init_payload = json.loads(init.read_text(encoding="utf-8"))
    bad_init_payload["head"]["initial_head_sha256"] = "f" * 64
    bad_init = tmp_path / "bad_init.json"
    bad_init.write_text(json.dumps(bad_init_payload), encoding="utf-8")
    with pytest.raises(RunnerArtifactError, match="hashes differ"):
        validate_test_metrics(
            cache_path=test,
            tasks=TASKS,
            layer=selected,
            seed=0,
            e0_path=e0,
            init_path=bad_init,
            trained_path=trained,
        )

    comparison_dir = tmp_path / "comparison"
    compare_results([e0, init, trained], comparison_dir)
    comparison = comparison_dir / "comparison.json"
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    if payload["overall_success"]:
        validate_comparison_artifact(comparison)
    else:
        with pytest.raises(RunnerArtifactError, match="scientific success"):
            validate_comparison_artifact(comparison)
        validate_comparison_artifact(comparison, allow_scientific_fail=True)
