from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.e0_e1 import train_e2e3 as training


def _cache_payload(*, split: str, proprio_mode: str, add_r3: bool = False) -> dict:
    variants = list(training.ACTIVE_VARIANTS)
    records: list[dict[str, object]] = []
    physical_states: list[dict[str, float]] = []
    token_rows: list[torch.Tensor] = []
    content_id = {"train": 0, "val": 30}[split]
    for state_index in range(3):
        physical_id = f"toy/content_{content_id:06d}/frame_{state_index:06d}"
        physical_states.append({"robot.q": float(state_index)})
        for view_index, variant in enumerate(variants):
            records.append(
                {
                    "task": "toy",
                    "physical_state_id": physical_id,
                    "trajectory_id": f"toy/content_{content_id:06d}",
                    "timestep": state_index * 10,
                    "trace_idx": state_index,
                    "content_id": content_id,
                    "variant": variant,
                    "split": split,
                }
            )
            tokens = torch.zeros(4, 16)
            tokens[:, state_index] = 1.0
            tokens[:, 8 + view_index] = 0.02
            token_rows.append(tokens)
    if add_r3:
        records[2]["variant"] = training.HOLDOUT_VARIANT
    visual = {
        state["physical_state_id"]: {
            variant: {
                "deployment_composite": f"{state_index + view_index + 1:064x}"[-64:],
                "encoded_rgb_by_camera": {
                    camera: f"{state_index * 100 + view_index * 10 + camera_index + 11:064x}"[-64:]
                    for camera_index, camera in enumerate(
                        ("head_camera", "left_camera", "right_camera")
                    )
                },
            }
            for view_index, variant in enumerate(variants)
        }
        for state_index, state in enumerate(records[::len(variants)])
    }
    return {
        "schema_version": 2,
        "variant_names": variants,
        "variants_per_state": len(variants),
        "records": records,
        "physical_states": physical_states,
        "proprio_raw": torch.stack(
            [torch.full((14,), float(index)) for index in range(3)]
        ),
        "visual_input_sha256_by_physical_state": visual,
        "tokens_by_layer": {"8": torch.stack(token_rows)},
        "pooled_by_layer": {},
        "provenance": {
            "protocol": training.PROTOCOL,
            "split": split,
            "tasks": ["toy"],
            "active_variants": variants,
            "holdout_variant": training.HOLDOUT_VARIANT,
            "proprio_mode": proprio_mode,
            "source_manifest_sha256": "a" * 64,
            "backbone": {"checkpoint": "toy", "native_prefill_verified": split == "train"},
            "task_prompt_sha256": {"toy": "0" * 64},
        },
    }


def _write_cache(path: Path, payload: dict) -> Path:
    torch.save(payload, path)
    return path


def _raw_load_cache(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def test_protocol_rejects_r3_and_experiment_proprio_mismatch() -> None:
    leaked = _cache_payload(split="train", proprio_mode="observed", add_r3=True)
    with pytest.raises(ValueError, match="R3 holdout leaked|record variants"):
        training._require_exact_training_protocol(
            leaked, split="train", experiment="E2", proprio_mode="observed"
        )

    e2 = _cache_payload(split="train", proprio_mode="observed")
    with pytest.raises(ValueError, match="requires proprio_mode"):
        training._require_exact_training_protocol(
            e2,
            split="train",
            experiment="E2",
            proprio_mode="constant_zero_normalized",
        )
    with pytest.raises(ValueError, match="requires proprio_mode"):
        training._require_exact_training_protocol(
            e2, split="train", experiment="E3", proprio_mode="observed"
        )


def test_best_validation_tie_keeps_earliest_step() -> None:
    assert training._is_better_validation(1.0, 1, best_value=None, best_step=None)
    assert training._is_better_validation(0.9, 2, best_value=1.0, best_step=1)
    assert not training._is_better_validation(1.0, 2, best_value=1.0, best_step=1)
    assert not training._is_better_validation(1.1, 2, best_value=1.0, best_step=1)


def test_e2_e3_share_training_contract_and_save_best_and_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production cache loader is being upgraded independently to schema v2.
    # Isolate this trainer test from that module while still exercising real,
    # serialized cache identities and content SHA-256 values.
    monkeypatch.setattr(training, "load_cache", _raw_load_cache)

    outputs: dict[str, tuple[Path, dict, dict]] = {}
    for experiment, mode in training.EXPERIMENT_PROPRIO_MODES.items():
        train_path = _write_cache(
            tmp_path / f"{experiment.lower()}_train.pt",
            _cache_payload(split="train", proprio_mode=mode),
        )
        val_path = _write_cache(
            tmp_path / f"{experiment.lower()}_val.pt",
            _cache_payload(split="val", proprio_mode=mode),
        )
        output_dir = tmp_path / experiment.lower()
        best_path = training.train_e2e3_head(
            experiment=experiment,
            proprio_mode=mode,
            train_cache_path=train_path,
            val_cache_path=val_path,
            layer=8,
            output_dir=output_dir,
            steps=2,
            groups_per_batch=2,
            val_every=1,
            seed=7,
            device="cpu",
            min_temporal_gap=1,
        )
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        final_path = output_dir / f"{experiment.lower()}_final_content_head.pt"
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        outputs[experiment] = (best_path, best, final)

        assert best_path.name == f"{experiment.lower()}_best_content_head.pt"
        assert best["checkpoint_kind"] == "best_val"
        assert final["checkpoint_kind"] == "final"
        assert best["step"] == best["best_step"]
        assert final["step"] == 2
        assert best["best_metric"]["r3_used"] is False
        assert best["training_config"]["experiment"] == experiment
        assert best["training_config"]["proprio_mode"] == mode
        assert len(best["train_cache_identity"]["sha256"]) == 64
        assert len(best["val_cache_identity"]["sha256"]) == 64
        assert len(best["train_scientific_cache_contract"]["records_sha256"]) == 64
        assert len(best["val_scientific_cache_contract"]["proprio_raw_sha256"]) == 64
        assert best["train_scientific_cache_contract"]["source_manifest_sha256"] == "a" * 64
        assert len(best["train_scientific_cache_contract"]["visual_inputs_sha256"]) == 64
        assert best["initial_head_sha256"] == final["initial_head_sha256"]
        assert (output_dir / "train_log.json").is_file()
        assert (output_dir / "train_log.csv").is_file()
        curve = output_dir / "training_curves.svg"
        assert curve.is_file()
        assert "best-val step=" in curve.read_text()
        summary = json.loads((output_dir / "training_summary.json").read_text())
        assert summary["selected_checkpoint"] == str(best_path.resolve())
        assert summary["best_step"] == best["best_step"]

    e2 = outputs["E2"][1]
    e3 = outputs["E3"][1]
    assert e2["initial_head_sha256"] == e3["initial_head_sha256"]
    assert (
        e2["controlled_training_config_sha256"]
        == e3["controlled_training_config_sha256"]
    )
    assert e2["controlled_training_config"] == e3["controlled_training_config"]
    assert (
        e2["train_scientific_cache_contract"]
        == e3["train_scientific_cache_contract"]
    )
    assert (
        e2["val_scientific_cache_contract"]
        == e3["val_scientific_cache_contract"]
    )
    assert e2["proprio_mode"] != e3["proprio_mode"]
