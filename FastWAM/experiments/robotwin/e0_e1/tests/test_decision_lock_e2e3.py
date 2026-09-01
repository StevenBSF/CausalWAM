from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.e0_e1.decision_lock_e2e3 import create_decision_lock
from experiments.robotwin.e0_e1.head import ContrastiveContentHead
from experiments.robotwin.e0_e1.io_utils import atomic_torch_save, file_identity, module_state_sha256


def _cache(path: Path) -> dict:
    atomic_torch_save(path, {
        "payload": torch.arange(4),
        "provenance": {
            "backbone": {"checkpoint": "toy", "capture_layers": [8]},
            "task_prompt_sha256": {"toy": "0" * 64},
        },
    })
    identity = file_identity(path)
    import hashlib
    identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def _checkpoint(
    path: Path,
    *,
    experiment: str,
    mode: str,
    train_identity: dict,
    val_identity: dict,
    initial_hash: str,
) -> Path:
    config = {
        "protocol": "r3_holdout_v1",
        "active_variants": ["clean", "style_00_seed_0", "style_01_seed_1"],
        "holdout_variant": "style_02_seed_2",
        "checkpoint_selection": {"r3_allowed": False},
    }
    atomic_torch_save(path, {
        "schema_version": 2,
        "experiment": experiment,
        "protocol": "r3_holdout_v1",
        "proprio_mode": mode,
        "checkpoint_kind": "best_val",
        "layer": 8,
        "step": 2,
        "best_step": 2,
        "best_metric": {
            "metric": "val_contrastive_loss", "mode": "min", "r3_used": False,
            "best_value": 1.0,
        },
        "controlled_training_config_sha256": "a" * 64,
        "initial_head_sha256": initial_hash,
        "controlled_training_config": config,
        "train_cache": train_identity["path"],
        "val_cache": val_identity["path"],
        "train_cache_identity": train_identity,
        "val_cache_identity": val_identity,
        "train_scientific_cache_contract": {
            "physical_state_ids": ["toy/content_000000/frame_000000"],
            "records_sha256": "c" * 64,
            "physical_states_sha256": "d" * 64,
            "proprio_raw_sha256": "e" * 64,
            "token_shapes_by_layer": {"8": [6, 4, 16]},
        },
        "val_scientific_cache_contract": {
            "physical_state_ids": ["toy/content_000030/frame_000000"],
            "records_sha256": "f" * 64,
            "physical_states_sha256": "0" * 64,
            "proprio_raw_sha256": "1" * 64,
            "token_shapes_by_layer": {"8": [6, 4, 16]},
        },
    })
    return path


def test_lock_is_created_only_before_test_outputs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    initial_hash = module_state_sha256(ContrastiveContentHead(backbone_dim=16))
    e2_train = _cache(tmp_path / "e2_train.pt")
    e2_val = _cache(tmp_path / "e2_val.pt")
    e3_train = _cache(tmp_path / "e3_train.pt")
    e3_val = _cache(tmp_path / "e3_val.pt")
    e2 = _checkpoint(
        tmp_path / "e2.pt", experiment="E2", mode="observed",
        train_identity=e2_train, val_identity=e2_val, initial_hash=initial_hash,
    )
    e3 = _checkpoint(
        tmp_path / "e3.pt", experiment="E3", mode="constant_zero_normalized",
        train_identity=e3_train, val_identity=e3_val, initial_hash=initial_hash,
    )
    selection = {
        "schema_version": 2,
        "protocol": "r3_holdout_v1",
        "evaluation_split": "val",
        "experiment": "E2-RawBackbone",
        "proprio_mode": "observed",
        "r3_used": False,
        "active_variants": ["clean", "style_00_seed_0", "style_01_seed_1"],
        "selected_layer": 8,
        "cache_identity": dict(e2_val),
        "candidates": [{"layer": layer} for layer in (8, 16, 24)],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    lock = create_decision_lock(
        selection_path=selection_path,
        e2_checkpoint=e2,
        e3_checkpoint=e3,
        e2_test_output=tmp_path / "e2_test.pt",
        e3_test_output=tmp_path / "e3_test.pt",
        output_path=tmp_path / "lock.json",
    )
    payload = json.loads(lock.read_text())
    assert payload["r3_access_before_lock"] is False
    assert set(payload["checkpoints"]) == {"E2", "E3"}
    assert payload["shared"]["initial_head_sha256"] == initial_hash

    (tmp_path / "e2_test.pt").touch()
    with pytest.raises(FileExistsError, match="already exists"):
        create_decision_lock(
            selection_path=selection_path,
            e2_checkpoint=e2,
            e3_checkpoint=e3,
            e2_test_output=tmp_path / "e2_test.pt",
            e3_test_output=tmp_path / "other_test.pt",
            output_path=tmp_path / "other_lock.json",
        )
