from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.robotwin.e0_e1.cache import build_cache_payload, load_cache, save_cache
from experiments.robotwin.e0_e1.compare import compare_results
from experiments.robotwin.e0_e1.evaluate import evaluate_cache
from experiments.robotwin.e0_e1.head import ContrastiveContentHead
from experiments.robotwin.e0_e1.train_e1 import (
    _physical_state_negative_mask,
    train_head,
)


def _cache(tmp_path: Path, split: str, states: int = 3) -> Path:
    samples = []
    token_rows = []
    content_id = {"train": 0, "val": 30, "test": 40}[split]
    for state_index in range(states):
        samples.append(
            {
                "physical_key": (
                    f"toy/content_{content_id:06d}/frame_{state_index:06d}"
                ),
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
                "physical_state_by_name": {"robot.q": float(state_index)},
            }
        )
        base = torch.zeros(4, 16)
        base[:, state_index] = 1.0
        for variant_index in range(4):
            tokens = base.clone()
            tokens[:, 8 + variant_index] += 0.02 * variant_index
            token_rows.append(tokens)
    payload = build_cache_payload(
        tokens_by_layer={8: torch.stack(token_rows)},
        samples=samples,
        provenance={
            "kind": "synthetic-test",
            "backbone": {"checkpoint": "toy"},
            "task_prompt_sha256": {"toy": "toy-prompt"},
        },
    )
    path = tmp_path / f"{split}.pt"
    save_cache(path, payload)
    return path


def test_cache_evaluate_train_and_compare_cpu(tmp_path: Path) -> None:
    train_cache = _cache(tmp_path, "train")
    val_cache = _cache(tmp_path, "val")
    test_cache = _cache(tmp_path, "test")
    assert load_cache(train_cache)["tokens_by_layer"]["8"].shape == (12, 4, 16)

    selection_dir = tmp_path / "selection_metrics"
    validation_e0 = evaluate_cache(
        cache_path=val_cache,
        layer=8,
        experiment="E0-RawBackbone",
        output_dir=selection_dir,
        device="cpu",
    )
    assert validation_e0[0]["retrieval_r1"] == 1.0
    validation_payload = json.loads(
        (selection_dir / "e0_rawbackbone_layer_08.json").read_text()
    )
    assert validation_payload["evaluation_split"] == "val"

    result_dir = tmp_path / "results"
    e0 = evaluate_cache(
        cache_path=test_cache,
        layer=8,
        experiment="E0-RawBackbone",
        output_dir=result_dir,
        device="cpu",
    )
    init = evaluate_cache(
        cache_path=test_cache,
        layer=8,
        experiment="E1-InitHead",
        output_dir=result_dir,
        device="cpu",
    )
    assert e0[0]["retrieval_r1"] == 1.0
    assert torch.isfinite(torch.tensor(init[0]["style_distance"]))
    assert json.loads(
        (result_dir / "e0_rawbackbone_layer_08.json").read_text()
    )["evaluation_split"] == "test"

    checkpoint = train_head(
        train_cache_path=train_cache,
        val_cache_path=val_cache,
        layer=8,
        output_dir=tmp_path / "train",
        steps=1,
        groups_per_batch=2,
        val_every=1,
        device="cpu",
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint_payload["negative_filter"] == {
        "min_temporal_gap": 8,
        "min_state_distance": 1e-5,
    }
    assert checkpoint_payload["train_cache_identity"]["path"] == str(
        train_cache.resolve()
    )
    assert checkpoint_payload["val_cache_identity"]["path"] == str(val_cache.resolve())
    trained = evaluate_cache(
        cache_path=test_cache,
        layer=8,
        experiment="E1-TrainedHead",
        output_dir=result_dir,
        head_checkpoint=checkpoint,
        device="cpu",
    )
    assert torch.isfinite(torch.tensor(trained[0]["state_style_ratio"]))
    summary = compare_results(
        [
            result_dir / "e0_rawbackbone_layer_08.json",
            result_dir / "e1_inithead_layer_08.json",
            result_dir / "e1_trainedhead_layer_08.json",
        ],
        tmp_path / "comparison",
    )
    assert len(summary) == 6  # task + macro-average, each with three experiments
    assert (tmp_path / "comparison/summary.md").is_file()
    comparison_payload = json.loads(
        (tmp_path / "comparison/comparison.json").read_text()
    )
    if not comparison_payload["overall_success"]:
        with pytest.raises(RuntimeError, match="success gate failed"):
            compare_results(
                [
                    result_dir / "e0_rawbackbone_layer_08.json",
                    result_dir / "e1_inithead_layer_08.json",
                    result_dir / "e1_trainedhead_layer_08.json",
                ],
                tmp_path / "required_comparison",
                require_success=True,
            )

    # Init and trained metrics must prove they derive from the exact same
    # random initialization; a relabeled or unrelated control is rejected.
    init_json = result_dir / "e1_inithead_layer_08.json"
    payload = json.loads(init_json.read_text())
    payload["head"]["initial_head_sha256"] = "not-the-training-init"
    bad_init = tmp_path / "bad_init.json"
    bad_init.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="initialization used for training"):
        compare_results(
            [
                result_dir / "e0_rawbackbone_layer_08.json",
                bad_init,
                result_dir / "e1_trainedhead_layer_08.json",
            ],
            tmp_path / "bad_comparison",
        )


def test_evaluate_rejects_train_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation or held-out test"):
        evaluate_cache(
            cache_path=_cache(tmp_path, "train"),
            layer=8,
            experiment="E0-RawBackbone",
            output_dir=tmp_path / "metrics",
            device="cpu",
        )


def test_training_negative_mask_expands_each_four_render_group(tmp_path: Path) -> None:
    cache = load_cache(_cache(tmp_path, "train", states=2))
    mask = _physical_state_negative_mask(
        cache,
        list(range(8)),
        min_temporal_gap=8,
        min_state_distance=1e-5,
    )
    assert mask.shape == (8, 8)
    assert torch.equal(mask[:4, :4], torch.zeros(4, 4, dtype=torch.bool))
    assert torch.equal(mask[4:, 4:], torch.zeros(4, 4, dtype=torch.bool))
    assert bool(mask[:4, 4:].all())
    assert bool(mask[4:, :4].all())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("task", "task sets differ"),
        ("backbone", "backbone provenance differs"),
        ("prompt", "missing task_prompt_sha256"),
    ),
)
def test_training_rejects_incompatible_cache_provenance(
    tmp_path: Path, mutation: str, message: str
) -> None:
    train_cache = _cache(tmp_path, "train")
    val_cache = _cache(tmp_path, "val")
    payload = load_cache(val_cache)
    if mutation == "task":
        for record in payload["records"]:
            record["task"] = "other-task"
    elif mutation == "backbone":
        payload["provenance"]["backbone"] = {"checkpoint": "other"}
    else:
        del payload["provenance"]["task_prompt_sha256"]
    save_cache(val_cache, payload)
    with pytest.raises(ValueError, match=message):
        train_head(
            train_cache_path=train_cache,
            val_cache_path=val_cache,
            layer=8,
            output_dir=tmp_path / "incompatible",
            steps=1,
            groups_per_batch=2,
            val_every=1,
            device="cpu",
        )


def test_training_accepts_audit_only_native_prefill_difference(tmp_path: Path) -> None:
    train_cache = _cache(tmp_path, "train")
    val_cache = _cache(tmp_path, "val")
    train_payload = load_cache(train_cache)
    val_payload = load_cache(val_cache)
    train_payload["provenance"]["backbone"]["native_prefill_verified"] = True
    val_payload["provenance"]["backbone"]["native_prefill_verified"] = False
    save_cache(train_cache, train_payload)
    save_cache(val_cache, val_payload)
    checkpoint = train_head(
        train_cache_path=train_cache,
        val_cache_path=val_cache,
        layer=8,
        output_dir=tmp_path / "audit-only-difference",
        steps=1,
        groups_per_batch=2,
        val_every=1,
        device="cpu",
    )
    assert checkpoint.is_file()


def test_final_comparison_rejects_validation_metrics(tmp_path: Path) -> None:
    train_cache = _cache(tmp_path, "train")
    val_cache = _cache(tmp_path, "val")
    checkpoint = train_head(
        train_cache_path=train_cache,
        val_cache_path=val_cache,
        layer=8,
        output_dir=tmp_path / "train",
        steps=1,
        groups_per_batch=2,
        val_every=1,
        device="cpu",
    )
    result_dir = tmp_path / "val_metrics"
    evaluate_cache(
        cache_path=val_cache,
        layer=8,
        experiment="E0-RawBackbone",
        output_dir=result_dir,
        device="cpu",
    )
    evaluate_cache(
        cache_path=val_cache,
        layer=8,
        experiment="E1-InitHead",
        output_dir=result_dir,
        device="cpu",
    )
    evaluate_cache(
        cache_path=val_cache,
        layer=8,
        experiment="E1-TrainedHead",
        output_dir=result_dir,
        head_checkpoint=checkpoint,
        device="cpu",
    )
    with pytest.raises(ValueError, match="held-out test split"):
        compare_results(
            [
                result_dir / "e0_rawbackbone_layer_08.json",
                result_dir / "e1_inithead_layer_08.json",
                result_dir / "e1_trainedhead_layer_08.json",
            ],
            tmp_path / "comparison",
        )


def test_frozen_backbone_gradient_contract_is_explicit() -> None:
    backbone = torch.nn.Linear(6, 16)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    head = ContrastiveContentHead(backbone_dim=16, embed_dim=32, num_queries=4, num_heads=4)
    observations = torch.randn(4, 5, 6)
    with torch.no_grad():
        tokens = backbone(observations).detach()
    embeddings = head(tokens)
    loss = (embeddings[0] - embeddings[1]).square().sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_training_rejects_physical_trajectory_leakage(tmp_path: Path) -> None:
    train_cache = _cache(tmp_path, "train")
    val_cache = _cache(tmp_path, "val")
    payload = load_cache(val_cache)
    for record in payload["records"]:
        record["trajectory_id"] = "toy/content_000000"
        record["physical_state_id"] = record["physical_state_id"].replace(
            "content_000030", "content_000000"
        )
    save_cache(val_cache, payload)
    with pytest.raises(ValueError, match="physical-trajectory leakage"):
        train_head(
            train_cache_path=train_cache,
            val_cache_path=val_cache,
            layer=8,
            output_dir=tmp_path / "leaked",
            steps=1,
            groups_per_batch=2,
            val_every=1,
            device="cpu",
        )
