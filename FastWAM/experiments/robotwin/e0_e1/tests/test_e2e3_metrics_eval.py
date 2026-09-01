from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from experiments.robotwin.e0_e1.cache import build_cache_payload, save_cache
from experiments.robotwin.e0_e1.evaluate_e2e3 import evaluate_e2e3_cache
from experiments.robotwin.e0_e1.head import ContrastiveContentHead
from experiments.robotwin.e0_e1.io_utils import (
    atomic_torch_save,
    file_identity,
    module_state_sha256,
)
from experiments.robotwin.e0_e1.metrics import (
    RepresentationRecord,
    compute_representation_metrics,
)
from experiments.robotwin.e0_e1.train_e2e3 import _controlled_training_config


SEEN = ("clean", "style_00_seed_0", "style_01_seed_1")
TEST = ("clean", "style_02_seed_2")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_inputs(variants: tuple[str, ...]) -> tuple[torch.Tensor, list[RepresentationRecord]]:
    embeddings: list[torch.Tensor] = []
    records: list[RepresentationRecord] = []
    for state in range(3):
        clean = torch.zeros(8)
        clean[state] = 1.0
        for view, variant in enumerate(variants):
            value = clean.clone()
            if view:
                value[4 + view] = 0.05 * view
            embeddings.append(F.normalize(value, dim=0))
            records.append(
                RepresentationRecord(
                    task="toy",
                    physical_state_id=f"toy/s{state}",
                    trajectory_id="toy/content_000040",
                    timestep=state * 10,
                    variant=variant,
                )
            )
    return torch.stack(embeddings), records


@pytest.mark.parametrize(
    ("variants", "styles", "expected_samples", "missing_style"),
    [
        (SEEN, ("r1", "r2"), 9, "r3"),
        (TEST, ("r3",), 6, "r1"),
    ],
)
def test_metrics_support_strict_seen_and_r3_protocols(
    variants: tuple[str, ...],
    styles: tuple[str, ...],
    expected_samples: int,
    missing_style: str,
) -> None:
    embeddings, records = _metric_inputs(variants)
    rows = compute_representation_metrics(
        embeddings,
        records,
        layer="video_block_08",
        experiment="strict",
        style_order=styles,
        state_negative_pairs=((0, len(variants)), (len(variants), 0)),
    )
    row = rows[0]
    assert row["num_samples"] == expected_samples
    assert row["evaluated_styles"] == list(styles)
    assert row[f"clean_{missing_style}_distance"] is None
    assert row[f"{missing_style}_to_clean_retrieval_at1"] is None
    if styles == ("r3",):
        assert row["style_distance"] == pytest.approx(row["clean_r3_distance"])
        assert row["retrieval_r1"] == pytest.approx(
            row["r3_to_clean_retrieval_at1"]
        )
        assert row["retrieval_r5"] == pytest.approx(
            row["r3_to_clean_retrieval_at5"]
        )


def _condition_provenance(mode: str, state: int) -> dict[str, object]:
    observed = hashlib.sha256(f"observed-{state}".encode()).hexdigest()
    effective = (
        observed
        if mode == "observed"
        else hashlib.sha256(b"constant-zero").hexdigest()
    )
    return {
        "normalized_proprio_sha256": effective,
        "context": {
            "proprio": {
                "mode": mode,
                "intervention_point": "post_normalizer_pre_proprio_encoder",
                "observed_normalized_sha256": observed,
                "effective_normalized_sha256": effective,
                "all_zero": mode == "constant_zero_normalized",
                "proprio_token_preserved": True,
            }
        }
    }


def _cache(tmp_path: Path, *, experiment: str, split: str) -> Path:
    mode = "observed" if experiment == "E2" else "constant_zero_normalized"
    variants = SEEN if split == "val" else TEST
    content = 30 if split == "val" else 40
    samples: list[dict[str, object]] = []
    token_rows: list[torch.Tensor] = []
    conditions: dict[str, dict[str, object]] = {}
    for state in range(3):
        key = f"toy/content_{content:06d}/frame_{state * 10:06d}"
        samples.append(
            {
                "physical_key": key,
                "task": "toy",
                "content_id": content,
                "frame_idx": state * 10,
                "trace_idx": state,
                "split": split,
                "variant_names": variants,
                "proprio_raw": torch.full((14,), float(state)),
                "physical_state_by_name": {"robot.q": float(state)},
                "visual_input_sha256": {
                    variant: {
                        "deployment_composite": hashlib.sha256(
                            f"{key}/{variant}/composite".encode()
                        ).hexdigest(),
                        "encoded_rgb_by_camera": {
                            camera: hashlib.sha256(
                                f"{key}/{variant}/{camera}".encode()
                            ).hexdigest()
                            for camera in (
                                "head_camera", "left_camera", "right_camera"
                            )
                        },
                    }
                    for variant in variants
                },
            }
        )
        conditions[key] = _condition_provenance(mode, state)
        base = torch.zeros(4, 16)
        base[:, state] = 1.0
        for variant_index in range(len(variants)):
            value = base.clone()
            value[:, 8 + variant_index] += 0.02 * variant_index
            token_rows.append(value)
    provenance = {
        "protocol": "r3_holdout_v1",
        "split": split,
        "active_variants": list(variants),
        "holdout_variant": "style_02_seed_2",
        "proprio_mode": mode,
        "backbone": {
            "checkpoint": "toy",
            "proprio_mode": mode,
        },
        "conditions_by_physical_state": conditions,
        "task_prompt_sha256": {"toy": "prompt"},
    }
    if split == "test":
        # Cache schema itself fail-closes before a final R3 artifact can be
        # written.  The real lock identity is bound immediately below by the
        # helper that creates the immutable lock.
        provenance["decision_lock_identity"] = {
            "path": str((tmp_path / "pending_lock.json").resolve()),
            "size_bytes": 1,
            "mtime_ns": 1,
            "sha256": "pending",
        }
        provenance["decision_lock_created_before_test"] = True
    payload = build_cache_payload(
        tokens_by_layer={8: torch.stack(token_rows)},
        samples=samples,
        provenance=provenance,
    )
    path = tmp_path / f"{experiment.lower()}_{split}.pt"
    save_cache(path, payload)
    return path


def _checkpoint(tmp_path: Path, *, experiment: str) -> tuple[Path, dict[str, object]]:
    mode = "observed" if experiment == "E2" else "constant_zero_normalized"
    torch.manual_seed(0)
    head = ContrastiveContentHead(
        backbone_dim=16, embed_dim=16, num_queries=2, num_heads=4
    )
    head_config = {
        "backbone_dim": 16,
        "embed_dim": 16,
        "num_queries": 2,
        "num_heads": 4,
    }
    initial_hash = module_state_sha256(head)
    controlled = _controlled_training_config(
        layer=8,
        head_config=head_config,
        steps=1,
        groups_per_batch=2,
        learning_rate=1e-4,
        weight_decay=1e-2,
        temperature=0.07,
        val_every=1,
        seed=0,
        min_temporal_gap=8,
        min_state_distance=1e-5,
    )
    controlled_hash = f"controlled-{experiment}"
    payload: dict[str, object] = {
        "schema_version": 2,
        "experiment": experiment,
        "protocol": "r3_holdout_v1",
        "proprio_mode": mode,
        "checkpoint_kind": "best_val",
        "layer": 8,
        "step": 1,
        "best_step": 1,
        "best_metric": {
            "metric": "val_contrastive_loss",
            "mode": "min",
            "tie_break": "earliest_step",
            "r3_used": False,
        },
        "head_config": head_config,
        "head": head.state_dict(),
        "controlled_training_config": controlled,
        "controlled_training_config_sha256": controlled_hash,
        "initial_head_sha256": initial_hash,
        "seed": 0,
    }
    path = tmp_path / f"{experiment.lower()}_best.pt"
    atomic_torch_save(path, payload)
    return path, payload


def _write_lock_and_bind_cache(
    tmp_path: Path,
    *,
    cache_path: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    experiment: str,
) -> Path:
    mode = "observed" if experiment == "E2" else "constant_zero_normalized"
    checkpoint_identity = {
        **file_identity(checkpoint_path),
        "sha256": _file_sha(checkpoint_path),
    }
    lock = {
        "schema_version": 1,
        "protocol": "r3_holdout_v1",
        "created_at_utc": "2026-08-13T00:00:00Z",
        "selected_layer": 8,
        "selection_identity": {"sha256": "selection"},
        "train_val_cache_identities": {"E2": {}, "E3": {}},
        "checkpoints": {
            experiment: {
                "checkpoint": checkpoint_identity,
                "experiment": experiment,
                "proprio_mode": mode,
                "best_step": 1,
                "controlled_training_config_sha256": checkpoint[
                    "controlled_training_config_sha256"
                ],
                "initial_head_sha256": checkpoint["initial_head_sha256"],
            }
        },
        "shared": {
            "controlled_training_config_sha256": checkpoint[
                "controlled_training_config_sha256"
            ],
            "initial_head_sha256": checkpoint["initial_head_sha256"],
        },
        "expected_test_outputs": {experiment: str(cache_path.resolve())},
    }
    lock_path = tmp_path / f"{experiment.lower()}_decision_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_identity = {
        **file_identity(lock_path),
        "sha256": _file_sha(lock_path),
    }
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    cache["provenance"]["decision_lock_identity"] = lock_identity
    cache["provenance"]["decision_lock_created_before_test"] = True
    # Test utility only: save_cache validates the updated cache structure.
    save_cache(cache_path, cache)
    return lock_path


def test_validation_e2_raw_uses_only_seen_styles(tmp_path: Path) -> None:
    cache = _cache(tmp_path, experiment="E2", split="val")
    rows = evaluate_e2e3_cache(
        cache_path=cache,
        layer=8,
        experiment="E2-RawBackbone",
        output_dir=tmp_path / "val_results",
        device="cpu",
    )
    assert rows[0]["evaluated_styles"] == ["r1", "r2"]
    payload = json.loads(
        (tmp_path / "val_results" / "e2_rawbackbone_layer_08.json").read_text()
    )
    assert payload["r3_used_for_selection"] is False
    assert payload["schema_version"] == 2
    assert payload["record_variants"] == list(SEEN)
    assert payload["metric_protocol"]["query"] == "mean(R1,R2)"


def test_final_r3_all_e2_controls_and_strict_lock(tmp_path: Path) -> None:
    cache = _cache(tmp_path, experiment="E2", split="test")
    checkpoint_path, checkpoint = _checkpoint(tmp_path, experiment="E2")
    lock_path = _write_lock_and_bind_cache(
        tmp_path,
        cache_path=cache,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        experiment="E2",
    )
    for label in ("E2-RawBackbone", "E2-InitHead", "E2-TrainedHead"):
        rows = evaluate_e2e3_cache(
            cache_path=cache,
            layer=8,
            experiment=label,
            output_dir=tmp_path / "test_results",
            head_checkpoint=checkpoint_path,
            decision_lock=lock_path,
            seed=0,
            device="cpu",
        )
        assert rows[0]["evaluated_styles"] == ["r3"]
        assert rows[0]["style_distance"] == pytest.approx(
            rows[0]["clean_r3_distance"]
        )
        assert rows[0]["retrieval_r1"] == pytest.approx(
            rows[0]["r3_to_clean_retrieval_at1"]
        )
        assert rows[0]["style_distance_R3"] == pytest.approx(
            rows[0]["clean_r3_distance"]
        )
        assert rows[0]["R3_to_Clean_R@1"] == pytest.approx(rows[0]["retrieval_r1"])

    with pytest.raises(ValueError, match="requires both"):
        evaluate_e2e3_cache(
            cache_path=cache,
            layer=8,
            experiment="E2-RawBackbone",
            output_dir=tmp_path / "invalid",
            head_checkpoint=checkpoint_path,
            device="cpu",
        )


def test_e3_requires_zero_proprio_provenance(tmp_path: Path) -> None:
    cache = _cache(tmp_path, experiment="E3", split="test")
    checkpoint_path, checkpoint = _checkpoint(tmp_path, experiment="E3")
    lock_path = _write_lock_and_bind_cache(
        tmp_path,
        cache_path=cache,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        experiment="E3",
    )
    rows = evaluate_e2e3_cache(
        cache_path=cache,
        layer=8,
        experiment="E3-NoProprio-RawBackbone",
        output_dir=tmp_path / "results",
        head_checkpoint=checkpoint_path,
        decision_lock=lock_path,
        device="cpu",
    )
    assert rows[0]["evaluated_styles"] == ["r3"]

    payload = torch.load(cache, map_location="cpu", weights_only=True)
    condition = next(iter(payload["provenance"]["conditions_by_physical_state"].values()))
    condition["context"]["proprio"]["all_zero"] = False
    bad_path = tmp_path / "bad_e3.pt"
    atomic_torch_save(bad_path, payload)
    with pytest.raises(ValueError, match="not proven exact zero|not zero"):
        evaluate_e2e3_cache(
            cache_path=bad_path,
            layer=8,
            experiment="E3-NoProprio-RawBackbone",
            output_dir=tmp_path / "invalid",
            head_checkpoint=checkpoint_path,
            decision_lock=lock_path,
            device="cpu",
        )
