from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.robotwin.e0_e1.head import (
    DEFAULT_PARAMETER_COUNT,
    ContrastiveContentHead,
    count_trainable_parameters,
    multi_positive_supcon_loss,
)
from experiments.robotwin.e0_e1.metrics import (
    RESULT_COLUMNS,
    RepresentationRecord,
    compute_representation_metrics,
    summarize_metric_rows,
)
from experiments.robotwin.e0_e1.negatives import build_state_negative_mask


def test_default_head_parameter_count_shape_and_normalization() -> None:
    head = ContrastiveContentHead()
    assert head.trainable_parameter_count() == DEFAULT_PARAMETER_COUNT == 2_070_144
    assert count_trainable_parameters(head) == DEFAULT_PARAMETER_COUNT

    visual_tokens = torch.randn(2, 5, 3072)
    embedding = head(visual_tokens)
    assert embedding.shape == (2, 384)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=-1),
        torch.ones(2),
        rtol=1e-5,
        atol=1e-5,
    )


def test_head_and_multi_positive_loss_forward_backward_are_finite() -> None:
    torch.manual_seed(7)
    head = ContrastiveContentHead(
        backbone_dim=16,
        embed_dim=32,
        num_queries=4,
        num_heads=4,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    visual_tokens = torch.randn(8, 6, 16)
    task_ids = ["task_a"] * 4 + ["task_b"] * 4
    physical_state_ids = ["a0", "a0", "a1", "a1", "b0", "b0", "b1", "b1"]

    before = head.content_queries.detach().clone()
    embeddings = head(visual_tokens)
    loss = multi_positive_supcon_loss(
        embeddings,
        physical_state_ids=physical_state_ids,
        task_ids=task_ids,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert sum(float(parameter.grad.norm().item()) for parameter in parameters) > 0.0
    optimizer.step()
    assert not torch.equal(before, head.content_queries.detach())


def test_multi_positive_loss_rejects_anchor_without_positive() -> None:
    embeddings = F.normalize(torch.randn(3, 8), dim=-1)
    with pytest.raises(ValueError, match="Every anchor must have"):
        multi_positive_supcon_loss(
            embeddings,
            physical_state_ids=[0, 1, 2],
            task_ids=["same-task"] * 3,
        )


def test_physical_negative_mask_filters_near_cross_task_and_obeys_priority() -> None:
    records = [
        {"task": "a", "physical_state_id": "a0", "trajectory_id": "ta", "timestep": 0},
        {"task": "a", "physical_state_id": "a_near", "trajectory_id": "ta", "timestep": 20},
        {"task": "a", "physical_state_id": "a_far", "trajectory_id": "ta", "timestep": 10},
        {"task": "a", "physical_state_id": "a_other", "trajectory_id": "tb", "timestep": 0},
        {"task": "b", "physical_state_id": "b0", "trajectory_id": "tc", "timestep": 0},
        {"task": "b", "physical_state_id": "b1", "trajectory_id": "td", "timestep": 0},
    ]
    states = [
        {"robot.q": 0.0},
        {"robot.q": 1e-7},  # temporally far, but physically too near
        {"robot.q": 1.0},
        {"robot.q": 2.0},
        {"robot.q": 3.0},
        {"robot.q": 4.0},
    ]
    mask = build_state_negative_mask(
        records, states, min_temporal_gap=8, min_state_distance=1e-5
    )

    assert mask.dtype == np.bool_
    assert mask[0, 2]  # same-trajectory far has priority
    assert not mask[0, 1]  # near-state filter
    assert not mask[0, 3]  # other trajectory excluded when far same-trajectory exists
    assert not mask[0, 4]  # cross-task excluded
    assert mask[3, 0] and mask[3, 2]  # no same-trajectory option: other trajectory
    assert mask[4, 5] and mask[5, 4]


def test_masked_supcon_keeps_positives_and_has_finite_backward() -> None:
    torch.manual_seed(13)
    raw = torch.randn(8, 12, requires_grad=True)
    embeddings = F.normalize(raw, dim=-1)
    state_ids = ["s0"] * 4 + ["s1"] * 4
    task_ids = ["task"] * 8
    negative_mask = torch.zeros(8, 8, dtype=torch.bool)
    negative_mask[:4, 4:] = True
    negative_mask[4:, :4] = True

    loss = multi_positive_supcon_loss(
        embeddings,
        state_ids,
        task_ids,
        negative_mask=negative_mask,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert float(raw.grad.norm().item()) > 0.0

    # Each anchor has cosine 1 with its positive and cosine 0 with its sole
    # selected negative.  The expected loss proves the positive remains in the
    # denominator even though the caller-supplied mask contains negatives only.
    exact_embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    exact_negative_mask = torch.tensor(
        [
            [False, False, True, False],
            [False, False, False, True],
            [True, False, False, False],
            [False, True, False, False],
        ]
    )
    exact_loss = multi_positive_supcon_loss(
        exact_embeddings,
        ["s0", "s0", "s1", "s1"],
        ["task"] * 4,
        temperature=0.5,
        negative_mask=exact_negative_mask,
    )
    assert exact_loss.item() == pytest.approx(math.log1p(math.exp(-2.0)))

    invalid_cross_task = negative_mask.clone()
    with pytest.raises(ValueError, match="cross-task"):
        multi_positive_supcon_loss(
            embeddings.detach(),
            state_ids,
            ["task"] * 4 + ["other"] * 4,
            negative_mask=invalid_cross_task,
        )


def test_toy_cross_domain_retrieval_and_summary() -> None:
    records: list[RepresentationRecord] = []
    embeddings: list[torch.Tensor] = []
    clean_indices: list[int] = []
    num_states = 6
    feature_dim = 8
    style_offsets = {
        "r1": (0.04, 6),
        "r2": (0.08, 7),
        "r3": (-0.12, 6),
    }

    for state_index in range(num_states):
        clean = torch.zeros(feature_dim)
        clean[state_index] = 1.0
        clean_indices.append(len(embeddings))
        embeddings.append(clean)
        records.append(
            RepresentationRecord(
                task="toy_task",
                physical_state_id=f"state_{state_index}",
                trajectory_id="trajectory_0",
                timestep=state_index * 10,
                variant="clean",
            )
        )
        for variant, (scale, offset_dim) in style_offsets.items():
            random_rendering = clean.clone()
            random_rendering[offset_dim] += scale
            embeddings.append(F.normalize(random_rendering, dim=0))
            records.append(
                RepresentationRecord(
                    task="toy_task",
                    physical_state_id=f"state_{state_index}",
                    trajectory_id="trajectory_0",
                    timestep=state_index * 10,
                    variant=variant,
                )
            )

    embedding_tensor = torch.stack(embeddings)
    state_negative_pairs = [
        (clean_indices[index], clean_indices[(index + 1) % num_states])
        for index in range(num_states)
    ]
    rows = compute_representation_metrics(
        embedding_tensor,
        records,
        layer="block_19",
        experiment="E0-RawBackbone",
        state_negative_pairs=state_negative_pairs,
    )

    assert len(rows) == 2
    task_row, average_row = rows
    assert task_row["task"] == "toy_task"
    assert average_row["task"] == "1-task-average"
    assert task_row["num_samples"] == num_states * 4
    assert task_row["num_physical_states"] == num_states
    assert task_row["retrieval_r1"] == pytest.approx(1.0)
    assert task_row["retrieval_r5"] == pytest.approx(1.0)
    for style in ("r1", "r2", "r3"):
        assert task_row[f"{style}_to_clean_retrieval_at1"] == pytest.approx(1.0)
        assert task_row[f"{style}_to_clean_retrieval_at5"] == pytest.approx(1.0)
    assert task_row["clean_r1_distance"] < task_row["clean_r2_distance"]
    assert task_row["clean_r2_distance"] < task_row["clean_r3_distance"]
    assert task_row["state_distance"] == pytest.approx(1.0)
    assert task_row["state_style_ratio"] > 10.0
    assert math.isfinite(task_row["state_style_ratio"])

    summary = summarize_metric_rows(rows)
    assert len(summary) == 2
    assert tuple(summary[0]) == RESULT_COLUMNS
    assert summary[0]["experiment"] == "E0-RawBackbone"


def test_cosine_roundoff_cannot_create_negative_distances() -> None:
    records: list[RepresentationRecord] = []
    values: list[torch.Tensor] = []
    # A high-dimensional normalized vector can have float32 self-dot slightly
    # above one.  Identical style renderings must still produce distance zero.
    clean = F.normalize(torch.full((3072,), 0.1, dtype=torch.float32), dim=0)
    for state in range(2):
        state_vector = clean.clone()
        if state:
            state_vector[0] *= -1
            state_vector = F.normalize(state_vector, dim=0)
        for variant in ("clean", "r1", "r2", "r3"):
            values.append(state_vector.clone())
            records.append(
                RepresentationRecord(
                    task="toy",
                    physical_state_id=f"s{state}",
                    trajectory_id="t0",
                    timestep=state * 10,
                    variant=variant,
                )
            )
    rows = compute_representation_metrics(
        torch.stack(values),
        records,
        layer="block",
        experiment="E0-RawBackbone",
        state_negative_pairs=((0, 4), (4, 0)),
    )
    assert rows[0]["style_distance"] >= 0.0
    assert rows[0]["state_distance"] >= 0.0
    assert rows[0]["state_style_ratio"] >= 0.0
