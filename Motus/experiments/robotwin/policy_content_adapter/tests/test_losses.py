from __future__ import annotations

import pytest
import torch

from experiments.robotwin.policy_content_adapter.losses import (
    multi_positive_supcon_loss,
)


def test_four_view_multi_positive_loss_is_finite_and_differentiable() -> None:
    embeddings = torch.randn(16, 8, requires_grad=True)
    # Two physical states per task, each represented by four scene variants.
    state_ids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
    task_ids = ["task0"] * 8 + ["task1"] * 8
    loss = multi_positive_supcon_loss(embeddings, state_ids, task_ids)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_loss_rejects_missing_same_task_negative() -> None:
    with pytest.raises(ValueError, match="different state"):
        multi_positive_supcon_loss(
            torch.randn(4, 8), ["same"] * 4, ["task"] * 4
        )


def test_loss_excludes_cross_task_samples_from_negatives() -> None:
    with pytest.raises(ValueError, match="different state"):
        multi_positive_supcon_loss(
            torch.randn(8, 8),
            ["a"] * 4 + ["b"] * 4,
            ["task0"] * 4 + ["task1"] * 4,
        )

