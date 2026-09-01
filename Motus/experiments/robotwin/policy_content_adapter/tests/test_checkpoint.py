from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.robotwin.policy_content_adapter.checkpoint import (
    CheckpointError,
    ResumeIdentity,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _identity() -> ResumeIdentity:
    return ResumeIdentity(
        control="m3_ours",
        regime="m_p2",
        training_seed=1,
        world_size=1,
        config_sha256="1" * 64,
        base_lineage_sha256="2" * 64,
        paired_manifest_sha256="3" * 64,
        official_manifest_sha256="4" * 64,
    )


def test_checkpoint_round_trip_restores_model_optimizer_scheduler(tmp_path: Path) -> None:
    torch.manual_seed(1)
    conditioner = nn.Linear(3, 2)
    action = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        list(conditioner.parameters()) + list(action.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    (conditioner(torch.randn(2, 3)).sum() + action(torch.randn(2, 2)).sum()).backward()
    optimizer.step()
    scheduler.step()
    expected_conditioner = {
        key: value.detach().clone() for key, value in conditioner.state_dict().items()
    }
    expected_action = {
        key: value.detach().clone() for key, value in action.state_dict().items()
    }
    root = tmp_path / "step_12"
    save_training_checkpoint(
        root,
        conditioner=conditioner,
        action_expert=action,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=12,
        epoch=2,
        identity=_identity(),
        official_sampler_state={"position": 9},
        paired_sampler_state={"position": 5},
    )
    with torch.no_grad():
        for parameter in conditioner.parameters():
            parameter.zero_()
        for parameter in action.parameters():
            parameter.zero_()
    result = load_training_checkpoint(
        root,
        conditioner=conditioner,
        action_expert=action,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_identity=_identity(),
        restore_rng=False,
    )
    assert result["global_step"] == 12 and result["epoch"] == 2
    assert result["official_sampler_state"] == {"position": 9}
    for key, value in conditioner.state_dict().items():
        torch.testing.assert_close(value, expected_conditioner[key])
    for key, value in action.state_dict().items():
        torch.testing.assert_close(value, expected_action[key])


def test_checkpoint_rejects_identity_drift_and_overwrite(tmp_path: Path) -> None:
    conditioner = nn.Linear(3, 2)
    action = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        list(conditioner.parameters()) + list(action.parameters()), lr=1e-3
    )
    root = tmp_path / "step_0"
    kwargs = dict(
        conditioner=conditioner,
        action_expert=action,
        optimizer=optimizer,
        scheduler=None,
        global_step=0,
        epoch=0,
        identity=_identity(),
        official_sampler_state={},
        paired_sampler_state={},
    )
    save_training_checkpoint(root, **kwargs)
    with pytest.raises(FileExistsError):
        save_training_checkpoint(root, **kwargs)
    with pytest.raises(CheckpointError, match="identity"):
        load_training_checkpoint(
            root,
            conditioner=conditioner,
            action_expert=action,
            optimizer=optimizer,
            scheduler=None,
            expected_identity=replace(_identity(), training_seed=2),
            restore_rng=False,
        )

