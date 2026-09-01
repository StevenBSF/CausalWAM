from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from types import SimpleNamespace

from experiments.robotwin.policy_content_adapter.model import (
    MotusPolicyContentConditioner,
)


def _deployment_module():
    path = (
        Path(__file__).parents[4]
        / "inference"
        / "robotwin"
        / "Motus"
        / "policy_content_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("motus_deploy_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_and_deployment_adapter_state_dicts_are_strictly_compatible() -> None:
    deployment = _deployment_module()
    train = MotusPolicyContentConditioner()
    deploy = deployment.MotusPolicyContentConditioner()
    train_state, deploy_state = train.state_dict(), deploy.state_dict()
    assert tuple(train_state) == tuple(deploy_state)
    assert {
        key: (tuple(value.shape), value.dtype) for key, value in train_state.items()
    } == {key: (tuple(value.shape), value.dtype) for key, value in deploy_state.items()}
    deploy.load_state_dict(train_state, strict=True)


def test_deployment_zero_gate_is_bit_exact() -> None:
    deployment = _deployment_module()
    conditioner = deployment.MotusPolicyContentConditioner()
    action = torch.randn(2, 5, 1024)
    content = torch.randn(2, 8, 384)
    output = conditioner.inject_action_tokens(action, content)
    assert conditioner.adapter.gate.detach().item() == 0.0
    assert torch.equal(output, action)


def test_deployment_observation_latent_matches_wan_dtype() -> None:
    deployment = _deployment_module()

    class Video:
        wan_model = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
        )

        def encode_video(self, value):
            return torch.ones(value.shape[0], 48, 1, 2, 2)

        def get_layer_features(self, latent, timestep, text, **kwargs):
            assert latent.dtype == torch.bfloat16
            return [torch.ones(latent.shape[0], 3, 3072, dtype=latent.dtype)]

    model = SimpleNamespace(
        video_model=Video(), device=torch.device("cpu"), dtype=torch.bfloat16
    )
    tokens = deployment.extract_observation_visual_tokens(
        model,
        first_frame=torch.rand(1, 3, 8, 8),
        language_embeddings=[torch.rand(2, 3)],
        capture_layer=16,
    )
    assert tokens.dtype == torch.bfloat16
