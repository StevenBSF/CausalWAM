from __future__ import annotations

import ast
from pathlib import Path


def test_motus_source_exposes_and_uses_optional_content_extension() -> None:
    root = Path(__file__).parents[4]
    source_path = root / "models" / "motus.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    motus = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Motus"
    )
    methods = {
        node.name: node
        for node in motus.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "set_policy_content_conditioner" in methods
    assert "_inject_policy_content_tokens" in methods
    training_source = ast.get_source_segment(
        source, methods["training_step"]
    )
    inference_source = ast.get_source_segment(
        source, methods["inference_step"]
    )
    assert training_source is not None
    assert inference_source is not None
    assert "_inject_policy_content_tokens" in training_source
    assert "_inject_policy_content_tokens" in inference_source
    assert "compute_video_loss" in training_source


def test_deployment_uses_package_relative_model_and_processor_imports() -> None:
    root = Path(__file__).parents[4]
    deployment = (root / "inference/robotwin/Motus/deploy_policy.py").read_text()
    inference_model = (
        root / "inference/robotwin/Motus/models/motus.py"
    ).read_text()
    assert "from .models.motus import" in deployment
    assert "from .qwen_processor import" in deployment
    assert "from ..utils.common import" in inference_model


def test_layer_feature_helpers_use_native_bf16_autocast() -> None:
    root = Path(__file__).parents[4]
    for path in (
        root / "models/wan_model.py",
        root / "inference/robotwin/Motus/models/wan_model.py",
    ):
        source = path.read_text()
        assert "with torch.autocast(" in source
        assert 'enabled=device.type == "cuda"' in source
