from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments.robotwin.policy_content_adapter.materialize import (
    materialize_pair,
)
CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def _file(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_materializer_resolves_artifacts_and_mp2_pair(tmp_path: Path) -> None:
    files = {
        name: _file(tmp_path / f"{name}.json")
        for name in (
            "base_lineage",
            "implementation_audit",
            "strict_load_audit",
            "zero_gate_audit",
            "official_manifest",
            "paired_manifest",
        )
    }
    token = tmp_path / "tokens"
    text = tmp_path / "text"
    _file(token / "manifest.json")
    _file(text / "audit.json")
    destination = tmp_path / "configs"
    result = materialize_pair(
        m1_template=CONFIG_ROOT / "m1_m_p1_smoke.yaml",
        m3_template=CONFIG_ROOT / "m3_m_p1_smoke.yaml",
        output_dir=destination,
        run_output_root=tmp_path / "runs",
        base_lineage=files["base_lineage"],
        implementation_audit=files["implementation_audit"],
        strict_load_audit=files["strict_load_audit"],
        zero_gate_audit=files["zero_gate_audit"],
        official_manifest=files["official_manifest"],
        paired_manifest=files["paired_manifest"],
        token_cache=token,
        task_text_cache=text,
        regime="m_p2",
        training_seed=3,
        world_size=2,
        per_device_batch=1,
        paired_groups_per_device=2,
        gradient_accumulation_steps=4,
        max_steps=5,
        checkpoint_interval=2,
    )
    assert result["status"] == "PASS" and result["global_batch"] == 8
    assert (destination / "m1.yaml").is_file()
    assert (destination / "m3.yaml").is_file()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["regime"] == "m_p2"


def test_materializer_locks_motus_five_epoch_profile(tmp_path: Path) -> None:
    files = {
        name: _file(tmp_path / f"{name}.json")
        for name in (
            "base_lineage",
            "implementation_audit",
            "strict_load_audit",
            "zero_gate_audit",
            "official_manifest",
            "paired_manifest",
        )
    }
    token = tmp_path / "tokens"
    text = tmp_path / "text"
    _file(token / "manifest.json")
    _file(text / "audit.json")
    destination = tmp_path / "formal_configs"
    result = materialize_pair(
        m1_template=CONFIG_ROOT / "m1_m_p1_smoke.yaml",
        m3_template=CONFIG_ROOT / "m3_m_p1_smoke.yaml",
        output_dir=destination,
        run_output_root=tmp_path / "formal_runs",
        base_lineage=files["base_lineage"],
        implementation_audit=files["implementation_audit"],
        strict_load_audit=files["strict_load_audit"],
        zero_gate_audit=files["zero_gate_audit"],
        official_manifest=files["official_manifest"],
        paired_manifest=files["paired_manifest"],
        token_cache=token,
        task_text_cache=text,
        regime="m_p2",
        training_seed=1,
        world_size=8,
        per_device_batch=8,
        paired_groups_per_device=2,
        gradient_accumulation_steps=1,
        max_steps=1_285,
        checkpoint_interval=257,
        profile="motus_author_5epoch_v1",
        head_adapter_lr=5.0e-5,
        action_expert_lr=5.0e-5,
    )
    assert result["profile"] == "motus_author_5epoch_v1"
    assert result["epochs"] == 5
    assert result["steps_per_epoch"] == 257
    assert result["max_steps"] == 1_285
    config = yaml.safe_load(
        (destination / "m3.yaml").read_text(encoding="utf-8")
    )
    assert config["training"]["global_batch"] == 64
    assert config["training"]["scheduler"] == "motus_author_linear"
    assert config["training"]["num_workers"] == 16
