"""Fail-closed M1/M3 configuration and fairness audit."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .paired_data import sha256_file
from .protocol import PROTOCOL_ID, TASKS, validate_control


CONFIG_SCHEMA = "motus_policy_content_adapter_run_config"
CONFIG_VERSION = 1
FORMAL_PROFILE = "motus_author_5epoch_v1"
AUTHOR_SMOKE_PROFILE = "motus_author_batch8_smoke_v1"


class ConfigAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigAuditError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _artifact(identity: Any, *, name: str, require_exists: bool) -> None:
    identity = _mapping(identity, name)
    path_text = str(identity.get("path", ""))
    sha = str(identity.get("sha256", ""))
    size = int(identity.get("size_bytes", -1))
    _require(path_text and not path_text.startswith("__"), f"{name} path is unresolved")
    _require(len(sha) == 64 and not sha.startswith("__"), f"{name} SHA is unresolved")
    _require(size >= 0, f"{name} size is invalid")
    if require_exists:
        path = Path(path_text)
        _require(path.is_file() or path.is_dir(), f"{name} path is missing: {path}")
        if path.is_file():
            _require(path.stat().st_size == size, f"{name} size changed")
            _require(sha256_file(path) == sha, f"{name} SHA changed")
        else:
            identity_name = str(identity.get("identity_file", ""))
            _require(identity_name and "/" not in identity_name, f"{name} directory identity_file is invalid")
            identity_path = path / identity_name
            _require(identity_path.is_file(), f"{name} identity file is missing")
            _require(identity_path.stat().st_size == size, f"{name} identity size changed")
            _require(sha256_file(identity_path) == sha, f"{name} identity SHA changed")


def validate_run_config(
    config: Mapping[str, Any], *, require_runnable: bool = False
) -> dict[str, Any]:
    _require(config.get("schema") == CONFIG_SCHEMA, "run config schema changed")
    _require(config.get("schema_version") == CONFIG_VERSION, "run config version changed")
    _require(config.get("protocol_id") == PROTOCOL_ID, "run protocol changed")
    runnable = config.get("runnable") is True
    if require_runnable:
        _require(runnable, "run config is not execution-ready")
    control = str(config.get("control", ""))
    objective = _mapping(config.get("objective"), "objective")
    validate_control(
        control=control,
        lambda_contrastive=float(objective.get("lambda_contrastive", -1)),
    )
    _require(objective.get("official_loss") == "action_flow_matching_only", "official stream must be action-only")
    _require(objective.get("paired_loss") == "same_task_four_view_supcon_only", "paired stream must be contrastive-only")
    _require(float(objective.get("temperature", 0)) > 0, "temperature must be positive")

    model = _mapping(config.get("model"), "model")
    regime = str(model.get("regime", ""))
    _require(regime in {"m_p1", "m_p2"}, "model regime is invalid")
    freeze = _mapping(model.get("freeze"), "model.freeze")
    _require(freeze.get("wan") is True and freeze.get("vae") is True, "WAN/VAE must be frozen")
    _require(freeze.get("vlm") is True and freeze.get("understanding_expert") is True, "VLM/Understanding must be frozen")
    _require(freeze.get("content_head_gca") is False, "Head/GCA must be trainable")
    _require(
        freeze.get("action_expert") is (regime == "m_p1"),
        "Action Expert freeze flag disagrees with regime",
    )
    architecture = _mapping(model.get("adapter"), "model.adapter")
    _require(architecture.get("capture_branch") == "current_observation_frozen_video_only_wan_t0", "content branch changed")
    _require(int(architecture.get("capture_layer", 0)) > 0, "capture layer is invalid")
    _require(architecture.get("content_queries") == 8 and architecture.get("content_dim") == 384, "Content Head architecture changed")
    _require(architecture.get("action_dim") == 1024 and architecture.get("zero_gate") is True, "GCA architecture changed")

    data = _mapping(config.get("data"), "data")
    _require(tuple(data.get("tasks", ())) == TASKS, "task order changed")
    official = _mapping(data.get("official"), "data.official")
    paired = _mapping(data.get("paired"), "data.paired")
    _require(official.get("episodes") == 1650, "official episode count changed")
    _require(official.get("action_chunk") == 16 and official.get("action_hz") == 10, "official temporal contract changed")
    _require(official.get("interpolation") is False, "official interpolation is forbidden")
    _require(official.get("action_values") == "raw_joint_action_vector", "official action value contract changed")
    _require(paired.get("physical_states") == 720 and paired.get("views_per_state") == 4, "paired counts changed")
    _require(paired.get("action_supervision") is False, "paired data may not provide action loss")

    training = _mapping(config.get("training"), "training")
    profile = str(training.get("profile", "engineering_smoke"))
    _require(
        profile in {
            "engineering_smoke",
            AUTHOR_SMOKE_PROFILE,
            FORMAL_PROFILE,
        },
        "training profile is invalid",
    )
    seed = int(training.get("seed", -1))
    _require(seed >= 0, "training seed must be non-negative")
    world = int(training.get("world_size", 0))
    micro = int(training.get("per_device_batch", 0))
    paired_groups = int(training.get("paired_groups_per_device", 0))
    accum = int(training.get("gradient_accumulation_steps", 0))
    global_batch = int(training.get("global_batch", 0))
    _require(world > 0 and micro > 0 and accum > 0, "training batch settings must be positive")
    _require(
        paired_groups >= 2,
        "paired_groups_per_device must provide a same-task negative",
    )
    _require(global_batch == world * micro * accum, "global batch formula changed")
    _require(int(training.get("max_steps", 0)) > 0, "max_steps must be positive")
    _require(int(training.get("checkpoint_interval", 0)) > 0, "checkpoint interval must be positive")
    _require(float(training.get("head_adapter_lr", 0)) > 0, "Head/GCA LR must be positive")
    action_lr = training.get("action_expert_lr")
    if regime == "m_p2":
        _require(action_lr is not None and float(action_lr) > 0, "M-P2 Action Expert LR is invalid")
    else:
        _require(action_lr is None, "M-P1 must not define an Action Expert LR")
    _require(training.get("mixed_precision") == "bf16", "training precision must be BF16")
    _require(float(training.get("weight_decay", -1)) == 0.01, "AdamW weight decay changed")
    _require(float(training.get("grad_clip_norm", -1)) == 0.5, "gradient clipping changed")
    scheduler = str(training.get("scheduler", ""))
    _require(
        scheduler in {"constant", "motus_author_linear"},
        "scheduler contract is invalid",
    )
    if profile in {AUTHOR_SMOKE_PROFILE, FORMAL_PROFILE}:
        _require(regime == "m_p2", "author-aligned profiles are defined for M-P2")
        _require(world == 8 and micro == 8 and accum == 1, "formal Motus batch contract changed")
        _require(global_batch == 64, "formal Motus global batch must be 64")
        _require(int(official.get("virtual_samples_per_epoch", 0)) == 16_500, "official virtual epoch changed")
        per_rank = math.ceil(16_500 / world)
        expected_steps = per_rank // micro
        _require(expected_steps == 257, "formal author-style epoch derivation changed")
        _require(int(training.get("steps_per_epoch", 0)) == expected_steps, "formal steps_per_epoch changed")
        _require(int(training.get("samples_per_epoch", 0)) == expected_steps * global_batch, "formal samples_per_epoch changed")
        expected_exposures = int(training["max_steps"]) * global_batch / 16_500
        _require(
            math.isclose(
                float(training.get("effective_dataset_exposures", -1)),
                expected_exposures,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            "formal exposure accounting changed",
        )
        _require(int(training.get("author_save_interval_reference", 0)) == 5000, "author save reference changed")
        _require(training.get("official_sampler") == "motus_distributed_drop_last_epoch_v1", "formal sampler changed")
        _require(training.get("drop_last") is True, "formal drop_last changed")
        _require(training.get("optimizer") == "adamw", "formal optimizer changed")
        _require(list(training.get("betas", ())) == [0.9, 0.95], "formal AdamW betas changed")
        _require(float(training.get("head_adapter_lr", 0)) == 5.0e-5, "formal Head/GCA LR changed")
        _require(float(action_lr) == 5.0e-5, "formal Action Expert LR changed")
        _require(scheduler == "motus_author_linear", "formal scheduler must match Motus")
        _require(int(training.get("warmup_steps", -1)) == 200, "formal warmup changed")
        _require(int(training.get("cycle_length", -1)) == 5_000_000, "formal scheduler cycle changed")
        _require(float(training.get("f_max", -1)) == 0.99, "formal f_max changed")
        _require(float(training.get("f_min", -1)) == 0.4, "formal f_min changed")
        _require(float(training.get("f_start", -1)) == 1.0e-6, "formal f_start changed")
        _require(int(training.get("num_workers", -1)) == 16, "formal worker count changed")
    if profile == FORMAL_PROFILE:
        _require(int(training.get("epochs", 0)) == 5, "formal run must use five epochs")
        _require(int(training.get("max_steps", 0)) == 5 * expected_steps, "formal max_steps changed")
        _require(int(training.get("checkpoint_interval", 0)) == expected_steps, "formal checkpoint cadence must be one epoch")
        _require(training.get("checkpoint_policy") == "every_author_style_epoch_for_exact_resume", "formal checkpoint policy changed")
    sequence = _mapping(training.get("sequence_contract"), "training.sequence_contract")
    _require(sequence.get("official_rng_stream") == "official", "official RNG stream changed")
    _require(sequence.get("paired_rng_stream") == "paired", "paired RNG stream changed")
    _require(sequence.get("matched_m1_m3") is True, "M1/M3 sequence matching is disabled")

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    if runnable or require_runnable:
        for name in (
            "base_lineage_manifest",
            "implementation_audit",
            "strict_load_audit",
            "zero_gate_audit",
            "official_manifest",
            "paired_manifest",
            "frozen_token_cache",
            "task_text_cache",
        ):
            _artifact(artifacts.get(name), name=name, require_exists=True)
    output = str(config.get("output_dir", ""))
    _require(output and not output.startswith("__"), "output_dir is unresolved")
    if require_runnable:
        _require(not Path(output).exists(), "formal output_dir already exists")
    return {
        "status": "PASS",
        "runnable": runnable,
        "control": control,
        "regime": regime,
        "training_seed": seed,
        "global_batch": global_batch,
        "training_profile": profile,
    }


def _fairness_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(config))
    projected.pop("control", None)
    projected.pop("output_dir", None)
    projected.pop("config_id", None)
    objective = dict(projected.get("objective", {}))
    objective.pop("lambda_contrastive", None)
    projected["objective"] = objective
    return projected


def validate_m1_m3_pair(
    m1: Mapping[str, Any],
    m3: Mapping[str, Any],
    *,
    require_runnable: bool = False,
) -> dict[str, Any]:
    first = validate_run_config(m1, require_runnable=require_runnable)
    second = validate_run_config(m3, require_runnable=require_runnable)
    _require(first["control"] == "m1_architecture_action_control", "first config is not M1")
    _require(second["control"] == "m3_ours", "second config is not M3")
    _require(_fairness_projection(m1) == _fairness_projection(m3), "M1/M3 differ outside control, lambda, config id, or output")
    return {
        "status": "PASS",
        "allowed_differences": [
            "control",
            "objective.lambda_contrastive",
            "config_id",
            "output_dir",
        ],
        "training_seed": first["training_seed"],
        "regime": first["regime"],
        "global_batch": first["global_batch"],
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "YAML config root must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", required=True)
    parser.add_argument("--m3")
    parser.add_argument("--require-runnable", action="store_true")
    args = parser.parse_args()
    m1 = load_yaml(args.m1)
    result = (
        validate_m1_m3_pair(m1, load_yaml(args.m3), require_runnable=args.require_runnable)
        if args.m3
        else validate_run_config(m1, require_runnable=args.require_runnable)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
