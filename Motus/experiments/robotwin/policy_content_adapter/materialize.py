"""Create execution-ready matched M1/M3 configs from audited artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .config_audit import load_yaml, validate_m1_m3_pair
from .paired_data import sha256_file


FORMAL_PROFILE = "motus_author_5epoch_v1"
AUTHOR_SMOKE_PROFILE = "motus_author_batch8_smoke_v1"
FORMAL_EPOCHS = 5
FORMAL_VIRTUAL_SAMPLES = 16_500
FORMAL_STEPS_PER_EPOCH = 257
FORMAL_MAX_STEPS = FORMAL_EPOCHS * FORMAL_STEPS_PER_EPOCH


def _file_identity(path: str | Path) -> dict[str, Any]:
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256_file(value),
    }


def _directory_manifest_identity(
    path: str | Path, identity_file: str
) -> dict[str, Any]:
    root = Path(path).resolve()
    identity_path = root / identity_file
    if not identity_path.is_file():
        raise FileNotFoundError(identity_path)
    return {
        "path": str(root),
        "identity_file": identity_file,
        "size_bytes": identity_path.stat().st_size,
        "sha256": sha256_file(identity_path),
    }


def materialize_pair(
    *,
    m1_template: str | Path,
    m3_template: str | Path,
    output_dir: str | Path,
    run_output_root: str | Path,
    base_lineage: str | Path,
    implementation_audit: str | Path,
    strict_load_audit: str | Path,
    zero_gate_audit: str | Path,
    official_manifest: str | Path,
    paired_manifest: str | Path,
    token_cache: str | Path,
    task_text_cache: str | Path,
    regime: str,
    training_seed: int,
    world_size: int,
    per_device_batch: int,
    paired_groups_per_device: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    checkpoint_interval: int,
    profile: str = "engineering_smoke",
    head_adapter_lr: float = 1.0e-4,
    action_expert_lr: float = 1.0e-5,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite materialization {destination}")
    if regime not in {"m_p1", "m_p2"}:
        raise ValueError("regime must be m_p1 or m_p2")
    if min(training_seed, world_size, per_device_batch, paired_groups_per_device, gradient_accumulation_steps, max_steps, checkpoint_interval) <= 0:
        raise ValueError("materialization numeric values must be positive")
    if head_adapter_lr <= 0 or action_expert_lr <= 0:
        raise ValueError("materialization learning rates must be positive")
    if profile not in {
        "engineering_smoke",
        AUTHOR_SMOKE_PROFILE,
        FORMAL_PROFILE,
    }:
        raise ValueError("unsupported training profile")
    if profile in {AUTHOR_SMOKE_PROFILE, FORMAL_PROFILE}:
        expected = {
            "world_size": 8,
            "per_device_batch": 8,
            "gradient_accumulation_steps": 1,
        }
        actual = {
            "world_size": world_size,
            "per_device_batch": per_device_batch,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
        if actual != expected:
            raise ValueError(
                f"formal Motus profile requires {expected}, got {actual}"
            )
        if head_adapter_lr != 5.0e-5 or action_expert_lr != 5.0e-5:
            raise ValueError("formal Motus profile requires LR=5e-5 for all trainable groups")
    if profile == FORMAL_PROFILE:
        if max_steps != FORMAL_MAX_STEPS or checkpoint_interval != FORMAL_STEPS_PER_EPOCH:
            raise ValueError(
                "formal Motus profile requires max_steps=1285 and checkpoint_interval=257"
            )
    run_root = Path(run_output_root).resolve()
    artifacts = {
        "base_lineage_manifest": _file_identity(base_lineage),
        "implementation_audit": _file_identity(implementation_audit),
        "strict_load_audit": _file_identity(strict_load_audit),
        "zero_gate_audit": _file_identity(zero_gate_audit),
        "official_manifest": _file_identity(official_manifest),
        "paired_manifest": _file_identity(paired_manifest),
        "frozen_token_cache": _directory_manifest_identity(
            token_cache, "manifest.json"
        ),
        "task_text_cache": _directory_manifest_identity(
            task_text_cache, "audit.json"
        ),
    }
    configs = [load_yaml(m1_template), load_yaml(m3_template)]
    controls = ("m1", "m3")
    for config, short_name in zip(configs, controls, strict=True):
        config["runnable"] = True
        config.pop("fail_closed_reason", None)
        config["config_id"] = (
            f"motus_{short_name}_{regime}_seed{training_seed}_v1"
        )
        config["model"]["regime"] = regime
        config["model"]["freeze"]["action_expert"] = regime == "m_p1"
        training = config["training"]
        training.update(
            {
                "profile": profile,
                "seed": training_seed,
                "world_size": world_size,
                "per_device_batch": per_device_batch,
                "paired_groups_per_device": paired_groups_per_device,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "global_batch": world_size
                * per_device_batch
                * gradient_accumulation_steps,
                "max_steps": max_steps,
                "checkpoint_interval": checkpoint_interval,
                "head_adapter_lr": head_adapter_lr,
                "action_expert_lr": action_expert_lr if regime == "m_p2" else None,
            }
        )
        if profile in {AUTHOR_SMOKE_PROFILE, FORMAL_PROFILE}:
            training.update(
                {
                    "epochs": (
                        FORMAL_EPOCHS if profile == FORMAL_PROFILE else None
                    ),
                    "virtual_samples_per_epoch": FORMAL_VIRTUAL_SAMPLES,
                    "steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
                    "samples_per_epoch": FORMAL_STEPS_PER_EPOCH
                    * world_size
                    * per_device_batch,
                    "effective_dataset_exposures": (
                        max_steps
                        * world_size
                        * per_device_batch
                        / FORMAL_VIRTUAL_SAMPLES
                    ),
                    "official_sampler": "motus_distributed_drop_last_epoch_v1",
                    "drop_last": True,
                    "optimizer": "adamw",
                    "betas": [0.9, 0.95],
                    "scheduler": "motus_author_linear",
                    "warmup_steps": 200,
                    "cycle_length": 5_000_000,
                    "f_max": 0.99,
                    "f_min": 0.4,
                    "f_start": 1.0e-6,
                    "num_workers": 16,
                    "checkpoint_policy": (
                        "every_author_style_epoch_for_exact_resume"
                        if profile == FORMAL_PROFILE
                        else "engineering_smoke_frequent"
                    ),
                    "author_save_interval_reference": 5000,
                }
            )
        config["artifacts"] = copy.deepcopy(artifacts)
        config["output_dir"] = str(run_root / short_name)
    validate_m1_m3_pair(configs[0], configs[1], require_runnable=True)
    destination.mkdir(parents=True)
    paths = []
    for config, short_name in zip(configs, controls, strict=True):
        path = destination / f"{short_name}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        paths.append(
            {
                "control": config["control"],
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "motus_policy_content_adapter_materialization",
        "schema_version": 1,
        "status": "PASS",
        "regime": regime,
        "training_seed": training_seed,
        "world_size": world_size,
        "global_batch": world_size
        * per_device_batch
        * gradient_accumulation_steps,
        "max_steps": max_steps,
        "profile": profile,
        "epochs": FORMAL_EPOCHS if profile == FORMAL_PROFILE else None,
        "steps_per_epoch": (
            FORMAL_STEPS_PER_EPOCH if profile == FORMAL_PROFILE else None
        ),
        "configs": paths,
        "artifacts": artifacts,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-template", required=True)
    parser.add_argument("--m3-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-output-root", required=True)
    parser.add_argument("--base-lineage", required=True)
    parser.add_argument("--implementation-audit", required=True)
    parser.add_argument("--strict-load-audit", required=True)
    parser.add_argument("--zero-gate-audit", required=True)
    parser.add_argument("--official-manifest", required=True)
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--task-text-cache", required=True)
    parser.add_argument("--regime", choices=("m_p1", "m_p2"), required=True)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--paired-groups-per-device", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument(
        "--profile",
        choices=(
            "engineering_smoke",
            AUTHOR_SMOKE_PROFILE,
            FORMAL_PROFILE,
        ),
        default="engineering_smoke",
    )
    parser.add_argument("--head-adapter-lr", type=float, default=1.0e-4)
    parser.add_argument("--action-expert-lr", type=float, default=1.0e-5)
    args = parser.parse_args()
    result = materialize_pair(**vars(args))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(Path(args.output_dir).resolve()),
                "regime": result["regime"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
