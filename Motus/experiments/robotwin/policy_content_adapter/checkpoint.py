"""Create-only compact checkpoints for adapter training and exact resume."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA = "motus_policy_content_adapter_training_checkpoint"
CHECKPOINT_VERSION = 1
MANIFEST_SCHEMA = "motus_policy_content_adapter_checkpoint_manifest"


class CheckpointError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResumeIdentity:
    control: str
    regime: str
    training_seed: int
    world_size: int
    config_sha256: str
    base_lineage_sha256: str
    paired_manifest_sha256: str
    official_manifest_sha256: str

    def validate(self) -> "ResumeIdentity":
        if self.control not in {"m1_architecture_action_control", "m3_ours"}:
            raise CheckpointError("resume control is invalid")
        if self.regime not in {"m_p1", "m_p2"}:
            raise CheckpointError("resume regime is invalid")
        if self.training_seed < 0 or self.world_size <= 0:
            raise CheckpointError("resume seed/world size is invalid")
        for name in (
            "config_sha256",
            "base_lineage_sha256",
            "paired_manifest_sha256",
            "official_manifest_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise CheckpointError(f"{name} is not a lower-case SHA-256")
        return self


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state:
        if not torch.cuda.is_available():
            raise CheckpointError("checkpoint contains CUDA RNG but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def save_training_checkpoint(
    output_dir: str | Path,
    *,
    conditioner: nn.Module,
    action_expert: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    identity: ResumeIdentity,
    official_sampler_state: Mapping[str, Any],
    paired_sampler_state: Mapping[str, Any],
    extra_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically create a resumable checkpoint directory.

    This helper is for unwrapped/single-process modules.  A DeepSpeed runner
    must first obtain consolidated state on rank zero or use Accelerator's
    distributed state save and then bind it with the same manifest fields.
    """

    identity.validate()
    if global_step < 0 or epoch < 0:
        raise CheckpointError("global_step and epoch must be non-negative")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_VERSION,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "identity": asdict(identity),
            "conditioner": conditioner.state_dict(),
            "action_expert": (
                action_expert.state_dict() if identity.regime == "m_p2" else None
            ),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": capture_rng_state(),
            "official_sampler_state": dict(official_sampler_state),
            "paired_sampler_state": dict(paired_sampler_state),
            "extra_audit": dict(extra_audit or {}),
        }
        checkpoint_path = staging / "checkpoint.pt"
        torch.save(payload, checkpoint_path)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": CHECKPOINT_VERSION,
            "status": "PASS",
            "checkpoint_file": {
                "name": checkpoint_path.name,
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            },
            "global_step": int(global_step),
            "epoch": int(epoch),
            "identity": asdict(identity),
            "contains_action_expert": identity.regime == "m_p2",
            "contains_optimizer": True,
            "contains_scheduler": scheduler is not None,
            "contains_rng": True,
            "contains_sampler_states": True,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    conditioner: nn.Module,
    action_expert: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_identity: ResumeIdentity,
    restore_rng: bool = True,
) -> dict[str, Any]:
    expected_identity.validate()
    root = Path(checkpoint_dir).resolve()
    manifest_path = root / "manifest.json"
    checkpoint_path = root / "checkpoint.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise CheckpointError("checkpoint directory is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "PASS":
        raise CheckpointError("checkpoint manifest is invalid")
    identity = manifest.get("identity")
    if identity != asdict(expected_identity):
        raise CheckpointError("checkpoint resume identity does not match this run")
    file_identity = manifest.get("checkpoint_file", {})
    if checkpoint_path.stat().st_size != int(file_identity.get("size_bytes", -1)):
        raise CheckpointError("checkpoint size changed")
    if _sha256(checkpoint_path) != file_identity.get("sha256"):
        raise CheckpointError("checkpoint SHA changed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("checkpoint payload schema changed")
    if payload.get("identity") != asdict(expected_identity):
        raise CheckpointError("checkpoint payload identity mismatch")
    conditioner.load_state_dict(payload["conditioner"], strict=True)
    if expected_identity.regime == "m_p2":
        if payload.get("action_expert") is None:
            raise CheckpointError("M-P2 checkpoint omitted Action Expert")
        action_expert.load_state_dict(payload["action_expert"], strict=True)
    elif payload.get("action_expert") is not None:
        raise CheckpointError("M-P1 checkpoint unexpectedly contains Action Expert")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload.get("scheduler") is None:
            raise CheckpointError("checkpoint omitted scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    elif payload.get("scheduler") is not None:
        raise CheckpointError("checkpoint has a scheduler but runtime does not")
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return {
        "status": "PASS",
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "official_sampler_state": payload["official_sampler_state"],
        "paired_sampler_state": payload["paired_sampler_state"],
        "extra_audit": payload.get("extra_audit", {}),
        "checkpoint_sha256": file_identity["sha256"],
    }

