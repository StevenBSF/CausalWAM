"""RoboTwin deployment bridge for compact policy-content-adapter checkpoints.

This module deliberately reuses the native FastWAM RoboTwin observation,
normalization, action-queue, and simulator-step semantics.  The only changed
part is checkpoint loading: a compact policy checkpoint is resolved by
``model.load_policy_checkpoint_into_model`` so the release checkpoint remains
immutable and is verified before policy-specific weights are attached.

The module is also executable as a no-SAPIEN one-action smoke test.  That path
loads the real model and checkpoint and performs one real FastWAM inference per
requested task against a tiny environment stub; it never imports RoboTwin or
constructs a renderer.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.robotwin.fastwam_policy.deploy_policy import (  # noqa: E402
    WorldActionRobotWinPolicy as _NativeRobotWinPolicy,
    _compose_sim_cfg,
    _is_none_like,
    _mixed_precision_to_model_dtype,
    _parse_bool,
    _parse_optional_float,
    _parse_optional_int,
    _resolve_dataset_stats_path,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)

if __package__:
    from .runtime_utils import audit_local_fastwam_source  # noqa: E402
else:  # Direct execution: ``python .../rollout_policy.py``.
    from experiments.robotwin.policy_content_adapter.runtime_utils import (  # noqa: E402
        audit_local_fastwam_source,
    )


logger = logging.getLogger(__name__)

POLICY_CHECKPOINT_LOADER = "load_policy_checkpoint_into_model"
DEFAULT_SIM_TASK = "robotwin_uncond_3cam_384_1e-4"
DEFAULT_TASKS = (
    "place_a2b_left",
    "open_microwave",
    "move_stapler_pad",
)
TASK_SMOKE_INSTRUCTIONS = {
    "move_stapler_pad": "Move the stapler to the colored mat.",
    "open_microwave": "Open the microwave door.",
    "place_a2b_left": "Place object A to the left of object B.",
}


@contextlib.contextmanager
def _temporary_environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_metadata(value: Any, *, field: str) -> Any:
    """Copy checkpoint provenance while refusing tensor-valued metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _copy_metadata(item, field=f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_copy_metadata(item, field=f"{field}[]") for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"policy checkpoint metadata {field} has unsupported type {type(value)!r}"
    )


def _checkpoint_contract() -> tuple[str, int, int, int]:
    if __package__:
        from .model import (
            EXPECTED_ADAPTER_PARAMETER_COUNT,
            EXPECTED_HEAD_PARAMETER_COUNT,
            POLICY_CHECKPOINT_SCHEMA,
            POLICY_CHECKPOINT_VERSION,
        )
    else:
        from experiments.robotwin.policy_content_adapter.model import (
            EXPECTED_ADAPTER_PARAMETER_COUNT,
            EXPECTED_HEAD_PARAMETER_COUNT,
            POLICY_CHECKPOINT_SCHEMA,
            POLICY_CHECKPOINT_VERSION,
        )

    return (
        POLICY_CHECKPOINT_SCHEMA,
        int(POLICY_CHECKPOINT_VERSION),
        int(EXPECTED_HEAD_PARAMETER_COUNT),
        int(EXPECTED_ADAPTER_PARAMETER_COUNT),
    )


def _extract_checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema, schema_version, _, _ = _checkpoint_contract()
    if payload.get("schema") != schema:
        raise ValueError(
            f"not a {schema!r} compact policy checkpoint: {payload.get('schema')!r}"
        )
    if int(payload.get("schema_version", -1)) != schema_version:
        raise ValueError(
            "unsupported compact policy checkpoint schema version: "
            f"{payload.get('schema_version')!r}"
        )
    regime = str(payload.get("regime", ""))
    if regime not in {"p_v1", "p_v2"}:
        raise ValueError(f"invalid policy checkpoint regime: {regime!r}")
    try:
        step = int(payload["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy checkpoint has no valid training step") from exc
    if step < 0:
        raise ValueError(f"policy checkpoint step must be non-negative, got {step}")

    base = payload.get("base_checkpoint")
    if not isinstance(base, Mapping):
        raise ValueError("policy checkpoint lacks base_checkpoint provenance")
    base_copy = _copy_metadata(base, field="base_checkpoint")
    base_text = str(base_copy.get("path", "")).strip()
    if not base_text:
        raise ValueError("base_checkpoint.path must be non-empty")
    base_path = Path(base_text).expanduser()
    if not base_path.is_absolute():
        raise ValueError("base_checkpoint.path must be absolute")
    base_path = base_path.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {base_path}")
    actual_size = int(base_path.stat().st_size)
    if int(base_copy.get("size_bytes", -1)) != actual_size:
        raise ValueError(
            "base checkpoint size differs from compact-checkpoint provenance"
        )
    base_sha = str(base_copy.get("sha256", "")).lower()
    if len(base_sha) != 64 or any(
        character not in "0123456789abcdef" for character in base_sha
    ):
        raise ValueError("base_checkpoint.sha256 must be a 64-character hex digest")
    base_copy["path"] = str(base_path)
    base_copy["sha256"] = base_sha

    run_config = payload.get("run_config")
    if not isinstance(run_config, Mapping):
        raise ValueError("policy checkpoint run_config must be a mapping")
    for name in ("head_config", "adapter_config"):
        if not isinstance(payload.get(name), Mapping):
            raise ValueError(f"policy checkpoint {name} must be a mapping")

    metadata = {
        "schema": schema,
        "schema_version": schema_version,
        "regime": regime,
        "step": step,
        "base_checkpoint": base_copy,
        "head_config": _copy_metadata(payload["head_config"], field="head_config"),
        "adapter_config": _copy_metadata(
            payload["adapter_config"], field="adapter_config"
        ),
        "run_config": _copy_metadata(run_config, field="run_config"),
    }
    artifact_identities = payload.get("artifact_identities")
    if not isinstance(artifact_identities, Mapping):
        raise ValueError("policy checkpoint artifact_identities must be a mapping")
    metadata["artifact_identities"] = _copy_metadata(
        artifact_identities, field="artifact_identities"
    )
    return metadata


def _read_checkpoint_provenance(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read only compact-checkpoint metadata using safe, mmap-backed loading."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint}")
    before = checkpoint.stat()
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise ValueError(
            f"cannot safely read compact policy checkpoint provenance: {checkpoint}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("policy checkpoint root must be a mapping")
    metadata = _extract_checkpoint_metadata(payload)
    del payload
    after = checkpoint.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity:
        raise RuntimeError("policy checkpoint changed while provenance was being read")
    return {
        "policy_checkpoint": {
            "path": str(checkpoint),
            "size_bytes": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
        },
        "metadata": metadata,
    }


def _resolve_model_base_path(
    checkpoint_path: str | Path,
    explicit_model_base_path: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve component weights from an explicit argument or bound provenance.

    No current-working-directory or ambient-environment fallback is allowed:
    those are especially dangerous when RoboTwin changes into its own root.
    """

    provenance = _read_checkpoint_provenance(checkpoint_path)
    metadata = provenance["metadata"]
    run_config = metadata["run_config"]
    declared = run_config.get("model_base_path")
    declared_text = "" if declared is None else str(declared).strip()

    if not _is_none_like(explicit_model_base_path):
        candidate = Path(str(explicit_model_base_path)).expanduser().resolve()
        source = "explicit_parameter"
    elif declared_text and not _is_none_like(declared_text):
        declared_path = Path(declared_text).expanduser()
        if not declared_path.is_absolute():
            raise ValueError(
                "checkpoint run_config.model_base_path must be absolute; "
                "pass model_base_path explicitly to relocate components"
            )
        candidate = declared_path.resolve()
        source = "checkpoint_run_config"
    else:
        base_path = Path(metadata["base_checkpoint"]["path"])
        if base_path.parent.name != "fastwam_release":
            raise ValueError(
                "cannot infer DIFFSYNTH_MODEL_BASE_PATH from checkpoint provenance: "
                "base checkpoint is not under <model-base>/fastwam_release; "
                "pass model_base_path explicitly"
            )
        candidate = base_path.parent.parent.resolve()
        source = "base_checkpoint_layout"

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Wan/VAE/text model component base not found: {candidate} "
            "(pass model_base_path explicitly if artifacts were relocated)"
        )
    audit = {
        **provenance,
        "resolved_model_base_path": str(candidate),
        "resolution_source": source,
        "declared_model_base_path": declared_text or None,
    }
    return candidate, audit


def _validate_fastwam_source_binding(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Verify rollout uses the exact FastWAM sources bound during training.

    The policy overlay is meaningful only relative to the native action path it
    was trained through.  A compact checkpoint therefore records hashes for the
    small set of native files that define that path and the official loader.
    Rollout fails closed if the provenance is missing, malformed, relocated, or
    differs byte-for-byte from the current workspace checkout.
    """

    run_config = metadata.get("run_config")
    if not isinstance(run_config, Mapping):
        raise ValueError("policy checkpoint run_config must be a mapping")
    runtime_provenance = run_config.get("runtime_provenance")
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError(
            "policy checkpoint lacks run_config.runtime_provenance"
        )
    expected = runtime_provenance.get("fastwam_source")
    if not isinstance(expected, Mapping):
        raise ValueError(
            "policy checkpoint lacks runtime_provenance.fastwam_source"
        )
    if expected.get("status") != "PASS":
        raise ValueError(
            "checkpoint FastWAM source provenance was not recorded as PASS"
        )
    if expected.get("scope") != "all_python_files_under_src_fastwam":
        raise ValueError("checkpoint FastWAM source audit scope is incomplete")

    expected_root_text = str(expected.get("source_root", "")).strip()
    if not expected_root_text:
        raise ValueError("checkpoint FastWAM source provenance lacks source_root")
    expected_root = Path(expected_root_text).expanduser()
    if not expected_root.is_absolute():
        raise ValueError("checkpoint FastWAM source_root must be absolute")
    expected_root = expected_root.resolve()

    expected_package_text = str(expected.get("package_file", "")).strip()
    if not expected_package_text:
        raise ValueError("checkpoint FastWAM source provenance lacks package_file")
    expected_package = Path(expected_package_text).expanduser()
    if not expected_package.is_absolute():
        raise ValueError("checkpoint FastWAM package_file must be absolute")
    expected_package = expected_package.resolve()
    if expected_package != expected_root / "fastwam/__init__.py":
        raise ValueError(
            "checkpoint FastWAM package_file is inconsistent with source_root"
        )

    expected_files = expected.get("files")
    if not isinstance(expected_files, Mapping) or not expected_files:
        raise ValueError("checkpoint FastWAM source provenance has no files")

    current = audit_local_fastwam_source()
    if current.get("scope") != expected.get("scope"):
        raise ValueError("rollout FastWAM source audit scope differs from checkpoint")
    current_root = Path(str(current["source_root"])).resolve()
    if current_root != expected_root:
        raise ValueError(
            "rollout FastWAM source_root differs from checkpoint provenance: "
            f"{current_root} != {expected_root}"
        )
    current_package = Path(str(current["package_file"])).resolve()
    if current_package != expected_package:
        raise ValueError(
            "rollout FastWAM package import differs from checkpoint provenance"
        )
    current_files = current.get("files")
    if not isinstance(current_files, Mapping):
        raise RuntimeError("current FastWAM source audit returned invalid files")
    if set(expected_files) != set(current_files):
        raise ValueError(
            "rollout FastWAM audited file set differs from checkpoint provenance: "
            f"current={sorted(current_files)}, expected={sorted(expected_files)}"
        )
    if int(expected.get("file_count", -1)) != len(expected_files):
        raise ValueError("checkpoint FastWAM source file_count is inconsistent")
    if int(current.get("file_count", -1)) != len(current_files):
        raise RuntimeError("current FastWAM source audit file_count is inconsistent")

    verified_files: dict[str, dict[str, Any]] = {}
    for relative in sorted(current_files):
        expected_identity = expected_files[relative]
        actual_identity = current_files[relative]
        if not isinstance(expected_identity, Mapping) or not isinstance(
            actual_identity, Mapping
        ):
            raise ValueError(
                f"FastWAM source identity for {relative!r} must be a mapping"
            )
        expected_path = Path(str(expected_identity.get("path", ""))).expanduser()
        if not expected_path.is_absolute():
            raise ValueError(
                f"checkpoint FastWAM source path for {relative!r} must be absolute"
            )
        if expected_path.resolve() != (expected_root / relative).resolve():
            raise ValueError(
                f"checkpoint FastWAM source path for {relative!r} is inconsistent"
            )
        expected_size = int(expected_identity.get("size_bytes", -1))
        actual_size = int(actual_identity.get("size_bytes", -2))
        if expected_size != actual_size:
            raise ValueError(
                f"rollout FastWAM source size differs for {relative!r}"
            )
        expected_sha = str(expected_identity.get("sha256", "")).lower()
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise ValueError(
                f"checkpoint FastWAM source SHA-256 is invalid for {relative!r}"
            )
        actual_sha = str(actual_identity.get("sha256", "")).lower()
        if actual_sha != expected_sha:
            raise ValueError(
                f"rollout FastWAM source SHA-256 differs for {relative!r}"
            )
        verified_files[str(relative)] = {
            "path": str(Path(str(actual_identity["path"])).resolve()),
            "size_bytes": actual_size,
            "sha256": actual_sha,
        }

    return {
        "status": "PASS",
        "scope": str(current["scope"]),
        "file_count": len(verified_files),
        "verification": "complete_python_tree_exact_source_root_size_sha256",
        "source_root": str(current_root),
        "package_file": str(current_package),
        "files": verified_files,
    }


def _dataset_stats_identity(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    run_config = metadata.get("run_config")
    if not isinstance(run_config, Mapping):
        return None
    official = run_config.get("official")
    candidates: list[Any] = []
    if isinstance(official, Mapping):
        candidates.extend(
            [official.get("dataset_stats_identity"), official.get("dataset_stats")]
        )
    for container in (
        metadata.get("artifact_identities"),
        run_config.get("artifact_identities"),
    ):
        if isinstance(container, Mapping):
            candidates.extend(
                [container.get("dataset_stats"), container.get("official_dataset_stats")]
            )
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("path"):
            return candidate
    return None


def _validate_dataset_stats_binding(
    dataset_stats_path: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = Path(dataset_stats_path).expanduser().resolve()
    if not supplied.is_file():
        raise FileNotFoundError(f"Dataset stats path not found: {supplied}")
    run_config = metadata.get("run_config")
    official = run_config.get("official") if isinstance(run_config, Mapping) else None
    declared_value = (
        official.get("dataset_stats") if isinstance(official, Mapping) else None
    )
    declared_path: Path | None = None
    if isinstance(declared_value, Mapping):
        if declared_value.get("path"):
            declared_path = Path(str(declared_value["path"])).expanduser()
    elif not _is_none_like(declared_value):
        declared_path = Path(str(declared_value)).expanduser()
    if declared_path is not None:
        if not declared_path.is_absolute():
            raise ValueError("checkpoint official.dataset_stats path must be absolute")
        declared_path = declared_path.resolve()

    identity = _dataset_stats_identity(metadata)
    if identity is None and declared_path is None:
        raise ValueError(
            "policy checkpoint does not bind the training dataset_stats artifact"
        )
    if identity is None:
        if supplied != declared_path:
            raise ValueError(
                "rollout dataset_stats differs from the path bound by training: "
                f"{supplied} != {declared_path}"
            )
        return {
            "path": str(supplied),
            "verification": "checkpoint_bound_path",
            "size_bytes": int(supplied.stat().st_size),
        }

    identity_path = Path(str(identity["path"])).expanduser()
    if not identity_path.is_absolute():
        raise ValueError("checkpoint dataset_stats identity path must be absolute")
    identity_path = identity_path.resolve()
    if declared_path is not None and identity_path != declared_path:
        raise ValueError(
            "checkpoint contains inconsistent dataset_stats path declarations"
        )
    expected_size = int(identity.get("size_bytes", -1))
    actual_size = int(supplied.stat().st_size)
    if expected_size != actual_size:
        raise ValueError("rollout dataset_stats size differs from checkpoint identity")
    expected_sha = str(identity.get("sha256", "")).lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError("checkpoint dataset_stats identity lacks a valid SHA-256")
    actual_sha = _sha256_file(supplied)
    if actual_sha != expected_sha:
        raise ValueError("rollout dataset_stats SHA-256 differs from checkpoint identity")
    return {
        "path": str(supplied),
        "verification": "sha256",
        "size_bytes": actual_size,
        "sha256": actual_sha,
    }


def _validate_rollout_device(device: str) -> str:
    try:
        parsed = torch.device(str(device))
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid rollout device: {device!r}") from exc
    if parsed.type != "cuda":
        return str(parsed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA rollout was requested ({device!r}) but CUDA is unavailable; "
            "refusing an implicit CPU fallback for the ~1B-parameter policy"
        )
    index = 0 if parsed.index is None else int(parsed.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; visible device count is "
            f"{torch.cuda.device_count()}"
        )
    return str(parsed)


def _validate_loaded_checkpoint_contract(
    metadata: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    _, _, expected_head, expected_adapter = _checkpoint_contract()
    if int(audit.get("head_parameter_count", -1)) != expected_head:
        raise ValueError("loaded policy head parameter count violates the protocol")
    if int(audit.get("adapter_parameter_count", -1)) != expected_adapter:
        raise ValueError("loaded policy adapter parameter count violates the protocol")
    expected_overlay = metadata.get("regime") == "p_v2"
    if audit.get("action_expert_overlaid") is not expected_overlay:
        raise ValueError("ActionDiT overlay audit disagrees with checkpoint regime")
    audited_base = audit.get("base_checkpoint")
    if not isinstance(audited_base, Mapping) or dict(audited_base) != dict(
        metadata["base_checkpoint"]
    ):
        raise ValueError("checkpoint loader base identity disagrees with payload")
    release = audit.get("release_load")
    if not isinstance(release, Mapping):
        raise ValueError("checkpoint loader did not return a release-load audit")
    if Path(str(release.get("path", ""))).expanduser().resolve() != Path(
        metadata["base_checkpoint"]["path"]
    ):
        raise ValueError("release-load audit path disagrees with base identity")
    if int(release.get("size_bytes", -1)) != int(
        metadata["base_checkpoint"]["size_bytes"]
    ):
        raise ValueError("release-load audit size disagrees with base identity")
    required_artifacts = {
        str(name): identity
        for name, identity in metadata["artifact_identities"].items()
        if isinstance(identity, Mapping)
        and bool(identity.get("required_for_rollout", False))
    }
    verified_artifacts = audit.get("verified_runtime_artifacts")
    if not isinstance(verified_artifacts, Mapping):
        raise ValueError("checkpoint loader did not audit runtime artifacts")
    if set(verified_artifacts) != set(required_artifacts):
        raise ValueError(
            "verified rollout artifacts differ from checkpoint requirements: "
            f"verified={sorted(verified_artifacts)}, "
            f"required={sorted(required_artifacts)}"
        )
    for name, identity in required_artifacts.items():
        actual = verified_artifacts[name]
        if not isinstance(actual, Mapping):
            raise ValueError(f"runtime artifact audit {name!r} is not a mapping")
        for field in ("kind", "size_bytes", "sha256"):
            expected = identity.get(field, "file" if field == "kind" else None)
            if actual.get(field) != expected:
                raise ValueError(
                    f"runtime artifact audit {name!r}/{field} disagrees with "
                    "checkpoint identity"
                )


def _load_policy_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    dataset_stats_path: str | Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Attach and strictly load the compact policy checkpoint into ``model``.

    Expected API supplied by this experiment's ``model.py``::

        load_policy_checkpoint_into_model(
            model,
            path,
            verify_base=True,
            verify_runtime_artifacts=True,
            runtime_artifacts={"dataset_stats": dataset_stats_path},
        )

    Its contract is to verify and strictly load the referenced release/base
    checkpoint first, attach the content head and gated adapter, strictly load
    their state, and (for P-v2) strictly load the ActionDiT state.  Keeping that
    logic in ``model.py`` makes training and rollout share one checkpoint
    implementation instead of maintaining two subtly different loaders.
    """

    if __package__:
        from .model import load_policy_checkpoint_into_model
    else:  # Direct execution: ``python .../rollout_policy.py``.
        from experiments.robotwin.policy_content_adapter.model import (
            load_policy_checkpoint_into_model,
        )

    if not callable(load_policy_checkpoint_into_model):
        raise TypeError(f"{POLICY_CHECKPOINT_LOADER} must be callable")

    signature = inspect.signature(load_policy_checkpoint_into_model)
    required_loader_parameters = {
        "verify_base",
        "verify_runtime_artifacts",
        "runtime_artifacts",
    }
    if not required_loader_parameters.issubset(signature.parameters):
        raise TypeError(
            f"{POLICY_CHECKPOINT_LOADER} must expose "
            f"{sorted(required_loader_parameters)}; got {signature}"
        )
    loaded = load_policy_checkpoint_into_model(
        model,
        str(Path(checkpoint_path).expanduser().resolve()),
        verify_base=True,
        verify_runtime_artifacts=True,
        runtime_artifacts={
            "dataset_stats": str(Path(dataset_stats_path).expanduser().resolve())
        },
    )
    if not isinstance(loaded, tuple) or len(loaded) != 3:
        raise TypeError(
            f"{POLICY_CHECKPOINT_LOADER} must return (runtime, payload, audit)"
        )
    runtime, payload, audit = loaded
    if not isinstance(payload, Mapping) or not isinstance(audit, Mapping):
        raise TypeError(
            f"{POLICY_CHECKPOINT_LOADER} returned invalid payload/audit types: "
            f"{type(payload)!r}, {type(audit)!r}"
        )
    metadata = _extract_checkpoint_metadata(payload)
    _validate_loaded_checkpoint_contract(metadata, audit)
    # Do not retain checkpoint tensors on CPU after strict loading, especially
    # the ~1B-parameter P-v2 ActionDiT overlay.  Keep only provenance metadata.
    return runtime, metadata, dict(audit)


class PolicyContentAdapterRobotWinPolicy(_NativeRobotWinPolicy):
    """Native RoboTwin policy semantics with compact adapter checkpoint load."""

    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        model_base_path: str | Path | None,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
    ) -> None:
        # This is the native constructor up to checkpoint load, with exactly
        # that load call replaced by the compact policy loader above.
        resolved_model_base, model_base_audit = _resolve_model_base_path(
            checkpoint_path,
            model_base_path,
        )
        fastwam_source_audit = _validate_fastwam_source_binding(
            model_base_audit["metadata"]
        )
        dataset_stats_audit = _validate_dataset_stats_binding(
            dataset_stats_path,
            model_base_audit["metadata"],
        )
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True
        # Instantiate only the structure. The compact loader strictly restores
        # the SHA-bound release checkpoint and then the policy overlay.
        model_cfg_copy.skip_dit_load_from_pretrain = True
        model_cfg_copy.action_dit_pretrained_path = None

        with _temporary_environment(
            "DIFFSYNTH_MODEL_BASE_PATH", str(resolved_model_base)
        ):
            self.model = instantiate(
                model_cfg_copy,
                model_dtype=model_dtype,
                device=device,
            )
            (
                self.policy_content_runtime,
                self.checkpoint_metadata,
                self.checkpoint_audit,
            ) = _load_policy_checkpoint(
                self.model,
                checkpoint_path,
                dataset_stats_path=dataset_stats_path,
            )
        if self.checkpoint_metadata != model_base_audit["metadata"]:
            raise RuntimeError(
                "policy checkpoint provenance changed between model construction "
                "and strict loading"
            )
        self.model_base_audit = model_base_audit
        self.checkpoint_audit["dataset_stats"] = dataset_stats_audit
        self.checkpoint_audit["model_base"] = {
            "path": str(resolved_model_base),
            "source": model_base_audit["resolution_source"],
        }
        self.checkpoint_audit["fastwam_source"] = fastwam_source_audit
        self.model = self.model.to(device=device, dtype=model_dtype).eval()
        # The runtime intentionally keeps its conditioner outside the native
        # module tree, so cast it explicitly after moving the native model.
        self.policy_content_runtime.conditioner.to(
            device=device,
            dtype=model_dtype,
        ).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized PolicyContentAdapterRobotWinPolicy | ckpt=%s | "
            "stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def reset(self) -> None:
        super().reset()
        # A fresh episode must never inherit the previous observation's Zc,
        # even though the empty native action queue would force a new prefill.
        self.policy_content_runtime.conditioner.clear_active_content()


def get_model(usr_args: Dict[str, Any]) -> PolicyContentAdapterRobotWinPolicy:
    """RoboTwin policy factory; mirrors the native FastWAM factory."""

    cfg = _compose_sim_cfg(
        sim_cfg_path=usr_args.get("sim_cfg_path"),
        sim_cfg_name=usr_args.get("sim_cfg_name"),
        sim_task=usr_args.get("sim_task"),
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must name a compact policy checkpoint.")
    resolved_checkpoint = Path(str(checkpoint_path)).expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {resolved_checkpoint}")

    device = _validate_rollout_device(
        str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    )

    mixed_precision = str(
        usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16")
    )
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    dataset_stats_path = _resolve_dataset_stats_path(usr_args.get("dataset_stats_path"))

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        configured_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = (
            configured_horizon
            if configured_horizon is not None
            else int(cfg.data.train.num_frames) - 1
        )
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(
            cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps)
        )
    if num_inference_steps <= 0:
        raise ValueError(
            f"`num_inference_steps` must be positive, got {num_inference_steps}"
        )

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(
        usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0))
    )
    negative_prompt = str(
        usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", ""))
    )
    rand_device = str(
        usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu"))
    )
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    return PolicyContentAdapterRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(resolved_checkpoint),
        model_base_path=usr_args.get("model_base_path"),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(
            (int(cfg.data.train.num_frames) - 1)
            // int(cfg.data.train.action_video_freq_ratio)
            + 1
        ),
    )


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def eval(
    task_env: Any,
    model: PolicyContentAdapterRobotWinPolicy,
    observation: Optional[Dict[str, Any]],
) -> None:
    model.step(task_env, encode_obs(observation))


def reset_model(model: PolicyContentAdapterRobotWinPolicy) -> None:
    model.reset()


class _OneActionEnv:
    def __init__(self, instruction: str) -> None:
        self.instruction = instruction
        self.actions: list[np.ndarray] = []

    def get_instruction(self) -> str:
        return self.instruction

    def take_action(self, action: np.ndarray, action_type: str) -> None:
        value = np.asarray(action, dtype=np.float32)
        if action_type != "qpos":
            raise AssertionError(f"Expected qpos action, got {action_type!r}")
        if value.shape != (14,):
            raise AssertionError(f"Expected one 14-D action, got {value.shape}")
        if not np.isfinite(value).all():
            raise AssertionError("Policy emitted a non-finite action")
        self.actions.append(value.copy())


def _synthetic_robotwin_observation() -> Dict[str, Any]:
    # Use native D435 raw dimensions; the inherited policy performs the exact
    # production resize/three-camera concatenation before inference.
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": image.copy()},
            "left_camera": {"rgb": image.copy()},
            "right_camera": {"rgb": image.copy()},
        },
        "joint_action": {"vector": np.zeros((14,), dtype=np.float32)},
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def run_one_action_smoke(args: argparse.Namespace) -> dict[str, Any]:
    requested_tasks = tuple(args.task or DEFAULT_TASKS)
    unknown = sorted(set(requested_tasks) - set(DEFAULT_TASKS))
    if unknown:
        raise ValueError(f"Unsupported smoke tasks: {unknown}")

    policy = get_model(
        {
            "sim_cfg_path": str(Path(args.sim_cfg_path).expanduser().resolve()),
            "sim_task": args.sim_task,
            "ckpt_setting": str(Path(args.checkpoint).expanduser().resolve()),
            "model_base_path": args.model_base_path,
            "dataset_stats_path": str(Path(args.dataset_stats).expanduser().resolve()),
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "timing_enabled": True,
        }
    )

    task_results: list[dict[str, Any]] = []
    for task_name in requested_tasks:
        environment = _OneActionEnv(TASK_SMOKE_INSTRUCTIONS[task_name])
        policy.reset()
        policy.step(environment, _synthetic_robotwin_observation())
        if len(environment.actions) != 1:
            raise AssertionError(
                f"{task_name}: expected exactly one executed action, "
                f"got {len(environment.actions)}"
            )
        action = environment.actions[0]
        conditioner = policy.policy_content_runtime.conditioner
        content_tokens = conditioner._active_content_tokens
        expected_content_shape = (
            1,
            conditioner.head.num_queries,
            conditioner.head.embed_dim,
        )
        if content_tokens is None or tuple(content_tokens.shape) != expected_content_shape:
            observed = None if content_tokens is None else tuple(content_tokens.shape)
            raise AssertionError(
                f"{task_name}: expected deployed Zc shape {expected_content_shape}, "
                f"got {observed}"
            )
        if not bool(torch.isfinite(content_tokens).all().item()):
            raise AssertionError(f"{task_name}: deployed Zc contains non-finite values")
        task_results.append(
            {
                "task": task_name,
                "executed_actions": 1,
                "action_shape": list(action.shape),
                "action_finite": bool(np.isfinite(action).all()),
                "action_l2": float(np.linalg.norm(action)),
                "zc_shape": list(content_tokens.shape),
                "zc_finite": True,
                "gate": conditioner.adapter.gate_value,
                "timing": policy.get_timing_rollout(),
            }
        )

    report = {
        "status": "PASS",
        "kind": "no_sapien_one_action_rollout_smoke",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "dataset_stats": str(Path(args.dataset_stats).expanduser().resolve()),
        "checkpoint_metadata": _jsonable(policy.checkpoint_metadata),
        "checkpoint_audit": _jsonable(policy.checkpoint_audit),
        "model_base_audit": _jsonable(policy.model_base_audit),
        "tasks": task_results,
        "sapien_imported": "sapien" in sys.modules,
    }
    if report["sapien_imported"]:
        raise AssertionError("No-SAPIEN smoke unexpectedly imported sapien")

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a compact policy checkpoint and execute one real action without SAPIEN."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument(
        "--model-base-path",
        help=(
            "Wan/VAE/text component directory. If omitted, use the absolute "
            "model_base_path stored in the compact checkpoint or its audited "
            "<base>/fastwam_release layout."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=DEFAULT_TASKS,
        help="Repeat to select tasks; default is all three smoke tasks.",
    )
    parser.add_argument(
        "--sim-cfg-path",
        default=str(PROJECT_ROOT / "configs" / "sim_robotwin.yaml"),
    )
    parser.add_argument("--sim-task", default=DEFAULT_SIM_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    return parser


if __name__ == "__main__":
    run_one_action_smoke(_build_parser().parse_args())
