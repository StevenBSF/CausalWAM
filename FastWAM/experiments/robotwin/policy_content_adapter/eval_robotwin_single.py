"""Independent RoboTwin evaluator for policy-content-adapter checkpoints.

Examples (from the FastWAM repository root)::

    # One task, one official phase.
    python experiments/robotwin/policy_content_adapter/eval_robotwin_single.py \
      ckpt=/path/policy_checkpoint.pt \
      EVALUATION.dataset_stats_path=/path/dataset_stats.json \
      EVALUATION.task_name=move_stapler_pad \
      EVALUATION.task_config=demo_clean

    # The same task under both official Clean and official Random protocols.
    python experiments/robotwin/policy_content_adapter/eval_robotwin_single.py \
      ckpt=/path/policy_checkpoint.pt \
      EVALUATION.dataset_stats_path=/path/dataset_stats.json \
      EVALUATION.task_name=move_stapler_pad \
      EVALUATION.task_config=both

``EVALUATION.task_name`` may also be a Hydra list.  Tasks and phases are run
sequentially in one command so the same GPU, seed, checkpoint, and inference
settings are used.  This entry creates an independent
``RoboTwin/policy/policy_content_adapter`` symlink and never touches the native
``fastwam_policy`` link.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from .p_mode_selection import (
    SEED_BANK_ID_PREFIX,
    build_seed_bank_descriptor,
    seed_bank_identity_payload,
    validate_seed_bank_descriptor,
    validate_formal_protocol_lock_manifest_payload,
    validate_selection_manifest_payload,
)
from .evaluation_protocol import (
    PROFILE as EVALUATION_PROTOCOL_PROFILE,
    SCHEMA_VERSION as EVALUATION_PROTOCOL_SCHEMA_VERSION,
)
from .formal_episode_protocol import (
    EPISODES_PER_CELL as FORMAL_EPISODES_PER_CELL,
    stable_file_identity as formal_stable_file_identity,
    select_realization_cell,
    validate_realization_bank,
    validate_replay_trace,
)
from .robotwin_gpu_runtime import (
    gpu_binding_environment,
    normalize_physical_gpu_id,
    preflight_gpu_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
POLICY_LINK_NAME = "policy_content_adapter"
POLICY_MODULE = "policy_content_adapter.rollout_policy"
DEPLOY_CONFIG = f"policy/{POLICY_LINK_NAME}/deploy_policy.yml"
PINNED_EVALUATOR_MODULE = (
    "experiments.robotwin.policy_content_adapter.pinned_eval_policy"
)
COMPLETED_ROLLOUTS_SCHEMA = "policy_content_adapter.completed_rollouts"
COMPLETED_ROLLOUTS_SCHEMA_VERSION = 5
FORMAL_COMPLETED_ROLLOUTS_SCHEMA_VERSION = 6
STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION = 7
PV2_FOLLOWUP_EVAL100_COMPLETED_ROLLOUTS_SCHEMA_VERSION = 8
OFFICIAL_TASK_CONFIGS = ("demo_clean", "demo_randomized")
TASK_CONFIG_TO_DOMAIN = {
    "demo_clean": "clean",
    "demo_randomized": "official_random",
}
SUPPORTED_CHECKPOINT_CONTROLS = {
    "p_v1",
    "p_v2",
    "c0_base",
    "c0_original",
    "c1_architecture_only",
    "c2_naive_aug",
    "c3_ours",
}
FORMAL_RECORD_CONTROL_ALIASES = {
    "c0_base": "c0_base",
    "c0_original": "c0_base",
    "c1_architecture_only": "c1_architecture_only",
    "c3_ours": "c3_ours",
}
FAIRNESS_RECORD_FIELDS = (
    "base_checkpoint_sha256",
    "dataset_stats_sha256",
    "base_lineage_manifest_sha256",
    "policy_regime",
    "head_init_sha256",
    "gca_init_sha256",
    "stage2_recipe_sha256",
    "p_mode_selection_manifest_sha256",
    "official_sample_sequence_sha256",
    "paired_physical_state_sequence_sha256",
    "matched_stream_contract_sha256",
    "runtime_source_sha256",
)
PV2_FOLLOWUP_STAGE = "mechanism_followup"
PV2_FOLLOWUP_ROLE = "post_hoc_actiondit_mechanism"
PV2_FOLLOWUP_PROTOCOL_KIND = "policy_pv2_actiondit_followup_protocol"


def _build_subprocess_environment(
    *,
    gpu_id: str | int,
    model_base_path: Path,
    gpu_runtime_binding: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    requested_gpu = normalize_physical_gpu_id(gpu_id)
    if gpu_runtime_binding is not None:
        bound_gpu = normalize_physical_gpu_id(
            gpu_runtime_binding.get("physical_gpu_index", "")
        )
        if bound_gpu != requested_gpu:
            raise ValueError(
                "GPU runtime binding differs from requested physical GPU: "
                f"{bound_gpu} != {requested_gpu}"
            )
    env = (
        os.environ.copy()
        if gpu_runtime_binding is None
        else gpu_binding_environment(gpu_runtime_binding)
    )
    python_entries = [str(PROJECT_ROOT), str(SRC_ROOT)]
    python_entries.extend(
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    )
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_entries))
    env["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)
    env["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base_path.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_runtime_binding is not None:
        env["ROBOTWIN_GPU_PREFLIGHT_JSON"] = json.dumps(
            dict(gpu_runtime_binding),
            sort_keys=True,
            separators=(",", ":"),
        )
    return env


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_output_dir(path_value: Any) -> Path:
    resolved = _resolve_path(str(path_value), base=PROJECT_ROOT)
    if not resolved.name:
        raise ValueError(f"Invalid EVALUATION.output_dir: {resolved}")
    return resolved


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(
        cfg.EVALUATION.dataset_stats_path,
        base=PROJECT_ROOT,
    )
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json explicitly."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_index = parts.index("runs")
        if runs_index + 2 >= len(parts):
            raise ValueError(
                "A checkpoint below runs must follow .../runs/<task>/<run>/..., "
                f"got {ckpt_path}"
            )
        return f"{parts[runs_index + 1]}_{parts[runs_index + 2]}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    link = policy_root / POLICY_LINK_NAME
    expected = policy_source_dir.resolve()
    if not link.exists() and not link.is_symlink():
        link.symlink_to(expected, target_is_directory=True)
        return link
    if link.is_symlink() and link.resolve() == expected:
        return link
    if link.is_symlink():
        raise RuntimeError(
            f"Independent policy symlink conflict: {link} -> {link.resolve()}, "
            f"expected {expected}"
        )
    raise RuntimeError(f"Policy link target exists and is not a symlink: {link}")


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(
    overrides: list[str],
    key: str,
    value: Any,
    *,
    skip_none: bool = True,
) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _resolve_tasks(value: Any) -> list[str]:
    if value is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")
    if isinstance(value, (list, tuple, ListConfig)):
        tasks = [str(item).strip() for item in value]
    else:
        tasks = [part.strip() for part in str(value).split(",")]
    tasks = [task for task in tasks if task]
    if not tasks:
        raise ValueError("`EVALUATION.task_name` resolved to an empty task list.")
    return list(dict.fromkeys(tasks))


def _resolve_task_configs(value: Any) -> list[str]:
    text = str(value).strip().lower()
    if text in {"both", "clean_random", "clean+random"}:
        return list(OFFICIAL_TASK_CONFIGS)
    if text in {"clean", "demo_clean"}:
        return ["demo_clean"]
    if text in {"random", "official_random", "demo_randomized"}:
        return ["demo_randomized"]
    raise ValueError(
        "EVALUATION.task_config must be clean/demo_clean, "
        "random/demo_randomized, or both."
    )


def _phase_name(task_config: str) -> str:
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"unsupported RoboTwin task_config {task_config!r}; only "
        "demo_clean/demo_randomized are Policy domains"
    )


def _result_path(output_dir: Path, phase: str) -> Path:
    return output_dir / f"_result_{phase}.txt"


def _parse_success_rate(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Expected RoboTwin result file not found: {path}")
    value: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = float(line.strip())
        except ValueError:
            continue
    if value is None:
        raise ValueError(f"No success rate found in RoboTwin result file: {path}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"RoboTwin success rate must be finite and within [0, 1], got {value}"
        )
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_sha256(value: Any, *, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return digest


def _audited_artifact_sha256(
    metadata: Mapping[str, Any],
    name: str,
) -> str:
    identity = _audited_artifact_identity(metadata, name)
    return _as_sha256(identity.get("sha256"), field=f"artifact_identities.{name}.sha256")


def _audited_artifact_identity(
    metadata: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    identities = metadata.get("artifact_identities")
    if not isinstance(identities, Mapping):
        raise ValueError("checkpoint metadata lacks artifact_identities")
    identity = identities.get(name)
    if not isinstance(identity, Mapping):
        raise ValueError(f"checkpoint lacks audited artifact identity {name!r}")
    if identity.get("verification_status") != "PASS":
        raise ValueError(f"checkpoint artifact identity {name!r} is not audited PASS")
    return identity


def _load_audited_json_artifact(
    metadata: Mapping[str, Any],
    name: str,
) -> tuple[dict[str, Any], str]:
    identity = _audited_artifact_identity(metadata, name)
    if identity.get("required_for_rollout") is not True:
        raise ValueError(f"checkpoint artifact {name!r} is not required for rollout")
    expected_sha = _as_sha256(
        identity.get("sha256"), field=f"artifact_identities.{name}.sha256"
    )
    path = Path(str(identity.get("path", ""))).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"checkpoint artifact {name!r} path is unavailable")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"checkpoint artifact {name!r} changed after audit")
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"checkpoint artifact {name!r} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint artifact {name!r} root must be an object")
    return payload, expected_sha


def _load_pv2_followup_protocol(
    run_config: Mapping[str, Any],
    *,
    declared_artifacts: Mapping[str, Any],
    simulator_seed_bank_manifest_sha256: str,
    simulator_seed_bank_id: str,
) -> tuple[dict[str, Any], str]:
    """Load the narrow post-hoc protocol sidecar embedded in a P-v2 config."""

    if (
        run_config.get("stage") != PV2_FOLLOWUP_STAGE
        or run_config.get("study_role") != PV2_FOLLOWUP_ROLE
        or run_config.get("formal") is not False
    ):
        raise ValueError("checkpoint is not a disclosed P-v2 mechanism follow-up")
    path = Path(str(run_config.get("mechanism_protocol_manifest", ""))).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError("P-v2 follow-up mechanism protocol is unavailable")
    expected_sha = _as_sha256(
        declared_artifacts.get("mechanism_protocol_manifest_sha256"),
        field="run_config.artifacts.mechanism_protocol_manifest_sha256",
    )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("P-v2 follow-up mechanism protocol changed after training")
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError("P-v2 follow-up mechanism protocol is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("P-v2 follow-up mechanism protocol root must be an object")
    if (
        payload.get("kind") != PV2_FOLLOWUP_PROTOCOL_KIND
        or payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
    ):
        raise ValueError("P-v2 follow-up mechanism protocol kind/version/status differs")
    study = payload.get("study_classification")
    if not isinstance(study, Mapping) or (
        study.get("role") != PV2_FOLLOWUP_ROLE
        or study.get("post_hoc_after_primary_results") is not True
        or study.get("primary_experiment_remains_unchanged") is not True
    ):
        raise ValueError("P-v2 follow-up post-hoc disclosure is invalid")
    training = payload.get("locked_training")
    if not isinstance(training, Mapping) or (
        training.get("policy_regime") != "p_v2"
        or training.get("action_dit_trainable") is not True
        or training.get("pilot_training_seed") != 1
        or training.get("max_steps") != 1800
    ):
        raise ValueError("P-v2 follow-up locked training recipe differs")
    pilot = payload.get("pilot_gate")
    if not isinstance(pilot, Mapping) or (
        pilot.get("simulator_seed") != 53
        or pilot.get("episodes_per_task_domain") != 20
        or pilot.get("seed_bank_manifest_sha256")
        != simulator_seed_bank_manifest_sha256
        or pilot.get("seed_bank_id") != simulator_seed_bank_id
    ):
        raise ValueError("P-v2 follow-up pilot seed bank/gate differs")
    history = payload.get("historical_p_mode_selection")
    if not isinstance(history, Mapping) or (
        history.get("winner") != "p_v1"
        or history.get("use") != "historical_context_not_treatment_selection"
        or history.get("sha256")
        != declared_artifacts.get("p_mode_selection_manifest_sha256")
    ):
        raise ValueError("P-v2 follow-up historical P-mode disclosure differs")
    return dict(payload), expected_sha


def _runtime_source_sha256(run_config: Mapping[str, Any]) -> str:
    provenance = run_config.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint run_config lacks runtime_provenance")
    source = provenance.get("fastwam_source")
    if not isinstance(source, Mapping):
        raise ValueError("checkpoint runtime provenance lacks fastwam_source")
    files = source.get("files")
    if (
        source.get("status") != "PASS"
        or source.get("scope") != "all_python_files_under_src_fastwam"
        or not isinstance(files, Mapping)
        or not files
        or source.get("file_count") != len(files)
    ):
        raise ValueError("checkpoint FastWAM source audit is incomplete")
    normalized_files: dict[str, dict[str, Any]] = {}
    for relative_name, raw_identity in sorted(files.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_identity, Mapping):
            raise ValueError(f"FastWAM source identity {relative_name!r} is not a mapping")
        size = _as_nonnegative_int(
            raw_identity.get("size_bytes"),
            field=f"FastWAM source {relative_name}.size_bytes",
        )
        normalized_files[str(relative_name)] = {
            "size_bytes": size,
            "sha256": _as_sha256(
                raw_identity.get("sha256"),
                field=f"FastWAM source {relative_name}.sha256",
            ),
        }
    return _canonical_sha256(
        {
            "schema": "policy_runtime_source_identity_v1",
            "scope": source["scope"],
            "files": normalized_files,
        }
    )


def _project_required_fields(
    value: Any,
    fields: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint run_config lacks {label}")
    missing = [name for name in fields if name not in value]
    if missing:
        raise ValueError(f"checkpoint {label} lacks common recipe fields: {missing}")
    projected = {name: value[name] for name in fields}
    for name, item in projected.items():
        if isinstance(item, str) and item.startswith(("__REQUIRED_", "__SELECT_")):
            raise ValueError(f"checkpoint {label}.{name} is an unresolved placeholder")
    return projected


def _stage2_common_recipe(
    run_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Build the hashable common Stage-2 recipe, excluding experiment variables.

    Deliberately absent are ``control``, paired/action-data paths, supervision
    mode, and loss coefficients.  Those are the C1/C2/C3 treatment variables.
    """

    training_fields = (
        "max_steps",
        "official_batch_size",
        "paired_groups_per_batch",
        "world_size",
        "gradient_accumulation_steps",
        "effective_official_global_batch",
        "effective_paired_groups_per_step",
        "num_workers",
        "mixed_precision",
        "model_dtype",
        "max_grad_norm",
        "require_cuda",
        "separate_stream_rng",
        "preserve_official_sequence_across_controls",
    )
    optimizer_fields = (
        "name",
        "lr_scheduler",
        "trainable_parameter_dtype",
        "head_adapter_lr",
        "action_dit_lr",
        "weight_decay",
        "betas",
    )
    official_fields = (
        "selection_mode",
        "expected_clean_per_task",
        "expected_random_per_task",
        "expected_total_per_task",
        "sampling_mode",
        "task_balanced",
        "balanced_tasks",
        "domain_label",
    )
    policy_fields = (
        "content_layer",
        "input_token_count",
        "input_dim",
        "queries",
        "content_dim",
        "attention_heads",
        "adapter_count",
        "action_hidden_dim",
        "gate_init_exact",
    )
    architecture_fields = (
        "content_head",
        "gated_action_adapter",
        "adapter_injection",
        "adapter_residual",
        "mean_pool_on_policy_path",
    )
    training = _project_required_fields(
        run_config.get("training"), training_fields, label="training"
    )
    optimizer = _project_required_fields(
        run_config.get("optimizer"), optimizer_fields, label="optimizer"
    )
    official = _project_required_fields(
        run_config.get("official"), official_fields, label="official"
    )
    policy = _project_required_fields(
        run_config.get("policy"), policy_fields, label="policy"
    )
    architecture = _project_required_fields(
        run_config.get("architecture"), architecture_fields, label="architecture"
    )
    max_steps = _as_positive_int(training["max_steps"], field="training.max_steps")
    if checkpoint_step != max_steps:
        raise ValueError(
            f"checkpoint step {checkpoint_step} differs from training.max_steps {max_steps}"
        )
    world_size = _as_positive_int(training["world_size"], field="training.world_size")
    accumulation = _as_positive_int(
        training["gradient_accumulation_steps"],
        field="training.gradient_accumulation_steps",
    )
    local_official = _as_positive_int(
        training["official_batch_size"], field="training.official_batch_size"
    )
    local_paired = _as_positive_int(
        training["paired_groups_per_batch"],
        field="training.paired_groups_per_batch",
    )
    if training["effective_official_global_batch"] != local_official * world_size * accumulation:
        raise ValueError("checkpoint effective official global batch is inconsistent")
    if training["effective_paired_groups_per_step"] != local_paired * world_size * accumulation:
        raise ValueError("checkpoint effective paired global groups are inconsistent")
    component_identities = {
        name: _audited_artifact_sha256(metadata, name)
        for name in ("official_manifest", "vae", "text_encoder", "tokenizer")
    }
    return {
        "schema": "policy_stage2_common_recipe_v1",
        "tasks": list(run_config.get("tasks", ())),
        "training": training,
        "optimizer": optimizer,
        "official": official,
        "policy": policy,
        "architecture": architecture,
        "common_artifacts": component_identities,
    }


def _as_nonnegative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _as_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _as_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return bool(value)


def _checkpoint_evaluation_contract(
    provenance: Mapping[str, Any],
    *,
    requested_tasks: Sequence[str],
    requested_domains: Sequence[str],
    episodes_per_task: int,
) -> dict[str, Any]:
    """Extract formal evaluation identity from safely loaded checkpoint metadata.

    ``provenance`` must be the mapping returned by the mmap-backed,
    ``weights_only=True`` checkpoint reader in :mod:`rollout_policy`.  Values
    are never inferred from a path, output directory, or command-line seed.
    """

    metadata = provenance.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("safe checkpoint provenance lacks metadata")
    checkpoint_identity = provenance.get("policy_checkpoint")
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError("safe checkpoint provenance lacks policy_checkpoint identity")
    checkpoint_path = Path(str(checkpoint_identity.get("path", ""))).expanduser()
    if not checkpoint_path.is_absolute():
        raise ValueError("safe checkpoint provenance path must be absolute")
    checkpoint_size = _as_positive_int(
        checkpoint_identity.get("size_bytes"), field="checkpoint size_bytes"
    )
    checkpoint_mtime_ns = _as_nonnegative_int(
        checkpoint_identity.get("mtime_ns"), field="checkpoint mtime_ns"
    )
    run_config = metadata.get("run_config")
    if not isinstance(run_config, Mapping):
        raise ValueError("checkpoint metadata lacks run_config")

    control = run_config.get("control")
    if not isinstance(control, str) or not control.strip():
        raise ValueError("checkpoint run_config lacks a non-empty control")
    control = control.strip()
    if control not in SUPPORTED_CHECKPOINT_CONTROLS:
        raise ValueError(f"checkpoint has unsupported control {control!r}")
    stage = str(run_config.get("stage", "")).strip()
    if not stage:
        raise ValueError("checkpoint run_config lacks stage")
    if control in {"p_v1", "p_v2"} and stage not in {"smoke", "dev_pilot"}:
        raise ValueError("P-v1/P-v2 checkpoint stage must be smoke or dev_pilot")
    is_pv2_followup = (
        control in {"c1_architecture_only", "c3_ours"}
        and stage == PV2_FOLLOWUP_STAGE
        and run_config.get("study_role") == PV2_FOLLOWUP_ROLE
        and run_config.get("formal") is False
    )
    is_c0 = control in {"c0_base", "c0_original"}

    training = run_config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint run_config lacks training metadata")
    if is_c0:
        if "seed" not in training or training.get("seed") is not None:
            raise ValueError("fixed C0 checkpoint training.seed must be null")
        training_seed: int | None = None
    else:
        training_seed = _as_nonnegative_int(
            training.get("seed"), field="checkpoint training.seed"
        )

    evaluation = run_config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("checkpoint run_config lacks evaluation metadata")
    protocol_id = evaluation.get("rollout_protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("checkpoint evaluation lacks rollout_protocol_id")
    protocol_id = protocol_id.strip()
    if protocol_id.startswith("__") or protocol_id.endswith("__"):
        raise ValueError("checkpoint rollout_protocol_id is an unresolved placeholder")
    declared_seed_bank_id = evaluation.get("simulator_seed_bank_id")
    if (
        not isinstance(declared_seed_bank_id, str)
        or not declared_seed_bank_id.strip()
    ):
        raise ValueError("checkpoint evaluation lacks simulator_seed_bank_id")
    declared_seed_bank_id = declared_seed_bank_id.strip()
    if declared_seed_bank_id.startswith("__") or declared_seed_bank_id.endswith(
        "__"
    ):
        raise ValueError(
            "checkpoint simulator_seed_bank_id is an unresolved placeholder"
        )
    declared_seed_bank_purpose = str(
        evaluation.get("simulator_seed_bank_purpose", "")
    ).strip()
    if not declared_seed_bank_purpose:
        raise ValueError("checkpoint evaluation lacks simulator_seed_bank_purpose")

    declared_episodes = _as_positive_int(
        evaluation.get("episodes_per_task"),
        field="checkpoint evaluation.episodes_per_task",
    )
    actual_episodes = _as_positive_int(
        episodes_per_task, field="runtime eval_num_episodes"
    )
    if declared_episodes != actual_episodes:
        raise ValueError(
            "runtime episodes differ from checkpoint rollout protocol: "
            f"{actual_episodes} != {declared_episodes}"
        )
    seed_bank_payload, simulator_seed_bank_manifest_sha256 = _load_audited_json_artifact(
        metadata, "simulator_seed_bank_manifest"
    )
    try:
        simulator_seed_bank = validate_seed_bank_descriptor(
            seed_bank_payload,
            expected_purpose=declared_seed_bank_purpose,
        )
    except ValueError as exc:
        raise ValueError(f"audited simulator seed-bank manifest is invalid: {exc}") from exc
    if simulator_seed_bank["simulator_seed_bank_id"] != declared_seed_bank_id:
        raise ValueError("checkpoint seed-bank manifest id differs from evaluation config")
    if simulator_seed_bank["episodes_per_cell"] != declared_episodes:
        raise ValueError("checkpoint seed-bank manifest episode count differs")
    formal_protocol_lock_manifest_sha256: str | None = None
    if declared_seed_bank_purpose == "final_test":
        lock_payload, formal_protocol_lock_manifest_sha256 = _load_audited_json_artifact(
            metadata, "formal_protocol_lock_manifest"
        )
        try:
            formal_lock = validate_formal_protocol_lock_manifest_payload(lock_payload)
        except ValueError as exc:
            raise ValueError(f"audited formal protocol lock is invalid: {exc}") from exc
        lock_ancestry = simulator_seed_bank.get("lock_ancestry")
        if not isinstance(lock_ancestry, Mapping):
            raise ValueError("final-test seed bank lacks lock ancestry")
        lock_identity = lock_ancestry.get("formal_protocol_lock_manifest")
        if (
            not isinstance(lock_identity, Mapping)
            or lock_identity.get("sha256") != formal_protocol_lock_manifest_sha256
        ):
            raise ValueError("final-test seed bank binds a different formal protocol lock")
    elif declared_seed_bank_purpose == "dev_selection":
        if not (
            (control in {"p_v1", "p_v2"} and stage == "dev_pilot")
            or is_pv2_followup
        ):
            raise ValueError("dev_selection seed bank is reserved for P-mode dev pilots")

    declared_tasks_raw = evaluation.get("tasks")
    if not isinstance(declared_tasks_raw, (list, tuple, ListConfig)):
        raise ValueError("checkpoint evaluation.tasks must be a list")
    declared_tasks = tuple(str(value).strip() for value in declared_tasks_raw)
    if any(not value for value in declared_tasks) or len(set(declared_tasks)) != len(
        declared_tasks
    ):
        raise ValueError("checkpoint evaluation.tasks is empty or contains duplicates")
    undeclared_tasks = sorted(set(requested_tasks) - set(declared_tasks))
    if undeclared_tasks:
        raise ValueError(
            f"runtime tasks are absent from checkpoint protocol: {undeclared_tasks}"
        )

    declared_domains_raw = evaluation.get("required_domains")
    if not isinstance(declared_domains_raw, (list, tuple, ListConfig)):
        raise ValueError("checkpoint evaluation.required_domains must be a list")
    declared_domains = tuple(str(value).strip() for value in declared_domains_raw)
    if (
        any(value not in {"clean", "official_random"} for value in declared_domains)
        or len(set(declared_domains)) != len(declared_domains)
    ):
        raise ValueError(
            "checkpoint evaluation.required_domains must contain only unique "
            "clean/official_random domains"
        )
    undeclared_domains = sorted(set(requested_domains) - set(declared_domains))
    if undeclared_domains:
        raise ValueError(
            "runtime domains are absent from checkpoint protocol: "
            f"{undeclared_domains}"
        )

    checkpoint_step = _as_nonnegative_int(
        metadata.get("step"), field="checkpoint step"
    )
    regime = metadata.get("regime")
    if not isinstance(regime, str) or regime not in {"p_v1", "p_v2"}:
        raise ValueError("checkpoint metadata lacks a supported policy regime")
    base_checkpoint_sha256 = _as_sha256(
        metadata.get("base_checkpoint", {}).get("sha256")
        if isinstance(metadata.get("base_checkpoint"), Mapping)
        else None,
        field="checkpoint base_checkpoint.sha256",
    )
    dataset_stats_sha256 = _audited_artifact_sha256(metadata, "dataset_stats")
    base_lineage_manifest_sha256 = _audited_artifact_sha256(
        metadata, "base_lineage_manifest"
    )
    runtime_source_sha256 = _runtime_source_sha256(run_config)
    dev_pilot_artifact_shas: dict[str, str] | None = None
    if control in {"p_v1", "p_v2"}:
        dev_pilot_artifact_shas = {
            "official_manifest_sha256": _audited_artifact_sha256(
                metadata, "official_manifest"
            ),
            "paired_action_manifest_sha256": _audited_artifact_sha256(
                metadata, "paired_action_manifest"
            ),
            "paired_state_bank_sha256": _audited_artifact_sha256(
                metadata, "paired_state_bank"
            ),
            "paired_text_cache_sha256": _audited_artifact_sha256(
                metadata, "paired_text_cache"
            ),
            "paired_cache_sha256": _audited_artifact_sha256(
                metadata, "paired_train_cache"
            ),
        }

    stage2_recipe: dict[str, Any] | None
    policy_regime: str | None
    head_init_sha256: str | None
    gca_init_sha256: str | None
    stage2_recipe_sha256: str | None
    p_mode_selection_manifest_sha256: str | None
    official_sample_sequence_sha256: str | None
    paired_physical_state_sequence_sha256: str | None
    matched_stream_contract_sha256: str | None
    mechanism_protocol_manifest_sha256: str | None = None
    if is_c0:
        semantics = run_config.get("c0_semantics")
        if (
            run_config.get("kind") != "policy_c0_eval_transport"
            or not isinstance(semantics, Mapping)
            or semantics.get("stage2_training") is not False
            or semantics.get("action_expert_overlay") is not False
            or semantics.get("head_gca_effect_on_action") != "none_exact_zero_gate"
            or training.get("stage2_steps") != 0
            or checkpoint_step != 0
            or regime != "p_v1"
        ):
            raise ValueError("C0 checkpoint does not prove native zero-Stage-2 semantics")
        if semantics.get("base_lineage_manifest_sha256") != base_lineage_manifest_sha256:
            raise ValueError("C0 author-release lineage identity is internally inconsistent")
        policy_regime = None
        head_init_sha256 = None
        gca_init_sha256 = None
        stage2_recipe = None
        stage2_recipe_sha256 = None
        p_mode_selection_manifest_sha256 = None
        official_sample_sequence_sha256 = None
        paired_physical_state_sequence_sha256 = None
        matched_stream_contract_sha256 = None
    else:
        if run_config.get("kind") != "policy_content_adapter_run":
            raise ValueError("trained Policy checkpoint has the wrong run_config kind")
        declared_artifacts = run_config.get("artifacts")
        if not isinstance(declared_artifacts, Mapping):
            raise ValueError("trained Policy checkpoint lacks run_config.artifacts")
        declared_sha_bindings = {
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "dataset_stats_sha256": dataset_stats_sha256,
            "base_lineage_manifest_sha256": base_lineage_manifest_sha256,
        }
        for field, expected_sha in declared_sha_bindings.items():
            if declared_artifacts.get(field) != expected_sha:
                raise ValueError(f"checkpoint run_config.artifacts.{field} is inconsistent")
        if (
            declared_artifacts.get("simulator_seed_bank_manifest_sha256")
            != simulator_seed_bank_manifest_sha256
        ):
            raise ValueError(
                "checkpoint run_config.artifacts.simulator_seed_bank_manifest_sha256 "
                "is inconsistent"
            )
        if is_pv2_followup:
            _, mechanism_protocol_manifest_sha256 = _load_pv2_followup_protocol(
                run_config,
                declared_artifacts=declared_artifacts,
                simulator_seed_bank_manifest_sha256=(
                    simulator_seed_bank_manifest_sha256
                ),
                simulator_seed_bank_id=declared_seed_bank_id,
            )
        resolved_base = run_config.get("resolved_base_checkpoint_identity")
        resolved_artifacts = run_config.get("resolved_artifact_identities")
        if not isinstance(resolved_base, Mapping) or not isinstance(resolved_artifacts, Mapping):
            raise ValueError("trained Policy checkpoint lacks resolved artifact audits")
        if resolved_base.get("sha256") != base_checkpoint_sha256:
            raise ValueError("resolved base checkpoint identity is inconsistent")
        for artifact_name, expected_sha in (
            ("dataset_stats", dataset_stats_sha256),
            ("base_lineage_manifest", base_lineage_manifest_sha256),
        ):
            resolved = resolved_artifacts.get(artifact_name)
            if not isinstance(resolved, Mapping) or resolved.get("sha256") != expected_sha:
                raise ValueError(f"resolved artifact {artifact_name!r} is inconsistent")
        policy = run_config.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("trained Policy checkpoint lacks policy config")
        declared_regime = str(policy.get("regime", "")).lower().replace("-", "_")
        if declared_regime != regime:
            raise ValueError("checkpoint payload regime differs from run_config policy.regime")
        initialization = run_config.get("resolved_initialization")
        if not isinstance(initialization, Mapping):
            raise ValueError("trained Policy checkpoint lacks resolved initialization audit")
        head_init_sha256 = _as_sha256(
            initialization.get("training_fp32_content_head_sha256"),
            field="resolved_initialization.training_fp32_content_head_sha256",
        )
        gca_init_sha256 = _as_sha256(
            initialization.get("training_fp32_adapter_sha256"),
            field="resolved_initialization.training_fp32_adapter_sha256",
        )
        if initialization.get("source_fp32_content_head_sha256") != head_init_sha256:
            raise ValueError("Content Head initialization restoration audit is inconsistent")
        if initialization.get("source_fp32_adapter_sha256") != gca_init_sha256:
            raise ValueError("GCA initialization restoration audit is inconsistent")
        policy_regime = regime
        sequence_audit = run_config.get("resolved_training_sequence_audit")
        if not isinstance(sequence_audit, Mapping) or sequence_audit.get("status") != "PASS":
            raise ValueError("trained Policy checkpoint lacks PASS training sequence audit")
        official_sample_sequence_sha256 = _as_sha256(
            sequence_audit.get("official_sample_sequence_sha256"),
            field="resolved_training_sequence_audit.official_sample_sequence_sha256",
        )
        paired_physical_state_sequence_sha256 = _as_sha256(
            sequence_audit.get("paired_physical_state_sequence_sha256"),
            field="resolved_training_sequence_audit.paired_physical_state_sequence_sha256",
        )
        matched_stream_contract_sha256 = _as_sha256(
            sequence_audit.get("matched_stream_contract_sha256"),
            field="resolved_training_sequence_audit.matched_stream_contract_sha256",
        )
        loss_config = run_config.get("loss")
        if not isinstance(loss_config, Mapping):
            raise ValueError("trained Policy checkpoint lacks loss config")
        try:
            lambda_contrastive: float | None = float(
                loss_config.get("lambda_contrastive")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("trained Policy checkpoint lacks numeric lambda_contrastive") from exc
        if not math.isfinite(lambda_contrastive) or lambda_contrastive < 0.0:
            raise ValueError("trained Policy checkpoint lambda_contrastive is invalid")
        if control == "c1_architecture_only" and lambda_contrastive != 0.0:
            raise ValueError("C1 checkpoint must set lambda_contrastive=0")
        if control == "c3_ours" and lambda_contrastive <= 0.0:
            raise ValueError("C3 checkpoint must enable contrastive supervision")
        stage2_recipe = _stage2_common_recipe(
            run_config,
            metadata,
            checkpoint_step=checkpoint_step,
        )
        stage2_recipe_sha256 = _canonical_sha256(stage2_recipe)
        selection_role: str | None = None
        if control in {"p_v1", "p_v2"}:
            selection_role = str(run_config.get("selection_role", ""))
            if stage == "dev_pilot" and (
                selection_role != "c1_lambda0" or lambda_contrastive != 0.0
            ):
                raise ValueError("P-mode dev checkpoint must be C1 lambda=0")
            if stage == "smoke" and (
                selection_role != "engineering_method_smoke"
                or lambda_contrastive <= 0.0
            ):
                raise ValueError("P-mode engineering smoke must exercise contrastive gradients")
        if control in {"c1_architecture_only", "c2_naive_aug", "c3_ours"}:
            selection_payload, p_mode_selection_manifest_sha256 = (
                _load_audited_json_artifact(metadata, "p_mode_selection_manifest")
            )
            try:
                validated_selection = validate_selection_manifest_payload(
                    selection_payload
                )
            except ValueError as exc:
                raise ValueError(
                    f"audited P-mode selection manifest is invalid: {exc}"
                ) from exc
            if is_pv2_followup:
                if (
                    validated_selection.get("winner") != "p_v1"
                    or policy_regime != "p_v2"
                ):
                    raise ValueError(
                        "post-hoc P-v2 follow-up must disclose the historical P-v1 winner"
                    )
            elif validated_selection.get("winner") != policy_regime:
                raise ValueError(
                    "P-mode selection winner differs from checkpoint policy regime"
                )
            shared_selection = validated_selection.get("shared_candidate_identity")
            if not isinstance(shared_selection, Mapping):
                raise ValueError("P-mode selection lacks shared candidate ancestry")
            selection_ancestry = {
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "dataset_stats_sha256": dataset_stats_sha256,
                "base_lineage_manifest_sha256": base_lineage_manifest_sha256,
                "runtime_source_sha256": runtime_source_sha256,
                "official_manifest_sha256": _audited_artifact_sha256(
                    metadata, "official_manifest"
                ),
                "paired_action_manifest_sha256": _audited_artifact_sha256(
                    metadata, "paired_action_manifest"
                ),
                "paired_state_bank_sha256": _audited_artifact_sha256(
                    metadata, "paired_state_bank"
                ),
                "paired_text_cache_sha256": _audited_artifact_sha256(
                    metadata, "paired_text_cache"
                ),
                "paired_cache_sha256": _audited_artifact_sha256(
                    metadata, "paired_train_cache"
                ),
            }
            for field, expected in selection_ancestry.items():
                if shared_selection.get(field) != expected:
                    raise ValueError(
                        f"formal checkpoint ancestry differs from P-mode selection: {field}"
                    )
            if (
                declared_artifacts.get("p_mode_selection_manifest_sha256")
                != p_mode_selection_manifest_sha256
            ):
                raise ValueError(
                    "checkpoint run_config P-mode selection SHA is inconsistent"
                )
            if declared_seed_bank_purpose == "final_test":
                lock_ancestry = simulator_seed_bank["lock_ancestry"]
                selected_identity = lock_ancestry.get("p_mode_selection_manifest")
                if (
                    not isinstance(selected_identity, Mapping)
                    or selected_identity.get("sha256")
                    != p_mode_selection_manifest_sha256
                ):
                    raise ValueError("final-test seed bank binds a different P-mode selection")
                if (
                    declared_artifacts.get("formal_protocol_lock_manifest_sha256")
                    != formal_protocol_lock_manifest_sha256
                ):
                    raise ValueError("checkpoint formal protocol lock SHA is inconsistent")
                if (
                    formal_lock["base_lineage_manifest"]["sha256"]
                    != base_lineage_manifest_sha256
                    or formal_lock["p_mode_selection_manifest"]["sha256"]
                    != p_mode_selection_manifest_sha256
                    or formal_lock["selected_policy_regime"] != policy_regime
                ):
                    raise ValueError("formal protocol lock ancestry/winner is inconsistent")
                if control in {"c1_architecture_only", "c3_ours"}:
                    lock_crosscheck = run_config.get("resolved_formal_protocol_lock")
                    if not isinstance(lock_crosscheck, Mapping) or lock_crosscheck.get("status") != "PASS":
                        raise ValueError("formal checkpoint lacks PASS protocol-lock projection crosscheck")
                    locked_row = formal_lock["resolved_configs"][control][int(training_seed) - 1]
                    expected_crosscheck = {
                        "formal_protocol_lock_manifest_sha256": formal_protocol_lock_manifest_sha256,
                        "control": control,
                        "training_seed": training_seed,
                        "selected_policy_regime": policy_regime,
                        "lambda_contrastive": float(run_config["loss"]["lambda_contrastive"]),
                        "protocol_projection_sha256": locked_row["protocol_projection_sha256"],
                        "source_config": locked_row["source_config"],
                    }
                    for field, expected in expected_crosscheck.items():
                        if lock_crosscheck.get(field) != expected:
                            raise ValueError(
                                f"formal protocol-lock crosscheck differs: {field}"
                            )
        else:
            if run_config.get("p_mode_selection_manifest") is not None or declared_artifacts.get(
                "p_mode_selection_manifest_sha256"
            ) is not None:
                raise ValueError("P-v1/P-v2 checkpoint must not bind a selection manifest")
            p_mode_selection_manifest_sha256 = None
    formal_flag = run_config.get("formal")
    if not isinstance(formal_flag, bool):
        raise ValueError("checkpoint run_config.formal must be boolean")
    has_formal_control = control in FORMAL_RECORD_CONTROL_ALIASES
    formal_evaluation_eligible = (
        has_formal_control and formal_flag and declared_seed_bank_purpose == "final_test"
    )
    if has_formal_control and formal_flag != (declared_seed_bank_purpose == "final_test"):
        raise ValueError("formal control eligibility and final-test seed-bank purpose disagree")
    return {
        "control": control,
        "stage": stage,
        "training_seed": training_seed,
        "regime": regime,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "dataset_stats_sha256": dataset_stats_sha256,
        "base_lineage_manifest_sha256": base_lineage_manifest_sha256,
        "policy_regime": policy_regime,
        "head_init_sha256": head_init_sha256,
        "gca_init_sha256": gca_init_sha256,
        "stage2_recipe": stage2_recipe,
        "stage2_recipe_sha256": stage2_recipe_sha256,
        "p_mode_selection_manifest_sha256": p_mode_selection_manifest_sha256,
        "official_sample_sequence_sha256": official_sample_sequence_sha256,
        "paired_physical_state_sequence_sha256": paired_physical_state_sequence_sha256,
        "matched_stream_contract_sha256": matched_stream_contract_sha256,
        "runtime_source_sha256": runtime_source_sha256,
        "dev_pilot_artifact_shas": dev_pilot_artifact_shas,
        "selection_role": selection_role if not is_c0 else None,
        "lambda_contrastive": lambda_contrastive if not is_c0 else None,
        "checkpoint_step": checkpoint_step,
        "checkpoint_identity": {
            "path": str(checkpoint_path.resolve()),
            "size_bytes": checkpoint_size,
            "mtime_ns": checkpoint_mtime_ns,
        },
        "rollout_protocol_id": protocol_id,
        "simulator_seed_bank_id": declared_seed_bank_id,
        "simulator_seed_bank_purpose": declared_seed_bank_purpose,
        "simulator_seed_bank_manifest_sha256": simulator_seed_bank_manifest_sha256,
        "formal_protocol_lock_manifest_sha256": formal_protocol_lock_manifest_sha256,
        "mechanism_protocol_manifest_sha256": mechanism_protocol_manifest_sha256,
        "formal_evaluation_eligible": formal_evaluation_eligible,
        "simulator_seed_bank_descriptor": simulator_seed_bank,
        "declared_tasks": list(declared_tasks),
        "declared_domains": list(declared_domains),
        "declared_episodes_per_task": declared_episodes,
        "source": "compact_checkpoint.audited_run_config_and_artifact_identities",
        "safe_load": "torch.load(weights_only=True,mmap=True)",
    }


def _build_rollout_settings(
    cfg: DictConfig,
    *,
    sim_cfg_path: Path,
    sim_task: Any,
) -> dict[str, Any]:
    evaluation = OmegaConf.to_container(cfg.EVALUATION, resolve=True)
    if not isinstance(evaluation, Mapping):
        raise ValueError("EVALUATION config must resolve to a mapping")
    instruction_type = str(evaluation.get("instruction_type", "")).strip()
    if not instruction_type:
        raise ValueError("EVALUATION.instruction_type must be non-empty")
    return {
        "schema": "robotwin.policy_content_adapter.rollout_settings",
        "schema_version": 1,
        "policy_module": POLICY_MODULE,
        "deploy_config": DEPLOY_CONFIG,
        "sim_cfg_path": str(sim_cfg_path.resolve()),
        "sim_task": None if sim_task is None else str(sim_task),
        "instruction_type": instruction_type,
        "episodes_per_task": _as_positive_int(
            evaluation.get("eval_num_episodes"),
            field="EVALUATION.eval_num_episodes",
        ),
        "mixed_precision": str(cfg.mixed_precision),
        "device": str(evaluation.get("device")),
        "action_horizon": evaluation.get("action_horizon"),
        "replan_steps": evaluation.get("replan_steps"),
        "num_inference_steps": evaluation.get("num_inference_steps"),
        "sigma_shift": evaluation.get("sigma_shift"),
        "text_cfg_scale": evaluation.get("text_cfg_scale"),
        "negative_prompt": str(evaluation.get("negative_prompt", "")),
        "rand_device": str(evaluation.get("rand_device")),
        "tiled": _as_bool(evaluation.get("tiled"), field="EVALUATION.tiled"),
        "timing_enabled": _as_bool(
            evaluation.get("timing_enabled"),
            field="EVALUATION.timing_enabled",
        ),
        "skip_get_obs_within_replan": _as_bool(
            evaluation.get("skip_get_obs_within_replan"),
            field="EVALUATION.skip_get_obs_within_replan",
        ),
    }


def _build_simulator_seed_bank(
    *,
    simulator_seed: int,
    episodes_per_task: int,
    evaluator_source: Path,
    purpose: str,
    disjoint_from: Sequence[Mapping[str, Any]] = (),
    lock_ancestry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_seed_bank_descriptor(
        simulator_seed=simulator_seed,
        episodes_per_cell=episodes_per_task,
        evaluator_source=evaluator_source,
        purpose=purpose,
        disjoint_from=disjoint_from,
        lock_ancestry=lock_ancestry,
    )


def _seed_bank_identity_payload(seed_bank: Mapping[str, Any]) -> dict[str, Any]:
    return seed_bank_identity_payload(seed_bank)


def _verify_checkpoint_seed_bank(
    checkpoint_contract: Mapping[str, Any],
    simulator_seed_bank: Mapping[str, Any],
) -> str:
    declared = str(checkpoint_contract.get("simulator_seed_bank_id", "")).strip()
    computed = str(simulator_seed_bank.get("simulator_seed_bank_id", "")).strip()
    if not declared or not computed or declared != computed:
        raise ValueError(
            "runtime simulator seed bank differs from checkpoint protocol: "
            f"{computed!r} != {declared!r}"
        )
    declared_purpose = str(
        checkpoint_contract.get("simulator_seed_bank_purpose", "")
    )
    if simulator_seed_bank.get("purpose") != declared_purpose:
        raise ValueError("runtime simulator seed-bank purpose differs from checkpoint")
    configured = checkpoint_contract.get("simulator_seed_bank_descriptor")
    if configured != simulator_seed_bank:
        raise ValueError("runtime simulator seed-bank descriptor differs from checkpoint artifact")
    return declared


def _validate_pv2_eval100_checkpoint_contract(
    amendment: Mapping[str, Any],
    *,
    checkpoint_contract: Mapping[str, Any],
    checkpoint_path: str | Path,
    requested_tasks: Sequence[str],
    requested_domains: Sequence[str],
    simulator_seed: int,
    episodes_per_task: int,
) -> dict[str, Any]:
    """Validate the narrow 20-to-100 episode override for one checkpoint."""

    from .pv2_followup_eval100_amendment import (
        DOMAINS as PV2_EVAL100_DOMAINS,
        PROFILE as PV2_EVAL100_PROFILE,
        RUNTIME_EPISODES_PER_CELL as PV2_EVAL100_EPISODES,
        SIMULATOR_SEED as PV2_EVAL100_SEED,
        TASKS as PV2_EVAL100_TASKS,
        matching_checkpoint_row,
    )

    if amendment.get("profile") != PV2_EVAL100_PROFILE:
        raise ValueError("P-v2 eval100 amendment profile differs")
    if simulator_seed != PV2_EVAL100_SEED:
        raise ValueError(
            f"P-v2 eval100 amendment requires simulator seed {PV2_EVAL100_SEED}"
        )
    if episodes_per_task != PV2_EVAL100_EPISODES:
        raise ValueError(
            f"P-v2 eval100 amendment requires {PV2_EVAL100_EPISODES} episodes"
        )
    if tuple(requested_tasks) != tuple(PV2_EVAL100_TASKS) or tuple(
        requested_domains
    ) != tuple(PV2_EVAL100_DOMAINS):
        raise ValueError(
            "P-v2 eval100 amendment requires all three tasks and both official domains"
        )
    if (
        checkpoint_contract.get("stage") != PV2_FOLLOWUP_STAGE
        or checkpoint_contract.get("policy_regime") != "p_v2"
        or checkpoint_contract.get("training_seed") != 1
        or checkpoint_contract.get("checkpoint_step") != 1800
        or checkpoint_contract.get("formal_evaluation_eligible") is not False
    ):
        raise ValueError("checkpoint is not the authorized P-v2 mechanism pilot")
    if (
        checkpoint_contract.get("mechanism_protocol_manifest_sha256")
        != amendment["mechanism_protocol"]["sha256"]
    ):
        raise ValueError("P-v2 eval100 amendment mechanism protocol differs")
    if (
        checkpoint_contract.get("simulator_seed_bank_id")
        != amendment["original_evaluation"]["seed_bank_id"]
        or checkpoint_contract.get("simulator_seed_bank_manifest_sha256")
        != amendment["original_evaluation"]["seed_bank"]["sha256"]
    ):
        raise ValueError(
            "P-v2 eval100 amendment does not bind the checkpoint's original bank"
        )
    return matching_checkpoint_row(
        amendment,
        checkpoint_path=checkpoint_path,
        control=str(checkpoint_contract["control"]),
        training_seed=int(checkpoint_contract["training_seed"]),
        checkpoint_step=int(checkpoint_contract["checkpoint_step"]),
    )


def _evaluation_record_control(checkpoint_control: str) -> str:
    try:
        return FORMAL_RECORD_CONTROL_ALIASES[checkpoint_control]
    except KeyError as exc:
        raise ValueError(
            f"checkpoint control {checkpoint_control!r} is not a formal "
            "evaluation_protocol control"
        ) from exc


def _fairness_identity_from_checkpoint_contract(
    checkpoint_contract: Mapping[str, Any],
    *,
    evaluation_control: str,
) -> dict[str, Any]:
    """Revalidate the checkpoint-derived identity embedded in a completed run."""

    ancestry = {
        name: _as_sha256(checkpoint_contract.get(name), field=f"checkpoint_contract.{name}")
        for name in (
            "base_checkpoint_sha256",
            "dataset_stats_sha256",
            "base_lineage_manifest_sha256",
            "runtime_source_sha256",
        )
    }
    if evaluation_control == "c0_base":
        for name in (
            "policy_regime",
            "head_init_sha256",
            "gca_init_sha256",
            "stage2_recipe_sha256",
            "p_mode_selection_manifest_sha256",
            "official_sample_sequence_sha256",
            "paired_physical_state_sequence_sha256",
            "matched_stream_contract_sha256",
        ):
            if checkpoint_contract.get(name) is not None:
                raise ValueError(f"C0 checkpoint_contract.{name} must be null")
        if checkpoint_contract.get("stage2_recipe") is not None:
            raise ValueError("C0 checkpoint_contract.stage2_recipe must be null")
        stage2 = {
            "policy_regime": None,
            "head_init_sha256": None,
            "gca_init_sha256": None,
            "stage2_recipe_sha256": None,
            "p_mode_selection_manifest_sha256": None,
            "official_sample_sequence_sha256": None,
            "paired_physical_state_sequence_sha256": None,
            "matched_stream_contract_sha256": None,
        }
    else:
        policy_regime = str(checkpoint_contract.get("policy_regime", ""))
        if policy_regime not in {"p_v1", "p_v2"}:
            raise ValueError("trained checkpoint_contract.policy_regime must be p_v1/p_v2")
        recipe = checkpoint_contract.get("stage2_recipe")
        if not isinstance(recipe, Mapping):
            raise ValueError("trained checkpoint_contract lacks Stage-2 recipe")
        recipe_sha = _as_sha256(
            checkpoint_contract.get("stage2_recipe_sha256"),
            field="checkpoint_contract.stage2_recipe_sha256",
        )
        if _canonical_sha256(recipe) != recipe_sha:
            raise ValueError("checkpoint_contract Stage-2 recipe SHA-256 differs")
        stage2 = {
            "policy_regime": policy_regime,
            "head_init_sha256": _as_sha256(
                checkpoint_contract.get("head_init_sha256"),
                field="checkpoint_contract.head_init_sha256",
            ),
            "gca_init_sha256": _as_sha256(
                checkpoint_contract.get("gca_init_sha256"),
                field="checkpoint_contract.gca_init_sha256",
            ),
            "stage2_recipe_sha256": recipe_sha,
            "p_mode_selection_manifest_sha256": _as_sha256(
                checkpoint_contract.get("p_mode_selection_manifest_sha256"),
                field="checkpoint_contract.p_mode_selection_manifest_sha256",
            ),
            "official_sample_sequence_sha256": _as_sha256(
                checkpoint_contract.get("official_sample_sequence_sha256"),
                field="checkpoint_contract.official_sample_sequence_sha256",
            ),
            "paired_physical_state_sequence_sha256": _as_sha256(
                checkpoint_contract.get("paired_physical_state_sequence_sha256"),
                field="checkpoint_contract.paired_physical_state_sequence_sha256",
            ),
            "matched_stream_contract_sha256": _as_sha256(
                checkpoint_contract.get("matched_stream_contract_sha256"),
                field="checkpoint_contract.matched_stream_contract_sha256",
            ),
        }
    return {**ancestry, **stage2}


def _records_from_completed_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate one completed manifest and reconstruct formal protocol records."""

    if payload.get("schema") != COMPLETED_ROLLOUTS_SCHEMA:
        raise ValueError("not a policy-content-adapter completed-rollouts manifest")
    manifest_schema_version = payload.get("schema_version")
    if manifest_schema_version not in {
        COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        FORMAL_COMPLETED_ROLLOUTS_SCHEMA_VERSION,
        STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported completed-rollouts schema_version")
    exact_formal_replay = (
        manifest_schema_version == FORMAL_COMPLETED_ROLLOUTS_SCHEMA_VERSION
    )
    stock_unpaired_profile = (
        manifest_schema_version == STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION
    )
    realization_bank: dict[str, Any] | None = None
    realization_bank_path: Path | None = None
    if exact_formal_replay:
        if payload.get("formal_exact_episode_replay") is not True:
            raise ValueError("formal completed manifest does not prove exact episode replay")
        realization_identity = payload.get("formal_episode_realization_bank")
        if not isinstance(realization_identity, Mapping):
            raise ValueError("formal completed manifest lacks realization-bank identity")
        actual_realization = formal_stable_file_identity(
            str(realization_identity.get("path", ""))
        )
        if dict(realization_identity) != actual_realization:
            raise ValueError("formal completed realization-bank identity changed")
        realization_bank, realization_bank_path = validate_realization_bank(
            actual_realization["path"]
        )
        if (
            payload.get("formal_episode_realization_bank_id")
            != realization_bank["realization_bank_id"]
        ):
            raise ValueError("formal completed realization-bank id differs")
    stock_amendment: dict[str, Any] | None = None
    stock_amendment_path: Path | None = None
    if stock_unpaired_profile:
        from .release_stock_eval_protocol import validate_stock_eval_amendment

        if payload.get("episode_pairing") != "not_claimed":
            raise ValueError(
                "author-stock completed manifest must disclaim episode pairing"
            )
        amendment_identity = payload.get("stock_protocol_amendment")
        if not isinstance(amendment_identity, Mapping):
            raise ValueError(
                "author-stock completed manifest lacks amendment identity"
            )
        actual_amendment = formal_stable_file_identity(
            str(amendment_identity.get("path", ""))
        )
        if dict(amendment_identity) != actual_amendment:
            raise ValueError("author-stock amendment identity changed")
        stock_amendment, stock_amendment_path = validate_stock_eval_amendment(
            actual_amendment["path"]
        )
        if (
            payload.get("stock_protocol_amendment_id")
            != stock_amendment["amendment_id"]
            or payload.get("evaluation_profile") != stock_amendment["profile"]
        ):
            raise ValueError("author-stock amendment/profile id differs")
    checkpoint_contract = payload.get("checkpoint_contract")
    if not isinstance(checkpoint_contract, Mapping):
        raise ValueError("completed manifest lacks checkpoint_contract")
    control = _evaluation_record_control(str(checkpoint_contract.get("control", "")))
    raw_training_seed = checkpoint_contract.get("training_seed")
    if control == "c0_base":
        if raw_training_seed is not None:
            raise ValueError("fixed C0 manifest training_seed must be null")
        training_seed: int | None = None
    else:
        training_seed = _as_nonnegative_int(
            raw_training_seed, field="manifest Stage-2 training_seed"
        )
    fairness_identity = _fairness_identity_from_checkpoint_contract(
        checkpoint_contract,
        evaluation_control=control,
    )
    if payload.get("checkpoint_fairness_identity") != fairness_identity:
        raise ValueError(
            "completed manifest checkpoint_fairness_identity differs from checkpoint_contract"
        )
    if stock_unpaired_profile:
        assert stock_amendment is not None
        checkpoint_identity = checkpoint_contract.get("checkpoint_identity")
        if not isinstance(checkpoint_identity, Mapping):
            raise ValueError("author-stock checkpoint contract lacks identity")
        checkpoint_path = Path(str(checkpoint_identity.get("path", ""))).resolve()
        actual_checkpoint = formal_stable_file_identity(checkpoint_path)
        matching_rows = [
            row
            for row in stock_amendment["checkpoints"]
            if row["control"] == checkpoint_contract["control"]
            and row["training_seed"] == checkpoint_contract["training_seed"]
        ]
        if len(matching_rows) != 1:
            raise ValueError("author-stock amendment checkpoint row is missing")
        for field in ("path", "size_bytes", "sha256"):
            if matching_rows[0][field] != actual_checkpoint[field]:
                raise ValueError(
                    f"author-stock completed checkpoint {field} differs"
                )
    protocol_id = str(payload.get("rollout_protocol_id", "")).strip()
    seed_bank_id = str(payload.get("simulator_seed_bank_id", "")).strip()
    settings_sha = str(payload.get("rollout_settings_sha256", "")).strip()
    if (
        not protocol_id
        or not seed_bank_id
        or len(settings_sha) != 64
        or any(character not in "0123456789abcdef" for character in settings_sha)
    ):
        raise ValueError(
            "completed manifest lacks rollout protocol, seed-bank, or settings identity"
        )
    rollout_settings = payload.get("rollout_settings")
    if not isinstance(rollout_settings, Mapping):
        raise ValueError("completed manifest lacks rollout_settings")
    if _canonical_sha256(rollout_settings) != settings_sha:
        raise ValueError("completed manifest rollout_settings SHA-256 differs")
    if rollout_settings.get("rollout_protocol_id") != protocol_id:
        raise ValueError("rollout_settings rollout_protocol_id differs from manifest")
    if stock_unpaired_profile:
        assert stock_amendment is not None
        expected_stock_settings = {
            "evaluation_profile": stock_amendment["profile"],
            "stock_protocol_amendment_id": stock_amendment["amendment_id"],
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
        }
        for field, expected in expected_stock_settings.items():
            if rollout_settings.get(field) != expected:
                raise ValueError(
                    f"author-stock rollout_settings disclaimer differs: {field}"
                )
    episodes = _as_positive_int(
        payload.get("episodes_per_task"), field="manifest episodes_per_task"
    )
    simulator_seed = _as_nonnegative_int(
        payload.get("simulator_seed"), field="manifest simulator_seed"
    )
    if rollout_settings.get("episodes_per_task") != episodes:
        raise ValueError("rollout_settings episodes differ from manifest")
    seed_bank = payload.get("simulator_seed_bank")
    if not isinstance(seed_bank, Mapping):
        raise ValueError("completed manifest lacks simulator_seed_bank")
    try:
        validated_seed_bank = validate_seed_bank_descriptor(seed_bank)
    except ValueError as exc:
        raise ValueError(f"completed manifest seed bank is invalid: {exc}") from exc
    if stock_unpaired_profile:
        assert stock_amendment is not None
        amended_seed_bank = validate_seed_bank_descriptor(
            stock_amendment["runtime_seed_bank"],
            expected_purpose="final_test",
        )
        if validated_seed_bank != amended_seed_bank:
            raise ValueError(
                "author-stock completed runtime seed bank differs from amendment"
            )
    seed_bank_payload = _seed_bank_identity_payload(validated_seed_bank)
    expected_seed_bank_id = SEED_BANK_ID_PREFIX + _canonical_sha256(seed_bank_payload)
    if seed_bank_id != expected_seed_bank_id:
        raise ValueError("completed manifest simulator seed-bank identity differs")
    if seed_bank.get("simulator_seed_bank_id") != seed_bank_id:
        raise ValueError("seed-bank descriptor id differs from manifest")
    if seed_bank.get("simulator_seed") != simulator_seed:
        raise ValueError("seed-bank simulator seed differs from manifest")
    if seed_bank.get("episodes_per_cell") != episodes:
        raise ValueError("seed-bank episodes differ from manifest")
    purpose = str(payload.get("simulator_seed_bank_purpose", ""))
    if (
        not purpose
        or seed_bank.get("purpose") != purpose
        or checkpoint_contract.get("simulator_seed_bank_purpose") != purpose
    ):
        raise ValueError("completed manifest seed-bank purpose differs")
    if purpose != "final_test":
        raise ValueError("formal evaluation records require a final_test seed bank")
    if checkpoint_contract.get("formal_evaluation_eligible") is not True:
        raise ValueError("checkpoint contract is not eligible for formal evaluation records")
    lock_ancestry = validated_seed_bank.get("lock_ancestry")
    if not isinstance(lock_ancestry, Mapping):
        raise ValueError("final-test seed bank lacks lock ancestry")
    formal_lock = lock_ancestry.get("formal_protocol_lock_manifest")
    selection_lock = lock_ancestry.get("p_mode_selection_manifest")
    if (
        not isinstance(formal_lock, Mapping)
        or checkpoint_contract.get("formal_protocol_lock_manifest_sha256")
        != formal_lock.get("sha256")
    ):
        raise ValueError("completed manifest formal protocol lock differs")
    expected_selection_sha = fairness_identity.get(
        "p_mode_selection_manifest_sha256"
    )
    if control != "c0_base" and (
        not isinstance(selection_lock, Mapping)
        or expected_selection_sha != selection_lock.get("sha256")
    ):
        raise ValueError("completed manifest P-mode selection lock differs")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("completed manifest runs must be a non-empty list")
    if exact_formal_replay and len(runs) != 1:
        raise ValueError("formal exact-replay completed manifest must contain one cell")
    if stock_unpaired_profile and len(runs) != 1:
        raise ValueError("author-stock completed manifest must contain one cell")

    records: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise ValueError(f"completed manifest run {index} must be an object")
        task_config = str(run.get("task_config", ""))
        if task_config not in OFFICIAL_TASK_CONFIGS:
            raise ValueError(
                f"run {index} has unsupported task_config {task_config!r}"
            )
        domain = TASK_CONFIG_TO_DOMAIN[task_config]
        if run.get("domain") != domain or run.get("phase") != _phase_name(task_config):
            raise ValueError(f"run {index} task_config/domain/phase disagree")
        if run.get("episodes") != episodes:
            raise ValueError(f"run {index} episodes differ from manifest")
        if run.get("simulator_seed") != simulator_seed:
            raise ValueError(f"run {index} simulator seed differs from manifest")
        if run.get("rollout_protocol_id") != protocol_id:
            raise ValueError(f"run {index} rollout protocol differs from manifest")
        if run.get("simulator_seed_bank_id") != seed_bank_id:
            raise ValueError(f"run {index} seed-bank identity differs from manifest")
        if run.get("simulator_seed_bank_purpose") != purpose:
            raise ValueError(f"run {index} seed-bank purpose differs from manifest")
        if run.get("rollout_settings_sha256") != settings_sha:
            raise ValueError(f"run {index} rollout settings differ from manifest")
        if exact_formal_replay:
            assert realization_bank is not None
            assert realization_bank_path is not None
            if run.get("formal_exact_episode_replay") is not True:
                raise ValueError(f"run {index} does not prove formal exact replay")
            if run.get("formal_episode_realization_bank_id") != realization_bank[
                "realization_bank_id"
            ]:
                raise ValueError(f"run {index} realization-bank id differs")
            _, realization_cell, _ = select_realization_cell(
                realization_bank_path,
                task=str(run.get("task", "")),
                task_config=task_config,
            )
            if run.get("formal_episode_realization_cell_id") != realization_cell[
                "cell_id"
            ]:
                raise ValueError(f"run {index} realization-cell id differs")
            if run.get("no_seed_replacement") is not True:
                raise ValueError(f"run {index} does not prove no seed replacement")
            if run.get("ordered_seed_instruction_sha256") != realization_cell[
                "ordered_seed_instruction_sha256"
            ]:
                raise ValueError(f"run {index} ordered seed/instruction SHA differs")
            trace_identity = run.get("episode_trace")
            if not isinstance(trace_identity, Mapping):
                raise ValueError(f"run {index} lacks episode trace identity")
            actual_trace = formal_stable_file_identity(
                str(trace_identity.get("path", ""))
            )
            if dict(trace_identity) != actual_trace:
                raise ValueError(f"run {index} episode trace identity changed")
            trace, _ = validate_replay_trace(
                actual_trace["path"],
                realization_bank_path=realization_bank_path,
                task=str(run.get("task", "")),
                task_config=task_config,
            )
            if run.get("ordered_seed_instruction_success_sha256") != trace[
                "ordered_seed_instruction_success_sha256"
            ]:
                raise ValueError(f"run {index} ordered outcome SHA differs")
            expected_outcomes = [
                {
                    "episode_index": row["episode_index"],
                    "simulator_seed": row["simulator_seed"],
                    "instruction_sha256": row["instruction_sha256"],
                    "success": row["success"],
                }
                for row in trace["episodes"]
            ]
            if run.get("episode_outcomes") != expected_outcomes:
                raise ValueError(f"run {index} embedded episode outcomes differ")
        if stock_unpaired_profile:
            assert stock_amendment is not None
            if (
                run.get("evaluation_profile") != stock_amendment["profile"]
                or run.get("stock_protocol_amendment_id")
                != stock_amendment["amendment_id"]
            ):
                raise ValueError(f"run {index} author-stock profile differs")
            required_disclaimers = {
                "episode_pairing": "not_claimed",
                "shared_starting_seed_only": True,
                "per_checkpoint_expert_filtering": True,
                "accepted_episode_sequence_recorded": False,
            }
            for field, expected in required_disclaimers.items():
                if run.get(field) != expected:
                    raise ValueError(
                        f"run {index} author-stock disclaimer differs: {field}"
                    )
            if "episode_trace" in run or "episode_outcomes" in run:
                raise ValueError(
                    f"run {index} author-stock profile must not imply exact replay"
                )
        try:
            success_rate = float(run["success_rate"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"run {index} success_rate must be numeric") from exc
        if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
            raise ValueError(f"run {index} success_rate must be within [0, 1]")
        if exact_formal_replay and not math.isclose(
            success_rate,
            float(trace["success_rate"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"run {index} success rate differs from exact trace")
        task = str(run.get("task", "")).strip()
        if not task:
            raise ValueError(f"run {index} lacks task")
        records.append(
            {
                "control": control,
                "training_seed": training_seed,
                **fairness_identity,
                "lambda_contrastive": checkpoint_contract.get(
                    "lambda_contrastive"
                ),
                "paired_contrastive_gradient_enabled": (
                    None
                    if control == "c0_base"
                    else float(checkpoint_contract["lambda_contrastive"]) > 0.0
                ),
                "task": task,
                "domain": domain,
                "episodes": episodes,
                "success_rate": success_rate,
                "rollout_protocol_id": protocol_id,
                "simulator_seed_bank_id": seed_bank_id,
            }
        )

    embedded = payload.get("evaluation_records")
    if embedded != records:
        raise ValueError(
            "embedded evaluation_records differ from audited manifest fields"
        )
    return records


def aggregate_completed_rollout_manifests(
    manifest_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Convert one or more completed manifests to evaluation_protocol records."""

    if not manifest_paths:
        raise ValueError("at least one completed-rollouts manifest is required")
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    settings_hashes: set[str] = set()
    protocol_ids: set[str] = set()
    seed_bank_ids: set[str] = set()
    seed_bank_purposes: set[str] = set()
    manifest_schema_versions: set[int] = set()
    realization_bank_ids: set[str] = set()
    stock_amendment_ids: set[str] = set()
    stock_profiles: set[str] = set()
    realized_sequence_by_cell: dict[tuple[str, str], str] = {}
    cells: set[tuple[str, int | None, str, str]] = set()
    fairness_by_control_seed: dict[tuple[str, int | None], tuple[Any, ...]] = {}
    for raw_path in manifest_paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            raw_bytes = path.read_bytes()
            payload = json.loads(raw_bytes)
        except Exception as exc:
            raise ValueError(f"cannot read completed manifest {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"completed manifest root must be an object: {path}")
        current = _records_from_completed_manifest(payload)
        manifest_schema_versions.add(int(payload["schema_version"]))
        if payload["schema_version"] == FORMAL_COMPLETED_ROLLOUTS_SCHEMA_VERSION:
            realization_bank_ids.add(
                str(payload.get("formal_episode_realization_bank_id", ""))
            )
            for run in payload["runs"]:
                cell_key = (str(run["task"]), str(run["domain"]))
                sequence_sha = str(run["ordered_seed_instruction_sha256"])
                previous_sequence = realized_sequence_by_cell.setdefault(
                    cell_key, sequence_sha
                )
                if previous_sequence != sequence_sha:
                    raise ValueError(
                        "formal candidates used different ordered seed/instruction "
                        f"sequences for {cell_key}"
                    )
        elif payload["schema_version"] == STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION:
            stock_amendment_ids.add(
                str(payload.get("stock_protocol_amendment_id", ""))
            )
            stock_profiles.add(str(payload.get("evaluation_profile", "")))
        settings_hashes.add(str(payload["rollout_settings_sha256"]))
        protocol_ids.add(str(payload["rollout_protocol_id"]))
        seed_bank_ids.add(str(payload["simulator_seed_bank_id"]))
        seed_bank_purposes.add(str(payload["simulator_seed_bank_purpose"]))
        for record in current:
            control_seed = (
                str(record["control"]),
                (
                    None
                    if record["training_seed"] is None
                    else int(record["training_seed"])
                ),
            )
            fairness = tuple(record[name] for name in FAIRNESS_RECORD_FIELDS)
            previous_fairness = fairness_by_control_seed.setdefault(control_seed, fairness)
            if previous_fairness != fairness:
                raise ValueError(
                    f"checkpoint fairness identity mismatch for {control_seed}"
                )
            key = (
                *control_seed,
                str(record["task"]),
                str(record["domain"]),
            )
            if key in cells:
                raise ValueError(f"duplicate evaluation record cell: {key}")
            cells.add(key)
            records.append(record)
        sources.append(
            {
                "path": str(path),
                "size_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        )
    if len(settings_hashes) != 1:
        raise ValueError(f"rollout settings mismatch: {sorted(settings_hashes)}")
    if len(protocol_ids) != 1:
        raise ValueError(f"rollout protocol mismatch: {sorted(protocol_ids)}")
    if len(seed_bank_ids) != 1:
        raise ValueError(f"simulator seed-bank mismatch: {sorted(seed_bank_ids)}")
    if len(seed_bank_purposes) != 1:
        raise ValueError(f"simulator seed-bank purpose mismatch: {sorted(seed_bank_purposes)}")
    if len(manifest_schema_versions) != 1:
        raise ValueError(
            "cannot mix legacy candidate-filtered and formal exact-replay manifests"
        )
    exact_formal_replay = manifest_schema_versions == {
        FORMAL_COMPLETED_ROLLOUTS_SCHEMA_VERSION
    }
    stock_unpaired_profile = manifest_schema_versions == {
        STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION
    }
    if exact_formal_replay:
        if len(realization_bank_ids) != 1 or "" in realization_bank_ids:
            raise ValueError("formal realization-bank identity mismatch")
    if stock_unpaired_profile:
        if (
            len(stock_amendment_ids) != 1
            or "" in stock_amendment_ids
            or len(stock_profiles) != 1
            or "" in stock_profiles
        ):
            raise ValueError("author-stock amendment/profile identity mismatch")
    return {
        "schema_version": EVALUATION_PROTOCOL_SCHEMA_VERSION,
        "profile": EVALUATION_PROTOCOL_PROFILE,
        "records": records,
        "source_manifests": sources,
        "rollout_settings_sha256": next(iter(settings_hashes)),
        "rollout_protocol_id": next(iter(protocol_ids)),
        "simulator_seed_bank_id": next(iter(seed_bank_ids)),
        "simulator_seed_bank_purpose": next(iter(seed_bank_purposes)),
        "formal_exact_episode_replay": exact_formal_replay,
        "formal_episode_realization_bank_id": (
            next(iter(realization_bank_ids)) if exact_formal_replay else None
        ),
        "evaluation_profile": (
            next(iter(stock_profiles)) if stock_unpaired_profile else None
        ),
        "stock_protocol_amendment_id": (
            next(iter(stock_amendment_ids))
            if stock_unpaired_profile
            else None
        ),
        "episode_pairing": (
            "not_claimed" if stock_unpaired_profile else None
        ),
        "shared_starting_seed_only": stock_unpaired_profile,
        "per_checkpoint_expert_filtering": stock_unpaired_profile,
        "ordered_seed_instruction_sha256_by_cell": (
            {
                f"{task}/{domain}": digest
                for (task, domain), digest in sorted(realized_sequence_by_cell.items())
            }
            if exact_formal_replay
            else None
        ),
    }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="sim_robotwin.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Compact policy checkpoint not found: {ckpt_path}")
    tasks = _resolve_tasks(cfg.EVALUATION.task_name)
    task_configs = _resolve_task_configs(cfg.EVALUATION.task_config)
    requested_domains = [TASK_CONFIG_TO_DOMAIN[value] for value in task_configs]
    formal_realization_bank_path = _resolve_optional_path(
        cfg.EVALUATION.get("formal_episode_realization_bank"),
        base=PROJECT_ROOT,
    )
    stock_protocol_amendment_path = _resolve_optional_path(
        cfg.EVALUATION.get("stock_protocol_amendment"),
        base=PROJECT_ROOT,
    )
    pv2_followup_eval_amendment_path = _resolve_optional_path(
        cfg.EVALUATION.get("pv2_followup_eval_amendment"),
        base=PROJECT_ROOT,
    )
    selected_profiles = sum(
        value is not None
        for value in (
            formal_realization_bank_path,
            stock_protocol_amendment_path,
            pv2_followup_eval_amendment_path,
        )
    )
    if selected_profiles > 1:
        raise ValueError(
            "exact replay, author-stock formal, and P-v2 eval100 profiles are mutually exclusive"
        )
    if formal_realization_bank_path is not None and (
        len(tasks) != 1 or len(task_configs) != 1
    ):
        raise ValueError(
            "exact formal replay is cell-scoped: pass exactly one task and one "
            "of demo_clean/demo_randomized"
        )
    if stock_protocol_amendment_path is not None and (
        len(tasks) != 1 or len(task_configs) != 1
    ):
        raise ValueError(
            "author-stock formal evaluation is cell-scoped: pass exactly one "
            "task and one of demo_clean/demo_randomized"
        )
    simulator_seed = _as_nonnegative_int(cfg.seed, field="simulator seed")
    episodes_per_task = _as_positive_int(
        cfg.EVALUATION.eval_num_episodes,
        field="EVALUATION.eval_num_episodes",
    )

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    policy_source = Path(__file__).resolve().parent

    requested_output = _resolve_output_dir(cfg.EVALUATION.output_dir)
    # EVALUATION.output_dir is an explicit artifact contract. Preserve its
    # complete parent path rather than retaining only the final path segment.
    run_output_dir = requested_output

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)
    for path in (PROJECT_ROOT, SRC_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiments.robotwin.policy_content_adapter.rollout_policy import (
        _resolve_model_base_path,
    )

    explicit_model_base = cfg.EVALUATION.get("model_base_path")
    model_base_path, model_base_audit = _resolve_model_base_path(
        ckpt_path,
        explicit_model_base,
    )
    # The compact checkpoint retains the original, immutable 20-episode pilot
    # contract.  A narrow external amendment may replace only the online
    # episode count/derived seed-bank identity; all training provenance is
    # still validated against the checkpoint's original declaration.
    checkpoint_declared_episodes = (
        20 if pv2_followup_eval_amendment_path is not None else episodes_per_task
    )
    checkpoint_contract = _checkpoint_evaluation_contract(
        model_base_audit,
        requested_tasks=tasks,
        requested_domains=requested_domains,
        episodes_per_task=checkpoint_declared_episodes,
    )
    formal_realization_bank: dict[str, Any] | None = None
    formal_realization_cell: dict[str, Any] | None = None
    formal_realization_bank_identity: dict[str, Any] | None = None
    stock_protocol_amendment: dict[str, Any] | None = None
    stock_protocol_amendment_identity: dict[str, Any] | None = None
    pv2_followup_eval_amendment: dict[str, Any] | None = None
    pv2_followup_eval_amendment_identity: dict[str, Any] | None = None
    if formal_realization_bank_path is not None:
        if episodes_per_task != FORMAL_EPISODES_PER_CELL:
            raise ValueError(
                f"exact formal replay requires {FORMAL_EPISODES_PER_CELL} episodes"
            )
        if checkpoint_contract["formal_evaluation_eligible"] is not True:
            raise ValueError(
                "exact formal episode replay requires a formal final-test checkpoint"
            )
        formal_realization_bank, formal_realization_cell, resolved_bank = (
            select_realization_cell(
                formal_realization_bank_path,
                task=tasks[0],
                task_config=task_configs[0],
            )
        )
        formal_realization_bank_identity = formal_stable_file_identity(resolved_bank)
        if (
            formal_realization_bank["candidate_seed_bank_id"]
            != checkpoint_contract["simulator_seed_bank_id"]
        ):
            raise ValueError(
                "formal realization bank derives from a different candidate seed bank"
            )
        if (
            formal_realization_bank["candidate_seed_bank"]["sha256"]
            != checkpoint_contract["simulator_seed_bank_manifest_sha256"]
        ):
            raise ValueError("formal realization candidate-bank SHA differs")
        if (
            formal_realization_bank["formal_protocol_lock"]["sha256"]
            != checkpoint_contract["formal_protocol_lock_manifest_sha256"]
        ):
            raise ValueError("formal realization protocol-lock SHA differs")
    if stock_protocol_amendment_path is not None:
        from .release_stock_eval_protocol import (
            EPISODES_PER_CELL as STOCK_EPISODES_PER_CELL,
            PROFILE as STOCK_EVAL_PROFILE,
            SIMULATOR_SEED as STOCK_SIMULATOR_SEED,
            validate_stock_eval_amendment,
        )

        stock_protocol_amendment, resolved_amendment = (
            validate_stock_eval_amendment(stock_protocol_amendment_path)
        )
        stock_protocol_amendment_identity = formal_stable_file_identity(
            resolved_amendment
        )
        if stock_protocol_amendment.get("profile") != STOCK_EVAL_PROFILE:
            raise ValueError("author-stock evaluation profile differs")
        if episodes_per_task != STOCK_EPISODES_PER_CELL:
            raise ValueError(
                f"author-stock evaluation requires {STOCK_EPISODES_PER_CELL} episodes"
            )
        if simulator_seed != STOCK_SIMULATOR_SEED:
            raise ValueError(
                f"author-stock evaluation requires simulator seed {STOCK_SIMULATOR_SEED}"
            )
        if checkpoint_contract["formal_evaluation_eligible"] is not True:
            raise ValueError(
                "author-stock formal profile requires a completed formal checkpoint"
            )
        if (
            stock_protocol_amendment["original_checkpoint_seed_bank_id"]
            != checkpoint_contract["simulator_seed_bank_id"]
            or stock_protocol_amendment[
                "original_checkpoint_seed_bank_sha256"
            ]
            != checkpoint_contract["simulator_seed_bank_manifest_sha256"]
        ):
            raise ValueError(
                "author-stock amendment does not bind the checkpoint's original seed bank"
            )
        if (
            stock_protocol_amendment["formal_protocol_lock_sha256"]
            != checkpoint_contract["formal_protocol_lock_manifest_sha256"]
        ):
            raise ValueError(
                "author-stock amendment formal protocol lock differs"
            )
        checkpoint_identity = formal_stable_file_identity(ckpt_path)
        matching_rows = [
            row
            for row in stock_protocol_amendment["checkpoints"]
            if row["control"] == checkpoint_contract["control"]
            and row["training_seed"] == checkpoint_contract["training_seed"]
        ]
        if len(matching_rows) != 1:
            raise ValueError(
                "author-stock amendment lacks one exact row for this checkpoint"
            )
        for field in ("path", "size_bytes", "sha256"):
            if matching_rows[0][field] != checkpoint_identity[field]:
                raise ValueError(
                    f"author-stock amendment checkpoint {field} differs"
                )
    if pv2_followup_eval_amendment_path is not None:
        from .pv2_followup_eval100_amendment import (
            validate_eval100_amendment,
        )

        pv2_followup_eval_amendment, resolved_amendment = (
            validate_eval100_amendment(pv2_followup_eval_amendment_path)
        )
        pv2_followup_eval_amendment_identity = formal_stable_file_identity(
            resolved_amendment
        )
        _validate_pv2_eval100_checkpoint_contract(
            pv2_followup_eval_amendment,
            checkpoint_path=ckpt_path,
            checkpoint_contract=checkpoint_contract,
            requested_tasks=tasks,
            requested_domains=requested_domains,
            simulator_seed=simulator_seed,
            episodes_per_task=episodes_per_task,
        )
    rollout_settings = _build_rollout_settings(
        cfg,
        sim_cfg_path=sim_cfg_path,
        sim_task=sim_task,
    )
    rollout_settings["rollout_protocol_id"] = checkpoint_contract[
        "rollout_protocol_id"
    ]
    if formal_realization_bank is not None:
        assert formal_realization_bank_identity is not None
        rollout_settings.update(
            {
                "formal_exact_episode_replay": True,
                "formal_episode_realization_bank_id": formal_realization_bank[
                    "realization_bank_id"
                ],
                "formal_episode_realization_bank_sha256": (
                    formal_realization_bank_identity["sha256"]
                ),
                "formal_no_seed_replacement": True,
            }
        )
    if stock_protocol_amendment is not None:
        assert stock_protocol_amendment_identity is not None
        rollout_settings.update(
            {
                "evaluation_profile": stock_protocol_amendment["profile"],
                "stock_protocol_amendment_id": stock_protocol_amendment[
                    "amendment_id"
                ],
                "stock_protocol_amendment_sha256": (
                    stock_protocol_amendment_identity["sha256"]
                ),
                "episode_pairing": "not_claimed",
                "shared_starting_seed_only": True,
                "per_checkpoint_expert_filtering": True,
            }
        )
    if pv2_followup_eval_amendment is not None:
        assert pv2_followup_eval_amendment_identity is not None
        rollout_settings.update(
            {
                "evaluation_profile": pv2_followup_eval_amendment["profile"],
                "pv2_followup_eval_amendment_id": (
                    pv2_followup_eval_amendment["amendment_id"]
                ),
                "pv2_followup_eval_amendment_sha256": (
                    pv2_followup_eval_amendment_identity["sha256"]
                ),
                "episode_pairing": "not_claimed",
                "shared_starting_seed_only": True,
                "per_checkpoint_expert_filtering": True,
                "fastwam_aligned_episodes_per_task_domain": 100,
                "superseded_partial_20_episode_results_used": False,
            }
        )
    rollout_settings_sha256 = _canonical_sha256(rollout_settings)
    computed_simulator_seed_bank = _build_simulator_seed_bank(
        simulator_seed=simulator_seed,
        episodes_per_task=episodes_per_task,
        evaluator_source=robotwin_root / "script" / "eval_policy.py",
        purpose=str(checkpoint_contract["simulator_seed_bank_purpose"]),
        disjoint_from=checkpoint_contract["simulator_seed_bank_descriptor"][
            "disjoint_from"
        ],
        lock_ancestry=checkpoint_contract["simulator_seed_bank_descriptor"][
            "lock_ancestry"
        ],
    )
    if stock_protocol_amendment is None and pv2_followup_eval_amendment is None:
        simulator_seed_bank = computed_simulator_seed_bank
        simulator_seed_bank_id = _verify_checkpoint_seed_bank(
            checkpoint_contract,
            simulator_seed_bank,
        )
    elif stock_protocol_amendment is not None:
        simulator_seed_bank = stock_protocol_amendment["runtime_seed_bank"]
        if simulator_seed_bank != computed_simulator_seed_bank:
            raise ValueError(
                "runtime author-stock seed bank differs from the audited amendment"
            )
        simulator_seed_bank_id = str(
            simulator_seed_bank["simulator_seed_bank_id"]
        )
    else:
        assert pv2_followup_eval_amendment is not None
        simulator_seed_bank = pv2_followup_eval_amendment["runtime_evaluation"][
            "seed_bank"
        ]
        simulator_seed_bank_path = simulator_seed_bank["path"]
        raw_runtime_bank = json.loads(
            Path(simulator_seed_bank_path).read_text(encoding="utf-8")
        )
        if raw_runtime_bank != computed_simulator_seed_bank:
            raise ValueError(
                "runtime P-v2 eval100 seed bank differs from the audited amendment"
            )
        simulator_seed_bank = raw_runtime_bank
        simulator_seed_bank_id = str(simulator_seed_bank["simulator_seed_bank_id"])

    # Bind CUDA inference and SAPIEN/Vulkan rendering to the same physical PCI
    # device before mutating RoboTwin or creating any rollout artifacts.  The
    # preflight imports SAPIEN only in a child process and never starts a task.
    gpu_runtime_binding = preflight_gpu_runtime(
        cfg.gpu_id,
        python_executable=sys.executable,
    )

    # Do not mutate RoboTwin or create output artifacts until every checkpoint
    # and rollout/runtime identity above has passed its fail-closed audit.
    _ensure_policy_symlink(robotwin_root, policy_source)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    env = _build_subprocess_environment(
        gpu_id=cfg.gpu_id,
        model_base_path=model_base_path,
        gpu_runtime_binding=gpu_runtime_binding,
    )

    completed: list[dict[str, Any]] = []
    for task_name in tasks:
        for task_config in task_configs:
            phase = _phase_name(task_config)
            robotwin_eval_base = run_output_dir / task_name
            formal_trace_path = (
                run_output_dir / f"formal_episode_trace_{task_name}_{phase}.json"
                if formal_realization_bank_path is not None
                else None
            )
            if formal_trace_path is not None and formal_trace_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite formal episode trace: {formal_trace_path}"
                )
            log_path = run_output_dir / (
                f"eval_{task_name}_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )

            overrides: list[str] = []
            _append_override(overrides, "task_name", task_name)
            _append_override(overrides, "task_config", task_config)
            _append_override(overrides, "ckpt_setting", str(ckpt_path))
            _append_override(overrides, "seed", cfg.seed)
            _append_override(overrides, "policy_name", POLICY_MODULE)
            _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
            _append_override(
                overrides,
                "eval_num_episodes",
                cfg.EVALUATION.eval_num_episodes,
            )
            _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
            _append_override(overrides, "sim_task", sim_task)
            _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
            _append_override(overrides, "mixed_precision", cfg.mixed_precision)
            _append_override(overrides, "device", cfg.EVALUATION.device)
            _append_override(overrides, "model_base_path", str(model_base_path))
            _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
            _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
            _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
            _append_override(
                overrides,
                "num_inference_steps",
                cfg.EVALUATION.num_inference_steps,
            )
            _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
            _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
            _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
            _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
            _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
            _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
            _append_override(
                overrides,
                "skip_get_obs_within_replan",
                cfg.EVALUATION.skip_get_obs_within_replan,
            )
            if formal_realization_bank_path is not None:
                assert formal_trace_path is not None
                _append_override(overrides, "formal_episode_mode", "replay")
                _append_override(
                    overrides,
                    "formal_episode_realization_bank",
                    str(formal_realization_bank_path),
                )
                _append_override(
                    overrides,
                    "formal_episode_trace_output",
                    str(formal_trace_path),
                )

            cmd = [
                sys.executable,
                "-u",
                "-m",
                PINNED_EVALUATOR_MODULE,
                "--config",
                DEPLOY_CONFIG,
                "--overrides",
                *overrides,
            ]
            print(
                f"START task={task_name} phase={phase} gpu={cfg.gpu_id} "
                f"pci={gpu_runtime_binding['pci_bus_id']} "
                f"log={log_path}",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(robotwin_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
                return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"RoboTwin evaluation failed for {task_name}/{phase} "
                    f"with return code {return_code}. Log: {log_path}"
                )

            phase_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            phase_cfg.EVALUATION.task_name = task_name
            phase_cfg.EVALUATION.task_config = task_config
            phase_cfg.EVALUATION.output_dir = str(run_output_dir)
            OmegaConf.save(
                config=phase_cfg,
                f=str(run_output_dir / f"eval_config_{task_name}_{phase}.yaml"),
            )
            result_path = _result_path(robotwin_eval_base, phase)
            success_rate = _parse_success_rate(result_path)
            run_record: dict[str, Any] = {
                "task": task_name,
                "phase": phase,
                "task_config": task_config,
                "domain": TASK_CONFIG_TO_DOMAIN[task_config],
                "episodes": episodes_per_task,
                "simulator_seed": simulator_seed,
                "rollout_protocol_id": checkpoint_contract[
                    "rollout_protocol_id"
                ],
                "rollout_settings_sha256": rollout_settings_sha256,
                "simulator_seed_bank_id": simulator_seed_bank_id,
                "simulator_seed_bank_purpose": simulator_seed_bank["purpose"],
                "physical_gpu_index": gpu_runtime_binding[
                    "physical_gpu_index"
                ],
                "render_device_alias": gpu_runtime_binding[
                    "render_device_alias"
                ],
                "log": str(log_path),
                "result": str(result_path),
                "success_rate": success_rate,
            }
            if formal_realization_bank_path is not None:
                assert formal_trace_path is not None
                assert formal_realization_bank is not None
                assert formal_realization_cell is not None
                trace, resolved_trace = validate_replay_trace(
                    formal_trace_path,
                    realization_bank_path=formal_realization_bank_path,
                    task=task_name,
                    task_config=task_config,
                )
                if (
                    formal_stable_file_identity(formal_realization_bank_path)
                    != formal_realization_bank_identity
                ):
                    raise RuntimeError(
                        "formal realization bank changed during online rollout"
                    )
                if not math.isclose(
                    float(trace["success_rate"]),
                    success_rate,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "RoboTwin result success rate differs from exact episode trace"
                    )
                run_record.update(
                    {
                        "formal_exact_episode_replay": True,
                        "formal_episode_realization_bank_id": (
                            formal_realization_bank["realization_bank_id"]
                        ),
                        "formal_episode_realization_cell_id": (
                            formal_realization_cell["cell_id"]
                        ),
                        "ordered_seed_instruction_sha256": trace[
                            "ordered_seed_instruction_sha256"
                        ],
                        "ordered_seed_instruction_success_sha256": trace[
                            "ordered_seed_instruction_success_sha256"
                        ],
                        "no_seed_replacement": True,
                        "episode_trace": formal_stable_file_identity(resolved_trace),
                        "episode_outcomes": [
                            {
                                "episode_index": row["episode_index"],
                                "simulator_seed": row["simulator_seed"],
                                "instruction_sha256": row["instruction_sha256"],
                                "success": row["success"],
                            }
                            for row in trace["episodes"]
                        ],
                    }
                )
            if stock_protocol_amendment is not None:
                assert stock_protocol_amendment_identity is not None
                if (
                    formal_stable_file_identity(stock_protocol_amendment_path)
                    != stock_protocol_amendment_identity
                ):
                    raise RuntimeError(
                        "author-stock evaluation amendment changed during rollout"
                    )
                run_record.update(
                    {
                        "evaluation_profile": stock_protocol_amendment[
                            "profile"
                        ],
                        "stock_protocol_amendment_id": (
                            stock_protocol_amendment["amendment_id"]
                        ),
                        "episode_pairing": "not_claimed",
                        "shared_starting_seed_only": True,
                        "per_checkpoint_expert_filtering": True,
                        "accepted_episode_sequence_recorded": False,
                    }
                )
            if pv2_followup_eval_amendment is not None:
                assert pv2_followup_eval_amendment_identity is not None
                if (
                    formal_stable_file_identity(pv2_followup_eval_amendment_path)
                    != pv2_followup_eval_amendment_identity
                ):
                    raise RuntimeError(
                        "P-v2 eval100 amendment changed during rollout"
                    )
                run_record.update(
                    {
                        "evaluation_profile": pv2_followup_eval_amendment["profile"],
                        "pv2_followup_eval_amendment_id": (
                            pv2_followup_eval_amendment["amendment_id"]
                        ),
                        "episode_pairing": "not_claimed",
                        "shared_starting_seed_only": True,
                        "per_checkpoint_expert_filtering": True,
                        "accepted_episode_sequence_recorded": False,
                        "superseded_partial_20_episode_results_used": False,
                    }
                )
            completed.append(run_record)
            print(
                f"DONE task={task_name} phase={phase} "
                f"success_rate={success_rate:.6f}",
                flush=True,
            )

    evaluation_control = (
        FORMAL_RECORD_CONTROL_ALIASES.get(str(checkpoint_contract["control"]))
        if checkpoint_contract["formal_evaluation_eligible"]
        else None
    )
    fairness_identity = (
        _fairness_identity_from_checkpoint_contract(
            checkpoint_contract,
            evaluation_control=evaluation_control,
        )
        if evaluation_control is not None
        else None
    )
    evaluation_records = (
        [
            {
                "control": evaluation_control,
                "training_seed": checkpoint_contract["training_seed"],
                **fairness_identity,
                "lambda_contrastive": checkpoint_contract[
                    "lambda_contrastive"
                ],
                "paired_contrastive_gradient_enabled": (
                    None
                    if evaluation_control == "c0_base"
                    else float(checkpoint_contract["lambda_contrastive"]) > 0.0
                ),
                "task": run["task"],
                "domain": run["domain"],
                "episodes": run["episodes"],
                "success_rate": run["success_rate"],
                "rollout_protocol_id": run["rollout_protocol_id"],
                "simulator_seed_bank_id": run["simulator_seed_bank_id"],
            }
            for run in completed
        ]
        if evaluation_control is not None
        else []
    )
    manifest = run_output_dir / "completed_rollouts.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": COMPLETED_ROLLOUTS_SCHEMA,
                "schema_version": (
                    PV2_FOLLOWUP_EVAL100_COMPLETED_ROLLOUTS_SCHEMA_VERSION
                    if pv2_followup_eval_amendment is not None
                    else (
                        STOCK_COMPLETED_ROLLOUTS_SCHEMA_VERSION
                        if stock_protocol_amendment is not None
                        else (
                            FORMAL_COMPLETED_ROLLOUTS_SCHEMA_VERSION
                            if formal_realization_bank is not None
                            else COMPLETED_ROLLOUTS_SCHEMA_VERSION
                        )
                    )
                ),
                "checkpoint": str(ckpt_path),
                "checkpoint_contract": checkpoint_contract,
                "checkpoint_fairness_identity": fairness_identity,
                "dataset_stats": str(dataset_stats_path),
                "model_base": model_base_audit,
                "gpu_runtime_binding": gpu_runtime_binding,
                "output_dir": str(run_output_dir),
                "simulator_seed": simulator_seed,
                "episodes_per_task": episodes_per_task,
                "rollout_protocol_id": checkpoint_contract[
                    "rollout_protocol_id"
                ],
                "rollout_settings": rollout_settings,
                "rollout_settings_sha256": rollout_settings_sha256,
                "simulator_seed_bank": simulator_seed_bank,
                "simulator_seed_bank_id": simulator_seed_bank_id,
                "simulator_seed_bank_purpose": simulator_seed_bank["purpose"],
                "formal_exact_episode_replay": (
                    formal_realization_bank is not None
                ),
                "formal_episode_realization_bank": (
                    formal_realization_bank_identity
                ),
                "formal_episode_realization_bank_id": (
                    None
                    if formal_realization_bank is None
                    else formal_realization_bank["realization_bank_id"]
                ),
                "evaluation_profile": (
                    pv2_followup_eval_amendment["profile"]
                    if pv2_followup_eval_amendment is not None
                    else (
                        None
                        if stock_protocol_amendment is None
                        else stock_protocol_amendment["profile"]
                    )
                ),
                "stock_protocol_amendment": (
                    stock_protocol_amendment_identity
                ),
                "stock_protocol_amendment_id": (
                    None
                    if stock_protocol_amendment is None
                    else stock_protocol_amendment["amendment_id"]
                ),
                "pv2_followup_eval_amendment": (
                    pv2_followup_eval_amendment_identity
                ),
                "pv2_followup_eval_amendment_id": (
                    None
                    if pv2_followup_eval_amendment is None
                    else pv2_followup_eval_amendment["amendment_id"]
                ),
                "episode_pairing": (
                    "not_claimed"
                    if (
                        stock_protocol_amendment is not None
                        or pv2_followup_eval_amendment is not None
                    )
                    else None
                ),
                "evaluation_protocol": {
                    "schema_version": EVALUATION_PROTOCOL_SCHEMA_VERSION,
                    "profile": EVALUATION_PROTOCOL_PROFILE,
                    "eligible": evaluation_control is not None,
                    "control": evaluation_control,
                    "reason": (
                        None
                        if evaluation_control is not None
                        else "checkpoint/purpose is not eligible for final C0/C1/C3 records"
                    ),
                },
                "evaluation_records": evaluation_records,
                "runs": completed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation finished successfully: {manifest}", flush=True)


if __name__ == "__main__":
    main()
