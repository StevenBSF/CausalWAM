"""Five-epoch, eight-GPU P-v2 C1/C3 experiment derived from author settings.

This is a post-hoc successor to the completed 1800-step mechanism pilot.  It
matches the author's RoboTwin epoch count (5) on the full three-task official
stream while retaining the method-specific freeze/LR/loss contract.  The
preferred execution is 8 GPUs x 16 official samples/GPU (global batch 128),
which yields exactly 18,215 optimizer steps for 466,240 samples x 5 epochs.
An audited OOM may authorize a separate global-batch-64 amendment; it is not
silently enabled by this module.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import train as train_module
from .config_audit import load_config, validate_execution_ready
from .materialize_release_engineering_smoke import _write_new_yaml
from .p_mode_selection import canonical_sha256
from .pv2_actiondit_followup_audit import _audit_action_gate, _audit_training_run
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pv2_actiondit_full5ep_protocol"
SCHEMA_VERSION = 1
MATERIALIZATION_KIND = "policy_pv2_actiondit_full5ep_materialization"
POSTTRAIN_KIND = "policy_pv2_actiondit_full5ep_posttrain_audit"
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
CONTROLS = {"c1": "c1_architecture_only", "c3": "c3_ours"}
TRAINING_SEEDS = (1, 2, 3)
OFFICIAL_SAMPLE_COUNT = 466_240
NUM_EPOCHS = 5
WORLD_SIZE = 8
PREFERRED_LOCAL_BATCH = 16
PREFERRED_GLOBAL_BATCH = WORLD_SIZE * PREFERRED_LOCAL_BATCH
PREFERRED_STEPS_PER_EPOCH = math.ceil(
    OFFICIAL_SAMPLE_COUNT / PREFERRED_GLOBAL_BATCH
)
PREFERRED_MAX_STEPS = PREFERRED_STEPS_PER_EPOCH * NUM_EPOCHS
FALLBACK_LOCAL_BATCH = 8
FALLBACK_GLOBAL_BATCH = WORLD_SIZE * FALLBACK_LOCAL_BATCH
FALLBACK_STEPS_PER_EPOCH = math.ceil(
    OFFICIAL_SAMPLE_COUNT / FALLBACK_GLOBAL_BATCH
)
FALLBACK_MAX_STEPS = FALLBACK_STEPS_PER_EPOCH * NUM_EPOCHS
PAIRED_GROUPS_PER_GPU = 2
EFFECTIVE_PAIRED_GROUPS_PER_STEP = WORLD_SIZE * PAIRED_GROUPS_PER_GPU
SMOKE_STEPS = 3
FORMAL_SAVE_EVERY = 2_000
SMOKE_SAVE_EVERY = SMOKE_STEPS
DEFAULT_SHORT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1"
).resolve()
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/pv2_actiondit_full5ep_v1"
).resolve()
AUTHOR_ROOT = Path("/mnt/cpfs-E/baoshifeng/FastWAM").resolve()
AUTHOR_TASK_CONFIG = AUTHOR_ROOT / "configs/task/robotwin_uncond_3cam_384_1e-4.yaml"
AUTHOR_TRAINER_SOURCE = AUTHOR_ROOT / "src/fastwam/trainer.py"
AUTHOR_README = AUTHOR_ROOT / "README.md"
AUTHOR_DATASET_INFO = AUTHOR_ROOT / "data/robotwin2.0/robotwin2.0/meta/info.json"


class Pv2Full5EpochError(ValueError):
    """The five-epoch execution contract is not proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2Full5EpochError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required artifact missing: {resolved}")
    before = resolved.stat()
    digest = _sha256(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"artifact changed while hashing: {resolved}",
    )
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _verify_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} identity must be an object")
    actual = _identity(str(value.get("path", "")))
    for field in ("path", "size_bytes", "sha256"):
        expected = (
            str(Path(str(value.get(field, ""))).expanduser().resolve())
            if field == "path"
            else value.get(field)
        )
        _require(actual[field] == expected, f"{label} {field} changed")
    return actual


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pv2Full5EpochError(f"cannot parse {label}: {resolved}") from exc
    _require(isinstance(payload, dict), f"{label} root must be an object")
    return payload, resolved


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise Pv2Full5EpochError(
                f"refusing to overwrite immutable artifact: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def _author_setting_audit() -> dict[str, Any]:
    task_text = AUTHOR_TASK_CONFIG.read_text(encoding="utf-8")
    readme_text = AUTHOR_README.read_text(encoding="utf-8")
    info, _ = _load_json(AUTHOR_DATASET_INFO, "author RoboTwin dataset info")
    _require("batch_size: 16" in task_text, "author RoboTwin batch_size is not 16")
    _require("num_epochs: 5" in task_text, "author RoboTwin num_epochs is not 5")
    _require(
        'lr_scheduler_type: "cosine"' in task_text
        and "learning_rate: 1e-4" in task_text
        and "gradient_accumulation_steps: 1" in task_text,
        "author scheduler/LR/accumulation setting differs",
    )
    _require(
        "For RoboTwin, we use 64 GPUs" in readme_text,
        "author README no longer declares 64-GPU RoboTwin training",
    )
    _require(
        info.get("total_episodes") == 27_500
        and info.get("total_frames") == 6_075_103
        and info.get("fps") == 50,
        "author RoboTwin dataset facts differ",
    )
    short_action_init, _ = _load_json(
        DEFAULT_SHORT_ROOT / "manifests/action_dit_initialization_audit.json",
        "release ActionDiT initialization audit",
    )
    _require(
        short_action_init.get("checkpoint_step") == 29_355
        and short_action_init.get("checkpoint_sha256")
        == "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63",
        "release checkpoint step/SHA differs",
    )
    return {
        "status": "PASS",
        "task_config": _identity(AUTHOR_TASK_CONFIG),
        "trainer_source": _identity(AUTHOR_TRAINER_SOURCE),
        "readme": _identity(AUTHOR_README),
        "dataset_info": _identity(AUTHOR_DATASET_INFO),
        "declared_num_epochs": 5,
        "declared_local_batch_size": 16,
        "declared_gpu_count": 64,
        "declared_effective_global_batch": 1024,
        "declared_gradient_accumulation_steps": 1,
        "declared_lr_scheduler": "cosine",
        "declared_learning_rate": 1.0e-4,
        "release_checkpoint_step": 29_355,
        "release_dataset_frames": 6_075_103,
    }


def _superseded_short_audit() -> dict[str, Any]:
    decision, decision_path = _load_json(
        DEFAULT_SHORT_ROOT / "pilot_decision.json", "short pilot decision"
    )
    _require(
        decision.get("pilot_gate_passed") is True
        and decision.get("episodes_per_task_domain") == 100,
        "short pilot decision differs",
    )
    live_processes = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="ignore"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(
            token in cmdline
            for token in (
                "pv2_actiondit_followup_expansion",
                "eval_robotwin_pv2_confirmatory",
                "run_pv2_actiondit_followup_confirmatory",
            )
        ):
            live_processes.append({"pid": int(entry.name), "cmdline": cmdline})
    _require(not live_processes, "superseded 1800-step expansion is still running")
    run_rows: dict[str, Any] = {}
    for seed in (2, 3):
        run_rows[str(seed)] = {}
        for short in CONTROLS:
            root = DEFAULT_SHORT_ROOT / "runs" / f"seed_{seed}" / short
            run_rows[str(seed)][short] = {
                "path": str(root),
                "exists": root.exists(),
                "checkpoint": (
                    _identity(root / "checkpoint.pt")
                    if (root / "checkpoint.pt").is_file()
                    else None
                ),
                "training_summary": (
                    _identity(root / "training_summary.json")
                    if (root / "training_summary.json").is_file()
                    else None
                ),
            }
    return {
        "status": "SUPERSEDED_NOT_SCIENTIFICALLY_TERMINAL",
        "reason": "user_required_author_matched_five_epoch_training_after_short_pilot",
        "short_pilot_decision": _identity(decision_path),
        "short_pilot_macro": decision["macro"],
        "short_pilot_delta": decision["delta"],
        "short_training_budget": {
            "optimizer_steps": 1800,
            "official_samples_per_step": 1,
            "official_sample_exposures": 1800,
            "fraction_of_one_full_three_task_epoch": 1800 / OFFICIAL_SAMPLE_COUNT,
        },
        "partial_seed2_seed3_runs": run_rows,
        "expansion_posttrain_audit_present": (
            DEFAULT_SHORT_ROOT / "expansion_posttrain_audit.json"
        ).is_file(),
        "confirmatory_amendment_present": (
            DEFAULT_SHORT_ROOT / "manifests/confirmatory_seed59_amendment_v1.json"
        ).is_file(),
        "live_processes": [],
        "artifacts_deleted": False,
    }


def _derive_config(
    source: Mapping[str, Any],
    *,
    output_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    seed: int,
    short: str,
    smoke: bool,
) -> dict[str, Any]:
    _require(seed in TRAINING_SEEDS, "training seed must be 1/2/3")
    _require(short in CONTROLS, "control must be c1/c3")
    value = copy.deepcopy(dict(source))
    suffix = "smoke_b128" if smoke else "full5ep_b128"
    value["experiment_id"] = f"pv2_actiondit_{suffix}_seed{seed}_{short}_v1"
    value["output_dir"] = str(
        (
            output_root
            / ("smoke" if smoke else "runs")
            / f"seed_{seed}"
            / short
        ).resolve()
    )
    value["training"].update(
        {
            "seed": seed,
            "max_steps": SMOKE_STEPS if smoke else PREFERRED_MAX_STEPS,
            "official_batch_size": PREFERRED_LOCAL_BATCH,
            "paired_groups_per_batch": PAIRED_GROUPS_PER_GPU,
            "world_size": WORLD_SIZE,
            "gradient_accumulation_steps": 1,
            "effective_official_global_batch": PREFERRED_GLOBAL_BATCH,
            "effective_paired_groups_per_step": EFFECTIVE_PAIRED_GROUPS_PER_STEP,
            "num_workers": 4,
            "save_every": SMOKE_SAVE_EVERY if smoke else FORMAL_SAVE_EVERY,
            "save_optimizer": True,
            "resume": None,
            "epoch_contract": {
                "dataset_samples": OFFICIAL_SAMPLE_COUNT,
                "num_epochs": NUM_EPOCHS,
                "steps_per_epoch": PREFERRED_STEPS_PER_EPOCH,
                "max_steps": PREFERRED_MAX_STEPS,
                "drop_last": False,
            },
        }
    )
    value["policy"]["head_init_seed"] = seed
    value["policy"]["adapter_init_seed"] = seed
    value["execution"].update(
        {
            "runner": "policy_content_adapter_pv2_full5ep",
            "runnable": True,
            "fail_closed": False,
            "long_formal_training": not smoke,
        }
    )
    value["full5ep_protocol_manifest"] = str(protocol_path.resolve())
    value["full5ep_protocol_manifest_sha256"] = protocol_sha256
    value["full5ep_execution_profile"] = "preferred_global_batch128"
    return value


def materialize(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    dry_run_source_root: str | Path = DEFAULT_SHORT_ROOT,
) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    source_root = Path(dry_run_source_root).expanduser().resolve()
    _require(not destination.exists(), f"refusing to reuse full5ep root: {destination}")
    author = _author_setting_audit()
    superseded = _superseded_short_audit()
    source_configs = {
        short: _identity(source_root / "configs/seed_1" / f"{short}.yaml")
        for short in CONTROLS
    }
    official_audit, official_audit_path = _load_json(
        source_root / "runs/seed_1/c1/official_subset_audit.json",
        "official full-three-task audit",
    )
    _require(
        official_audit.get("total_selected_samples") == OFFICIAL_SAMPLE_COUNT
        and official_audit.get("total_selected_episodes") == 1_650
        and official_audit.get("explicit_episode_native_loader", {}).get(
            "selection_mode"
        )
        == "full_550_per_task"
        and official_audit.get("explicit_episode_native_loader", {})
        .get("native_split", {})
        .get("val_set_proportion")
        == 0.0,
        "official full-three-task sample/selection contract differs",
    )
    source_paths = {
        "config_audit.py": Path(__file__).with_name("config_audit.py"),
        "model.py": Path(__file__).with_name("model.py"),
        "train.py": Path(__file__).with_name("train.py"),
        "training_audit.py": Path(__file__).with_name("training_audit.py"),
        "pv2_actiondit_full5ep.py": Path(__file__).resolve(),
        "run_pv2_actiondit_full5ep.sh": Path(__file__).with_name(
            "run_pv2_actiondit_full5ep.sh"
        ),
    }
    protocol_path = destination / "manifests/full5ep_protocol_v1.json"
    core = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "study_classification": {
            "post_hoc_after_short_pilot": True,
            "short_pilot_superseded_for_absolute_training_adequacy": True,
            "primary_pv1_remains_unchanged": True,
            "result_driven_hyperparameter_tuning": (
                "duration changed by explicit user request to match author epochs; "
                "lambda/LR/model/loss remain unchanged"
            ),
        },
        "author_setting_audit": author,
        "superseded_short_setting": superseded,
        "official_three_task_audit": _identity(official_audit_path),
        "official_data": {
            "tasks": list(TASKS),
            "episodes_per_task": 550,
            "clean_episodes_per_task": 50,
            "random_episodes_per_task": 500,
            "selected_samples": OFFICIAL_SAMPLE_COUNT,
            "val_set_proportion": 0.0,
            "num_epochs": NUM_EPOCHS,
            "nominal_unique_sample_exposures": OFFICIAL_SAMPLE_COUNT * NUM_EPOCHS,
        },
        "preferred_execution": {
            "world_size": WORLD_SIZE,
            "local_official_batch": PREFERRED_LOCAL_BATCH,
            "effective_global_official_batch": PREFERRED_GLOBAL_BATCH,
            "gradient_accumulation_steps": 1,
            "steps_per_epoch": PREFERRED_STEPS_PER_EPOCH,
            "max_steps": PREFERRED_MAX_STEPS,
            "save_every": FORMAL_SAVE_EVERY,
            "resume_engine": "accelerate_save_state_load_state",
            "actual_official_batch_slots": (
                PREFERRED_MAX_STEPS * PREFERRED_GLOBAL_BATCH
            ),
            "distributed_sampler_padding_slots": (
                PREFERRED_MAX_STEPS * PREFERRED_GLOBAL_BATCH
                - OFFICIAL_SAMPLE_COUNT * NUM_EPOCHS
            ),
            "paired_groups_per_gpu_step": PAIRED_GROUPS_PER_GPU,
            "effective_paired_groups_per_step": EFFECTIVE_PAIRED_GROUPS_PER_STEP,
            "total_paired_group_draws": (
                PREFERRED_MAX_STEPS * EFFECTIVE_PAIRED_GROUPS_PER_STEP
            ),
            "paired_training_state_count": 720,
            "paired_group_draws_per_state_equivalent": (
                PREFERRED_MAX_STEPS * EFFECTIVE_PAIRED_GROUPS_PER_STEP / 720
            ),
        },
        "oom_fallback_intent": {
            "only_after_audited_pre_optimizer_oom": True,
            "world_size": WORLD_SIZE,
            "local_official_batch": FALLBACK_LOCAL_BATCH,
            "effective_global_official_batch": FALLBACK_GLOBAL_BATCH,
            "steps_per_epoch": FALLBACK_STEPS_PER_EPOCH,
            "max_steps": FALLBACK_MAX_STEPS,
            "silent_fallback_forbidden": True,
        },
        "method_contract": {
            "video_dit_frozen": True,
            "vae_frozen": True,
            "t5_frozen": True,
            "action_dit_trainable": True,
            "head_gca_trainable": True,
            "head_adapter_lr": 1.0e-4,
            "action_dit_lr": 1.0e-5,
            "lr_scheduler": "constant_method_specific_preserved_from_pilot",
            "c1_lambda_contrastive": 0.0,
            "c3_lambda_contrastive": 0.1,
            "only_c1_c3_difference": "contrastive_coefficient_and_gradient",
            "paired_stream_consumed_same_order_by_c1_c3": True,
        },
        "training_seeds": list(TRAINING_SEEDS),
        "source_seed1_configs": source_configs,
        "authorized_config_changes": [
            "experiment_id",
            "output_dir",
            "training.seed",
            "training.max_steps",
            "training.official_batch_size",
            "training.world_size",
            "training.effective_official_global_batch",
            "training.effective_paired_groups_per_step",
            "training.epoch_contract",
            "training.save_every",
            "training.save_optimizer",
            "training.resume",
            "policy.head_init_seed",
            "policy.adapter_init_seed",
            "execution",
            "full5ep_protocol_manifest",
            "full5ep_protocol_manifest_sha256",
            "full5ep_execution_profile",
        ],
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
    }
    protocol = {
        **core,
        "protocol_id": "pv2-full5ep-v1:" + canonical_sha256(core),
    }
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    _write_new_json(protocol_path, protocol)
    configs: dict[str, Any] = {"smoke": {}, "full": {}}
    for seed in TRAINING_SEEDS:
        configs["full"][str(seed)] = {}
        for short in CONTROLS:
            source = load_config(source_configs[short]["path"])
            full = _derive_config(
                source,
                output_root=destination,
                protocol_path=protocol_path,
                protocol_sha256=protocol_sha,
                seed=seed,
                short=short,
                smoke=False,
            )
            path = destination / "configs" / f"seed_{seed}" / f"{short}.yaml"
            _write_new_yaml(path, full)
            validate_config(path)
            configs["full"][str(seed)][short] = _identity(path)
    for short in CONTROLS:
        source = load_config(source_configs[short]["path"])
        smoke = _derive_config(
            source,
            output_root=destination,
            protocol_path=protocol_path,
            protocol_sha256=protocol_sha,
            seed=1,
            short=short,
            smoke=True,
        )
        path = destination / "smoke/configs" / f"{short}.yaml"
        _write_new_yaml(path, smoke)
        validate_config(path)
        configs["smoke"][short] = _identity(path)
    audit = {
        "kind": MATERIALIZATION_KIND,
        "schema_version": 1,
        "status": "PASS",
        "gpu_training_started": False,
        "preferred_batch_smoke_started": False,
        "fallback_authorized": False,
        "protocol": _identity(protocol_path),
        "configs": configs,
        "preferred_max_steps": PREFERRED_MAX_STEPS,
        "fallback_max_steps": FALLBACK_MAX_STEPS,
        "formal_save_every": FORMAL_SAVE_EVERY,
        "resume_engine": "accelerate_save_state_load_state",
    }
    _write_new_json(destination / "materialization_audit.json", audit)
    return audit


def validate_protocol(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, resolved = _load_json(path, "full5ep protocol")
    _require(
        payload.get("kind") == KIND
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "PASS",
        "full5ep protocol kind/version/status differs",
    )
    core = dict(payload)
    protocol_id = core.pop("protocol_id", None)
    _require(
        protocol_id == "pv2-full5ep-v1:" + canonical_sha256(core),
        "full5ep protocol id differs",
    )
    author = payload.get("author_setting_audit")
    _require(isinstance(author, Mapping) and author.get("status") == "PASS", "author audit missing")
    for name in ("task_config", "trainer_source", "readme", "dataset_info"):
        _verify_identity(author.get(name), f"author {name}")
    official_identity = _verify_identity(
        payload.get("official_three_task_audit"), "official three-task audit"
    )
    official, _ = _load_json(official_identity["path"], "official three-task audit")
    _require(official.get("total_selected_samples") == OFFICIAL_SAMPLE_COUNT, "official sample count changed")
    preferred = payload.get("preferred_execution")
    _require(
        isinstance(preferred, Mapping)
        and preferred.get("world_size") == WORLD_SIZE
        and preferred.get("local_official_batch") == PREFERRED_LOCAL_BATCH
        and preferred.get("effective_global_official_batch") == PREFERRED_GLOBAL_BATCH
        and preferred.get("steps_per_epoch") == PREFERRED_STEPS_PER_EPOCH
        and preferred.get("max_steps") == PREFERRED_MAX_STEPS
        and preferred.get("save_every") == FORMAL_SAVE_EVERY
        and preferred.get("resume_engine") == "accelerate_save_state_load_state",
        "preferred full5ep execution differs",
    )
    source_sha = payload.get("source_sha256")
    _require(isinstance(source_sha, Mapping), "full5ep source SHA map missing")
    source_root = Path(__file__).resolve().parent
    _require(
        set(source_sha)
        == {
            "config_audit.py",
            "model.py",
            "train.py",
            "training_audit.py",
            "pv2_actiondit_full5ep.py",
            "run_pv2_actiondit_full5ep.sh",
        },
        "full5ep source set differs",
    )
    for name, digest in source_sha.items():
        _require(_sha256(source_root / name) == digest, f"full5ep source drifted: {name}")
    return payload, resolved


def validate_config(path_or_config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    config = (
        copy.deepcopy(dict(path_or_config))
        if isinstance(path_or_config, Mapping)
        else load_config(path_or_config)
    )
    protocol_path = Path(str(config.get("full5ep_protocol_manifest", ""))).expanduser()
    _require(protocol_path.is_absolute(), "full5ep config lacks protocol path")
    protocol, resolved_protocol = validate_protocol(protocol_path)
    _require(
        config.get("full5ep_protocol_manifest_sha256") == _sha256(resolved_protocol),
        "full5ep config protocol SHA differs",
    )
    seed = config.get("training", {}).get("seed")
    _require(seed in TRAINING_SEEDS, "full5ep training seed differs")
    control = str(config.get("control", ""))
    shorts = [short for short, value in CONTROLS.items() if value == control]
    _require(len(shorts) == 1, "full5ep control differs")
    short = shorts[0]
    smoke = config.get("training", {}).get("max_steps") == SMOKE_STEPS
    source_identity = protocol["source_seed1_configs"][short]
    source_path = _verify_identity(source_identity, f"source seed1 {short} config")["path"]
    source = load_config(source_path)
    validate_execution_ready(source)
    expected = _derive_config(
        source,
        output_root=Path(protocol_path).parents[1],
        protocol_path=resolved_protocol,
        protocol_sha256=_sha256(resolved_protocol),
        seed=int(seed),
        short=short,
        smoke=smoke,
    )
    _require(config == expected, "full5ep config differs outside authorized transform")
    return {
        "status": "PASS",
        "seed": int(seed),
        "short": short,
        "control": control,
        "smoke": smoke,
        "global_batch": PREFERRED_GLOBAL_BATCH,
        "max_steps": SMOKE_STEPS if smoke else PREFERRED_MAX_STEPS,
        "protocol_id": protocol["protocol_id"],
    }


def run_training(
    config_path: str | Path, *, resume_from: str | Path | None = None
) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    validate_config(config_file)
    original_validator = train_module.validate_execution_ready
    train_module.validate_execution_ready = validate_config
    try:
        return train_module.run(config_file, resume_from=resume_from)
    finally:
        train_module.validate_execution_ready = original_validator


def audit_pair(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int,
    smoke: bool,
    require_action_gate: bool = True,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    expected_steps = SMOKE_STEPS if smoke else PREFERRED_MAX_STEPS
    runs: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    for short, control in CONTROLS.items():
        config_path = (
            root / "smoke/configs" / f"{short}.yaml"
            if smoke
            else root / "configs" / f"seed_{seed}" / f"{short}.yaml"
        )
        config = load_config(config_path)
        validate_config(config)
        configs[short] = config
        run_root = Path(config["output_dir"])
        run = _audit_training_run(
            run_root,
            control=control,
            expected_steps=expected_steps,
            expected_lambda=0.0 if short == "c1" else 0.1,
        )
        if require_action_gate:
            run["action_gate"] = _audit_action_gate(
                run_root / "pre_online_action_gate.json", control
            )
        runs[short] = run
    for field in (
        "official_sample_sequence_sha256",
        "paired_physical_state_sequence_sha256",
        "matched_stream_contract_sha256",
    ):
        _require(
            runs["c1"]["summary"][field] == runs["c3"]["summary"][field],
            f"C1/C3 {field} differs",
        )
    for field in (
        "source_fp32_content_head_sha256",
        "source_fp32_adapter_sha256",
        "training_fp32_content_head_sha256",
        "training_fp32_adapter_sha256",
    ):
        _require(
            runs["c1"]["summary"]["initialization"][field]
            == runs["c3"]["summary"]["initialization"][field],
            f"C1/C3 initialization differs at {field}",
        )
    _require(
        runs["c1"]["step_rng_rows_sha256"]
        == runs["c3"]["step_rng_rows_sha256"],
        "C1/C3 step RNG rows differ",
    )
    return {
        "kind": POSTTRAIN_KIND,
        "schema_version": 1,
        "status": "PASS",
        "stage": "smoke" if smoke else "full5ep",
        "seed": seed,
        "steps_per_control": expected_steps,
        "world_size": WORLD_SIZE,
        "global_batch": PREFERRED_GLOBAL_BATCH,
        "runs": {
            short: {
                "checkpoint": runs[short]["checkpoint"],
                "action_dit_update": runs[short]["updates"]["action_dit"],
                "head_gca_update": runs[short]["updates"]["head_and_adapter"],
                "action_gate": runs[short].get("action_gate"),
                "final_gate_raw": runs[short]["summary"]["final_gate_raw"],
            }
            for short in CONTROLS
        },
        "shared_sequences": {
            field: runs["c1"]["summary"][field]
            for field in (
                "official_sample_sequence_sha256",
                "paired_physical_state_sequence_sha256",
                "matched_stream_contract_sha256",
            )
        },
        "shared_step_rng_rows_sha256": runs["c1"]["step_rng_rows_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    validate_parser = sub.add_parser("validate-config")
    validate_parser.add_argument("--config", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume the strict native Accelerate state (default: latest).",
    )
    audit_parser = sub.add_parser("audit-pair")
    audit_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    audit_parser.add_argument("--seed", type=int, required=True)
    audit_parser.add_argument("--smoke", action="store_true")
    audit_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize(output_root=args.output_root)
    elif args.command == "validate-config":
        result = validate_config(args.config)
    elif args.command == "train":
        result = {
            "status": "PASS",
            "output_dir": str(
                run_training(args.config, resume_from=args.resume)
            ),
        }
    else:
        result = audit_pair(
            output_root=args.output_root,
            seed=args.seed,
            smoke=args.smoke,
        )
        _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Pv2Full5EpochError",
    "audit_pair",
    "materialize",
    "run_training",
    "validate_config",
    "validate_protocol",
]
