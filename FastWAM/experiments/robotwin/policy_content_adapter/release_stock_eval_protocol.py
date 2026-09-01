"""Create and validate the author-stock seed-42 evaluation amendment.

This module is deliberately CPU-only.  It records a transparent evaluation
profile amendment for the already-trained formal retry-1 C1/C3 checkpoints;
it never loads a checkpoint with torch, launches RoboTwin, or touches a GPU.

The author-stock evaluator starts its candidate scan at
``100000 * (1 + seed)`` and independently expert-filters candidates for every
checkpoint/task/domain invocation.  Consequently the candidate pool can be
locked before evaluation, but the 100 accepted episodes are *not* a paired
episode set and no paired-episode claim is made by this protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .evaluation_protocol import TASKS as _TASKS
from .p_mode_selection import (
    build_seed_bank_descriptor,
    validate_formal_protocol_lock_manifest_payload,
    validate_seed_bank_descriptor,
)
from .runtime_utils import PROJECT_ROOT


PROFILE = "author_stock_seed42_unpaired_v1"
SIMULATOR_SEED = 42
EPISODES_PER_CELL = 100
TASKS = tuple(_TASKS)
TASK_CONFIGS = ("demo_clean", "demo_randomized")
DOMAINS = {"demo_clean": "clean", "demo_randomized": "random"}
FORMAL_SEEDS = (1, 2, 3)
CONTROLS = {"c1": "c1_architecture_only", "c3": "c3_ours"}
KIND = "policy_release_author_stock_eval_protocol_amendment"
SCHEMA_VERSION = 1
DEFAULT_FORMAL_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "formal_c1_c3_release_v1_retry1"
).resolve()
DEFAULT_AMENDMENT_RELATIVE = Path(
    "manifests/author_stock_seed42_unpaired_v1.json"
)


class StockEvalProtocolError(ValueError):
    """The stock evaluation amendment cannot be proven from immutable inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StockEvalProtocolError(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required file is missing: {resolved}")
    before = resolved.stat()
    digest = _file_sha256(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {resolved}",
    )
    return {
        "kind": "file",
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StockEvalProtocolError(f"cannot parse {label}: {resolved}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, resolved


def _verify_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} identity must be an object")
    path = Path(str(value.get("path", ""))).expanduser()
    _require(path.is_absolute(), f"{label} path must be absolute")
    actual = _file_identity(path)
    for field in ("kind", "path", "size_bytes", "sha256"):
        expected = (
            str(Path(str(value.get(field, ""))).expanduser().resolve())
            if field == "path"
            else value.get(field)
        )
        _require(actual[field] == expected, f"{label} {field} changed")
    return actual


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StockEvalProtocolError(f"cannot parse {label}: {path}: {exc}") from exc
    _require(isinstance(value, Mapping), f"{label} root must be an object")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
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
            raise StockEvalProtocolError(
                f"refusing to overwrite immutable amendment: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def _runtime_artifacts(project_root: Path, robotwin_root: Path) -> dict[str, Any]:
    task_config_root = robotwin_root / "task_config"
    environment_sources = {
        task: _file_identity(robotwin_root / f"envs/{task}.py") for task in TASKS
    }
    artifacts = {
        "author_stock_eval_policy": _file_identity(robotwin_root / "script/eval_policy.py"),
        "pinned_eval_launcher": _file_identity(
            project_root
            / "experiments/robotwin/policy_content_adapter/pinned_eval_policy.py"
        ),
        "robotwin_base_task": _file_identity(robotwin_root / "envs/_base_task.py"),
        "sim_robotwin_config": _file_identity(project_root / "configs/sim_robotwin.yaml"),
        "train_defaults_config": _file_identity(project_root / "configs/train.yaml"),
        "task_configs": {
            task_config: _file_identity(task_config_root / f"{task_config}.yml")
            for task_config in TASK_CONFIGS
        },
        "camera_config": _file_identity(task_config_root / "_camera_config.yml"),
        "embodiment_registry": _file_identity(
            task_config_root / "_embodiment_config.yml"
        ),
        "resolved_embodiment_config": _file_identity(
            robotwin_root / "assets/embodiments/aloha-agilex/config.yml"
        ),
        "eval_step_limits": _file_identity(task_config_root / "_eval_step_limit.yml"),
        "task_environment_sources": environment_sources,
    }

    sim = _load_yaml(project_root / "configs/sim_robotwin.yaml", "sim config")
    evaluation = sim.get("EVALUATION")
    _require(isinstance(evaluation, Mapping), "sim config lacks EVALUATION")
    _require(
        evaluation.get("eval_num_episodes") == EPISODES_PER_CELL,
        "sim config eval_num_episodes is not 100",
    )
    _require(
        evaluation.get("instruction_type") == "unseen",
        "sim config instruction_type is not unseen",
    )
    _require(evaluation.get("replan_steps") == 24, "sim config replan_steps is not 24")
    _require(
        evaluation.get("skip_get_obs_within_replan") is True,
        "sim config skip_get_obs_within_replan is not true",
    )
    _require(
        evaluation.get("num_inference_steps") == "${eval_num_inference_steps}",
        "sim config no longer inherits eval_num_inference_steps",
    )
    train_defaults = _load_yaml(project_root / "configs/train.yaml", "train defaults")
    _require(
        train_defaults.get("eval_num_inference_steps") == 10,
        "train defaults eval_num_inference_steps is not 10",
    )

    step_limits = _load_yaml(
        task_config_root / "_eval_step_limit.yml", "eval-step limits"
    )
    for task in TASKS:
        _require(
            isinstance(step_limits.get(task), int) and step_limits[task] > 0,
            f"eval-step limit is missing for {task}",
        )
    for task_config in TASK_CONFIGS:
        config = _load_yaml(
            task_config_root / f"{task_config}.yml", f"{task_config} config"
        )
        _require(
            config.get("embodiment") == ["aloha-agilex"],
            f"{task_config} embodiment is not aloha-agilex",
        )
        camera = config.get("camera")
        _require(isinstance(camera, Mapping), f"{task_config} camera block is missing")
        _require(
            camera.get("collect_head_camera") is True
            and camera.get("collect_wrist_camera") is True,
            f"{task_config} does not enable head and wrist cameras",
        )
    return artifacts


def _checkpoint_rows(posttrain: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(posttrain.get("status") == "PASS", "posttrain audit is not PASS")
    _require(
        posttrain.get("formal_training_complete") is True,
        "posttrain audit does not prove completed formal training",
    )
    _require(
        posttrain.get("online_rollout_started") is False,
        "posttrain audit says online rollout already started",
    )
    raw = posttrain.get("checkpoints")
    _require(isinstance(raw, Mapping), "posttrain audit lacks checkpoints")
    rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        pair = raw.get(str(seed))
        _require(isinstance(pair, Mapping), f"posttrain audit lacks seed {seed}")
        _require(set(pair) == set(CONTROLS), f"seed {seed} checkpoint pair changed")
        for short, control in CONTROLS.items():
            identity = _verify_identity(
                {"kind": "file", **dict(pair[short])},
                f"seed {seed}/{short} checkpoint",
            )
            rows.append(
                {
                    "control": control,
                    "training_seed": seed,
                    "path": identity["path"],
                    "size_bytes": identity["size_bytes"],
                    "sha256": identity["sha256"],
                }
            )
    return rows


def materialize_stock_eval_amendment(
    *,
    formal_root: str | Path = DEFAULT_FORMAL_ROOT,
    output: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    robotwin_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Create one immutable stock-evaluation amendment and validate it."""

    root = Path(formal_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    robotwin = (
        Path(robotwin_root).expanduser().resolve()
        if robotwin_root is not None
        else (project / "third_party/RoboTwin").resolve()
    )
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else (root / DEFAULT_AMENDMENT_RELATIVE).resolve()
    )
    _require(not destination.exists(), f"refusing to overwrite immutable amendment: {destination}")

    posttrain, posttrain_path = _load_json(
        root / "strict_posttrain_pair_audit.json", "strict posttrain pair audit"
    )
    checkpoints = _checkpoint_rows(posttrain)
    prelaunch = posttrain.get("prelaunch")
    _require(isinstance(prelaunch, Mapping), "posttrain audit lacks prelaunch ancestry")
    materialization_path = Path(
        str(prelaunch.get("materialization_manifest", ""))
    ).expanduser().resolve()
    _require(
        materialization_path == (root / "materialization_manifest.json").resolve(),
        "posttrain materialization path differs from formal root",
    )
    materialization, _ = _load_json(materialization_path, "materialization manifest")
    _require(materialization.get("status") == "PASS", "materialization is not PASS")
    artifacts = materialization.get("artifacts")
    _require(isinstance(artifacts, Mapping), "materialization lacks artifacts")
    original_bank_identity = _verify_identity(
        artifacts.get("final_test_seed_bank"), "original final-test seed bank"
    )
    formal_lock_identity = _verify_identity(
        artifacts.get("formal_protocol_lock"), "formal protocol lock"
    )
    original_raw, _ = _load_json(
        original_bank_identity["path"], "original final-test seed bank"
    )
    try:
        original_bank = validate_seed_bank_descriptor(
            original_raw, expected_purpose="final_test"
        )
    except ValueError as exc:
        raise StockEvalProtocolError(f"invalid original final-test bank: {exc}") from exc
    _require(
        original_bank["simulator_seed"] == 47,
        "original checkpoint-bound final-test bank is not seed 47",
    )
    _require(
        original_bank["simulator_seed_bank_id"]
        == artifacts.get("final_test_seed_bank_id")
        == prelaunch.get("final_test_seed_bank_id"),
        "original final-test seed-bank id ancestry differs",
    )
    _require(
        original_bank_identity["sha256"]
        == prelaunch.get("final_test_seed_bank_sha256"),
        "original final-test seed-bank SHA ancestry differs",
    )
    formal_lock_raw, _ = _load_json(
        formal_lock_identity["path"], "formal protocol lock"
    )
    try:
        validate_formal_protocol_lock_manifest_payload(formal_lock_raw)
    except ValueError as exc:
        raise StockEvalProtocolError(f"invalid formal protocol lock: {exc}") from exc
    _require(
        formal_lock_identity["sha256"]
        == prelaunch.get("formal_protocol_lock_sha256"),
        "formal protocol-lock SHA ancestry differs",
    )

    runtime_artifacts = _runtime_artifacts(project, robotwin)
    runtime_seed_bank = build_seed_bank_descriptor(
        simulator_seed=SIMULATOR_SEED,
        episodes_per_cell=EPISODES_PER_CELL,
        evaluator_source=runtime_artifacts["author_stock_eval_policy"]["path"],
        purpose="final_test",
        disjoint_from=original_bank["disjoint_from"],
        lock_ancestry=original_bank["lock_ancestry"],
    )

    amendment_projection = {
        "profile": PROFILE,
        "simulator_seed": SIMULATOR_SEED,
        "episodes_per_cell": EPISODES_PER_CELL,
        "episode_pairing": "not_claimed",
        "runtime_seed_bank_id": runtime_seed_bank["simulator_seed_bank_id"],
        "original_checkpoint_seed_bank_id": original_bank["simulator_seed_bank_id"],
        "original_checkpoint_seed_bank_sha256": original_bank_identity["sha256"],
        "formal_protocol_lock_sha256": formal_lock_identity["sha256"],
        "posttrain_audit_sha256": _file_identity(posttrain_path)["sha256"],
        "checkpoint_sha256": [row["sha256"] for row in checkpoints],
        "runtime_artifact_sha256": {
            "author_stock_eval_policy": runtime_artifacts["author_stock_eval_policy"]["sha256"],
            "pinned_eval_launcher": runtime_artifacts["pinned_eval_launcher"]["sha256"],
            "robotwin_base_task": runtime_artifacts["robotwin_base_task"]["sha256"],
            "sim_robotwin_config": runtime_artifacts["sim_robotwin_config"]["sha256"],
            "train_defaults_config": runtime_artifacts["train_defaults_config"]["sha256"],
            "task_configs": {
                name: item["sha256"]
                for name, item in runtime_artifacts["task_configs"].items()
            },
            "camera_config": runtime_artifacts["camera_config"]["sha256"],
            "embodiment_registry": runtime_artifacts["embodiment_registry"]["sha256"],
            "resolved_embodiment_config": runtime_artifacts["resolved_embodiment_config"]["sha256"],
            "eval_step_limits": runtime_artifacts["eval_step_limits"]["sha256"],
            "task_environment_sources": {
                name: item["sha256"]
                for name, item in runtime_artifacts["task_environment_sources"].items()
            },
        },
    }
    amendment_id = "author-stock-eval-amendment-v1:" + _canonical_sha256(
        amendment_projection
    )
    statement = (
        "This amendment overrides only the evaluation simulator seed/profile "
        "for the six already-audited checkpoints. It does not mutate model "
        "weights, checkpoints, training, or evaluator/config source bytes. "
        "RoboTwin's author-stock evaluator independently expert-filters "
        "candidate episodes for every checkpoint/task/domain invocation; the "
        "actual 100 accepted episodes are therefore NOT paired across models "
        "or cells, and no paired-episode claim is made."
    )
    payload = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "profile": PROFILE,
        "amendment_id": amendment_id,
        "simulator_seed": SIMULATOR_SEED,
        "episodes_per_cell": EPISODES_PER_CELL,
        "tasks": list(TASKS),
        "task_configs": list(TASK_CONFIGS),
        "domains": dict(DOMAINS),
        "episode_pairing": "not_claimed",
        "checkpoints": checkpoints,
        "runtime_seed_bank": runtime_seed_bank,
        "original_checkpoint_seed_bank_id": original_bank["simulator_seed_bank_id"],
        "original_checkpoint_seed_bank_sha256": original_bank_identity["sha256"],
        "formal_protocol_lock_sha256": formal_lock_identity["sha256"],
        "ancestry": {
            "strict_posttrain_pair_audit": _file_identity(posttrain_path),
            "materialization_manifest": _file_identity(materialization_path),
            "original_final_test_seed_bank": original_bank_identity,
            "formal_protocol_lock": formal_lock_identity,
        },
        "runtime_artifacts": runtime_artifacts,
        "evaluation_settings": {
            "instruction_type": "unseen",
            "replan_steps": 24,
            "num_inference_steps": 10,
            "skip_get_obs_within_replan": True,
            "action_horizon": None,
        },
        "scope": {
            "checkpoint_count": 6,
            "cells_per_checkpoint": len(TASKS) * len(TASK_CONFIGS),
            "accepted_episodes_per_cell": EPISODES_PER_CELL,
            "accepted_episodes_per_checkpoint": len(TASKS)
            * len(TASK_CONFIGS)
            * EPISODES_PER_CELL,
            "total_accepted_rollouts": 6
            * len(TASKS)
            * len(TASK_CONFIGS)
            * EPISODES_PER_CELL,
            "instruction_type": "unseen",
        },
        "amendment": {
            "overrides_only": [
                "evaluation.simulator_seed",
                "evaluation.seed_profile",
            ],
            "weights_mutated": False,
            "checkpoints_rewritten": False,
            "training_protocol_mutated": False,
            "runtime_source_mutated": False,
            "candidate_pool_locked_before_evaluation": True,
            "accepted_members_locked_before_evaluation": False,
            "expert_filtering": "independent_per_checkpoint_task_domain",
            "runtime_disclosure": (
                "The vendored author-stock eval_policy.py is bound byte-for-byte. "
                "The experiment-owned pinned launcher and the local RoboTwin "
                "_base_task.py are separately bound and disclosed as "
                "hardware/transport integration sources; this protocol does not "
                "claim a byte-identical upstream RoboTwin runtime tree."
            ),
            "transparent_statement": statement,
        },
    }
    _write_new_json(destination, payload)
    return validate_stock_eval_amendment(destination)


def validate_stock_eval_amendment(
    path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Validate schema, semantics, ancestry, and every currently-bound byte."""

    value, resolved = _load_json(path, "stock evaluation amendment")
    _require(value.get("kind") == KIND, "amendment kind changed")
    _require(value.get("schema_version") == SCHEMA_VERSION, "schema version changed")
    _require(value.get("status") == "PASS", "amendment is not PASS")
    _require(value.get("profile") == PROFILE, "evaluation profile changed")
    _require(value.get("simulator_seed") == SIMULATOR_SEED, "simulator seed changed")
    _require(
        value.get("episodes_per_cell") == EPISODES_PER_CELL,
        "episodes per cell changed",
    )
    _require(value.get("tasks") == list(TASKS), "task set/order changed")
    _require(
        value.get("task_configs") == list(TASK_CONFIGS),
        "task-config set/order changed",
    )
    _require(value.get("domains") == DOMAINS, "domain mapping changed")
    _require(value.get("episode_pairing") == "not_claimed", "pairing claim changed")

    rows = value.get("checkpoints")
    _require(isinstance(rows, list) and len(rows) == 6, "six checkpoint rows required")
    expected_pairs = [
        (control, seed)
        for seed in FORMAL_SEEDS
        for control in CONTROLS.values()
    ]
    observed_pairs: list[tuple[str, int]] = []
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"checkpoint row {index} is not an object")
        control = row.get("control")
        seed = row.get("training_seed")
        observed_pairs.append((control, seed))
        identity = _verify_identity(
            {"kind": "file", **dict(row)}, f"checkpoint row {index}"
        )
        normalized_rows.append(
            {
                "control": control,
                "training_seed": seed,
                "path": identity["path"],
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    _require(observed_pairs == expected_pairs, "checkpoint control/seed rows changed")

    ancestry = value.get("ancestry")
    _require(isinstance(ancestry, Mapping), "amendment ancestry is missing")
    posttrain_identity = _verify_identity(
        ancestry.get("strict_posttrain_pair_audit"), "strict posttrain pair audit"
    )
    _verify_identity(ancestry.get("materialization_manifest"), "materialization manifest")
    original_identity = _verify_identity(
        ancestry.get("original_final_test_seed_bank"), "original final-test seed bank"
    )
    lock_identity = _verify_identity(
        ancestry.get("formal_protocol_lock"), "formal protocol lock"
    )
    _require(
        value.get("original_checkpoint_seed_bank_sha256") == original_identity["sha256"],
        "original checkpoint seed-bank SHA differs",
    )
    _require(
        value.get("formal_protocol_lock_sha256") == lock_identity["sha256"],
        "formal protocol-lock SHA differs",
    )
    posttrain, _ = _load_json(posttrain_identity["path"], "strict posttrain pair audit")
    expected_rows = _checkpoint_rows(posttrain)
    _require(normalized_rows == expected_rows, "checkpoint rows differ from posttrain audit")

    original_raw, _ = _load_json(original_identity["path"], "original final-test bank")
    try:
        original = validate_seed_bank_descriptor(
            original_raw, expected_purpose="final_test"
        )
    except ValueError as exc:
        raise StockEvalProtocolError(f"invalid original final-test bank: {exc}") from exc
    _require(original["simulator_seed"] == 47, "original bank is not seed 47")
    _require(
        value.get("original_checkpoint_seed_bank_id")
        == original["simulator_seed_bank_id"],
        "original checkpoint seed-bank id differs",
    )
    lock_raw, _ = _load_json(lock_identity["path"], "formal protocol lock")
    try:
        validate_formal_protocol_lock_manifest_payload(lock_raw)
    except ValueError as exc:
        raise StockEvalProtocolError(f"invalid formal protocol lock: {exc}") from exc

    runtime = value.get("runtime_artifacts")
    _require(isinstance(runtime, Mapping), "runtime artifacts are missing")
    for name in (
        "author_stock_eval_policy",
        "pinned_eval_launcher",
        "robotwin_base_task",
        "sim_robotwin_config",
        "train_defaults_config",
        "camera_config",
        "embodiment_registry",
        "resolved_embodiment_config",
        "eval_step_limits",
    ):
        _verify_identity(runtime.get(name), f"runtime artifact {name}")
    for collection_name, expected_names in (
        ("task_configs", TASK_CONFIGS),
        ("task_environment_sources", TASKS),
    ):
        collection = runtime.get(collection_name)
        _require(
            isinstance(collection, Mapping) and set(collection) == set(expected_names),
            f"runtime {collection_name} set changed",
        )
        for name in expected_names:
            _verify_identity(collection[name], f"runtime {collection_name}/{name}")

    try:
        runtime_bank = validate_seed_bank_descriptor(
            value.get("runtime_seed_bank"), expected_purpose="final_test"
        )
    except ValueError as exc:
        raise StockEvalProtocolError(f"invalid runtime seed bank: {exc}") from exc
    _require(runtime_bank["simulator_seed"] == SIMULATOR_SEED, "runtime bank seed changed")
    _require(
        runtime_bank["candidate_start_seed"] == 4_300_000,
        "stock candidate start is not 4.3m",
    )
    _require(
        runtime_bank["episodes_per_cell"] == EPISODES_PER_CELL,
        "runtime bank episode count changed",
    )
    _require(
        runtime_bank["evaluator_source_sha256"]
        == runtime["author_stock_eval_policy"]["sha256"],
        "runtime bank does not bind author stock eval_policy.py",
    )
    _require(
        runtime_bank["lock_ancestry"] == original["lock_ancestry"],
        "runtime bank changed the original formal lock ancestry",
    )

    settings = value.get("evaluation_settings")
    _require(
        settings
        == {
            "instruction_type": "unseen",
            "replan_steps": 24,
            "num_inference_steps": 10,
            "skip_get_obs_within_replan": True,
            "action_horizon": None,
        },
        "stock evaluation settings changed",
    )

    amendment = value.get("amendment")
    _require(isinstance(amendment, Mapping), "amendment semantics are missing")
    _require(
        amendment.get("overrides_only")
        == ["evaluation.simulator_seed", "evaluation.seed_profile"],
        "amendment override scope changed",
    )
    for field in (
        "weights_mutated",
        "checkpoints_rewritten",
        "training_protocol_mutated",
        "runtime_source_mutated",
    ):
        _require(amendment.get(field) is False, f"amendment unexpectedly sets {field}")
    _require(
        amendment.get("accepted_members_locked_before_evaluation") is False
        and amendment.get("expert_filtering")
        == "independent_per_checkpoint_task_domain",
        "independent expert-filtering semantics changed",
    )
    disclosure = amendment.get("runtime_disclosure")
    _require(
        isinstance(disclosure, str)
        and "does not claim a byte-identical upstream RoboTwin runtime tree" in disclosure,
        "local runtime-source disclosure is missing",
    )
    statement = amendment.get("transparent_statement")
    _require(
        isinstance(statement, str)
        and "does not mutate model weights" in statement
        and "NOT paired" in statement,
        "transparent non-mutation/non-pairing statement is missing",
    )

    projection = {
        "profile": PROFILE,
        "simulator_seed": SIMULATOR_SEED,
        "episodes_per_cell": EPISODES_PER_CELL,
        "episode_pairing": "not_claimed",
        "runtime_seed_bank_id": runtime_bank["simulator_seed_bank_id"],
        "original_checkpoint_seed_bank_id": original["simulator_seed_bank_id"],
        "original_checkpoint_seed_bank_sha256": original_identity["sha256"],
        "formal_protocol_lock_sha256": lock_identity["sha256"],
        "posttrain_audit_sha256": posttrain_identity["sha256"],
        "checkpoint_sha256": [row["sha256"] for row in normalized_rows],
        "runtime_artifact_sha256": {
            "author_stock_eval_policy": runtime["author_stock_eval_policy"]["sha256"],
            "pinned_eval_launcher": runtime["pinned_eval_launcher"]["sha256"],
            "robotwin_base_task": runtime["robotwin_base_task"]["sha256"],
            "sim_robotwin_config": runtime["sim_robotwin_config"]["sha256"],
            "train_defaults_config": runtime["train_defaults_config"]["sha256"],
            "task_configs": {
                name: runtime["task_configs"][name]["sha256"] for name in TASK_CONFIGS
            },
            "camera_config": runtime["camera_config"]["sha256"],
            "embodiment_registry": runtime["embodiment_registry"]["sha256"],
            "resolved_embodiment_config": runtime["resolved_embodiment_config"]["sha256"],
            "eval_step_limits": runtime["eval_step_limits"]["sha256"],
            "task_environment_sources": {
                name: runtime["task_environment_sources"][name]["sha256"]
                for name in TASKS
            },
        },
    }
    expected_amendment_id = "author-stock-eval-amendment-v1:" + _canonical_sha256(
        projection
    )
    _require(value.get("amendment_id") == expected_amendment_id, "amendment id differs")
    return value, resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("materialize", help="create-only amendment materialization")
    create.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    create.add_argument("--output", default=None)
    create.add_argument("--project-root", default=str(PROJECT_ROOT))
    create.add_argument("--robotwin-root", default=None)
    validate = subparsers.add_parser("validate", help="validate an existing amendment")
    validate.add_argument("--path", required=True)
    args = parser.parse_args(argv)
    if args.command == "materialize":
        payload, destination = materialize_stock_eval_amendment(
            formal_root=args.formal_root,
            output=args.output,
            project_root=args.project_root,
            robotwin_root=args.robotwin_root,
        )
    else:
        payload, destination = validate_stock_eval_amendment(args.path)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "profile": payload["profile"],
                "amendment_id": payload["amendment_id"],
                "path": str(destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "PROFILE",
    "SIMULATOR_SEED",
    "EPISODES_PER_CELL",
    "TASKS",
    "TASK_CONFIGS",
    "DEFAULT_FORMAL_ROOT",
    "DEFAULT_AMENDMENT_RELATIVE",
    "StockEvalProtocolError",
    "materialize_stock_eval_amendment",
    "validate_stock_eval_amendment",
]
