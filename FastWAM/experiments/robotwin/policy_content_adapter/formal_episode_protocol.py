"""Exact, policy-independent RoboTwin episode realization and replay.

The checkpoint-bound final-test seed bank is a *candidate pool*.  RoboTwin's
stock evaluator filters that pool separately inside every policy process, so
two checkpoints can silently execute different accepted seeds.  This module
keeps the immutable candidate bank, but realizes it once (before any policy is
loaded) into six exact task/domain lists.  Every policy then replays the same
ordered ``(seed, instruction)`` entries without replacement.

No vendored RoboTwin source is edited.  :mod:`pinned_eval_policy` installs the
``eval_policy`` closure below into its dynamically loaded evaluator module.
Realization and replay both use the already pinned CUDA/SAPIEN PCI device.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .p_mode_selection import (
    canonical_sha256,
    validate_formal_protocol_lock_manifest_payload,
    validate_seed_bank_descriptor,
)
from .robotwin_gpu_runtime import canonical_nvidia_pci_address


CELL_SCHEMA = "policy_content_adapter.formal_episode_realization_cell"
CELL_SCHEMA_VERSION = 1
BANK_SCHEMA = "policy_content_adapter.formal_episode_realization_bank"
BANK_SCHEMA_VERSION = 1
TRACE_SCHEMA = "policy_content_adapter.formal_episode_replay_trace"
TRACE_SCHEMA_VERSION = 1
CELL_ID_PREFIX = "robotwin-formal-cell-v1:"
BANK_ID_PREFIX = "robotwin-formal-realization-v1:"
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
TASK_CONFIGS = ("demo_clean", "demo_randomized")
TASK_CONFIG_TO_DOMAIN = {
    "demo_clean": "clean",
    "demo_randomized": "official_random",
}
EPISODES_PER_CELL = 100
DEFAULT_ROBOTWIN_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party/RoboTwin"
).resolve()
INSTRUCTION_CHOICE_SCHEMA = "robotwin.formal_deterministic_instruction_choice.v1"
SELECTION_SEMANTICS = (
    "ascending candidate-bank members; one stock setup_demo/play_once expert pass; "
    "accept iff plan_success and check_success; first 100 accepted; no policy loaded; "
    "instruction deterministically selected from stock unseen candidate descriptions"
)


class FormalEpisodeProtocolError(RuntimeError):
    """Exact episode realization or replay violated the locked protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalEpisodeProtocolError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required artifact is missing: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"artifact changed while hashing: {resolved}",
    )
    return {
        "kind": "file",
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _validate_file_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a file identity")
    path = Path(str(value.get("path", ""))).expanduser()
    _require(path.is_absolute(), f"{label}.path must be absolute")
    actual = stable_file_identity(path)
    for field in ("kind", "size_bytes", "sha256"):
        _require(value.get(field) == actual[field], f"{label}.{field} differs")
    return actual


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FormalEpisodeProtocolError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, resolved


def _exclusive_json(path: str | Path, value: Mapping[str, Any]) -> Path:
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
            raise FormalEpisodeProtocolError(
                f"refusing to overwrite immutable artifact: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def task_config_artifacts(robotwin_root: str | Path, task_config: str) -> dict[str, Any]:
    root = Path(robotwin_root).expanduser().resolve()
    _require(task_config in TASK_CONFIGS, f"unsupported task_config {task_config!r}")
    paths = {
        "domain_task_config": root / f"task_config/{task_config}.yml",
        "camera_config": root / "task_config/_camera_config.yml",
        "embodiment_config": root / "task_config/_embodiment_config.yml",
        "eval_step_limit": root / "task_config/_eval_step_limit.yml",
    }
    return {name: stable_file_identity(path) for name, path in paths.items()}


def _source_artifacts(robotwin_root: str | Path) -> dict[str, Any]:
    root = Path(robotwin_root).expanduser().resolve()
    here = Path(__file__).resolve()
    return {
        "stock_evaluator": stable_file_identity(root / "script/eval_policy.py"),
        "stock_instruction_generator": stable_file_identity(
            root / "description/utils/generate_episode_instructions.py"
        ),
        "pinned_evaluator": stable_file_identity(here.parent / "pinned_eval_policy.py"),
        "gpu_runtime": stable_file_identity(here.parent / "robotwin_gpu_runtime.py"),
        "formal_episode_protocol": stable_file_identity(here),
    }


def _parent_artifacts(
    candidate_bank_path: str | Path,
    formal_lock_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_bank, bank_path = _load_json(candidate_bank_path, "candidate seed bank")
    try:
        candidate_bank = validate_seed_bank_descriptor(
            raw_bank, expected_purpose="final_test"
        )
    except ValueError as exc:
        raise FormalEpisodeProtocolError(f"candidate seed bank is invalid: {exc}") from exc
    raw_lock, lock_path = _load_json(formal_lock_path, "formal protocol lock")
    try:
        formal_lock = validate_formal_protocol_lock_manifest_payload(raw_lock)
    except ValueError as exc:
        raise FormalEpisodeProtocolError(f"formal protocol lock is invalid: {exc}") from exc
    bank_identity = stable_file_identity(bank_path)
    lock_identity = stable_file_identity(lock_path)
    bound_lock = candidate_bank.get("lock_ancestry", {}).get(
        "formal_protocol_lock_manifest"
    )
    _require(
        isinstance(bound_lock, Mapping),
        "candidate seed bank lacks a formal protocol-lock binding",
    )
    for field in ("path", "size_bytes", "sha256"):
        expected = lock_identity[field]
        actual = (
            str(Path(str(bound_lock.get(field, ""))).expanduser().resolve())
            if field == "path"
            else bound_lock.get(field)
        )
        _require(
            actual == expected,
            f"candidate seed bank formal-lock {field} differs",
        )
    bound_selection = candidate_bank.get("lock_ancestry", {}).get(
        "p_mode_selection_manifest"
    )
    locked_selection = formal_lock.get("p_mode_selection_manifest")
    _require(
        isinstance(bound_selection, Mapping) and isinstance(locked_selection, Mapping),
        "candidate bank/formal lock lacks a P-mode selection binding",
    )
    for field in ("path", "size_bytes", "sha256"):
        left = (
            str(Path(str(bound_selection.get(field, ""))).expanduser().resolve())
            if field == "path"
            else bound_selection.get(field)
        )
        right = (
            str(Path(str(locked_selection.get(field, ""))).expanduser().resolve())
            if field == "path"
            else locked_selection.get(field)
        )
        _require(
            left == right,
            f"candidate bank/formal lock P-mode selection {field} differs",
        )
    _require(
        candidate_bank["candidate_start_seed"] == candidate_bank["members"][0],
        "candidate start seed differs from the first explicit member",
    )
    _require(
        len(candidate_bank["members"]) >= EPISODES_PER_CELL,
        "candidate pool is smaller than the required final-test cell",
    )
    _require(
        candidate_bank["selection"]
        == "ascending_integer_candidates_filtered_by_setup_demo_play_once_"
        "plan_success_and_check_success",
        "candidate seed bank binds a different formal protocol lock",
    )
    _require(
        candidate_bank["episodes_per_cell"] == EPISODES_PER_CELL,
        f"candidate bank must declare {EPISODES_PER_CELL} episodes per cell",
    )
    return candidate_bank, formal_lock, bank_identity, lock_identity


def audit_selector_inputs(
    *,
    robotwin_root: str | Path,
    candidate_bank_path: str | Path,
    formal_lock_path: str | Path,
) -> dict[str, Any]:
    """CPU-only audit of the immutable inputs consumed by all six cells."""

    root = Path(robotwin_root).expanduser().resolve()
    _require(root.is_dir(), f"RoboTwin root is missing: {root}")
    candidate, lock, candidate_identity, lock_identity = _parent_artifacts(
        candidate_bank_path, formal_lock_path
    )
    sources = _source_artifacts(root)
    evaluator = sources["stock_evaluator"]
    _require(
        evaluator["size_bytes"] == candidate["evaluator_source_size_bytes"],
        "candidate-bank evaluator source size differs from vendored evaluator",
    )
    _require(
        evaluator["sha256"] == candidate["evaluator_source_sha256"],
        "candidate-bank evaluator source SHA differs from vendored evaluator",
    )
    configs = {
        task_config: task_config_artifacts(root, task_config)
        for task_config in TASK_CONFIGS
    }
    return {
        "status": "PASS",
        "gpu_started": False,
        "candidate_seed_bank": candidate_identity,
        "candidate_seed_bank_id": candidate["simulator_seed_bank_id"],
        "candidate_members_sha256": candidate["members_sha256"],
        "candidate_member_count": len(candidate["members"]),
        "formal_protocol_lock": lock_identity,
        "selected_policy_regime": lock["selected_policy_regime"],
        "source_artifacts": sources,
        "task_config_artifacts": configs,
    }


def ordered_seed_instruction_payload(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": int(episode["episode_index"]),
            "simulator_seed": int(episode["simulator_seed"]),
            "instruction": str(episode["instruction"]),
            "instruction_sha256": str(episode["instruction_sha256"]),
        }
        for episode in episodes
    ]


def validate_runtime_binding_payload(
    value: Any,
    *,
    verify_files: bool,
) -> dict[str, Any]:
    """Normalize the exact CUDA/SAPIEN PCI binding recorded by one cell."""

    _require(isinstance(value, Mapping), "runtime_binding must be an object")
    raw_gpu = value.get("physical_gpu_index")
    _require(
        isinstance(raw_gpu, int) and not isinstance(raw_gpu, bool) and raw_gpu >= 0,
        "runtime_binding physical GPU index is invalid",
    )
    try:
        pci = canonical_nvidia_pci_address(str(value.get("pci_bus_id", "")))
    except Exception as exc:
        raise FormalEpisodeProtocolError(
            "runtime_binding PCI address is invalid"
        ) from exc
    _require(
        value.get("render_device_alias") == f"pci:{pci}",
        "runtime_binding render alias differs from PCI address",
    )

    def artifact(raw: Any, label: str) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            return (
                _validate_file_identity(raw, label)
                if verify_files
                else {
                    "kind": str(raw.get("kind", "")),
                    "path": str(Path(str(raw.get("path", ""))).expanduser().resolve()),
                    "size_bytes": raw.get("size_bytes"),
                    "sha256": str(raw.get("sha256", "")),
                }
            )
        _require(isinstance(raw, str) and raw, f"{label} path is missing")
        identity = stable_file_identity(raw)
        return identity

    vulkan = artifact(value.get("vulkan_icd"), "runtime_binding.vulkan_icd")
    egl = artifact(value.get("egl_vendor"), "runtime_binding.egl_vendor")
    sapien = value.get("sapien")
    _require(isinstance(sapien, Mapping), "runtime_binding lacks SAPIEN preflight")
    try:
        sapien_pci = canonical_nvidia_pci_address(str(sapien.get("pci_bus_id", "")))
    except Exception as exc:
        raise FormalEpisodeProtocolError(
            "runtime_binding SAPIEN PCI address is invalid"
        ) from exc
    _require(sapien_pci == pci, "runtime_binding SAPIEN resolved a different PCI GPU")
    _require(
        sapien.get("logical_cuda_id") == 0,
        "runtime_binding SAPIEN device is not logical CUDA device zero",
    )
    _require(
        sapien.get("can_render") is True,
        "runtime_binding SAPIEN device cannot render",
    )
    return {
        "schema": "robotwin.policy_content_adapter.exact_cell_runtime_binding",
        "schema_version": 1,
        "status": "PASS",
        "physical_gpu_index": int(raw_gpu),
        "pci_bus_id": pci,
        "render_device_alias": f"pci:{pci}",
        "gpu_name": str(value.get("gpu_name", "")),
        "driver_version": str(value.get("driver_version", "")),
        "vulkan_icd": vulkan,
        "egl_vendor": egl,
        "sapien": {
            "version": str(sapien.get("version", "")),
            "device_name": str(sapien.get("device_name", "")),
            "logical_cuda_id": 0,
            "pci_bus_id": pci,
            "can_render": True,
        },
    }


def _cell_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"cell_id", "cell_payload_sha256"}
    return {key: value[key] for key in value if key not in excluded}


def _bank_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"realization_bank_id", "bank_payload_sha256"}
    return {key: value[key] for key in value if key not in excluded}


def _validate_attempt_prefix(
    attempts: Sequence[Mapping[str, Any]],
    *,
    candidate_members: Sequence[int],
    accepted_episode_seeds: Sequence[int],
) -> None:
    _require(bool(attempts), "realization attempts must be recorded")
    _require(
        len(attempts) <= len(candidate_members),
        "attempt log exceeds the explicit candidate pool",
    )
    observed_accepted: list[int] = []
    for index, attempt in enumerate(attempts):
        _require(isinstance(attempt, Mapping), f"attempt {index} is not an object")
        _require(
            attempt.get("candidate_index") == index,
            f"attempt {index} candidate index is not contiguous",
        )
        _require(
            attempt.get("simulator_seed") == int(candidate_members[index]),
            f"attempt {index} is not the corresponding candidate-bank member",
        )
        for flag in (
            "expert_setup_ok",
            "expert_play_once_ok",
            "expert_plan_success",
            "expert_check_evaluated",
            "expert_check_success",
            "accepted",
        ):
            _require(
                isinstance(attempt.get(flag), bool),
                f"attempt {index} {flag} is not boolean",
            )
        accepted = bool(attempt["accepted"])
        rejection = attempt.get("rejection")
        if accepted:
            _require(
                all(
                    bool(attempt[field])
                    for field in (
                        "expert_setup_ok",
                        "expert_play_once_ok",
                        "expert_plan_success",
                        "expert_check_evaluated",
                        "expert_check_success",
                    )
                ),
                f"attempt {index} was accepted without a successful stock expert check",
            )
            _require(rejection is None, f"accepted attempt {index} claims a rejection")
            _require(
                attempt.get("accepted_episode_index") == len(observed_accepted),
                f"attempt {index} accepted episode index differs",
            )
            observed_accepted.append(int(attempt["simulator_seed"]))
        else:
            _require(
                attempt.get("accepted_episode_index") is None,
                f"rejected attempt {index} claims an accepted episode index",
            )
            _require(
                isinstance(rejection, Mapping),
                f"rejected attempt {index} lacks a reason",
            )
            reason = str(rejection.get("type", ""))
            _require(
                reason in {"UnStableError", "plan_success_false", "check_success_false"},
                f"attempt {index} records a non-permitted rejection type {reason!r}",
            )
            if reason == "UnStableError":
                _require(
                    not accepted,
                    f"unstable attempt {index} cannot be accepted",
                )
            elif reason == "plan_success_false":
                _require(
                    attempt["expert_play_once_ok"]
                    and not attempt["expert_plan_success"]
                    and not attempt["expert_check_evaluated"],
                    f"attempt {index} plan-failure flags differ from stock short-circuit",
                )
            else:
                _require(
                    attempt["expert_play_once_ok"]
                    and attempt["expert_plan_success"]
                    and attempt["expert_check_evaluated"]
                    and not attempt["expert_check_success"],
                    f"attempt {index} check-failure flags differ",
                )
    _require(
        observed_accepted == [int(seed) for seed in accepted_episode_seeds],
        "accepted attempts do not map one-to-one to realized episodes",
    )
    _require(
        len(observed_accepted) == EPISODES_PER_CELL,
        "attempt log does not prove exactly 100 accepted seeds",
    )
    _require(
        bool(attempts[-1]["accepted"]),
        "attempt scan continued past the 100th accepted candidate",
    )


def build_realization_cell_manifest(
    *,
    robotwin_root: str | Path,
    task: str,
    task_config: str,
    instruction_type: str,
    candidate_bank_path: str | Path,
    formal_lock_path: str | Path,
    episodes: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(task in TASKS, f"unsupported formal task {task!r}")
    _require(task_config in TASK_CONFIGS, f"unsupported task_config {task_config!r}")
    _require(instruction_type == "unseen", "formal instruction_type must be unseen")
    _require(len(episodes) == EPISODES_PER_CELL, "realization must contain exactly 100 episodes")
    candidate, lock, candidate_identity, lock_identity = _parent_artifacts(
        candidate_bank_path, formal_lock_path
    )
    normalized_episodes = [_jsonable(dict(item)) for item in episodes]
    normalized_attempts = [_jsonable(dict(item)) for item in attempts]
    normalized_runtime = validate_runtime_binding_payload(
        runtime_binding, verify_files=True
    )
    indices = [item.get("episode_index") for item in normalized_episodes]
    seeds = [item.get("simulator_seed") for item in normalized_episodes]
    _require(indices == list(range(EPISODES_PER_CELL)), "episode indices are not exactly 0..99")
    _require(all(isinstance(seed, int) for seed in seeds), "episode seeds must be integers")
    _require(len(set(seeds)) == len(seeds), "realized episode seeds contain duplicates")
    candidate_members = list(candidate["members"])
    candidate_positions = {seed: index for index, seed in enumerate(candidate_members)}
    _require(all(seed in candidate_positions for seed in seeds), "episode seed is outside candidate bank")
    _require(
        [candidate_positions[seed] for seed in seeds]
        == sorted(candidate_positions[seed] for seed in seeds),
        "realized seeds are not in candidate-bank order",
    )
    for index, episode in enumerate(normalized_episodes):
        instruction = episode.get("instruction")
        _require(isinstance(instruction, str) and instruction, f"episode {index} instruction is empty")
        instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        _require(
            episode.get("instruction_sha256") == instruction_sha,
            f"episode {index} instruction SHA differs",
        )
        info_sha = str(episode.get("expert_episode_info_sha256", ""))
        _require(
            len(info_sha) == 64
            and all(character in "0123456789abcdef" for character in info_sha),
            f"episode {index} expert info SHA is invalid",
        )
        _validate_instruction_proof(
            episode,
            candidate_seed_bank_id=candidate["simulator_seed_bank_id"],
            task=task,
            task_config=task_config,
            instruction_type=instruction_type,
            episode_index=index,
        )
    _require(bool(normalized_attempts), "realization attempts must be recorded")
    _validate_attempt_prefix(
        normalized_attempts,
        candidate_members=candidate_members,
        accepted_episode_seeds=seeds,
    )
    sequence_payload = ordered_seed_instruction_payload(normalized_episodes)
    payload: dict[str, Any] = {
        "schema": CELL_SCHEMA,
        "schema_version": CELL_SCHEMA_VERSION,
        "status": "PASS",
        "purpose": "policy_independent_exact_final_test_realization",
        "task": task,
        "task_config": task_config,
        "domain": TASK_CONFIG_TO_DOMAIN[task_config],
        "instruction_type": instruction_type,
        "episodes_per_cell": EPISODES_PER_CELL,
        "candidate_seed_bank": candidate_identity,
        "candidate_seed_bank_id": candidate["simulator_seed_bank_id"],
        "candidate_members_sha256": candidate["members_sha256"],
        "formal_protocol_lock": lock_identity,
        "selected_policy_regime": lock["selected_policy_regime"],
        "task_config_artifacts": task_config_artifacts(robotwin_root, task_config),
        "source_artifacts": _source_artifacts(robotwin_root),
        "runtime_binding": normalized_runtime,
        "selection_semantics": SELECTION_SEMANTICS,
        "attempt_count": len(normalized_attempts),
        "attempts": normalized_attempts,
        "episodes": normalized_episodes,
        "ordered_seed_instruction_sha256": canonical_sha256(sequence_payload),
    }
    digest = canonical_sha256(_cell_identity_payload(payload))
    payload["cell_payload_sha256"] = digest
    payload["cell_id"] = CELL_ID_PREFIX + digest
    return validate_realization_cell_payload(payload, verify_files=True)


def validate_realization_cell_payload(
    value: Any,
    *,
    verify_files: bool,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "realization cell must be an object")
    payload = dict(value)
    _require(payload.get("schema") == CELL_SCHEMA, "realization cell schema changed")
    _require(payload.get("schema_version") == CELL_SCHEMA_VERSION, "realization cell version changed")
    _require(payload.get("status") == "PASS", "realization cell is not PASS")
    _require(payload.get("task") in TASKS, "realization cell task is invalid")
    _require(payload.get("task_config") in TASK_CONFIGS, "realization cell config is invalid")
    _require(
        payload.get("domain") == TASK_CONFIG_TO_DOMAIN[payload["task_config"]],
        "realization cell task_config/domain differ",
    )
    _require(payload.get("instruction_type") == "unseen", "instruction type differs")
    _require(payload.get("episodes_per_cell") == EPISODES_PER_CELL, "episode count differs")
    _require(
        payload.get("selection_semantics") == SELECTION_SEMANTICS,
        "realization selection semantics differ",
    )
    digest = canonical_sha256(_cell_identity_payload(payload))
    _require(payload.get("cell_payload_sha256") == digest, "realization cell payload SHA differs")
    _require(payload.get("cell_id") == CELL_ID_PREFIX + digest, "realization cell id differs")
    episodes = payload.get("episodes")
    attempts = payload.get("attempts")
    _require(isinstance(episodes, list) and len(episodes) == EPISODES_PER_CELL, "cell episodes differ")
    _require(isinstance(attempts, list) and attempts, "cell attempts are missing")
    _require(
        payload.get("attempt_count") == len(attempts),
        "cell attempt_count differs from attempts",
    )
    indices = [episode.get("episode_index") for episode in episodes if isinstance(episode, Mapping)]
    seeds = [episode.get("simulator_seed") for episode in episodes if isinstance(episode, Mapping)]
    _require(indices == list(range(EPISODES_PER_CELL)), "cell episode order differs")
    _require(len(seeds) == EPISODES_PER_CELL and len(set(seeds)) == EPISODES_PER_CELL, "cell seeds differ")
    for index, episode in enumerate(episodes):
        _require(isinstance(episode, Mapping), f"episode {index} is not an object")
        instruction = str(episode.get("instruction", ""))
        _require(bool(instruction), f"episode {index} instruction is empty")
        _require(
            episode.get("instruction_sha256")
            == hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            f"episode {index} instruction SHA differs",
        )
        info_sha = str(episode.get("expert_episode_info_sha256", ""))
        _require(
            len(info_sha) == 64
            and all(character in "0123456789abcdef" for character in info_sha),
            f"episode {index} expert info SHA is invalid",
        )
        _validate_instruction_proof(
            episode,
            candidate_seed_bank_id=str(payload.get("candidate_seed_bank_id", "")),
            task=str(payload["task"]),
            task_config=str(payload["task_config"]),
            instruction_type=str(payload["instruction_type"]),
            episode_index=index,
        )
    _require(
        payload.get("ordered_seed_instruction_sha256")
        == canonical_sha256(ordered_seed_instruction_payload(episodes)),
        "ordered seed/instruction SHA differs",
    )
    if verify_files:
        candidate_identity = _validate_file_identity(
            payload.get("candidate_seed_bank"), "candidate_seed_bank"
        )
        lock_identity = _validate_file_identity(
            payload.get("formal_protocol_lock"), "formal_protocol_lock"
        )
        candidate, lock, expected_candidate, expected_lock = _parent_artifacts(
            candidate_identity["path"], lock_identity["path"]
        )
        _require(candidate_identity == expected_candidate, "candidate identity differs")
        _require(lock_identity == expected_lock, "formal lock identity differs")
        _require(
            payload.get("candidate_seed_bank_id") == candidate["simulator_seed_bank_id"],
            "candidate seed-bank id differs",
        )
        _require(
            payload.get("candidate_members_sha256") == candidate["members_sha256"],
            "candidate member SHA differs",
        )
        _require(
            payload.get("selected_policy_regime") == lock["selected_policy_regime"],
            "selected policy regime differs",
        )
        candidate_positions = {seed: i for i, seed in enumerate(candidate["members"])}
        _require(all(seed in candidate_positions for seed in seeds), "cell seed left candidate pool")
        _require(
            [candidate_positions[seed] for seed in seeds]
            == sorted(candidate_positions[seed] for seed in seeds),
            "cell seed ordering differs from candidate pool",
        )
        _validate_attempt_prefix(
            attempts,
            candidate_members=candidate["members"],
            accepted_episode_seeds=seeds,
        )
        source_group = payload.get("source_artifacts")
        _require(isinstance(source_group, Mapping), "source_artifacts is missing")
        stock = source_group.get("stock_evaluator")
        _require(isinstance(stock, Mapping), "stock evaluator identity is missing")
        stock_path = Path(str(stock.get("path", ""))).expanduser().resolve()
        _require(
            stock_path.name == "eval_policy.py" and stock_path.parent.name == "script",
            "stock evaluator path is not the vendored RoboTwin evaluator",
        )
        robotwin_root = stock_path.parent.parent
        expected_sources = _source_artifacts(robotwin_root)
        _require(
            set(source_group) == set(expected_sources),
            "selector source-artifact set differs",
        )
        for name, expected_identity in expected_sources.items():
            actual_identity = _validate_file_identity(
                source_group[name], f"source_artifacts.{name}"
            )
            _require(
                actual_identity == expected_identity,
                f"source_artifacts.{name} differs from the selector-owned source",
            )
        _require(
            expected_sources["stock_evaluator"]["sha256"]
            == candidate["evaluator_source_sha256"]
            and expected_sources["stock_evaluator"]["size_bytes"]
            == candidate["evaluator_source_size_bytes"],
            "cell vendored evaluator differs from the checkpoint-bound candidate bank",
        )
        config_group = payload.get("task_config_artifacts")
        _require(isinstance(config_group, Mapping), "task_config_artifacts is missing")
        expected_configs = task_config_artifacts(robotwin_root, payload["task_config"])
        _require(
            set(config_group) == set(expected_configs),
            "task-config artifact set differs",
        )
        for name, expected_identity in expected_configs.items():
            actual_identity = _validate_file_identity(
                config_group[name], f"task_config_artifacts.{name}"
            )
            _require(
                actual_identity == expected_identity,
                f"task_config_artifacts.{name} differs from vendored RoboTwin",
            )
        validate_runtime_binding_payload(
            payload.get("runtime_binding"), verify_files=True
        )
    else:
        validate_runtime_binding_payload(
            payload.get("runtime_binding"), verify_files=False
        )
    return payload


def validate_realization_cell(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, resolved = _load_json(path, "realization cell")
    return validate_realization_cell_payload(payload, verify_files=True), resolved


def finalize_realization_bank(
    *,
    candidate_bank_path: str | Path,
    formal_lock_path: str | Path,
    cell_paths: Sequence[str | Path],
) -> dict[str, Any]:
    candidate, lock, candidate_identity, lock_identity = _parent_artifacts(
        candidate_bank_path, formal_lock_path
    )
    _require(len(cell_paths) == 6, "realization bank requires exactly six cells")
    cells: list[dict[str, Any]] = []
    observed: set[tuple[str, str]] = set()
    shared_sources: dict[str, Any] | None = None
    config_artifacts: dict[str, Any] = {}
    for raw_path in cell_paths:
        cell, path = validate_realization_cell(raw_path)
        key = (str(cell["task"]), str(cell["task_config"]))
        _require(key not in observed, f"duplicate realization cell: {key}")
        observed.add(key)
        _require(
            cell["candidate_seed_bank_id"] == candidate["simulator_seed_bank_id"],
            "realization cells bind a different candidate bank",
        )
        _require(
            cell["candidate_seed_bank"] == candidate_identity,
            "realization cells bind a different candidate-bank artifact",
        )
        _require(
            cell["formal_protocol_lock"] == lock_identity,
            "realization cells bind a different formal lock",
        )
        if shared_sources is None:
            shared_sources = dict(cell["source_artifacts"])
        else:
            _require(
                cell["source_artifacts"] == shared_sources,
                "realization cells used different evaluator/selector sources",
            )
        previous_config = config_artifacts.setdefault(
            cell["task_config"], cell["task_config_artifacts"]
        )
        _require(
            previous_config == cell["task_config_artifacts"],
            f"realization cells used different {cell['task_config']} artifacts",
        )
        cells.append(
            {
                "task": cell["task"],
                "task_config": cell["task_config"],
                "domain": cell["domain"],
                "cell_id": cell["cell_id"],
                "ordered_seed_instruction_sha256": cell[
                    "ordered_seed_instruction_sha256"
                ],
                "runtime_binding_sha256": canonical_sha256(
                    cell["runtime_binding"]
                ),
                "physical_gpu_index": cell["runtime_binding"][
                    "physical_gpu_index"
                ],
                "pci_bus_id": cell["runtime_binding"]["pci_bus_id"],
                "manifest": stable_file_identity(path),
            }
        )
    expected = {(task, config) for task in TASKS for config in TASK_CONFIGS}
    _require(observed == expected, "realization bank task/domain matrix differs")
    _require(shared_sources is not None, "realization bank lacks source artifacts")
    _require(
        set(config_artifacts) == set(TASK_CONFIGS),
        "realization bank lacks one task-config artifact set per domain",
    )
    cells.sort(key=lambda row: (TASKS.index(row["task"]), TASK_CONFIGS.index(row["task_config"])))
    payload: dict[str, Any] = {
        "schema": BANK_SCHEMA,
        "schema_version": BANK_SCHEMA_VERSION,
        "status": "PASS",
        "purpose": "policy_independent_exact_final_test_realization",
        "candidate_seed_bank": candidate_identity,
        "candidate_seed_bank_id": candidate["simulator_seed_bank_id"],
        "candidate_members_sha256": candidate["members_sha256"],
        "formal_protocol_lock": lock_identity,
        "selected_policy_regime": lock["selected_policy_regime"],
        "source_artifacts": shared_sources,
        "task_config_artifacts": {
            task_config: config_artifacts[task_config]
            for task_config in TASK_CONFIGS
        },
        "tasks": list(TASKS),
        "task_configs": list(TASK_CONFIGS),
        "episodes_per_cell": EPISODES_PER_CELL,
        "cell_count": 6,
        "cells": cells,
    }
    digest = canonical_sha256(_bank_identity_payload(payload))
    payload["bank_payload_sha256"] = digest
    payload["realization_bank_id"] = BANK_ID_PREFIX + digest
    return validate_realization_bank_payload(payload, verify_files=True)


def validate_realization_bank_payload(
    value: Any,
    *,
    verify_files: bool,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "realization bank must be an object")
    payload = dict(value)
    _require(payload.get("schema") == BANK_SCHEMA, "realization bank schema changed")
    _require(payload.get("schema_version") == BANK_SCHEMA_VERSION, "realization bank version changed")
    _require(payload.get("status") == "PASS", "realization bank is not PASS")
    _require(payload.get("tasks") == list(TASKS), "realization bank tasks differ")
    _require(payload.get("task_configs") == list(TASK_CONFIGS), "realization bank configs differ")
    _require(payload.get("episodes_per_cell") == EPISODES_PER_CELL, "bank episode count differs")
    _require(payload.get("cell_count") == 6, "realization bank cell_count differs")
    digest = canonical_sha256(_bank_identity_payload(payload))
    _require(payload.get("bank_payload_sha256") == digest, "realization bank payload SHA differs")
    _require(payload.get("realization_bank_id") == BANK_ID_PREFIX + digest, "realization bank id differs")
    cells = payload.get("cells")
    _require(isinstance(cells, list) and len(cells) == 6, "realization bank cells differ")
    expected_order = [(task, config) for task in TASKS for config in TASK_CONFIGS]
    observed_order = [(cell.get("task"), cell.get("task_config")) for cell in cells if isinstance(cell, Mapping)]
    _require(observed_order == expected_order, "realization bank cell order differs")
    if verify_files:
        candidate_identity = _validate_file_identity(payload.get("candidate_seed_bank"), "candidate_seed_bank")
        lock_identity = _validate_file_identity(payload.get("formal_protocol_lock"), "formal_protocol_lock")
        candidate, lock, _, _ = _parent_artifacts(candidate_identity["path"], lock_identity["path"])
        _require(payload.get("candidate_seed_bank_id") == candidate["simulator_seed_bank_id"], "bank candidate id differs")
        _require(
            payload.get("candidate_members_sha256") == candidate["members_sha256"],
            "bank candidate members SHA differs",
        )
        _require(payload.get("selected_policy_regime") == lock["selected_policy_regime"], "bank regime differs")
        bank_sources = payload.get("source_artifacts")
        bank_configs = payload.get("task_config_artifacts")
        _require(isinstance(bank_sources, Mapping), "bank source artifacts are missing")
        _require(
            isinstance(bank_configs, Mapping)
            and set(bank_configs) == set(TASK_CONFIGS),
            "bank task-config artifacts differ",
        )
        for index, descriptor in enumerate(cells):
            _require(isinstance(descriptor, Mapping), f"cell descriptor {index} is invalid")
            identity = _validate_file_identity(descriptor.get("manifest"), f"cells[{index}].manifest")
            cell, _ = validate_realization_cell(identity["path"])
            for field in (
                "task",
                "task_config",
                "domain",
                "cell_id",
                "ordered_seed_instruction_sha256",
            ):
                _require(descriptor.get(field) == cell.get(field), f"cell descriptor {index} {field} differs")
            _require(
                descriptor.get("runtime_binding_sha256")
                == canonical_sha256(cell["runtime_binding"]),
                f"cell descriptor {index} runtime binding SHA differs",
            )
            _require(
                descriptor.get("physical_gpu_index")
                == cell["runtime_binding"]["physical_gpu_index"]
                and descriptor.get("pci_bus_id")
                == cell["runtime_binding"]["pci_bus_id"],
                f"cell descriptor {index} PCI binding differs",
            )
            _require(
                cell["candidate_seed_bank"] == candidate_identity
                and cell["formal_protocol_lock"] == lock_identity,
                f"cell descriptor {index} parent artifact differs",
            )
            _require(
                cell["source_artifacts"] == bank_sources,
                f"cell descriptor {index} source artifacts differ from bank",
            )
            _require(
                cell["task_config_artifacts"]
                == bank_configs[cell["task_config"]],
                f"cell descriptor {index} task-config artifacts differ from bank",
            )
    return payload


def validate_realization_bank(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, resolved = _load_json(path, "realization bank")
    return validate_realization_bank_payload(payload, verify_files=True), resolved


def select_realization_cell(
    realization_bank_path: str | Path,
    *,
    task: str,
    task_config: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    bank, bank_path = validate_realization_bank(realization_bank_path)
    matches = [
        descriptor
        for descriptor in bank["cells"]
        if descriptor["task"] == task and descriptor["task_config"] == task_config
    ]
    _require(len(matches) == 1, f"realization bank lacks one exact cell for {task}/{task_config}")
    cell, _ = validate_realization_cell(matches[0]["manifest"]["path"])
    return bank, cell, bank_path


def validate_replay_trace(
    path: str | Path,
    *,
    realization_bank_path: str | Path,
    task: str,
    task_config: str,
) -> tuple[dict[str, Any], Path]:
    """Validate one exact-list replay trace against its realized cell."""

    trace, resolved = _load_json(path, "formal episode replay trace")
    _require(trace.get("schema") == TRACE_SCHEMA, "formal replay trace schema changed")
    _require(
        trace.get("schema_version") == TRACE_SCHEMA_VERSION,
        "formal replay trace version changed",
    )
    _require(trace.get("status") == "PASS", "formal replay trace is not PASS")
    bank, cell, bank_path = select_realization_cell(
        realization_bank_path,
        task=task,
        task_config=task_config,
    )
    _require(trace.get("realization_bank_id") == bank["realization_bank_id"], "trace realization bank differs")
    _require(trace.get("cell_id") == cell["cell_id"], "trace realization cell differs")
    _require(trace.get("task") == task, "trace task differs")
    _require(trace.get("task_config") == task_config, "trace task_config differs")
    _require(trace.get("domain") == TASK_CONFIG_TO_DOMAIN[task_config], "trace domain differs")
    _require(trace.get("no_seed_replacement") is True, "trace does not prove no seed replacement")
    bank_identity = _validate_file_identity(trace.get("realization_bank"), "trace.realization_bank")
    _require(bank_identity == stable_file_identity(bank_path), "trace realization bank identity differs")
    episodes = trace.get("episodes")
    _require(isinstance(episodes, list) and len(episodes) == EPISODES_PER_CELL, "trace must contain 100 episodes")
    expected = ordered_seed_instruction_payload(cell["episodes"])
    observed = ordered_seed_instruction_payload(episodes)
    _require(observed == expected, "trace ordered seed/instruction entries differ from realization")
    _require(
        trace.get("ordered_seed_instruction_sha256")
        == cell["ordered_seed_instruction_sha256"]
        == canonical_sha256(observed),
        "trace ordered seed/instruction SHA differs",
    )
    successes = 0
    for index, episode in enumerate(episodes):
        _require(isinstance(episode.get("success"), bool), f"trace episode {index} success is not boolean")
        successes += int(episode["success"])
    _require(trace.get("episode_count") == EPISODES_PER_CELL, "trace episode_count differs")
    _require(trace.get("successes") == successes, "trace successes count differs")
    rate = trace.get("success_rate")
    _require(isinstance(rate, (int, float)) and math.isfinite(float(rate)), "trace success_rate is invalid")
    _require(math.isclose(float(rate), successes / EPISODES_PER_CELL), "trace success_rate differs")
    _require(
        trace.get("ordered_seed_instruction_success_sha256") == canonical_sha256(episodes),
        "trace ordered outcome SHA differs",
    )
    validate_runtime_binding_payload(
        trace.get("pinned_runtime_binding"),
        verify_files=True,
    )
    return trace, resolved


def _instruction_context(
    *,
    candidate_seed_bank_id: str,
    task: str,
    task_config: str,
    simulator_seed: int,
    instruction_type: str,
) -> dict[str, Any]:
    return {
        "schema": INSTRUCTION_CHOICE_SCHEMA,
        "candidate_seed_bank_id": candidate_seed_bank_id,
        "task": task,
        "task_config": task_config,
        "simulator_seed": int(simulator_seed),
        "instruction_type": instruction_type,
    }


def _deterministic_instruction_choice(
    module: Any,
    *,
    candidate_seed_bank_id: str,
    task: str,
    task_config: str,
    simulator_seed: int,
    episode_info: Mapping[str, Any],
    instruction_type: str,
    test_num: int,
) -> tuple[str, dict[str, Any]]:
    """Freeze one instruction while retaining proof of the legal option set.

    RoboTwin's generator uses the process-global :mod:`random` module for both
    template order and object synonyms.  We scope that state to a hash of the
    locked cell/seed and restore it afterwards.  Selection from the resulting
    legal descriptions is a second explicit SHA-based rule; NumPy's ambient
    RNG state is never consulted.
    """

    context = _instruction_context(
        candidate_seed_bank_id=candidate_seed_bank_id,
        task=task,
        task_config=task_config,
        simulator_seed=simulator_seed,
        instruction_type=instruction_type,
    )
    context_sha = canonical_sha256(context)
    generation_seed = int(context_sha[:16], 16)
    generator = module.generate_episode_descriptions
    generator_random = getattr(generator, "__globals__", {}).get("random")
    _require(
        generator_random is not None
        and callable(getattr(generator_random, "getstate", None))
        and callable(getattr(generator_random, "setstate", None))
        and callable(getattr(generator_random, "seed", None)),
        "stock instruction generator does not expose its Python RNG",
    )
    state = generator_random.getstate()
    try:
        generator_random.seed(generation_seed, version=2)
        descriptions = generator(task, [dict(episode_info)], test_num)
    finally:
        generator_random.setstate(state)
    _require(isinstance(descriptions, list) and descriptions, "instruction generator returned no descriptions")
    options = descriptions[0].get(instruction_type)
    _require(isinstance(options, (list, tuple)) and options, "instruction generator returned no unseen instructions")
    normalized_options = [str(option) for option in options]
    _require(
        all(option for option in normalized_options),
        "instruction generator returned an empty candidate description",
    )
    candidates_sha = canonical_sha256(normalized_options)
    choice_sha = canonical_sha256(
        {
            **context,
            "instruction_candidates_sha256": candidates_sha,
            "instruction_candidate_count": len(normalized_options),
        }
    )
    choice_index = int(choice_sha, 16) % len(normalized_options)
    instruction = normalized_options[choice_index]
    _require(bool(instruction), "instruction generator selected an empty instruction")
    return instruction, {
        "instruction_choice_schema": INSTRUCTION_CHOICE_SCHEMA,
        "instruction_generation_context_sha256": context_sha,
        "instruction_generation_seed": generation_seed,
        "instruction_candidates": normalized_options,
        "instruction_candidate_count": len(normalized_options),
        "instruction_candidates_sha256": candidates_sha,
        "instruction_choice_sha256": choice_sha,
        "instruction_choice_index": choice_index,
    }


def _validate_instruction_proof(
    episode: Mapping[str, Any],
    *,
    candidate_seed_bank_id: str,
    task: str,
    task_config: str,
    instruction_type: str,
    episode_index: int,
) -> None:
    seed = episode.get("simulator_seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        f"episode {episode_index} simulator seed is invalid",
    )
    context = _instruction_context(
        candidate_seed_bank_id=candidate_seed_bank_id,
        task=task,
        task_config=task_config,
        simulator_seed=int(seed),
        instruction_type=instruction_type,
    )
    context_sha = canonical_sha256(context)
    _require(
        episode.get("instruction_choice_schema") == INSTRUCTION_CHOICE_SCHEMA,
        f"episode {episode_index} instruction choice schema differs",
    )
    _require(
        episode.get("instruction_generation_context_sha256") == context_sha,
        f"episode {episode_index} instruction generation context differs",
    )
    _require(
        episode.get("instruction_generation_seed") == int(context_sha[:16], 16),
        f"episode {episode_index} instruction generation seed differs",
    )
    options = episode.get("instruction_candidates")
    _require(
        isinstance(options, list) and bool(options),
        f"episode {episode_index} legal instruction candidates are missing",
    )
    _require(
        all(isinstance(option, str) and option for option in options),
        f"episode {episode_index} legal instruction candidates are invalid",
    )
    _require(
        episode.get("instruction_candidate_count") == len(options),
        f"episode {episode_index} instruction candidate count differs",
    )
    candidates_sha = canonical_sha256(options)
    _require(
        episode.get("instruction_candidates_sha256") == candidates_sha,
        f"episode {episode_index} instruction candidate SHA differs",
    )
    choice_sha = canonical_sha256(
        {
            **context,
            "instruction_candidates_sha256": candidates_sha,
            "instruction_candidate_count": len(options),
        }
    )
    _require(
        episode.get("instruction_choice_sha256") == choice_sha,
        f"episode {episode_index} instruction choice SHA differs",
    )
    choice_index = int(choice_sha, 16) % len(options)
    _require(
        episode.get("instruction_choice_index") == choice_index,
        f"episode {episode_index} instruction choice index differs",
    )
    _require(
        episode.get("instruction") == options[choice_index],
        f"episode {episode_index} instruction is not the deterministic legal candidate",
    )


def _close_env(
    task_env: Any,
    *,
    suppress_errors: bool = False,
    **kwargs: Any,
) -> None:
    try:
        task_env.close_env(**kwargs)
    except Exception:
        if not suppress_errors:
            raise


def scan_expert_candidates(
    module: Any,
    *,
    task_name: str,
    task_config: str,
    task_env: Any,
    args: dict[str, Any],
    candidate_bank: Mapping[str, Any],
    instruction_type: str,
    test_num: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the policy-independent stock expert filter for exactly one cell."""

    _require(test_num == EPISODES_PER_CELL, "realization test_num must be exactly 100")
    _require(task_name in TASKS, f"unsupported formal task {task_name!r}")
    _require(task_config in TASK_CONFIGS, f"unsupported task_config {task_config!r}")
    _require(instruction_type == "unseen", "realization instruction type must be unseen")
    members = candidate_bank.get("members")
    _require(isinstance(members, list), "candidate bank lacks explicit members")
    unstable_error = getattr(module, "UnStableError", None)
    _require(
        isinstance(unstable_error, type) and issubclass(unstable_error, Exception),
        "stock evaluator lacks RoboTwin UnStableError",
    )
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for candidate_index, raw_seed in enumerate(members):
        candidate_seed = int(raw_seed)
        episode_index = len(accepted)
        attempt: dict[str, Any] = {
            "candidate_index": candidate_index,
            "simulator_seed": candidate_seed,
            "expert_setup_ok": False,
            "expert_play_once_ok": False,
            "expert_plan_success": False,
            "expert_check_evaluated": False,
            "expert_check_success": False,
            "accepted": False,
            "accepted_episode_index": None,
            "rejection": None,
        }
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        episode_info: Any = None
        try:
            try:
                task_env.setup_demo(
                    now_ep_num=episode_index,
                    seed=candidate_seed,
                    is_test=True,
                    **args,
                )
                attempt["expert_setup_ok"] = True
                episode_info = task_env.play_once()
                attempt["expert_play_once_ok"] = True
                attempt["expert_plan_success"] = bool(task_env.plan_success)
                if attempt["expert_plan_success"]:
                    attempt["expert_check_evaluated"] = True
                    attempt["expert_check_success"] = bool(task_env.check_success())
            except unstable_error:
                attempt["rejection"] = {
                    "stage": "expert_environment",
                    "type": "UnStableError",
                }
                _close_env(task_env, suppress_errors=True)
                attempts.append(attempt)
                continue
            except Exception as exc:
                _close_env(task_env, suppress_errors=True)
                raise FormalEpisodeProtocolError(
                    "non-UnStableError during expert seed selection for "
                    f"{task_name}/{task_config} candidate_index={candidate_index} "
                    f"seed={candidate_seed}: {type(exc).__name__}"
                ) from exc
            _close_env(task_env)
        finally:
            args["render_freq"] = render_freq

        if not attempt["expert_plan_success"]:
            attempt["rejection"] = {
                "stage": "expert_validity",
                "type": "plan_success_false",
            }
            attempts.append(attempt)
            continue
        if not attempt["expert_check_success"]:
            attempt["rejection"] = {
                "stage": "expert_validity",
                "type": "check_success_false",
            }
            attempts.append(attempt)
            continue

        _require(
            isinstance(episode_info, Mapping)
            and isinstance(episode_info.get("info"), Mapping),
            f"stock expert returned invalid episode info for seed {candidate_seed}",
        )
        info = _jsonable(episode_info["info"])
        _require(
            isinstance(info, Mapping),
            f"stock expert episode info is not an object for seed {candidate_seed}",
        )
        instruction, proof = _deterministic_instruction_choice(
            module,
            candidate_seed_bank_id=str(candidate_bank["simulator_seed_bank_id"]),
            task=task_name,
            task_config=task_config,
            simulator_seed=candidate_seed,
            episode_info=info,
            instruction_type=instruction_type,
            test_num=test_num,
        )
        accepted.append(
            {
                "episode_index": episode_index,
                "simulator_seed": candidate_seed,
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "expert_episode_info_sha256": canonical_sha256(info),
                **proof,
            }
        )
        attempt["accepted"] = True
        attempt["accepted_episode_index"] = episode_index
        attempts.append(attempt)
        if len(accepted) == EPISODES_PER_CELL:
            break
    _require(
        len(accepted) == EPISODES_PER_CELL,
        "candidate bank exhausted before 100 exact policy-independent episodes "
        f"were realized (accepted={len(accepted)}, members={len(members)})",
    )
    _validate_attempt_prefix(
        attempts,
        candidate_members=members,
        accepted_episode_seeds=[entry["simulator_seed"] for entry in accepted],
    )
    return accepted, attempts


def make_realization_eval_policy(
    module: Any,
    user_args: Mapping[str, Any],
    *,
    runtime_binding: Mapping[str, Any],
) -> Callable[..., tuple[int, int]]:
    candidate_path = Path(str(user_args.get("formal_candidate_seed_bank", ""))).expanduser().resolve()
    lock_path = Path(str(user_args.get("formal_protocol_lock", ""))).expanduser().resolve()
    output_path = Path(str(user_args.get("formal_episode_cell_output", ""))).expanduser().resolve()
    robotwin_root = Path.cwd().resolve()
    _require(not output_path.exists(), f"refusing to overwrite realization cell: {output_path}")
    audit_selector_inputs(
        robotwin_root=robotwin_root,
        candidate_bank_path=candidate_path,
        formal_lock_path=lock_path,
    )
    normalized_runtime = validate_runtime_binding_payload(
        runtime_binding, verify_files=True
    )

    def realize(
        task_name: str,
        task_env: Any,
        args: dict[str, Any],
        model: Any,
        st_seed: int,
        test_num: int = 100,
        video_size: Any = None,
        instruction_type: str | None = None,
        skip_get_obs_within_replan: bool = False,
    ) -> tuple[int, int]:
        del model, video_size, skip_get_obs_within_replan
        candidate, _, _, _ = _parent_artifacts(candidate_path, lock_path)
        _require(test_num == EPISODES_PER_CELL, "realization test_num must be exactly 100")
        _require(st_seed == candidate["candidate_start_seed"], "realization start seed differs from candidate bank")
        task_config = str(args.get("task_config", ""))
        _require(task_name in TASKS and task_config in TASK_CONFIGS, "realization cell is outside formal matrix")
        _require(instruction_type == "unseen", "realization instruction type must be unseen")
        task_env.suc = 0
        task_env.test_num = 0
        accepted, attempts = scan_expert_candidates(
            module,
            task_name=task_name,
            task_config=task_config,
            task_env=task_env,
            args=args,
            candidate_bank=candidate,
            instruction_type=str(instruction_type),
            test_num=test_num,
        )
        payload = build_realization_cell_manifest(
            robotwin_root=robotwin_root,
            task=task_name,
            task_config=task_config,
            instruction_type=str(instruction_type),
            candidate_bank_path=candidate_path,
            formal_lock_path=lock_path,
            episodes=accepted,
            attempts=attempts,
            runtime_binding=normalized_runtime,
        )
        _exclusive_json(output_path, payload)
        return int(accepted[-1]["simulator_seed"]) + 1, 0

    return realize


def _run_policy_episode(
    module: Any,
    *,
    task_env: Any,
    args: Mapping[str, Any],
    model: Any,
    eval_func: Callable[..., Any],
    reset_func: Callable[..., Any],
    video_size: Any,
    skip_get_obs_within_replan: bool,
    episode_index: int,
) -> bool:
    current_video_path: Path | None = None
    ffmpeg: subprocess.Popen[bytes] | None = None
    if task_env.eval_video_path is not None:
        _require(video_size is not None, "formal video logging lacks video_size")
        current_video_path = Path(task_env.eval_video_path) / f"episode{task_env.test_num}.mp4"
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                str(video_size),
                "-framerate",
                "10",
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(current_video_path),
            ],
            stdin=subprocess.PIPE,
        )
        task_env._set_eval_video_ffmpeg(ffmpeg)
    success = False
    try:
        reset_func(model)
        while task_env.take_action_cnt < task_env.step_lim:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())
            observation = task_env.get_obs() if need_obs else None
            eval_func(task_env, model, observation)
            if task_env.eval_success:
                success = True
                break
    finally:
        if task_env.eval_video_path is not None and ffmpeg is not None:
            task_env._del_eval_video_ffmpeg()
    if task_env.eval_video_path is not None:
        _require(
            current_video_path is not None and current_video_path.exists(),
            f"expected formal eval video is missing: {current_video_path}",
        )
        randomized = "randomized" in str(args["task_config"]).lower()
        renamed = Path(task_env.eval_video_path) / (
            f"episode{episode_index}_randomized-{str(randomized).lower()}_"
            f"success-{str(success).lower()}.mp4"
        )
        _require(not renamed.exists(), f"refusing to overwrite formal eval video: {renamed}")
        current_video_path.rename(renamed)
    return success


def make_replay_eval_policy(
    module: Any,
    user_args: Mapping[str, Any],
    *,
    runtime_binding: Mapping[str, Any],
) -> Callable[..., tuple[int, int]]:
    bank_path = Path(str(user_args.get("formal_episode_realization_bank", ""))).expanduser().resolve()
    trace_path = Path(str(user_args.get("formal_episode_trace_output", ""))).expanduser().resolve()
    _require(not trace_path.exists(), f"refusing to overwrite formal episode trace: {trace_path}")
    normalized_runtime = validate_runtime_binding_payload(
        runtime_binding,
        verify_files=True,
    )

    def replay(
        task_name: str,
        task_env: Any,
        args: dict[str, Any],
        model: Any,
        st_seed: int,
        test_num: int = 100,
        video_size: Any = None,
        instruction_type: str | None = None,
        skip_get_obs_within_replan: bool = False,
    ) -> tuple[int, int]:
        task_config = str(args.get("task_config", ""))
        bank, cell, resolved_bank = select_realization_cell(
            bank_path,
            task=task_name,
            task_config=task_config,
        )
        _require(test_num == EPISODES_PER_CELL, "formal replay test_num must be exactly 100")
        _require(instruction_type == "unseen", "formal replay instruction type must be unseen")
        raw_candidate, _ = _load_json(
            cell["candidate_seed_bank"]["path"], "replay candidate seed bank"
        )
        try:
            candidate = validate_seed_bank_descriptor(
                raw_candidate, expected_purpose="final_test"
            )
        except ValueError as exc:
            raise FormalEpisodeProtocolError(
                f"replay candidate seed bank is invalid: {exc}"
            ) from exc
        _require(
            int(st_seed) == int(candidate["candidate_start_seed"]),
            "formal replay start seed differs from checkpoint-bound candidate bank",
        )
        eval_func = module.eval_function_decorator(args["policy_name"], "eval")
        reset_func = module.eval_function_decorator(args["policy_name"], "reset_model")
        task_env.suc = 0
        task_env.test_num = 0
        args["eval_mode"] = True
        results: list[dict[str, Any]] = []
        clear_cache_freq = int(args["clear_cache_freq"])
        for entry in cell["episodes"]:
            episode_index = int(entry["episode_index"])
            seed = int(entry["simulator_seed"])
            instruction = str(entry["instruction"])
            # Exact-list replay is deliberately fail-closed: a setup failure is
            # a cell failure, never permission to substitute another seed.
            try:
                task_env.setup_demo(
                    now_ep_num=episode_index,
                    seed=seed,
                    is_test=True,
                    **args,
                )
            except Exception as exc:
                _close_env(task_env)
                raise FormalEpisodeProtocolError(
                    f"exact replay setup failed without replacement for "
                    f"{task_name}/{task_config} episode={episode_index} seed={seed}: "
                    f"{type(exc).__name__}"
                ) from exc
            try:
                task_env.set_instruction(instruction=instruction)
                success = _run_policy_episode(
                    module,
                    task_env=task_env,
                    args=args,
                    model=model,
                    eval_func=eval_func,
                    reset_func=reset_func,
                    video_size=video_size,
                    skip_get_obs_within_replan=skip_get_obs_within_replan,
                    episode_index=episode_index,
                )
            finally:
                # A close failure is a cell failure.  It is not silently
                # swallowed because later exact seeds could inherit state.
                task_env.close_env(
                    clear_cache=((episode_index + 2) % clear_cache_freq == 0)
                )
            if success:
                task_env.suc += 1
                print("\033[92mSuccess!\033[0m")
            else:
                print("\033[91mFail!\033[0m")
            task_env.test_num += 1
            results.append(
                {
                    "episode_index": episode_index,
                    "simulator_seed": seed,
                    "instruction": instruction,
                    "instruction_sha256": entry["instruction_sha256"],
                    "success": bool(success),
                }
            )
            if getattr(task_env, "render_freq", 0) and getattr(task_env, "viewer", None):
                task_env.viewer.close()
            print(
                f"{task_name} | {args['policy_name']} | {task_config} | "
                f"{args['ckpt_setting']}\nSuccess rate: {task_env.suc}/{task_env.test_num}, "
                f"fixed seed: {seed}\n",
                flush=True,
            )
        sequence_sha = canonical_sha256(ordered_seed_instruction_payload(results))
        _require(
            sequence_sha == cell["ordered_seed_instruction_sha256"],
            "replay ordered seed/instruction sequence differs from realization",
        )
        successes = sum(bool(row["success"]) for row in results)
        trace_payload: dict[str, Any] = {
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "status": "PASS",
            "realization_bank": stable_file_identity(resolved_bank),
            "realization_bank_id": bank["realization_bank_id"],
            "cell_id": cell["cell_id"],
            "task": task_name,
            "task_config": task_config,
            "domain": TASK_CONFIG_TO_DOMAIN[task_config],
            "episodes": results,
            "episode_count": len(results),
            "successes": successes,
            "success_rate": successes / len(results),
            "ordered_seed_instruction_sha256": sequence_sha,
            "ordered_seed_instruction_success_sha256": canonical_sha256(results),
            "no_seed_replacement": True,
            "pinned_runtime_binding": normalized_runtime,
        }
        _exclusive_json(trace_path, trace_payload)
        return int(results[-1]["simulator_seed"]) + 1, successes

    return replay


def install_formal_episode_mode(
    module: Any,
    user_args: Mapping[str, Any],
    *,
    runtime_binding: Mapping[str, Any],
) -> None:
    mode = str(user_args.get("formal_episode_mode", "")).strip().lower()
    _require(mode in {"realize", "replay"}, f"unsupported formal_episode_mode {mode!r}")
    if mode == "realize":
        module.eval_policy = make_realization_eval_policy(
            module,
            user_args,
            runtime_binding=runtime_binding,
        )
    else:
        module.eval_policy = make_replay_eval_policy(
            module,
            user_args,
            runtime_binding=runtime_binding,
        )


# Lightweight policy hooks used only by the policy-independent realization
# pass.  The patched realization eval_policy never calls eval/reset_model.
def get_model(usr_args: Mapping[str, Any]) -> None:
    _require(
        str(usr_args.get("formal_episode_mode", "")) == "realize",
        "formal_episode_protocol.get_model is only valid for realization",
    )
    return None


def eval(task_env: Any, model: Any, observation: Any) -> None:  # pragma: no cover
    raise FormalEpisodeProtocolError("realization must never execute a policy action")


def reset_model(model: Any) -> None:  # pragma: no cover
    raise FormalEpisodeProtocolError("realization must never reset a policy model")


def _stock_task_arguments(
    module: Any,
    *,
    robotwin_root: Path,
    task: str,
    task_config: str,
) -> dict[str, Any]:
    """Construct the same task arguments used by stock ``eval_policy.main``."""

    config_path = robotwin_root / f"task_config/{task_config}.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        args = module.yaml.load(handle.read(), Loader=module.yaml.FullLoader)
    _require(isinstance(args, dict), f"invalid RoboTwin task config: {config_path}")
    args["task_name"] = task
    args["task_config"] = task_config
    args["ckpt_setting"] = "formal_exact_expert_selector"
    args["policy_name"] = (
        "experiments.robotwin.policy_content_adapter.formal_episode_protocol"
    )
    args["eval_mode"] = True

    embodiment_type = args.get("embodiment")
    _require(
        isinstance(embodiment_type, list) and len(embodiment_type) in {1, 3},
        "RoboTwin embodiment config changed",
    )
    embodiment_config_path = Path(module.CONFIGS_PATH) / "_embodiment_config.yml"
    with embodiment_config_path.open("r", encoding="utf-8") as handle:
        embodiment_types = module.yaml.load(
            handle.read(), Loader=module.yaml.FullLoader
        )
    _require(isinstance(embodiment_types, Mapping), "embodiment inventory is invalid")

    def embodiment_file(name: str) -> str:
        row = embodiment_types.get(name)
        _require(isinstance(row, Mapping), f"unknown embodiment {name!r}")
        path = row.get("file_path")
        _require(isinstance(path, str) and path, f"embodiment {name!r} lacks file_path")
        return path

    camera_config_path = Path(module.CONFIGS_PATH) / "_camera_config.yml"
    with camera_config_path.open("r", encoding="utf-8") as handle:
        camera_types = module.yaml.load(handle.read(), Loader=module.yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_types[head_camera_type]["h"]
    args["head_camera_w"] = camera_types[head_camera_type]["w"]
    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(str(embodiment_type[0]))
        args["right_robot_file"] = embodiment_file(str(embodiment_type[0]))
        args["dual_arm_embodied"] = True
    else:
        args["left_robot_file"] = embodiment_file(str(embodiment_type[0]))
        args["right_robot_file"] = embodiment_file(str(embodiment_type[1]))
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = module.get_embodiment_config(
        args["left_robot_file"]
    )
    args["right_embodiment_config"] = module.get_embodiment_config(
        args["right_robot_file"]
    )
    return args


@contextlib.contextmanager
def _temporary_process_context(
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> Any:
    previous_cwd = Path.cwd()
    previous_environment = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(environment)
        os.chdir(cwd)
        yield
    finally:
        os.chdir(previous_cwd)
        os.environ.clear()
        os.environ.update(previous_environment)


def realize_cell_on_gpu(
    *,
    robotwin_root: str | Path,
    candidate_bank_path: str | Path,
    formal_lock_path: str | Path,
    task: str,
    task_config: str,
    gpu_id: int | str,
    output: str | Path,
) -> dict[str, Any]:
    """Create one immutable expert-only realization trace on one pinned GPU."""

    destination = Path(output).expanduser().resolve()
    _require(
        not destination.exists(),
        f"refusing to overwrite realization cell: {destination}",
    )
    _require(task in TASKS, f"unsupported formal task {task!r}")
    _require(task_config in TASK_CONFIGS, f"unsupported task_config {task_config!r}")
    root = Path(robotwin_root).expanduser().resolve()
    audit_selector_inputs(
        robotwin_root=root,
        candidate_bank_path=candidate_bank_path,
        formal_lock_path=formal_lock_path,
    )

    # Importing the GPU helpers is safe here: neither imports torch/SAPIEN.
    # CUDA visibility is then installed before the vendored evaluator is
    # dynamically imported for the first time in this process.
    from .pinned_eval_policy import (
        _load_robotwin_eval_module,
        install_setup_demo_pin,
        validate_pinned_environment,
    )
    from .robotwin_gpu_runtime import (
        gpu_binding_environment,
        preflight_gpu_runtime,
    )

    binding = preflight_gpu_runtime(
        gpu_id,
        python_executable=sys.executable,
        check_vulkan=True,
        check_sapien=True,
    )
    normalized_runtime = validate_runtime_binding_payload(binding, verify_files=True)
    environment = gpu_binding_environment(binding)
    with _temporary_process_context(cwd=root, environment=environment):
        pinned = validate_pinned_environment()
        _require(
            pinned["physical_gpu_index"]
            == normalized_runtime["physical_gpu_index"]
            and pinned["pci_bus_id"] == normalized_runtime["pci_bus_id"]
            and pinned["render_device_alias"]
            == normalized_runtime["render_device_alias"],
            "selector process environment differs from preflight PCI binding",
        )
        module = _load_robotwin_eval_module(root)
        install_setup_demo_pin(
            module,
            render_device_alias=normalized_runtime["render_device_alias"],
        )
        from test_render import Sapien_TEST

        probe = Sapien_TEST(
            render_device_alias=normalized_runtime["render_device_alias"]
        )
        del probe
        gc.collect()
        args = _stock_task_arguments(
            module,
            robotwin_root=root,
            task=task,
            task_config=task_config,
        )
        candidate, _, _, _ = _parent_artifacts(
            candidate_bank_path, formal_lock_path
        )
        task_env = module.class_decorator(task)
        task_env.suc = 0
        task_env.test_num = 0
        accepted, attempts = scan_expert_candidates(
            module,
            task_name=task,
            task_config=task_config,
            task_env=task_env,
            args=args,
            candidate_bank=candidate,
            instruction_type="unseen",
            test_num=EPISODES_PER_CELL,
        )
        payload = build_realization_cell_manifest(
            robotwin_root=root,
            task=task,
            task_config=task_config,
            instruction_type="unseen",
            candidate_bank_path=candidate_bank_path,
            formal_lock_path=formal_lock_path,
            episodes=accepted,
            attempts=attempts,
            runtime_binding=normalized_runtime,
        )
        written = _exclusive_json(destination, payload)
    return {
        "status": "PASS",
        "gpu_started": True,
        "task": task,
        "task_config": task_config,
        "cell_id": payload["cell_id"],
        "ordered_seed_instruction_sha256": payload[
            "ordered_seed_instruction_sha256"
        ],
        "cell": stable_file_identity(written),
    }


def build_realization_cell_commands(
    *,
    robotwin_root: str | Path,
    candidate_bank_path: str | Path,
    formal_lock_path: str | Path,
    output_root: str | Path,
    gpu_ids: str | Sequence[int | str],
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """CPU-only construction of six independent cell commands plus merge."""

    if isinstance(gpu_ids, str):
        raw_ids = [item.strip() for item in gpu_ids.split(",")]
    else:
        raw_ids = [str(item).strip() for item in gpu_ids]
    _require(len(raw_ids) == 6, "exact cell realization requires six GPU ids")
    _require(
        all(item.isdigit() for item in raw_ids),
        "exact cell realization GPU ids must be non-negative integers",
    )
    physical_gpus = [int(item) for item in raw_ids]
    _require(
        len(set(physical_gpus)) == 6,
        "exact cell realization GPU ids must be unique",
    )
    audit = audit_selector_inputs(
        robotwin_root=robotwin_root,
        candidate_bank_path=candidate_bank_path,
        formal_lock_path=formal_lock_path,
    )
    root = Path(robotwin_root).expanduser().resolve()
    candidate = Path(candidate_bank_path).expanduser().resolve()
    lock = Path(formal_lock_path).expanduser().resolve()
    outputs = Path(output_root).expanduser().resolve()
    module_name = "experiments.robotwin.policy_content_adapter.formal_episode_protocol"
    cells = [(task, config) for task in TASKS for config in TASK_CONFIGS]
    jobs: list[dict[str, Any]] = []
    for index, ((task, config), gpu) in enumerate(
        zip(cells, physical_gpus, strict=True)
    ):
        output = outputs / "cells" / task / f"{config}.json"
        _require(
            not output.exists(),
            f"refusing to plan over an existing realization cell: {output}",
        )
        command = [
            str(Path(python_executable).expanduser().resolve()),
            "-m",
            module_name,
            "realize-cell",
            "--robotwin-root",
            str(root),
            "--candidate-bank",
            str(candidate),
            "--formal-lock",
            str(lock),
            "--task",
            task,
            "--task-config",
            config,
            "--gpu-id",
            str(gpu),
            "--output",
            str(output),
        ]
        jobs.append(
            {
                "job_index": index,
                "task": task,
                "task_config": config,
                "physical_gpu_index": gpu,
                "output": str(output),
                "command": command,
            }
        )
    bank_output = outputs / "realization_bank.json"
    _require(
        not bank_output.exists(),
        f"refusing to plan over an existing realization bank: {bank_output}",
    )
    merge_command = [
        str(Path(python_executable).expanduser().resolve()),
        "-m",
        module_name,
        "finalize",
        "--candidate-bank",
        str(candidate),
        "--formal-lock",
        str(lock),
    ]
    for job in jobs:
        merge_command.extend(["--cell", job["output"]])
    merge_command.extend(["--output", str(bank_output)])
    return {
        "status": "PASS",
        "gpu_started": False,
        "candidate_seed_bank": audit["candidate_seed_bank"],
        "candidate_seed_bank_id": audit["candidate_seed_bank_id"],
        "formal_protocol_lock": audit["formal_protocol_lock"],
        "output_root": str(outputs),
        "jobs": jobs,
        "merge_output": str(bank_output),
        "merge_command": merge_command,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser(
        "audit-inputs", help="CPU-only audit of candidate bank, lock, and sources"
    )
    inputs.add_argument("--robotwin-root", default=str(DEFAULT_ROBOTWIN_ROOT))
    inputs.add_argument("--candidate-bank", required=True)
    inputs.add_argument("--formal-lock", required=True)
    realize = commands.add_parser(
        "realize-cell",
        help="GPU expert-only selection for one exact task/domain cell",
    )
    realize.add_argument("--robotwin-root", default=str(DEFAULT_ROBOTWIN_ROOT))
    realize.add_argument("--candidate-bank", required=True)
    realize.add_argument("--formal-lock", required=True)
    realize.add_argument("--task", required=True, choices=TASKS)
    realize.add_argument("--task-config", required=True, choices=TASK_CONFIGS)
    realize.add_argument("--gpu-id", required=True)
    realize.add_argument("--output", required=True)
    cell_commands = commands.add_parser(
        "cell-commands",
        help="CPU-only JSON argv plan for six realization cells and merge",
    )
    cell_commands.add_argument("--robotwin-root", default=str(DEFAULT_ROBOTWIN_ROOT))
    cell_commands.add_argument("--candidate-bank", required=True)
    cell_commands.add_argument("--formal-lock", required=True)
    cell_commands.add_argument("--output-root", required=True)
    cell_commands.add_argument("--gpu-ids", default="0,1,2,4,5,6")
    cell_commands.add_argument("--python", default=sys.executable)
    finalize = commands.add_parser("finalize", help="create the six-cell realization bank")
    finalize.add_argument("--candidate-bank", required=True)
    finalize.add_argument("--formal-lock", required=True)
    finalize.add_argument("--cell", action="append", required=True)
    finalize.add_argument("--output", required=True)
    audit = commands.add_parser("audit", help="validate a realization cell or bank")
    audit.add_argument("--cell")
    audit.add_argument("--bank")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "audit-inputs":
        print(
            json.dumps(
                audit_selector_inputs(
                    robotwin_root=args.robotwin_root,
                    candidate_bank_path=args.candidate_bank,
                    formal_lock_path=args.formal_lock,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "realize-cell":
        print(
            json.dumps(
                realize_cell_on_gpu(
                    robotwin_root=args.robotwin_root,
                    candidate_bank_path=args.candidate_bank,
                    formal_lock_path=args.formal_lock,
                    task=args.task,
                    task_config=args.task_config,
                    gpu_id=args.gpu_id,
                    output=args.output,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "cell-commands":
        print(
            json.dumps(
                build_realization_cell_commands(
                    robotwin_root=args.robotwin_root,
                    candidate_bank_path=args.candidate_bank,
                    formal_lock_path=args.formal_lock,
                    output_root=args.output_root,
                    gpu_ids=args.gpu_ids,
                    python_executable=args.python,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "finalize":
        _require(
            not Path(args.output).expanduser().resolve().exists(),
            f"refusing to overwrite realization bank: {Path(args.output).expanduser().resolve()}",
        )
        payload = finalize_realization_bank(
            candidate_bank_path=args.candidate_bank,
            formal_lock_path=args.formal_lock,
            cell_paths=args.cell,
        )
        output = _exclusive_json(args.output, payload)
        print(json.dumps({"status": "PASS", "bank": stable_file_identity(output)}, indent=2))
    elif args.cell:
        payload, path = validate_realization_cell(args.cell)
        print(json.dumps({"status": "PASS", "cell_id": payload["cell_id"], "path": str(path)}, indent=2))
    elif args.bank:
        payload, path = validate_realization_bank(args.bank)
        print(json.dumps({"status": "PASS", "realization_bank_id": payload["realization_bank_id"], "path": str(path)}, indent=2))
    else:
        raise SystemExit("audit requires --cell or --bank")


if __name__ == "__main__":
    main()


__all__ = [
    "BANK_SCHEMA",
    "CELL_SCHEMA",
    "EPISODES_PER_CELL",
    "FormalEpisodeProtocolError",
    "TASKS",
    "TASK_CONFIGS",
    "TASK_CONFIG_TO_DOMAIN",
    "TRACE_SCHEMA",
    "audit_selector_inputs",
    "build_realization_cell_commands",
    "build_realization_cell_manifest",
    "finalize_realization_bank",
    "install_formal_episode_mode",
    "ordered_seed_instruction_payload",
    "realize_cell_on_gpu",
    "scan_expert_candidates",
    "select_realization_cell",
    "stable_file_identity",
    "validate_realization_bank",
    "validate_realization_cell",
    "validate_replay_trace",
    "validate_runtime_binding_payload",
]
