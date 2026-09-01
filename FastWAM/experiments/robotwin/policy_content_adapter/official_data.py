"""Strict three-task view over the official FastWAM RoboTwin policy data.

The native :class:`RobotVideoDataset` deliberately retries failed reads with a
random global sample in ``__getitem__``.  That behaviour is convenient for
large-scale pretraining but is unsafe for a task-restricted experiment: a
failed target-task read can silently become a frame from another task.  This
module therefore calls the native ``_get(base_index)`` path directly and adds
an inner index guard which rejects any retry that changed the requested frame.

The task-to-episode mapping is not inferred from prompt keywords at runtime.
It is accepted only for the exact release metadata identified by the compiled
SHA-256 contract below.  Both a changed metadata file and a changed manifest
range fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


OFFICIAL_TASKS = (
    "place_a2b_left",
    "open_microwave",
    "move_stapler_pad",
)

# These identities describe the release dataset at
# /mnt/cpfs-E/baoshifeng/FastWAM/data/robotwin2.0/robotwin2.0.  Keep the
# contract in code as well as JSON: otherwise editing the manifest and its
# ranges together could silently admit another task.
EXPECTED_META_FILES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "meta/info.json": MappingProxyType(
            {
                "size_bytes": 4_601,
                "sha256": "441a98fffe047bb642dba617bd6d89bbe313dd1379858744844536d73c493609",
            }
        ),
        "meta/episodes.jsonl": MappingProxyType(
            {
                "size_bytes": 200_586_956,
                "sha256": "7489b3636f8960d04491df4319711f6f12d38f72b8a775408a5d7af4689070b9",
            }
        ),
        "meta/tasks.jsonl": MappingProxyType(
            {
                "size_bytes": 130_765_390,
                "sha256": "ba499d6bd77debf911456d7352baa4fb91e4efc685caa6474f308a6df614b99f",
            }
        ),
    }
)

EXPECTED_DATASET_FACTS: Mapping[str, int | str] = MappingProxyType(
    {
        "codebase_version": "v2.1",
        "total_episodes": 27_500,
        "total_frames": 6_075_103,
        "total_tasks": 921_032,
        "total_videos": 82_500,
        "fps": 50,
    }
)

# Inclusive canonical episode ranges.  Each task is one contiguous 550-episode
# block in the hash-bound release metadata.
EXPECTED_TASK_EPISODE_RANGES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "place_a2b_left": (11_000, 11_549),
        "open_microwave": (9_350, 9_899),
        "move_stapler_pad": (8_250, 8_799),
    }
)

# Protocol-v2 range contract for the hash-bound FastWAM release.  The release
# metadata itself does not contain a ``domain`` column: these labels are valid
# only because the approved protocol fixes the first 50 episodes in each
# 550-episode task block as Clean and the following 500 as Official Random.
# Keeping the exact subranges compiled into code prevents a manifest edit from
# silently changing either domain or count.
EXPECTED_TASK_DOMAIN_EPISODE_RANGES: Mapping[
    str, Mapping[str, tuple[int, int]]
] = MappingProxyType(
    {
        "place_a2b_left": MappingProxyType(
            {"clean": (11_000, 11_049), "official_random": (11_050, 11_549)}
        ),
        "open_microwave": MappingProxyType(
            {"clean": (9_350, 9_399), "official_random": (9_400, 9_899)}
        ),
        "move_stapler_pad": MappingProxyType(
            {"clean": (8_250, 8_299), "official_random": (8_300, 8_799)}
        ),
    }
)
OFFICIAL_DOMAINS = ("clean", "official_random")
EPISODE_SELECTION_MODES = ("native_99pct", "full_550_per_task")


class OfficialDataContractError(ValueError):
    """The input cannot prove the official three-task data contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialDataContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OfficialDataContractError(f"duplicate JSON key in manifest: {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except OfficialDataContractError:
        raise
    except Exception as exc:
        raise OfficialDataContractError(f"cannot parse JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


@dataclass(frozen=True)
class TaskEpisodeRange:
    """One canonical inclusive task episode block."""

    task: str
    episode_start: int
    episode_end_inclusive: int

    @property
    def episode_count(self) -> int:
        return self.episode_end_inclusive - self.episode_start + 1

    def contains(self, episode_index: int) -> bool:
        return self.episode_start <= int(episode_index) <= self.episode_end_inclusive


@dataclass(frozen=True)
class TaskDomainEpisodeRange:
    """One inclusive domain subrange inside a task's canonical block."""

    task: str
    domain: str
    episode_start: int
    episode_end_inclusive: int

    @property
    def episode_count(self) -> int:
        return self.episode_end_inclusive - self.episode_start + 1

    def contains(self, episode_index: int) -> bool:
        return self.episode_start <= int(episode_index) <= self.episode_end_inclusive


@dataclass(frozen=True)
class VerifiedOfficialManifest:
    """Manifest whose declarations and dataset files passed strict checks."""

    manifest_path: Path
    manifest_sha256: str
    dataset_root: Path
    task_ranges: tuple[TaskEpisodeRange, ...]
    task_domain_ranges: tuple[TaskDomainEpisodeRange, ...]
    meta_files: tuple[tuple[str, int, str], ...]

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(item.task for item in self.task_ranges)

    def domain_for_episode(self, task: str, episode_index: int) -> str:
        matches = tuple(
            item.domain
            for item in self.task_domain_ranges
            if item.task == task and item.contains(episode_index)
        )
        _require(
            len(matches) == 1,
            f"episode {episode_index} has no unique protocol-v2 domain for {task}",
        )
        return matches[0]

    def as_provenance(self) -> dict[str, Any]:
        return {
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
            },
            "dataset_root": str(self.dataset_root),
            "domain_label": "protocol_v2_hash_bound_range_partition",
            "domain_verified": True,
            "domain_verification_scope": "episode_index_ranges_in_hash_bound_release",
            "intrinsic_metadata_domain_field": False,
            "meta_files": {
                name: {"size_bytes": size, "sha256": digest}
                for name, size, digest in self.meta_files
            },
            "tasks": {
                item.task: {
                    "episode_start": item.episode_start,
                    "episode_end_inclusive": item.episode_end_inclusive,
                    "episode_count": item.episode_count,
                    "domains": {
                        domain_item.domain: {
                            "episode_start": domain_item.episode_start,
                            "episode_end_inclusive": domain_item.episode_end_inclusive,
                            "episode_count": domain_item.episode_count,
                        }
                        for domain_item in self.task_domain_ranges
                        if domain_item.task == item.task
                    },
                }
                for item in self.task_ranges
            },
        }


@dataclass(frozen=True)
class NativeSplitEpisodeSelection:
    """Exact manifest intersection with FastWAM's native episode split.

    ``episode_ids`` deliberately preserves the shuffled order produced by
    ``BaseLerobotDataset``.  Keeping that order is important because the
    native LeRobot dataset concatenates selected parquet files in precisely
    this sequence, so local frame indices and all downstream transforms stay
    native-compatible.
    """

    manifest_sha256: str
    dataset_root: Path
    seed: int
    val_set_proportion: float
    is_training_set: bool
    native_split_episode_count: int
    episode_ids: tuple[int, ...]
    episodes_by_task: tuple[tuple[str, tuple[int, ...]], ...]
    selection_mode: Literal["native_99pct", "full_550_per_task"] = "native_99pct"
    episodes_by_task_domain: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ] = ()

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(task for task, _ in self.episodes_by_task)

    @property
    def target_episode_set(self) -> frozenset[int]:
        return frozenset(self.episode_ids)

    def as_provenance(self) -> dict[str, Any]:
        domain_counts = {
            task: {domain: len(values) for domain, values in domains}
            for task, domains in self.episodes_by_task_domain
        }
        return {
            "implementation": "experiment_local_explicit_episode_native_loader",
            "selection_mode": self.selection_mode,
            "manifest_sha256": self.manifest_sha256,
            "dataset_root": str(self.dataset_root),
            "native_split": {
                "algorithm": "numpy.default_rng(seed).shuffle_then_floor_split",
                "seed": self.seed,
                "val_set_proportion": self.val_set_proportion,
                "is_training_set": self.is_training_set,
                "pre_intersection_episode_count": self.native_split_episode_count,
            },
            "loaded_episode_count": len(self.episode_ids),
            "loaded_episode_counts_by_task": {
                task: len(episode_ids) for task, episode_ids in self.episodes_by_task
            },
            "loaded_episode_counts_by_task_domain": domain_counts,
            "only_manifest_split_intersection_loaded": True,
        }


def _validate_manifest_declarations(
    manifest: Mapping[str, Any],
) -> tuple[tuple[TaskEpisodeRange, ...], tuple[TaskDomainEpisodeRange, ...]]:
    _require(manifest.get("schema_version") == 2, "official manifest schema_version must be 2")
    _require(
        manifest.get("manifest_id") == "fastwam_robotwin2_release_three_task_v2",
        "unexpected official manifest_id",
    )
    dataset = manifest.get("dataset")
    _require(isinstance(dataset, Mapping), "manifest.dataset must be an object")

    declared_files = dataset.get("meta_files")
    _require(isinstance(declared_files, Mapping), "manifest.dataset.meta_files must be an object")
    _require(
        set(declared_files) == set(EXPECTED_META_FILES),
        "manifest metadata file set differs from the compiled release contract",
    )
    for relative_path, expected in EXPECTED_META_FILES.items():
        declared = declared_files.get(relative_path)
        _require(isinstance(declared, Mapping), f"manifest entry missing for {relative_path}")
        _require(
            declared.get("size_bytes") == expected["size_bytes"],
            f"manifest size mismatch for {relative_path}",
        )
        _require(
            declared.get("sha256") == expected["sha256"],
            f"manifest SHA-256 mismatch for {relative_path}",
        )

    facts = dataset.get("facts")
    _require(isinstance(facts, Mapping), "manifest.dataset.facts must be an object")
    for key, expected in EXPECTED_DATASET_FACTS.items():
        _require(facts.get(key) == expected, f"manifest dataset fact mismatch for {key}")

    declared_order = tuple(str(value) for value in manifest.get("task_order", ()))
    _require(declared_order == OFFICIAL_TASKS, f"task_order must be exactly {OFFICIAL_TASKS}")
    declared_tasks = manifest.get("tasks")
    _require(isinstance(declared_tasks, Mapping), "manifest.tasks must be an object")
    _require(
        set(declared_tasks) == set(OFFICIAL_TASKS),
        "manifest task set differs from the exact three-task contract",
    )

    result: list[TaskEpisodeRange] = []
    domain_result: list[TaskDomainEpisodeRange] = []
    for task in OFFICIAL_TASKS:
        declared = declared_tasks.get(task)
        _require(isinstance(declared, Mapping), f"manifest task entry missing for {task}")
        expected_start, expected_end = EXPECTED_TASK_EPISODE_RANGES[task]
        _require(
            declared.get("episode_start") == expected_start,
            f"manifest episode_start mismatch for {task}",
        )
        _require(
            declared.get("episode_end_inclusive") == expected_end,
            f"manifest episode_end_inclusive mismatch for {task}",
        )
        expected_count = expected_end - expected_start + 1
        _require(
            declared.get("episode_count") == expected_count,
            f"manifest episode_count mismatch for {task}",
        )
        result.append(TaskEpisodeRange(task, expected_start, expected_end))

        declared_domains = declared.get("domains")
        _require(isinstance(declared_domains, Mapping), f"manifest domains missing for {task}")
        _require(
            tuple(declared_domains) == OFFICIAL_DOMAINS,
            f"manifest domain order for {task} must be exactly {OFFICIAL_DOMAINS}",
        )
        previous_end: int | None = None
        for domain in OFFICIAL_DOMAINS:
            domain_declared = declared_domains.get(domain)
            _require(
                isinstance(domain_declared, Mapping),
                f"manifest domain entry missing for {task}/{domain}",
            )
            domain_start, domain_end = EXPECTED_TASK_DOMAIN_EPISODE_RANGES[task][domain]
            _require(
                domain_declared.get("episode_start") == domain_start,
                f"manifest episode_start mismatch for {task}/{domain}",
            )
            _require(
                domain_declared.get("episode_end_inclusive") == domain_end,
                f"manifest episode_end_inclusive mismatch for {task}/{domain}",
            )
            _require(
                domain_declared.get("episode_count") == domain_end - domain_start + 1,
                f"manifest episode_count mismatch for {task}/{domain}",
            )
            if previous_end is None:
                _require(domain_start == expected_start, f"first domain does not start {task}")
            else:
                _require(
                    domain_start == previous_end + 1,
                    f"domain ranges are not contiguous for {task}",
                )
            previous_end = domain_end
            domain_result.append(TaskDomainEpisodeRange(task, domain, domain_start, domain_end))
        _require(previous_end == expected_end, f"domain ranges do not cover all of {task}")

    domain = manifest.get("domain")
    _require(isinstance(domain, Mapping), "manifest.domain must be an object")
    _require(domain.get("verified") is True, "protocol-v2 domain partition must be verified")
    _require(
        domain.get("label") == "protocol_v2_hash_bound_range_partition",
        "unexpected release domain label",
    )
    _require(
        domain.get("verification_scope") == "episode_index_ranges_in_hash_bound_release",
        "unexpected domain verification scope",
    )
    _require(
        domain.get("intrinsic_metadata_domain_field") is False,
        "manifest must not claim an intrinsic metadata domain field",
    )
    return tuple(result), tuple(domain_result)


def verify_official_task_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
) -> VerifiedOfficialManifest:
    """Verify manifest constants and hash-bind them to ``dataset_root``.

    The hashes are recomputed on every call.  No mtime-based cache is used,
    because a strict provenance gate must detect in-place metadata changes.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    _require(manifest_file.is_file(), f"official task manifest not found: {manifest_file}")
    _require(root.is_dir(), f"official dataset root not found: {root}")
    manifest = _load_json_object(manifest_file)
    task_ranges, task_domain_ranges = _validate_manifest_declarations(manifest)

    identities: list[tuple[str, int, str]] = []
    for relative_path, expected in EXPECTED_META_FILES.items():
        path = root / relative_path
        _require(path.is_file(), f"official metadata file not found: {path}")
        actual_size = path.stat().st_size
        _require(
            actual_size == expected["size_bytes"],
            f"official metadata size mismatch for {relative_path}: "
            f"expected {expected['size_bytes']}, got {actual_size}",
        )
        actual_sha256 = _sha256(path)
        _require(
            actual_sha256 == expected["sha256"],
            f"official metadata SHA-256 mismatch for {relative_path}: "
            f"expected {expected['sha256']}, got {actual_sha256}",
        )
        identities.append((relative_path, actual_size, actual_sha256))

    info = _load_json_object(root / "meta/info.json")
    for key, expected in EXPECTED_DATASET_FACTS.items():
        _require(info.get(key) == expected, f"official info.json fact mismatch for {key}")

    return VerifiedOfficialManifest(
        manifest_path=manifest_file,
        manifest_sha256=_sha256(manifest_file),
        dataset_root=root,
        task_ranges=task_ranges,
        task_domain_ranges=task_domain_ranges,
        meta_files=tuple(identities),
    )


def select_official_episodes_from_native_split(
    verified: VerifiedOfficialManifest,
    *,
    val_set_proportion: float,
    is_training_set: bool,
    seed: int = 42,
) -> NativeSplitEpisodeSelection:
    """Reproduce the native split, then retain only manifest episodes.

    This is intentionally a literal implementation of the release
    ``BaseLerobotDataset`` split contract: a fresh NumPy ``default_rng``
    shuffles all canonical episode IDs, the split point is floor-truncated by
    ``int``, and training selects the prefix while validation selects the
    suffix.  For proportions below ``1e-6`` the native implementation selects
    all episodes for either mode.
    """

    _require(
        isinstance(verified, VerifiedOfficialManifest),
        "verified must be a VerifiedOfficialManifest",
    )
    try:
        proportion = float(val_set_proportion)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OfficialDataContractError(
            f"val_set_proportion is not numeric: {val_set_proportion!r}"
        ) from exc
    _require(math.isfinite(proportion), "val_set_proportion must be finite")
    _require(
        0.0 <= proportion <= 1.0,
        "val_set_proportion must be within [0, 1]",
    )
    _require(
        isinstance(is_training_set, (bool, np.bool_)),
        "is_training_set must be bool",
    )
    _require(
        isinstance(seed, (int, np.integer)) and not isinstance(seed, bool),
        "native split seed must be an integer",
    )

    total_episodes = int(EXPECTED_DATASET_FACTS["total_episodes"])
    if proportion < 1e-6:
        native_split = list(range(total_episodes))
    else:
        split_index = int(total_episodes * (1.0 - proportion))
        shuffled = list(range(total_episodes))
        rng = np.random.default_rng(int(seed))
        rng.shuffle(shuffled)
        native_split = (
            shuffled[:split_index]
            if bool(is_training_set)
            else shuffled[split_index:]
        )

    range_by_task = {item.task: item for item in verified.task_ranges}
    domain_ranges = {
        (item.task, item.domain): item for item in verified.task_domain_ranges
    }
    selected_by_task: dict[str, list[int]] = {task: [] for task in OFFICIAL_TASKS}
    selected_by_task_domain: dict[str, dict[str, list[int]]] = {
        task: {domain: [] for domain in OFFICIAL_DOMAINS} for task in OFFICIAL_TASKS
    }
    selected: list[int] = []
    for episode_index in native_split:
        task = next(
            (
                task_name
                for task_name in OFFICIAL_TASKS
                if range_by_task[task_name].contains(episode_index)
            ),
            None,
        )
        if task is not None:
            selected.append(int(episode_index))
            selected_by_task[task].append(int(episode_index))
            domains = tuple(
                domain
                for domain in OFFICIAL_DOMAINS
                if domain_ranges[(task, domain)].contains(episode_index)
            )
            _require(
                len(domains) == 1,
                f"native split episode {episode_index} has no unique domain for {task}",
            )
            selected_by_task_domain[task][domains[0]].append(int(episode_index))

    _require(len(selected) == len(set(selected)), "native split intersection has duplicates")
    for task in OFFICIAL_TASKS:
        _require(
            selected_by_task[task],
            f"native split contains no manifest episodes for required task {task}",
        )
        _require(
            all(range_by_task[task].contains(value) for value in selected_by_task[task]),
            f"native split intersection leaked a non-{task} episode",
        )

    return NativeSplitEpisodeSelection(
        manifest_sha256=verified.manifest_sha256,
        dataset_root=verified.dataset_root,
        seed=int(seed),
        val_set_proportion=proportion,
        is_training_set=bool(is_training_set),
        native_split_episode_count=len(native_split),
        episode_ids=tuple(selected),
        episodes_by_task=tuple(
            (task, tuple(selected_by_task[task])) for task in OFFICIAL_TASKS
        ),
        selection_mode="native_99pct",
        episodes_by_task_domain=tuple(
            (
                task,
                tuple(
                    (domain, tuple(selected_by_task_domain[task][domain]))
                    for domain in OFFICIAL_DOMAINS
                ),
            )
            for task in OFFICIAL_TASKS
        ),
    )


def select_official_full_550_per_task(
    verified: VerifiedOfficialManifest,
    *,
    seed: int = 42,
) -> NativeSplitEpisodeSelection:
    """Select exactly 50 Clean + 500 Official Random episodes per task.

    ``episode_ids`` is globally ascending because the native loader is forced
    to ``val_set_proportion=0`` in this mode and therefore presents canonical
    ``range(total_episodes)`` order before the explicit three-task
    intersection.  No 99/1 split is applied.
    """

    _require(
        isinstance(verified, VerifiedOfficialManifest),
        "verified must be a VerifiedOfficialManifest",
    )
    _require(
        isinstance(seed, (int, np.integer)) and not isinstance(seed, bool),
        "native split seed must be an integer",
    )
    by_task: dict[str, tuple[int, ...]] = {}
    by_task_domain: dict[str, dict[str, tuple[int, ...]]] = {}
    all_ids: list[int] = []
    for task in OFFICIAL_TASKS:
        per_domain: dict[str, tuple[int, ...]] = {}
        for domain in OFFICIAL_DOMAINS:
            start, end = EXPECTED_TASK_DOMAIN_EPISODE_RANGES[task][domain]
            values = tuple(range(start, end + 1))
            expected_count = 50 if domain == "clean" else 500
            _require(
                len(values) == expected_count,
                f"compiled full-selection count mismatch for {task}/{domain}",
            )
            per_domain[domain] = values
            all_ids.extend(values)
        task_values = tuple(
            value for domain in OFFICIAL_DOMAINS for value in per_domain[domain]
        )
        expected_task_range = next(item for item in verified.task_ranges if item.task == task)
        _require(
            task_values
            == tuple(
                range(
                    expected_task_range.episode_start,
                    expected_task_range.episode_end_inclusive + 1,
                )
            ),
            f"protocol-v2 domains do not exactly cover {task}",
        )
        by_task[task] = task_values
        by_task_domain[task] = per_domain

    episode_ids = tuple(sorted(all_ids))
    _require(len(episode_ids) == 1_650, "full selection must contain 1,650 episodes")
    _require(len(set(episode_ids)) == 1_650, "full selection contains duplicate episodes")
    return NativeSplitEpisodeSelection(
        manifest_sha256=verified.manifest_sha256,
        dataset_root=verified.dataset_root,
        seed=int(seed),
        val_set_proportion=0.0,
        is_training_set=True,
        native_split_episode_count=int(EXPECTED_DATASET_FACTS["total_episodes"]),
        episode_ids=episode_ids,
        episodes_by_task=tuple((task, by_task[task]) for task in OFFICIAL_TASKS),
        selection_mode="full_550_per_task",
        episodes_by_task_domain=tuple(
            (
                task,
                tuple(
                    (domain, by_task_domain[task][domain])
                    for domain in OFFICIAL_DOMAINS
                ),
            )
            for task in OFFICIAL_TASKS
        ),
    )


def _scalar_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise OfficialDataContractError(f"{name} must be an integer, not bool")
    if isinstance(value, torch.Tensor):
        _require(value.numel() == 1, f"{name} tensor must contain exactly one value")
        value = value.detach().cpu().item()
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OfficialDataContractError(f"{name} is not integer-like: {value!r}") from exc
    _require(integer == value, f"{name} is not an exact integer: {value!r}")
    return integer


def _integer_tuple(values: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        _require(values.ndim == 1, f"{name} must be one-dimensional")
        values = values.detach().cpu().tolist()
    _require(isinstance(values, Sequence), f"{name} must be a sequence")
    return tuple(_scalar_int(value, name=f"{name} item") for value in values)


class _FailClosedLerobotProxy:
    """Delegate to BaseLerobotDataset but reject its random-index recovery."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __len__(self) -> int:
        return len(self._target)

    def __getitem__(self, requested_index: int) -> Any:
        sample = self._target[requested_index]
        _require(isinstance(sample, Mapping), "native BaseLerobotDataset returned a non-mapping")
        _require(
            "idx" in sample,
            "native BaseLerobotDataset result lacks idx; cannot exclude random fallback",
        )
        actual_index = _scalar_int(sample["idx"], name="native returned idx")
        _require(
            actual_index == int(requested_index),
            "native loader replaced requested frame "
            f"{requested_index} with random frame {actual_index}",
        )
        return sample


@dataclass(frozen=True)
class OfficialSampleIndex:
    task: str
    domain: str
    episode_index: int
    base_index: int
    frame_offset: int


class OfficialThreeTaskDataset(Dataset[dict[str, Any]]):
    """Fail-closed exact three-task view over a native RobotVideoDataset.

    ``episode_anchor`` selects the lower midpoint frame of every available
    episode.  ``all_frames`` selects every frame from every available episode.
    "Available" means present in the native dataset's already-selected
    train/validation episode list; the wrapper never changes that split.
    """

    _ATTACHED_KEYS = (
        "official_task",
        "official_domain",
        "official_episode_index",
        "official_base_index",
        "official_subset_index",
        "task_name",
        "episode_index",
        "base_index",
    )

    def __init__(
        self,
        dataset: Dataset,
        *,
        dataset_root: str | Path,
        manifest_path: str | Path,
        sampling_mode: Literal["all_frames", "episode_anchor"] = "all_frames",
    ) -> None:
        _require(
            sampling_mode in ("all_frames", "episode_anchor"),
            "sampling_mode must be 'all_frames' or 'episode_anchor'",
        )
        _require(callable(getattr(dataset, "_get", None)), "native dataset must expose callable _get")
        _require(
            not bool(getattr(dataset, "skip_padding_as_possible", False)),
            "skip_padding_as_possible must be false; it can change the requested frame",
        )
        verified = verify_official_task_manifest(manifest_path, dataset_root)

        lerobot = getattr(dataset, "lerobot_dataset", None)
        _require(lerobot is not None, "native dataset lacks lerobot_dataset")
        multi_dataset = getattr(lerobot, "multi_dataset", None)
        inner_datasets = getattr(multi_dataset, "_datasets", None)
        _require(
            isinstance(inner_datasets, Sequence) and len(inner_datasets) == 1,
            "official three-task wrapper requires exactly one underlying dataset",
        )
        inner = inner_datasets[0]
        inner_root = Path(str(getattr(inner, "root", ""))).expanduser().resolve()
        _require(
            inner_root == verified.dataset_root,
            f"native dataset root {inner_root} differs from verified root {verified.dataset_root}",
        )

        total_episodes = int(EXPECTED_DATASET_FACTS["total_episodes"])
        selected = getattr(inner, "episodes", None)
        if selected is None:
            episode_ids = tuple(range(total_episodes))
        else:
            episode_ids = _integer_tuple(selected, name="underlying selected episodes")
        _require(len(episode_ids) > 0, "underlying native dataset selected no episodes")
        _require(len(set(episode_ids)) == len(episode_ids), "underlying episode list has duplicates")
        _require(
            all(0 <= value < total_episodes for value in episode_ids),
            "underlying episode list contains an out-of-range canonical episode",
        )
        explicit_selection = getattr(
            dataset, "_official_explicit_episode_selection", None
        )
        if explicit_selection is not None:
            _require(
                isinstance(explicit_selection, NativeSplitEpisodeSelection),
                "native dataset explicit episode provenance has an invalid type",
            )
            _require(
                explicit_selection.dataset_root == verified.dataset_root,
                "explicit episode provenance dataset root differs from verified root",
            )
            _require(
                explicit_selection.manifest_sha256 == verified.manifest_sha256,
                "explicit episode provenance manifest identity differs from verification",
            )
            _require(
                episode_ids == explicit_selection.episode_ids,
                "native loaded episode order differs from the audited split intersection",
            )

        episode_data_index = getattr(inner, "episode_data_index", None)
        _require(isinstance(episode_data_index, Mapping), "underlying episode_data_index is missing")
        starts = _integer_tuple(episode_data_index.get("from"), name="episode_data_index.from")
        ends = _integer_tuple(episode_data_index.get("to"), name="episode_data_index.to")
        _require(
            len(starts) == len(episode_ids) == len(ends),
            "episode IDs and episode_data_index lengths differ",
        )
        _require(starts[0] == 0, "episode_data_index must begin at frame zero")
        _require(
            all(start < end for start, end in zip(starts, ends, strict=True)),
            "episode_data_index contains an empty/reversed episode",
        )
        _require(
            all(ends[index] == starts[index + 1] for index in range(len(starts) - 1)),
            "episode_data_index is not contiguous",
        )
        native_length = len(dataset)
        _require(
            ends[-1] == native_length,
            f"episode_data_index ends at {ends[-1]} but native dataset length is {native_length}",
        )

        range_by_task = {item.task: item for item in verified.task_ranges}
        records: list[OfficialSampleIndex] = []
        subset_indices_by_task: dict[str, list[int]] = {task: [] for task in OFFICIAL_TASKS}
        episodes_by_task: dict[str, list[int]] = {task: [] for task in OFFICIAL_TASKS}
        episodes_by_task_domain: dict[str, dict[str, list[int]]] = {
            task: {domain: [] for domain in OFFICIAL_DOMAINS} for task in OFFICIAL_TASKS
        }
        for local_episode, canonical_episode in enumerate(episode_ids):
            task = next(
                (
                    task_name
                    for task_name in OFFICIAL_TASKS
                    if range_by_task[task_name].contains(canonical_episode)
                ),
                None,
            )
            if task is None:
                continue
            domain = verified.domain_for_episode(task, canonical_episode)
            start, end = starts[local_episode], ends[local_episode]
            episodes_by_task[task].append(canonical_episode)
            episodes_by_task_domain[task][domain].append(canonical_episode)
            if sampling_mode == "episode_anchor":
                base_indices = (start + (end - start - 1) // 2,)
            else:
                base_indices = range(start, end)
            for base_index in base_indices:
                subset_index = len(records)
                records.append(
                    OfficialSampleIndex(
                        task=task,
                        domain=domain,
                        episode_index=canonical_episode,
                        base_index=base_index,
                        frame_offset=base_index - start,
                    )
                )
                subset_indices_by_task[task].append(subset_index)

        for task in OFFICIAL_TASKS:
            _require(episodes_by_task[task], f"native split contains no episodes for required task {task}")
            _require(subset_indices_by_task[task], f"native split contains no samples for required task {task}")
            _require(
                all(range_by_task[task].contains(value) for value in episodes_by_task[task]),
                f"episode leakage detected for task {task}",
            )
        _require(records, "strict official three-task subset is empty")
        base_indices = [record.base_index for record in records]
        _require(
            len(base_indices) == len(set(base_indices)),
            "one native frame was assigned to more than one task",
        )

        # A shallow runner preserves the native _get implementation and all
        # transforms, while replacing only the inner indexing boundary with a
        # guard against BaseLerobotDataset's random retry.
        try:
            runner = copy.copy(dataset)
        except Exception as exc:
            raise OfficialDataContractError(f"cannot create guarded native dataset view: {exc}") from exc
        runner.lerobot_dataset = _FailClosedLerobotProxy(lerobot)

        self._dataset = dataset
        self._runner = runner
        self._verified_manifest = verified
        self._explicit_episode_selection = explicit_selection
        self._records = tuple(records)
        self._indices_by_task = {
            task: tuple(subset_indices_by_task[task]) for task in OFFICIAL_TASKS
        }
        self._episodes_by_task = {
            task: tuple(sorted(episodes_by_task[task])) for task in OFFICIAL_TASKS
        }
        self._episodes_by_task_domain = {
            task: {
                domain: tuple(sorted(episodes_by_task_domain[task][domain]))
                for domain in OFFICIAL_DOMAINS
            }
            for task in OFFICIAL_TASKS
        }
        self.sampling_mode = str(sampling_mode)

    @property
    def task_names(self) -> tuple[str, ...]:
        return OFFICIAL_TASKS

    @property
    def indices_by_task(self) -> dict[str, tuple[int, ...]]:
        return dict(self._indices_by_task)

    @property
    def episodes_by_task(self) -> dict[str, tuple[int, ...]]:
        return dict(self._episodes_by_task)

    @property
    def episodes_by_task_domain(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return copy.deepcopy(self._episodes_by_task_domain)

    @property
    def manifest(self) -> VerifiedOfficialManifest:
        return self._verified_manifest

    @property
    def provenance(self) -> dict[str, Any]:
        result = self._verified_manifest.as_provenance()
        result["sampling_mode"] = self.sampling_mode
        result["selected_episode_counts"] = {
            task: len(values) for task, values in self._episodes_by_task.items()
        }
        result["selected_episode_counts_by_domain"] = {
            task: {
                domain: len(values)
                for domain, values in self._episodes_by_task_domain[task].items()
            }
            for task in OFFICIAL_TASKS
        }
        result["selected_sample_counts"] = {
            task: len(values) for task, values in self._indices_by_task.items()
        }
        result["native_get_only"] = True
        result["random_fallback_rejected_by_idx_guard"] = True
        if self._explicit_episode_selection is not None:
            result["explicit_episode_native_loader"] = (
                self._explicit_episode_selection.as_provenance()
            )
            result["explicit_episode_native_loader"]["patch_scope_exited"] = bool(
                getattr(
                    self._dataset,
                    "_official_explicit_loader_patch_scope_exited",
                    False,
                )
            )
            result["explicit_episode_native_loader"]["metadata_audit"] = copy.deepcopy(
                getattr(
                    self._dataset,
                    "_official_explicit_metadata_audit",
                    None,
                )
            )
        else:
            result["explicit_episode_native_loader"] = None
        return result

    @property
    def audit_report(self) -> dict[str, Any]:
        """JSON-safe task-selection evidence for run logs/checkpoints."""

        result = self.provenance
        result["task_order"] = list(OFFICIAL_TASKS)
        result["task_histogram"] = {
            task: {
                "episodes": len(self._episodes_by_task[task]),
                "samples": len(self._indices_by_task[task]),
                "domains": {
                    domain: len(self._episodes_by_task_domain[task][domain])
                    for domain in OFFICIAL_DOMAINS
                },
            }
            for task in OFFICIAL_TASKS
        }
        result["total_selected_episodes"] = sum(
            len(values) for values in self._episodes_by_task.values()
        )
        result["total_selected_samples"] = len(self._records)
        return result

    def record_for_index(self, index: int) -> OfficialSampleIndex:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"subset index must be int, got {type(index).__name__}")
        if index < 0 or index >= len(self._records):
            raise IndexError(f"subset index {index} out of bounds for {len(self._records)} samples")
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.record_for_index(index)
        # Deliberately do not catch: native read/processing errors must remain
        # failures rather than crossing the task boundary through random retry.
        sample = self._runner._get(record.base_index)  # noqa: SLF001
        _require(isinstance(sample, Mapping), "native RobotVideoDataset._get returned a non-mapping")
        overlap = set(sample).intersection(self._ATTACHED_KEYS)
        _require(not overlap, f"native sample already contains reserved audit keys: {sorted(overlap)}")
        result = dict(sample)
        result.update(
            {
                "official_task": record.task,
                "official_domain": record.domain,
                "official_episode_index": record.episode_index,
                "official_base_index": record.base_index,
                "official_subset_index": index,
                # Short aliases make CSV/debug consumers convenient.  The
                # official_* names remain the unambiguous provenance fields.
                "task_name": record.task,
                "episode_index": record.episode_index,
                "base_index": record.base_index,
            }
        )
        return result


class ThreeTaskRoundRobinSampler(Sampler[int]):
    """Deterministic task-balanced sampler in exact ``OFFICIAL_TASKS`` order.

    Every complete consecutive three-sample round contains one item from each
    task.  Shorter task lists are cycled (and reshuffled between cycles when
    requested), so the default epoch has equal task counts rather than being
    dominated by the task with more frames.
    """

    def __init__(
        self,
        dataset: OfficialThreeTaskDataset,
        *,
        seed: int = 0,
        num_samples: int | None = None,
        shuffle: bool = True,
    ) -> None:
        _require(
            tuple(dataset.task_names) == OFFICIAL_TASKS,
            f"sampler requires exact task order {OFFICIAL_TASKS}",
        )
        groups = dataset.indices_by_task
        _require(set(groups) == set(OFFICIAL_TASKS), "sampler dataset task set is not exact")
        _require(all(groups[task] for task in OFFICIAL_TASKS), "sampler received an empty task")
        if num_samples is None:
            num_samples = len(OFFICIAL_TASKS) * max(len(groups[task]) for task in OFFICIAL_TASKS)
        _require(
            isinstance(num_samples, int) and not isinstance(num_samples, bool) and num_samples > 0,
            "num_samples must be a positive integer",
        )
        self._groups = {task: tuple(groups[task]) for task in OFFICIAL_TASKS}
        self.seed = int(seed)
        self.num_samples = int(num_samples)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    @property
    def task_schedule(self) -> tuple[str, ...]:
        """Deterministic task identity at every yielded sampler position."""

        return tuple(
            OFFICIAL_TASKS[index % len(OFFICIAL_TASKS)]
            for index in range(self.num_samples)
        )

    @property
    def schedule(self) -> tuple[str, ...]:
        """Backward-readable alias for :attr:`task_schedule`."""

        return self.task_schedule

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        queues: dict[str, list[int]] = {}
        positions = {task: 0 for task in OFFICIAL_TASKS}
        for task in OFFICIAL_TASKS:
            values = list(self._groups[task])
            if self.shuffle:
                rng.shuffle(values)
            queues[task] = values

        for output_index in range(self.num_samples):
            task = OFFICIAL_TASKS[output_index % len(OFFICIAL_TASKS)]
            position = positions[task]
            values = queues[task]
            if position == len(values):
                position = 0
                if self.shuffle:
                    rng.shuffle(values)
            yield values[position]
            positions[task] = position + 1


# Readable aliases retained for callers that prefer "subset" terminology.
OfficialThreeTaskSubset = OfficialThreeTaskDataset
TaskBalancedRoundRobinSampler = ThreeTaskRoundRobinSampler


__all__ = [
    "EPISODE_SELECTION_MODES",
    "EXPECTED_DATASET_FACTS",
    "EXPECTED_META_FILES",
    "EXPECTED_TASK_DOMAIN_EPISODE_RANGES",
    "EXPECTED_TASK_EPISODE_RANGES",
    "NativeSplitEpisodeSelection",
    "OFFICIAL_DOMAINS",
    "OFFICIAL_TASKS",
    "OfficialDataContractError",
    "OfficialSampleIndex",
    "OfficialThreeTaskDataset",
    "OfficialThreeTaskSubset",
    "TaskBalancedRoundRobinSampler",
    "TaskDomainEpisodeRange",
    "TaskEpisodeRange",
    "ThreeTaskRoundRobinSampler",
    "VerifiedOfficialManifest",
    "select_official_full_550_per_task",
    "select_official_episodes_from_native_split",
    "verify_official_task_manifest",
]
