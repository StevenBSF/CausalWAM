"""Strict construction helpers shared by training, smoke, and rollout audits."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import sys
import threading
import types
from array import array
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.robotwin.e0_e1.backbone import strict_load_release_checkpoint
from experiments.robotwin.policy_content_adapter.official_data import (
    EPISODE_SELECTION_MODES,
    EXPECTED_DATASET_FACTS,
    EXPECTED_TASK_EPISODE_RANGES,
    NativeSplitEpisodeSelection,
    OfficialDataContractError,
    _require,
    select_official_full_550_per_task,
    select_official_episodes_from_native_split,
    verify_official_task_manifest,
)
from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import aggregate_stats
from fastwam.datasets.lerobot.lerobot.datasets.utils import (
    cast_stats_to_numpy,
    load_annotations,
    load_info,
)
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import (
    LeRobotDatasetMetadata as _NativeLeRobotDatasetMetadata,
)


CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_TASK_CONFIG = "robotwin_uncond_3cam_384_1e-4"
DEFAULT_OFFICIAL_MANIFEST = (
    Path(__file__).resolve().parent / "configs/official_three_task_manifest.json"
)
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audited_fastwam_source_files() -> tuple[str, ...]:
    """Return the complete, deterministic in-workspace FastWAM Python tree."""

    package_root = (SRC_ROOT / "fastwam").resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"FastWAM package source root is missing: {package_root}")
    relative_files: list[str] = []
    for candidate in sorted(package_root.rglob("*.py")):
        resolved = candidate.resolve()
        if not resolved.is_file():
            continue
        if not resolved.is_relative_to(package_root):
            raise RuntimeError(
                f"audited FastWAM source symlink escapes package root: {candidate}"
            )
        relative_files.append(resolved.relative_to(SRC_ROOT.resolve()).as_posix())
    if not relative_files:
        raise RuntimeError("FastWAM source audit found no Python files")
    if len(relative_files) != len(set(relative_files)):
        raise RuntimeError("FastWAM source audit resolved duplicate Python paths")
    return tuple(relative_files)


def audit_local_fastwam_source() -> dict[str, Any]:
    """Hash the complete FastWAM Python tree from this workspace checkout."""

    import fastwam

    package_file = Path(str(fastwam.__file__)).resolve()
    expected_package = (SRC_ROOT / "fastwam").resolve()
    if package_file.parent != expected_package:
        raise RuntimeError(
            "fastwam import resolved outside the current workspace: "
            f"{package_file} vs {expected_package}"
        )
    files: dict[str, dict[str, Any]] = {}
    relative_files = _audited_fastwam_source_files()
    for relative in relative_files:
        path = (SRC_ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"audited FastWAM source file is missing: {path}")
        files[relative] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    return {
        "status": "PASS",
        "scope": "all_python_files_under_src_fastwam",
        "file_count": len(files),
        "package_file": str(package_file),
        "source_root": str(SRC_ROOT.resolve()),
        "files": files,
    }


@dataclass
class _ExplicitLoaderContext:
    selection: NativeSplitEpisodeSelection
    metadata_audits: list[dict[str, Any]] = field(default_factory=list)


_ACTIVE_EXPLICIT_LOADER: ContextVar[_ExplicitLoaderContext | None] = ContextVar(
    "policy_content_adapter_explicit_loader", default=None
)
_EXPLICIT_LOADER_PATCH_LOCK = threading.RLock()


def _active_explicit_loader() -> _ExplicitLoaderContext:
    context = _ACTIVE_EXPLICIT_LOADER.get()
    if context is None:
        raise OfficialDataContractError(
            "explicit official metadata loader used outside its guarded patch scope"
        )
    return context


def _validate_metadata_root(root: Path, context: _ExplicitLoaderContext) -> Path:
    resolved = Path(root).expanduser().resolve()
    _require(
        resolved == context.selection.dataset_root,
        f"native metadata root {resolved} differs from verified root "
        f"{context.selection.dataset_root}",
    )
    return resolved


class _InfoOnlyOfficialMetadata:
    """Metadata surface used only while native Base computes its split.

    BaseLerobotDataset needs just ``repo_id``, ``fps`` and ``total_episodes``
    before it delegates actual loading.  Reading 400 MiB of JSONL at this
    stage is therefore both unnecessary and the source of the original CPFS
    initialization stall.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ) -> None:
        del revision
        _require(not force_cache_sync, "official loader forbids remote cache synchronization")
        context = _active_explicit_loader()
        self.repo_id = repo_id
        self.root = _validate_metadata_root(
            Path(root) if root is not None else Path(repo_id), context
        )
        info_path = self.root / "meta/info.json"
        _require(info_path.is_file(), f"official info.json not found: {info_path}")
        self.info = load_info(self.root)
        for key, expected in EXPECTED_DATASET_FACTS.items():
            _require(self.info.get(key) == expected, f"official info fact mismatch for {key}")

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])


class _IndexedOfficialTasks(Mapping[int, str]):
    """Read-only exact task mapping with compact JSONL byte offsets.

    The task file is hash-bound by the manifest.  Initialization performs one
    sequential binary scan but does not parse or retain 921,032 prompt strings;
    requested task records are decoded on demand and checked against their
    canonical line/index identity.
    """

    def __init__(self, path: Path, *, expected_count: int) -> None:
        _require(path.is_file(), f"official tasks metadata not found: {path}")
        offsets = array("Q")
        offset = 0
        with path.open("rb") as handle:
            for line in handle:
                _require(bool(line.strip()), f"blank line in official tasks metadata: {path}")
                offsets.append(offset)
                offset += len(line)
        _require(
            len(offsets) == int(expected_count),
            f"official task line count mismatch: expected {expected_count}, got {len(offsets)}",
        )
        self.path = path
        self.offsets = offsets
        self._cache: dict[int, str] = {}
        self._reverse_cache: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __iter__(self):
        return iter(range(len(self.offsets)))

    def __getitem__(self, task_index: int) -> str:
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise KeyError(task_index)
        if task_index < 0 or task_index >= len(self.offsets):
            raise KeyError(task_index)
        cached = self._cache.get(task_index)
        if cached is not None:
            return cached
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[task_index])
            raw = handle.readline()
        try:
            record = json.loads(raw, strict=False)
        except Exception as exc:
            raise OfficialDataContractError(
                f"cannot parse official task record {task_index}: {exc}"
            ) from exc
        _require(isinstance(record, dict), "official task record must be an object")
        _require(
            record.get("task_index") == task_index,
            f"official task line/index mismatch at {task_index}",
        )
        task = record.get("task")
        _require(isinstance(task, str) and bool(task), f"empty official task {task_index}")
        self._cache[task_index] = task
        self._reverse_cache[task] = task_index
        return task


class _ReverseOfficialTasks(Mapping[str, int]):
    """Lazy reverse view; normal training never needs a full prompt dictionary."""

    def __init__(self, forward: _IndexedOfficialTasks) -> None:
        self.forward = forward

    def __len__(self) -> int:
        return len(self.forward)

    def __iter__(self):
        for index in self.forward:
            yield self.forward[index]

    def __getitem__(self, task: str) -> int:
        cached = self.forward._reverse_cache.get(task)  # noqa: SLF001
        if cached is not None:
            return cached
        # This compatibility path is intentionally lazy.  Read-only policy
        # training indexes tasks in the forward direction, so it is not paid
        # during normal initialization or sampling.
        for index in self.forward:
            if self.forward[index] == task:
                return index
        raise KeyError(task)


def _load_selected_jsonl_records(
    path: Path,
    selected_episode_ids: tuple[int, ...],
    *,
    record_name: str,
) -> dict[int, dict[str, Any]]:
    _require(path.is_file(), f"official {record_name} metadata not found: {path}")
    wanted = frozenset(selected_episode_ids)
    _require(bool(wanted), f"official {record_name} selection is empty")
    maximum = max(wanted)
    records: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line_index, raw in enumerate(handle):
            if line_index > maximum:
                break
            if line_index not in wanted:
                continue
            try:
                record = json.loads(raw, strict=False)
            except Exception as exc:
                raise OfficialDataContractError(
                    f"cannot parse official {record_name} record {line_index}: {exc}"
                ) from exc
            _require(isinstance(record, dict), f"official {record_name} record must be an object")
            _require(
                record.get("episode_index") == line_index,
                f"official {record_name} line/index mismatch at {line_index}",
            )
            records[line_index] = record
    missing = wanted.difference(records)
    _require(
        not missing,
        f"official {record_name} metadata lacks selected episodes: {sorted(missing)[:8]}",
    )
    return records


class _ExplicitEpisodeMetadata(_NativeLeRobotDatasetMetadata):
    """Native-compatible metadata restricted to audited explicit episodes."""

    def load_metadata(self) -> None:
        context = _active_explicit_loader()
        root = _validate_metadata_root(self.root, context)
        selection = context.selection
        self.info = load_info(root)
        for key, expected in EXPECTED_DATASET_FACTS.items():
            _require(self.info.get(key) == expected, f"official info fact mismatch for {key}")

        tasks = _IndexedOfficialTasks(
            root / "meta/tasks.jsonl",
            expected_count=int(EXPECTED_DATASET_FACTS["total_tasks"]),
        )
        self.tasks = tasks
        self.task_to_task_index = _ReverseOfficialTasks(tasks)
        if (root / "annotations").exists():
            self.annotations = load_annotations(root)

        self.episodes = _load_selected_jsonl_records(
            root / "meta/episodes.jsonl",
            selection.episode_ids,
            record_name="episodes",
        )
        raw_episode_stats = _load_selected_jsonl_records(
            root / "meta/episodes_stats.jsonl",
            selection.episode_ids,
            record_name="episodes_stats",
        )
        self.episodes_stats = {
            episode_index: cast_stats_to_numpy(record["stats"])
            for episode_index, record in raw_episode_stats.items()
        }
        self.stats = aggregate_stats(list(self.episodes_stats.values()))
        audit = {
            "metadata_mode": "selected_records_only",
            "episodes_records_loaded": len(self.episodes),
            "episode_stats_records_loaded": len(self.episodes_stats),
            "task_json_records_parsed_at_init": 0,
            "task_json_line_offsets_indexed": len(tasks),
        }
        self.explicit_load_audit = audit
        context.metadata_audits.append(dict(audit))


@contextlib.contextmanager
def temporary_environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@contextlib.contextmanager
def _temporary_explicit_episode_native_loader(
    selection: NativeSplitEpisodeSelection,
) -> Iterator[_ExplicitLoaderContext]:
    """Narrow native construction inside a serialized, exception-safe scope.

    The native classes resolve these three symbols only during synchronous
    dataset construction.  Every symbol and the active context are restored in
    ``finally`` before the dataset is returned or an exception propagates, so
    DataLoader workers and later native users never observe the patch.
    """

    from fastwam.datasets.lerobot import base_lerobot_dataset as base_module
    from fastwam.datasets.lerobot.lerobot import lerobot_dataset as lerobot_module

    _require(
        isinstance(selection, NativeSplitEpisodeSelection),
        "explicit loader requires a NativeSplitEpisodeSelection",
    )
    manifest_episode_set = frozenset(
        episode_index
        for start, end in EXPECTED_TASK_EPISODE_RANGES.values()
        for episode_index in range(start, end + 1)
    )

    with _EXPLICIT_LOADER_PATCH_LOCK:
        original_base_metadata = base_module.LeRobotDatasetMetadata
        original_multi_dataset = base_module.MultiLeRobotDataset
        original_lerobot_metadata = lerobot_module.LeRobotDatasetMetadata
        multi_signature = inspect.signature(original_multi_dataset)

        def explicit_multi_dataset(*args, **kwargs):
            bound = multi_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            dataset_dirs = list(bound.arguments.get("dataset_dirs") or ())
            _require(
                len(dataset_dirs) == 1,
                "explicit official loader requires exactly one dataset root",
            )
            resolved_root = Path(dataset_dirs[0]).expanduser().resolve()
            _require(
                resolved_root == selection.dataset_root,
                f"native multi-dataset root {resolved_root} differs from verified root "
                f"{selection.dataset_root}",
            )
            episode_map = bound.arguments.get("episodes")
            _require(isinstance(episode_map, Mapping), "native split did not provide episode IDs")
            _require(
                len(episode_map) == 1,
                "explicit official loader received multiple native episode sets",
            )
            try:
                native_split_ids = tuple(int(value) for value in next(iter(episode_map.values())))
            except Exception as exc:
                raise OfficialDataContractError(
                    f"cannot read native split episode IDs: {exc}"
                ) from exc
            actual_intersection = tuple(
                episode_index
                for episode_index in native_split_ids
                if episode_index in manifest_episode_set
            )
            _require(
                actual_intersection == selection.episode_ids,
                "native train/validation split intersection differs from the audited selection",
            )
            dataset_key = next(iter(episode_map))
            bound.arguments["episodes"] = {
                dataset_key: list(selection.episode_ids)
            }
            return original_multi_dataset(*bound.args, **bound.kwargs)

        context = _ExplicitLoaderContext(selection=selection)
        token = _ACTIVE_EXPLICIT_LOADER.set(context)
        base_module.LeRobotDatasetMetadata = _InfoOnlyOfficialMetadata
        base_module.MultiLeRobotDataset = explicit_multi_dataset
        lerobot_module.LeRobotDatasetMetadata = _ExplicitEpisodeMetadata
        try:
            yield context
        finally:
            lerobot_module.LeRobotDatasetMetadata = original_lerobot_metadata
            base_module.MultiLeRobotDataset = original_multi_dataset
            base_module.LeRobotDatasetMetadata = original_base_metadata
            _ACTIVE_EXPLICIT_LOADER.reset(token)


def compose_robotwin_config(task_config: str = DEFAULT_TASK_CONFIG):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        return compose(
            config_name="sim_robotwin.yaml",
            overrides=[f"task={task_config}"],
        )


def dtype_from_name(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32", "no"}:
        return torch.float32
    raise ValueError(f"unsupported dtype name: {name!r}")


def instantiate_release_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    load_text_encoder: bool = True,
    task_config: str = DEFAULT_TASK_CONFIG,
    model_base_path: str | Path | None = None,
    compute_checkpoint_sha256: bool = False,
):
    """Instantiate structure, then strictly overwrite it with release weights."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"release checkpoint not found: {checkpoint}")
    cfg = compose_robotwin_config(task_config)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = bool(load_text_encoder)
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    if model_base_path is None:
        if checkpoint.parent.name == "fastwam_release":
            model_base = checkpoint.parent.parent
        else:
            model_base = PROJECT_ROOT / "checkpoints"
    else:
        model_base = Path(model_base_path).expanduser().resolve()
    if not Path(model_base).is_dir():
        raise FileNotFoundError(f"model component base not found: {model_base}")
    with temporary_environment("DIFFSYNTH_MODEL_BASE_PATH", str(model_base)):
        model = instantiate(model_cfg, model_dtype=dtype, device=str(device))
    audit = strict_load_release_checkpoint(
        model,
        checkpoint,
        compute_sha256=compute_checkpoint_sha256,
    )
    model.eval()
    return model, cfg, audit


def instantiate_official_dataset(
    cfg,
    *,
    dataset_root: str | Path,
    dataset_stats_path: str | Path | None,
    text_cache_dir: str | Path | None,
    model_for_on_the_fly_text=None,
    manifest_path: str | Path | None = None,
    episode_selection_mode: str = "native_99pct",
    allow_compute_dataset_stats: bool = False,
):
    """Instantiate native data for only the audited three-task split.

    On-the-fly text encoding is intended only for num_workers=0 smoke. A formal
    run must provide a precomputed cache so workers never share a CUDA model.
    Native ``RobotVideoDataset``/``_get``/processor/transform semantics are
    retained, but its construction is scoped through an experiment-local
    explicit-episode loader so it never opens the other ~25.8k episodes.
    """

    _require(
        episode_selection_mode in EPISODE_SELECTION_MODES,
        f"episode_selection_mode must be one of {EPISODE_SELECTION_MODES}",
    )
    root = Path(dataset_root).expanduser().resolve()
    stats = (
        None
        if dataset_stats_path is None
        else Path(dataset_stats_path).expanduser().resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(f"official dataset root not found: {root}")
    if stats is None and not allow_compute_dataset_stats:
        raise FileNotFoundError(
            "official dataset stats are required unless allow_compute_dataset_stats=true"
        )
    if stats is not None and not stats.is_file():
        raise FileNotFoundError(f"official dataset stats not found: {stats}")
    manifest = (
        DEFAULT_OFFICIAL_MANIFEST
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    verified_manifest = verify_official_task_manifest(manifest, root)
    data_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.data.train, resolve=True)
    )
    data_cfg.dataset_dirs = [str(root)]
    data_cfg.pretrained_norm_stats = None if stats is None else str(stats)
    data_cfg.text_embedding_cache_dir = (
        None
        if text_cache_dir is None
        else str(Path(text_cache_dir).expanduser().resolve())
    )
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset

    seed_parameter = inspect.signature(BaseLerobotDataset.__init__).parameters.get("seed")
    if seed_parameter is None or seed_parameter.default is inspect.Parameter.empty:
        raise OfficialDataContractError(
            "native BaseLerobotDataset no longer exposes a default split seed"
        )
    native_seed = seed_parameter.default
    if episode_selection_mode == "full_550_per_task":
        _require(
            bool(data_cfg.is_training_set),
            "full_550_per_task is a Stage1 training selection, not a validation split",
        )
        # The native constructor must expose its canonical full episode order;
        # the scoped explicit loader then replaces it with the audited 1,650
        # episode list.  This is selection, not data resampling.
        data_cfg.val_set_proportion = 0.0
        selection = select_official_full_550_per_task(
            verified_manifest,
            seed=native_seed,
        )
    else:
        selection = select_official_episodes_from_native_split(
            verified_manifest,
            val_set_proportion=float(data_cfg.val_set_proportion),
            is_training_set=bool(data_cfg.is_training_set),
            seed=native_seed,
        )
    with _temporary_explicit_episode_native_loader(selection) as loader_context:
        dataset = instantiate(data_cfg)

    lerobot = getattr(dataset, "lerobot_dataset", None)
    multi_dataset = getattr(lerobot, "multi_dataset", None)
    inner_datasets = getattr(multi_dataset, "_datasets", None)
    _require(
        isinstance(inner_datasets, (list, tuple)) and len(inner_datasets) == 1,
        "explicit native loader did not construct exactly one inner dataset",
    )
    loaded_episodes = tuple(int(value) for value in inner_datasets[0].episodes)
    _require(
        loaded_episodes == selection.episode_ids,
        "loaded native episodes differ from the audited explicit selection",
    )
    _require(
        len(dataset) < int(EXPECTED_DATASET_FACTS["total_frames"]),
        "explicit native loader unexpectedly exposed the full release frame count",
    )
    dataset._official_explicit_episode_selection = selection  # noqa: SLF001
    dataset._verified_official_manifest = verified_manifest  # noqa: SLF001
    dataset._official_explicit_loader_patch_scope_exited = True  # noqa: SLF001
    dataset._official_explicit_metadata_audit = {  # noqa: SLF001
        "constructor_metadata_instances": len(loader_context.metadata_audits),
        "instances": list(loader_context.metadata_audits),
    }
    if text_cache_dir is None:
        if model_for_on_the_fly_text is None:
            raise FileNotFoundError(
                "official text cache is required for formal training; "
                "only smoke may pass a loaded model_for_on_the_fly_text"
            )
        if getattr(model_for_on_the_fly_text, "text_encoder", None) is None:
            raise ValueError("on-the-fly text smoke requires a loaded text encoder")
        model = model_for_on_the_fly_text

        def encode_instead_of_cache(_dataset, prompt: str):
            with torch.no_grad():
                context, mask = model.encode_prompt(prompt)
            return (
                context[0].detach().to(device="cpu"),
                mask[0].detach().to(device="cpu", dtype=torch.bool),
            )

        dataset._get_cached_text_context = types.MethodType(  # noqa: SLF001
            encode_instead_of_cache, dataset
        )
    return dataset


def instantiate_native_paired_action_dataset(
    cfg,
    *,
    dataset_root: str | Path,
    dataset_stats_path: str | Path,
    text_cache_dir: str | Path | None,
    model_for_on_the_fly_text=None,
    manifest_path: str | Path,
    audit_path: str | Path,
    state_bank_path: str | Path,
    expected_state_bank_sha256: str | None,
    split: str = "train",
    expected_tasks=None,
    require_full_protocol_counts: bool = False,
    state_bank_states_per_trajectory: int = 8,
    state_bank_sampling_algorithm: str = "sha256_rank_endpoint_safe_v1",
    state_bank_sampling_version: int = 1,
    state_bank_sampling_seed: int = 42,
):
    """Build the native 50 Hz RobotVideoDataset, then strict four-scene wrapper.

    This path deliberately does not reuse the official-release metadata patch:
    the paired root is an independent LeRobot-v2.1 dataset with its own
    collector manifest/audit.  All episodes are exposed in canonical order;
    ``NativePairedActionDataset`` applies the manifest split and guards native
    random read recovery.
    """

    from experiments.robotwin.policy_content_adapter.data import (
        NativePairedActionDataset,
    )

    root = Path(dataset_root).expanduser().resolve()
    stats = Path(dataset_stats_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    audit = Path(audit_path).expanduser().resolve()
    state_bank = Path(state_bank_path).expanduser().resolve()
    if str(split) != "train":
        raise ValueError("shared Policy state-bank action loading supports train only")
    if not root.is_dir():
        raise FileNotFoundError(f"paired native action root not found: {root}")
    if not stats.is_file():
        raise FileNotFoundError(f"paired native dataset stats not found: {stats}")
    if not manifest.is_file():
        raise FileNotFoundError(f"paired action manifest not found: {manifest}")
    if not audit.is_file():
        raise FileNotFoundError(f"paired action audit not found: {audit}")
    if not state_bank.is_file():
        raise FileNotFoundError(f"paired state bank not found: {state_bank}")

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    data_cfg.dataset_dirs = [str(root)]
    data_cfg.pretrained_norm_stats = str(stats)
    data_cfg.text_embedding_cache_dir = (
        None
        if text_cache_dir is None
        else str(Path(text_cache_dir).expanduser().resolve())
    )
    # The Policy manifest, rather than FastWAM's global random 99/1 splitter,
    # owns train/val/test membership.  Load every paired episode so the strict
    # wrapper can select a manifest split without accidental omission.
    data_cfg.val_set_proportion = 0.0
    data_cfg.is_training_set = True
    base_dataset = instantiate(data_cfg)

    if text_cache_dir is None:
        if model_for_on_the_fly_text is None:
            raise FileNotFoundError(
                "paired text cache is required for formal training; only smoke may "
                "pass model_for_on_the_fly_text"
            )
        if getattr(model_for_on_the_fly_text, "text_encoder", None) is None:
            raise ValueError("on-the-fly paired text smoke requires a loaded text encoder")
        model = model_for_on_the_fly_text

        def encode_instead_of_cache(_dataset, prompt: str):
            with torch.no_grad():
                context, mask = model.encode_prompt(prompt)
            return (
                context[0].detach().to(device="cpu"),
                mask[0].detach().to(device="cpu", dtype=torch.bool),
            )

        base_dataset._get_cached_text_context = types.MethodType(  # noqa: SLF001
            encode_instead_of_cache, base_dataset
        )

    return NativePairedActionDataset(
        base_dataset,
        dataset_root=root,
        manifest_path=manifest,
        audit_path=audit,
        state_bank_path=state_bank,
        expected_state_bank_sha256=expected_state_bank_sha256,
        split=split,
        expected_tasks=expected_tasks,
        require_full_protocol_counts=require_full_protocol_counts,
        state_bank_states_per_trajectory=state_bank_states_per_trajectory,
        state_bank_sampling_algorithm=state_bank_sampling_algorithm,
        state_bank_sampling_version=state_bank_sampling_version,
        state_bank_sampling_seed=state_bank_sampling_seed,
    )


def move_batch_to_cpu(batch: Any) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu()
    if isinstance(batch, dict):
        return {key: move_batch_to_cpu(value) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_cpu(value) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_cpu(value) for value in batch)
    return batch


__all__ = [
    "DEFAULT_OFFICIAL_MANIFEST",
    "DEFAULT_TASK_CONFIG",
    "PROJECT_ROOT",
    "SRC_ROOT",
    "audit_local_fastwam_source",
    "compose_robotwin_config",
    "dtype_from_name",
    "instantiate_official_dataset",
    "instantiate_native_paired_action_dataset",
    "instantiate_release_model",
    "move_batch_to_cpu",
    "temporary_environment",
]
