"""Adapt the audited FastWAM C/R1/R2/R3 observations to Motus.

The adapter reuses only current RGB observations and physical-state grouping.
It neither interpolates actions nor claims that the 50 Hz / 32-step action
targets match Motus's native 30 Hz / 16-step policy training contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .protocol import (
    CAMERAS,
    MOTUS_IMAGE_HEIGHT,
    MOTUS_IMAGE_WIDTH,
    PAIRED_SCENE_COUNT,
    PAIRED_STATE_COUNT,
    PAIRED_VIEW_COUNT,
    PROTOCOL_ID,
    TASKS,
    VARIANTS,
)


MANIFEST_SCHEMA = "motus_policy_paired_observation_manifest"
MANIFEST_VERSION = 1
SOURCE_BINDING_KIND = "policy_release_native50hz_paired_binding"
SOURCE_PROTOCOL_ID = "policy_native50hz_four_scene_v1"


class PairedDataError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PairedDataError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedDataError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(
                    isinstance(value, dict),
                    f"{path}:{line_number} is not an object",
                )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedDataError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _verify_bound_file(identity: Mapping[str, Any], *, name: str) -> Path:
    path = Path(str(identity.get("path", "")))
    _require(path.is_file(), f"bound {name} file is missing: {path}")
    expected_size = int(identity.get("size_bytes", -1))
    _require(path.stat().st_size == expected_size, f"bound {name} size changed")
    expected_sha = str(identity.get("sha256", ""))
    _require(len(expected_sha) == 64, f"bound {name} SHA is invalid")
    _require(sha256_file(path) == expected_sha, f"bound {name} SHA changed")
    return path


def _video_path(root: Path, episode_index: int, camera: str) -> Path:
    chunk = episode_index // 1000
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{camera}"
        / f"episode_{episode_index:06d}.mp4"
    )


def _parquet_path(root: Path, episode_index: int) -> Path:
    return (
        root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def build_paired_observation_manifest(
    binding_path: str | Path,
    *,
    verify_source_paths: bool = True,
) -> dict[str, Any]:
    """Build the immutable 720-state Motus observation manifest."""

    binding_path = Path(binding_path).resolve()
    binding = _load_json(binding_path)
    _require(isinstance(binding, dict), "source binding must be a JSON object")
    _require(binding.get("status") == "PASS", "source paired binding is not PASS")
    _require(
        binding.get("kind") == SOURCE_BINDING_KIND,
        "source paired binding kind changed",
    )
    _require(
        binding.get("protocol_id") == SOURCE_PROTOCOL_ID,
        "source paired protocol changed",
    )

    paired = binding.get("paired_dataset")
    _require(isinstance(paired, dict), "paired_dataset is missing")
    _require(paired.get("scene_episode_count") == 600, "expected 600 scene episodes")
    _require(
        paired.get("physical_trajectory_count") == 150,
        "expected 150 physical trajectories",
    )
    _require(
        paired.get("train_physical_trajectory_count") == 90,
        "expected 90 train trajectories",
    )
    _require(
        paired.get("state_bank_anchor_count") == PAIRED_STATE_COUNT,
        "expected 720 paired train states",
    )
    _require(
        paired.get("scene_variants_per_state") == PAIRED_VIEW_COUNT,
        "expected four scene variants",
    )
    _require(tuple(paired.get("task_order", ())) == TASKS, "task order changed")

    meta = binding.get("meta_artifacts")
    _require(isinstance(meta, dict), "source meta artifact identities are missing")
    state_bank_path = _verify_bound_file(
        meta.get("policy_paired_state_bank", {}), name="state bank"
    )
    paired_contents_path = _verify_bound_file(
        meta.get("paired_contents", {}), name="paired contents"
    )
    state_bank = _load_json(state_bank_path)
    content_rows = _load_jsonl(paired_contents_path)
    _require(state_bank.get("schema") == "policy_paired_state_bank_v1", "state bank schema changed")
    _require(state_bank.get("split") == "train", "state bank is not the train split")
    _require(tuple(state_bank.get("variant_names", ())) == (
        "clean", "style_00_seed_0", "style_01_seed_1", "style_02_seed_2"
    ), "state-bank variant order changed")
    states = state_bank.get("states")
    _require(isinstance(states, list) and len(states) == PAIRED_STATE_COUNT, "state count changed")

    content_by_trajectory: dict[str, dict[str, Any]] = {}
    for row in content_rows:
        trajectory = str(row.get("physical_trajectory_id", ""))
        _require(trajectory and trajectory not in content_by_trajectory, "duplicate trajectory row")
        content_by_trajectory[trajectory] = row

    root = Path(str(paired.get("root", ""))).resolve()
    _require(root.is_dir(), f"paired dataset root is missing: {root}")
    records: list[dict[str, Any]] = []
    unique_episodes: set[int] = set()
    unique_files: set[str] = set()
    per_task: dict[str, int] = {task: 0 for task in TASKS}
    for state in states:
        _require(isinstance(state, dict), "state record is not an object")
        task = str(state.get("task", ""))
        trajectory = str(state.get("trajectory_id", ""))
        physical_state_id = str(state.get("physical_state_id", ""))
        frame_offset = int(state.get("frame_offset", -1))
        _require(task in TASKS, f"unexpected task {task!r}")
        row = content_by_trajectory.get(trajectory)
        _require(row is not None, f"no content row for {trajectory}")
        _require(row.get("task") == task, "state/content task mismatch")
        _require(row.get("split") == "train", "state points outside train split")
        _require(row.get("state_action_exactly_equal") is True, "four scenes are not exact-state paired")
        _require(0 <= frame_offset < int(row.get("equal_length", -1)), "state frame is out of bounds")
        episode_map = row.get("scene_variant_episode_indices")
        _require(isinstance(episode_map, dict) and tuple(episode_map) == VARIANTS, "variant map changed")
        views: list[dict[str, Any]] = []
        for variant in VARIANTS:
            episode_index = int(episode_map[variant])
            unique_episodes.add(episode_index)
            camera_paths: dict[str, str] = {}
            for camera in CAMERAS:
                path = _video_path(root, episode_index, camera)
                if verify_source_paths:
                    _require(path.is_file(), f"missing source video {path}")
                camera_paths[camera] = str(path)
                unique_files.add(str(path))
            parquet = _parquet_path(root, episode_index)
            if verify_source_paths:
                _require(parquet.is_file(), f"missing source parquet {parquet}")
            unique_files.add(str(parquet))
            views.append(
                {
                    "variant": variant,
                    "episode_index": episode_index,
                    "frame_offset": frame_offset,
                    "camera_paths": camera_paths,
                    "parquet_path": str(parquet),
                }
            )
        records.append(
            {
                "physical_state_id": physical_state_id,
                "physical_trajectory_id": trajectory,
                "task": task,
                "content_id": int(state.get("content_id", -1)),
                "frame_offset": frame_offset,
                "views": views,
            }
        )
        per_task[task] += 1

    _require(len(records) == PAIRED_STATE_COUNT, "record count changed")
    _require(len(unique_episodes) == 360, "train scene episode inventory changed")
    _require(len(unique_files) == 1440, "selected train file inventory changed")
    _require(per_task == {task: 240 for task in TASKS}, "per-task state counts changed")
    selected = binding.get("selected_train_artifacts")
    _require(isinstance(selected, dict), "selected train artifact binding is missing")
    _require(selected.get("file_count") == 1440, "source selected file count changed")
    _require(selected.get("episode_count") == 360, "source selected episode count changed")

    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "purpose": "motus_current_observation_contrastive_only",
        "action_supervision_allowed": False,
        "temporal_conversion": "none",
        "source_binding": {
            "path": str(binding_path),
            "size_bytes": binding_path.stat().st_size,
            "sha256": sha256_file(binding_path),
        },
        "source_selected_train_artifacts": selected,
        "source_state_bank": {
            "path": str(state_bank_path),
            "size_bytes": state_bank_path.stat().st_size,
            "sha256": sha256_file(state_bank_path),
        },
        "source_paired_contents": {
            "path": str(paired_contents_path),
            "size_bytes": paired_contents_path.stat().st_size,
            "sha256": sha256_file(paired_contents_path),
        },
        "source_dataset_root": str(root),
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "cameras": list(CAMERAS),
        "motus_preprocessing": {
            "layout": "head_full_top_left_right_half_bottom",
            "target_height": MOTUS_IMAGE_HEIGHT,
            "target_width": MOTUS_IMAGE_WIDTH,
            "resize": "opencv_linear_aspect_preserving_center_zero_pad",
        },
        "counts": {
            "physical_states": len(records),
            "views_per_state": PAIRED_VIEW_COUNT,
            "scene_views": len(records) * PAIRED_VIEW_COUNT,
            "selected_scene_episodes": len(unique_episodes),
            "selected_files": len(unique_files),
            "per_task_physical_states": per_task,
        },
        "record_inventory_sha256": canonical_json_sha256(records),
        "records": records,
    }


def validate_paired_observation_manifest(
    manifest: Mapping[str, Any], *, verify_source_paths: bool = False
) -> dict[str, Any]:
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema changed")
    _require(manifest.get("schema_version") == MANIFEST_VERSION, "manifest version changed")
    _require(manifest.get("status") == "PASS", "manifest is not PASS")
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "protocol id changed")
    _require(manifest.get("action_supervision_allowed") is False, "paired observations may not supervise Motus actions")
    _require(tuple(manifest.get("tasks", ())) == TASKS, "task order changed")
    _require(tuple(manifest.get("variants", ())) == VARIANTS, "variant order changed")
    _require(tuple(manifest.get("cameras", ())) == CAMERAS, "camera order changed")
    records = manifest.get("records")
    _require(isinstance(records, list) and len(records) == PAIRED_STATE_COUNT, "manifest must contain 720 states")
    _require(
        canonical_json_sha256(records) == manifest.get("record_inventory_sha256"),
        "record inventory SHA changed",
    )
    state_ids: set[str] = set()
    counts = {task: 0 for task in TASKS}
    for record in records:
        state_id = str(record.get("physical_state_id", ""))
        _require(state_id and state_id not in state_ids, "duplicate physical state")
        state_ids.add(state_id)
        task = str(record.get("task", ""))
        _require(task in TASKS, "record task is invalid")
        counts[task] += 1
        views = record.get("views")
        _require(isinstance(views, list) and len(views) == PAIRED_VIEW_COUNT, "state does not have four views")
        _require(tuple(view.get("variant") for view in views) == VARIANTS, "view order changed")
        for view in views:
            paths = view.get("camera_paths")
            _require(isinstance(paths, dict) and tuple(paths) == CAMERAS, "camera path order changed")
            if verify_source_paths:
                for path in paths.values():
                    _require(Path(path).is_file(), f"source video disappeared: {path}")
                _require(Path(str(view.get("parquet_path", ""))).is_file(), "source parquet disappeared")
    _require(counts == {task: 240 for task in TASKS}, "per-task state count changed")
    return {
        "status": "PASS",
        "physical_states": len(records),
        "scene_views": len(records) * PAIRED_VIEW_COUNT,
        "record_inventory_sha256": manifest["record_inventory_sha256"],
    }


def resize_with_padding(
    frame: np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be HWC RGB")
    target_height, target_width = target_size
    height, width = frame.shape[:2]
    scale = min(target_height / height, target_width / width)
    new_height, new_width = int(height * scale), int(width * scale)
    resized = cv2.resize(frame, (new_width, new_height))
    output = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    output[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return output


def build_motus_t_shaped_frame(
    head: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    *,
    target_size: tuple[int, int] = (MOTUS_IMAGE_HEIGHT, MOTUS_IMAGE_WIDTH),
) -> np.ndarray:
    """Match the author converter followed by Motus resize-with-padding."""

    if head.ndim != 3 or head.shape[2] != 3:
        raise ValueError("head frame must be HWC RGB")
    if left_wrist.shape != head.shape or right_wrist.shape != head.shape:
        raise ValueError("three camera frames must have identical shapes")
    height, width = head.shape[:2]
    left = cv2.resize(left_wrist, (width // 2, height // 2))
    right = cv2.resize(right_wrist, (width // 2, height // 2))
    composite = np.vstack([head, np.hstack([left, right])])
    return resize_with_padding(composite, target_size)


class _PyAVReaderCache:
    """Sequential software AV1 decoding for the ordered state-bank access.

    Motus's bundled decord and OpenCV builds cannot decode the AV1 files in
    the paired LeRobot export on this host.  PyAV is linked with libdav1d and
    reliably decodes them in software.  State offsets are sorted within each
    trajectory, so retaining one iterator per camera avoids reopening a video
    for every selected state.
    """

    def __init__(self, max_open: int = 32) -> None:
        self.max_open = int(max_open)
        self._readers: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @staticmethod
    def _open(path: str) -> dict[str, Any]:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required to decode paired AV1 videos") from exc
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return {
            "container": container,
            "iterator": iter(container.decode(stream)),
            "next_index": 0,
        }

    @staticmethod
    def _close(entry: Mapping[str, Any]) -> None:
        try:
            entry["container"].close()
        except Exception:
            pass

    def __call__(self, path: str, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise IndexError("frame index must be non-negative")
        entry = self._readers.pop(path, None)
        if entry is None:
            entry = self._open(path)
        elif frame_index < int(entry["next_index"]):
            self._close(entry)
            entry = self._open(path)
        self._readers[path] = entry
        while len(self._readers) > self.max_open:
            _, evicted = self._readers.popitem(last=False)
            self._close(evicted)
        decoded = None
        try:
            while int(entry["next_index"]) <= frame_index:
                frame = next(entry["iterator"])
                current = int(entry["next_index"])
                entry["next_index"] = current + 1
                if current == frame_index:
                    decoded = frame.to_ndarray(format="rgb24")
                    break
        except StopIteration as exc:
            raise IndexError(f"frame {frame_index} outside decoded video {path}") from exc
        if decoded is None:
            raise PairedDataError(f"failed to decode frame {frame_index} from {path}")
        if decoded.dtype != np.uint8 or decoded.ndim != 3 or decoded.shape[2] != 3:
            raise PairedDataError(f"decoded frame has invalid format: {path}")
        return decoded


class MotusPairedObservationDataset(Dataset):
    """One item is an ordered C/R1/R2/R3 current-observation group."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        decoder: Callable[[str, int], np.ndarray] | None = None,
        verify_source_paths: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = _load_json(self.manifest_path)
        validate_paired_observation_manifest(
            self.manifest, verify_source_paths=verify_source_paths
        )
        self.records = self.manifest["records"]
        self.decoder = decoder if decoder is not None else _PyAVReaderCache()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        images: list[torch.Tensor] = []
        for view in record["views"]:
            paths = view["camera_paths"]
            frame_index = int(view["frame_offset"])
            head = self.decoder(paths["cam_high"], frame_index)
            left = self.decoder(paths["cam_left_wrist"], frame_index)
            right = self.decoder(paths["cam_right_wrist"], frame_index)
            composite = build_motus_t_shaped_frame(head, left, right)
            images.append(
                torch.from_numpy(composite.copy()).permute(2, 0, 1).float() / 255.0
            )
        return {
            "images": torch.stack(images, dim=0),
            "physical_state_id": record["physical_state_id"],
            "task": record["task"],
            "variants": tuple(VARIANTS),
        }


def collate_paired_groups(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("paired batch is empty")
    images = torch.stack([item["images"] for item in batch], dim=0)
    if images.ndim != 5 or images.shape[1] != PAIRED_VIEW_COUNT:
        raise ValueError("paired images must be [G,4,3,H,W]")
    return {
        "images": images,
        "physical_state_ids": [
            item["physical_state_id"]
            for item in batch
            for _ in range(PAIRED_VIEW_COUNT)
        ],
        "task_ids": [
            item["task"] for item in batch for _ in range(PAIRED_VIEW_COUNT)
        ],
        "variants": list(VARIANTS) * len(batch),
    }


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--binding", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--skip-source-existence", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--verify-source-paths", action="store_true")
    args = parser.parse_args()

    if args.command == "build":
        manifest = build_paired_observation_manifest(
            args.binding,
            verify_source_paths=not args.skip_source_existence,
        )
        validate_paired_observation_manifest(
            manifest, verify_source_paths=not args.skip_source_existence
        )
        output = Path(args.output).resolve()
        _write_create_only(output, manifest)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "path": str(output),
                    "sha256": sha256_file(output),
                    "physical_states": PAIRED_STATE_COUNT,
                    "scene_views": PAIRED_SCENE_COUNT,
                },
                sort_keys=True,
            )
        )
        return
    manifest_path = Path(args.manifest).resolve()
    result = validate_paired_observation_manifest(
        _load_json(manifest_path), verify_source_paths=args.verify_source_paths
    )
    result.update(path=str(manifest_path), sha256=sha256_file(manifest_path))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
