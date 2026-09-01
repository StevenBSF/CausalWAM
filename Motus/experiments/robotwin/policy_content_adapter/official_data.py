"""Deterministic Motus view of the three-task FastWAM RoboTwin release.

The source trajectories are native 50 Hz.  Selecting every five source frames
produces the same 10 Hz physical action cadence as Motus's author protocol
(30 Hz source, stride three), without interpolation.  Sixteen action targets
span 1.6 seconds and eight video targets are sampled every two action points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .paired_data import (
    build_motus_t_shaped_frame,
    canonical_json_sha256,
    sha256_file,
)
from .protocol import (
    CAMERAS,
    MOTUS_ACTION_CHUNK,
    MOTUS_ACTION_DIM,
    PROTOCOL_ID,
    TASKS,
)


MANIFEST_SCHEMA = "motus_policy_official_three_task_manifest"
MANIFEST_VERSION = 4
SOURCE_FPS = 50
ACTION_STRIDE = 5
ACTION_HZ = SOURCE_FPS // ACTION_STRIDE
VIDEO_ACTION_RATIO = 2
VIDEO_FRAMES = 8
PHYSICAL_SPAN_FRAMES = MOTUS_ACTION_CHUNK * ACTION_STRIDE
SAMPLES_PER_EPISODE = 10

TASK_DOMAIN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "place_a2b_left": {
        "clean": (11000, 11049),
        "official_random": (11050, 11549),
    },
    "open_microwave": {
        "clean": (9350, 9399),
        "official_random": (9400, 9899),
    },
    "move_stapler_pad": {
        "clean": (8250, 8299),
        "official_random": (8300, 8799),
    },
}

MOTUS_META_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)

TASK_INSTRUCTIONS = {
    "place_a2b_left": "use appropriate arm to place object A on the left of object B",
    "open_microwave": "Use one arm to open the microwave.",
    "move_stapler_pad": "use appropriate arm to move the stapler to a colored mat",
}
TASK_PROMPTS = {
    task: MOTUS_META_PREFIX + instruction
    for task, instruction in TASK_INSTRUCTIONS.items()
}

EXPECTED_INFO = {
    "size_bytes": 4601,
    "sha256": "441a98fffe047bb642dba617bd6d89bbe313dd1379858744844536d73c493609",
}
EXPECTED_OFFICIAL_PARTITION = {
    "size_bytes": 3704,
    "sha256": "15f1d60e6f662f047385069ec1afe4715f69aeb3137419f9b4f37a811ec55126",
}


class OfficialDataError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialDataError(message)


def _identity(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_expected(path: Path, expected: Mapping[str, Any], name: str) -> dict[str, Any]:
    identity = _identity(path)
    _require(identity["size_bytes"] == expected["size_bytes"], f"{name} size changed")
    _require(identity["sha256"] == expected["sha256"], f"{name} SHA changed")
    return identity


def _episode_paths(root: Path, episode_index: int) -> tuple[Path, dict[str, Path]]:
    chunk = episode_index // 1000
    parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    videos = {
        camera: root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{camera}"
        / f"episode_{episode_index:06d}.mp4"
        for camera in CAMERAS
    }
    return parquet, videos


def _load_stats(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    identity = _identity(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    stats = value.get("robotwin2")
    _require(isinstance(stats, dict), "robotwin2 normalization stats are missing")
    minimum = np.asarray(stats.get("min"), dtype=np.float32)
    maximum = np.asarray(stats.get("max"), dtype=np.float32)
    _require(minimum.shape == (MOTUS_ACTION_DIM,), "normalization minimum shape changed")
    _require(maximum.shape == (MOTUS_ACTION_DIM,), "normalization maximum shape changed")
    _require(np.isfinite(minimum).all() and np.isfinite(maximum).all(), "normalization stats are non-finite")
    _require(np.all(maximum > minimum), "normalization ranges must be positive")
    return minimum, maximum, identity


def build_official_manifest(
    *,
    dataset_root: str | Path,
    official_partition_manifest: str | Path,
    normalization_stats: str | Path,
    verify_episode_files: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    _require(root.is_dir(), f"official dataset root is missing: {root}")
    info_identity = _verify_expected(
        root / "meta" / "info.json", EXPECTED_INFO, "release info"
    )
    partition_path = Path(official_partition_manifest).resolve()
    partition_identity = _verify_expected(
        partition_path, EXPECTED_OFFICIAL_PARTITION, "official partition"
    )
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    _require(tuple(partition.get("task_order", ())) == TASKS, "partition task order changed")
    _, _, stats_identity = _load_stats(Path(normalization_stats).resolve())

    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {
        task: {"clean": 0, "official_random": 0} for task in TASKS
    }
    for task in TASKS:
        for domain in ("clean", "official_random"):
            start, end = TASK_DOMAIN_RANGES[task][domain]
            for episode_index in range(start, end + 1):
                parquet, videos = _episode_paths(root, episode_index)
                if verify_episode_files:
                    _require(parquet.is_file(), f"missing parquet {parquet}")
                    for path in videos.values():
                        _require(path.is_file(), f"missing video {path}")
                metadata = pq.ParquetFile(parquet).metadata
                length = int(metadata.num_rows)
                _require(
                    length > PHYSICAL_SPAN_FRAMES,
                    f"episode {episode_index} is too short for Motus: {length}",
                )
                records.append(
                    {
                        "task": task,
                        "domain": domain,
                        "episode_index": episode_index,
                        "length": length,
                        "eligible_condition_frames": length - PHYSICAL_SPAN_FRAMES,
                        "parquet_path": str(parquet),
                        "camera_paths": {
                            camera: str(videos[camera]) for camera in CAMERAS
                        },
                        "prompt": TASK_PROMPTS[task],
                    }
                )
                counts[task][domain] += 1
    _require(len(records) == 1650, "official episode count must be 1650")
    _require(
        counts
        == {
            task: {"clean": 50, "official_random": 500}
            for task in TASKS
        },
        "official task/domain counts changed",
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "dataset_root": str(root),
        "source_release_info": info_identity,
        "source_partition_manifest": partition_identity,
        "normalization_stats": stats_identity,
        "action_value_contract": {
            "training": "raw_joint_action_vector",
            "state_source": "action_column_matching_motus_qpos_pt",
            "future_action_source": "action_column_matching_motus_qpos_pt",
            "fastwam_observation_state_used": False,
            "deployment": "raw_state_and_raw_predicted_actions",
            "stats_loaded_but_unused_by_author_deploy_path": True,
        },
        "tasks": list(TASKS),
        "task_domain_ranges": {
            task: {
                domain: list(bounds)
                for domain, bounds in TASK_DOMAIN_RANGES[task].items()
            }
            for task in TASKS
        },
        "counts": {
            "episodes": len(records),
            "per_task_domain": counts,
            "virtual_samples_per_epoch": len(records) * SAMPLES_PER_EPISODE,
        },
        "temporal_contract": {
            "source_fps": SOURCE_FPS,
            "source_stride": ACTION_STRIDE,
            "action_hz": ACTION_HZ,
            "action_chunk": MOTUS_ACTION_CHUNK,
            "action_dim": MOTUS_ACTION_DIM,
            "video_frames": VIDEO_FRAMES,
            "video_action_ratio": VIDEO_ACTION_RATIO,
            "physical_span_seconds": PHYSICAL_SPAN_FRAMES / SOURCE_FPS,
            "interpolation": False,
            "equivalence": "50Hz_stride5_equals_author_30Hz_stride3_at_10Hz",
        },
        "camera_contract": {
            "source_cameras": list(CAMERAS),
            "layout": "head_full_top_left_right_half_bottom",
            "motus_size_hw": [384, 320],
        },
        "record_inventory_sha256": canonical_json_sha256(records),
        "records": records,
    }


def validate_official_manifest(manifest: Mapping[str, Any], *, verify_paths: bool = False) -> dict[str, Any]:
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "official manifest schema changed")
    _require(manifest.get("schema_version") == MANIFEST_VERSION, "official manifest version changed")
    _require(manifest.get("status") == "PASS", "official manifest is not PASS")
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "official protocol changed")
    _require(tuple(manifest.get("tasks", ())) == TASKS, "official task order changed")
    temporal = manifest.get("temporal_contract", {})
    expected_temporal = {
        "source_fps": 50,
        "source_stride": 5,
        "action_hz": 10,
        "action_chunk": 16,
        "action_dim": 14,
        "video_frames": 8,
        "video_action_ratio": 2,
        "physical_span_seconds": 1.6,
        "interpolation": False,
        "equivalence": "50Hz_stride5_equals_author_30Hz_stride3_at_10Hz",
    }
    _require(temporal == expected_temporal, "official temporal contract changed")
    _require(
        manifest.get("action_value_contract")
        == {
            "training": "raw_joint_action_vector",
            "state_source": "action_column_matching_motus_qpos_pt",
            "future_action_source": "action_column_matching_motus_qpos_pt",
            "fastwam_observation_state_used": False,
            "deployment": "raw_state_and_raw_predicted_actions",
            "stats_loaded_but_unused_by_author_deploy_path": True,
        },
        "official action value contract changed",
    )
    records = manifest.get("records")
    _require(isinstance(records, list) and len(records) == 1650, "official manifest must contain 1650 episodes")
    _require(canonical_json_sha256(records) == manifest.get("record_inventory_sha256"), "official record SHA changed")
    counts = {task: {"clean": 0, "official_random": 0} for task in TASKS}
    episode_ids: set[int] = set()
    for record in records:
        task, domain = record.get("task"), record.get("domain")
        _require(task in TASKS and domain in {"clean", "official_random"}, "official record label changed")
        episode = int(record.get("episode_index", -1))
        _require(episode not in episode_ids, "duplicate official episode")
        episode_ids.add(episode)
        start, end = TASK_DOMAIN_RANGES[task][domain]
        _require(start <= episode <= end, "episode is outside its bound domain")
        length = int(record.get("length", -1))
        _require(length > PHYSICAL_SPAN_FRAMES, "official episode is too short")
        _require(record.get("eligible_condition_frames") == length - PHYSICAL_SPAN_FRAMES, "eligible condition count changed")
        paths = record.get("camera_paths")
        _require(isinstance(paths, dict) and tuple(paths) == CAMERAS, "official camera paths changed")
        if verify_paths:
            _require(Path(record["parquet_path"]).is_file(), "official parquet disappeared")
            for path in paths.values():
                _require(Path(path).is_file(), "official camera video disappeared")
        counts[task][domain] += 1
    _require(counts == {task: {"clean": 50, "official_random": 500} for task in TASKS}, "official counts changed")
    return {
        "status": "PASS",
        "episodes": len(records),
        "virtual_samples_per_epoch": len(records) * SAMPLES_PER_EPISODE,
        "record_inventory_sha256": manifest["record_inventory_sha256"],
    }


def _stable_integer(*values: Any) -> int:
    digest = hashlib.sha256("\0".join(str(value) for value in values).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class _PyAVBatchDecoder:
    def __call__(self, path: str, indices: list[int]) -> list[np.ndarray]:
        if not indices or indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("frame indices must be sorted and unique")
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required for AV1 official videos") from exc
        wanted = set(indices)
        frames: dict[int, np.ndarray] = {}
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    frames[index] = frame.to_ndarray(format="rgb24")
                if index >= indices[-1]:
                    break
        if set(frames) != wanted:
            raise IndexError(f"failed to decode requested indices from {path}")
        return [frames[index] for index in indices]


class MotusOfficialDataset(Dataset):
    """Deterministic 16,500-sample virtual epoch over the official episodes."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        task_embeddings: Mapping[str, torch.Tensor],
        training_seed: int,
        decoder: Callable[[str, list[int]], list[np.ndarray]] | None = None,
        parquet_loader: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_official_manifest(self.manifest)
        self.records = self.manifest["records"]
        self.training_seed = int(training_seed)
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        self.epoch = 0
        if tuple(task_embeddings) != TASKS:
            raise ValueError("task embedding map must use canonical task order")
        self.task_embeddings = {
            task: task_embeddings[task].detach().cpu() for task in TASKS
        }
        self.decoder = decoder if decoder is not None else _PyAVBatchDecoder()
        self.parquet_loader = parquet_loader or self._read_parquet
        stats_path = Path(self.manifest["normalization_stats"]["path"])
        minimum, maximum, identity = _load_stats(stats_path)
        _require(identity["sha256"] == self.manifest["normalization_stats"]["sha256"], "normalization stats changed after manifest")
        # The release training/deployment path uses raw qpos.  Stats remain
        # bound only to detect accidental changes in the author deployment
        # package; they are intentionally not applied here.
        self.minimum = torch.from_numpy(minimum)
        self.maximum = torch.from_numpy(maximum)

    @staticmethod
    def _read_parquet(path: str) -> Mapping[str, Any]:
        table = pq.read_table(path, columns=["observation.state", "action", "frame_index"])
        return table.to_pydict()

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * SAMPLES_PER_EPISODE

    def _record_and_condition(
        self, index: int, *, epoch: int | None = None
    ) -> tuple[dict[str, Any], int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = index // SAMPLES_PER_EPISODE
        record = self.records[record_index]
        eligible = int(record["eligible_condition_frames"])
        condition = _stable_integer(
            PROTOCOL_ID,
            self.training_seed,
            self.epoch if epoch is None else int(epoch),
            index,
            record["episode_index"],
        ) % eligible
        return record, int(condition)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("official tuple index must be (epoch,index)")
            epoch, sample_index = int(index[0]), int(index[1])
            if epoch < 0:
                raise IndexError("official epoch must be non-negative")
        else:
            epoch, sample_index = self.epoch, int(index)
        record, condition = self._record_and_condition(
            sample_index, epoch=epoch
        )
        action_indices = [condition + (step + 1) * ACTION_STRIDE for step in range(MOTUS_ACTION_CHUNK)]
        video_indices = [
            action_indices[(step + 1) * VIDEO_ACTION_RATIO - 1]
            for step in range(VIDEO_FRAMES)
        ]
        all_indices = [condition] + video_indices
        camera_batches = {
            camera: self.decoder(record["camera_paths"][camera], all_indices)
            for camera in CAMERAS
        }
        composites = []
        for position in range(len(all_indices)):
            composites.append(
                build_motus_t_shaped_frame(
                    camera_batches["cam_high"][position],
                    camera_batches["cam_left_wrist"][position],
                    camera_batches["cam_right_wrist"][position],
                )
            )
        images = torch.stack(
            [torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0 for image in composites]
        )
        columns = self.parquet_loader(record["parquet_path"])
        states = torch.as_tensor(np.asarray(columns["observation.state"], dtype=np.float32))
        actions = torch.as_tensor(np.asarray(columns["action"], dtype=np.float32))
        if states.shape != actions.shape or states.shape[1] != MOTUS_ACTION_DIM:
            raise OfficialDataError("parquet state/action shape changed")
        # Motus's converter saves HDF5 joint_action/vector as qpos.pt and its
        # loader selects both current state and future targets from that one
        # tensor.  The LeRobot observation.state field is therefore not used.
        initial_state = actions[condition]
        action_sequence = actions[action_indices]
        if not torch.isfinite(initial_state).all() or not torch.isfinite(action_sequence).all():
            raise FloatingPointError("normalized official state/action is non-finite")
        return {
            "first_frame": images[0],
            "video_frames": images[1:],
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "language_embedding": self.task_embeddings[record["task"]].clone(),
            "text_instruction": record["prompt"],
            "task": record["task"],
            "domain": record["domain"],
            "episode_index": record["episode_index"],
            "condition_frame_index": condition,
            "virtual_epoch": epoch,
            "virtual_sample_index": sample_index,
            "action_frame_indices": torch.tensor(action_indices, dtype=torch.long),
            "video_frame_indices": torch.tensor(video_indices, dtype=torch.long),
        }


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dataset-root", required=True)
    build.add_argument("--official-partition", required=True)
    build.add_argument("--normalization-stats", required=True)
    build.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--verify-paths", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        manifest = build_official_manifest(
            dataset_root=args.dataset_root,
            official_partition_manifest=args.official_partition,
            normalization_stats=args.normalization_stats,
        )
        validate_official_manifest(manifest, verify_paths=True)
        output = Path(args.output).resolve()
        _write_create_only(output, manifest)
        print(json.dumps({"status": "PASS", "path": str(output), "sha256": sha256_file(output), "episodes": 1650}, sort_keys=True))
        return
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = validate_official_manifest(manifest, verify_paths=args.verify_paths)
    result.update(path=str(path), sha256=sha256_file(path))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
