from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.robotwin.policy_content_adapter import collect_native50hz_paired as collector
from experiments.robotwin.policy_content_adapter import native50hz_paired as module


def test_direct_collector_default_uses_fresh_rgb640x480_namespace() -> None:
    root = collector.default_output_root("open_microwave")
    assert collector.DEFAULT_RAW_DATASET_NAME == "policy_native50hz_paired_rgb640x480_v1"
    assert root.name == "raw"
    assert root.parent.name == collector.DEFAULT_RAW_DATASET_NAME
    assert "policy_native50hz_paired/raw" not in str(root)


def test_checked_config_is_exact_native_250_to_50hz_without_interpolation() -> None:
    report = module.validate_collection_config()
    assert report["physics_hz"] == 250
    assert report["sample_every_physics_steps"] == 5
    assert report["fps"] == 50
    assert report["timestamp_delta_seconds"] == 0.02
    assert report["action_horizon"] == 32
    assert report["action_dim"] == 14
    assert report["scene_variants"] == ["C", "R1", "R2", "R3"]
    assert report["interpolation"] == "forbidden"
    assert report["camera_type"] == "Large_D435"
    assert report["image_shape_hwc"] == [480, 640, 3]
    assert len(report["camera_catalog_sha256"]) == 64


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("save_freq",), 15),
        (("timestep",), 1 / 30),
        (("policy_native50hz_contract", "interpolation"), "linear"),
        (("policy_native50hz_contract", "fps"), 30),
        (("camera", "head_camera_type"), "D435"),
        (("camera", "wrist_camera_type"), "D435"),
        (("policy_native50hz_contract", "camera_type"), "D435"),
        (("policy_native50hz_contract", "image_shape_hwc"), [240, 320, 3]),
    ],
)
def test_config_tampering_fails_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    config = yaml.safe_load(module.DEFAULT_COLLECTION_CONFIG.read_text(encoding="utf-8"))
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    target = tmp_path / "tampered.yml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(module.Native50HzContractError):
        module.validate_collection_config(target)


def test_collection_contract_refuses_to_claim_simulator_snapshot_support() -> None:
    contract = module.collection_contract_value(
        task="open_microwave",
        requested_contents=1,
    )
    assert contract["trajectory_contract"]["simulator_snapshot_support_claimed"] is False
    assert contract["sampling"]["30_to_50_interpolation"] == "explicitly_forbidden"
    assert contract["output_contract"]["future_action_steps"] == 32
    assert contract["output_contract"]["camera_keys"] == list(module.CAMERA_PATHS)


def test_av1_encoder_uses_file_backed_stderr_instead_of_a_deadlocking_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeStdin:
        def write(self, payload: bytes) -> int:
            observed["stdin_bytes"] = int(observed.get("stdin_bytes", 0)) + len(payload)
            return len(payload)

        def close(self) -> None:
            observed["stdin_closed"] = True

    class FakeProcess:
        def __init__(self, command, *, stdin, stderr) -> None:
            observed["command"] = command
            observed["stderr_is_pipe"] = stderr is module.subprocess.PIPE
            # Exceed this host's observed 8 KiB pipe capacity.  A real file
            # accepts the complete SVT warning stream without blocking stdin.
            stderr.write(b"svt-warning\n" * 2048)
            self.stdin = FakeStdin()

        def wait(self) -> int:
            return 0

        def kill(self) -> None:  # pragma: no cover - success path only
            raise AssertionError("successful fake encoder must not be killed")

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        module,
        "_probe_video",
        lambda path, *, expected_frames: {"path": str(path), "frames": expected_frames},
    )
    frame = np.zeros((module.IMAGE_HEIGHT, module.IMAGE_WIDTH, 3), dtype=np.uint8)
    report = module._encode_av1([frame], tmp_path / "episode.mp4")

    assert observed["stderr_is_pipe"] is False
    assert observed["stdin_closed"] is True
    assert observed["stdin_bytes"] == frame.nbytes
    command = observed["command"]
    assert command[command.index("-svtav1-params") + 1] == "lp=16"
    assert report["frames"] == 1


def test_scoped_sampler_publishes_only_global_physics_steps_0_5_10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBase:
        def set_pair_trace_hook(self, hook):
            self._pair_trace_hook = hook

        def _take_picture(self):
            self._pair_trace_hook.frame_trace_indices.append(
                len(self._pair_trace_hook._state_rows["left_qpos"]) - 1
            )

        def _pair_trace_scene_step(self, semantic_action):
            del semantic_action
            self._pair_trace_hook._state_rows["left_qpos"].append(np.zeros(1))

    fake_module = types.ModuleType("envs._base_task")
    fake_module.Base_Task = FakeBase
    monkeypatch.setitem(sys.modules, "envs._base_task", fake_module)
    recorder = types.SimpleNamespace(
        _state_rows={"left_qpos": [np.zeros(1)]},
        frame_trace_indices=[],
    )
    instance = FakeBase()
    instance.save_data = True
    with collector._native_global_physics_sampler():
        instance.set_pair_trace_hook(recorder)
        for _ in range(12):
            instance._pair_trace_scene_step("test")
            # Primitive-local/boundary calls cannot duplicate a global frame.
            instance._take_picture()
    assert recorder.frame_trace_indices == [0, 5, 10]
    assert FakeBase._take_picture.__name__ == "_take_picture"


def _write_synthetic_variant(path: Path, *, tamper_trace: bool = False) -> None:
    cv2 = pytest.importorskip("cv2")
    h5py = pytest.importorskip("h5py")
    path.mkdir(parents=True)
    (path / "data").mkdir()
    # Forty exported frames (41 raw states before the one-step action shift)
    # are the formal minimum: 40 - 32 = eight endpoint-safe anchors.
    frame_count = 41
    trace = np.arange(frame_count, dtype=np.int64) * 5
    if tamper_trace:
        trace[17] += 1
    state_arrays = {
        "frame_trace_index": trace,
        "left_qpos": np.arange(frame_count * 2, dtype=np.float64).reshape(frame_count, 2),
    }
    action_arrays = {
        "semantic_action": np.asarray(["step"] * (frame_count - 1), dtype="<U8"),
        "left_drive_target": np.arange((frame_count - 1) * 2, dtype=np.float64).reshape(
            frame_count - 1, 2
        ),
    }
    np.savez(path / "state_trace.npz", **state_arrays)
    np.savez(path / "action_trace.npz", **action_arrays)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    encoded_bytes = encoded.tobytes()
    joints = np.arange(frame_count * 14, dtype=np.float64).reshape(frame_count, 14)
    with h5py.File(path / "data" / "episode0.hdf5", "w") as handle:
        handle.create_dataset("joint_action/vector", data=joints)
        for raw_path in module.CAMERA_PATHS.values():
            handle.create_dataset(
                raw_path,
                data=np.asarray([encoded_bytes] * frame_count, dtype=f"S{len(encoded_bytes)}"),
            )
    (path / "metadata.json").write_text(
        json.dumps({"frame_count": frame_count, "content_seed": 7}) + "\n",
        encoding="utf-8",
    )


def _write_synthetic_content(root: Path, *, tamper_variant: str | None = None) -> Path:
    content = root / "content_000000"
    for name in module.VARIANT_DIRS:
        _write_synthetic_variant(content / name, tamper_trace=name == tamper_variant)
    (content / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    return content


def test_raw_content_proves_four_equal_native_traces_and_future_horizon(tmp_path: Path) -> None:
    content = _write_synthetic_content(tmp_path)
    report = module.validate_raw_content(
        content,
        expected_task="place_a2b_left",
        expected_content_id=0,
        decode_all_frames=False,
        run_legacy_validator=False,
    )
    assert report["status"] == "PASS"
    assert report["scene_variants"] == {
        "C": "clean",
        "R1": "style_00_seed_0",
        "R2": "style_01_seed_1",
        "R3": "style_02_seed_2",
    }
    assert report["raw_native_frame_count"] == 41
    assert report["converted_frame_count"] == 40
    assert report["valid_future_action_windows"] == 8
    assert report["exact_state_action_trace_identity"] is True
    assert report["interpolation_used"] is False


def test_non_native_or_unequal_trace_fails_closed(tmp_path: Path) -> None:
    content = _write_synthetic_content(tmp_path, tamper_variant="style_01_seed_1")
    with pytest.raises(module.Native50HzContractError, match="native every-5"):
        module.validate_raw_content(
            content,
            expected_task="place_a2b_left",
            expected_content_id=0,
            decode_all_frames=False,
            run_legacy_validator=False,
        )


def test_parquet_writer_emits_14d_vectors_and_exact_50hz_timestamps(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    state = np.arange(33 * 14, dtype=np.float32).reshape(33, 14)
    action = state + 1
    path = tmp_path / "episode.parquet"
    module._write_parquet(
        path,
        state=state,
        action=action,
        episode_index=0,
        global_start=0,
        task_index=0,
    )
    loaded_state, loaded_action, timestamps, frame_index = module._read_parquet_vectors(path)
    assert np.array_equal(loaded_state, state)
    assert np.array_equal(loaded_action, action)
    assert np.array_equal(frame_index, np.arange(len(state), dtype=np.int64))
    assert np.array_equal(
        timestamps.astype(np.float32),
        module._native_float32_timestamp_grid(len(timestamps)),
    )


def test_long_float32_episode_uses_exact_native_ticks_without_delta_false_positive(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    # The 1,480-frame Open Microwave pilot previously failed because valid
    # float32 adjacent differences deviate from 0.02 by up to ~1.45e-6.
    length = 1480
    state = np.zeros((length, 14), dtype=np.float32)
    path = tmp_path / "long_episode.parquet"
    module._write_parquet(
        path,
        state=state,
        action=state.copy(),
        episode_index=0,
        global_start=0,
        task_index=1,
    )
    _, _, timestamps, frame_index = module._read_parquet_vectors(path)
    module._require_native_float32_timestamp_grid(timestamps, label="long episode")
    recovered_ticks = np.rint(timestamps * module.FPS).astype(np.int64)
    assert np.array_equal(recovered_ticks, np.arange(length, dtype=np.int64))
    assert np.array_equal(frame_index, np.arange(length, dtype=np.int64))

    tampered = timestamps.copy()
    tampered[1000] = np.nextafter(
        tampered[1000],
        np.float32(np.inf),
        dtype=np.float32,
    )
    with pytest.raises(module.Native50HzContractError, match="native 50 Hz ticks"):
        module._require_native_float32_timestamp_grid(tampered, label="tampered episode")

    wrong_rate = (np.arange(length, dtype=np.float64) / 30.0).astype(np.float32)
    with pytest.raises(module.Native50HzContractError, match="native 50 Hz ticks"):
        module._require_native_float32_timestamp_grid(wrong_rate, label="30 Hz episode")
