from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin.policy_content_adapter.data import (
    DataContractError,
    DualStreamIterator,
    FrozenPairedTokenDataset,
    NativePairedActionDataset,
    PolicyPhysicalStateAnchor,
    SameTaskPhysicalStateBatchSampler,
    audit_file_identity,
    audit_frozen_token_cache,
    audit_native_paired_action_contract,
    audit_native_paired_action_dataset,
    build_dual_stream_provenance,
    collate_paired_action_groups,
    collate_paired_token_groups,
    flatten_paired_action_batch,
    physical_state_inventory_sha256,
    policy_state_bank_offsets,
    validate_native_paired_action_batch,
    verify_native_paired_action_manifest,
    verify_policy_state_bank,
)
from experiments.robotwin.policy_content_adapter.protocol import (
    POLICY_ACTION_MANIFEST_SCHEMA,
    POLICY_ACTION_MANIFEST_VERSION,
    POLICY_PROTOCOL_ID,
    POLICY_R3_ROLE,
    POLICY_STATE_BANK_SAMPLING_ALGORITHM,
    POLICY_STATE_BANK_SAMPLING_VERSION,
    POLICY_STATE_BANK_SCHEMA,
    POLICY_STATE_BANK_SCHEMA_VERSION,
    POLICY_STATE_BANK_SEED,
    POLICY_STATES_PER_TRAJECTORY,
    POLICY_TOKEN_CACHE_SCHEMA_VERSION,
    POLICY_VARIANTS,
)


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
BASE_SHA = "a" * 64
BASE_LINEAGE_SHA = "b" * 64
RELEASE_PAIRED_BINDING_SHA = "c" * 64
EXTRACTION_CONTRACT = {"schema": "test_extraction_contract_v1"}


def _protocol(split: str = "train") -> dict[str, object]:
    return {
        "protocol_id": POLICY_PROTOCOL_ID,
        "variant_names": list(POLICY_VARIANTS),
        "view_count": 4,
        "r3_role": POLICY_R3_ROLE,
        "camera_names": list(CAMERAS),
        "camera_count": 3,
        "native_fps": 50,
        "action_steps": 32,
        "action_dim": 14,
        "temporal_resampling": "none",
        "native_action_targets": True,
        "split": split,
    }


def _token_payload(
    *,
    state_bank,
    split: str = "train",
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    physical_states: list[dict[str, float]] = []
    proprio: list[torch.Tensor] = []
    group_by_content = {
        (group.task, group.content_id): group
        for group in state_bank.native_manifest.groups_for_split("train")
    }
    for state_index, anchor in enumerate(state_bank.anchors):
        state_id = anchor.physical_state_id
        physical_states.append({"robot.q": float(state_index)})
        proprio.append(torch.arange(14, dtype=torch.float32) + state_index)
        group = group_by_content[(anchor.task, anchor.content_id)]
        for variant in POLICY_VARIANTS:
            records.append(
                {
                    "task": anchor.task,
                    "physical_state_id": state_id,
                    "trajectory_id": anchor.trajectory_id,
                    "content_id": anchor.content_id,
                    "frame_offset": anchor.frame_offset,
                    "episode_index": group.episode_by_variant[variant],
                    "split": split,
                    "variant": variant,
                }
            )
    count = len(records)
    tokens = torch.arange(1, count * 4 * 8 + 1, dtype=torch.float32).reshape(count, 4, 8)
    provenance = {
        **_protocol(split),
        "backbone_checkpoint": {"sha256": BASE_SHA, "path": "/release/base.pt"},
        "base_lineage_manifest": {
            "sha256": BASE_LINEAGE_SHA,
            "path": "/release/base_lineage.json",
        },
        "release_paired_binding_manifest": {
            "sha256": RELEASE_PAIRED_BINDING_SHA,
            "path": "/release/paired_binding.json",
        },
        "paired_action_manifest_sha256": state_bank.native_manifest.sha256,
        "paired_action_audit_sha256": state_bank.native_manifest.audit_sha256,
        "paired_state_bank_sha256": state_bank.sha256,
        "physical_state_inventory_sha256": state_bank.physical_state_inventory_sha256,
        "extraction_contract": EXTRACTION_CONTRACT,
        "native_prefill_identity_audit": {
            "status": "PASS",
            "checked_states": 1,
            "comparison": "bit_exact_K_and_V_for_every_layer",
            "rtol": 0.0,
            "atol": 0.0,
        },
    }
    return {
        "schema": "policy_frozen_token_cache_v1",
        "schema_version": POLICY_TOKEN_CACHE_SCHEMA_VERSION,
        "variant_names": list(POLICY_VARIANTS),
        "tokens_by_layer": {"16": tokens},
        "records": records,
        "physical_states": physical_states,
        "proprio_raw": torch.stack(proprio),
        "provenance": provenance,
    }


def _write_token_cache(tmp_path: Path, state_bank, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "policy_cache.pt"
    torch.save(_token_payload(state_bank=state_bank) if payload is None else payload, path)
    return path


def test_frozen_dataset_returns_ordered_four_scene_group_with_r3_positive(
    tmp_path: Path,
) -> None:
    *_contract, state_bank = _verified_state_bank(tmp_path)
    dataset = FrozenPairedTokenDataset(
        _write_token_cache(tmp_path, state_bank),
        state_bank=state_bank,
        expected_extraction_contract=EXTRACTION_CONTRACT,
        expected_backbone_sha256=BASE_SHA,
    )
    assert len(dataset) == 32
    assert tuple(
        dataset.physical_state_id_for_index(index) for index in range(len(dataset))
    ) == tuple(anchor.physical_state_id for anchor in state_bank.anchors)
    assert dataset.variant_names == POLICY_VARIANTS
    assert dataset.token_shape == (4, 8)
    item = dataset[0]
    assert item["tokens"].shape == (4, 4, 8)
    assert item["r3_role"] == "training_positive"
    assert tuple(record["variant"] for record in item["records"]) == POLICY_VARIANTS

    batch = collate_paired_token_groups([dataset[0], dataset[1]])
    assert batch["tokens"].shape == (2, 4, 4, 8)
    assert batch["variant_names"] == POLICY_VARIANTS
    assert batch["r3_role"] == "training_positive"
    assert batch["supervision_mode"] == "contrastive"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["provenance"].update({"native_fps": 30}), "native_fps"),
        (lambda payload: payload["provenance"].update({"r3_role": "holdout"}), "r3_role"),
        (
            lambda payload: payload["provenance"].update(
                {"variant_names": list(POLICY_VARIANTS[:3]), "view_count": 3}
            ),
            "C/R1/R2/R3",
        ),
    ),
)
def test_frozen_dataset_rejects_non_policy_or_interpolated_cache(
    tmp_path: Path, mutation, message: str
) -> None:
    *_contract, state_bank = _verified_state_bank(tmp_path)
    payload = _token_payload(state_bank=state_bank)
    mutation(payload)
    with pytest.raises(DataContractError, match=message):
        FrozenPairedTokenDataset(
            _write_token_cache(tmp_path, state_bank, payload),
            state_bank=state_bank,
            expected_extraction_contract=EXTRACTION_CONTRACT,
        )


def test_frozen_dataset_rejects_cache_from_different_release_base(tmp_path: Path) -> None:
    *_contract, state_bank = _verified_state_bank(tmp_path)
    with pytest.raises(DataContractError, match="base-lineage checkpoint"):
        FrozenPairedTokenDataset(
            _write_token_cache(tmp_path, state_bank),
            state_bank=state_bank,
            expected_extraction_contract=EXTRACTION_CONTRACT,
            expected_backbone_sha256="b" * 64,
        )


def test_same_task_sampler_keeps_distinct_physical_states(tmp_path: Path) -> None:
    *_contract, state_bank = _verified_state_bank(tmp_path)
    dataset = FrozenPairedTokenDataset(
        _write_token_cache(tmp_path, state_bank),
        state_bank=state_bank,
        expected_extraction_contract=EXTRACTION_CONTRACT,
    )
    sampler = SameTaskPhysicalStateBatchSampler(dataset, groups_per_batch=2, seed=17)
    batches = list(sampler)
    assert len(batches) == 16
    for indices in batches:
        samples = [dataset[index] for index in indices]
        assert len(indices) == len(set(indices)) == 2
        assert len({sample["task"] for sample in samples}) == 1
        assert len({sample["physical_state_id"] for sample in samples}) == 2


def test_frozen_cache_rejects_runtime_extraction_contract_drift(tmp_path: Path) -> None:
    *_contract, state_bank = _verified_state_bank(tmp_path)
    with pytest.raises(DataContractError, match="extraction dependencies differ"):
        FrozenPairedTokenDataset(
            _write_token_cache(tmp_path, state_bank),
            state_bank=state_bank,
            expected_extraction_contract={"schema": "different"},
        )


class _FakeLerobot:
    def __init__(self, root: Path, episodes: tuple[int, ...]) -> None:
        self.multi_dataset = SimpleNamespace(
            _datasets=[SimpleNamespace(root=root, episodes=episodes)]
        )
        self.episode_length = 40
        self.episode_data_index = {
            "from": torch.arange(len(episodes), dtype=torch.long) * self.episode_length,
            "to": torch.arange(1, len(episodes) + 1, dtype=torch.long) * self.episode_length,
        }

    def __len__(self) -> int:
        return len(self.multi_dataset._datasets[0].episodes) * self.episode_length

    def __getitem__(self, index: int) -> dict[str, object]:
        episode_position, frame_offset = divmod(index, self.episode_length)
        episode = self.multi_dataset._datasets[0].episodes[episode_position]
        content_group = episode // 4
        action = (
            torch.arange(32 * 14, dtype=torch.float32).reshape(32, 14)
            + content_group
            + frame_offset
        )
        state = (
            torch.arange(33 * 14, dtype=torch.float32).reshape(33, 14)
            + content_group
            + frame_offset
        )
        return {
            "idx": index,
            "action": action,
            "proprio": state,
            # Meta tensors make the full-resolution shape test allocation-free.
            "pixel_values": torch.empty((3, 33, 3, 240, 320), device="meta"),
        }


class _FakeRobotVideoDataset:
    def __init__(self, root: Path, episodes: tuple[int, ...]) -> None:
        self.lerobot_dataset = _FakeLerobot(root, episodes)
        self.skip_padding_as_possible = False

    def _get(self, index: int) -> dict[str, object]:
        raw = self.lerobot_dataset[index]
        episode_position = index // self.lerobot_dataset.episode_length
        episode = self.lerobot_dataset.multi_dataset._datasets[0].episodes[episode_position]
        variant_index = episode % 4
        return {
            "video": torch.full((3, 9, 4, 4), float(variant_index)),
            "action": raw["action"].clone(),
            "proprio": raw["proprio"][:-1].clone(),
            "context": torch.ones((5, 6), dtype=torch.float32),
            "context_mask": torch.ones((5,), dtype=torch.bool),
            "action_is_pad": torch.zeros((32,), dtype=torch.bool),
            "prompt": "one shared task prompt",
        }


def _write_action_contract(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, tuple[int, ...]]:
    root = tmp_path / "paired_lerobot"
    root.mkdir()
    groups = []
    episode = 0
    for task in ("task_a", "task_b"):
        for content_id in range(2):
            episodes = {}
            for variant in POLICY_VARIANTS:
                episodes[variant] = episode
                episode += 1
            groups.append(
                {
                    "task": task,
                    "content_id": content_id,
                    "split": "train",
                    "trajectory_id": f"{task}/content_{content_id:06d}",
                    "episode_length": 40,
                    "valid_action_anchor_count": 8,
                    "episodes": episodes,
                }
            )
    manifest = {
        "schema": POLICY_ACTION_MANIFEST_SCHEMA,
        "schema_version": POLICY_ACTION_MANIFEST_VERSION,
        **_protocol("train"),
        "dataset_root": str(root.resolve()),
        "groups": groups,
    }
    manifest_path = tmp_path / "paired_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    audit = {
        "status": "PASS",
        **_protocol("train"),
        "dataset_root": str(root.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checks": {
            "three_camera_sync": True,
            "native_50hz": True,
            "action_window_32x14": True,
            "state_window_33x14": True,
            "cross_scene_state_exact": True,
            "cross_scene_action_exact": True,
            "temporal_resampling_absent": True,
            "endpoint_safe_state_bank_supported": True,
        },
    }
    audit_path = tmp_path / "paired_audit.json"
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    anchors: list[PolicyPhysicalStateAnchor] = []
    for group in groups:
        for frame_offset in policy_state_bank_offsets(
            task=group["task"],
            content_id=group["content_id"],
            episode_length=group["episode_length"],
        ):
            anchors.append(
                PolicyPhysicalStateAnchor(
                    task=group["task"],
                    content_id=group["content_id"],
                    trajectory_id=group["trajectory_id"],
                    frame_offset=frame_offset,
                )
            )
    state_bank = {
        "schema": POLICY_STATE_BANK_SCHEMA,
        "schema_version": POLICY_STATE_BANK_SCHEMA_VERSION,
        **_protocol("train"),
        "paired_action_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "paired_action_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "sampling": {
            "algorithm": POLICY_STATE_BANK_SAMPLING_ALGORITHM,
            "version": POLICY_STATE_BANK_SAMPLING_VERSION,
            "seed": POLICY_STATE_BANK_SEED,
            "states_per_trajectory": POLICY_STATES_PER_TRAJECTORY,
            "endpoint_rule": "33_state_frames_and_32_actions_without_padding",
            "short_trajectory_policy": "fail_closed",
        },
        "physical_state_inventory_sha256": physical_state_inventory_sha256(anchors),
        "states": [anchor.as_dict() for anchor in anchors],
    }
    state_bank_path = tmp_path / "paired_state_bank.json"
    state_bank_path.write_text(json.dumps(state_bank) + "\n", encoding="utf-8")
    return root, manifest_path, audit_path, state_bank_path, tuple(range(episode))


def _verified_state_bank(tmp_path: Path):
    root, manifest, audit, state_bank, episodes = _write_action_contract(tmp_path)
    verified_manifest = verify_native_paired_action_manifest(
        manifest, dataset_root=root, audit_path=audit
    )
    verified_bank = verify_policy_state_bank(
        state_bank,
        native_manifest=verified_manifest,
        expected_tasks=("task_a", "task_b"),
    )
    return root, manifest, audit, state_bank, episodes, verified_bank


def test_state_bank_is_exact_deterministic_eight_offset_inventory(tmp_path: Path) -> None:
    root, manifest, audit, state_bank_path, _episodes, verified = _verified_state_bank(tmp_path)
    assert len(verified.anchors) == 2 * 2 * 8
    assert all(len(values) == 8 for values in verified.anchors_by_content.values())
    value = json.loads(state_bank_path.read_text(encoding="utf-8"))
    value["states"][0]["frame_offset"], value["states"][1]["frame_offset"] = (
        value["states"][1]["frame_offset"],
        value["states"][0]["frame_offset"],
    )
    state_bank_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    native = verify_native_paired_action_manifest(
        manifest, dataset_root=root, audit_path=audit
    )
    with pytest.raises(DataContractError, match="canonical deterministic inventory"):
        verify_policy_state_bank(state_bank_path, native_manifest=native)


def test_native_manifest_rejects_short_trajectory_for_state_bank(tmp_path: Path) -> None:
    root, manifest, audit, _state_bank, _episodes = _write_action_contract(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["groups"][0]["episode_length"] = 39
    value["groups"][0]["valid_action_anchor_count"] = 7
    manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
    audit_value = json.loads(audit.read_text(encoding="utf-8"))
    audit_value["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audit.write_text(json.dumps(audit_value) + "\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="too short for eight"):
        verify_native_paired_action_manifest(manifest, dataset_root=root, audit_path=audit)


def test_native_action_wrapper_preserves_official_shape_and_four_scene_identity(
    tmp_path: Path,
) -> None:
    root, manifest, audit, state_bank_path, episodes, state_bank = _verified_state_bank(tmp_path)
    dataset = NativePairedActionDataset(
        _FakeRobotVideoDataset(root, episodes),
        dataset_root=root,
        manifest_path=manifest,
        audit_path=audit,
        state_bank_path=state_bank_path,
        expected_state_bank_sha256=state_bank.sha256,
        expected_tasks=("task_a", "task_b"),
    )
    assert len(dataset) == 32
    item = dataset[0]
    assert item["video"].shape == (4, 3, 9, 4, 4)
    assert item["action"].shape == (4, 32, 14)
    assert item["state_window"].shape == (4, 33, 14)
    assert item["r3_role"] == "training_positive"
    assert torch.equal(item["action"], item["action"][:1].expand_as(item["action"]))

    batch = collate_paired_action_groups([dataset[0], dataset[1]])
    validate_native_paired_action_batch(batch)
    flat = flatten_paired_action_batch(batch)
    assert flat["video"].shape == (8, 3, 9, 4, 4)
    assert flat["action"].shape == (8, 32, 14)
    report = audit_native_paired_action_dataset(dataset)
    assert report["native_fps"] == 50
    assert report["r3_training_positive"] is True


def test_native_action_batch_rejects_cross_scene_action_drift(tmp_path: Path) -> None:
    root, manifest, audit, state_bank_path, episodes, state_bank = _verified_state_bank(tmp_path)
    dataset = NativePairedActionDataset(
        _FakeRobotVideoDataset(root, episodes),
        dataset_root=root,
        manifest_path=manifest,
        audit_path=audit,
        state_bank_path=state_bank_path,
        expected_state_bank_sha256=state_bank.sha256,
    )
    batch = collate_paired_action_groups([dataset[0], dataset[1]])
    bad = copy.deepcopy(batch)
    bad["action"][0, 3, 0, 0] += 1
    with pytest.raises(DataContractError, match="action is not exact"):
        validate_native_paired_action_batch(bad)


def test_native_action_manifest_rejects_30hz_audit(tmp_path: Path) -> None:
    root, manifest, audit, state_bank_path, episodes, _state_bank = _verified_state_bank(tmp_path)
    value = json.loads(audit.read_text(encoding="utf-8"))
    value["native_fps"] = 30
    audit.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="native_fps"):
        NativePairedActionDataset(
            _FakeRobotVideoDataset(root, episodes),
            dataset_root=root,
            manifest_path=manifest,
            audit_path=audit,
            state_bank_path=state_bank_path,
        )


def test_native_action_manifest_enforces_content_id_split_and_global_uniqueness(
    tmp_path: Path,
) -> None:
    root, manifest, audit, state_bank_path, episodes, _state_bank = _verified_state_bank(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["groups"][0]["split"] = "val"
    manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
    audit_value = json.loads(audit.read_text(encoding="utf-8"))
    audit_value["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audit.write_text(json.dumps(audit_value) + "\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="belongs to 'train'"):
        NativePairedActionDataset(
            _FakeRobotVideoDataset(root, episodes),
            dataset_root=root,
            manifest_path=manifest,
            audit_path=audit,
            state_bank_path=state_bank_path,
        )


def test_native_action_contract_reports_splits_and_full_gate_fails_closed(
    tmp_path: Path,
) -> None:
    root, manifest, audit, _state_bank_path, _episodes = _write_action_contract(tmp_path)
    report = audit_native_paired_action_contract(
        dataset_root=root,
        manifest_path=manifest,
        audit_path=audit,
        expected_tasks=("task_a", "task_b"),
    )
    assert report["content_groups_by_task_split"]["task_a"] == {
        "train": 2,
        "val": 0,
        "test": 0,
    }
    with pytest.raises(DataContractError, match="full paired manifest counts"):
        audit_native_paired_action_contract(
            dataset_root=root,
            manifest_path=manifest,
            audit_path=audit,
            expected_tasks=("task_a", "task_b"),
            require_full_protocol_counts=True,
        )


def test_dual_stream_and_provenance_never_concatenate(tmp_path: Path) -> None:
    official = [object(), object(), object()]
    paired_values = [object()]
    iterator = DualStreamIterator(official, paired_values)
    outputs = [next(iterator) for _ in range(4)]
    assert [value["official"] for value in outputs] == [
        official[0],
        official[1],
        official[2],
        official[0],
    ]
    assert all(value["paired"] is paired_values[0] for value in outputs)
    assert iterator.cycles == {"official": 1, "paired": 3}

    *_contract, state_bank = _verified_state_bank(tmp_path)
    path = _write_token_cache(tmp_path, state_bank)
    identity = audit_file_identity(path)
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    paired = audit_frozen_token_cache(
        path,
        state_bank=state_bank,
        expected_extraction_contract=EXTRACTION_CONTRACT,
        expected_backbone_sha256=BASE_SHA,
    )
    audit = build_dual_stream_provenance(
        official={"dataset_manifest": identity}, paired=paired
    )
    assert audit["stream_contract"]["paired_role"] == "content_invariance_supervision"
    assert audit["paired"]["r3_training_positive"] is True
    assert len(audit["audit_sha256"]) == 64
