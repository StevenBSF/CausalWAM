from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from experiments.robotwin.policy_content_adapter import extract_policy_cache as module
from experiments.robotwin.policy_content_adapter.data import (
    NativePairedActionDataset,
    NativePairedEpisodeGroup,
    NativePairedFrameRecord,
    PolicyPhysicalStateAnchor,
    VerifiedNativeActionManifest,
    VerifiedPolicyStateBank,
    build_policy_cache_extraction_contract,
    selected_episode_artifact_aggregate,
)
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS
from experiments.robotwin.policy_content_adapter.protocol import (
    POLICY_TOKEN_CACHE_SCHEMA,
    POLICY_TOKEN_CACHE_SCHEMA_VERSION,
    POLICY_VARIANTS,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_identity(digest: str = "a" * 64) -> dict[str, object]:
    return {"kind": "file", "path": "/immutable/file", "size_bytes": 1, "sha256": digest}


def _directory_identity(digest: str = "b" * 64) -> dict[str, object]:
    return {
        "kind": "directory",
        "path": "/immutable/directory",
        "file_count": 1,
        "size_bytes": 1,
        "sha256": digest,
    }


def test_cache_dataset_init_workdir_exists_and_restores_on_error(tmp_path: Path) -> None:
    from fastwam.utils import misc

    original = getattr(misc, "_WORK_DIR", None)
    previous = tmp_path / "previous-workdir"
    misc.register_work_dir(previous)
    output = tmp_path / "formal-cache.pt"
    try:
        with pytest.raises(RuntimeError, match="synthetic init failure"):
            with module._cache_dataset_initialization_work_dir(output) as staging:
                assert staging.is_dir()
                assert Path(misc.get_work_dir()).resolve() == staging
                (staging / "dataset_stats.json").write_text("{}", encoding="utf-8")
                assert not output.exists()
                raise RuntimeError("synthetic init failure")
        assert not staging.exists()
        assert Path(misc.get_work_dir()).resolve() == previous.resolve()
    finally:
        misc._WORK_DIR = original  # noqa: SLF001 - restore shared upstream state


def test_state_bank_extraction_plan_carries_train_split(tmp_path: Path) -> None:
    records = []
    anchors = []
    for index in range(720):
        content_id = index // 8
        frame_offset = index % 8
        trajectory_id = f"place_a2b_left/content_{content_id:06d}"
        records.append(
            NativePairedFrameRecord(
                task="place_a2b_left",
                content_id=content_id,
                split="train",
                trajectory_id=trajectory_id,
                frame_offset=frame_offset,
                base_indices=(index * 4, index * 4 + 1, index * 4 + 2, index * 4 + 3),
                episode_indices=(index * 4, index * 4 + 1, index * 4 + 2, index * 4 + 3),
            )
        )
        anchors.append(
            PolicyPhysicalStateAnchor(
                task="place_a2b_left",
                content_id=content_id,
                trajectory_id=trajectory_id,
                frame_offset=frame_offset,
            )
        )
    dataset = object.__new__(NativePairedActionDataset)
    dataset._records = tuple(records)  # noqa: SLF001 - metadata-only contract fixture
    native_manifest = VerifiedNativeActionManifest(
        path=tmp_path / "native.json",
        sha256="a" * 64,
        dataset_root=tmp_path,
        groups=(),
        audit_path=tmp_path / "audit.json",
        audit_sha256="b" * 64,
        protocol={},
    )
    state_bank = VerifiedPolicyStateBank(
        path=tmp_path / "state-bank.json",
        sha256="c" * 64,
        native_manifest=native_manifest,
        anchors=tuple(anchors),
        physical_state_inventory_sha256="d" * 64,
        protocol={},
        sampling={},
    )
    selected, plan = module.select_indices_from_verified_state_bank(dataset, state_bank)
    assert len(selected) == len(plan) == 720
    assert {row["split"] for row in plan} == {"train"}


def test_release_base_wrapper_delegates_all_immutable_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_verify(path, **kwargs):
        observed.update({"manifest": path, **kwargs})
        return {"status": "PASS", "base_kind": "author_release"}

    monkeypatch.setattr(module, "verify_author_release_lineage", fake_verify)
    report = module.verify_release_base_lineage(
        "/release/lineage.json",
        checkpoint="/release/base.pt",
        dataset_stats="/release/stats.json",
        official_manifest="/release/official.json",
        expected_manifest_sha256="a" * 64,
    )
    assert report["base_kind"] == "author_release"
    assert observed == {
        "manifest": "/release/lineage.json",
        "checkpoint_path": "/release/base.pt",
        "dataset_stats_path": "/release/stats.json",
        "official_manifest_path": "/release/official.json",
        "expected_manifest_sha256": "a" * 64,
    }


def _write_90_trajectory_artifacts(tmp_path: Path) -> VerifiedNativeActionManifest:
    root = tmp_path / "paired"
    groups: list[NativePairedEpisodeGroup] = []
    episode = 0
    for task in OFFICIAL_TASKS:
        for content_id in range(30):
            episodes: list[tuple[str, int]] = []
            for variant in POLICY_VARIANTS:
                current = episode
                episodes.append((variant, current))
                chunk = current // 1000
                parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{current:06d}.parquet"
                parquet.parent.mkdir(parents=True, exist_ok=True)
                parquet.write_bytes(f"parquet:{current}".encode("ascii"))
                for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    video = (
                        root
                        / "videos"
                        / f"chunk-{chunk:03d}"
                        / f"observation.images.{camera}"
                        / f"episode_{current:06d}.mp4"
                    )
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.write_bytes(f"av1:{current}:{camera}".encode("ascii"))
                episode += 1
            groups.append(
                NativePairedEpisodeGroup(
                    task=task,
                    content_id=content_id,
                    split="train",
                    trajectory_id=f"{task}/content_{content_id:06d}",
                    episode_length=40,
                    valid_action_anchor_count=8,
                    episodes=tuple(episodes),
                )
            )
    return VerifiedNativeActionManifest(
        path=tmp_path / "manifest.json",
        sha256="c" * 64,
        dataset_root=root,
        groups=tuple(groups),
        audit_path=tmp_path / "audit.json",
        audit_sha256="d" * 64,
        protocol={},
    )


def test_selected_train_aggregate_binds_360_episodes_and_1440_files(
    tmp_path: Path,
) -> None:
    manifest = _write_90_trajectory_artifacts(tmp_path)
    first = selected_episode_artifact_aggregate(manifest)
    assert first["episode_count"] == 360
    assert first["file_count"] == 1_440
    assert len(first["sha256"]) == 64

    target = (
        manifest.dataset_root
        / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    )
    target.write_bytes(target.read_bytes() + b"tamper")
    second = selected_episode_artifact_aggregate(manifest)
    assert second["sha256"] != first["sha256"]


def _full_payload() -> tuple[dict[str, object], dict[str, object]]:
    identities = {
        "base_lineage": _file_identity("0" * 64),
        "release_paired_binding": _file_identity("f" * 64),
        "dataset_stats": _file_identity("1" * 64),
        "vae": _file_identity("2" * 64),
        "text_encoder": _file_identity("3" * 64),
        "tokenizer": _directory_identity("4" * 64),
        "text_cache": _directory_identity("5" * 64),
        "extractor": _file_identity("6" * 64),
    }
    fastwam_source = {
        "status": "PASS",
        "scope": "all_python_files_under_src_fastwam",
        "file_count": 1,
        "files": {
            "fastwam/runtime.py": {
                "path": "/src/fastwam/runtime.py",
                "size_bytes": 1,
                "sha256": "7" * 64,
            }
        },
    }
    selected = {
        "algorithm": "relative_path_size_and_bytes_sha256_v1",
        "dataset_root": "/paired",
        "split": "train",
        "episode_count": 360,
        "file_count": 1_440,
        "size_bytes": 1_440,
        "sha256": "8" * 64,
    }
    extraction_contract = build_policy_cache_extraction_contract(
        base_lineage_identity=identities["base_lineage"],
        release_paired_binding_identity=identities["release_paired_binding"],
        dataset_stats_identity=identities["dataset_stats"],
        vae_identity=identities["vae"],
        text_encoder_identity=identities["text_encoder"],
        tokenizer_identity=identities["tokenizer"],
        text_cache_identity=identities["text_cache"],
        fastwam_source_audit=fastwam_source,
        extractor_source_identity=identities["extractor"],
        extractor_support_source_identities={
            "frozen_backbone": _file_identity("a" * 64),
            "runtime_utils": _file_identity("b" * 64),
            "policy_data": _file_identity("c" * 64),
            "policy_protocol": _file_identity("d" * 64),
        },
        selected_episode_artifacts=selected,
    )
    records: list[dict[str, object]] = []
    states: list[dict[str, object]] = []
    for state_index in range(720):
        task = OFFICIAL_TASKS[state_index // 240]
        content_id = (state_index % 240) // 8
        frame_offset = state_index % 8
        state_id = f"{task}/content_{content_id:06d}/frame_{frame_offset:06d}"
        states.append(
            {
                "task": task,
                "trajectory_id": f"{task}/content_{content_id:06d}",
                "content_id": content_id,
                "frame_offset": frame_offset,
                "physical_state_id": state_id,
            }
        )
        for view_index, variant in enumerate(POLICY_VARIANTS):
            records.append(
                {
                    "task": task,
                    "trajectory_id": f"{task}/content_{content_id:06d}",
                    "content_id": content_id,
                    "frame_offset": frame_offset,
                    "physical_state_id": state_id,
                    "split": "train",
                    "variant": variant,
                    "episode_index": content_id * 4 + view_index,
                    "view_index": view_index,
                }
            )
    backbone_sha = "9" * 64
    manifest_sha = "a" * 64
    audit_sha = "b" * 64
    state_bank_sha = "c" * 64
    inventory_sha = "d" * 64
    provenance = {
        **module._policy_metadata(),
        "backbone_checkpoint": _file_identity(backbone_sha),
        "base_lineage_manifest": identities["base_lineage"],
        "release_paired_binding_manifest": identities[
            "release_paired_binding"
        ],
        "dataset_stats": identities["dataset_stats"],
        "components": {
            name: identities[name] for name in ("vae", "text_encoder", "tokenizer")
        },
        "fastwam_source": fastwam_source,
        "extractor_source": identities["extractor"],
        "selected_episode_artifacts": selected,
        "text_cache": identities["text_cache"],
        "paired_text_cache_audit": _file_identity("e" * 64),
        "extraction_contract": extraction_contract,
        "native_prefill_identity_audit": {
            "status": "PASS",
            "checked_states": 1,
            "checked_physical_state_id": states[0]["physical_state_id"],
            "comparison": "bit_exact_K_and_V_for_every_layer",
            "rtol": 0.0,
            "atol": 0.0,
        },
        "paired_action_manifest_sha256": manifest_sha,
        "paired_action_audit_sha256": audit_sha,
        "paired_state_bank_sha256": state_bank_sha,
        "physical_state_inventory_sha256": inventory_sha,
    }
    # Expanded tensors prove the full logical 720x4 contract without allocating
    # a multi-gigabyte synthetic cache in the unit test.
    tokens = torch.zeros(1, 1, 1).expand(2_880, 120, 3_072)
    proprio = torch.zeros(1, 1).expand(720, 14)
    payload: dict[str, object] = {
        "schema": POLICY_TOKEN_CACHE_SCHEMA,
        "schema_version": POLICY_TOKEN_CACHE_SCHEMA_VERSION,
        "variant_names": list(POLICY_VARIANTS),
        "tokens_by_layer": {"16": tokens},
        "records": records,
        "physical_states": states,
        "proprio_raw": proprio,
        "provenance": provenance,
    }
    expected = {
        "extraction_contract": extraction_contract,
        "backbone": backbone_sha,
        "base_lineage": identities["base_lineage"]["sha256"],
        "release_paired_binding": identities["release_paired_binding"]["sha256"],
        "manifest": manifest_sha,
        "audit": audit_sha,
        "state_bank": state_bank_sha,
        "inventory": inventory_sha,
    }
    return payload, expected


def _validate(payload: dict[str, object], expected: dict[str, object]) -> dict[str, object]:
    return module.validate_policy_cache_payload(
        payload,
        expected_extraction_contract=expected["extraction_contract"],
        expected_backbone_sha256=str(expected["backbone"]),
        expected_base_lineage_sha256=str(expected["base_lineage"]),
        expected_release_paired_binding_sha256=str(
            expected["release_paired_binding"]
        ),
        expected_manifest_sha256=str(expected["manifest"]),
        expected_audit_sha256=str(expected["audit"]),
        expected_state_bank_sha256=str(expected["state_bank"]),
        expected_inventory_sha256=str(expected["inventory"]),
    )


def test_full_720_by_four_payload_and_dependency_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, expected = _full_payload()
    # Shape/order/provenance are the target here; avoid scanning one billion
    # repeated synthetic zeros while production retains its full finite scan.
    monkeypatch.setattr(module.torch, "isfinite", lambda _value: torch.tensor(True))
    report = _validate(payload, expected)
    assert report["physical_state_count"] == 720
    assert report["record_count"] == 2_880
    assert report["layer16_shape"] == [2_880, 120, 3_072]

    state_tamper = {**payload, "provenance": dict(payload["provenance"])}
    state_tamper["provenance"]["paired_state_bank_sha256"] = "e" * 64
    with pytest.raises(module.PolicyCacheContractError, match="state-bank identity"):
        _validate(state_tamper, expected)

    text_tamper = {**payload, "provenance": dict(payload["provenance"])}
    changed_contract = copy.deepcopy(expected["extraction_contract"])
    changed_contract["runtime_artifacts"]["text_cache"]["sha256"] = "f" * 64
    text_tamper["provenance"]["extraction_contract"] = changed_contract
    with pytest.raises(module.PolicyCacheContractError, match="extraction dependencies"):
        _validate(text_tamper, expected)
