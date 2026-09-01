from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from experiments.robotwin.e0_e1.cache import build_cache_payload, validate_cache
from experiments.robotwin.e0_e1.data import (
    R3_HOLDOUT_PROTOCOL,
    R3_VARIANT,
    SEEN_VARIANTS,
    UNSEEN_TEST_VARIANTS,
    VARIANTS,
)
from experiments.robotwin.e0_e1.extract import extract_cache


def _samples(split: str, variants: tuple[str, ...], count: int = 2):
    content_id = {"train": 0, "val": 30, "test": 40}[split]
    return [
        {
            "physical_key": f"toy/content_{content_id:06d}/frame_{index:06d}",
            "task": "toy",
            "content_id": content_id,
            "frame_idx": index,
            "trace_idx": index,
            "split": split,
            "variant_names": variants,
            "proprio_raw": torch.arange(14, dtype=torch.float32) + index,
            "physical_state_by_name": {"robot.q": float(index)},
            "visual_input_sha256": {
                variant: {
                    "deployment_composite": f"{index * 10 + view + 1:064x}",
                    "encoded_rgb_by_camera": {
                        camera: f"{index * 100 + view * 10 + camera_index + 101:064x}"
                        for camera_index, camera in enumerate(
                            ("head_camera", "left_camera", "right_camera")
                        )
                    },
                }
                for view, variant in enumerate(variants)
            },
        }
        for index in range(count)
    ]


def _payload(
    *,
    split: str,
    variants: tuple[str, ...],
    proprio_mode: str = "observed",
):
    samples = _samples(split, variants)
    zero_hash = "0" * 64
    conditions = {}
    for index, sample in enumerate(samples):
        observed = f"{index + 1:064x}"
        effective = observed if proprio_mode == "observed" else zero_hash
        conditions[str(sample["physical_key"])] = {
            "normalized_proprio_sha256": effective,
            "context": {
                "proprio": {
                    "mode": proprio_mode,
                    "effective_normalized_sha256": effective,
                    "all_zero": proprio_mode == "constant_zero_normalized",
                }
            },
        }
    provenance = {
        "protocol": R3_HOLDOUT_PROTOCOL,
        "split": split,
        "active_variants": list(variants),
        "holdout_variant": R3_VARIANT,
        "proprio_mode": proprio_mode,
        "conditions_by_physical_state": conditions,
    }
    if split == "test":
        provenance.update(
            {
                "decision_lock_identity": {
                    "path": "/decision/lock.json",
                    "size_bytes": 1,
                    "mtime_ns": 2,
                    "sha256": "a" * 64,
                },
                "decision_lock_created_before_test": True,
            }
        )
    return build_cache_payload(
        tokens_by_layer={8: torch.randn(len(samples) * len(variants), 4, 16)},
        samples=samples,
        provenance=provenance,
    )


@pytest.mark.parametrize(
    ("split", "variants"),
    (("train", SEEN_VARIANTS), ("val", SEEN_VARIANTS), ("test", UNSEEN_TEST_VARIANTS)),
)
def test_schema_v2_is_dynamic_and_self_describing(split: str, variants: tuple[str, ...]) -> None:
    payload = _payload(split=split, variants=variants)
    assert payload["schema_version"] == 2
    assert tuple(payload["variant_names"]) == variants
    assert payload["variants_per_state"] == len(variants)
    assert len(payload["records"]) == len(payload["physical_states"]) * len(variants)
    validate_cache(payload)


def test_holdout_records_are_authoritative_over_forged_provenance() -> None:
    with pytest.raises(ValueError, match="requires variants"):
        _payload(split="train", variants=VARIANTS)

    payload = _payload(split="train", variants=SEEN_VARIANTS)
    payload["records"][2]["variant"] = R3_VARIANT
    with pytest.raises(ValueError, match="variant groups|R3"):
        validate_cache(payload)

    payload = _payload(split="train", variants=SEEN_VARIANTS)
    payload["provenance"]["active_variants"] = list(VARIANTS)
    with pytest.raises(ValueError, match="active_variants"):
        validate_cache(payload)


def test_test_cache_requires_preexisting_decision_lock_identity() -> None:
    payload = _payload(split="test", variants=UNSEEN_TEST_VARIANTS)
    payload["provenance"].pop("decision_lock_identity")
    with pytest.raises(ValueError, match="decision-lock identity"):
        validate_cache(payload)


def test_holdout_proprio_mode_is_strict() -> None:
    _payload(
        split="train",
        variants=SEEN_VARIANTS,
        proprio_mode="constant_zero_normalized",
    )
    with pytest.raises(ValueError, match="proprio_mode"):
        _payload(split="train", variants=SEEN_VARIANTS, proprio_mode="zero")


def test_no_proprio_condition_provenance_must_prove_one_zero_input() -> None:
    payload = _payload(
        split="train",
        variants=SEEN_VARIANTS,
        proprio_mode="constant_zero_normalized",
    )
    state_ids = sorted({record["physical_state_id"] for record in payload["records"]})
    zero_hash = "a" * 64
    payload["provenance"]["conditions_by_physical_state"] = {
        state_id: {
            "normalized_proprio_sha256": zero_hash,
            "context": {
                "proprio": {
                    "mode": "constant_zero_normalized",
                    "all_zero": True,
                    "effective_normalized_sha256": zero_hash,
                }
            },
        }
        for state_id in state_ids
    }
    validate_cache(payload)
    payload["provenance"]["conditions_by_physical_state"][state_ids[-1]][
        "normalized_proprio_sha256"
    ] = "b" * 64
    with pytest.raises(
        ValueError,
        match="one identical effective proprio|outer/inner effective proprio hashes disagree",
    ):
        validate_cache(payload)


def test_dynamic_groups_must_be_contiguous() -> None:
    payload = _payload(split="train", variants=SEEN_VARIANTS)
    payload["records"][1], payload["records"][3] = (
        payload["records"][3],
        payload["records"][1],
    )
    with pytest.raises(ValueError, match="non-contiguous|variant groups"):
        validate_cache(payload)


def test_schema_v1_four_variant_cache_remains_load_compatible() -> None:
    samples = _samples("test", VARIANTS)
    payload = build_cache_payload(
        tokens_by_layer={8: torch.randn(len(samples) * len(VARIANTS), 4, 16)},
        samples=samples,
        provenance={"kind": "legacy-test"},
    )
    legacy = copy.deepcopy(payload)
    legacy["schema_version"] = 1
    del legacy["variant_names"]
    del legacy["variants_per_state"]
    validate_cache(legacy)


def test_r3_test_extraction_fails_before_dataset_without_decision_lock(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="pre-existing decision lock"):
        extract_cache(
            data_root=tmp_path / "must-not-be-read",
            tasks=("place_a2b_left",),
            split="test",
            states_per_trajectory=1,
            checkpoint=tmp_path / "must-not-be-read.pt",
            dataset_stats=tmp_path / "must-not-be-read.json",
            model_base_path=tmp_path,
            output_path=tmp_path / "test_cache.pt",
            device="cpu",
            protocol=R3_HOLDOUT_PROTOCOL,
            proprio_mode="observed",
        )


def test_r3_test_extraction_rejects_non_locked_layer_before_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.robotwin.e0_e1.decision_lock_e2e3 as lock_module

    monkeypatch.setattr(
        lock_module,
        "load_decision_lock",
        lambda *args, **kwargs: (
            {"selected_layer": 16},
            {
                "path": str(tmp_path / "lock.json"),
                "size_bytes": 1,
                "mtime_ns": 1,
                "sha256": "a" * 64,
            },
        ),
    )
    with pytest.raises(ValueError, match="decision-locked selected layer"):
        extract_cache(
            data_root=tmp_path / "must-not-be-read",
            tasks=("place_a2b_left",),
            split="test",
            states_per_trajectory=1,
            checkpoint=tmp_path / "must-not-be-read.pt",
            dataset_stats=tmp_path / "must-not-be-read.json",
            model_base_path=tmp_path,
            output_path=tmp_path / "test_cache.pt",
            layers=(8, 16, 24),
            device="cpu",
            protocol=R3_HOLDOUT_PROTOCOL,
            proprio_mode="observed",
            decision_lock=tmp_path / "lock.json",
        )
