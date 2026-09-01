"""Physical-state-aware clean negative selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .data import VARIANTS, compatible_state_vectors


def _standardized_state_distance(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    _, left_values, right_values = compatible_state_vectors(left, right)
    scale = np.maximum(np.maximum(np.abs(left_values), np.abs(right_values)), 1.0)
    return float(
        np.linalg.norm((left_values - right_values) / scale) / np.sqrt(scale.size)
    )


def build_state_negative_mask(
    records: Sequence[Mapping[str, Any]],
    physical_states: Sequence[Mapping[str, float]],
    *,
    min_temporal_gap: int = 8,
    min_state_distance: float = 1e-5,
) -> np.ndarray:
    """Return the admissible directed state-negative relation for a batch.

    ``records`` and ``physical_states`` are parallel at the representation level:
    callers normally repeat each cached physical-state mapping for its four
    clean/R1/R2/R3 records.  For every anchor, same-trajectory states beyond the
    temporal gap are preferred.  Only when none exist are physically distinct
    states from other trajectories admitted.  Cross-task and near-state pairs
    are always excluded.

    The relation is intentionally directed because the preference is defined per
    anchor.  A later loss may symmetrize it if desired, but must not silently add
    pairs that failed this physical-state filter.
    """

    if min_temporal_gap < 0 or min_state_distance < 0:
        raise ValueError("negative thresholds must be non-negative")
    if len(records) != len(physical_states):
        raise ValueError("records and physical_states must have equal lengths")
    if not records:
        raise ValueError("negative-mask construction requires at least one record")

    count = len(records)
    mask = np.zeros((count, count), dtype=np.bool_)
    for anchor_index, anchor in enumerate(records):
        same_trajectory_far: list[int] = []
        other_trajectory: list[int] = []
        for candidate_index, candidate in enumerate(records):
            if candidate_index == anchor_index or candidate["task"] != anchor["task"]:
                continue
            if candidate["physical_state_id"] == anchor["physical_state_id"]:
                continue
            state_distance = _standardized_state_distance(
                physical_states[anchor_index], physical_states[candidate_index]
            )
            if not np.isfinite(state_distance) or state_distance <= min_state_distance:
                continue
            same_trajectory = candidate["trajectory_id"] == anchor["trajectory_id"]
            temporal_gap = abs(int(candidate["timestep"]) - int(anchor["timestep"]))
            if same_trajectory and temporal_gap >= min_temporal_gap:
                same_trajectory_far.append(candidate_index)
            elif not same_trajectory:
                other_trajectory.append(candidate_index)
        eligible = same_trajectory_far or other_trajectory
        if not eligible:
            raise ValueError(
                "No physically distinct same-task negative for "
                f"{anchor['physical_state_id']}"
            )
        mask[anchor_index, eligible] = True
    return mask


def _group_level_records_and_states(
    records: Sequence[Mapping[str, Any]],
    physical_states: Sequence[Mapping[str, float]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, float]], list[int]]:
    if (
        len(records) % len(VARIANTS) != 0
        or len(physical_states) * len(VARIANTS) != len(records)
    ):
        raise ValueError("records and physical states do not form four-render groups")
    clean_indices = list(range(0, len(records), len(VARIANTS)))
    clean_records: list[Mapping[str, Any]] = []
    for group_index, clean_index in enumerate(clean_indices):
        group = records[clean_index : clean_index + len(VARIANTS)]
        if tuple(str(item["variant"]) for item in group) != VARIANTS:
            raise ValueError(f"group {group_index} does not use canonical variant order")
        keys = {
            (str(item["task"]), str(item["physical_state_id"])) for item in group
        }
        if len(keys) != 1:
            raise ValueError(f"group {group_index} is not one aligned physical state")
        clean_records.append(records[clean_index])
    return clean_records, list(physical_states), clean_indices


def select_state_negative_pairs(
    records: Sequence[Mapping[str, Any]],
    physical_states: Sequence[Mapping[str, float]],
    *,
    min_temporal_gap: int = 8,
    min_state_distance: float = 1e-5,
) -> list[tuple[int, int]]:
    """Return one clean-clean negative for every physical state.

    State mappings are stored once per four-render physical group while records
    are flattened in clean/R1/R2/R3 order.  Same-trajectory far timesteps are
    preferred.  All candidates must differ in named robot/object state after a
    scale-aware filter; otherwise construction fails closed.
    """

    if min_temporal_gap < 0 or min_state_distance < 0:
        raise ValueError("negative thresholds must be non-negative")
    clean_records, clean_states, clean_indices = _group_level_records_and_states(
        records, physical_states
    )
    admissible = build_state_negative_mask(
        clean_records,
        clean_states,
        min_temporal_gap=min_temporal_gap,
        min_state_distance=min_state_distance,
    )

    pairs: list[tuple[int, int]] = []
    for anchor_group, anchor_index in enumerate(clean_indices):
        candidate_groups = np.flatnonzero(admissible[anchor_group]).tolist()
        negative_group = max(
            candidate_groups,
            key=lambda candidate_group: (
                abs(
                    int(clean_records[candidate_group]["timestep"])
                    - int(clean_records[anchor_group]["timestep"])
                ),
                _standardized_state_distance(
                    clean_states[anchor_group], clean_states[candidate_group]
                ),
                -candidate_group,
            ),
        )
        negative_index = clean_indices[negative_group]
        pairs.append((anchor_index, negative_index))
    return pairs


__all__ = ["build_state_negative_mask", "select_state_negative_pairs"]
