from __future__ import annotations

from collections import Counter

from experiments.robotwin.policy_content_adapter.pair280_protocol import (
    PAIR280_ACTIVE_STEPS,
    PAIR280_GROUPS,
    PAIR280_INACTIVE_STEPS,
    PAIR280_PROFILE_ID,
    PAIR280_TOTAL_STEPS,
    audit_pair280_active_schedule,
    paired_active_count,
    paired_active_index,
    paired_is_active,
)
from experiments.robotwin.policy_content_adapter.pair280_sampler import (
    ExactPair280GlobalBatchSampler,
    audit_global_distinct_sampler,
)


TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")


class _MetadataDataset:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.indices_by_task = {task: [] for task in TASKS}
        for task in TASKS:
            for content_id in range(30):
                trajectory = f"{task}/content_{content_id:06d}"
                for state in range(280):
                    index = len(self.rows)
                    self.rows.append(
                        (task, trajectory, f"{trajectory}/frame_{state:06d}")
                    )
                    self.indices_by_task[task].append(index)

    def trajectory_id_for_index(self, index: int) -> str:
        return self.rows[index][1]

    def physical_state_id_for_index(self, index: int) -> str:
        return self.rows[index][2]


def test_uniform_active_schedule_is_exact() -> None:
    audit = audit_pair280_active_schedule()
    assert audit["status"] == "PASS"
    assert audit["profile_id"] == PAIR280_PROFILE_ID
    assert audit["active_steps"] == PAIR280_ACTIVE_STEPS
    assert audit["inactive_steps"] == PAIR280_INACTIVE_STEPS
    assert paired_active_count(PAIR280_TOTAL_STEPS) == PAIR280_ACTIVE_STEPS
    active_indices = [
        paired_active_index(step)
        for step in range(1, PAIR280_TOTAL_STEPS + 1)
        if paired_is_active(step)
    ]
    assert active_indices == list(range(PAIR280_ACTIVE_STEPS))


def test_exact_sampler_covers_every_state_ten_times_without_same_trajectory_pair() -> None:
    dataset = _MetadataDataset()
    sampler = ExactPair280GlobalBatchSampler(dataset, seed=1)
    assert len(sampler) == PAIR280_GROUPS * 10 // 2
    counts: Counter[int] = Counter()
    epoch_local_batches = PAIR280_GROUPS // 2
    epoch_seen: set[int] = set()
    epoch_count = 0
    for batch_index, batch in enumerate(sampler, start=1):
        assert len(batch) == 2
        assert dataset.trajectory_id_for_index(batch[0]) != dataset.trajectory_id_for_index(batch[1])
        counts.update(batch)
        epoch_seen.update(batch)
        if batch_index % epoch_local_batches == 0:
            assert len(epoch_seen) == PAIR280_GROUPS
            epoch_seen.clear()
            epoch_count += 1
    assert epoch_count == 10
    assert len(counts) == PAIR280_GROUPS
    assert set(counts.values()) == {10}


def test_exact_sampler_is_deterministic_and_seeded() -> None:
    dataset = _MetadataDataset()
    first = list(ExactPair280GlobalBatchSampler(dataset, seed=1))[:64]
    repeated = list(ExactPair280GlobalBatchSampler(dataset, seed=1))[:64]
    other = list(ExactPair280GlobalBatchSampler(dataset, seed=2))[:64]
    assert first == repeated
    assert first != other


def test_global_step_has_no_same_task_trajectory_repeat() -> None:
    audit = audit_global_distinct_sampler(_MetadataDataset(), seed=1)
    assert audit["status"] == "PASS"
    assert audit["global_active_steps"] == PAIR280_ACTIVE_STEPS
    assert audit["same_task_trajectory_repeats_within_global_step"] == 0
