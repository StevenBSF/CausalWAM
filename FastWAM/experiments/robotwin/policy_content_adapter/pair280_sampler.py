"""Exact Pair-280 sampler with global-step trajectory distinctness."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Mapping

from torch.utils.data import Dataset, Sampler

from .official_data import OFFICIAL_TASKS
from .pair280_protocol import (
    PAIR280_ACTIVE_STEPS,
    PAIR280_EPOCHS,
    PAIR280_GROUPS,
    PAIR280_LOCAL_GROUPS,
    PAIR280_STATES_PER_TRAJECTORY,
    PAIR280_WORLD_SIZE,
    Pair280ContractError,
)


PAIR280_SAMPLER_ID = "exact_no_replacement_global_distinct_trajectory_v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pair280ContractError(message)


class ExactPair280GlobalBatchSampler(Sampler[list[int]]):
    """Emit ten exact passes with no same-task trajectory repeat per global step.

    Eight consecutive emitted lists are the eight rank-local batches for one
    optimizer step.  Every list contains two same-task states, while all states
    of that task across those eight lists come from distinct trajectories.
    """

    def __init__(self, dataset: Dataset, *, seed: int) -> None:
        self.seed = int(seed)
        tasks = getattr(dataset, "indices_by_task", None)
        trajectory_lookup = getattr(dataset, "trajectory_id_for_index", None)
        _require(isinstance(tasks, Mapping), "Pair-280 dataset lacks indices_by_task")
        _require(callable(trajectory_lookup), "Pair-280 dataset lacks trajectory lookup")
        by_task_trajectory: dict[str, dict[str, list[int]]] = {}
        trajectory_by_index: dict[int, str] = {}
        for task, indices in tasks.items():
            grouped: dict[str, list[int]] = defaultdict(list)
            for index in indices:
                trajectory = str(trajectory_lookup(int(index)))
                grouped[trajectory].append(int(index))
                trajectory_by_index[int(index)] = trajectory
            _require(len(grouped) == 30, f"Pair-280 task {task} must contain 30 trajectories")
            _require(
                all(
                    len(values) == PAIR280_STATES_PER_TRAJECTORY
                    for values in grouped.values()
                ),
                f"Pair-280 task {task} trajectory state counts changed",
            )
            by_task_trajectory[str(task)] = dict(sorted(grouped.items()))
        _require(
            tuple(sorted(by_task_trajectory)) == tuple(sorted(OFFICIAL_TASKS)),
            "Pair-280 task set changed",
        )
        self._by_task_trajectory = by_task_trajectory
        self._trajectory_by_index = trajectory_by_index

    def __len__(self) -> int:
        return PAIR280_ACTIVE_STEPS * PAIR280_WORLD_SIZE

    def _epoch_state_streams(self, epoch: int) -> dict[str, list[int]]:
        streams: dict[str, list[int]] = {}
        for task_index, task in enumerate(OFFICIAL_TASKS):
            rng = random.Random(
                self.seed * 10_000_019 + epoch * 1_000_003 + task_index
            )
            trajectories = list(self._by_task_trajectory[task])
            rng.shuffle(trajectories)
            states = {
                trajectory: list(self._by_task_trajectory[task][trajectory])
                for trajectory in trajectories
            }
            for values in states.values():
                rng.shuffle(values)
            positions = {trajectory: 0 for trajectory in trajectories}
            stream: list[int] = []
            # Repeating one trajectory permutation makes every contiguous
            # window of at most 30 exposures trajectory-distinct, including
            # windows that cross a cycle boundary.
            for _cycle in range(PAIR280_STATES_PER_TRAJECTORY):
                for trajectory in trajectories:
                    position = positions[trajectory]
                    stream.append(states[trajectory][position])
                    positions[trajectory] = position + 1
            _require(len(stream) == 8_400, f"Pair-280 task {task} exposure count changed")
            streams[task] = stream
        return streams

    def __iter__(self) -> Iterator[list[int]]:
        # These three rows allocate 8 local batches/task in every three global
        # steps: (3,3,2), (3,2,3), and (2,3,3).
        base_layout = (
            (0, 1, 2, 0, 1, 2, 0, 1),
            (2, 0, 1, 2, 0, 1, 2, 0),
            (1, 2, 0, 1, 2, 0, 1, 2),
        )
        for epoch in range(PAIR280_EPOCHS):
            streams = self._epoch_state_streams(epoch)
            positions = {task: 0 for task in OFFICIAL_TASKS}
            layout_rng = random.Random(self.seed * 97_409 + epoch)
            # 1,575 global steps / 3 = 525 balanced layout blocks.
            for _block in range(525):
                task_aliases = list(OFFICIAL_TASKS)
                layout_rng.shuffle(task_aliases)
                rank_rotation = layout_rng.randrange(PAIR280_WORLD_SIZE)
                for raw_row in base_layout:
                    row = raw_row[rank_rotation:] + raw_row[:rank_rotation]
                    local_batches: list[list[int]] = []
                    trajectories_by_task: dict[str, list[str]] = defaultdict(list)
                    for task_slot in row:
                        task = task_aliases[task_slot]
                        position = positions[task]
                        pair = streams[task][position : position + PAIR280_LOCAL_GROUPS]
                        _require(len(pair) == 2, "Pair-280 task stream exhausted early")
                        positions[task] = position + 2
                        local_batches.append(pair)
                        # Metadata lookup is available without loading token shards.
                        # Keep this assertion inside generation so every emitted
                        # global step proves the stronger contract.
                        trajectories_by_task[task].extend(
                            self._trajectory_by_index[index] for index in pair
                        )
                    for task, trajectories in trajectories_by_task.items():
                        _require(
                            len(trajectories) == len(set(trajectories)),
                            f"Pair-280 global step repeats a {task} trajectory",
                        )
                    yield from local_batches
            _require(
                set(positions.values()) == {8_400},
                "Pair-280 epoch task exposure counts changed",
            )


def audit_global_distinct_sampler(dataset: Dataset, *, seed: int) -> dict[str, object]:
    sampler = ExactPair280GlobalBatchSampler(dataset, seed=seed)
    trajectory_lookup = getattr(dataset, "trajectory_id_for_index")
    task_lookup = {
        int(index): str(task)
        for task, indices in getattr(dataset, "indices_by_task").items()
        for index in indices
    }
    state_counts: dict[int, int] = defaultdict(int)
    global_step: list[list[int]] = []
    checked_steps = 0
    for batch in sampler:
        global_step.append(batch)
        for index in batch:
            state_counts[int(index)] += 1
        if len(global_step) != PAIR280_WORLD_SIZE:
            continue
        trajectories_by_task: dict[str, list[str]] = defaultdict(list)
        for pair in global_step:
            _require(len(pair) == 2, "Pair-280 local batch width changed")
            tasks = {task_lookup[int(index)] for index in pair}
            _require(len(tasks) == 1, "Pair-280 local batch mixes tasks")
            task = next(iter(tasks))
            trajectories_by_task[task].extend(
                str(trajectory_lookup(int(index))) for index in pair
            )
        _require(
            all(len(values) == len(set(values)) for values in trajectories_by_task.values()),
            "Pair-280 global trajectory distinctness audit failed",
        )
        checked_steps += 1
        global_step = []
    _require(checked_steps == PAIR280_ACTIVE_STEPS, "Pair-280 global step count changed")
    _require(len(state_counts) == PAIR280_GROUPS, "Pair-280 state coverage changed")
    _require(set(state_counts.values()) == {PAIR280_EPOCHS}, "Pair-280 exposures are not exact10")
    return {
        "status": "PASS",
        "sampler_id": PAIR280_SAMPLER_ID,
        "global_active_steps": checked_steps,
        "physical_state_groups": len(state_counts),
        "exact_exposures_per_state": PAIR280_EPOCHS,
        "same_task_trajectory_repeats_within_global_step": 0,
    }


class Pair280AnchorMetadataDataset(Dataset[int]):
    """Metadata-only dataset used to audit the immutable state bank schedule."""

    def __init__(self, anchors) -> None:
        self._anchors = tuple(anchors)
        self.indices_by_task: dict[str, list[int]] = defaultdict(list)
        for index, anchor in enumerate(self._anchors):
            self.indices_by_task[str(anchor.task)].append(index)
        _require(len(self._anchors) == PAIR280_GROUPS, "Pair-280 anchor count changed")

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def trajectory_id_for_index(self, index: int) -> str:
        return str(self._anchors[index].trajectory_id)

    def physical_state_id_for_index(self, index: int) -> str:
        return str(self._anchors[index].physical_state_id)


def audit_state_bank_global_distinct(state_bank, *, seed: int) -> dict[str, object]:
    return audit_global_distinct_sampler(
        Pair280AnchorMetadataDataset(state_bank.anchors), seed=seed
    )
