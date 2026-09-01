"""Stateless step-addressed samplers for exact M1/M3 resume."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


def stream_seed(training_seed: int, stream: str) -> int:
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    if stream not in {"official", "paired"}:
        raise ValueError("stream must be official or paired")
    digest = hashlib.sha256(f"motus-policy-v1\0{training_seed}\0{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def action_step_seed(training_seed: int, rank: int, micro_step: int) -> int:
    if training_seed < 0 or rank < 0 or micro_step < 0:
        raise ValueError("seed, rank and micro_step must be non-negative")
    if rank >= 1024 or micro_step >= 1_000_000_000:
        raise ValueError("rank or micro_step exceeds the collision-free contract")
    # Injective over the declared domain; suitable for torch manual_seed.
    return (training_seed * 1024 + rank) * 1_000_000_000 + micro_step


class DeterministicStepBatchSampler(Sampler[list[object]]):
    """Map every trainer micro-step directly to a rank-local batch.

    Resume supplies ``start_micro_step``.  DataLoader prefetch cannot change
    future samples because no mutable sampler cursor is checkpointed.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        local_batch_size: int,
        world_size: int,
        rank: int,
        training_seed: int,
        stream: str,
        total_micro_steps: int,
        start_micro_step: int = 0,
        include_epoch_in_index: bool,
    ) -> None:
        if dataset_size <= 0 or local_batch_size <= 0 or world_size <= 0:
            raise ValueError("dataset/batch/world values must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank is outside world size")
        if not 0 <= start_micro_step <= total_micro_steps:
            raise ValueError("start_micro_step is invalid")
        self.dataset_size = int(dataset_size)
        self.local_batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.base_seed = stream_seed(training_seed, stream)
        self.total_micro_steps = int(total_micro_steps)
        self.start_micro_step = int(start_micro_step)
        self.include_epoch_in_index = bool(include_epoch_in_index)
        self.global_micro_batch = self.local_batch_size * self.world_size

    def __len__(self) -> int:
        return self.total_micro_steps - self.start_micro_step

    def _permutation(self, epoch: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed((self.base_seed + epoch) % (1 << 63))
        return torch.randperm(self.dataset_size, generator=generator)

    def batch_for_micro_step(self, micro_step: int) -> list[object]:
        if not 0 <= micro_step < self.total_micro_steps:
            raise IndexError(micro_step)
        result: list[object] = []
        global_start = micro_step * self.global_micro_batch + self.rank * self.local_batch_size
        permutations: dict[int, torch.Tensor] = {}
        for offset in range(self.local_batch_size):
            address = global_start + offset
            epoch, position = divmod(address, self.dataset_size)
            permutation = permutations.setdefault(epoch, self._permutation(epoch))
            index = int(permutation[position].item())
            result.append((epoch, index) if self.include_epoch_in_index else index)
        return result

    def __iter__(self) -> Iterator[list[object]]:
        for micro_step in range(self.start_micro_step, self.total_micro_steps):
            yield self.batch_for_micro_step(micro_step)


class DeterministicMotusEpochBatchSampler(Sampler[list[object]]):
    """Step-addressable equivalent of Motus's author DataLoader layout.

    The author uses ``DistributedSampler(drop_last=False)`` followed by a
    rank-local ``DataLoader(drop_last=True)``.  This sampler reproduces that
    padding, rank striding, local tail dropping, and epoch reshuffle while
    retaining exact optimizer-boundary resume for the adapter trainer.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        local_batch_size: int,
        world_size: int,
        rank: int,
        training_seed: int,
        total_micro_steps: int,
        start_micro_step: int = 0,
        include_epoch_in_index: bool,
    ) -> None:
        if dataset_size <= 0 or local_batch_size <= 0 or world_size <= 0:
            raise ValueError("dataset/batch/world values must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank is outside world size")
        if not 0 <= start_micro_step <= total_micro_steps:
            raise ValueError("start_micro_step is invalid")
        self.dataset_size = int(dataset_size)
        self.local_batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.base_seed = stream_seed(training_seed, "official")
        self.total_micro_steps = int(total_micro_steps)
        self.start_micro_step = int(start_micro_step)
        self.include_epoch_in_index = bool(include_epoch_in_index)
        self.num_samples_per_rank = math.ceil(self.dataset_size / self.world_size)
        self.padded_total_size = self.num_samples_per_rank * self.world_size
        self.steps_per_epoch = self.num_samples_per_rank // self.local_batch_size
        if self.steps_per_epoch <= 0:
            raise ValueError("author-style epoch contains no complete local batch")
        self.samples_per_epoch = (
            self.steps_per_epoch * self.local_batch_size * self.world_size
        )
        self._rank_epoch_cache: dict[int, tuple[int, ...]] = {}

    def __len__(self) -> int:
        return self.total_micro_steps - self.start_micro_step

    def _rank_indices(self, epoch: int) -> tuple[int, ...]:
        cached = self._rank_epoch_cache.get(epoch)
        if cached is not None:
            return cached
        generator = torch.Generator()
        generator.manual_seed((self.base_seed + epoch) % (1 << 63))
        indices = torch.randperm(
            self.dataset_size, generator=generator
        ).tolist()
        padding_size = self.padded_total_size - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        rank_indices = tuple(
            indices[self.rank : self.padded_total_size : self.world_size]
        )
        if len(rank_indices) != self.num_samples_per_rank:
            raise RuntimeError("author-style rank shard length changed")
        self._rank_epoch_cache[epoch] = rank_indices
        return rank_indices

    def batch_for_micro_step(self, micro_step: int) -> list[object]:
        if not 0 <= micro_step < self.total_micro_steps:
            raise IndexError(micro_step)
        epoch, batch_in_epoch = divmod(micro_step, self.steps_per_epoch)
        start = batch_in_epoch * self.local_batch_size
        stop = start + self.local_batch_size
        indices = self._rank_indices(epoch)[start:stop]
        if len(indices) != self.local_batch_size:
            raise RuntimeError("author-style local tail was not dropped")
        return [
            (epoch, index) if self.include_epoch_in_index else index
            for index in indices
        ]

    def __iter__(self) -> Iterator[list[object]]:
        for micro_step in range(self.start_micro_step, self.total_micro_steps):
            yield self.batch_for_micro_step(micro_step)


class DeterministicSameTaskBatchSampler(Sampler[list[int]]):
    """Step-addressed paired batches with valid local SupCon negatives.

    Every rank receives distinct physical states from exactly one task.  Ranks
    cycle deterministically across tasks, and ranks assigned the same task in
    one step consume disjoint slices of a shared task permutation.
    """

    def __init__(
        self,
        *,
        task_labels: Sequence[str],
        local_batch_size: int,
        world_size: int,
        rank: int,
        training_seed: int,
        total_micro_steps: int,
        start_micro_step: int = 0,
    ) -> None:
        if local_batch_size < 2:
            raise ValueError("paired local batch needs at least two states")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("paired sampler rank/world size is invalid")
        if not 0 <= start_micro_step <= total_micro_steps:
            raise ValueError("start_micro_step is invalid")
        pools: dict[str, list[int]] = {}
        for index, task in enumerate(task_labels):
            if not isinstance(task, str) or not task:
                raise ValueError("paired task labels must be non-empty strings")
            pools.setdefault(task, []).append(index)
        if not pools:
            raise ValueError("paired sampler received no task labels")
        maximum_rank_count = (world_size + len(pools) - 1) // len(pools)
        required_per_task = maximum_rank_count * local_batch_size
        if any(len(indices) < required_per_task for indices in pools.values()):
            raise ValueError(
                "each paired task needs enough states for disjoint rank-local batches"
            )
        self.local_batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.base_seed = stream_seed(training_seed, "paired")
        self.total_micro_steps = int(total_micro_steps)
        self.start_micro_step = int(start_micro_step)
        self.tasks = tuple(sorted(pools))
        self.pools = {task: tuple(pools[task]) for task in self.tasks}
        self.task_phase = self.base_seed % len(self.tasks)

    def __len__(self) -> int:
        return self.total_micro_steps - self.start_micro_step

    def _task_for_rank(self, micro_step: int, rank: int) -> str:
        address = micro_step * self.world_size + rank + self.task_phase
        return self.tasks[address % len(self.tasks)]

    def _task_permutation(self, micro_step: int, task: str) -> torch.Tensor:
        payload = f"{self.base_seed}\0{micro_step}\0{task}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        generator = torch.Generator()
        generator.manual_seed(seed & ((1 << 63) - 1))
        return torch.randperm(len(self.pools[task]), generator=generator)

    def batch_for_micro_step(self, micro_step: int) -> list[int]:
        if not 0 <= micro_step < self.total_micro_steps:
            raise IndexError(micro_step)
        task = self._task_for_rank(micro_step, self.rank)
        same_task_rank_offset = sum(
            self._task_for_rank(micro_step, other_rank) == task
            for other_rank in range(self.rank)
        )
        start = same_task_rank_offset * self.local_batch_size
        stop = start + self.local_batch_size
        permutation = self._task_permutation(micro_step, task)
        pool = self.pools[task]
        return [pool[int(position)] for position in permutation[start:stop]]

    def __iter__(self) -> Iterator[list[int]]:
        for micro_step in range(self.start_micro_step, self.total_micro_steps):
            yield self.batch_for_micro_step(micro_step)


def sampler_sequence_sha256(
    sampler: DeterministicStepBatchSampler
    | DeterministicMotusEpochBatchSampler
    | DeterministicSameTaskBatchSampler,
) -> str:
    digest = hashlib.sha256()
    for micro_step in range(sampler.start_micro_step, sampler.total_micro_steps):
        digest.update(str(micro_step).encode())
        digest.update(b"\0")
        for index in sampler.batch_for_micro_step(micro_step):
            digest.update(repr(index).encode())
            digest.update(b"\0")
    return digest.hexdigest()
