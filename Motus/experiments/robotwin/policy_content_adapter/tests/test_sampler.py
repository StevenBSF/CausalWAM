from __future__ import annotations

from experiments.robotwin.policy_content_adapter.sampler import (
    DeterministicMotusEpochBatchSampler,
    DeterministicSameTaskBatchSampler,
    DeterministicStepBatchSampler,
    action_step_seed,
    sampler_sequence_sha256,
)
from torch.utils.data import DistributedSampler


def _sampler(rank: int, start: int = 0):
    return DeterministicStepBatchSampler(
        dataset_size=17,
        local_batch_size=2,
        world_size=3,
        rank=rank,
        training_seed=7,
        stream="official",
        total_micro_steps=10,
        start_micro_step=start,
        include_epoch_in_index=True,
    )


def test_step_batches_are_rank_disjoint_and_resume_exact() -> None:
    full = _sampler(1)
    resumed = _sampler(1, start=4)
    assert list(resumed) == list(full)[4:]
    for step in range(10):
        batches = [_sampler(rank).batch_for_micro_step(step) for rank in range(3)]
        # Addresses in one global microbatch are unique, including across an
        # epoch boundary; tuple epoch keeps repeated dataset indices distinct.
        flat = [item for batch in batches for item in batch]
        assert len(flat) == len(set(flat))


def test_m1_m3_sequence_sha_is_stable_and_seed_sensitive() -> None:
    first = _sampler(0)
    second = _sampler(0)
    assert sampler_sequence_sha256(first) == sampler_sequence_sha256(second)
    changed = DeterministicStepBatchSampler(
        dataset_size=17,
        local_batch_size=2,
        world_size=3,
        rank=0,
        training_seed=8,
        stream="official",
        total_micro_steps=10,
        include_epoch_in_index=True,
    )
    assert sampler_sequence_sha256(first) != sampler_sequence_sha256(changed)


def test_formal_sampler_matches_distributed_sampler_plus_local_drop_last() -> None:
    dataset = list(range(17))
    world_size = 3
    local_batch = 2
    for rank in range(world_size):
        formal = DeterministicMotusEpochBatchSampler(
            dataset_size=len(dataset),
            local_batch_size=local_batch,
            world_size=world_size,
            rank=rank,
            training_seed=7,
            total_micro_steps=6,
            include_epoch_in_index=True,
        )
        reference = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=formal.base_seed,
            drop_last=False,
        )
        assert formal.steps_per_epoch == 3
        for epoch in (0, 1):
            reference.set_epoch(epoch)
            rank_indices = list(reference)
            expected = [
                (epoch, value)
                for value in rank_indices[: formal.steps_per_epoch * local_batch]
            ]
            observed = [
                item
                for step in range(
                    epoch * formal.steps_per_epoch,
                    (epoch + 1) * formal.steps_per_epoch,
                )
                for item in formal.batch_for_micro_step(step)
            ]
            assert observed == expected


def test_formal_three_task_epoch_accounting() -> None:
    sampler = DeterministicMotusEpochBatchSampler(
        dataset_size=16_500,
        local_batch_size=8,
        world_size=8,
        rank=0,
        training_seed=1,
        total_micro_steps=1_285,
        include_epoch_in_index=True,
    )
    assert sampler.steps_per_epoch == 257
    assert sampler.samples_per_epoch == 16_448
    assert sampler.total_micro_steps == 5 * sampler.steps_per_epoch


def test_action_step_seeds_are_collision_free_in_formal_domain() -> None:
    values = {
        action_step_seed(seed, rank, step)
        for seed in (1, 2, 3)
        for rank in range(8)
        for step in range(20)
    }
    assert len(values) == 3 * 8 * 20


def _paired_sampler(rank: int, *, seed: int = 7, start: int = 0):
    return DeterministicSameTaskBatchSampler(
        task_labels=[task for task in ("a", "b", "c") for _ in range(12)],
        local_batch_size=2,
        world_size=8,
        rank=rank,
        training_seed=seed,
        total_micro_steps=12,
        start_micro_step=start,
    )


def test_paired_batches_have_same_task_distinct_states_and_resume_exact() -> None:
    labels = [task for task in ("a", "b", "c") for _ in range(12)]
    full = _paired_sampler(4)
    assert list(_paired_sampler(4, start=5)) == list(full)[5:]
    for step in range(12):
        batches = [
            _paired_sampler(rank).batch_for_micro_step(step)
            for rank in range(8)
        ]
        for batch in batches:
            assert len(batch) == len(set(batch)) == 2
            assert len({labels[index] for index in batch}) == 1
        for task in ("a", "b", "c"):
            task_indices = [
                index
                for batch in batches
                for index in batch
                if labels[index] == task
            ]
            assert len(task_indices) == len(set(task_indices))


def test_paired_sequence_is_stable_and_seed_sensitive() -> None:
    assert sampler_sequence_sha256(_paired_sampler(0)) == sampler_sequence_sha256(
        _paired_sampler(0)
    )
    assert sampler_sequence_sha256(_paired_sampler(0)) != sampler_sequence_sha256(
        _paired_sampler(0, seed=8)
    )
