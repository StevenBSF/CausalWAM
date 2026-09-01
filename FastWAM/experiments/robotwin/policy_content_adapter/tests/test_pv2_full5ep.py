from experiments.robotwin.policy_content_adapter import pv2_actiondit_full5ep


def test_author_matched_three_task_epoch_math() -> None:
    assert pv2_actiondit_full5ep.OFFICIAL_SAMPLE_COUNT == 466_240
    assert pv2_actiondit_full5ep.NUM_EPOCHS == 5
    assert pv2_actiondit_full5ep.WORLD_SIZE == 8
    assert pv2_actiondit_full5ep.PREFERRED_LOCAL_BATCH == 16
    assert pv2_actiondit_full5ep.PREFERRED_GLOBAL_BATCH == 128
    assert pv2_actiondit_full5ep.PREFERRED_STEPS_PER_EPOCH == 3_643
    assert pv2_actiondit_full5ep.PREFERRED_MAX_STEPS == 18_215
    assert pv2_actiondit_full5ep.FORMAL_SAVE_EVERY == 2_000
    assert pv2_actiondit_full5ep.SMOKE_SAVE_EVERY == 3
    assert pv2_actiondit_full5ep.FALLBACK_LOCAL_BATCH == 8
    assert pv2_actiondit_full5ep.FALLBACK_GLOBAL_BATCH == 64
    assert pv2_actiondit_full5ep.FALLBACK_STEPS_PER_EPOCH == 7_285
    assert pv2_actiondit_full5ep.FALLBACK_MAX_STEPS == 36_425


def test_preferred_batch_covers_five_epochs_with_only_sampler_padding() -> None:
    slots = (
        pv2_actiondit_full5ep.PREFERRED_MAX_STEPS
        * pv2_actiondit_full5ep.PREFERRED_GLOBAL_BATCH
    )
    nominal = (
        pv2_actiondit_full5ep.OFFICIAL_SAMPLE_COUNT
        * pv2_actiondit_full5ep.NUM_EPOCHS
    )
    assert slots == 2_331_520
    assert nominal == 2_331_200
    assert slots - nominal == 320
    assert (
        pv2_actiondit_full5ep.PREFERRED_MAX_STEPS
        * pv2_actiondit_full5ep.EFFECTIVE_PAIRED_GROUPS_PER_STEP
        == 291_440
    )
