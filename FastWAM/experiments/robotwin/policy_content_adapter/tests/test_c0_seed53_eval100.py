from experiments.robotwin.policy_content_adapter import c0_seed53_eval100


def test_c0_seed53_eval100_protocol_constants() -> None:
    assert c0_seed53_eval100.SIMULATOR_SEED == 53
    assert c0_seed53_eval100.EPISODES_PER_CELL == 100
    assert c0_seed53_eval100.TASKS == (
        "place_a2b_left",
        "open_microwave",
        "move_stapler_pad",
    )
    assert c0_seed53_eval100.DOMAINS == ("clean", "official_random")
