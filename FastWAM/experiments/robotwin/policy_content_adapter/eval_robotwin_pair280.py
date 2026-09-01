#!/usr/bin/env python3
"""Process-local RoboTwin evaluator for the Pair-280 seed53/100 protocol."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from . import eval_robotwin_single
from . import pv2_followup_eval100_amendment as legacy_eval_module
from .pair280_eval100 import (
    CHECKPOINT_STEP,
    DOMAINS,
    EPISODES_PER_CELL,
    PROFILE,
    SIMULATOR_SEED,
    TASKS,
    matching_checkpoint_row,
    validate,
)


def _amendment_from_argv(argv: Sequence[str]) -> Path:
    matches = [
        item.split("=", 1)[1]
        for item in argv
        if item.startswith("+EVALUATION.pv2_followup_eval_amendment=")
        or item.startswith("EVALUATION.pv2_followup_eval_amendment=")
    ]
    if len(matches) != 1:
        raise ValueError("Pair-280 evaluator requires exactly one amendment")
    return Path(matches[0]).expanduser().resolve()


def _validate_runtime_contract(
    amendment: Mapping[str, Any],
    *,
    checkpoint_contract: Mapping[str, Any],
    checkpoint_path: str | Path,
    requested_tasks: Sequence[str],
    requested_domains: Sequence[str],
    simulator_seed: int,
    episodes_per_task: int,
) -> dict[str, Any]:
    if amendment.get("profile") != PROFILE:
        raise ValueError("Pair-280 evaluation profile differs")
    if simulator_seed != SIMULATOR_SEED or episodes_per_task != EPISODES_PER_CELL:
        raise ValueError("Pair-280 evaluation requires seed53 and 100 episodes")
    tasks = tuple(requested_tasks)
    domains = tuple(requested_domains)
    full_matrix = tasks == TASKS and domains == DOMAINS
    one_cell = (
        len(tasks) == 1
        and tasks[0] in TASKS
        and len(domains) == 1
        and domains[0] in DOMAINS
    )
    if not (full_matrix or one_cell):
        raise ValueError("Pair-280 evaluation requires the full matrix or one cell")
    expected = amendment["checkpoint_contract"]
    checks = {
        "control": "c3_ours",
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "training_seed": 1,
        "checkpoint_step": CHECKPOINT_STEP,
        "formal_evaluation_eligible": False,
        "lambda_contrastive": 0.1,
        "mechanism_protocol_manifest_sha256": expected[
            "mechanism_protocol_manifest_sha256"
        ],
        "official_sample_sequence_sha256": expected[
            "official_sample_sequence_sha256"
        ],
        "paired_physical_state_sequence_sha256": expected[
            "paired_physical_state_sequence_sha256"
        ],
        "matched_stream_contract_sha256": expected[
            "matched_stream_contract_sha256"
        ],
        "simulator_seed_bank_id": expected["simulator_seed_bank_id"],
        "simulator_seed_bank_manifest_sha256": expected[
            "simulator_seed_bank_manifest_sha256"
        ],
    }
    for field, value in checks.items():
        if checkpoint_contract.get(field) != value:
            raise ValueError(f"Pair-280 checkpoint contract differs: {field}")
    return matching_checkpoint_row(
        amendment,
        checkpoint_path=checkpoint_path,
        control=str(checkpoint_contract["control"]),
        training_seed=int(checkpoint_contract["training_seed"]),
        checkpoint_step=int(checkpoint_contract["checkpoint_step"]),
    )


def _selection_validator_with_pair280_projection(
    amendment: Mapping[str, Any],
    original_validator,
):
    projection = amendment.get("selection_ancestry_projection")
    if not isinstance(projection, Mapping) or projection.get("status") != "PASS":
        raise ValueError("Pair-280 selection ancestry projection is not PASS")
    allowed = tuple(projection.get("allowed_fields", ()))
    historical = projection.get("historical_p_mode_values")
    effective = projection.get("effective_pair280_values")
    if (
        set(allowed)
        != {
            "paired_state_bank_sha256",
            "paired_text_cache_sha256",
            "paired_cache_sha256",
        }
        or not isinstance(historical, Mapping)
        or not isinstance(effective, Mapping)
    ):
        raise ValueError("Pair-280 selection ancestry projection scope differs")

    def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
        validated = original_validator(payload)
        shared = validated.get("shared_candidate_identity")
        if not isinstance(shared, Mapping):
            raise ValueError("validated P-mode selection lacks shared identity")
        projected = dict(shared)
        for field in allowed:
            if projected.get(field) != historical.get(field):
                raise ValueError(
                    f"historical P-mode selection ancestry differs: {field}"
                )
            projected[field] = effective[field]
        result = copy.deepcopy(validated)
        result["shared_candidate_identity"] = projected
        return result

    return _validate


def install(path: str | Path) -> dict[str, Any]:
    payload, _ = validate(path)
    bank_path = Path(payload["runtime_evaluation"]["seed_bank"]["path"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    legacy_eval_module.PROFILE = PROFILE
    legacy_eval_module.SIMULATOR_SEED = SIMULATOR_SEED
    legacy_eval_module.RUNTIME_EPISODES_PER_CELL = EPISODES_PER_CELL
    legacy_eval_module.TASKS = TASKS
    legacy_eval_module.DOMAINS = DOMAINS
    legacy_eval_module.validate_eval100_amendment = validate
    legacy_eval_module.matching_checkpoint_row = matching_checkpoint_row
    eval_robotwin_single._validate_pv2_eval100_checkpoint_contract = (
        _validate_runtime_contract
    )
    eval_robotwin_single.validate_selection_manifest_payload = (
        _selection_validator_with_pair280_projection(
            payload, eval_robotwin_single.validate_selection_manifest_payload
        )
    )

    def _bank(**_: Any) -> dict[str, Any]:
        return copy.deepcopy(bank)

    eval_robotwin_single._build_simulator_seed_bank = _bank
    return payload


def _compose_eval_config(argv: Sequence[str]):
    config_dir = (eval_robotwin_single.PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_dir), job_name="pair280_eval"
    ):
        cfg = compose(
            config_name="sim_robotwin.yaml",
            overrides=list(argv),
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def main() -> None:
    path = _amendment_from_argv(sys.argv[1:])
    install(path)
    cfg = _compose_eval_config(sys.argv[1:])
    eval_robotwin_single.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
