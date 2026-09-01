"""Process-local adapter for the full-5-epoch seed-53 development evaluation."""

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
from .pv2_full5ep_eval100 import (
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
        raise ValueError("full5ep evaluator requires exactly one evaluation amendment")
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
        raise ValueError("full5ep evaluation profile differs")
    if simulator_seed != SIMULATOR_SEED or episodes_per_task != EPISODES_PER_CELL:
        raise ValueError("full5ep evaluation requires seed53 and 100 episodes")
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
        raise ValueError("full5ep evaluation requires the full matrix or one exact cell")
    expected = amendment["checkpoint_contract"]
    checks = {
        "stage": "mechanism_followup",
        "policy_regime": "p_v2",
        "training_seed": 1,
        "checkpoint_step": CHECKPOINT_STEP,
        "formal_evaluation_eligible": False,
        "mechanism_protocol_manifest_sha256": expected[
            "mechanism_protocol_manifest_sha256"
        ],
        "simulator_seed_bank_id": expected["simulator_seed_bank_id"],
        "simulator_seed_bank_manifest_sha256": expected[
            "simulator_seed_bank_manifest_sha256"
        ],
    }
    for field, value in checks.items():
        if checkpoint_contract.get(field) != value:
            raise ValueError(f"full5ep checkpoint contract differs: {field}")
    return matching_checkpoint_row(
        amendment,
        checkpoint_path=checkpoint_path,
        control=str(checkpoint_contract["control"]),
        training_seed=int(checkpoint_contract["training_seed"]),
        checkpoint_step=int(checkpoint_contract["checkpoint_step"]),
    )


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

    def _bank(**_: Any) -> dict[str, Any]:
        return copy.deepcopy(bank)

    eval_robotwin_single._build_simulator_seed_bank = _bank
    return payload


def _compose_eval_config(argv: Sequence[str]):
    """Compose from the absolute project config dir, independent of caller."""

    config_dir = (eval_robotwin_single.PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_dir), job_name="pv2_full5ep_eval"
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
