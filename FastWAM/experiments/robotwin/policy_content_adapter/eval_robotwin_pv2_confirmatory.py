"""Seed-59 confirmatory entry that preserves the completed pilot evaluator.

The seed-53 pilot binds the exact bytes of :mod:`eval_robotwin_single`; those
bytes remain untouched.  This entry validates a new confirmatory amendment and
installs a process-local compatibility adapter so the already-audited evaluator
can consume the seed-59 bank.  No file or checkpoint is modified.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

from . import eval_robotwin_single
from . import pv2_followup_eval100_amendment as pilot_amendment_module
from .pv2_actiondit_followup_confirmatory import (
    DOMAINS,
    EPISODES_PER_CELL,
    PROFILE,
    SIMULATOR_SEED,
    TASKS,
    matching_checkpoint_row,
    validate_confirmatory_amendment,
)


class ConfirmatoryLauncherError(ValueError):
    """The process-local confirmatory evaluator adapter is invalid."""


def _amendment_from_argv(argv: list[str]) -> Path:
    keys = (
        "+EVALUATION.pv2_followup_eval_amendment=",
        "EVALUATION.pv2_followup_eval_amendment=",
    )
    matches = [
        argument.split("=", 1)[1]
        for argument in argv
        if any(argument.startswith(key) for key in keys)
    ]
    if len(matches) != 1:
        raise ConfirmatoryLauncherError(
            "confirmatory CLI requires exactly one "
            "+EVALUATION.pv2_followup_eval_amendment=... override"
        )
    path = Path(matches[0]).expanduser().resolve()
    if not path.is_file():
        raise ConfirmatoryLauncherError(f"confirmatory amendment missing: {path}")
    return path


def install_confirmatory_profile(path: str | Path) -> dict[str, Any]:
    """Validate the amendment and install only process-local profile hooks."""

    payload, _ = validate_confirmatory_amendment(path)
    runtime_bank_identity = payload["runtime_evaluation"]["seed_bank"]
    runtime_bank_path = Path(runtime_bank_identity["path"])
    import json

    runtime_bank = json.loads(runtime_bank_path.read_text(encoding="utf-8"))
    if not isinstance(runtime_bank, dict):
        raise ConfirmatoryLauncherError("confirmatory seed bank root must be an object")

    # _validate_pv2_eval100_checkpoint_contract imports these symbols lazily;
    # replacing them in this process reuses its complete checkpoint/ancestry
    # validation while changing only the external online evaluation profile.
    pilot_amendment_module.PROFILE = PROFILE
    pilot_amendment_module.SIMULATOR_SEED = SIMULATOR_SEED
    pilot_amendment_module.RUNTIME_EPISODES_PER_CELL = EPISODES_PER_CELL
    pilot_amendment_module.TASKS = TASKS
    pilot_amendment_module.DOMAINS = DOMAINS
    pilot_amendment_module.validate_eval100_amendment = (
        validate_confirmatory_amendment
    )
    pilot_amendment_module.matching_checkpoint_row = matching_checkpoint_row

    def _confirmatory_seed_bank(**_: Any) -> dict[str, Any]:
        return copy.deepcopy(runtime_bank)

    eval_robotwin_single._build_simulator_seed_bank = _confirmatory_seed_bank
    return payload


def main() -> None:
    amendment = _amendment_from_argv(sys.argv[1:])
    install_confirmatory_profile(amendment)
    eval_robotwin_single.main()


if __name__ == "__main__":
    main()


__all__ = ["ConfirmatoryLauncherError", "install_confirmatory_profile"]
