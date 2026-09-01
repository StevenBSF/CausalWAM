#!/usr/bin/env python3
"""Export audited three-task native-50Hz raw pairs to LeRobot v2.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from experiments.robotwin.policy_content_adapter.native50hz_paired import (
    export_paired_lerobot_v21,
)
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for task in OFFICIAL_TASKS:
        parser.add_argument(f"--{task.replace('_', '-')}-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-contents", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_roots = {
        task: getattr(args, f"{task}_root")
        for task in OFFICIAL_TASKS
    }
    try:
        report = export_paired_lerobot_v21(
            raw_roots,
            output_root=args.output_root,
            expected_contents=args.expected_contents,
        )
    except Exception as exc:
        print(f"native-50Hz LeRobot export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
