#!/usr/bin/env python3
"""Validate raw native-50Hz pairs or an exported LeRobot-v2.1 root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from experiments.robotwin.policy_content_adapter.native50hz_paired import (
    atomic_write_json,
    validate_lerobot_v21_root,
    validate_raw_task_root,
)
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    raw = subparsers.add_parser("raw")
    raw.add_argument("--task", required=True, choices=OFFICIAL_TASKS)
    raw.add_argument("--root", required=True, type=Path)
    raw.add_argument("--expected-contents", required=True, type=int)
    raw.add_argument("--report", type=Path)
    raw.add_argument(
        "--fast-camera-check",
        action="store_true",
        help="decode only first/last camera frames; formal gates must omit this",
    )
    exported = subparsers.add_parser("lerobot")
    exported.add_argument("--root", required=True, type=Path)
    exported.add_argument("--expected-contents", required=True, type=int)
    exported.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "raw":
            report = validate_raw_task_root(
                args.root,
                expected_task=args.task,
                expected_contents=args.expected_contents,
                decode_all_frames=not args.fast_camera_check,
                run_legacy_validator=True,
            )
        else:
            report = validate_lerobot_v21_root(
                args.root,
                expected_contents=args.expected_contents,
            )
        if args.report is not None:
            atomic_write_json(args.report, report)
    except Exception as exc:
        print(f"native-50Hz validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
