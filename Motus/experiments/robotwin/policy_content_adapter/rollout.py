"""CLI for Motus RoboTwin rollout settings, cell plans, and receipts."""

import argparse
import json
from .rollout_common import write_json
from .rollout_finalize import audit_cell, finalize_cell
from .rollout_plan import build_plan
from .rollout_settings import build_settings


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    settings = sub.add_parser("settings")
    settings.add_argument("--lineage", required=True)
    settings.add_argument("--robotwin-root", required=True)
    settings.add_argument("--motus-root", required=True)
    settings.add_argument("--simulator-seed", type=int, default=42)
    settings.add_argument("--output", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--settings", required=True)
    plan.add_argument("--checkpoint", required=True)
    plan.add_argument("--training-summary", required=True)
    plan.add_argument("--control", required=True)
    plan.add_argument("--training-seed", type=int, required=True)
    plan.add_argument("--task", required=True)
    plan.add_argument("--domain", required=True)
    plan.add_argument("--cell-output-dir", required=True)
    plan.add_argument("--output", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--plan", required=True)
    finish.add_argument("--result", required=True)
    finish.add_argument("--log", required=True)
    finish.add_argument("--output", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--cell", required=True)
    args = parser.parse_args()
    if args.command == "settings":
        value = build_settings(
            args.lineage, args.robotwin_root, args.motus_root, args.simulator_seed
        )
        write_json(args.output, value)
    elif args.command == "plan":
        value = build_plan(
            args.settings,
            args.checkpoint,
            args.training_summary,
            args.control,
            args.training_seed,
            args.task,
            args.domain,
            args.cell_output_dir,
        )
        write_json(args.output, value)
    elif args.command == "finalize":
        value = finalize_cell(args.plan, args.result, args.log, args.output)
    else:
        value = audit_cell(args.cell)
    print(
        json.dumps(
            {
                "status": "PASS",
                "command": args.command,
                "control": value.get("control"),
                "task": value.get("task"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
