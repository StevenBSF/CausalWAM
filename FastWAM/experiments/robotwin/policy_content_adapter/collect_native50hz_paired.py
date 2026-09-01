#!/usr/bin/env python3
"""Collect one RoboTwin task with strict global native-50Hz sampling."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator, Sequence

from experiments.robotwin.policy_content_adapter.native50hz_paired import (
    CONTRACT_FILENAME,
    DEFAULT_COLLECTION_CONFIG,
    ROBOTWIN_ROOT,
    SAMPLE_EVERY_PHYSICS_STEPS,
    atomic_write_json,
    collection_contract_value,
    validate_collection_config,
    validate_raw_task_root,
)
from experiments.robotwin.policy_content_adapter.official_data import OFFICIAL_TASKS


DEFAULT_RAW_DATASET_NAME = "policy_native50hz_paired_rgb640x480_v1"


def default_output_root(task: str) -> Path:
    if task not in OFFICIAL_TASKS:
        raise ValueError(f"task must be one of {OFFICIAL_TASKS}")
    return ROBOTWIN_ROOT / "data" / task / DEFAULT_RAW_DATASET_NAME / "raw"


@contextlib.contextmanager
def _native_global_physics_sampler() -> Iterator[None]:
    """Patch only the live collection scope to capture global steps 0,5,10...

    RoboTwin normally invokes ``_take_picture`` at the start/end of every
    primitive and on a primitive-local counter.  Merely setting save_freq=5
    therefore does not establish global 50 Hz.  This scoped adapter calls the
    original camera capture after every physics step and admits it only when
    the trace's global physics index is divisible by five.  Duplicate boundary
    calls are rejected.  It does not synthesize states or images.
    """

    from envs._base_task import Base_Task

    original_take_picture = Base_Task._take_picture
    original_scene_step = Base_Task._pair_trace_scene_step
    original_set_hook = Base_Task.set_pair_trace_hook

    def native_take_picture(self: Any) -> None:
        if not bool(getattr(self, "save_data", False)):
            return
        recorder = getattr(self, "_pair_trace_hook", None)
        state_rows = getattr(recorder, "_state_rows", None)
        frame_indices = getattr(recorder, "frame_trace_indices", None)
        if not isinstance(state_rows, dict) or not isinstance(frame_indices, list):
            # Fail closed by publishing no untraced frame.  The collector then
            # rejects an empty/inconsistent cache before publication.
            return
        left_rows = state_rows.get("left_qpos")
        if not isinstance(left_rows, list) or not left_rows:
            return
        physics_index = len(left_rows) - 1
        if physics_index % SAMPLE_EVERY_PHYSICS_STEPS != 0:
            return
        if frame_indices and frame_indices[-1] == physics_index:
            return
        original_take_picture(self)

    def native_scene_step(self: Any, semantic_action: str) -> None:
        original_scene_step(self, semantic_action)
        # The patched capture method performs global-index filtering.
        self._take_picture()

    def native_set_hook(self: Any, hook: Any) -> None:
        original_set_hook(self, hook)
        # Establish the native t=0 observation even if the first task
        # primitive would otherwise disable or delay camera saving.
        self._take_picture()

    Base_Task._take_picture = native_take_picture
    Base_Task._pair_trace_scene_step = native_scene_step
    Base_Task.set_pair_trace_hook = native_set_hook
    try:
        yield
    finally:
        Base_Task.set_pair_trace_hook = original_set_hook
        Base_Task._pair_trace_scene_step = original_scene_step
        Base_Task._take_picture = original_take_picture


def _import_reused_collector() -> Any:
    script_dir = ROBOTWIN_ROOT / "script"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import collect_paired_random_background as collector

    return collector


def collect_native50hz(
    *,
    task: str,
    num_contents: int,
    output_root: Path,
    start_seed: int,
    max_attempts: int,
    config_path: Path,
) -> dict[str, Any]:
    if task not in OFFICIAL_TASKS:
        raise ValueError(f"task must be one of {OFFICIAL_TASKS}")
    validate_collection_config(config_path)
    collector = _import_reused_collector()
    collector.CONFIG_PATH = config_path.resolve()
    collector.TASK_CONFIG_NAME = "policy_native50hz_paired"
    original_collect = collector.collect

    def sampled_collect(**kwargs: Any) -> None:
        with _native_global_physics_sampler():
            original_collect(**kwargs)

    collector.collect = sampled_collect
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROBOTWIN_ROOT)
        result = collector.main(
            [
                "--task",
                task,
                "--num-contents",
                str(num_contents),
                "--output-root",
                str(output_root.resolve()),
                "--start-seed",
                str(start_seed),
                "--max-attempts",
                str(max_attempts),
            ]
        )
    finally:
        collector.collect = original_collect
        os.chdir(previous_cwd)
    if result != 0:
        raise RuntimeError(f"reused paired collector returned {result}")

    contract_path = output_root.resolve() / CONTRACT_FILENAME
    contract = collection_contract_value(
        task=task,
        requested_contents=num_contents,
        config_path=config_path,
    )
    atomic_write_json(contract_path, contract)
    try:
        post_audit = validate_raw_task_root(
            output_root,
            expected_task=task,
            expected_contents=num_contents,
            decode_all_frames=False,
            run_legacy_validator=True,
        )
    except Exception:
        # A PASS contract must never survive a failed post-collection audit.
        if contract_path.exists():
            contract_path.unlink()
        raise
    contract["post_collection_audit"] = {
        "status": "PASS",
        "content_count": post_audit["content_count"],
        "scene_episode_count": post_audit["scene_episode_count"],
        "minimum_future_action_windows": post_audit["minimum_future_action_windows"],
    }
    atomic_write_json(contract_path, contract)
    return post_audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=OFFICIAL_TASKS)
    parser.add_argument("--num-contents", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--config", type=Path, default=DEFAULT_COLLECTION_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.num_contents <= 50:
        raise SystemExit("--num-contents must be in [1, 50]")
    if args.start_seed < 0 or args.max_attempts < 1:
        raise SystemExit("--start-seed must be nonnegative and --max-attempts positive")
    output_root = args.output_root
    if output_root is None:
        output_root = default_output_root(args.task)
    try:
        report = collect_native50hz(
            task=args.task,
            num_contents=args.num_contents,
            output_root=output_root,
            start_seed=args.start_seed,
            max_attempts=args.max_attempts,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"native-50Hz paired collection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(json_summary(report), flush=True)
    return 0


def json_summary(report: dict[str, Any]) -> str:
    return (
        f"native-50Hz PASS task={report['task']} contents={report['content_count']} "
        f"episodes={report['scene_episode_count']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
