"""Deterministic, task-level prompts for the paired rendering experiment.

The paired collector does not store the episode language instruction.  A
single prompt per task therefore removes language variation as a confound.
These strings are the tasks' checked-in ``full_description`` fields, not
invented episode labels.
"""

from __future__ import annotations

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


TASK_INSTRUCTIONS = {
    "place_a2b_left": "use appropriate arm to place object A on the left of object B",
    "open_microwave": "Use one arm to open the microwave.",
    "move_stapler_pad": "use appropriate arm to move the stapler to a colored mat",
}


def prompt_for_task(task: str) -> str:
    if task not in TASK_INSTRUCTIONS:
        raise KeyError(f"No audited deterministic instruction for task {task!r}.")
    return DEFAULT_PROMPT.format(task=TASK_INSTRUCTIONS[task])


__all__ = ["TASK_INSTRUCTIONS", "prompt_for_task"]
