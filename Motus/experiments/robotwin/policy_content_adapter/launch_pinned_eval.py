"""Launch the pinned evaluator under one audited GPU-binding environment."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .robotwin_gpu_runtime import binding_environment, validate_binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--motus-root", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    binding = json.loads(Path(args.binding).read_text())
    validate_binding(binding)
    robotwin = Path(args.robotwin_root).resolve()
    motus = Path(args.motus_root).resolve()
    if not (robotwin / "script/eval_policy.py").is_file():
        raise FileNotFoundError(robotwin)
    env = binding_environment(binding)
    env["MOTUS_ROBOTWIN_GPU_BINDING_JSON"] = json.dumps(binding, sort_keys=True)
    env["DS_IGNORE_CUDA_DETECTION"] = "1"
    paths = [
        str(motus),
        str(motus / "inference/robotwin"),
        str(robotwin),
        str(robotwin / "script"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    env["MPLCONFIGDIR"] = str(motus / "outputs/policy_content_adapter/matplotlib_cache")
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [
        sys.executable,
        "-m",
        "experiments.robotwin.policy_content_adapter.pinned_eval_policy",
        *forwarded,
    ]
    raise SystemExit(
        subprocess.run(command, cwd=robotwin, env=env, check=False).returncode
    )


if __name__ == "__main__":
    main()
