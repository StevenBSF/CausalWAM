"""Run the stock RoboTwin evaluator with an audited SAPIEN PCI binding."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .robotwin_gpu_runtime import binding_environment, validate_binding


class PinnedEvalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PinnedEvalError(message)


def binding_from_environment() -> dict[str, Any]:
    raw = os.environ.get("MOTUS_ROBOTWIN_GPU_BINDING_JSON", "")
    _require(bool(raw), "MOTUS_ROBOTWIN_GPU_BINDING_JSON is missing")
    try:
        binding = json.loads(raw)
    except Exception as exc:
        raise PinnedEvalError("GPU binding environment is invalid JSON") from exc
    _require(isinstance(binding, Mapping), "GPU binding is not an object")
    result = validate_binding(binding)
    expected = binding_environment(result)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "MOTUS_ROBOTWIN_PHYSICAL_GPU",
        "MOTUS_ROBOTWIN_EXPECTED_PCI",
        "MOTUS_ROBOTWIN_RENDER_ALIAS",
        "VK_DRIVER_FILES",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
    ):
        _require(
            os.environ.get(name) == expected[name],
            f"runtime environment differs: {name}",
        )
    return result


def _load_evaluator(root: Path) -> ModuleType:
    script = root / "script" / "eval_policy.py"
    _require(script.is_file(), f"RoboTwin evaluator is missing: {script}")
    for path in (
        root,
        root / "script",
        root / "policy",
        root / "description" / "utils",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("motus_pinned_robotwin_eval", script)
    _require(
        spec is not None and spec.loader is not None, "cannot load RoboTwin evaluator"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_setup_demo_pin(module: ModuleType, *, render_alias: str) -> None:
    original = module.class_decorator

    def pinned_class_decorator(task_name: str):
        environment = original(task_name)
        original_setup = environment.setup_demo

        def pinned_setup_demo(*args, **kwargs):
            supplied = kwargs.get("render_device_alias")
            if supplied is not None and supplied != render_alias:
                raise PinnedEvalError("task requested a different render device")
            kwargs["render_device_alias"] = render_alias
            return original_setup(*args, **kwargs)

        environment.setup_demo = pinned_setup_demo
        return environment

    module.class_decorator = pinned_class_decorator


def main() -> None:
    binding = binding_from_environment()
    root = Path.cwd().resolve()
    module = _load_evaluator(root)
    alias = str(binding["render_device_alias"])
    install_setup_demo_pin(module, render_alias=alias)
    from test_render import Sapien_TEST

    probe = Sapien_TEST(render_device_alias=alias)
    del probe
    gc.collect()
    arguments = module.parse_args_and_config()
    module.main(arguments)


if __name__ == "__main__":
    main()
