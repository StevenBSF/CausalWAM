#!/usr/bin/env python3
"""Run RoboTwin's evaluator with an exact CUDA-to-SAPIEN PCI binding.

This experiment-owned launcher deliberately leaves ``script/eval_policy.py``
unchanged.  It injects ``render_device_alias`` into every ``setup_demo`` call,
including the expert-validity pass and the policy rollout pass, and invokes the
existing render probe with the same explicit device.
"""

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

from .robotwin_gpu_runtime import canonical_nvidia_pci_address


class PinnedEvalRuntimeError(RuntimeError):
    """The child environment cannot prove one exact physical render GPU."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PinnedEvalRuntimeError(message)


def validate_pinned_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    visible = str(env.get("CUDA_VISIBLE_DEVICES", ""))
    physical = str(env.get("ROBOTWIN_PHYSICAL_GPU_INDEX", ""))
    _require(visible.isdigit(), "CUDA_VISIBLE_DEVICES must be one numeric GPU")
    _require(
        physical == visible,
        "ROBOTWIN_PHYSICAL_GPU_INDEX must exactly match CUDA_VISIBLE_DEVICES",
    )
    _require(
        env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "CUDA_DEVICE_ORDER must be PCI_BUS_ID",
    )
    pci = canonical_nvidia_pci_address(
        str(env.get("ROBOTWIN_EXPECTED_GPU_PCI", ""))
    )
    alias = str(env.get("ROBOTWIN_RENDER_DEVICE_ALIAS", ""))
    _require(alias == f"pci:{pci}", "ROBOTWIN render alias differs from PCI identity")
    vulkan = str(env.get("VK_DRIVER_FILES", ""))
    legacy_vulkan = str(env.get("VK_ICD_FILENAMES", ""))
    egl = str(env.get("__EGL_VENDOR_LIBRARY_FILENAMES", ""))
    _require(vulkan != "" and Path(vulkan).is_file(), "NVIDIA Vulkan ICD is unavailable")
    _require(legacy_vulkan == vulkan, "VK_DRIVER_FILES and VK_ICD_FILENAMES differ")
    _require(egl != "" and Path(egl).is_file(), "NVIDIA EGL vendor manifest is unavailable")
    _require(
        env.get("__GLX_VENDOR_LIBRARY_NAME") == "nvidia",
        "GLX vendor must be explicitly NVIDIA",
    )
    return {
        "physical_gpu_index": int(physical),
        "pci_bus_id": pci,
        "render_device_alias": alias,
        "vulkan_icd": str(Path(vulkan).resolve()),
        "egl_vendor": str(Path(egl).resolve()),
    }


def _load_robotwin_eval_module(robotwin_root: Path) -> ModuleType:
    script_root = (robotwin_root / "script").resolve()
    source = script_root / "eval_policy.py"
    _require(source.is_file(), f"RoboTwin evaluator is unavailable: {source}")
    for path in (robotwin_root, script_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("_robotwin_pinned_eval_policy", source)
    _require(spec is not None and spec.loader is not None, "cannot import RoboTwin evaluator")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_setup_demo_pin(module: ModuleType, *, render_device_alias: str) -> None:
    original_decorator = getattr(module, "class_decorator", None)
    _require(callable(original_decorator), "RoboTwin evaluator lacks class_decorator")

    def pinned_class_decorator(task_name: str) -> Any:
        task_environment = original_decorator(task_name)
        original_setup_demo = getattr(task_environment, "setup_demo", None)
        _require(callable(original_setup_demo), "RoboTwin task lacks setup_demo")

        def pinned_setup_demo(*args: Any, **kwargs: Any) -> Any:
            requested = kwargs.get("render_device_alias")
            _require(
                requested in (None, render_device_alias),
                "task requested a render device different from the audited PCI binding",
            )
            kwargs["render_device_alias"] = render_device_alias
            return original_setup_demo(*args, **kwargs)

        task_environment.setup_demo = pinned_setup_demo
        return task_environment

    module.class_decorator = pinned_class_decorator


def main() -> None:
    binding = validate_pinned_environment()
    robotwin_root = Path.cwd().resolve()
    module = _load_robotwin_eval_module(robotwin_root)
    install_setup_demo_pin(
        module, render_device_alias=str(binding["render_device_alias"])
    )

    # Preserve RoboTwin's existing render gate, but make its physical device
    # explicit.  No task is attempted unless this exact-device scene succeeds.
    from test_render import Sapien_TEST

    probe = Sapien_TEST(render_device_alias=str(binding["render_device_alias"]))
    del probe
    gc.collect()

    user_args = module.parse_args_and_config()
    formal_episode_mode = str(
        user_args.get("formal_episode_mode", "") or ""
    ).strip()
    if formal_episode_mode:
        # The formal extension is installed only after the stock evaluator has
        # been loaded and its SAPIEN device has been pinned.  This keeps the
        # vendored RoboTwin tree byte-for-byte unchanged while replacing its
        # policy-dependent candidate filtering with an exact-list realization
        # or replay loop.
        from .formal_episode_protocol import install_formal_episode_mode

        encoded_preflight = os.environ.get("ROBOTWIN_GPU_PREFLIGHT_JSON", "")
        _require(bool(encoded_preflight), "formal episode mode lacks raw GPU preflight binding")
        try:
            rich_binding = json.loads(encoded_preflight)
        except Exception as exc:
            raise PinnedEvalRuntimeError(
                "formal raw GPU preflight binding is invalid JSON"
            ) from exc
        _require(isinstance(rich_binding, Mapping), "formal raw GPU binding is not an object")
        for field in (
            "physical_gpu_index",
            "pci_bus_id",
            "render_device_alias",
            "vulkan_icd",
            "egl_vendor",
        ):
            _require(
                rich_binding.get(field) == binding.get(field),
                f"formal raw GPU preflight differs from pinned environment: {field}",
            )

        install_formal_episode_mode(
            module,
            user_args,
            runtime_binding=rich_binding,
        )
    module.main(user_args)


if __name__ == "__main__":
    main()


__all__ = [
    "PinnedEvalRuntimeError",
    "install_setup_demo_pin",
    "validate_pinned_environment",
]
