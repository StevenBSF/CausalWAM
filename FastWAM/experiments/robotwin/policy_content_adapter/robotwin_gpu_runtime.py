#!/usr/bin/env python3
"""Fail-closed CUDA/Vulkan/SAPIEN binding for RoboTwin online evaluation.

The stock RoboTwin evaluator accepts a CUDA device for the policy, but its
SAPIEN compatibility renderer otherwise chooses a Vulkan device implicitly.
That is unsafe when two policy candidates are evaluated concurrently.  This
module resolves one numeric ``nvidia-smi`` index to its canonical PCI address,
checks that the exact SAPIEN device can render, and returns the environment
contract consumed by :mod:`pinned_eval_policy`.

The ``preflight`` CLI is read-only: it does not create an environment episode,
load a policy checkpoint, or write an evaluation artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_NVIDIA_VULKAN_ICD = Path("/etc/vulkan/icd.d/nvidia_icd.json")
DEFAULT_NVIDIA_EGL_VENDOR = Path(
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
)
_NVIDIA_PCI_RE = re.compile(
    r"(?P<domain>[0-9a-fA-F]{4,8}):(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\.(?P<function>[0-7])"
)
_SAPIEN_MARKER = "ROBOTWIN_SAPIEN_DEVICE_JSON="


class RobotwinGpuPreflightError(RuntimeError):
    """The requested online CUDA/render-device contract is not usable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RobotwinGpuPreflightError(message)


def normalize_physical_gpu_id(value: str | int) -> int:
    text = str(value).strip()
    _require(
        re.fullmatch(r"[0-9]+", text) is not None,
        "RoboTwin online evaluation requires one numeric physical GPU index",
    )
    return int(text)


def canonical_nvidia_pci_address(value: str) -> str:
    match = _NVIDIA_PCI_RE.fullmatch(str(value).strip())
    _require(match is not None, f"invalid NVIDIA PCI address: {value!r}")
    assert match is not None
    domain = int(match.group("domain"), 16)
    _require(domain <= 0xFFFF, f"NVIDIA PCI domain is out of range: {value!r}")
    return (
        f"{domain:04x}:{match.group('bus').lower()}:"
        f"{match.group('device').lower()}.{match.group('function')}"
    )


def _runtime_paths() -> tuple[Path, Path]:
    vulkan = Path(
        os.environ.get(
            "ROBOTWIN_NVIDIA_VULKAN_ICD", str(DEFAULT_NVIDIA_VULKAN_ICD)
        )
    ).expanduser().resolve()
    egl = Path(
        os.environ.get(
            "ROBOTWIN_NVIDIA_EGL_VENDOR", str(DEFAULT_NVIDIA_EGL_VENDOR)
        )
    ).expanduser().resolve()
    _require(vulkan.is_file(), f"NVIDIA Vulkan ICD is unavailable: {vulkan}")
    _require(egl.is_file(), f"NVIDIA EGL vendor manifest is unavailable: {egl}")
    return vulkan, egl


def gpu_binding_environment(
    binding: Mapping[str, Any],
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that binds CUDA and SAPIEN to one PCI device."""

    gpu = normalize_physical_gpu_id(binding.get("physical_gpu_index", ""))
    pci = canonical_nvidia_pci_address(str(binding.get("pci_bus_id", "")))
    alias = str(binding.get("render_device_alias", ""))
    _require(alias == f"pci:{pci}", "render-device alias differs from PCI identity")
    vulkan = Path(str(binding.get("vulkan_icd", ""))).expanduser().resolve()
    egl = Path(str(binding.get("egl_vendor", ""))).expanduser().resolve()
    _require(vulkan.is_file(), f"bound NVIDIA Vulkan ICD is unavailable: {vulkan}")
    _require(egl.is_file(), f"bound NVIDIA EGL manifest is unavailable: {egl}")

    env = dict(os.environ if base_environment is None else base_environment)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "ROBOTWIN_PHYSICAL_GPU_INDEX": str(gpu),
            "ROBOTWIN_EXPECTED_GPU_PCI": pci,
            "ROBOTWIN_RENDER_DEVICE_ALIAS": alias,
            "VK_DRIVER_FILES": str(vulkan),
            "VK_ICD_FILENAMES": str(vulkan),
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(egl),
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        }
    )
    return env


def _run_checked(
    command: list[str],
    *,
    label: str,
    env: Mapping[str, str] | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=None if env is None else dict(env),
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        raise RobotwinGpuPreflightError(f"{label} failed{suffix}") from exc


def _query_nvidia_gpu(
    gpu: int,
    *,
    nvidia_smi: str = "nvidia-smi",
) -> dict[str, Any]:
    result = _run_checked(
        [
            nvidia_smi,
            "--id",
            str(gpu),
            "--query-gpu=index,pci.bus_id,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        label=f"nvidia-smi query for GPU {gpu}",
    )
    rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
    _require(len(rows) == 1 and len(rows[0]) == 6, "nvidia-smi returned an invalid GPU row")
    row = [item.strip() for item in rows[0]]
    _require(row[0] == str(gpu), "nvidia-smi returned a different physical GPU index")
    try:
        total_mib = int(row[4])
        free_mib = int(row[5])
    except ValueError as exc:
        raise RobotwinGpuPreflightError(
            "nvidia-smi returned non-integer memory capacity"
        ) from exc
    _require(total_mib > 0 and 0 <= free_mib <= total_mib, "invalid GPU memory inventory")
    return {
        "physical_gpu_index": gpu,
        "pci_bus_id": canonical_nvidia_pci_address(row[1]),
        "gpu_name": row[2],
        "driver_version": row[3],
        "memory_total_mib": total_mib,
        "memory_free_mib_at_preflight": free_mib,
    }


def _check_vulkan(env: Mapping[str, str], *, vulkaninfo: str = "vulkaninfo") -> None:
    result = _run_checked(
        [vulkaninfo, "--summary"],
        label="NVIDIA Vulkan instance preflight",
        env=env,
        timeout=60,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    _require(
        re.search(r"deviceName\s*=.*NVIDIA|GPU[0-9]+:.*NVIDIA", combined, re.I)
        is not None,
        "Vulkan did not enumerate an NVIDIA rendering device",
    )


def _check_sapien_device(
    env: Mapping[str, str],
    *,
    python_executable: str,
    gpu: int,
    pci: str,
) -> dict[str, Any]:
    program = (
        "import json,sapien,sys; "
        "d=sapien.Device(sys.argv[1]); "
        "print('" + _SAPIEN_MARKER + "'+json.dumps({"
        "'name':str(d.name),'pci_bus_id':d.pci_string,'cuda_id':int(d.cuda_id),"
        "'can_render':bool(d.can_render()),"
        "'version':str(getattr(sapien,'__version__','unknown'))},sort_keys=True))"
    )
    result = _run_checked(
        [python_executable, "-c", program, f"pci:{pci}"],
        label=f"SAPIEN exact-device preflight for GPU {gpu}",
        env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
    )
    marker_lines = [
        line[len(_SAPIEN_MARKER) :]
        for line in result.stdout.splitlines()
        if line.startswith(_SAPIEN_MARKER)
    ]
    _require(len(marker_lines) == 1, "SAPIEN preflight did not return one device record")
    try:
        payload = json.loads(marker_lines[0])
    except Exception as exc:
        raise RobotwinGpuPreflightError("SAPIEN device record is invalid JSON") from exc
    _require(isinstance(payload, dict), "SAPIEN device record is not an object")
    actual_pci = canonical_nvidia_pci_address(str(payload.get("pci_bus_id", "")))
    _require(actual_pci == pci, "SAPIEN resolved a different physical PCI device")
    # CUDA_VISIBLE_DEVICES contains exactly one physical GPU, so SAPIEN must
    # expose the explicitly selected PCI device as logical CUDA device zero.
    _require(payload.get("cuda_id") == 0, "SAPIEN device is not logical CUDA device zero")
    _require(payload.get("can_render") is True, "SAPIEN device cannot render")
    return {
        "version": str(payload.get("version")),
        "device_name": str(payload.get("name")),
        "logical_cuda_id": int(payload["cuda_id"]),
        "pci_bus_id": actual_pci,
        "can_render": True,
    }


def preflight_gpu_runtime(
    gpu_id: str | int,
    *,
    python_executable: str = sys.executable,
    nvidia_smi: str = "nvidia-smi",
    vulkaninfo: str = "vulkaninfo",
    check_vulkan: bool = True,
    check_sapien: bool = True,
) -> dict[str, Any]:
    """Resolve and validate one online evaluation GPU without running a task."""

    gpu = normalize_physical_gpu_id(gpu_id)
    inventory = _query_nvidia_gpu(gpu, nvidia_smi=nvidia_smi)
    vulkan, egl = _runtime_paths()
    binding: dict[str, Any] = {
        "schema": "robotwin.policy_content_adapter.gpu_runtime_binding",
        "schema_version": 1,
        "status": "PASS",
        **inventory,
        "render_device_alias": f"pci:{inventory['pci_bus_id']}",
        "vulkan_icd": str(vulkan),
        "egl_vendor": str(egl),
        "vulkan_preflight": bool(check_vulkan),
        "sapien_preflight": bool(check_sapien),
    }
    env = gpu_binding_environment(binding)
    if check_vulkan:
        _check_vulkan(env, vulkaninfo=vulkaninfo)
    if check_sapien:
        binding["sapien"] = _check_sapien_device(
            env,
            python_executable=python_executable,
            gpu=gpu,
            pci=str(inventory["pci_bus_id"]),
        )
    return binding


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight", help="read-only NVIDIA Vulkan/SAPIEN device preflight"
    )
    preflight.add_argument("--gpu-id", required=True)
    preflight.add_argument("--skip-vulkan", action="store_true")
    preflight.add_argument("--skip-sapien", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command != "preflight":  # pragma: no cover - argparse enforces this
        raise SystemExit(2)
    result = preflight_gpu_runtime(
        args.gpu_id,
        check_vulkan=not args.skip_vulkan,
        check_sapien=not args.skip_sapien,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "RobotwinGpuPreflightError",
    "canonical_nvidia_pci_address",
    "gpu_binding_environment",
    "normalize_physical_gpu_id",
    "preflight_gpu_runtime",
]
