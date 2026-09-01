"""Resolve one physical NVIDIA GPU for pinned RoboTwin rendering."""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

SCHEMA = "motus_policy_content_adapter_gpu_runtime_binding"
PCI = re.compile(r"([0-9A-Fa-f]{4,8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])")


def canonical_pci(value):
    m = PCI.fullmatch(str(value).strip())
    if not m:
        raise ValueError("invalid PCI address")
    return f"{int(m[1], 16):04x}:{m[2].lower()}:{m[3].lower()}.{m[4]}"


def validate_binding(value):
    if value.get("schema") != SCHEMA or value.get("status") != "PASS":
        raise ValueError("invalid binding")
    if value.get("render_device_alias") != f"pci:{canonical_pci(value['pci_bus_id'])}":
        raise ValueError("PCI alias mismatch")
    for key in ("vulkan_icd", "egl_vendor"):
        if not Path(value[key]).is_file():
            raise FileNotFoundError(value[key])
    return dict(value)


def preflight(gpu):
    cmd = [
        "nvidia-smi",
        "--id",
        str(gpu),
        "--query-gpu=index,pci.bus_id,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    row = next(csv.reader(subprocess.check_output(cmd, text=True).splitlines()))
    row = [x.strip() for x in row]
    if row[0] != str(gpu):
        raise RuntimeError("GPU mismatch")
    pci = canonical_pci(row[1])
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "physical_gpu_index": gpu,
        "pci_bus_id": pci,
        "render_device_alias": f"pci:{pci}",
        "gpu_name": row[2],
        "driver_version": row[3],
        "memory_total_mib": int(row[4]),
        "memory_free_mib_at_preflight": int(row[5]),
        "vulkan_icd": os.getenv(
            "MOTUS_NVIDIA_VULKAN_ICD", "/etc/vulkan/icd.d/nvidia_icd.json"
        ),
        "egl_vendor": os.getenv(
            "MOTUS_NVIDIA_EGL_VENDOR", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        ),
    }
    return validate_binding(result)


def binding_environment(value, base=None):
    value = validate_binding(value)
    env = dict(os.environ if base is None else base)
    gpu = value["physical_gpu_index"]
    pci = canonical_pci(value["pci_bus_id"])
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MOTUS_ROBOTWIN_PHYSICAL_GPU": str(gpu),
            "MOTUS_ROBOTWIN_EXPECTED_PCI": pci,
            "MOTUS_ROBOTWIN_RENDER_ALIAS": f"pci:{pci}",
            "VK_DRIVER_FILES": value["vulkan_icd"],
            "VK_ICD_FILENAMES": value["vulkan_icd"],
            "__EGL_VENDOR_LIBRARY_FILENAMES": value["egl_vendor"],
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        }
    )
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-id", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    value = preflight(a.gpu_id)
    env = binding_environment(value)
    text = subprocess.check_output(["vulkaninfo", "--summary"], text=True, env=env)
    if "NVIDIA" not in text:
        raise RuntimeError("Vulkan has no NVIDIA GPU")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
