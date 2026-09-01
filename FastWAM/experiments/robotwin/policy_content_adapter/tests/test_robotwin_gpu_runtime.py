from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.robotwin.policy_content_adapter import pinned_eval_policy
from experiments.robotwin.policy_content_adapter import robotwin_gpu_runtime


def _binding(tmp_path, *, gpu: int = 1) -> dict:
    vulkan = tmp_path / "nvidia_icd.json"
    egl = tmp_path / "10_nvidia.json"
    vulkan.write_text("{}\n", encoding="utf-8")
    egl.write_text("{}\n", encoding="utf-8")
    return {
        "physical_gpu_index": gpu,
        "pci_bus_id": f"0000:00:0{gpu + 1}.0",
        "render_device_alias": f"pci:0000:00:0{gpu + 1}.0",
        "vulkan_icd": str(vulkan),
        "egl_vendor": str(egl),
    }


def test_pci_normalization_and_numeric_gpu_are_fail_closed() -> None:
    assert (
        robotwin_gpu_runtime.canonical_nvidia_pci_address("00000000:0A:0B.3")
        == "0000:0a:0b.3"
    )
    assert robotwin_gpu_runtime.normalize_physical_gpu_id("7") == 7
    with pytest.raises(robotwin_gpu_runtime.RobotwinGpuPreflightError):
        robotwin_gpu_runtime.normalize_physical_gpu_id("0,1")
    with pytest.raises(robotwin_gpu_runtime.RobotwinGpuPreflightError):
        robotwin_gpu_runtime.canonical_nvidia_pci_address("GPU-uuid")


def test_binding_environment_pins_cuda_vulkan_and_sapien(tmp_path) -> None:
    binding = _binding(tmp_path)
    env = robotwin_gpu_runtime.gpu_binding_environment(
        binding, base_environment={"KEEP": "yes"}
    )
    assert env["KEEP"] == "yes"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["ROBOTWIN_PHYSICAL_GPU_INDEX"] == "1"
    assert env["ROBOTWIN_EXPECTED_GPU_PCI"] == "0000:00:02.0"
    assert env["ROBOTWIN_RENDER_DEVICE_ALIAS"] == "pci:0000:00:02.0"
    assert env["VK_DRIVER_FILES"] == env["VK_ICD_FILENAMES"]
    assert env["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"


def test_preflight_combines_exact_device_checks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vulkan = tmp_path / "nvidia_icd.json"
    egl = tmp_path / "10_nvidia.json"
    vulkan.write_text("{}\n", encoding="utf-8")
    egl.write_text("{}\n", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(robotwin_gpu_runtime, "_runtime_paths", lambda: (vulkan, egl))
    monkeypatch.setattr(
        robotwin_gpu_runtime,
        "_query_nvidia_gpu",
        lambda gpu, nvidia_smi="nvidia-smi": {
            "physical_gpu_index": gpu,
            "pci_bus_id": "0000:00:01.0",
            "gpu_name": "NVIDIA test GPU",
            "driver_version": "580.1",
            "memory_total_mib": 72_000,
            "memory_free_mib_at_preflight": 70_000,
        },
    )
    monkeypatch.setattr(
        robotwin_gpu_runtime,
        "_check_vulkan",
        lambda env, vulkaninfo="vulkaninfo": calls.append("vulkan"),
    )
    monkeypatch.setattr(
        robotwin_gpu_runtime,
        "_check_sapien_device",
        lambda env, python_executable, gpu, pci: {
            "version": "3.0.0b1",
            "device_name": "NVIDIA test GPU",
            "logical_cuda_id": 0,
            "pci_bus_id": pci,
            "can_render": True,
        },
    )
    result = robotwin_gpu_runtime.preflight_gpu_runtime(0)
    assert result["status"] == "PASS"
    assert result["render_device_alias"] == "pci:0000:00:01.0"
    assert result["sapien"]["can_render"] is True
    assert calls == ["vulkan"]


def test_pinned_launcher_injects_alias_into_every_setup_demo(tmp_path) -> None:
    binding = _binding(tmp_path, gpu=0)
    env = robotwin_gpu_runtime.gpu_binding_environment(
        binding, base_environment={}
    )
    assert pinned_eval_policy.validate_pinned_environment(env) == {
        "physical_gpu_index": 0,
        "pci_bus_id": "0000:00:01.0",
        "render_device_alias": "pci:0000:00:01.0",
        "vulkan_icd": str((tmp_path / "nvidia_icd.json").resolve()),
        "egl_vendor": str((tmp_path / "10_nvidia.json").resolve()),
    }

    observed: list[dict] = []

    class FakeTask:
        def setup_demo(self, *args, **kwargs):
            observed.append(dict(kwargs))
            return "ok"

    module = SimpleNamespace(class_decorator=lambda task_name: FakeTask())
    pinned_eval_policy.install_setup_demo_pin(
        module, render_device_alias="pci:0000:00:01.0"
    )
    task = module.class_decorator("place_a2b_left")
    assert task.setup_demo(seed=4) == "ok"
    assert task.setup_demo(seed=5, render_device_alias="pci:0000:00:01.0") == "ok"
    assert [item["render_device_alias"] for item in observed] == [
        "pci:0000:00:01.0",
        "pci:0000:00:01.0",
    ]
    with pytest.raises(pinned_eval_policy.PinnedEvalRuntimeError):
        task.setup_demo(render_device_alias="pci:0000:00:02.0")
