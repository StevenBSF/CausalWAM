from experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime import (
    binding_environment,
    canonical_pci,
    validate_binding,
)


def test_pci_normalization_and_environment(tmp_path):
    vulkan = tmp_path / "nvidia.json"
    egl = tmp_path / "egl.json"
    vulkan.write_text("{}")
    egl.write_text("{}")
    value = {
        "schema": "motus_policy_content_adapter_gpu_runtime_binding",
        "schema_version": 1,
        "status": "PASS",
        "physical_gpu_index": 7,
        "pci_bus_id": "00000000:AF:0B.0",
        "render_device_alias": "pci:0000:af:0b.0",
        "vulkan_icd": str(vulkan),
        "egl_vendor": str(egl),
    }
    assert canonical_pci("00000000:AF:0B.0") == "0000:af:0b.0"
    assert validate_binding(value)["physical_gpu_index"] == 7
    env = binding_environment(value, {})
    assert (
        env["CUDA_VISIBLE_DEVICES"] == "7"
        and env["MOTUS_ROBOTWIN_EXPECTED_PCI"] == "0000:af:0b.0"
    )
