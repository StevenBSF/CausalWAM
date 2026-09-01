import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict

import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict


class Sapien_TEST(gym.Env):

    def __init__(self, render_device_alias=None):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        try:
            self.setup_scene(render_device_alias=render_device_alias)
            print("\033[32m" + "Render Well" + "\033[0m")
        except Exception:
            print("\033[31m" + "Render Error" + "\033[0m")
            # Do not hide renderer/Vulkan initialization failures or report
            # them to the shell as a successful process.  The traceback is
            # required to distinguish a missing GPU/driver from a SAPIEN
            # configuration error.
            raise

    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.
        """
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        scene_config = sapien.SceneConfig()
        sapien.physx.set_scene_config(scene_config)
        alias = kwargs.get("render_device_alias")
        device = sapien.Device(alias) if alias is not None else None
        if device is not None and not device.can_render():
            raise RuntimeError(f"SAPIEN device {alias!r} cannot render")
        render_system = sapien.render.RenderSystem(device)
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
        actual = self.scene.render_system.device
        physical_gpu_index = os.environ.get("ROBOTWIN_PHYSICAL_GPU_INDEX")
        if alias is not None:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            expected_pci = os.environ.get("ROBOTWIN_EXPECTED_GPU_PCI")
            if (
                physical_gpu_index is None
                or not physical_gpu_index.isdigit()
                or visible_devices != physical_gpu_index
                or expected_pci is None
                or alias != f"pci:{expected_pci}"
            ):
                raise RuntimeError(
                    "explicit SAPIEN rendering requires matching CUDA GPU index "
                    "and nvidia-smi PCI provenance environment values"
                )
            if (
                actual.cuda_id != device.cuda_id
                or actual.pci_string != device.pci_string
            ):
                raise RuntimeError(
                    "SAPIEN RenderSystem did not retain the requested device: "
                    f"requested={device}, actual={actual}"
                )
            if self.scene.render_system is not render_system:
                raise RuntimeError(
                    "SAPIEN scene did not retain the explicitly constructed RenderSystem"
                )
            actual_pci = actual.pci_string
            if not isinstance(actual_pci, str) or actual_pci.lower() != expected_pci.lower():
                raise RuntimeError(
                    "SAPIEN RenderSystem selected the wrong physical GPU: "
                    f"expected PCI {expected_pci}, actual PCI {actual_pci}"
                )
        self.render_device_info = {
            "requested_alias": alias,
            "physical_gpu_index": (
                int(physical_gpu_index) if physical_gpu_index is not None else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": str(actual.name),
            "logical_cuda_id": int(actual.cuda_id),
            "pci_string": (
                actual.pci_string.lower()
                if isinstance(actual.pci_string, str)
                else actual.pci_string
            ),
            "can_render": bool(actual.can_render()),
        }
        print(f"SAPIEN render device: {self.render_device_info}", flush=True)


if __name__ == "__main__":
    a = Sapien_TEST()
