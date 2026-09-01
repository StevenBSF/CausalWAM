"""Action-facing content adapter experiments for RoboTwin FastWAM.

The RoboTwin evaluator imports this package through a policy-directory
symlink, before the FastWAM repository has necessarily been added to
``sys.path``.  Keep package import side-effect free and resolve the optional
model API only when a caller actually asks for one of its symbols.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GatedCrossAttentionAdapter",
    "PolicyContentConditioner",
    "PolicyContentHead",
    "build_optimizer_param_groups",
    "configure_trainable_modules",
    "install_policy_content_adapter",
    "load_policy_checkpoint_into_model",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = import_module(".model", __name__)
    value = getattr(model, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
