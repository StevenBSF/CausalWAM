"""Policy Content Adapter for Motus.

The package is intentionally independent from the author training entrypoint.
Importing it does not load Motus backbones or touch CUDA.
"""

from .losses import multi_positive_supcon_loss
from .model import (
    GatedCrossAttentionAdapter,
    MotusContentHead,
    MotusPolicyContentConditioner,
)

__all__ = [
    "GatedCrossAttentionAdapter",
    "MotusContentHead",
    "MotusPolicyContentConditioner",
    "multi_positive_supcon_loss",
]

