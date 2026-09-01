"""Immutable first-version contracts for the Motus adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PROTOCOL_ID = "motus_policy_content_adapter_v1"
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
VARIANTS = ("C", "R1", "R2", "R3")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CONTROLS = ("m1_architecture_action_control", "m3_ours")
REGIMES = ("m_p1", "m_p2")

DEFAULT_CAPTURE_LAYER = 16
DEFAULT_BACKBONE_DIM = 3072
DEFAULT_CONTENT_DIM = 384
DEFAULT_CONTENT_QUERIES = 8
DEFAULT_CONTENT_HEADS = 8
DEFAULT_ACTION_DIM = 1024
DEFAULT_ACTION_HEADS = 8
DEFAULT_TEMPERATURE = 0.07
DEFAULT_LAMBDA_CONTRASTIVE = 0.1

PAIRED_STATE_COUNT = 720
PAIRED_VIEW_COUNT = 4
PAIRED_SCENE_COUNT = 2880
MOTUS_IMAGE_HEIGHT = 384
MOTUS_IMAGE_WIDTH = 320
MOTUS_ACTION_CHUNK = 16
MOTUS_ACTION_DIM = 14


class ProtocolError(ValueError):
    """A requested run violates the Motus adapter protocol."""


@dataclass(frozen=True)
class AdapterArchitecture:
    capture_layer: int = DEFAULT_CAPTURE_LAYER
    backbone_dim: int = DEFAULT_BACKBONE_DIM
    content_dim: int = DEFAULT_CONTENT_DIM
    content_queries: int = DEFAULT_CONTENT_QUERIES
    content_heads: int = DEFAULT_CONTENT_HEADS
    action_dim: int = DEFAULT_ACTION_DIM
    action_heads: int = DEFAULT_ACTION_HEADS

    def validate(self) -> "AdapterArchitecture":
        if self.capture_layer <= 0:
            raise ProtocolError("capture_layer is one-based and must be positive")
        for name in ("backbone_dim", "content_dim", "content_queries", "action_dim"):
            if int(getattr(self, name)) <= 0:
                raise ProtocolError(f"{name} must be positive")
        if self.content_dim % self.content_heads:
            raise ProtocolError("content_heads must divide content_dim")
        if self.action_dim % self.action_heads:
            raise ProtocolError("action_heads must divide action_dim")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_control(*, control: str, lambda_contrastive: float) -> None:
    if control not in CONTROLS:
        raise ProtocolError(f"unsupported control {control!r}")
    value = float(lambda_contrastive)
    if control == "m1_architecture_action_control" and value != 0.0:
        raise ProtocolError("M1 must use lambda_contrastive=0")
    if control == "m3_ours" and value <= 0.0:
        raise ProtocolError("M3 must use a positive contrastive weight")

