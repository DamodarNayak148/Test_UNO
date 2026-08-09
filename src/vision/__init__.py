"""
src/vision — WALLE Real-Time Vision Perception Module

Exports:
    VisionEngine   — main per-frame processor
    VisionResult   — unified detection result dataclass
"""

from src.vision.vision_result import (
    VisionResult, FaceResult, EyeResult, HandResult, PoseResult
)
from src.vision.vision_engine import VisionEngine

__all__ = [
    "VisionEngine",
    "VisionResult",
    "FaceResult",
    "EyeResult",
    "HandResult",
    "PoseResult",
]
