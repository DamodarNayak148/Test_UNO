"""
vision_result.py — Unified detection result dataclasses for WALLE Vision.

All detectors write into this structure. The UI and any future consumer
(e.g. a re-attached Game Master) reads from here.

Backward compatible: all new fields carry default values so existing
code that constructs these dataclasses without keyword args still works.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Sub-result types
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """
    Face detection and perception result.
    Primary face detector: YuNet (cv2.FaceDetectorYN) for high-accuracy bounding boxes,
    confidence scores, and 5 keypoints.
    Primary face mesh landmarker: MediaPipe FaceLandmarker (478 landmarks) for eyelids,
    iris, EAR, MAR, smile, eyebrow elevation, and expression.
    """
    detected: bool = False
    count: int = 0

    # Bounding boxes for all detected faces: [(x, y, w, h), ...]
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Pixel centre of the primary (largest/first) face
    center: Tuple[int, int] = (0, 0)

    # Detector confidence (YuNet detection confidence score 0.0-1.0)
    confidence: float = 0.0

    # YuNet 5 2D keypoints [(x, y), ...] for primary face:
    # [right_eye, left_eye, nose_tip, right_mouth, left_mouth]
    keypoints: List[Tuple[int, int]] = field(default_factory=list)

    # --- MediaPipe FaceLandmarker additions ---
    # Raw NormalizedLandmark objects (up to 478 with attention mesh model)
    landmarks: List[Any] = field(default_factory=list)
    # Pixel-space (x, y) for each landmark — parallel to `landmarks`
    landmark_pixels: List[Tuple[int, int]] = field(default_factory=list)

    # Key feature positions in pixel space (None if not available)
    left_eye_center:  Optional[Tuple[int, int]] = None   # iris landmark 468
    right_eye_center: Optional[Tuple[int, int]] = None   # iris landmark 473
    nose_tip:         Optional[Tuple[int, int]] = None   # landmark 4 or YuNet keypoint

    # --- Facial Configuration & Expression ---
    mouth_open: bool = False
    mouth_ar: float = 0.0             # Mouth Aspect Ratio
    smile: bool = False
    eyebrows_raised: bool = False
    expression: str = "NEUTRAL"       # "NEUTRAL" | "SMILE" | "SURPRISED" | "ANGRY" | "SAD"
    detector_source: str = "YuNet"    # "YuNet" | "MediaPipe" | "Haar"


@dataclass
class EyeResult:
    """
    Eye landmark, EAR, and Blink detection result.
    Primary source: Iris and Eyelid landmarks from FaceLandmarker.
    Fallback: Haar eye cascade centres.
    """
    detected: bool = False

    # Pixel centres of iris / eye centres. None = not found.
    left:  Optional[Tuple[int, int]] = None
    right: Optional[Tuple[int, int]] = None

    # Eye contour landmark pixels for drawing (from face mesh)
    left_landmarks:  List[Tuple[int, int]] = field(default_factory=list)
    right_landmarks: List[Tuple[int, int]] = field(default_factory=list)

    # --- Eye State & Blink additions ---
    left_open: bool = False
    right_open: bool = False
    left_ear: float = 0.0             # Eye Aspect Ratio (left)
    right_ear: float = 0.0            # Eye Aspect Ratio (right)
    blink_detected: bool = False      # True during a blink transition
    blink_count: int = 0              # Cumulative blink counter

    @property
    def count(self) -> int:
        """Number of eyes detected (0, 1, or 2)."""
        return sum(1 for e in (self.left, self.right) if e is not None)


@dataclass
class HandResult:
    """
    MediaPipe Hands result for one hand (left or right).
    Uses Tasks API HandLandmarker (21 landmarks per hand).
    """
    detected: bool = False

    # Raw NormalizedLandmark objects (21 points)
    landmarks: List[Any] = field(default_factory=list)
    # Pixel-space (x, y) for each landmark
    landmark_pixels: List[Tuple[int, int]] = field(default_factory=list)

    # Finger extension state: [thumb, index, middle, ring, pinky] — True = extended
    finger_states: List[bool] = field(default_factory=lambda: [False] * 5)

    # Structured finger dict for readable access: {"thumb": bool, ...}
    fingers: Dict[str, bool] = field(default_factory=dict)

    # Number of currently extended fingers
    fingers_up: int = 0

    # MediaPipe handedness label: "Left" | "Right" | ""
    handedness: str = ""

    # Detection confidence from MediaPipe handedness score (0.0–1.0)
    confidence: float = 0.0

    # Key positions in pixel space
    wrist:  Optional[Tuple[int, int]] = None   # landmark 0
    center: Optional[Tuple[int, int]] = None   # palm centre (avg of MCP landmarks)


@dataclass
class PoseResult:
    """
    MediaPipe Pose detection result (33 landmarks).
    Includes basic pose classification from landmark geometry.
    """
    detected: bool = False

    # Raw landmark list (33 landmarks from MediaPipe Pose)
    landmarks: List[Any] = field(default_factory=list)
    # Pixel-space (x, y) for each pose landmark
    landmark_pixels: List[Tuple[int, int]] = field(default_factory=list)

    # --- Gesture / state flags ---
    left_hand_raised:  bool = False
    right_hand_raised: bool = False
    hands_raised:      bool = False   # both hands raised simultaneously

    # Body position classification (from knee/hip geometry)
    standing: bool = False
    sitting:  bool = False

    # Average visibility of key landmarks (0.0–1.0)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Top-level result container
# ---------------------------------------------------------------------------

@dataclass
class VisionResult:
    """
    Unified snapshot of a single processed camera frame.

    Produced by VisionEngine.process_frame() and consumed by the UI and
    any future reasoning layer (e.g., Game Master).
    """

    face:       FaceResult = field(default_factory=FaceResult)
    eyes:       EyeResult  = field(default_factory=EyeResult)
    left_hand:  HandResult = field(default_factory=HandResult)
    right_hand: HandResult = field(default_factory=HandResult)
    pose:       PoseResult = field(default_factory=PoseResult)

    avg_brightness: float = 0.0
    camera_dark:    bool  = False   # True when frame was too dark to process
    fps:            float = 0.0
    timestamp:      float = 0.0

    @staticmethod
    def dark_frame(brightness: float = 0.0, fps: float = 0.0) -> "VisionResult":
        """Return an all-undetected result for a dark/covered frame."""
        return VisionResult(avg_brightness=brightness, camera_dark=True, fps=fps)

    @staticmethod
    def empty(fps: float = 0.0) -> "VisionResult":
        """Return an all-undetected result (e.g. None frame)."""
        return VisionResult(fps=fps)
