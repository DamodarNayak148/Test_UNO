"""
vision_engine.py — WALLE Real-Time Vision Engine.

Runs all detectors on every camera frame and returns:
  - An annotated BGR frame ready for display
  - A VisionResult with all detection data

Detector pipeline (each isolated -- one crash never halts the others):
  1. Brightness guard         -- dark/covered-camera early exit
  2. MediaPipe FaceLandmarker -- 478-landmark face mesh + iris centres
     + Eye Aspect Ratio (EAR) open/closed detection
     + State-machine Blink detection
     + Facial configuration & expression recognition (Smile, Mouth Open, Eyebrows)
  3. Eye landmarks            -- derived from face mesh (iris 468/473, EAR contours)
  4. MediaPipe HandLandmarker -- 21-landmark hand model, left + right
  5. Finger detection         -- 3D orientation-invariant landmark geometry (Thumb & Fingers 1-4)
  6. MediaPipe PoseLandmarker -- 33-landmark body skeleton + pose classification

Architecture preserved:
  - HAL camera is injected; VisionEngine NEVER owns or opens a camera stream
  - cv2.VideoCapture() is NEVER called here
  - Runs on frames from any HAL camera (SimCamera, UNOQCamera, etc.)
  - Tasks API is tried first, legacy mp.solutions fallback where needed
  - Compatible with mediapipe==0.10.35
"""

# Suppress C++ backend (glog / TFLite) INFO and WARNING messages.
import os as _os
_os.environ.setdefault("GLOG_minloglevel",    "2")   # show C++ ERROR+ only
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")   # show TFLite ERROR+ only

import math
import os
import time
import threading
import urllib.request
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from src.vision.vision_result import (
    VisionResult, FaceResult, EyeResult, HandResult, PoseResult,
)

# ---------------------------------------------------------------------------
# Drawing colour palette (BGR)
# ---------------------------------------------------------------------------
_COL_FACE_MESH   = (0,   200,  70)   # Green      -- face mesh / oval
_COL_FACE_BOX    = (0,   220,  80)   # Bright green -- face bounding box
_COL_EYE         = (0,   255, 255)   # Cyan       -- eye contour / iris
_COL_EYE_CLOSED  = (0,   100, 255)   # Orange-Red -- closed eye indicator
_COL_EYEBROW     = (120, 220, 255)   # Light blue -- eyebrows
_COL_NOSE        = (180, 255, 180)   # Pale green -- nose
_COL_LIPS        = (100, 160, 255)   # Soft red   -- lips
_COL_HAND_L      = (0,   165, 255)   # Orange     -- left hand
_COL_HAND_R      = (255, 200,   0)   # Yellow     -- right hand
_COL_POSE        = (180, 180, 255)   # Lilac      -- pose skeleton
_COL_FINGER_UP   = (0,   255, 120)   # Bright green -- extended finger tip
_COL_FINGER_DOWN = (60,   60, 100)   # Dark       -- curled finger tip
_COL_TEXT        = (255, 255, 255)   # White      -- labels

# ---------------------------------------------------------------------------
# MediaPipe hand landmark connection pairs (21 landmarks)
# ---------------------------------------------------------------------------
_HAND_CONNECTIONS = [
    (0, 1),(1, 2),(2, 3),(3, 4),          # Thumb
    (0, 5),(5, 6),(6, 7),(7, 8),          # Index
    (0, 9),(9,10),(10,11),(11,12),         # Middle
    (0,13),(13,14),(14,15),(15,16),        # Ring
    (0,17),(17,18),(18,19),(19,20),        # Pinky
    (5, 9),(9,13),(13,17),                 # Palm transversals
]

# Finger landmark indices
_FINGER_TIPS = [4, 8, 12, 16, 20]   # [thumb, index, middle, ring, pinky]
_FINGER_PIPS = [3, 6, 10, 14, 18]
_FINGER_MCPS = [2, 5,  9, 13, 17]
_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# ---------------------------------------------------------------------------
# MediaPipe model URLs and local cache paths
# ---------------------------------------------------------------------------
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_POSE_MODEL_PATH = "models/pose_landmarker_lite.task"

_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_MODEL_PATH = "models/hand_landmarker.task"

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_FACE_MODEL_PATH = "models/face_landmarker.task"

# ---------------------------------------------------------------------------
# Key pose landmark indices (MediaPipe Pose 33-landmark set)
# ---------------------------------------------------------------------------
_POSE_IDX = {
    "nose":       0,
    "l_ear":      7,  "r_ear":      8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}

_POSE_CONNECTIONS = [
    (7, 8),                         # ear-ear
    (11,12),                        # shoulder-shoulder
    (11,13),(13,15),                 # left arm
    (12,14),(14,16),                 # right arm
    (11,23),(12,24),                 # torso sides
    (23,24),                        # hips
    (23,25),(25,27),                 # left leg
    (24,26),(26,28),                 # right leg
]

# ---------------------------------------------------------------------------
# Face mesh landmark indices (MediaPipe 478-landmark model)
# ---------------------------------------------------------------------------
_FACE_LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246, 33]
_FACE_RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                     466, 388, 387, 386, 385, 384, 398, 362]
_FACE_LEFT_BROW  = [276, 283, 282, 295, 285, 336, 296, 334]
_FACE_RIGHT_BROW = [46,  53,  52,  65,  55, 107,  66, 105]
_FACE_NOSE       = [168, 6, 197, 195, 5, 4, 1, 2, 98, 97, 326, 327]
_FACE_LIPS_OUT   = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                     375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
_FACE_OVAL       = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                     361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                     176, 149, 150, 136, 172,  58, 132,  93, 234, 127,
                     162,  21,  54, 103,  67, 109, 10]
_FACE_LEFT_IRIS_IDX  = 468
_FACE_RIGHT_IRIS_IDX = 473
_FACE_NOSE_TIP_IDX   = 4

# Eye Aspect Ratio (EAR) 6-point landmark indices
# Left eye (person's left / camera right)
_EAR_LEFT_CORNER1 = 362
_EAR_LEFT_CORNER2 = 263
_EAR_LEFT_TOP1    = 385
_EAR_LEFT_BOT1    = 380
_EAR_LEFT_TOP2    = 387
_EAR_LEFT_BOT2    = 373

# Right eye (person's right / camera left)
_EAR_RIGHT_CORNER1 = 33
_EAR_RIGHT_CORNER2 = 133
_EAR_RIGHT_TOP1    = 160
_EAR_RIGHT_BOT1    = 144
_EAR_RIGHT_TOP2    = 158
_EAR_RIGHT_BOT2    = 153

# Mouth & eyebrow landmark indices
_MOUTH_TOP    = 13
_MOUTH_BOT    = 14
_MOUTH_LEFT   = 61
_MOUTH_RIGHT  = 291
_BROW_R_CTR   = 105
_EYE_R_TOP    = 159
_BROW_L_CTR   = 334
_EYE_L_TOP    = 386
_FACE_TOP     = 10
_FACE_BOT     = 152
_FACE_L_SIDE  = 234
_FACE_R_SIDE  = 454


# ---------------------------------------------------------------------------
# Temporal smoothing helper
# ---------------------------------------------------------------------------

class DetectionHistory:
    """
    Rolling-window boolean smoother for detector on/off flickering.

    A detector is considered "active" only when it has fired in at least
    `min_hits` of the last `window` frames.
    """

    def __init__(self, window: int = 5, min_hits: int = 2) -> None:
        self._history: deque = deque([False] * window, maxlen=window)
        self._min_hits = min_hits

    def push(self, detected: bool) -> bool:
        """Record this frame's detection state; return smoothed bool."""
        self._history.append(detected)
        return int(sum(self._history)) >= self._min_hits

    def reset(self) -> None:
        """Clear history (e.g. when camera stream restarts)."""
        for i in range(len(self._history)):
            self._history[i] = False


# ---------------------------------------------------------------------------
# VisionEngine
# ---------------------------------------------------------------------------

class VisionEngine:
    """
    Continuous real-time vision processor.

    Usage::
        engine = VisionEngine()
        annotated_frame, result = engine.process_frame(bgr_frame)

    Thread safety:
        process_frame() is NOT thread-safe by design. Call it only from
        the Qt timer callback (single GUI thread).

    HAL contract:
        VisionEngine NEVER calls cv2.VideoCapture().
        Frames come from the HAL camera (SimCamera, UNOQCamera, etc.)
        and are passed in via process_frame(bgr_frame).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # FPS tracking
        self._fps: float = 0.0
        self._frame_times: List[float] = []
        self._fps_window = 30

        # ---- Temporal smoothers (5-frame window, 2-hit threshold) ----
        self._face_smoother = DetectionHistory(window=5, min_hits=2)
        self._lh_smoother   = DetectionHistory(window=5, min_hits=2)
        self._rh_smoother   = DetectionHistory(window=5, min_hits=2)
        self._pose_smoother = DetectionHistory(window=5, min_hits=2)

        # ---- Blink state machine tracking ----
        self._blink_count: int = 0
        self._closed_frames: int = 0
        self._blink_active_frames: int = 0  # Hold blink_detected True for 2 frames so UI/user can see it

        # ---- Detector state ----
        self._mp: Any = None

        # Haar (kept, loaded, but NOT called in process_frame)
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._eye_cascade:  Optional[cv2.CascadeClassifier] = None
        self._init_haar()

        # MediaPipe detectors (Tasks API)
        self._face_landmarker: Any = None
        self._pose_landmarker: Any = None
        self._pose_is_tasks_api: bool = False
        self._hands_solution: Any = None
        self._hands_is_tasks_api: bool = False

        self._init_mediapipe()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_haar(self) -> None:
        """Load Haar cascades (kept as fallback; not active in main loop)."""
        try:
            base = getattr(cv2.data, "haarcascades", "")
            fc = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
            if not fc.empty():
                self._face_cascade = fc
                print("[VisionEngine] Haar face cascade loaded (standby).")
            ec = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
            if not ec.empty():
                self._eye_cascade = ec
                print("[VisionEngine] Haar eye cascade loaded (standby).")
        except Exception as e:
            print(f"[VisionEngine] Haar init warning: {e}")

    def _init_mediapipe(self) -> None:
        """Import MediaPipe and initialise all three Tasks API detectors."""
        try:
            import sys
            import mediapipe as mp
            self._mp = mp

            try:
                import absl.logging as _absl
                _absl.set_verbosity(_absl.ERROR)
            except Exception:
                pass

            print(f"[VisionEngine] MediaPipe imported successfully: {mp.__version__}")
            print(f"[VisionEngine] Python executable: {sys.executable}")
            print(f"[VisionEngine] Python version:    {sys.version.split()[0]}")

            self._init_face_landmarker(mp)
            self._init_pose(mp)
            self._init_hands(mp)

        except ImportError as e:
            import sys
            print(f"[VisionEngine] MediaPipe import FAILED: {type(e).__name__}: {e}")
            print(f"[VisionEngine] Python executable: {sys.executable}")
            print(f"[VisionEngine] Python version:    {sys.version}")
            print("[VisionEngine] Pose, Hands and Face Landmarker are disabled.")
            print(f"[VisionEngine] Fix: {sys.executable} -m pip install mediapipe==0.10.35")
        except Exception as e:
            import sys, traceback
            print(f"[VisionEngine] MediaPipe init error: {type(e).__name__}: {e}")
            print(f"[VisionEngine] Python executable: {sys.executable}")
            traceback.print_exc()

    def _init_face_landmarker(self, mp: Any) -> None:
        """
        Initialise MediaPipe FaceLandmarker (Tasks API).
        Provides 478-landmark attention-mesh including iris landmarks 468/473.
        This replaces the Haar face cascade in the active detection path.
        """
        if not hasattr(mp, "tasks"):
            print("[VisionEngine] FaceLandmarker: Tasks API not available.")
            return
        try:
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core import base_options as bo

            if not os.path.exists(_FACE_MODEL_PATH):
                os.makedirs(os.path.dirname(_FACE_MODEL_PATH), exist_ok=True)
                print(f"[VisionEngine] Downloading face model -> {_FACE_MODEL_PATH} ...")
                urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)

            options = mp_vision.FaceLandmarkerOptions(
                base_options=bo.BaseOptions(model_asset_path=_FACE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=3,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            print("[VisionEngine] Face Landmarker (Tasks API) initialized.")
        except Exception as e:
            print(f"[VisionEngine] Face Landmarker init FAILED: {type(e).__name__}: {e}")

    def _init_pose(self, mp: Any) -> None:
        """Try Tasks API PoseLandmarker, fall back to legacy mp.solutions.pose."""
        if hasattr(mp, "tasks"):
            try:
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks.python.core import base_options as bo

                if not os.path.exists(_POSE_MODEL_PATH):
                    os.makedirs(os.path.dirname(_POSE_MODEL_PATH), exist_ok=True)
                    print(f"[VisionEngine] Downloading pose model -> {_POSE_MODEL_PATH} ...")
                    urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL_PATH)

                options = mp_vision.PoseLandmarkerOptions(
                    base_options=bo.BaseOptions(model_asset_path=_POSE_MODEL_PATH),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                )
                self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                self._pose_is_tasks_api = True
                print("[VisionEngine] Pose Landmarker (Tasks API) initialized.")
                return
            except Exception as e:
                print(f"[VisionEngine] Tasks API pose failed: {e} -- trying legacy.")

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                self._pose_landmarker = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_is_tasks_api = False
                print("[VisionEngine] Pose Landmarker (legacy solutions) initialized.")
            except Exception as e:
                print(f"[VisionEngine] Legacy pose init failed: {e}")

    def _init_hands(self, mp: Any) -> None:
        """Initialise MediaPipe Hands (Tasks API preferred, legacy fallback)."""
        if hasattr(mp, "tasks"):
            try:
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks.python.core import base_options as bo

                if not os.path.exists(_HAND_MODEL_PATH):
                    os.makedirs(os.path.dirname(_HAND_MODEL_PATH), exist_ok=True)
                    print(f"[VisionEngine] Downloading hand model -> {_HAND_MODEL_PATH} ...")
                    urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_PATH)

                options = mp_vision.HandLandmarkerOptions(
                    base_options=bo.BaseOptions(model_asset_path=_HAND_MODEL_PATH),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._hands_solution = mp_vision.HandLandmarker.create_from_options(options)
                self._hands_is_tasks_api = True
                print("[VisionEngine] Hand Landmarker (Tasks API) initialized.")
                return
            except Exception as e:
                print(f"[VisionEngine] Tasks API hands failed: {e} -- trying legacy.")

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self._hands_solution = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._hands_is_tasks_api = False
                print("[VisionEngine] Hand Landmarker (legacy solutions) initialized.")
            except Exception as e:
                print(f"[VisionEngine] Hands init failed: {e}")
        else:
            print("[VisionEngine] No MediaPipe Hands API available.")

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, VisionResult]:
        """
        Process one BGR camera frame.

        Returns:
            annotated -- BGR frame with all overlays drawn (safe to display)
            result    -- VisionResult with all detection data

        Never raises. Returns a blank frame on any error.
        """
        fps = self._update_fps()

        # Guard: None / malformed frame
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "NO CAMERA FRAME", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
            return blank, VisionResult.empty(fps=fps)

        annotated = frame.copy()

        # Brightness guard
        try:
            hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
            avg_brightness = float(np.mean(hsv[:, :, 2]))
        except Exception:
            avg_brightness = 128.0

        if avg_brightness < 35.0:
            cv2.putText(
                annotated,
                f"CAMERA COVERED / DARK  (V={avg_brightness:.0f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2,
            )
            return annotated, VisionResult.dark_frame(brightness=avg_brightness, fps=fps)

        # ---------- Run detectors (each fully isolated) ----------
        face_result, eye_result = self._detect_faces_mp(annotated, frame)
        lh, rh                  = self._detect_hands(annotated, frame)
        pose_result             = self._detect_pose(annotated, frame)

        # ---------- Apply temporal smoothing to 'detected' flags ----------
        face_result.detected = self._face_smoother.push(face_result.detected)
        lh.detected          = self._lh_smoother.push(lh.detected)
        rh.detected          = self._rh_smoother.push(rh.detected)
        pose_result.detected = self._pose_smoother.push(pose_result.detected)

        # Eyes are derived from face -- clear if face not smoothed-active
        if not face_result.detected:
            eye_result = EyeResult(blink_count=self._blink_count)
            self._closed_frames = 0

        # ---------- FPS overlay ----------
        self._draw_fps(annotated, fps)

        result = VisionResult(
            face=face_result,
            eyes=eye_result,
            left_hand=lh,
            right_hand=rh,
            pose=pose_result,
            avg_brightness=avg_brightness,
            camera_dark=False,
            fps=fps,
            timestamp=time.time(),
        )
        return annotated, result

    # ------------------------------------------------------------------
    # Detector 1: MediaPipe FaceLandmarker + Eye/Blink/Expression Analysis
    # ------------------------------------------------------------------

    def _detect_faces_mp(
        self, annotated: np.ndarray, original: np.ndarray
    ) -> Tuple[FaceResult, EyeResult]:
        """
        Run MediaPipe FaceLandmarker.
        Returns (FaceResult, EyeResult) derived from the face mesh.
        """
        if self._face_landmarker is not None:
            return self._detect_faces_landmarker(annotated, original)

        if self._face_cascade is not None:
            return self._detect_faces_haar(annotated), EyeResult()

        return FaceResult(), EyeResult()

    def _detect_faces_landmarker(
        self, annotated: np.ndarray, original: np.ndarray
    ) -> Tuple[FaceResult, EyeResult]:
        """MediaPipe FaceLandmarker path (Tasks API)."""
        try:
            h, w = original.shape[:2]
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb
            )
            result = self._face_landmarker.detect(mp_img)

            if not result or not result.face_landmarks:
                # Reset closed frames on no face detected
                self._closed_frames = 0
                return FaceResult(), EyeResult(blink_count=self._blink_count)

            face_data = []
            for face_idx, face_lms in enumerate(result.face_landmarks):
                pixels = [
                    (int(lm.x * w), int(lm.y * h)) for lm in face_lms
                ]

                # Bounding box from landmark extents
                xs = [p[0] for p in pixels]
                ys = [p[1] for p in pixels]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                left_iris  = pixels[_FACE_LEFT_IRIS_IDX]  if _FACE_LEFT_IRIS_IDX  < len(pixels) else None
                right_iris = pixels[_FACE_RIGHT_IRIS_IDX] if _FACE_RIGHT_IRIS_IDX < len(pixels) else None
                nose_tip   = pixels[_FACE_NOSE_TIP_IDX]   if _FACE_NOSE_TIP_IDX   < len(pixels) else None

                # Compute EAR (Eye Aspect Ratio) for Left & Right eyes
                left_ear  = self._compute_ear(face_lms, _EAR_LEFT_CORNER1, _EAR_LEFT_CORNER2,
                                             _EAR_LEFT_TOP1, _EAR_LEFT_BOT1, _EAR_LEFT_TOP2, _EAR_LEFT_BOT2)
                right_ear = self._compute_ear(face_lms, _EAR_RIGHT_CORNER1, _EAR_RIGHT_CORNER2,
                                              _EAR_RIGHT_TOP1, _EAR_RIGHT_BOT1, _EAR_RIGHT_TOP2, _EAR_RIGHT_BOT2)

                left_open  = left_ear > 0.20
                right_open = right_ear > 0.20

                # Compute MAR (Mouth Aspect Ratio), Smile, Eyebrows & Expression
                mouth_ar, mouth_open = self._compute_mouth(face_lms)
                smile = self._compute_smile(face_lms, mouth_ar)
                eyebrows_raised, eyebrow_ratio = self._compute_eyebrows(face_lms)
                expression = self._classify_expression(face_lms, smile, mouth_open, eyebrows_raised, eyebrow_ratio)

                # Draw face mesh contours and expression overlay
                self._draw_face_mesh(annotated, pixels, face_idx == 0, expression, left_open, right_open)

                face_data.append({
                    "bbox":            bbox,
                    "pixels":          pixels,
                    "landmarks":       list(face_lms),
                    "left_iris":       left_iris,
                    "right_iris":      right_iris,
                    "nose_tip":        nose_tip,
                    "left_ear":        left_ear,
                    "right_ear":       right_ear,
                    "left_open":       left_open,
                    "right_open":      right_open,
                    "mouth_ar":        mouth_ar,
                    "mouth_open":      mouth_open,
                    "smile":           smile,
                    "eyebrows_raised": eyebrows_raised,
                    "expression":      expression,
                })

            if not face_data:
                return FaceResult(), EyeResult(blink_count=self._blink_count)

            primary = face_data[0]
            bx, by, bw, bh = primary["bbox"]
            cx = bx + bw // 2
            cy = by + bh // 2

            # --- Blink State Machine ---
            left_open  = primary["left_open"]
            right_open = primary["right_open"]
            both_closed = (not left_open) and (not right_open)

            blink_detected = False
            if both_closed:
                self._closed_frames += 1
            else:
                # If eyes just re-opened after 1..10 closed frames, count a blink!
                if 1 <= self._closed_frames <= 10:
                    self._blink_count += 1
                    self._blink_active_frames = 3  # Display blink_detected=True for 3 frames
                self._closed_frames = 0

            if self._blink_active_frames > 0:
                blink_detected = True
                self._blink_active_frames -= 1

            face_result = FaceResult(
                detected=True,
                count=len(face_data),
                bounding_boxes=[d["bbox"] for d in face_data],
                center=(cx, cy),
                landmarks=primary["landmarks"],
                landmark_pixels=primary["pixels"],
                left_eye_center=primary["left_iris"],
                right_eye_center=primary["right_iris"],
                nose_tip=primary["nose_tip"],
                confidence=1.0,
                mouth_open=primary["mouth_open"],
                mouth_ar=primary["mouth_ar"],
                smile=primary["smile"],
                eyebrows_raised=primary["eyebrows_raised"],
                expression=primary["expression"],
            )

            pxs = primary["pixels"]
            left_eye_px  = [pxs[i] for i in _FACE_LEFT_EYE  if i < len(pxs)]
            right_eye_px = [pxs[i] for i in _FACE_RIGHT_EYE if i < len(pxs)]

            eye_result = EyeResult(
                detected=True,
                left=primary["left_iris"],
                right=primary["right_iris"],
                left_landmarks=left_eye_px,
                right_landmarks=right_eye_px,
                left_open=primary["left_open"],
                right_open=primary["right_open"],
                left_ear=primary["left_ear"],
                right_ear=primary["right_ear"],
                blink_detected=blink_detected,
                blink_count=self._blink_count,
            )

            return face_result, eye_result

        except Exception as e:
            print(f"[VisionEngine] FaceLandmarker detect error: {e}")
            return FaceResult(), EyeResult(blink_count=self._blink_count)

    # ------------------------------------------------------------------
    # Facial Landmark Analysis Helpers (EAR, MAR, Smile, Eyebrows, Expression)
    # ------------------------------------------------------------------

    @staticmethod
    def _dist_3d(p1: Any, p2: Any) -> float:
        """3D Euclidean distance between two NormalizedLandmark objects."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _compute_ear(
        self,
        landmarks: List[Any],
        c1_idx: int, c2_idx: int,
        t1_idx: int, b1_idx: int,
        t2_idx: int, b2_idx: int,
    ) -> float:
        """Calculate Eye Aspect Ratio (EAR) for 6 landmark points."""
        try:
            if max(c1_idx, c2_idx, t1_idx, b1_idx, t2_idx, b2_idx) >= len(landmarks):
                return 0.0
            c1, c2 = landmarks[c1_idx], landmarks[c2_idx]
            t1, b1 = landmarks[t1_idx], landmarks[b1_idx]
            t2, b2 = landmarks[t2_idx], landmarks[b2_idx]

            horiz = self._dist_3d(c1, c2)
            if horiz < 1e-6:
                return 0.0

            vert1 = self._dist_3d(t1, b1)
            vert2 = self._dist_3d(t2, b2)
            return float((vert1 + vert2) / (2.0 * horiz))
        except Exception:
            return 0.0

    def _compute_mouth(self, landmarks: List[Any]) -> Tuple[float, bool]:
        """Calculate Mouth Aspect Ratio (MAR) and mouth_open flag."""
        try:
            if _MOUTH_RIGHT >= len(landmarks):
                return 0.0, False
            top, bot = landmarks[_MOUTH_TOP], landmarks[_MOUTH_BOT]
            left, right = landmarks[_MOUTH_LEFT], landmarks[_MOUTH_RIGHT]
            horiz = self._dist_3d(left, right)
            if horiz < 1e-6:
                return 0.0, False
            vert = self._dist_3d(top, bot)
            mar = vert / horiz
            return float(mar), mar > 0.35
        except Exception:
            return 0.0, False

    def _compute_smile(self, landmarks: List[Any], mouth_ar: float) -> bool:
        """Determine if person is smiling based on mouth width / cheek ratio & corner lift."""
        try:
            if _FACE_R_SIDE >= len(landmarks):
                return False
            m_left, m_right = landmarks[_MOUTH_LEFT], landmarks[_MOUTH_RIGHT]
            f_left, f_right = landmarks[_FACE_L_SIDE], landmarks[_FACE_R_SIDE]
            m_width = self._dist_3d(m_left, m_right)
            f_width = self._dist_3d(f_left, f_right)
            if f_width < 1e-6:
                return False
            width_ratio = m_width / f_width

            # Elevation of mouth corners relative to upper lip
            top_lip = landmarks[_MOUTH_TOP]
            avg_corner_y = (m_left.y + m_right.y) / 2.0
            corner_lift = top_lip.y - avg_corner_y  # positive if corners pulled up

            return bool(width_ratio > 0.43 or (width_ratio > 0.40 and mouth_ar > 0.15) or corner_lift > 0.008)
        except Exception:
            return False

    def _compute_eyebrows(self, landmarks: List[Any]) -> Tuple[bool, float]:
        """Calculate normalized eyebrow elevation ratio."""
        try:
            if _FACE_BOT >= len(landmarks):
                return False, 0.0
            r_brow, r_eye = landmarks[_BROW_R_CTR], landmarks[_EYE_R_TOP]
            l_brow, l_eye = landmarks[_BROW_L_CTR], landmarks[_EYE_L_TOP]
            f_top, f_bot  = landmarks[_FACE_TOP], landmarks[_FACE_BOT]

            f_height = self._dist_3d(f_top, f_bot)
            if f_height < 1e-6:
                return False, 0.0

            r_dist = self._dist_3d(r_brow, r_eye)
            l_dist = self._dist_3d(l_brow, l_eye)
            ratio = (r_dist + l_dist) / (2.0 * f_height)
            return ratio > 0.075, float(ratio)
        except Exception:
            return False, 0.0

    def _classify_expression(
        self,
        landmarks: List[Any],
        smile: bool,
        mouth_open: bool,
        eyebrows_raised: bool,
        eyebrow_ratio: float,
    ) -> str:
        """Classify expression into NEUTRAL, SMILE, SURPRISED, ANGRY, or SAD."""
        try:
            if mouth_open and eyebrows_raised:
                return "SURPRISED"
            if smile:
                return "SMILE"
            if eyebrows_raised:
                return "SURPRISED"
            if eyebrow_ratio > 0.0 and eyebrow_ratio < 0.042 and not mouth_open:
                return "ANGRY"

            # Check sad (corners pulled below lower lip)
            m_left, m_right = landmarks[_MOUTH_LEFT], landmarks[_MOUTH_RIGHT]
            bot_lip = landmarks[_MOUTH_BOT]
            if (m_left.y + m_right.y) / 2.0 > bot_lip.y + 0.005:
                return "SAD"

            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    def _draw_face_mesh(
        self,
        annotated: np.ndarray,
        pixels: List[Tuple[int, int]],
        is_primary: bool,
        expression: str,
        left_open: bool,
        right_open: bool,
    ) -> None:
        """Draw key face mesh contours and concise expression label."""
        n = len(pixels)

        def _polyline(indices: List[int], colour: Tuple, thickness: int = 1) -> None:
            pts = [pixels[i] for i in indices if i < n]
            if len(pts) > 1:
                cv2.polylines(
                    annotated,
                    [np.array(pts, dtype=np.int32)],
                    isClosed=False,
                    color=colour,
                    thickness=thickness,
                )

        if is_primary:
            _polyline(_FACE_OVAL, _COL_FACE_MESH, 1)
            _polyline(_FACE_LEFT_EYE,  _COL_EYE if left_open else _COL_EYE_CLOSED, 1)
            _polyline(_FACE_RIGHT_EYE, _COL_EYE if right_open else _COL_EYE_CLOSED, 1)
            _polyline(_FACE_LEFT_BROW,  _COL_EYEBROW, 1)
            _polyline(_FACE_RIGHT_BROW, _COL_EYEBROW, 1)
            _polyline(_FACE_NOSE, _COL_NOSE, 1)
            _polyline(_FACE_LIPS_OUT, _COL_LIPS, 1)

            # Draw concise status badge above face
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            x_min, y_min = min(xs), min(ys)
            eye_str = "Eyes:Open" if (left_open and right_open) else ("Blink" if not (left_open or right_open) else "1-Eye")
            badge = f"Face | {expression} | {eye_str}"
            cv2.putText(
                annotated, badge,
                (max(10, x_min), max(20, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL_FACE_BOX, 1,
            )

        # Iris dots
        for iris_idx, eye_open in [(_FACE_LEFT_IRIS_IDX, left_open), (_FACE_RIGHT_IRIS_IDX, right_open)]:
            if iris_idx < n:
                dot_col = _COL_EYE if eye_open else _COL_EYE_CLOSED
                cv2.circle(annotated, pixels[iris_idx], 4, dot_col, -1)

    def _detect_faces_haar(self, annotated: np.ndarray) -> FaceResult:
        """Haar face cascade fallback (bounding boxes only, no landmarks)."""
        try:
            gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            raw = self._face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60)
            )
            if len(raw) == 0:
                return FaceResult()

            boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw]
            primary = max(boxes, key=lambda b: b[2] * b[3])
            cx = primary[0] + primary[2] // 2
            cy = primary[1] + primary[3] // 2

            for i, (x, y, bw, bh) in enumerate(boxes):
                col = _COL_FACE_BOX if (x, y, bw, bh) == primary else (100, 200, 100)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), col, 2)
                cv2.putText(annotated, "FACE(Haar)", (x, max(12, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            return FaceResult(
                detected=True,
                count=len(boxes),
                bounding_boxes=boxes,
                center=(cx, cy),
                confidence=0.5,
            )
        except Exception as e:
            print(f"[VisionEngine] Haar face fallback error: {e}")
            return FaceResult()

    # ------------------------------------------------------------------
    # Detector 2: MediaPipe Hands + 3D Orientation-Invariant Fingers
    # ------------------------------------------------------------------

    def _detect_hands(
        self, annotated: np.ndarray, original: np.ndarray
    ) -> Tuple[HandResult, HandResult]:
        """Run MediaPipe HandLandmarker; return (left_hand, right_hand)."""
        lh = HandResult(handedness="Left")
        rh = HandResult(handedness="Right")

        if self._hands_solution is None:
            return lh, rh

        try:
            h, w = original.shape[:2]
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            if self._hands_is_tasks_api:
                mp_img = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB, data=rgb
                )
                result = self._hands_solution.detect(mp_img)
                if not result or not result.hand_landmarks:
                    return lh, rh

                for idx, hand_lms in enumerate(result.hand_landmarks):
                    label = "Right"
                    confidence = 0.0
                    if result.handedness and idx < len(result.handedness):
                        cat = result.handedness[idx][0]
                        label      = cat.display_name
                        confidence = float(cat.score)

                    hand = self._build_hand_result(
                        annotated, hand_lms, w, h, label, confidence
                    )
                    if label == "Left":
                        lh = hand
                    else:
                        rh = hand

            else:
                results = self._hands_solution.process(rgb)
                if not results or not results.multi_hand_landmarks:
                    return lh, rh

                handedness_list = results.multi_handedness or []
                for idx, hand_lms_proto in enumerate(results.multi_hand_landmarks):
                    label = "Right"
                    confidence = 0.0
                    if idx < len(handedness_list):
                        cls = handedness_list[idx].classification[0]
                        label      = cls.label
                        confidence = float(cls.score)

                    hand = self._build_hand_result(
                        annotated, hand_lms_proto.landmark, w, h, label, confidence
                    )
                    if label == "Left":
                        lh = hand
                    else:
                        rh = hand

        except Exception as e:
            print(f"[VisionEngine] Hand detect error: {e}")

        return lh, rh

    def _build_hand_result(
        self,
        annotated: np.ndarray,
        landmarks: Any,
        w: int,
        h: int,
        label: str,
        confidence: float,
    ) -> HandResult:
        """Compute pixel positions, 3D finger states, and draw for one hand."""
        pixels: List[Tuple[int, int]] = [
            (int(lm.x * w), int(lm.y * h)) for lm in landmarks
        ]

        wrist = pixels[0] if pixels else None

        mcp_indices = [5, 9, 13, 17]
        palm_pts = [pixels[i] for i in mcp_indices if i < len(pixels)]
        if palm_pts:
            cx = sum(p[0] for p in palm_pts) // len(palm_pts)
            cy = sum(p[1] for p in palm_pts) // len(palm_pts)
            center = (cx, cy)
        else:
            center = wrist

        finger_states = self._compute_finger_states(landmarks, label)
        fingers_up    = sum(finger_states)
        fingers_dict  = {name: state for name, state in zip(_FINGER_NAMES, finger_states)}

        colour = _COL_HAND_L if label == "Left" else _COL_HAND_R
        self._draw_hand(annotated, pixels, finger_states, colour, label)

        return HandResult(
            detected=True,
            landmarks=list(landmarks),
            landmark_pixels=pixels,
            finger_states=finger_states,
            fingers=fingers_dict,
            fingers_up=fingers_up,
            handedness=label,
            confidence=confidence,
            wrist=wrist,
            center=center,
        )

    def _compute_finger_states(
        self, landmarks: Any, handedness: str
    ) -> List[bool]:
        """
        Determine which of the 5 fingers are extended using 3D vector geometry.

        Algorithm:
          Fingers 1-4 (index through pinky):
            1. Compute palm direction unit vector: wrist(0) -> middle_MCP(9)
            2. Project (finger_MCP -> finger_TIP) onto palm direction.
            3. If projection > threshold -> finger extended.
            This works for any hand orientation (upright, sideways, inverted).

          Thumb (0):
            1. Distance from Thumb TIP (4) to Pinky MCP (17) relative to Palm Size (Wrist 0 -> Middle MCP 9).
               When extended (open hand / thumbs up), ratio > 1.08.
               When folded across palm (fist / 4 fingers / peace sign), ratio < 0.90.
            2. IP joint straightness alignment: dot(CMC->MCP, IP->TIP) > 0.25.
            3. Combined condition ensures thumb works when upright, rotated, or tilted!
        """
        states = [False] * 5

        try:
            if len(landmarks) < 21:
                return states

            wrist      = landmarks[0]
            middle_mcp = landmarks[9]
            pinky_mcp  = landmarks[17]
            idx_mcp    = landmarks[5]

            # Palm direction: wrist -> middle_MCP
            pdx = middle_mcp.x - wrist.x
            pdy = middle_mcp.y - wrist.y
            pdz = middle_mcp.z - wrist.z
            pd_len = math.sqrt(pdx*pdx + pdy*pdy + pdz*pdz)
            if pd_len < 1e-6:
                pd_len = 1.0
            palm_dir = (pdx / pd_len, pdy / pd_len, pdz / pd_len)

            # Fingers 1-4: project MCP->TIP onto palm direction
            finger_mcps = [5, 9, 13, 17]   # index, middle, ring, pinky
            finger_tips = [8, 12, 16, 20]
            threshold = 0.04 * pd_len

            for i, (mcp_i, tip_i) in enumerate(zip(finger_mcps, finger_tips)):
                mcp = landmarks[mcp_i]
                tip = landmarks[tip_i]
                vx = tip.x - mcp.x
                vy = tip.y - mcp.y
                vz = tip.z - mcp.z
                proj = vx * palm_dir[0] + vy * palm_dir[1] + vz * palm_dir[2]
                states[i + 1] = proj > threshold

            # --- Improved Thumb Extension Detection ---
            thumb_cmc = landmarks[1]
            thumb_mcp = landmarks[2]
            thumb_ip  = landmarks[3]
            thumb_tip = landmarks[4]

            # 3D Distance from Thumb TIP (4) to Pinky MCP (17)
            d_tip_pinky = self._dist_3d(thumb_tip, pinky_mcp)
            d_tip_idx   = self._dist_3d(thumb_tip, idx_mcp)
            d_cmc_idx   = self._dist_3d(thumb_cmc, idx_mcp)

            ratio_pinky = d_tip_pinky / pd_len
            ratio_idx   = d_tip_idx / pd_len

            # IP joint straightness (CMC->MCP dot IP->TIP)
            v1 = (thumb_mcp.x - thumb_cmc.x, thumb_mcp.y - thumb_cmc.y, thumb_mcp.z - thumb_cmc.z)
            v2 = (thumb_tip.x - thumb_ip.x,   thumb_tip.y - thumb_ip.y,   thumb_tip.z - thumb_ip.z)
            v1_len = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            v2_len = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

            dot_align = 0.0
            if v1_len > 1e-6 and v2_len > 1e-6:
                dot_align = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (v1_len * v2_len)

            # Thumb is extended if it spreads away from palm (high ratio_pinky or high ratio_idx)
            # AND the thumb joints are relatively uncurled.
            thumb_extended = (
                (ratio_pinky > 1.08 and dot_align > 0.25) or
                (ratio_idx > 0.85 and d_tip_idx > d_cmc_idx * 1.15)
            )

            states[0] = bool(thumb_extended)

        except Exception as e:
            print(f"[VisionEngine] Finger state error: {e}")

        return states

    def _draw_hand(
        self,
        annotated:     np.ndarray,
        pixels:        List[Tuple[int, int]],
        finger_states: List[bool],
        colour:        Tuple[int, int, int],
        label:         str,
    ) -> None:
        """Draw hand skeleton, joints, and per-finger-tip state indicators."""
        # Connections
        for a, b in _HAND_CONNECTIONS:
            if a < len(pixels) and b < len(pixels):
                cv2.line(annotated, pixels[a], pixels[b], colour, 2)

        # Joints / tips
        for i, pt in enumerate(pixels):
            if i in _FINGER_TIPS:
                fi = _FINGER_TIPS.index(i)
                dot_col = _COL_FINGER_UP if finger_states[fi] else _COL_FINGER_DOWN
                cv2.circle(annotated, pt, 7, dot_col, -1)
                cv2.circle(annotated, pt, 7, colour, 1)
            else:
                cv2.circle(annotated, pt, 4, colour, -1)

        # Label above wrist: "LH 3/5"
        if pixels:
            wx, wy = pixels[0]
            cv2.putText(
                annotated,
                f"{label[0]}H {sum(finger_states)}/5",
                (wx - 20, max(14, wy - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
            )

    # ------------------------------------------------------------------
    # Detector 3: MediaPipe Pose + classification
    # ------------------------------------------------------------------

    def _detect_pose(
        self, annotated: np.ndarray, original: np.ndarray
    ) -> PoseResult:
        """Run MediaPipe PoseLandmarker; return PoseResult."""
        if self._pose_landmarker is None:
            return PoseResult()

        try:
            h, w = original.shape[:2]
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            lms_raw: List[Any] = []

            if self._pose_is_tasks_api:
                mp_img = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB, data=rgb
                )
                result = self._pose_landmarker.detect(mp_img)
                if not (result and result.pose_landmarks and result.pose_landmarks):
                    return PoseResult()
                lms_raw = result.pose_landmarks[0]
            else:
                results = self._pose_landmarker.process(rgb)
                if not (results and results.pose_landmarks):
                    return PoseResult()
                lms_raw = list(results.pose_landmarks.landmark)

            pixels: List[Tuple[int, int]] = [
                (int(lm.x * w), int(lm.y * h)) for lm in lms_raw
            ]

            self._draw_pose(annotated, lms_raw, pixels)

            def _vis(lm: Any) -> float:
                return float(getattr(lm, "visibility", 1.0))

            n = len(lms_raw)

            # ---- Arm-raised detection ----
            left_hand_raised  = False
            right_hand_raised = False

            if n > 16:
                l_shoulder = lms_raw[11]
                r_shoulder = lms_raw[12]
                l_wrist    = lms_raw[15]
                r_wrist    = lms_raw[16]
                l_ear      = lms_raw[7]
                r_ear      = lms_raw[8]

                if _vis(l_wrist) > 0.4 and _vis(l_shoulder) > 0.4:
                    left_hand_raised = (
                        l_wrist.y < l_shoulder.y - 0.05 or
                        l_wrist.y < l_ear.y
                    )
                if _vis(r_wrist) > 0.4 and _vis(r_shoulder) > 0.4:
                    right_hand_raised = (
                        r_wrist.y < r_shoulder.y - 0.05 or
                        r_wrist.y < r_ear.y
                    )

            # ---- Standing / sitting classification ----
            standing = False
            sitting  = False

            if n > 28:
                l_hip   = lms_raw[23]
                l_knee  = lms_raw[25]
                r_hip   = lms_raw[24]
                r_knee  = lms_raw[26]

                left_ok  = _vis(l_hip) > 0.35 and _vis(l_knee) > 0.35
                right_ok = _vis(r_hip) > 0.35 and _vis(r_knee) > 0.35

                if left_ok or right_ok:
                    hip   = l_hip   if left_ok else r_hip
                    knee  = l_knee  if left_ok else r_knee
                    delta = knee.y - hip.y
                    if delta > 0.12:
                        standing = True
                    elif abs(delta) < 0.08:
                        sitting  = True

            key_idxs = [11, 12, 23, 24]
            vis_vals = [_vis(lms_raw[i]) for i in key_idxs if i < n]
            confidence = float(sum(vis_vals) / len(vis_vals)) if vis_vals else 0.0

            return PoseResult(
                detected=True,
                landmarks=lms_raw,
                landmark_pixels=pixels,
                left_hand_raised=left_hand_raised,
                right_hand_raised=right_hand_raised,
                hands_raised=left_hand_raised and right_hand_raised,
                standing=standing,
                sitting=sitting,
                confidence=confidence,
            )

        except Exception as e:
            print(f"[VisionEngine] Pose detect error: {e}")
            return PoseResult()

    def _draw_pose(
        self,
        annotated:  np.ndarray,
        landmarks:  List[Any],
        pixels:     List[Tuple[int, int]],
    ) -> None:
        """Draw pose skeleton and key-joint dots."""
        n = len(landmarks)

        for a, b in _POSE_CONNECTIONS:
            if a < n and b < n:
                va = getattr(landmarks[a], "visibility", 1.0)
                vb = getattr(landmarks[b], "visibility", 1.0)
                if va > 0.3 and vb > 0.3:
                    cv2.line(annotated, pixels[a], pixels[b], _COL_POSE, 2)

        for name, idx in _POSE_IDX.items():
            if idx < n:
                vis = getattr(landmarks[idx], "visibility", 1.0)
                if vis > 0.3:
                    cv2.circle(annotated, pixels[idx], 5, _COL_POSE, -1)
                    cv2.circle(annotated, pixels[idx], 5, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # FPS tracking
    # ------------------------------------------------------------------

    def _update_fps(self) -> float:
        now = time.perf_counter()
        self._frame_times.append(now)
        if len(self._frame_times) > self._fps_window:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
        return round(self._fps, 1)

    def _draw_fps(self, annotated: np.ndarray, fps: float) -> None:
        cv2.putText(
            annotated, f"FPS: {fps:.1f}",
            (annotated.shape[1] - 90, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )
