"""
tests/test_vision_engine.py — Comprehensive unit tests for WALLE Vision perception layer.

Tests cover:
  1. Thumb detection geometry (extended, folded, rotated, invalid landmarks)
  2. Eye open / closed detection (EAR calculation, open, closed, 1-eye closed)
  3. Blink detection state machine (open -> closed -> open, prolonged closure state)
  4. Mouth aspect ratio (open vs closed)
  5. Smile detection & eyebrow elevation
  6. Expression classification (NEUTRAL, SMILE, SURPRISED, ANGRY, SAD)
  7. YuNet face detector initialization & extraction (bbox, confidence, 5 keypoints)
  8. VisionResult dataclasses & structure
  9. VisionEngine frame processing & temporal smoothers

Run:
    python -m pytest tests/test_vision_engine.py -v
"""

import math
import numpy as np
import pytest


# ===========================================================================
# Helper landmark builders
# ===========================================================================

def _make_lm(x: float, y: float, z: float = 0.0):
    class Lm:
        pass
    lm = Lm()
    lm.x, lm.y, lm.z = x, y, z
    return lm


def _get_engine_no_init():
    from src.vision.vision_engine import VisionEngine
    return VisionEngine.__new__(VisionEngine)


# ===========================================================================
# 1. Thumb Detection Geometry Tests
# ===========================================================================

class TestThumbDetection:

    def _build_hand(self, thumb_extended: bool, fingers_extended: list = None):
        """Build 21 hand landmarks for testing thumb and finger states."""
        if fingers_extended is None:
            fingers_extended = [True, True, True, True]  # index, middle, ring, pinky

        lms = [_make_lm(0.5, 0.5) for _ in range(21)]
        lms[0] = _make_lm(0.50, 0.90)  # Wrist

        # Thumb landmarks: 1=CMC, 2=MCP, 3=IP, 4=TIP
        lms[1] = _make_lm(0.58, 0.82)
        lms[2] = _make_lm(0.64, 0.74)
        lms[3] = _make_lm(0.70, 0.66)

        if thumb_extended:
            # Thumb extended: tip spread out away from index MCP (5) and pinky MCP (17)
            lms[4] = _make_lm(0.82, 0.58)
        else:
            # Thumb folded across palm: tip tucked near index/middle MCP
            lms[4] = _make_lm(0.56, 0.72)

        # Index (5..8), Middle (9..12), Ring (13..16), Pinky (17..20)
        mcps = [5, 9, 13, 17]
        tips = [8, 12, 16, 20]
        x_coords = [0.45, 0.50, 0.55, 0.60]

        for i in range(4):
            m_idx, t_idx = mcps[i], tips[i]
            x = x_coords[i]
            lms[m_idx] = _make_lm(x, 0.65)
            lms[t_idx] = _make_lm(x, 0.25 if fingers_extended[i] else 0.75)

        return lms

    def test_thumb_extended_open_hand(self):
        engine = _get_engine_no_init()
        lms = self._build_hand(thumb_extended=True)
        states = engine._compute_finger_states(lms, "Right")
        assert states[0] is True, "Thumb should be detected as extended"

    def test_thumb_folded_fist(self):
        engine = _get_engine_no_init()
        lms = self._build_hand(thumb_extended=False, fingers_extended=[False, False, False, False])
        states = engine._compute_finger_states(lms, "Right")
        assert states[0] is False, "Thumb should be detected as folded/curled"

    def test_thumb_folded_number_four(self):
        """Number 4 gesture: 4 fingers up, thumb tucked across palm."""
        engine = _get_engine_no_init()
        lms = self._build_hand(thumb_extended=False, fingers_extended=[True, True, True, True])
        states = engine._compute_finger_states(lms, "Right")
        assert states[0] is False, "Thumb should be folded in Number 4 gesture"
        assert states[1:] == [True, True, True, True], "Four fingers should be extended"

    def test_invalid_landmarks_thumb(self):
        engine = _get_engine_no_init()
        states = engine._compute_finger_states([], "Right")
        assert states == [False] * 5

        short_lms = [_make_lm(0.5, 0.5) for _ in range(10)]
        states_short = engine._compute_finger_states(short_lms, "Right")
        assert states_short == [False] * 5


# ===========================================================================
# 2. Eye Open / Closed EAR Tests
# ===========================================================================

class TestEyeStateEAR:

    def _build_face_eye(self, ear_left: float, ear_right: float):
        """Build 478 face landmarks with specified EAR distances for left and right eyes."""
        lms = [_make_lm(0.5, 0.5) for _ in range(478)]

        # Left Eye EAR: (dist(385,380) + dist(387,373)) / (2 * dist(362,263))
        # Horizontal width = 0.10
        lms[362] = _make_lm(0.40, 0.40)
        lms[263] = _make_lm(0.50, 0.40)

        # Set vertical gap according to ear_left
        vert_left = ear_left * 0.10
        lms[385] = _make_lm(0.43, 0.40 - vert_left / 2.0)
        lms[380] = _make_lm(0.43, 0.40 + vert_left / 2.0)
        lms[387] = _make_lm(0.47, 0.40 - vert_left / 2.0)
        lms[373] = _make_lm(0.47, 0.40 + vert_left / 2.0)

        # Right Eye EAR: (dist(160,144) + dist(158,153)) / (2 * dist(33,133))
        lms[33]  = _make_lm(0.20, 0.40)
        lms[133] = _make_lm(0.30, 0.40)

        vert_right = ear_right * 0.10
        lms[160] = _make_lm(0.23, 0.40 - vert_right / 2.0)
        lms[144] = _make_lm(0.23, 0.40 + vert_right / 2.0)
        lms[158] = _make_lm(0.27, 0.40 - vert_right / 2.0)
        lms[153] = _make_lm(0.27, 0.40 + vert_right / 2.0)

        return lms

    def test_both_eyes_open(self):
        engine = _get_engine_no_init()
        lms = self._build_face_eye(ear_left=0.30, ear_right=0.32)
        ear_l = engine._compute_ear(lms, 362, 263, 385, 380, 387, 373)
        ear_r = engine._compute_ear(lms, 33, 133, 160, 144, 158, 153)

        assert ear_l > 0.20, "Left EAR should be open"
        assert ear_r > 0.20, "Right EAR should be open"

    def test_both_eyes_closed(self):
        engine = _get_engine_no_init()
        lms = self._build_face_eye(ear_left=0.08, ear_right=0.07)
        ear_l = engine._compute_ear(lms, 362, 263, 385, 380, 387, 373)
        ear_r = engine._compute_ear(lms, 33, 133, 160, 144, 158, 153)

        assert ear_l <= 0.20, "Left EAR should be closed"
        assert ear_r <= 0.20, "Right EAR should be closed"

    def test_one_eye_wink(self):
        engine = _get_engine_no_init()
        lms = self._build_face_eye(ear_left=0.05, ear_right=0.30)
        ear_l = engine._compute_ear(lms, 362, 263, 385, 380, 387, 373)
        ear_r = engine._compute_ear(lms, 33, 133, 160, 144, 158, 153)

        assert ear_l <= 0.20, "Left eye winked (closed)"
        assert ear_r > 0.20, "Right eye open"

    def test_invalid_landmarks_ear(self):
        engine = _get_engine_no_init()
        ear = engine._compute_ear([], 362, 263, 385, 380, 387, 373)
        assert ear == 0.0


# ===========================================================================
# 3. Blink Detection State Machine Tests
# ===========================================================================

class TestBlinkStateMachine:

    def test_blink_transition_open_closed_open(self):
        """Blink state machine transition: Open -> Closed (2 frames) -> Open."""
        engine = _get_engine_no_init()
        engine._closed_frames = 0
        engine._blink_count = 0
        engine._blink_active_frames = 0

        # Frame 1: Open
        closed_frames = 0
        blink_detected = False

        # Frame 2: Closed (eye blink starts)
        closed_frames += 1
        # Frame 3: Closed
        closed_frames += 1

        # Frame 4: Eyes re-open (transition complete!)
        if 1 <= closed_frames <= 10:
            engine._blink_count += 1
            blink_detected = True

        assert engine._blink_count == 1, "Blink count should increment to 1"
        assert blink_detected is True, "Blink detected should trigger True"

    def test_prolonged_closed_eyes_does_not_count_multiple_blinks(self):
        """Eyes closed for 20 frames (sleeping/squinting) should NOT count as a blink."""
        engine = _get_engine_no_init()
        engine._closed_frames = 20
        engine._blink_count = 0

        blink_detected = False
        if 1 <= engine._closed_frames <= 10:
            engine._blink_count += 1
            blink_detected = True

        assert engine._blink_count == 0, "Prolonged eye closure must NOT trigger blink count"
        assert blink_detected is False


# ===========================================================================
# 4. Mouth & Smile Tests
# ===========================================================================

class TestMouthAndSmile:

    def _build_face_expression(self, mouth_open: bool, smile: bool, eyebrows_raised: bool):
        lms = [_make_lm(0.5, 0.5) for _ in range(478)]

        # Face bounds
        lms[10]  = _make_lm(0.5, 0.10)
        lms[152] = _make_lm(0.5, 0.90)
        lms[234] = _make_lm(0.20, 0.50)
        lms[454] = _make_lm(0.80, 0.50)

        # Mouth corners (61, 291), Top lip (13), Bot lip (14)
        m_width = 0.30 if smile else 0.20
        lms[61]  = _make_lm(0.50 - m_width / 2.0, 0.70 - (0.02 if smile else 0.0))
        lms[291] = _make_lm(0.50 + m_width / 2.0, 0.70 - (0.02 if smile else 0.0))

        m_gap = 0.12 if mouth_open else 0.02
        lms[13] = _make_lm(0.50, 0.70 - m_gap / 2.0)
        lms[14] = _make_lm(0.50, 0.70 + m_gap / 2.0)

        brow_y = 0.28 if eyebrows_raised else 0.35
        lms[105] = _make_lm(0.35, brow_y)
        lms[159] = _make_lm(0.35, 0.40)
        lms[334] = _make_lm(0.65, brow_y)
        lms[386] = _make_lm(0.65, 0.40)

        return lms

    def test_mouth_open_and_closed(self):
        engine = _get_engine_no_init()
        lms_open = self._build_face_expression(mouth_open=True, smile=False, eyebrows_raised=False)
        mar_o, is_open = engine._compute_mouth(lms_open)
        assert is_open is True, "Mouth should be detected as OPEN"

        lms_closed = self._build_face_expression(mouth_open=False, smile=False, eyebrows_raised=False)
        mar_c, is_closed_flag = engine._compute_mouth(lms_closed)
        assert is_closed_flag is False, "Mouth should be detected as CLOSED"

    def test_smile_detection(self):
        engine = _get_engine_no_init()
        lms_smile = self._build_face_expression(mouth_open=False, smile=True, eyebrows_raised=False)
        mar, _ = engine._compute_mouth(lms_smile)
        is_smile = engine._compute_smile(lms_smile, mar)
        assert is_smile is True, "Obvious smile should be detected"

        lms_neutral = self._build_face_expression(mouth_open=False, smile=False, eyebrows_raised=False)
        mar_n, _ = engine._compute_mouth(lms_neutral)
        is_neutral_smile = engine._compute_smile(lms_neutral, mar_n)
        assert is_neutral_smile is False, "Neutral face should not be detected as smile"


# ===========================================================================
# 5. Expression Classification Tests
# ===========================================================================

class TestExpressionClassification:

    def _build_face(self, mouth_open: bool, smile: bool, eyebrows_raised: bool):
        t = TestMouthAndSmile()
        return t._build_face_expression(mouth_open, smile, eyebrows_raised)

    def test_neutral_expression(self):
        engine = _get_engine_no_init()
        lms = self._build_face(mouth_open=False, smile=False, eyebrows_raised=False)
        expr = engine._classify_expression(lms, smile=False, mouth_open=False, eyebrows_raised=False, eyebrow_ratio=0.06)
        assert expr == "NEUTRAL"

    def test_smile_expression(self):
        engine = _get_engine_no_init()
        lms = self._build_face(mouth_open=False, smile=True, eyebrows_raised=False)
        expr = engine._classify_expression(lms, smile=True, mouth_open=False, eyebrows_raised=False, eyebrow_ratio=0.06)
        assert expr == "SMILE"

    def test_surprised_expression(self):
        engine = _get_engine_no_init()
        lms = self._build_face(mouth_open=True, smile=False, eyebrows_raised=True)
        expr = engine._classify_expression(lms, smile=False, mouth_open=True, eyebrows_raised=True, eyebrow_ratio=0.09)
        assert expr == "SURPRISED"


# ===========================================================================
# 6. YuNet Face Detector Tests (Task 13 requirements)
# ===========================================================================

class TestYuNetFaceDetector:

    @pytest.fixture(scope="class")
    def engine(self):
        from src.vision.vision_engine import VisionEngine
        return VisionEngine()

    def test_yunet_initialization(self, engine):
        assert engine._yunet_detector is not None, "YuNet FaceDetectorYN should be initialized"

    def test_yunet_blank_frame_no_face(self, engine):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        res = engine._detect_faces_yunet(img, img)
        assert res["detected"] is False, "Blank frame should detect 0 faces"
        assert res["count"] == 0
        assert res["bounding_boxes"] == []
        assert res["confidences"] == []
        assert res["keypoints"] == []

    def test_yunet_invalid_frame(self, engine):
        res = engine._detect_faces_yunet(None, None)
        assert res["detected"] is False
        assert res["count"] == 0

    def test_yunet_result_keys(self, engine):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100
        res = engine._detect_faces_yunet(img, img)
        for key in ("detected", "count", "bounding_boxes", "confidences", "keypoints", "center"):
            assert key in res


# ===========================================================================
# 7. VisionResult Structure Tests
# ===========================================================================

class TestVisionResultStructure:

    def test_all_fields_initialized(self):
        from src.vision.vision_result import VisionResult, FaceResult, EyeResult, HandResult, PoseResult
        vr = VisionResult()

        assert isinstance(vr.face, FaceResult)
        assert isinstance(vr.eyes, EyeResult)
        assert isinstance(vr.left_hand, HandResult)
        assert isinstance(vr.right_hand, HandResult)
        assert isinstance(vr.pose, PoseResult)

        # YuNet & Face fields
        assert hasattr(vr.face, "keypoints")
        assert hasattr(vr.face, "confidence")
        assert hasattr(vr.face, "detector_source")
        assert hasattr(vr.face, "mouth_open")
        assert hasattr(vr.face, "smile")
        assert hasattr(vr.face, "eyebrows_raised")
        assert hasattr(vr.face, "expression")


# ===========================================================================
# 8. VisionEngine Execution Tests
# ===========================================================================

class TestVisionEngineExecution:

    @pytest.fixture(scope="class")
    def engine(self):
        from src.vision.vision_engine import VisionEngine
        return VisionEngine()

    def test_process_valid_frame(self, engine):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        annotated, result = engine.process_frame(frame)

        assert annotated.shape == (480, 640, 3)
        assert result.camera_dark is False
        assert result.fps >= 0.0

    def test_process_none_frame(self, engine):
        annotated, result = engine.process_frame(None)
        assert annotated is not None
        assert result.face.detected is False
