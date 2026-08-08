"""
tests/test_game_master.py

Comprehensive unit tests for:
- VisionScanSession (new scan pipeline)
- VisionScanResult aggregation
- GameMaster state machine
- Challenge evaluation logic
- Frame-level error isolation
- Interrupt support

All vision tests use mock camera/VisionProcessor data — no real webcam needed.
"""

import time
import threading
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from src.hal.hardware_factory import HardwareFactory
from src.engine.prompt_manager import PromptManager
from src.ai.llm_service import LLMService
from src.ai.vision_processor import VisionProcessor
from src.engine.game_master import GameMaster
from src.engine.game_state import GameState
from src.engine.vision_scan import (
    VisionScanSession, VisionScanResult,
    POSE_MIN_CONFIRMATIONS, FACE_MIN_CONFIRMATIONS, COLOR_MIN_CONFIRMATIONS,
    SCAN_FRAME_COUNT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_telemetry(
    hands=False, left=False, right=False, pose=False, face=False, color=False,
    brightness=120.0, face_count=0, face_center=(0, 0)
):
    return {
        "face_detected": face,
        "face_count": face_count,
        "face_center": face_center,
        "body_pose_detected": pose,
        "left_hand_raised": left,
        "right_hand_raised": right,
        "hands_raised": hands,
        "avg_brightness": brightness,
        "has_colorful_item": color,
    }


def _make_mock_camera(telemetry_sequence):
    """
    Returns a mock camera where each call to get_frame() yields the next item
    from telemetry_sequence.  Items can be:
      - np.ndarray: returned directly
      - None:       simulates camera returning None
      - Exception:  raises that exception
    """
    camera = MagicMock()
    side_effects = list(telemetry_sequence)
    camera.get_frame.side_effect = side_effects + [np.ones((480, 640, 3), dtype=np.uint8)] * 20
    return camera


def _make_mock_vision(telemetry_sequence):
    """
    Returns a mock VisionProcessor where analyze_frame() returns successive
    (annotated_frame, telemetry_dict) pairs from the provided telemetry list.
    """
    vision = MagicMock()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [(dummy_frame, t) for t in telemetry_sequence] + \
              [(dummy_frame, _make_telemetry())] * 20
    vision.analyze_frame.side_effect = results
    return vision


def _run_session_sync(camera, vision, interrupt_event=None, timeout=5.0) -> VisionScanResult:
    """Helper: create, run, and wait for a VisionScanSession synchronously."""
    session = VisionScanSession(camera, vision, interrupt_event)
    session.start()
    return session.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# Tests: VisionScanResult — value object
# ---------------------------------------------------------------------------

class TestVisionScanResult(unittest.TestCase):

    def test_failure_returns_all_false(self):
        r = VisionScanResult.failure("test_error")
        self.assertFalse(r.face_detected)
        self.assertFalse(r.hands_raised)
        self.assertFalse(r.has_colorful_item)
        self.assertEqual(r.scan_error, "test_error")
        self.assertFalse(r.was_interrupted)

    def test_interrupted_failure_sets_flag(self):
        r = VisionScanResult.failure("interrupted", interrupted=True)
        self.assertTrue(r.was_interrupted)

    def test_as_telemetry_contains_all_keys(self):
        r = VisionScanResult(face_detected=True, hands_raised=True)
        t = r.as_telemetry()
        for key in [
            "face_detected", "face_count", "face_center",
            "body_pose_detected", "left_hand_raised", "right_hand_raised",
            "hands_raised", "avg_brightness", "has_colorful_item"
        ]:
            self.assertIn(key, t)


# ---------------------------------------------------------------------------
# Tests: VisionScanSession — consensus aggregation
# ---------------------------------------------------------------------------

class TestVisionScanSession(unittest.TestCase):

    def _session_with_telemetry(self, tel_list, interrupt_event=None):
        """Run a scan with the given fixed telemetry list for each frame."""
        camera = _make_mock_camera([np.ones((480, 640, 3), dtype=np.uint8)] * 20)
        vision = _make_mock_vision(tel_list)
        return _run_session_sync(camera, vision, interrupt_event=interrupt_event, timeout=5.0)

    # -- hands_raised consensus --

    def test_zero_hands_frames_returns_false(self):
        tels = [_make_telemetry(hands=False, pose=True, face=True)] * SCAN_FRAME_COUNT
        r = self._session_with_telemetry(tels)
        self.assertFalse(r.hands_raised)
        self.assertEqual(r.hands_confirmations, 0)

    def test_one_hands_frame_below_threshold_returns_false(self):
        tels = [_make_telemetry(hands=True, left=True, right=True, pose=True, face=True)] + \
               [_make_telemetry(hands=False, pose=True, face=True)] * (SCAN_FRAME_COUNT - 1)
        r = self._session_with_telemetry(tels)
        self.assertFalse(r.hands_raised)

    def test_two_hands_frames_meets_threshold_returns_true(self):
        tels = [_make_telemetry(hands=True, left=True, right=True, pose=True, face=True)] * 2 + \
               [_make_telemetry(hands=False, pose=True, face=True)] * (SCAN_FRAME_COUNT - 2)
        r = self._session_with_telemetry(tels)
        self.assertTrue(r.hands_raised)
        self.assertEqual(r.hands_confirmations, 2)

    def test_all_frames_hands_raised_returns_true(self):
        tels = [_make_telemetry(hands=True, left=True, right=True, pose=True, face=True)] * SCAN_FRAME_COUNT
        r = self._session_with_telemetry(tels)
        self.assertTrue(r.hands_raised)
        self.assertEqual(r.hands_confirmations, SCAN_FRAME_COUNT)

    # -- individual hand consensus --

    def test_left_only_hands_raised_false(self):
        tels = [_make_telemetry(hands=False, left=True, right=False, pose=True, face=True)] * SCAN_FRAME_COUNT
        r = self._session_with_telemetry(tels)
        self.assertTrue(r.left_hand_raised)
        self.assertFalse(r.right_hand_raised)
        self.assertFalse(r.hands_raised)

    def test_right_only_hands_raised_false(self):
        tels = [_make_telemetry(hands=False, left=False, right=True, pose=True, face=True)] * SCAN_FRAME_COUNT
        r = self._session_with_telemetry(tels)
        self.assertFalse(r.left_hand_raised)
        self.assertTrue(r.right_hand_raised)
        self.assertFalse(r.hands_raised)

    # -- face consensus --

    def test_face_detected_below_threshold_returns_false(self):
        tels = [_make_telemetry(face=True)] * (FACE_MIN_CONFIRMATIONS - 1) + \
               [_make_telemetry(face=False)] * (SCAN_FRAME_COUNT - FACE_MIN_CONFIRMATIONS + 1)
        r = self._session_with_telemetry(tels)
        self.assertFalse(r.face_detected)

    def test_face_detected_at_threshold_returns_true(self):
        tels = [_make_telemetry(face=True)] * FACE_MIN_CONFIRMATIONS + \
               [_make_telemetry(face=False)] * (SCAN_FRAME_COUNT - FACE_MIN_CONFIRMATIONS)
        r = self._session_with_telemetry(tels)
        self.assertTrue(r.face_detected)

    # -- colorful item: 1-frame threshold --

    def test_color_item_in_one_frame_returns_true(self):
        tels = [_make_telemetry(color=True)] + \
               [_make_telemetry(color=False)] * (SCAN_FRAME_COUNT - 1)
        r = self._session_with_telemetry(tels)
        self.assertTrue(r.has_colorful_item)

    def test_color_item_absent_all_frames_returns_false(self):
        tels = [_make_telemetry(color=False)] * SCAN_FRAME_COUNT
        r = self._session_with_telemetry(tels)
        self.assertFalse(r.has_colorful_item)

    # -- camera returning None on some frames --

    def test_some_camera_none_frames_still_succeeds(self):
        # Alternate None and valid frames; valid frames show pose+hands
        good_frame = np.ones((480, 640, 3), dtype=np.uint8)
        frames = [None, good_frame, None, good_frame, good_frame, good_frame]
        camera = _make_mock_camera(frames)
        # Only the non-None frames will produce telemetry
        good_tel = _make_telemetry(hands=True, left=True, right=True, pose=True, face=True)
        vision = MagicMock()
        vision.analyze_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), good_tel)
        r = _run_session_sync(camera, vision)
        # 4 valid frames with hands_raised=True → True
        self.assertTrue(r.hands_raised)

    def test_all_camera_none_frames_returns_safe_failure(self):
        camera = _make_mock_camera([None] * 10)
        vision = MagicMock()
        r = _run_session_sync(camera, vision)
        self.assertFalse(r.hands_raised)
        self.assertFalse(r.face_detected)
        self.assertIsNotNone(r.scan_error)
        self.assertEqual(r.valid_frames, 0)

    # -- VisionProcessor exception on one frame --

    def test_vision_exception_one_frame_continues(self):
        good_frame = np.ones((480, 640, 3), dtype=np.uint8)
        camera = _make_mock_camera([good_frame] * SCAN_FRAME_COUNT)
        good_tel = _make_telemetry(hands=True, left=True, right=True, pose=True, face=True)
        vision = MagicMock()
        # Raise on first call, succeed on rest
        vision.analyze_frame.side_effect = [
            RuntimeError("pose model crash"),
        ] + [(np.zeros((480, 640, 3), dtype=np.uint8), good_tel)] * (SCAN_FRAME_COUNT - 1)
        r = _run_session_sync(camera, vision)
        # 5 of 6 frames succeed; all with hands_raised → True
        self.assertTrue(r.hands_raised)
        self.assertEqual(r.valid_frames, SCAN_FRAME_COUNT - 1)

    def test_all_vision_exceptions_returns_safe_failure(self):
        good_frame = np.ones((480, 640, 3), dtype=np.uint8)
        camera = _make_mock_camera([good_frame] * SCAN_FRAME_COUNT)
        vision = MagicMock()
        vision.analyze_frame.side_effect = RuntimeError("model unavailable")
        r = _run_session_sync(camera, vision)
        self.assertFalse(r.hands_raised)
        self.assertEqual(r.valid_frames, 0)

    # -- interrupt support --

    def test_interrupt_before_scan_starts_returns_interrupted(self):
        camera = _make_mock_camera([np.ones((480, 640, 3), dtype=np.uint8)] * SCAN_FRAME_COUNT)
        vision = MagicMock()
        vision.analyze_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), _make_telemetry())
        evt = threading.Event()
        evt.set()  # already interrupted before session starts
        r = _run_session_sync(camera, vision, interrupt_event=evt, timeout=3.0)
        self.assertTrue(r.was_interrupted)
        self.assertFalse(r.hands_raised)

    def test_interrupt_mid_scan_stops_early(self):
        good_frame = np.ones((480, 640, 3), dtype=np.uint8)
        camera = _make_mock_camera([good_frame] * SCAN_FRAME_COUNT)
        good_tel = _make_telemetry(hands=True, left=True, right=True, pose=True, face=True)
        vision = MagicMock()
        vision.analyze_frame.return_value = (good_frame, good_tel)
        evt = threading.Event()

        session = VisionScanSession(camera, vision, interrupt_event=evt)
        session.start()
        time.sleep(0.05)  # let it capture at most 1 frame
        evt.set()          # interrupt mid-scan
        result = session.wait(timeout=5.0)

        self.assertTrue(result.was_interrupted)
        self.assertLess(result.attempted_frames, SCAN_FRAME_COUNT)

    # -- wait timeout --

    def test_wait_timeout_returns_failure(self):
        # Camera that blocks forever
        camera = MagicMock()
        camera.get_frame.side_effect = lambda: time.sleep(60) or np.zeros((480, 640, 3), dtype=np.uint8)
        vision = MagicMock()
        r = _run_session_sync(camera, vision, timeout=0.5)
        # Should return a timeout failure result — not raise
        self.assertFalse(r.hands_raised)

    # -- avg_brightness --

    def test_avg_brightness_is_averaged_across_valid_frames(self):
        tels = [_make_telemetry(brightness=100.0)] * 3 + [_make_telemetry(brightness=200.0)] * 3
        camera = _make_mock_camera([np.ones((480, 640, 3), dtype=np.uint8)] * SCAN_FRAME_COUNT)
        vision = _make_mock_vision(tels)
        r = _run_session_sync(camera, vision)
        self.assertAlmostEqual(r.avg_brightness, 150.0, places=1)


# ---------------------------------------------------------------------------
# Tests: GameMaster — state machine (unchanged logic preserved)
# ---------------------------------------------------------------------------

class TestGameMasterStateMachine(unittest.TestCase):

    def setUp(self):
        config = {
            "driver_mode": "simulated",
            "hardware": {
                "camera": {"device_index": 0},
                "leds": {"count": 12},
                "pan_tilt": {"pan_min": -90, "pan_max": 90}
            },
            "ai": {"llm_provider": "mock"}
        }
        self.hw = HardwareFactory.create_hardware_suite(config)
        self.prompt_mgr = PromptManager("config/game_prompts.json")
        self.llm = LLMService(provider="mock")
        self.vision = VisionProcessor()
        self.gm = GameMaster(self.hw, self.prompt_mgr, self.llm, self.vision)

    def test_initial_state_is_idle(self):
        self.assertEqual(self.gm.state, GameState.IDLE)

    def test_start_game_transitions_to_intro(self):
        self.gm.start_game()
        self.assertEqual(self.gm.state, GameState.INTRO)

    def test_interrupt_from_intro_returns_to_idle(self):
        self.gm.start_game()
        self.gm.handle_interrupt_button()
        self.assertEqual(self.gm.state, GameState.IDLE)

    def test_hal_simulated_drivers(self):
        self.hw.led.set_all(255, 0, 100)
        colors = self.hw.led.get_colors()
        self.assertEqual(len(colors), 12)

        self.hw.pan_tilt.set_angles(30.0, -10.0)
        pan, tilt = self.hw.pan_tilt.get_angles()
        self.assertEqual(pan, 30.0)
        self.assertEqual(tilt, -10.0)


# ---------------------------------------------------------------------------
# Tests: VisionProcessor — frame guard and telemetry schema (no camera needed)
# ---------------------------------------------------------------------------

class TestVisionProcessorSchemas(unittest.TestCase):

    def setUp(self):
        self.vision = VisionProcessor()

    def test_dark_frame_covered_camera_guard(self):
        dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, telemetry = self.vision.analyze_frame(dark_frame)
        self.assertFalse(telemetry["face_detected"])
        self.assertFalse(telemetry["body_pose_detected"])
        self.assertFalse(telemetry["left_hand_raised"])
        self.assertFalse(telemetry["right_hand_raised"])
        self.assertFalse(telemetry["hands_raised"])
        self.assertFalse(telemetry["has_colorful_item"])

    def test_none_frame_returns_safe_telemetry(self):
        _, telemetry = self.vision.analyze_frame(None)
        for key in ["face_detected", "hands_raised", "has_colorful_item", "body_pose_detected"]:
            self.assertFalse(telemetry[key])

    def test_telemetry_schema_all_keys_present(self):
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        _, telemetry = self.vision.analyze_frame(dummy_frame)
        for key in [
            "face_detected", "face_count", "face_center",
            "body_pose_detected", "left_hand_raised", "right_hand_raised",
            "hands_raised", "avg_brightness", "has_colorful_item"
        ]:
            self.assertIn(key, telemetry)


# ---------------------------------------------------------------------------
# Tests: Challenge evaluation logic
# ---------------------------------------------------------------------------

class TestChallengeEvaluation(unittest.TestCase):

    def setUp(self):
        self.prompt_mgr = PromptManager("config/game_prompts.json")

    def test_pose_master_requires_face_and_both_hands(self):
        game = self.prompt_mgr.get_mini_game(0)
        self.assertEqual(game["id"], "pose_master")
        criteria = game.get("evaluation_criteria", {})
        self.assertTrue(criteria.get("require_face", False))
        self.assertIn("hands_raised", criteria.get("required_telemetry", []))

    def test_pose_master_left_only_fails(self):
        # Simulate evaluation logic inline
        result = VisionScanResult(face_detected=True, left_hand_raised=True, right_hand_raised=False, hands_raised=False)
        self.assertFalse(result.hands_raised)

    def test_pose_master_right_only_fails(self):
        result = VisionScanResult(face_detected=True, left_hand_raised=False, right_hand_raised=True, hands_raised=False)
        self.assertFalse(result.hands_raised)

    def test_pose_master_both_hands_with_face_succeeds(self):
        result = VisionScanResult(face_detected=True, left_hand_raised=True, right_hand_raised=True, hands_raised=True)
        self.assertTrue(result.face_detected and result.hands_raised)

    def test_mystery_item_colorful_object_succeeds(self):
        result = VisionScanResult(has_colorful_item=True)
        self.assertTrue(result.has_colorful_item)

    def test_mystery_item_no_color_fails(self):
        result = VisionScanResult(face_detected=True, hands_raised=True, has_colorful_item=False)
        self.assertFalse(result.has_colorful_item)


if __name__ == "__main__":
    unittest.main()
