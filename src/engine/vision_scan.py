"""
vision_scan.py — Self-contained vision scan session for WALLE Game Master.

Responsible for:
- Safe multi-frame camera acquisition
- Per-frame error isolation
- Temporal consensus aggregation
- Interrupt support
- Clean scan result production

GameMaster delegates the entire SCANNING_VISION phase to VisionScanSession.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Scan configuration constants — one place, no hardcoding elsewhere
# ---------------------------------------------------------------------------
SCAN_FRAME_COUNT       = 6        # Total frames to attempt per scan
SCAN_FRAME_INTERVAL    = 0.18     # Seconds between frame captures
SCAN_INITIAL_DELAY     = 0.30     # Seconds to wait before first frame (pose stabilisation)
POSE_MIN_CONFIRMATIONS = 2        # Frames with hands_raised required for True consensus
FACE_MIN_CONFIRMATIONS = 2        # Frames with face required for True consensus
COLOR_MIN_CONFIRMATIONS = 1       # Frames with colorful item required for True (brief visibility)


# ---------------------------------------------------------------------------
# VisionScanResult — immutable value object representing one completed scan
# ---------------------------------------------------------------------------
@dataclass
class VisionScanResult:
    """Aggregated result of a completed vision scan session."""
    face_detected: bool = False
    face_count: int = 0
    face_center: tuple = (0, 0)
    body_pose_detected: bool = False
    left_hand_raised: bool = False
    right_hand_raised: bool = False
    hands_raised: bool = False
    avg_brightness: float = 0.0
    has_colorful_item: bool = False

    # Diagnostic counters (for logging/tests)
    valid_frames: int = 0
    attempted_frames: int = 0
    hands_confirmations: int = 0
    left_confirmations: int = 0
    right_confirmations: int = 0
    face_confirmations: int = 0
    pose_confirmations: int = 0
    color_confirmations: int = 0
    was_interrupted: bool = False
    scan_error: Optional[str] = None

    def as_telemetry(self) -> Dict[str, Any]:
        """Return a telemetry dict compatible with the GameMaster criteria evaluator."""
        return {
            "face_detected":     self.face_detected,
            "face_count":        self.face_count,
            "face_center":       self.face_center,
            "body_pose_detected": self.body_pose_detected,
            "left_hand_raised":  self.left_hand_raised,
            "right_hand_raised": self.right_hand_raised,
            "hands_raised":      self.hands_raised,
            "avg_brightness":    self.avg_brightness,
            "has_colorful_item": self.has_colorful_item,
        }

    @staticmethod
    def failure(reason: str = "scan_failed", interrupted: bool = False) -> "VisionScanResult":
        """Return a safe all-False result for error/interrupt paths."""
        return VisionScanResult(
            scan_error=reason,
            was_interrupted=interrupted,
        )


# ---------------------------------------------------------------------------
# VisionScanSession — executes one bounded scan and returns VisionScanResult
# ---------------------------------------------------------------------------
class VisionScanSession:
    """
    Encapsulates a single bounded vision scan session.

    Usage:
        session = VisionScanSession(camera, vision_processor)
        session.start()          # non-blocking — runs in background thread
        ...                      # later...
        result = session.wait()  # blocks until done or interrupted
    """

    def __init__(self, camera, vision_processor, interrupt_event: Optional[threading.Event] = None):
        """
        Args:
            camera:            HAL camera driver (anything with .get_frame() -> Optional[np.ndarray])
            vision_processor:  VisionProcessor with .analyze_frame(frame) -> (annotated, telemetry)
            interrupt_event:   Optional threading.Event; set it to request early abort.
        """
        self._camera = camera
        self._vision = vision_processor
        self._interrupt = interrupt_event or threading.Event()
        self._result: Optional[VisionScanResult] = None
        self._done_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the scan in a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="VisionScanThread")
        self._thread.start()

    def wait(self, timeout: float = 10.0) -> VisionScanResult:
        """
        Block until the scan completes or timeout expires.
        Always returns a VisionScanResult — never raises.
        """
        finished = self._done_event.wait(timeout=timeout)
        if not finished or self._result is None:
            return VisionScanResult.failure(
                reason="scan_timeout",
                interrupted=self._interrupt.is_set(),
            )
        return self._result

    def interrupt(self) -> None:
        """Signal the scan to abort early."""
        self._interrupt.set()

    # ------------------------------------------------------------------
    # Internal scan execution
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main scan loop — always signals _done_event before returning."""
        try:
            self._result = self._execute_scan()
        except Exception as exc:
            print(f"[SCAN] Unexpected top-level scan error: {exc}")
            self._result = VisionScanResult.failure(reason=str(exc))
        finally:
            self._done_event.set()

    def _execute_scan(self) -> VisionScanResult:
        """Perform the bounded multi-frame scan and return a result."""
        print("[SCAN] Starting vision scan...")

        # Initial stabilisation delay — interruptible
        if self._interrupt.wait(timeout=SCAN_INITIAL_DELAY):
            print("[SCAN] Interrupted during initial delay.")
            return VisionScanResult.failure(reason="interrupted_before_start", interrupted=True)

        observations: List[Dict[str, Any]] = []
        attempted = 0
        brightness_values: List[float] = []
        best_face_center = (0, 0)
        max_face_count = 0

        for i in range(SCAN_FRAME_COUNT):
            # Respect interrupt between frames
            if self._interrupt.is_set():
                print(f"[SCAN] Interrupted at frame {i+1}/{SCAN_FRAME_COUNT}.")
                break

            attempted += 1
            frame_ok, frame_tel = self._acquire_and_analyze_frame(i + 1, SCAN_FRAME_COUNT)

            if frame_ok and frame_tel is not None:
                observations.append(frame_tel)
                brightness_values.append(frame_tel.get("avg_brightness", 0.0))
                fc = frame_tel.get("face_center", (0, 0))
                if fc != (0, 0):
                    best_face_center = fc
                fcount = frame_tel.get("face_count", 0)
                if fcount > max_face_count:
                    max_face_count = fcount

            # Interruptible inter-frame delay
            if i < SCAN_FRAME_COUNT - 1:
                if self._interrupt.wait(timeout=SCAN_FRAME_INTERVAL):
                    print(f"[SCAN] Interrupted during frame interval after frame {i+1}.")
                    break

        return self._aggregate(
            observations=observations,
            attempted=attempted,
            brightness_values=brightness_values,
            best_face_center=best_face_center,
            max_face_count=max_face_count,
            interrupted=self._interrupt.is_set(),
        )

    def _acquire_and_analyze_frame(self, frame_num: int, total: int):
        """
        Safely acquire one camera frame and run VisionProcessor on it.
        Returns (success: bool, telemetry: Optional[dict]).
        Never raises.
        """
        try:
            frame = self._camera.get_frame()

            # Validate frame
            if frame is None:
                print(f"[SCAN] Frame {frame_num}/{total}: camera returned None — skipping.")
                return False, None

            if not hasattr(frame, 'shape') or len(frame.shape) != 3 or frame.size == 0:
                print(f"[SCAN] Frame {frame_num}/{total}: invalid frame shape — skipping.")
                return False, None

            # Run vision analysis
            _, telemetry = self._vision.analyze_frame(frame)

            hands    = telemetry.get("hands_raised", False)
            left_h   = telemetry.get("left_hand_raised", False)
            right_h  = telemetry.get("right_hand_raised", False)
            pose     = telemetry.get("body_pose_detected", False)
            face     = telemetry.get("face_detected", False)
            color    = telemetry.get("has_colorful_item", False)

            print(
                f"[SCAN] Frame {frame_num}/{total}: "
                f"face={face} pose={pose} "
                f"left={left_h} right={right_h} hands={hands} "
                f"color={color}"
            )
            return True, telemetry

        except Exception as exc:
            print(f"[SCAN] Frame {frame_num}/{total}: analysis exception — {exc} — skipping.")
            return False, None

    def _aggregate(
        self,
        observations: List[Dict[str, Any]],
        attempted: int,
        brightness_values: List[float],
        best_face_center: tuple,
        max_face_count: int,
        interrupted: bool,
    ) -> VisionScanResult:
        """Build a VisionScanResult from collected observations using temporal consensus."""
        valid = len(observations)

        if valid == 0:
            print("[SCAN] No valid frames collected — returning safe failure result.")
            return VisionScanResult.failure(
                reason="no_valid_frames",
                interrupted=interrupted,
            )

        # Count confirmations
        hands_conf = sum(1 for t in observations if t.get("hands_raised", False))
        left_conf  = sum(1 for t in observations if t.get("left_hand_raised", False))
        right_conf = sum(1 for t in observations if t.get("right_hand_raised", False))
        pose_conf  = sum(1 for t in observations if t.get("body_pose_detected", False))
        face_conf  = sum(1 for t in observations if t.get("face_detected", False))
        color_conf = sum(1 for t in observations if t.get("has_colorful_item", False))

        # Apply consensus thresholds
        hands_result = hands_conf >= POSE_MIN_CONFIRMATIONS
        left_result  = left_conf  >= POSE_MIN_CONFIRMATIONS
        right_result = right_conf >= POSE_MIN_CONFIRMATIONS
        pose_result  = pose_conf  >= POSE_MIN_CONFIRMATIONS
        face_result  = face_conf  >= FACE_MIN_CONFIRMATIONS
        color_result = color_conf >= COLOR_MIN_CONFIRMATIONS

        avg_brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 0.0

        # Print summary
        print(f"\n[SCAN RESULT]")
        print(f"  Valid frames: {valid}/{attempted}")
        print(f"  Face:       {face_conf}/{valid} -> {face_result}")
        print(f"  Pose:       {pose_conf}/{valid} -> {pose_result}")
        print(f"  Left hand:  {left_conf}/{valid} -> {left_result}")
        print(f"  Right hand: {right_conf}/{valid} -> {right_result}")
        print(f"  Both hands: {hands_conf}/{valid} -> {hands_result}")
        print(f"  Color item: {color_conf}/{valid} -> {color_result}")
        if interrupted:
            print(f"  [NOTE] Scan was interrupted early.")
        print()

        return VisionScanResult(
            face_detected      = face_result,
            face_count         = max_face_count if face_result else 0,
            face_center        = best_face_center if face_result else (0, 0),
            body_pose_detected = pose_result,
            left_hand_raised   = left_result,
            right_hand_raised  = right_result,
            hands_raised       = hands_result,
            avg_brightness     = avg_brightness,
            has_colorful_item  = color_result,
            valid_frames       = valid,
            attempted_frames   = attempted,
            hands_confirmations = hands_conf,
            left_confirmations  = left_conf,
            right_confirmations = right_conf,
            face_confirmations  = face_conf,
            pose_confirmations  = pose_conf,
            color_confirmations = color_conf,
            was_interrupted    = interrupted,
        )
