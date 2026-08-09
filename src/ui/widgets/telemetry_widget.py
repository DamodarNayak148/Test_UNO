"""
telemetry_widget.py — Live Vision Telemetry Panel for WALLE Vision Dashboard.

Displays real-time detection status for face (YuNet / MediaPipe), eyes, blink,
facial expression comparison (Heuristic vs Trained AI Model), hands, AI hand gestures,
individual fingers, pose and FPS.

Updated every frame via update_telemetry(result: VisionResult).
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from src.vision.vision_result import VisionResult

# Colour tokens
_YES   = "#00ff88"   # bright green — detected / open / up
_NO    = "#ff4466"   # red          — not detected / closed / down
_MAYBE = "#ffb830"   # amber        — partial / blink / active
_DIM   = "#5a6478"   # grey         — inactive label
_HEAD  = "#00d4ff"   # cyan         — section headers
_BG    = "#0e101a"
_CARD  = "#141623"
_BORDER= "#252a3d"


class _Section(QFrame):
    """A card-styled section grouping related telemetry rows."""

    def __init__(self, title: str, parent: QWidget = None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background-color: {_CARD}; border-radius: 8px; "
            f"border: 1px solid {_BORDER}; }}"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(5)

        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet(
            f"color: {_HEAD}; font-size: 10px; font-weight: bold; "
            f"letter-spacing: 1.5px; border: none; background: transparent;"
        )
        self._layout.addWidget(title_lbl)

    def add_row(self, label_text: str) -> QLabel:
        """Add a telemetry row and return its value label for later update."""
        row = QWidget()
        row.setStyleSheet("background: transparent; border: none;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        key = QLabel(label_text)
        key.setStyleSheet(f"color: {_DIM}; font-size: 11px; border: none; background: transparent;")
        key.setFixedWidth(115)

        val = QLabel("—")
        val.setStyleSheet(f"color: {_DIM}; font-size: 11px; font-weight: bold; border: none; background: transparent;")
        val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        rl.addWidget(key)
        rl.addWidget(val, 1)

        self._layout.addWidget(row)
        return val


class TelemetryWidget(QFrame):
    """
    Right-hand live telemetry panel for the WALLE Vision dashboard.

    Updated every frame by calling:
        self.update_telemetry(result: VisionResult)
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background-color: {_BG}; border: none; }}"
        )
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self.setMinimumWidth(230)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(8)

        # --- Header ---
        hdr = QLabel("VISION TELEMETRY")
        hdr.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(
            f"color: {_HEAD}; letter-spacing: 2px; background: transparent; border: none;"
        )
        root.addWidget(hdr)

        # --- Face section ---
        face_sec = _Section("Face Detection", self)
        self._v_face_detected = face_sec.add_row("Detector:")
        self._v_face_conf     = face_sec.add_row("Confidence:")
        self._v_face_mouth    = face_sec.add_row("Mouth:")
        self._v_face_brow     = face_sec.add_row("Eyebrows:")
        root.addWidget(face_sec)

        # --- Expression Comparison section (Heuristic vs Trained AI) ---
        expr_sec = _Section("Expression Comparison", self)
        self._v_expr_heuristic = expr_sec.add_row("Heuristic:")
        self._v_expr_ai        = expr_sec.add_row("AI Model:")
        self._v_expr_probs     = expr_sec.add_row("Top Probs:")
        root.addWidget(expr_sec)

        # --- Hand Gestures section ---
        gest_sec = _Section("Hand Gestures", self)
        self._v_gest_left  = gest_sec.add_row("Left Hand:")
        self._v_gest_right = gest_sec.add_row("Right Hand:")
        root.addWidget(gest_sec)

        # --- Eyes & Blink section ---
        eye_sec = _Section("Eyes & Blink", self)
        self._v_eye_left     = eye_sec.add_row("Left Eye:")
        self._v_eye_right    = eye_sec.add_row("Right Eye:")
        self._v_eye_blink    = eye_sec.add_row("Blink:")
        root.addWidget(eye_sec)

        # --- Left hand section ---
        lh_sec = _Section("Left Hand", self)
        self._v_lh_detected  = lh_sec.add_row("Detected:")
        self._v_lh_fingers   = lh_sec.add_row("Fingers up:")
        self._v_lh_states    = lh_sec.add_row("Thumb/Fingers:")
        root.addWidget(lh_sec)

        # --- Right hand section ---
        rh_sec = _Section("Right Hand", self)
        self._v_rh_detected  = rh_sec.add_row("Detected:")
        self._v_rh_fingers   = rh_sec.add_row("Fingers up:")
        self._v_rh_states    = rh_sec.add_row("Thumb/Fingers:")
        root.addWidget(rh_sec)

        # --- Body Pose section ---
        pose_sec = _Section("Body Pose", self)
        self._v_pose_detected = pose_sec.add_row("Detected:")
        self._v_pose_state    = pose_sec.add_row("Posture:")
        self._v_pose_arms     = pose_sec.add_row("Arms Raised:")
        root.addWidget(pose_sec)

        # --- Perf section ---
        perf_sec = _Section("Performance", self)
        self._v_fps        = perf_sec.add_row("FPS:")
        self._v_brightness = perf_sec.add_row("Brightness:")
        root.addWidget(perf_sec)

        root.addStretch(1)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update_telemetry(self, result: VisionResult) -> None:
        """Refresh all labels from the latest VisionResult."""
        self._update_face(result)
        self._update_expression(result)
        self._update_gestures(result)
        self._update_eyes(result)
        self._update_hand(
            result.left_hand,
            self._v_lh_detected, self._v_lh_fingers, self._v_lh_states,
        )
        self._update_hand(
            result.right_hand,
            self._v_rh_detected, self._v_rh_fingers, self._v_rh_states,
        )
        self._update_pose(result)
        self._update_perf(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set(self, label: QLabel, text: str, colour: str = _DIM) -> None:
        label.setText(text)
        label.setStyleSheet(
            f"color: {colour}; font-size: 11px; font-weight: bold; "
            f"border: none; background: transparent;"
        )

    def _update_face(self, result: VisionResult) -> None:
        f = result.face
        if f.detected:
            self._set(self._v_face_detected, f"{f.detector_source} ({f.count})", _YES)
            conf_pct = int(f.confidence * 100)
            self._set(self._v_face_conf, f"{conf_pct}%", _YES)
            mouth_str = "OPEN" if f.mouth_open else "Closed"
            self._set(self._v_face_mouth, mouth_str, _MAYBE if f.mouth_open else _DIM)
            brow_str = "RAISED ↑" if f.eyebrows_raised else "Normal"
            self._set(self._v_face_brow, brow_str, _YES if f.eyebrows_raised else _DIM)
        else:
            self._set(self._v_face_detected, "NO", _NO)
            self._set(self._v_face_conf, "—", _DIM)
            self._set(self._v_face_mouth, "—", _DIM)
            self._set(self._v_face_brow, "—", _DIM)

    def _update_expression(self, result: VisionResult) -> None:
        f = result.face
        if f.detected:
            h_expr = f.heuristic_expression
            h_col = _YES if h_expr in ("SMILE", "SURPRISED") else _HEAD
            self._set(self._v_expr_heuristic, h_expr, h_col)

            ai_expr = f.ai_expression
            ai_pct  = int(f.ai_confidence * 100)
            ai_str  = f"{ai_expr} {ai_pct}%" if f.ai_confidence > 0 else ai_expr
            ai_col  = _YES if ai_expr in ("Happiness", "Surprise") else _HEAD
            self._set(self._v_expr_ai, ai_str, ai_col)

            # Format top 3 probabilities
            probs = f.ai_expression_probabilities
            if probs:
                sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:3]
                probs_str = " ".join([f"{k[:4]}:{int(v*100)}%" for k, v in sorted_probs])
                self._set(self._v_expr_probs, probs_str, _YES)
            else:
                self._set(self._v_expr_probs, "—", _DIM)
        else:
            self._set(self._v_expr_heuristic, "—", _DIM)
            self._set(self._v_expr_ai,        "—", _DIM)
            self._set(self._v_expr_probs,     "—", _DIM)

    def _update_gestures(self, result: VisionResult) -> None:
        lh = result.left_hand
        rh = result.right_hand

        if lh.detected and lh.gesture != "UNKNOWN":
            conf_pct = int(lh.gesture_confidence * 100)
            c_str = f" {conf_pct}%" if conf_pct > 0 else ""
            self._set(self._v_gest_left, f"{lh.gesture}{c_str}", _YES)
        elif lh.detected:
            self._set(self._v_gest_left, "Unknown", _MAYBE)
        else:
            self._set(self._v_gest_left, "—", _DIM)

        if rh.detected and rh.gesture != "UNKNOWN":
            conf_pct = int(rh.gesture_confidence * 100)
            c_str = f" {conf_pct}%" if conf_pct > 0 else ""
            self._set(self._v_gest_right, f"{rh.gesture}{c_str}", _YES)
        elif rh.detected:
            self._set(self._v_gest_right, "Unknown", _MAYBE)
        else:
            self._set(self._v_gest_right, "—", _DIM)

    def _update_eyes(self, result: VisionResult) -> None:
        e = result.eyes
        if e.detected:
            l_str = "OPEN" if e.left_open else "CLOSED"
            r_str = "OPEN" if e.right_open else "CLOSED"
            self._set(self._v_eye_left,  l_str, _YES if e.left_open else _NO)
            self._set(self._v_eye_right, r_str, _YES if e.right_open else _NO)

            blink_str = f"YES ({e.blink_count})" if e.blink_detected else f"No ({e.blink_count})"
            self._set(self._v_eye_blink, blink_str, _MAYBE if e.blink_detected else _DIM)
        else:
            self._set(self._v_eye_left,  "—", _DIM)
            self._set(self._v_eye_right, "—", _DIM)
            self._set(self._v_eye_blink, f"No ({e.blink_count})", _DIM)

    def _update_hand(
        self,
        hand,
        v_detected: QLabel,
        v_fingers: QLabel,
        v_states: QLabel,
    ) -> None:
        if hand.detected:
            self._set(v_detected, "YES ✓", _YES)
            n = hand.fingers_up
            total = 5
            colour = _YES if n == total else (_MAYBE if n > 0 else _NO)
            self._set(v_fingers, f"{n}/{total}", colour)

            names = ["T", "I", "M", "R", "P"]
            icons = "".join(
                f"[{n}]" if up else f" {n} "
                for n, up in zip(names, hand.finger_states)
            )
            self._set(v_states, icons, _YES if n > 0 else _DIM)
        else:
            self._set(v_detected, "NO", _NO)
            self._set(v_fingers, "—", _DIM)
            self._set(v_states, "—", _DIM)

    def _update_pose(self, result: VisionResult) -> None:
        p = result.pose
        if p.detected:
            self._set(self._v_pose_detected, "YES ✓", _YES)
            posture = "Standing" if p.standing else ("Sitting" if p.sitting else "Tracked")
            self._set(self._v_pose_state, posture, _YES)
            arms_str = "Both ↑" if p.hands_raised else ("Left ↑" if p.left_hand_raised else ("Right ↑" if p.right_hand_raised else "Down"))
            self._set(self._v_pose_arms, arms_str, _YES if (p.left_hand_raised or p.right_hand_raised) else _DIM)
        else:
            self._set(self._v_pose_detected, "NO", _NO)
            self._set(self._v_pose_state, "—", _DIM)
            self._set(self._v_pose_arms, "—", _DIM)

    def _update_perf(self, result: VisionResult) -> None:
        fps = result.fps
        fps_colour = _YES if fps >= 20 else (_MAYBE if fps >= 10 else _NO)
        self._set(self._v_fps, f"{fps:.1f}", fps_colour)
        b = result.avg_brightness
        b_colour = _NO if result.camera_dark else (_YES if b > 60 else _MAYBE)
        self._set(self._v_brightness, f"{b:.0f}" + (" [DARK]" if result.camera_dark else ""), b_colour)
