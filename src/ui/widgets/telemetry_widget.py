"""
telemetry_widget.py — Redesigned WALLE Diagnostic Telemetry Panel.

Consolidated 5-Card Ergonomic Layout:
  1. SYSTEM & PERFORMANCE  -- FPS, Process Latency (ms), Brightness
  2. FACE & EXPRESSION     -- YuNet/Mesh, AI Expression, Eyes, Blink, Mouth, Eyebrows
  3. HANDS & GESTURES      -- Two-column Left & Right Hand gestures, confidence & finger states
  4. BODY POSE             -- Posture (Standing/Sitting) & Arms Raised status
  5. VOICE INPUT (Vosk)    -- Status (Ready/Listening/Transcribing), Hotkey & Transcript
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from src.vision.vision_result import VisionResult
from src.voice.voice_result import VoiceResult

# Color tokens
_YES   = "#00ff88"   # bright green -- detected / open / ready
_NO    = "#ff4466"   # red          -- not detected / closed / error
_MAYBE = "#ffb830"   # amber        -- listening / transcribing / partial
_DIM   = "#5a6478"   # grey         -- inactive label
_HEAD  = "#00d4ff"   # cyan         -- section headers
_BG    = "#080a12"
_CARD  = "#121522"
_BORDER= "#21263b"


class _Card(QFrame):
    """Compact card grouping related telemetry metrics."""

    def __init__(self, title: str, parent: QWidget = None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background-color: {_CARD}; border-radius: 8px; "
            f"border: 1px solid {_BORDER}; }}"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(4)

        hdr = QLabel(title.upper())
        hdr.setStyleSheet(
            f"color: {_HEAD}; font-size: 10px; font-weight: bold; "
            f"letter-spacing: 1.5px; border: none; background: transparent;"
        )
        self._layout.addWidget(hdr)

    def add_row(self, label_text: str, key_width: int = 100) -> QLabel:
        """Add a key-value row and return the value label for updates."""
        row = QWidget()
        row.setStyleSheet("background: transparent; border: none;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        key = QLabel(label_text)
        key.setStyleSheet(f"color: {_DIM}; font-size: 11px; border: none; background: transparent;")
        key.setFixedWidth(key_width)

        val = QLabel("—")
        val.setStyleSheet(f"color: {_DIM}; font-size: 11px; font-weight: bold; border: none; background: transparent;")
        val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        rl.addWidget(key)
        rl.addWidget(val, 1)

        self._layout.addWidget(row)
        return val


class TelemetryWidget(QFrame):
    """
    WALLE Telemetry Panel — compact 5-card diagnostic layout with scrolling support.

    Updated via:
        update_telemetry(vision_result)
        update_voice_telemetry(voice_result)
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background-color: {_BG}; border: none; }}")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(260)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area container so the telemetry panel never clips at smaller window sizes
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{ background: {_BG}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: {_BORDER}; border-radius: 3px; }}"
        )

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        root = QVBoxLayout(content)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(8)

        # Header
        hdr = QLabel("WALLE DIAGNOSTICS")
        hdr.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(f"color: {_HEAD}; letter-spacing: 2px; background: transparent; border: none;")
        root.addWidget(hdr)

        # --- CARD 1: SYSTEM & PERFORMANCE ---
        perf_card = _Card("System & Performance", content)
        self._v_fps       = perf_card.add_row("Vision FPS:")
        self._v_proc_time = perf_card.add_row("Process Latency:")
        self._v_bright    = perf_card.add_row("Brightness:")
        root.addWidget(perf_card)

        # --- CARD 2: FACE & EXPRESSION ---
        face_card = _Card("Face & Expression", content)
        self._v_face_det   = face_card.add_row("Detector:")
        self._v_expr_ai    = face_card.add_row("AI Emotion:")
        self._v_expr_probs = face_card.add_row("Top Probs:")
        self._v_eyes       = face_card.add_row("Eyes / Blink:")
        self._v_mouth_brow = face_card.add_row("Mouth / Brow:")
        root.addWidget(face_card)

        # --- CARD 3: HANDS & GESTURES (Two-Column Layout) ---
        hands_card = QFrame(content)
        hands_card.setStyleSheet(
            f"QFrame {{ background-color: {_CARD}; border-radius: 8px; border: 1px solid {_BORDER}; }}"
        )
        h_layout = QVBoxLayout(hands_card)
        h_layout.setContentsMargins(10, 8, 10, 8)
        h_layout.setSpacing(4)

        h_title = QLabel("HANDS & GESTURES")
        h_title.setStyleSheet(f"color: {_HEAD}; font-size: 10px; font-weight: bold; letter-spacing: 1.5px; border: none; background: transparent;")
        h_layout.addWidget(h_title)

        cols_widget = QWidget()
        cols_widget.setStyleSheet("background: transparent; border: none;")
        cols_hl = QHBoxLayout(cols_widget)
        cols_hl.setContentsMargins(0, 0, 0, 0)
        cols_hl.setSpacing(10)

        # Left Hand Col
        lh_box = QVBoxLayout()
        lh_lbl = QLabel("LEFT HAND")
        lh_lbl.setStyleSheet(f"color: {_HEAD}; font-size: 9px; font-weight: bold;")
        self._v_lh_gest   = QLabel("—")
        self._v_lh_fingers = QLabel("—")
        self._v_lh_states  = QLabel("—")
        for lbl in (self._v_lh_gest, self._v_lh_fingers, self._v_lh_states):
            lbl.setStyleSheet(f"color: {_DIM}; font-size: 10px; font-weight: bold;")

        lh_box.addWidget(lh_lbl)
        lh_box.addWidget(self._v_lh_gest)
        lh_box.addWidget(self._v_lh_fingers)
        lh_box.addWidget(self._v_lh_states)

        # Right Hand Col
        rh_box = QVBoxLayout()
        rh_lbl = QLabel("RIGHT HAND")
        rh_lbl.setStyleSheet(f"color: {_HEAD}; font-size: 9px; font-weight: bold;")
        self._v_rh_gest   = QLabel("—")
        self._v_rh_fingers = QLabel("—")
        self._v_rh_states  = QLabel("—")
        for lbl in (self._v_rh_gest, self._v_rh_fingers, self._v_rh_states):
            lbl.setStyleSheet(f"color: {_DIM}; font-size: 10px; font-weight: bold;")

        rh_box.addWidget(rh_lbl)
        rh_box.addWidget(self._v_rh_gest)
        rh_box.addWidget(self._v_rh_fingers)
        rh_box.addWidget(self._v_rh_states)

        cols_hl.addLayout(lh_box, 1)
        cols_hl.addLayout(rh_box, 1)
        h_layout.addWidget(cols_widget)
        root.addWidget(hands_card)

        # --- CARD 4: BODY POSE ---
        pose_card = _Card("Body Pose", content)
        self._v_pose_state = pose_card.add_row("Posture:")
        self._v_pose_arms  = pose_card.add_row("Arms Raised:")
        root.addWidget(pose_card)

        # --- CARD 5: VOICE INPUT (Vosk STT) ---
        voice_card = _Card("Voice Input (Vosk STT)", content)
        self._v_voice_status     = voice_card.add_row("Status:")
        self._v_voice_hotkey     = voice_card.add_row("Hotkey:")
        self._v_voice_transcript = voice_card.add_row("Transcript:")
        self._set(self._v_voice_hotkey, "[ Hold SPACE ]", _HEAD)
        root.addWidget(voice_card)

        root.addStretch(1)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Telemetry Updates
    # ------------------------------------------------------------------

    def update_telemetry(self, result: VisionResult) -> None:
        """Update vision telemetry metrics."""
        # 1. Performance Card
        fps = result.fps
        fps_col = _YES if fps >= 18.0 else (_MAYBE if fps >= 10.0 else _NO)
        self._set(self._v_fps, f"{fps:.1f} FPS", fps_col)

        proc_ms = result.process_time_ms
        self._set(self._v_proc_time, f"{proc_ms:.1f} ms", _YES if proc_ms < 55.0 else _MAYBE)

        b = result.avg_brightness
        b_str = f"{b:.0f}" + (" [DARK]" if result.camera_dark else "")
        self._set(self._v_bright, b_str, _NO if result.camera_dark else _YES)

        # 2. Face Card
        f = result.face
        if f.detected:
            self._set(self._v_face_det, f"{f.detector_source} ({f.count})", _YES)

            ai_expr = f.ai_expression
            ai_conf = int(f.ai_confidence * 100)
            ai_str  = f"{ai_expr} ({ai_conf}%)" if f.ai_confidence > 0 else ai_expr
            self._set(self._v_expr_ai, ai_str, _YES if ai_expr in ("Happiness", "Surprise") else _HEAD)

            probs = f.ai_expression_probabilities
            if probs:
                sorted_p = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:3]
                p_str = " ".join([f"{k[:4]}:{int(v*100)}%" for k, v in sorted_p])
                self._set(self._v_expr_probs, p_str, _YES)
            else:
                self._set(self._v_expr_probs, "—", _DIM)

            e = result.eyes
            l_str = "OPEN" if e.left_open else "CLOSED"
            r_str = "OPEN" if e.right_open else "CLOSED"
            blink_str = f"Blink ({e.blink_count})" if e.blink_detected else "No Blink"
            self._set(self._v_eyes, f"{l_str}/{r_str} | {blink_str}", _YES if (e.left_open or e.right_open) else _NO)

            m_str = "Mouth:OPEN" if f.mouth_open else "Mouth:Closed"
            b_str = "Brow:RAISED" if f.eyebrows_raised else "Brow:Normal"
            self._set(self._v_mouth_brow, f"{m_str} | {b_str}", _MAYBE if (f.mouth_open or f.eyebrows_raised) else _DIM)
        else:
            self._set(self._v_face_det, "NO FACE", _NO)
            self._set(self._v_expr_ai, "—", _DIM)
            self._set(self._v_expr_probs, "—", _DIM)
            self._set(self._v_eyes, "—", _DIM)
            self._set(self._v_mouth_brow, "—", _DIM)

        # 3. Hands Card (Two-column)
        lh = result.left_hand
        rh = result.right_hand

        if lh.detected:
            g_str = f"{lh.gesture}" if lh.gesture != "UNKNOWN" else "Tracked"
            self._set(self._v_lh_gest, f"● {g_str}", _YES)
            self._set(self._v_lh_fingers, f"Fingers: {lh.fingers_up}/5", _YES)
            icons = "".join(f"[{n[0].upper()}]" if u else f" {n[0].upper()} " for n, u in zip(["t","i","m","r","p"], lh.finger_states))
            self._set(self._v_lh_states, icons, _YES)
        else:
            self._set(self._v_lh_gest, "○ None", _DIM)
            self._set(self._v_lh_fingers, "Fingers: 0/5", _DIM)
            self._set(self._v_lh_states, "—", _DIM)

        if rh.detected:
            g_str = f"{rh.gesture}" if rh.gesture != "UNKNOWN" else "Tracked"
            self._set(self._v_rh_gest, f"● {g_str}", _YES)
            self._set(self._v_rh_fingers, f"Fingers: {rh.fingers_up}/5", _YES)
            icons = "".join(f"[{n[0].upper()}]" if u else f" {n[0].upper()} " for n, u in zip(["t","i","m","r","p"], rh.finger_states))
            self._set(self._v_rh_states, icons, _YES)
        else:
            self._set(self._v_rh_gest, "○ None", _DIM)
            self._set(self._v_rh_fingers, "Fingers: 0/5", _DIM)
            self._set(self._v_rh_states, "—", _DIM)

        # 4. Pose Card
        p = result.pose
        if p.detected:
            posture = "Standing" if p.standing else ("Sitting" if p.sitting else "Tracked")
            self._set(self._v_pose_state, f"● {posture}", _YES)
            arms = "Both ↑" if p.hands_raised else ("Left ↑" if p.left_hand_raised else ("Right ↑" if p.right_hand_raised else "Down"))
            self._set(self._v_pose_arms, arms, _YES if (p.left_hand_raised or p.right_hand_raised) else _DIM)
        else:
            self._set(self._v_pose_state, "○ None", _DIM)
            self._set(self._v_pose_arms, "—", _DIM)

    def update_voice_telemetry(self, result: VoiceResult) -> None:
        """Update voice telemetry metrics."""
        if result.is_listening:
            self._set(self._v_voice_status, "● LISTENING...", _MAYBE)
            self._set(self._v_voice_transcript, "...", _MAYBE)
        elif result.is_transcribing:
            self._set(self._v_voice_status, "● Transcribing...", _MAYBE)
            self._set(self._v_voice_transcript, "Processing...", _MAYBE)
        elif result.error:
            self._set(self._v_voice_status, "Error", _NO)
            self._set(self._v_voice_transcript, f"[{result.error}]", _NO)
        else:
            self._set(self._v_voice_status, "● Ready", _YES)
            t_str = f'"{result.text}"' if result.text else "—"
            self._set(self._v_voice_transcript, t_str, _YES if result.text else _DIM)

    def _set(self, label: QLabel, text: str, colour: str = _DIM) -> None:
        label.setText(text)
        label.setStyleSheet(
            f"color: {colour}; font-size: 11px; font-weight: bold; "
            f"border: none; background: transparent;"
        )
