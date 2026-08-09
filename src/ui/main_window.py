"""
main_window.py — WALLE Vision & Voice Dashboard.

Layout:
  ┌─────────────────────────────────────────────┐
  │         WALLE VISION & VOICE  header        │
  ├───────────────────────────┬─────────────────┤
  │                           │                 │
  │   CameraWidget            │ TelemetryWidget │
  │   (live annotated feed)   │ (live stats)    │
  │                           │                 │
  └───────────────────────────┴─────────────────┘

Features:
  - Real-time AI Vision (YuNet + MediaPipe + HSEmotionONNX + GestureRecognizer + Pose)
  - Push-to-Talk Local Speech-to-Text Voice Engine (Hold SPACE to talk)
  - Non-blocking Qt UI architecture
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSplitter, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeyEvent

from src.hal.hardware_factory import HardwareSuite
from src.vision.vision_engine import VisionEngine
from src.voice.voice_engine import VoiceEngine
from src.ui.widgets.camera_widget import CameraWidget
from src.ui.widgets.telemetry_widget import TelemetryWidget


class MainWindow(QMainWindow):
    """WALLE Vision & Voice Dashboard — real-time human perception display."""

    def __init__(self, hw: HardwareSuite, vision_engine: VisionEngine):
        super().__init__()
        self.hw = hw
        self._engine = vision_engine

        # Voice Subsystem
        self.voice_engine = VoiceEngine()

        self.setWindowTitle("WALLE Vision & Voice — Local Perception Dashboard")
        self.resize(1280, 760)
        self.setMinimumSize(900, 600)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #080a12;
                color: #e0e6ed;
                font-family: 'Segoe UI', sans-serif;
            }
            QSplitter::handle {
                background-color: #1e2235;
                width: 3px;
            }
        """)

        # ── Root layout ──────────────────────────────────────────────────
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)

        # ── Header bar ───────────────────────────────────────────────────
        header = self._build_header()
        root.addWidget(header)

        # ── Main content: camera (left) + telemetry (right) ──────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Telemetry panel
        self.telemetry = TelemetryWidget()

        # Connect voice result signals to telemetry panel
        self.voice_engine.result_updated.connect(self.telemetry.update_voice_telemetry)
        self.telemetry.update_voice_telemetry(self.voice_engine.get_result())

        # Camera widget — wired to VisionEngine and telemetry updater
        self.camera_widget = CameraWidget(
            camera=hw.camera,
            vision_engine=vision_engine,
            on_result=self.telemetry.update_telemetry,
            fps=30,
        )

        splitter.addWidget(self.camera_widget)
        splitter.addWidget(self.telemetry)
        splitter.setSizes([880, 300])

        root.addWidget(splitter, stretch=1)

        # ── Status bar ───────────────────────────────────────────────────
        status = self._build_status_bar()
        root.addWidget(status)

    # ------------------------------------------------------------------
    # Push-to-Talk Keyboard Event Handling (Hold SPACE to talk)
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Key press event: Hold SPACE to start recording."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self.voice_engine.is_listening():
                self.voice_engine.start_recording()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Key release event: Release SPACE to transcribe audio."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self.voice_engine.is_listening():
                self.voice_engine.stop_recording_and_transcribe()
        else:
            super().keyReleaseEvent(event)

    # ------------------------------------------------------------------
    # Header & Status Bar
    # ------------------------------------------------------------------

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setFixedHeight(52)
        frame.setStyleSheet(
            "QFrame { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0,"
            "stop:0 #0d1b3e, stop:0.5 #0e2040, stop:1 #0d1b3e);"
            "border-radius: 8px; border: 1px solid #1f2d55; }"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 0, 16, 0)

        logo = QLabel("⬡  WALLE VISION & VOICE")
        logo.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        logo.setStyleSheet("color: #00d4ff; border: none; background: transparent; letter-spacing: 2px;")

        sub = QLabel("Real-Time Perception  ·  Vision + Local Vosk STT Voice Layer")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color: #4a6fa5; font-size: 11px; border: none; background: transparent;")

        badge = QLabel("LIVE  ●")
        badge.setStyleSheet(
            "color: #00ff88; font-size: 12px; font-weight: bold; "
            "background: #0a2215; border: 1px solid #00ff88; "
            "border-radius: 4px; padding: 2px 8px;"
        )

        layout.addWidget(logo)
        layout.addWidget(sub, stretch=1)
        layout.addWidget(badge)
        return frame

    def _build_status_bar(self) -> QWidget:
        frame = QFrame()
        frame.setFixedHeight(24)
        frame.setStyleSheet(
            "QFrame { background: transparent; border: none; }"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4, 0, 4, 0)

        info = QLabel(
            "Perception: YuNet Face  ·  HSEmotion AI  ·  Gesture AI  ·  Vosk Local STT [Hold SPACE to Talk]    "
            "|    HAL: Independent    |    Press ESC to quit"
        )
        info.setStyleSheet("color: #3a4560; font-size: 10px;")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)
        return frame
