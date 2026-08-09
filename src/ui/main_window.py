"""
main_window.py — WALLE Vision Dashboard (replaces game-master dashboard).

Layout:
  ┌─────────────────────────────────────────────┐
  │            WALLE VISION   header            │
  ├───────────────────────────┬─────────────────┤
  │                           │                 │
  │   CameraWidget            │ TelemetryWidget │
  │   (live annotated feed)   │ (live stats)    │
  │                           │                 │
  └───────────────────────────┴─────────────────┘

The Game Master, LLM, PromptManager are intentionally NOT imported here.
They remain on disk for future re-integration.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSplitter, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from src.hal.hardware_factory import HardwareSuite
from src.vision.vision_engine import VisionEngine
from src.ui.widgets.camera_widget import CameraWidget
from src.ui.widgets.telemetry_widget import TelemetryWidget


class MainWindow(QMainWindow):
    """WALLE Vision Dashboard — real-time human perception display."""

    def __init__(self, hw: HardwareSuite, vision_engine: VisionEngine):
        super().__init__()
        self.hw = hw
        self._engine = vision_engine

        self.setWindowTitle("WALLE Vision — Real-Time Human Perception")
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

        # Telemetry panel (created first so we can pass its callback)
        self.telemetry = TelemetryWidget()

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
    # Header
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

        # Left: logo text
        logo = QLabel("⬡  WALLE VISION")
        logo.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        logo.setStyleSheet("color: #00d4ff; border: none; background: transparent; letter-spacing: 2px;")

        # Centre: subtitle
        sub = QLabel("Real-Time Human Perception  ·  AI Vision Layer")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color: #4a6fa5; font-size: 11px; border: none; background: transparent;")

        # Right: mode badge
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

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> QWidget:
        frame = QFrame()
        frame.setFixedHeight(24)
        frame.setStyleSheet(
            "QFrame { background: transparent; border: none; }"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4, 0, 4, 0)

        info = QLabel(
            "Detectors: Haar Face  ·  Haar Eyes  ·  MediaPipe Hands  ·  MediaPipe Pose    "
            "|    HAL: SimCamera    |    Press ESC to quit"
        )
        info.setStyleSheet("color: #3a4560; font-size: 10px;")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)
        return frame

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)
