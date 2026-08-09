"""
camera_widget.py — WALLE Vision live camera feed widget.

Drives the VisionEngine on every QTimer tick:
  1. Grabs a frame from the HAL camera
  2. Passes it through VisionEngine.process_frame()
  3. Displays the annotated BGR frame
  4. Fires on_result(VisionResult) callback so the telemetry panel can update
"""

import cv2
import numpy as np
from typing import Optional, Callable

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from src.hal.base_camera import BaseCamera
from src.vision.vision_engine import VisionEngine
from src.vision.vision_result import VisionResult


class CameraWidget(QFrame):
    """
    Live camera viewport that runs VisionEngine on every frame.

    Parameters
    ----------
    camera : BaseCamera
        HAL camera driver (SimCamera or future UNO-Q driver).
    vision_engine : VisionEngine
        Shared VisionEngine instance.  Owned by the caller; not started/stopped here.
    on_result : optional callable(VisionResult)
        Called after each frame is processed so the telemetry panel can refresh.
    fps : int
        Target update rate (default 30).
    """

    def __init__(
        self,
        camera: BaseCamera,
        vision_engine: VisionEngine,
        on_result: Optional[Callable[[VisionResult], None]] = None,
        fps: int = 30,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.camera = camera
        self._engine = vision_engine
        self._on_result = on_result

        self.setMinimumSize(480, 360)
        self.setStyleSheet(
            "QFrame { background-color: #0b0d16; border-radius: 10px; "
            "border: 1px solid #252a3d; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("INITIALIZING WALLE VISION...")
        self.image_label.setStyleSheet(
            "color: #00d4ff; font-weight: bold; font-size: 14px; background: transparent; border: none;"
        )
        layout.addWidget(self.image_label)

        # Drive the loop via Qt timer — keeps everything on the GUI thread
        interval_ms = max(16, 1000 // fps)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(interval_ms)

    # ------------------------------------------------------------------
    # Internal frame update
    # ------------------------------------------------------------------

    def _update_frame(self) -> None:
        """Called every timer tick.  Grab → process → display → notify."""
        if not self.camera or not self.camera.is_opened():
            return

        frame = self.camera.get_frame()

        # Run VisionEngine (handles None/invalid frames internally)
        try:
            annotated, result = self._engine.process_frame(frame)
        except Exception as e:
            print(f"[CameraWidget] VisionEngine exception: {e}")
            return

        # Convert BGR (OpenCV) → RGB (Qt) for display
        try:
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            q_img = QImage(
                rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_img)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
        except Exception as e:
            print(f"[CameraWidget] Frame display error: {e}")
            return

        # Notify telemetry panel
        if self._on_result is not None:
            try:
                self._on_result(result)
            except Exception as e:
                print(f"[CameraWidget] on_result callback error: {e}")
