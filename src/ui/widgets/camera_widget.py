import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from src.hal.base_camera import BaseCamera

class CameraWidget(QFrame):
    """Simulated Camera Viewport rendering live webcam / synthetic vision feed."""

    def __init__(self, camera: BaseCamera, parent: QWidget = None):
        super().__init__(parent)
        self.camera = camera
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #12141d; border-radius: 8px; border: 1px solid #2a2e3d;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("INITIALIZING CAMERA STREAM...")
        self.image_label.setStyleSheet("color: #00ffc8; font-weight: bold;")
        layout.addWidget(self.image_label)

        # 30 FPS Video Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)

    def _update_frame(self) -> None:
        if not self.camera or not self.camera.is_opened():
            return

        frame = self.camera.get_frame()
        if frame is None:
            return

        # Convert BGR (OpenCV) to RGB (Qt)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
