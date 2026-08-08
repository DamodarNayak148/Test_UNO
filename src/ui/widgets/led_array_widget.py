from typing import List, Tuple
from PySide6.QtWidgets import QWidget, QFrame
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QColor, QRadialGradient, QBrush, QPen

class LEDArrayWidget(QFrame):
    """Simulated NeoPixel RGB LED Strip Widget with radial glow effects."""

    def __init__(self, count: int = 12, parent: QWidget = None):
        super().__init__(parent)
        self.count = count
        self.led_colors: List[Tuple[int, int, int]] = [(0, 0, 0)] * count
        self.setMinimumHeight(60)
        self.setStyleSheet("background-color: #12141d; border-radius: 8px; border: 1px solid #2a2e3d;")

    def update_leds(self, colors: List[Tuple[int, int, int]]) -> None:
        """Update LED color array and trigger widget repaint."""
        self.led_colors = colors
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        padding = 15
        available_width = width - (2 * padding)
        spacing = available_width / max(1, self.count)
        radius = min(18, int(spacing * 0.35))

        cy = height // 2

        for i in range(self.count):
            cx = int(padding + (i + 0.5) * spacing)
            r, g, b = self.led_colors[i] if i < len(self.led_colors) else (0, 0, 0)
            
            # Base dark circle outline
            painter.setPen(QPen(QColor(50, 55, 75), 2))
            painter.setBrush(QBrush(QColor(20, 22, 30)))
            painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

            if r > 10 or g > 10 or b > 10:
                # Radial glow effect
                gradient = QRadialGradient(cx, cy, radius * 1.8)
                gradient.setColorAt(0.0, QColor(r, g, b, 255))
                gradient.setColorAt(0.5, QColor(r, g, b, 180))
                gradient.setColorAt(1.0, QColor(r, g, b, 0))

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(cx - int(radius * 1.8), cy - int(radius * 1.8), int(radius * 3.6), int(radius * 3.6))

                # Center bright core
                painter.setBrush(QBrush(QColor(r, g, b)))
                painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
