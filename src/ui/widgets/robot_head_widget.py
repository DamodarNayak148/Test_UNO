from PySide6.QtWidgets import QWidget, QFrame
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient, QFont

class RobotHeadWidget(QFrame):
    """Simulated Robot Head viewport illustrating Servo Pan-Tilt position and facial expressions."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setMinimumSize(280, 240)
        self.setStyleSheet("background-color: #12141d; border-radius: 8px; border: 1px solid #2a2e3d;")

        self.pan: float = 0.0
        self.tilt: float = 0.0
        self.expression: str = "neutral"
        self.blink_state: bool = False

        # Idle blink timer
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._toggle_blink)
        self.blink_timer.start(3500)

    def update_head(self, pan: float, tilt: float, expression: str) -> None:
        self.pan = pan
        self.tilt = tilt
        self.expression = expression
        self.update()

    def _toggle_blink(self) -> None:
        self.blink_state = True
        self.update()
        QTimer.singleShot(180, self._end_blink)

    def _end_blink(self) -> None:
        self.blink_state = False
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Map pan/tilt to pixel offsets
        dx = int((self.pan / 90.0) * 45)
        dy = int((self.tilt / 45.0) * 30)

        cx = w // 2 + dx
        cy = h // 2 + dy - 10

        head_w, head_h = 160, 140

        # Neck joint base
        painter.setPen(QPen(QColor(40, 45, 60), 2))
        painter.setBrush(QBrush(QColor(25, 28, 38)))
        painter.drawRect(w // 2 - 20, h // 2 + 50, 40, 30)

        # Main Robot Head Frame
        painter.setPen(QPen(QColor(0, 200, 255), 2))
        head_grad = QRadialGradient(cx, cy, 100)
        head_grad.setColorAt(0.0, QColor(30, 35, 50))
        head_grad.setColorAt(1.0, QColor(15, 18, 25))
        painter.setBrush(QBrush(head_grad))
        painter.drawRoundedRect(cx - head_w // 2, cy - head_h // 2, head_w, head_h, 20, 20)

        # Visor Screen
        visor_w, visor_h = 120, 65
        visor_x = cx - visor_w // 2
        visor_y = cy - 25
        painter.setPen(QPen(QColor(0, 255, 200, 150), 1))
        painter.setBrush(QBrush(QColor(5, 10, 18)))
        painter.drawRoundedRect(visor_x, visor_y, visor_w, visor_h, 12, 12)

        # Draw Eyes based on Expression
        eye_color = QColor(0, 255, 200)
        if self.expression == "happy":
            eye_color = QColor(0, 255, 120)
        elif self.expression == "angry":
            eye_color = QColor(255, 50, 50)
        elif self.expression == "surprised":
            eye_color = QColor(255, 200, 0)

        left_eye_x = cx - 30
        right_eye_x = cx + 30
        eye_y = cy - 5

        if self.blink_state:
            # Closed eyes (blinking line)
            painter.setPen(QPen(eye_color, 4))
            painter.drawLine(left_eye_x - 12, eye_y, left_eye_x + 12, eye_y)
            painter.drawLine(right_eye_x - 12, eye_y, right_eye_x + 12, eye_y)
        elif self.expression == "surprised":
            painter.setPen(QPen(eye_color, 3))
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawEllipse(left_eye_x - 14, eye_y - 14, 28, 28)
            painter.drawEllipse(right_eye_x - 14, eye_y - 14, 28, 28)
        else:
            # Normal Glowing Eyes
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(eye_color))
            painter.drawEllipse(left_eye_x - 10, eye_y - 10, 20, 20)
            painter.drawEllipse(right_eye_x - 10, eye_y - 10, 20, 20)

        # Telemetry Labels below head
        painter.setPen(QColor(150, 160, 180))
        painter.setFont(QFont("Segoe UI", 9))
        info_text = f"Pan: {self.pan:.1f}° | Tilt: {self.tilt:.1f}° | Mood: {self.expression.upper()}"
        painter.drawText(10, h - 12, info_text)
