import time
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtGui import QFont

class LogWidget(QFrame):
    """Telemetry and event console logging Game Master actions in real-time."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #12141d; border-radius: 8px; border: 1px solid #2a2e3d;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Game Master Activity Log & Telemetry")
        title.setStyleSheet("color: #8a99ad; font-weight: bold; font-size: 11px;")
        layout.addWidget(title)

        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QFont("Consolas", 9))
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0b0c12;
                color: #00ffc8;
                border: 1px solid #1f2333;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.text_area)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.text_area.append(f"[{timestamp}] {message}")
        # Auto scroll to bottom
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())
