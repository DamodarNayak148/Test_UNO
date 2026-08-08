from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QLabel, QGroupBox
)
from PySide6.QtCore import Qt
from src.hal.base_buttons import BaseButtons

class PlayerControlsWidget(QFrame):
    """Simulated Player Control Panel triggering physical player inputs & setting GM settings."""

    def __init__(self, buttons: BaseButtons, parent: QWidget = None):
        super().__init__(parent)
        self.buttons = buttons
        self.setStyleSheet("""
            QFrame {
                background-color: #12141d;
                border-radius: 8px;
                border: 1px solid #2a2e3d;
            }
            QLabel {
                color: #e0e6ed;
                font-weight: bold;
            }
            QComboBox {
                background-color: #1b1e2e;
                color: #00ffc8;
                border: 1px solid #3b4261;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }
        """)

        main_layout = QVBoxLayout(self)

        # 1. Personality Selection Row
        pers_layout = QHBoxLayout()
        pers_label = QLabel("GM Personality:")
        self.combo_personality = QComboBox()
        self.combo_personality.addItems(["DungeonMaster", "SciFiAI", "PartyHost"])
        pers_layout.addWidget(pers_label)
        pers_layout.addWidget(self.combo_personality)
        main_layout.addLayout(pers_layout)

        # 2. Simulated Physical Buttons Layout
        btn_box = QGroupBox("Simulated Physical Buttons (Keyboard Key Mappings)")
        btn_box.setStyleSheet("QGroupBox { color: #8a99ad; font-size: 11px; border: 1px solid #2a2e3d; margin-top: 10px; }")
        btn_layout = QHBoxLayout(btn_box)

        self.btn_p1 = QPushButton("PLAYER 1\n[Key: A]")
        self.btn_p1.setStyleSheet("background-color: #2b5cff; color: white;")
        self.btn_p1.clicked.connect(lambda: self.buttons.trigger_button("player1"))

        self.btn_p2 = QPushButton("PLAYER 2\n[Key: S]")
        self.btn_p2.setStyleSheet("background-color: #9d2bff; color: white;")
        self.btn_p2.clicked.connect(lambda: self.buttons.trigger_button("player2"))

        self.btn_action = QPushButton("ACTION / CONFIRM\n[Key: D]")
        self.btn_action.setStyleSheet("background-color: #00b87c; color: white;")
        self.btn_action.clicked.connect(lambda: self.buttons.trigger_button("action"))

        self.btn_interrupt = QPushButton("INTERRUPT / RESET\n[Spacebar]")
        self.btn_interrupt.setStyleSheet("background-color: #ff3b5c; color: white;")
        self.btn_interrupt.clicked.connect(lambda: self.buttons.trigger_button("interrupt"))

        btn_layout.addWidget(self.btn_p1)
        btn_layout.addWidget(self.btn_p2)
        btn_layout.addWidget(self.btn_action)
        btn_layout.addWidget(self.btn_interrupt)

        main_layout.addWidget(btn_box)
