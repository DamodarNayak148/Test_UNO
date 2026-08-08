from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent

from src.hal.hardware_factory import HardwareSuite
from src.engine.game_master import GameMaster
from src.ui.widgets.led_array_widget import LEDArrayWidget
from src.ui.widgets.robot_head_widget import RobotHeadWidget
from src.ui.widgets.camera_widget import CameraWidget
from src.ui.widgets.player_controls import PlayerControlsWidget
from src.ui.widgets.log_widget import LogWidget

class MainWindow(QMainWindow):
    """Main Windows Prototype Dashboard Window."""

    def __init__(self, hw: HardwareSuite, gm: GameMaster):
        super().__init__()
        self.hw = hw
        self.gm = gm

        self.setWindowTitle("Physical AI Game Master - PC Simulation Dashboard")
        self.resize(1024, 720)
        self.setStyleSheet("background-color: #0b0c10; color: #ffffff;")

        # Central Widget & Main Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Header Title Banner
        header = QLabel("PHYSICAL AI GAME MASTER (SIMULATION DASHBOARD)")
        header.setStyleSheet("color: #00ffc8; font-size: 16px; font-weight: bold; padding: 4px;")
        main_layout.addWidget(header)

        # 1. RGB LED Simulation Array (Top Bar)
        self.led_widget = LEDArrayWidget(count=12)
        main_layout.addWidget(self.led_widget)

        # 2. Middle Row Splitter: Robot Head (Left) + Camera Feed (Right)
        mid_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.head_widget = RobotHeadWidget()
        self.camera_widget = CameraWidget(camera=self.hw.camera)
        
        mid_splitter.addWidget(self.head_widget)
        mid_splitter.addWidget(self.camera_widget)
        mid_splitter.setSizes([450, 550])
        main_layout.addWidget(mid_splitter)

        # 3. Control Panel & Console Log
        self.controls_widget = PlayerControlsWidget(buttons=self.hw.buttons)
        self.log_widget = LogWidget()

        main_layout.addWidget(self.controls_widget)
        main_layout.addWidget(self.log_widget)

        # Wire Hardware Callbacks to UI Widgets
        if hasattr(self.hw.led, "set_update_callback"):
            self.hw.led.set_update_callback(self.led_widget.update_leds)

        if hasattr(self.hw.pan_tilt, "set_update_callback"):
            self.hw.pan_tilt.set_update_callback(self.head_widget.update_head)

        if hasattr(self.hw.audio, "set_state_change_callback"):
            self.hw.audio.set_state_change_callback(
                lambda is_speaking, text: self.log_widget.log(f"[TTS Audio]: Speaking '{text}'...") if is_speaking else None
            )

        # Connect Game Master state observer to log widget
        self.gm.set_state_change_callback(
            lambda state, msg: self.log_widget.log(f"[{state.name}] {msg}")
        )

        # Connect Personality selector change
        self.controls_widget.combo_personality.currentTextChanged.connect(
            self._on_personality_changed
        )

        # Install Keyboard Event Filter
        self.installEventFilter(self)
        self.log_widget.log("System initialized. Press 'A' (P1), 'S' (P2), 'D' (Action) or 'Space' (Interrupt).")

    def _on_personality_changed(self, text: str) -> None:
        self.gm.set_personality(text)
        self.log_widget.log(f"Selected GM Personality: {text}")

    def eventFilter(self, watched, event) -> bool:
        """Capture physical keyboard shortcuts for hardware button simulation."""
        if event.type() == QEvent.Type.KeyPress:
            key_event: QKeyEvent = event
            key = key_event.key()
            if key == Qt.Key.Key_A:
                self.hw.buttons.trigger_button("player1")
                return True
            elif key == Qt.Key.Key_S:
                self.hw.buttons.trigger_button("player2")
                return True
            elif key == Qt.Key.Key_D:
                self.hw.buttons.trigger_button("action")
                return True
            elif key == Qt.Key.Key_Space:
                self.hw.buttons.trigger_button("interrupt")
                return True

        return super().eventFilter(watched, event)
