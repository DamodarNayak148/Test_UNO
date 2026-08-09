"""
main.py — WALLE Vision entry point.

Active flow:
    Webcam → HAL (SimCamera) → VisionEngine → VisionResult → WALLE Vision Dashboard

Game Master, LLM, PromptManager are NOT imported here.
They remain in src/engine/ for future re-integration once the vision
perception layer is stable.
"""

import sys
import yaml
import os
from PySide6.QtWidgets import QApplication

from src.hal.hardware_factory import HardwareFactory
from src.vision.vision_engine import VisionEngine
from src.ui.main_window import MainWindow


def load_config(config_path: str = "config/default_config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    print("=" * 54)
    print("         WALLE VISION — Real-Time Human Perception      ")
    print("=" * 54)

    # 1. Load Configuration
    config = load_config()

    # 2. Instantiate Hardware Suite via HAL Factory (unchanged)
    hw = HardwareFactory.create_hardware_suite(config)
    hw.camera.start_stream()

    # 3. Instantiate Vision Engine
    vision_engine = VisionEngine()

    # 4. Launch PySide6 Vision Dashboard
    app = QApplication(sys.argv)
    window = MainWindow(hw=hw, vision_engine=vision_engine)
    window.show()

    # Run Event Loop
    exit_code = app.exec()

    # Cleanup Hardware Resources on Close
    hw.camera.stop_stream()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
