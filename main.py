import sys
import yaml
import os
from PySide6.QtWidgets import QApplication

from src.hal.hardware_factory import HardwareFactory
from src.engine.prompt_manager import PromptManager
from src.ai.llm_service import LLMService
from src.ai.vision_processor import VisionProcessor
from src.engine.game_master import GameMaster
from src.ui.main_window import MainWindow

def load_config(config_path: str = "config/default_config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def main():
    print("==================================================")
    print("      PHYSICAL AI GAME MASTER - WINDOWS PC        ")
    print("==================================================")

    # 1. Load Configuration
    config = load_config()

    # 2. Instantiate Hardware Suite via HAL Factory
    hw = HardwareFactory.create_hardware_suite(config)
    hw.camera.start_stream()

    # 3. Instantiate AI Services & Prompt Manager
    ai_conf = config.get("ai", {})
    prompt_mgr = PromptManager("config/game_prompts.json")
    llm_service = LLMService(
        provider=ai_conf.get("llm_provider", "mock"),
        api_key=ai_conf.get("api_key", ""),
        model_name=ai_conf.get("model_name", "gemini-2.5-flash")
    )
    vision_processor = VisionProcessor()

    # 4. Instantiate Core Game Master Engine
    gm = GameMaster(
        hw=hw,
        prompt_mgr=prompt_mgr,
        llm_service=llm_service,
        vision_proc=vision_processor
    )

    # 5. Launch PySide6 GUI Dashboard
    app = QApplication(sys.argv)
    window = MainWindow(hw=hw, gm=gm)
    window.show()

    # Run Event Loop
    exit_code = app.exec()

    # Cleanup Hardware Resources on Close
    hw.camera.stop_stream()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
