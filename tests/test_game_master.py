import unittest
import numpy as np
from src.hal.hardware_factory import HardwareFactory, HardwareSuite
from src.engine.prompt_manager import PromptManager
from src.ai.llm_service import LLMService
from src.ai.vision_processor import VisionProcessor
from src.engine.game_master import GameMaster
from src.engine.game_state import GameState

class TestPhysicalAIGameMaster(unittest.TestCase):

    def setUp(self):
        config = {
            "driver_mode": "simulated",
            "hardware": {
                "camera": {"device_index": 0},
                "leds": {"count": 12},
                "pan_tilt": {"pan_min": -90, "pan_max": 90}
            },
            "ai": {"llm_provider": "mock"}
        }
        self.hw = HardwareFactory.create_hardware_suite(config)
        self.prompt_mgr = PromptManager("config/game_prompts.json")
        self.llm = LLMService(provider="mock")
        self.vision = VisionProcessor()
        self.gm = GameMaster(self.hw, self.prompt_mgr, self.llm, self.vision)

    def test_hal_simulated_drivers(self):
        self.hw.led.set_all(255, 0, 100)
        colors = self.hw.led.get_colors()
        self.assertEqual(len(colors), 12)
        
        self.hw.pan_tilt.set_angles(30.0, -10.0)
        pan, tilt = self.hw.pan_tilt.get_angles()
        self.assertEqual(pan, 30.0)
        self.assertEqual(tilt, -10.0)

    def test_vision_processor(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, telemetry = self.vision.analyze_frame(dummy_frame)
        self.assertIn("face_detected", telemetry)
        self.assertIn("has_colorful_item", telemetry)

    def test_game_master_state_transitions(self):
        self.assertEqual(self.gm.state, GameState.IDLE)
        self.gm.start_game()
        self.assertEqual(self.gm.state, GameState.INTRO)

        self.gm.handle_interrupt_button()
        self.assertEqual(self.gm.state, GameState.IDLE)

if __name__ == "__main__":
    unittest.main()
