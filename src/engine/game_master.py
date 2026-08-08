import time
import threading
from typing import Callable, Optional, Dict, Any
from src.hal.hardware_factory import HardwareSuite
from src.engine.game_state import GameState
from src.engine.prompt_manager import PromptManager
from src.ai.vision_processor import VisionProcessor
from src.ai.llm_service import LLMService

class GameMaster:
    """Core Game Master Engine connecting HAL drivers, AI vision/LLM, and game state logic."""

    def __init__(self, hw: HardwareSuite, prompt_mgr: PromptManager, llm_service: LLMService, vision_proc: VisionProcessor):
        self.hw = hw
        self.prompt_mgr = prompt_mgr
        self.llm = llm_service
        self.vision = vision_proc

        self.state: GameState = GameState.IDLE
        self.active_personality: Dict[str, Any] = self.prompt_mgr.get_personality("DungeonMaster")
        self.current_game_index: int = 0
        self.current_player: str = "Player 1"

        self._on_state_change_cb: Optional[Callable[[GameState, str], None]] = None

        # Register physical/simulated button listeners
        self.hw.buttons.register_callback("player1", lambda: self.handle_player_button("Player 1"))
        self.hw.buttons.register_callback("player2", lambda: self.handle_player_button("Player 2"))
        self.hw.buttons.register_callback("action", self.handle_action_button)
        self.hw.buttons.register_callback("interrupt", self.handle_interrupt_button)

    def set_state_change_callback(self, cb: Callable[[GameState, str], None]) -> None:
        """Register UI state change observer."""
        self._on_state_change_cb = cb
        self._notify_state("System Initialized & Ready.")

    def set_personality(self, personality_key: str) -> None:
        self.active_personality = self.prompt_mgr.get_personality(personality_key)
        print(f"[GameMaster] Personality selected: {self.active_personality.get('name')}")

    def start_game(self) -> None:
        """Begin a new Game Master session."""
        self.state = GameState.INTRO
        greeting = self.active_personality.get("intro_greeting", "Welcome!")
        self._notify_state(f"Intro: {greeting}")

        # Hardware Reactions: Mystic Blue LEDs + Head Center
        self.hw.led.set_all(0, 150, 255)
        self.hw.pan_tilt.center()
        self.hw.pan_tilt.set_expression("happy")

        def _after_intro():
            time.sleep(1.0)
            self.start_challenge()

        self.hw.audio.speak(greeting, on_complete=_after_intro)

    def start_challenge(self) -> None:
        """Present current mini-game challenge."""
        self.state = GameState.PROMPT_CHALLENGE
        game = self.prompt_mgr.get_mini_game(self.current_game_index)
        instructions = f"Challenge: {game.get('title')}! {game.get('instructions')}"
        self._notify_state(instructions)

        # Hardware Reactions: Green waiting lights + Curious Head Tilt
        self.hw.led.set_all(0, 255, 100)
        self.hw.pan_tilt.express_emotion("curious")

        def _ready_for_player():
            self.state = GameState.WAITING_PLAYER_ACTION
            self._notify_state("Waiting for player button press or gesture...")

        self.hw.audio.speak(instructions, on_complete=_ready_for_player)

    def handle_player_button(self, player_id: str) -> None:
        if self.state != GameState.WAITING_PLAYER_ACTION:
            print(f"[GameMaster] Button ignored: state is {self.state.name}")
            return

        self.current_player = player_id
        self.state = GameState.SCANNING_VISION
        self._notify_state(f"{player_id} triggered action! Game Master scanning camera...")

        # Hardware Reactions: Yellow scan LEDs + Look at player
        self.hw.led.set_all(255, 200, 0)
        pan_angle = -25 if player_id == "Player 1" else 25
        self.hw.pan_tilt.set_angles(pan_angle, -10)
        self.hw.audio.play_sfx("scanning")

        # Non-blocking vision evaluation sequence
        threading.Thread(target=self._process_vision_scan, daemon=True).start()

    def handle_action_button(self) -> None:
        if self.state == GameState.IDLE:
            self.start_game()
        elif self.state == GameState.WAITING_PLAYER_ACTION:
            self.handle_player_button("Player 1")

    def handle_interrupt_button(self) -> None:
        print("[GameMaster] Emergency Interrupt Triggered!")
        self.hw.audio.stop_speaking()
        self.state = GameState.IDLE
        self.hw.led.clear()
        self.hw.pan_tilt.center()
        self.hw.pan_tilt.set_expression("neutral")
        self._notify_state("Game interrupted and reset to IDLE.")

    def _process_vision_scan(self) -> None:
        time.sleep(1.2) # Allow camera capture window
        frame = self.hw.camera.get_frame()
        _, telemetry = self.vision.analyze_frame(frame)

        self.state = GameState.EVALUATING
        self._notify_state(f"Evaluating telemetry: {telemetry}")

        # Query LLM for Game Master response
        game = self.prompt_mgr.get_mini_game(self.current_game_index)
        context = f"Game: {game.get('title')}. Player: {self.current_player} performed action."
        ai_commentary = self.llm.generate_gm_response(self.active_personality, context, telemetry)

        # Determine success status
        success = telemetry.get("hands_raised", False) or telemetry.get("has_colorful_item", False) or telemetry.get("face_detected", False)

        self._execute_reaction(ai_commentary, success)

    def _execute_reaction(self, commentary: str, success: bool) -> None:
        self.state = GameState.REACTION
        self._notify_state(f"Reaction: {commentary}")

        if success:
            self.hw.led.set_all(0, 255, 0)
            self.hw.pan_tilt.express_emotion("celebrate")
            self.hw.audio.play_sfx("fanfare")
        else:
            self.hw.led.set_all(255, 0, 0)
            self.hw.pan_tilt.express_emotion("shake")
            self.hw.audio.play_sfx("buzzer")

        def _finish_round():
            time.sleep(1.0)
            # Cycle to next mini-game
            self.current_game_index = (self.current_game_index + 1) % len(self.prompt_mgr.mini_games)
            self.state = GameState.IDLE
            self._notify_state("Round Complete! Press Action/Player Button to play next round.")

        self.hw.audio.speak(commentary, on_complete=_finish_round)

    def _notify_state(self, message: str) -> None:
        print(f"[GameMaster State: {self.state.name}] {message}")
        if self._on_state_change_cb:
            self._on_state_change_cb(self.state, message)
