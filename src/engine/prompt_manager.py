import json
import os
from typing import Dict, Any, List

class PromptManager:
    """Manages Game Master personalities and mini-game rule prompts."""

    def __init__(self, config_path: str = "config/game_prompts.json"):
        self.config_path = config_path
        self.personalities: Dict[str, Any] = {}
        self.mini_games: List[Dict[str, Any]] = []
        self.load_config()

    def load_config(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.personalities = data.get("personalities", {})
                    self.mini_games = data.get("mini_games", [])
            except Exception as e:
                print(f"[PromptManager] Failed to load prompt config: {e}")

    def get_personality(self, name: str) -> Dict[str, Any]:
        return self.personalities.get(name, {
            "name": "Game Master",
            "style": "Balanced",
            "intro_greeting": "Welcome to the game!",
            "success_phrases": ["Success!"],
            "failure_phrases": ["Failure!"]
        })

    def get_mini_game(self, index: int = 0) -> Dict[str, Any]:
        if 0 <= index < len(self.mini_games):
            return self.mini_games[index]
        return {
            "id": "default",
            "title": "General Challenge",
            "description": "Perform your best move!",
            "instructions": "Press your button and pose!"
        }
