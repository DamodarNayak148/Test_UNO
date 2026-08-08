from typing import Dict, List, Callable
from src.hal.base_buttons import BaseButtons

class SimButtons(BaseButtons):
    """Simulated Player Buttons driver for keyboard & GUI triggers."""

    def __init__(self):
        self._callbacks: Dict[str, List[Callable[[], None]]] = {
            "player1": [],
            "player2": [],
            "action": [],
            "interrupt": []
        }

    def register_callback(self, button_name: str, callback: Callable[[], None]) -> None:
        button_key = button_name.lower()
        if button_key not in self._callbacks:
            self._callbacks[button_key] = []
        self._callbacks[button_key].append(callback)

    def trigger_button(self, button_name: str) -> None:
        button_key = button_name.lower()
        print(f"[SimButtons Triggered]: '{button_key}'")
        if button_key in self._callbacks:
            for cb in self._callbacks[button_key]:
                try:
                    cb()
                except Exception as e:
                    print(f"[SimButtons Callback Error]: {e}")
