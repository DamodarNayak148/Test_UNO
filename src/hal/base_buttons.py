from abc import ABC, abstractmethod
from typing import Callable, Dict

class BaseButtons(ABC):
    """Abstract interface for physical or simulated player button inputs."""

    @abstractmethod
    def register_callback(self, button_name: str, callback: Callable[[], None]) -> None:
        """Register a trigger callback for a specific button event (e.g. 'player1', 'player2', 'action', 'interrupt')."""
        pass

    @abstractmethod
    def trigger_button(self, button_name: str) -> None:
        """Programmatically trigger a button press event."""
        pass
