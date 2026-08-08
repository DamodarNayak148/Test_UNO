from abc import ABC, abstractmethod

class BasePanTilt(ABC):
    """Abstract interface for Game Master Robot Head Servo Pan-Tilt mechanism."""

    @abstractmethod
    def set_angles(self, pan: float, tilt: float) -> None:
        """Set absolute servo angles in degrees (Pan: -90 to +90, Tilt: -45 to +45)."""
        pass

    @abstractmethod
    def get_angles(self) -> tuple[float, float]:
        """Return current target (pan, tilt) angles."""
        pass

    @abstractmethod
    def center(self) -> None:
        """Reset head to center position (0, 0)."""
        pass

    @abstractmethod
    def express_emotion(self, emotion: str) -> None:
        """Perform a movement gesture pattern based on emotion (nod, shake, curious_tilt, celebrate)."""
        pass

    @abstractmethod
    def set_expression(self, expression: str) -> None:
        """Set facial expression state (neutral, happy, surprised, thinking, angry, mysterious)."""
        pass
