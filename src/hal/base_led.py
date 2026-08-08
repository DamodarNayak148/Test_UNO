from abc import ABC, abstractmethod
from typing import Tuple

class BaseLED(ABC):
    """Abstract interface for Game Master RGB LED lighting array (NeoPixels/Simulated)."""

    @abstractmethod
    def set_led_color(self, index: int, r: int, g: int, b: int) -> None:
        """Set a specific LED index to an RGB color (0-255)."""
        pass

    @abstractmethod
    def set_all(self, r: int, g: int, b: int) -> None:
        """Set all LEDs in the array to a uniform RGB color."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Turn off all LEDs."""
        pass

    @abstractmethod
    def set_brightness(self, brightness: float) -> None:
        """Set global LED brightness scaling factor (0.0 to 1.0)."""
        pass

    @abstractmethod
    def pulse(self, r: int, g: int, b: int, speed_hz: float = 1.0) -> None:
        """Set pulsing/breathing animation effect."""
        pass
