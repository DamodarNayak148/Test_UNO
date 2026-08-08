from typing import List, Tuple, Callable, Optional
from src.hal.base_led import BaseLED

class SimLED(BaseLED):
    """Simulated LED driver managing an array of RGB colors with UI callbacks."""

    def __init__(self, count: int = 12, brightness: float = 0.8):
        self.count = count
        self.brightness = brightness
        self.led_colors: List[Tuple[int, int, int]] = [(0, 0, 0)] * count
        self._on_update_callback: Optional[Callable[[List[Tuple[int, int, int]]], None]] = None

    def set_update_callback(self, callback: Callable[[List[Tuple[int, int, int]]], None]) -> None:
        """Set callback function to notify UI when LED states change."""
        self._on_update_callback = callback
        self._notify_update()

    def set_led_color(self, index: int, r: int, g: int, b: int) -> None:
        if 0 <= index < self.count:
            self.led_colors[index] = (r, g, b)
            self._notify_update()

    def set_all(self, r: int, g: int, b: int) -> None:
        self.led_colors = [(r, g, b)] * self.count
        self._notify_update()

    def clear(self) -> None:
        self.set_all(0, 0, 0)

    def set_brightness(self, brightness: float) -> None:
        self.brightness = max(0.0, min(1.0, brightness))
        self._notify_update()

    def pulse(self, r: int, g: int, b: int, speed_hz: float = 1.0) -> None:
        # Set base colors; UI animation handles breathing oscillation
        self.set_all(r, g, b)

    def get_colors(self) -> List[Tuple[int, int, int]]:
        """Return current scaled RGB colors for display rendering."""
        return [
            (
                int(r * self.brightness),
                int(g * self.brightness),
                int(b * self.brightness),
            )
            for r, g, b in self.led_colors
        ]

    def _notify_update(self) -> None:
        if self._on_update_callback:
            self._on_update_callback(self.get_colors())
