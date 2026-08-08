import time
import threading
from typing import Tuple, Callable, Optional
from src.hal.base_pantilt import BasePanTilt

class SimPanTilt(BasePanTilt):
    """Simulated Pan-Tilt driver managing head angles and facial expression states."""

    def __init__(self, pan_range: Tuple[float, float] = (-90, 90), tilt_range: Tuple[float, float] = (-45, 45)):
        self.pan_min, self.pan_max = pan_range
        self.tilt_min, self.tilt_max = tilt_range
        self._pan: float = 0.0
        self._tilt: float = 0.0
        self._expression: str = "neutral"
        self._on_update_callback: Optional[Callable[[float, float, str], None]] = None

    def set_update_callback(self, callback: Callable[[float, float, str], None]) -> None:
        """Register UI update callback (pan, tilt, expression)."""
        self._on_update_callback = callback
        self._notify_update()

    def set_angles(self, pan: float, tilt: float) -> None:
        self._pan = max(self.pan_min, min(self.pan_max, pan))
        self._tilt = max(self.tilt_min, min(self.tilt_max, tilt))
        self._notify_update()

    def get_angles(self) -> Tuple[float, float]:
        return (self._pan, self._tilt)

    def center(self) -> None:
        self.set_angles(0.0, 0.0)

    def set_expression(self, expression: str) -> None:
        self._expression = expression
        self._notify_update()

    def express_emotion(self, emotion: str) -> None:
        """Execute a simulated animated gesture in a non-blocking background thread."""
        threading.Thread(target=self._run_emotion_routine, args=(emotion,), daemon=True).start()

    def _run_emotion_routine(self, emotion: str) -> None:
        if emotion == "nod":
            self.set_expression("happy")
            for t in [15, -10, 15, 0]:
                self.set_angles(self._pan, t)
                time.sleep(0.15)
        elif emotion == "shake":
            self.set_expression("thinking")
            for p in [30, -30, 20, -20, 0]:
                self.set_angles(p, self._tilt)
                time.sleep(0.15)
        elif emotion == "curious":
            self.set_expression("thinking")
            self.set_angles(25, -15)
        elif emotion == "celebrate":
            self.set_expression("surprised")
            for i in range(3):
                self.set_angles(-30, 20)
                time.sleep(0.12)
                self.set_angles(30, -10)
                time.sleep(0.12)
            self.center()
            self.set_expression("happy")

    def _notify_update(self) -> None:
        if self._on_update_callback:
            self._on_update_callback(self._pan, self._tilt, self._expression)
