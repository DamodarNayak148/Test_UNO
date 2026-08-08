from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseCamera(ABC):
    """Abstract interface for Game Master Camera input."""

    @abstractmethod
    def start_stream(self) -> bool:
        """Initialize and start the camera stream."""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop the camera stream and release resources."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture and return the current camera frame (BGR image format)."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera stream is active and readable."""
        pass
