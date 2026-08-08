import cv2
import numpy as np
import time
from typing import Optional
from src.hal.base_camera import BaseCamera

class SimCamera(BaseCamera):
    """Simulated Camera driver using OpenCV webcam feed with synthetic fallback."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._synthetic_angle = 0.0

    def start_stream(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._running = True
                return True
        except Exception as e:
            print(f"[SimCamera] Could not open physical camera (index {self.device_index}): {e}")
        
        print("[SimCamera] Using synthetic camera generator mode.")
        self._running = True
        return True

    def stop_stream(self) -> None:
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._running:
            return None

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame

        # Fallback: Synthetic animated camera frame
        return self._generate_synthetic_frame()

    def is_opened(self) -> bool:
        return self._running

    def _generate_synthetic_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw dark background grid pattern
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (20, 20, 30), 1)
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (20, 20, 30), 1)

        # Draw camera overlay details
        self._synthetic_angle += 0.05
        cx = int(self.width / 2 + np.sin(self._synthetic_angle) * 80)
        cy = int(self.height / 2 + np.cos(self._synthetic_angle * 0.7) * 30)

        # Simulated player face indicator
        cv2.circle(frame, (cx, cy), 50, (0, 200, 255), 2) # Face circle
        cv2.circle(frame, (cx - 18, cy - 15), 6, (0, 255, 255), -1) # Left eye
        cv2.circle(frame, (cx + 18, cy - 15), 6, (0, 255, 255), -1) # Right eye
        cv2.ellipse(frame, (cx, cy + 15), (20, 10), 0, 0, 180, (0, 255, 255), 2) # Smile

        cv2.putText(frame, "SIMULATED WEBCAM FEED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (20, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame
