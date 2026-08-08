import cv2
import numpy as np
from typing import Tuple, Dict, Any

class VisionProcessor:
    """Computer Vision analysis engine using OpenCV."""

    def __init__(self):
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier'):
                cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
                classifier = cv2.CascadeClassifier(cascade_path)
                if not classifier.empty():
                    self.face_cascade = classifier
        except Exception as e:
            print(f"[VisionProcessor] CascadeClassifier init warning: {e}")

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process camera frame, draw overlays, and extract vision telemetry."""
        if frame is None:
            return frame, {"face_detected": False, "hands_raised": False, "avg_brightness": 0}

        annotated = frame.copy()
        gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
        
        # 1. Face Detection
        faces = []
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            except Exception:
                faces = []

        face_detected = len(faces) > 0

        cx, cy = 0, 0
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "PLAYER DETECTED", (x, max(10, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2

        # 2. Brightness & Color Analysis (for Mystery Item Quiz)
        hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
        avg_brightness = float(np.mean(hsv[:, :, 2]))

        # High saturation mask for colorful objects
        color_mask = cv2.inRange(hsv, (0, 70, 50), (180, 255, 255))
        colorful_pixels = int(np.count_nonzero(color_mask))
        has_colorful_item = colorful_pixels > (frame.shape[0] * frame.shape[1] * 0.05)

        # 3. Simple motion/pose gesture heuristic (Hands Raised Detection)
        # Inspect top 35% region for motion or bright objects
        top_region = hsv[:int(frame.shape[0] * 0.35), :]
        top_saturation = np.mean(top_region[:, :, 1])
        hands_raised = top_saturation > 45 or (face_detected and cy > int(frame.shape[0] * 0.5))

        telemetry = {
            "face_detected": face_detected,
            "face_count": len(faces),
            "face_center": (cx, cy),
            "hands_raised": hands_raised,
            "avg_brightness": avg_brightness,
            "has_colorful_item": has_colorful_item,
        }

        # Status overlay on video preview
        status_text = f"Faces: {len(faces)} | Colorful Item: {'YES' if has_colorful_item else 'NO'}"
        cv2.putText(annotated, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated, telemetry
