import os
import time
import cv2
import numpy as np
import urllib.request
from typing import Tuple, Dict, Any

class VisionProcessor:
    """Computer Vision analysis engine using OpenCV & MediaPipe Pose."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    MODEL_PATH = "models/pose_landmarker_lite.task"

    def __init__(self):
        # 1. Initialize Haar Cascade for Face Detection
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier'):
                cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
                classifier = cv2.CascadeClassifier(cascade_path)
                if not classifier.empty():
                    self.face_cascade = classifier
        except Exception as e:
            print(f"[VisionProcessor] CascadeClassifier init warning: {e}")

        # 2. Initialize MediaPipe Pose Engine (Supports Tasks API & Legacy Solutions)
        self.mp = None
        self.landmarker = None
        self.drawing_utils = None
        self.pose_connections = None
        self._init_mediapipe_pose()

    def _init_mediapipe_pose(self) -> None:
        try:
            import mediapipe as mp
            self.mp = mp

            # Try MediaPipe Tasks API
            if hasattr(mp, 'tasks'):
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core import base_options

                # Ensure model file exists locally
                if not os.path.exists(self.MODEL_PATH):
                    os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
                    print(f"[VisionProcessor] Downloading MediaPipe pose model to {self.MODEL_PATH}...")
                    urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)

                options = vision.PoseLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=self.MODEL_PATH),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5
                )
                self.landmarker = vision.PoseLandmarker.create_from_options(options)
                self.drawing_utils = vision.drawing_utils
                self.pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
                print("[VisionProcessor] MediaPipe Tasks PoseLandmarker initialized successfully.")

            # Legacy Solutions Fallback
            elif hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                self.mp_pose_solution = mp.solutions.pose
                self.landmarker = self.mp_pose_solution.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.5
                )
                self.drawing_utils = mp.solutions.drawing_utils
                self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
                print("[VisionProcessor] MediaPipe Legacy Pose initialized successfully.")

        except Exception as e:
            print(f"[VisionProcessor] MediaPipe Pose init warning: {e}")
            self.landmarker = None

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process camera frame, draw overlays, and extract vision telemetry."""
        if frame is None:
            return frame, {
                "face_detected": False,
                "face_count": 0,
                "face_center": (0, 0),
                "body_pose_detected": False,
                "left_hand_raised": False,
                "right_hand_raised": False,
                "hands_raised": False,
                "avg_brightness": 0.0,
                "has_colorful_item": False,
            }

        annotated = frame.copy()
        hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
        avg_brightness = float(np.mean(hsv[:, :, 2]))

        # 1. Dark / Covered-Camera Guard (Camera covered or pitch dark)
        if avg_brightness < 35.0:
            status_text = f"CAMERA COVERED/DARK (V: {avg_brightness:.1f})"
            cv2.putText(annotated, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return annotated, {
                "face_detected": False,
                "face_count": 0,
                "face_center": (0, 0),
                "body_pose_detected": False,
                "left_hand_raised": False,
                "right_hand_raised": False,
                "hands_raised": False,
                "avg_brightness": avg_brightness,
                "has_colorful_item": False,
            }

        # 2. Haar Face Detection
        gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        faces = []
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray_eq, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
            except Exception:
                faces = []

        face_detected = len(faces) > 0
        cx, cy = 0, 0
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "PLAYER DETECTED", (x, max(10, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2

        # 3. Brightness & Colorful Object Detection (High S and High V)
        color_mask = cv2.inRange(hsv, (0, 90, 80), (180, 255, 255))
        colorful_pixels = int(np.count_nonzero(color_mask))
        has_colorful_item = colorful_pixels > (frame.shape[0] * frame.shape[1] * 0.05)

        # 4. MediaPipe Pose Landmark Detection & Gesture Analysis
        body_pose_detected = False
        left_hand_raised = False
        right_hand_raised = False
        hands_raised = False

        if self.landmarker is not None and self.mp is not None:
            try:
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # Tasks API path
                if hasattr(self.landmarker, 'detect'):
                    mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
                    result = self.landmarker.detect(mp_img)
                    if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                        landmarks = result.pose_landmarks[0]

                        l_shoulder = landmarks[11]
                        r_shoulder = landmarks[12]
                        l_wrist = landmarks[15]
                        r_wrist = landmarks[16]
                        l_ear = landmarks[7]
                        r_ear = landmarks[8]

                        l_wrist_valid = getattr(l_wrist, 'visibility', 1.0) > 0.4
                        r_wrist_valid = getattr(r_wrist, 'visibility', 1.0) > 0.4
                        l_shoulder_valid = getattr(l_shoulder, 'visibility', 1.0) > 0.4
                        r_shoulder_valid = getattr(r_shoulder, 'visibility', 1.0) > 0.4

                        if (l_shoulder_valid or r_shoulder_valid) and (l_wrist_valid or r_wrist_valid):
                            body_pose_detected = True

                            if l_wrist_valid and l_shoulder_valid:
                                left_hand_raised = (l_wrist.y < l_shoulder.y - 0.05) or (l_wrist.y < l_ear.y)

                            if r_wrist_valid and r_shoulder_valid:
                                right_hand_raised = (r_wrist.y < r_shoulder.y - 0.05) or (r_wrist.y < r_ear.y)

                            hands_raised = left_hand_raised and right_hand_raised

                        # Draw pose landmarks overlay
                        if self.drawing_utils and self.pose_connections:
                            self.drawing_utils.draw_landmarks(annotated, landmarks, self.pose_connections)

                # Legacy solution path
                elif hasattr(self.landmarker, 'process'):
                    results = self.landmarker.process(rgb_frame)
                    if results and results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        l_shoulder, r_shoulder = landmarks[11], landmarks[12]
                        l_wrist, r_wrist = landmarks[15], landmarks[16]
                        l_ear, r_ear = landmarks[7], landmarks[8]

                        l_wrist_valid = l_wrist.visibility > 0.4
                        r_wrist_valid = r_wrist.visibility > 0.4

                        if l_wrist_valid or r_wrist_valid:
                            body_pose_detected = True
                            if l_wrist_valid:
                                left_hand_raised = (l_wrist.y < l_shoulder.y - 0.05) or (l_wrist.y < l_ear.y)
                            if r_wrist_valid:
                                right_hand_raised = (r_wrist.y < r_shoulder.y - 0.05) or (r_wrist.y < r_ear.y)

                            hands_raised = left_hand_raised and right_hand_raised

                        if self.drawing_utils and self.pose_connections:
                            self.drawing_utils.draw_landmarks(annotated, results.pose_landmarks, self.pose_connections)

            except Exception as e:
                print(f"[VisionProcessor] Pose detection exception: {e}")

        telemetry = {
            "face_detected": face_detected,
            "face_count": len(faces),
            "face_center": (cx, cy),
            "body_pose_detected": body_pose_detected,
            "left_hand_raised": left_hand_raised,
            "right_hand_raised": right_hand_raised,
            "hands_raised": hands_raised,
            "avg_brightness": avg_brightness,
            "has_colorful_item": has_colorful_item,
        }

        # Status Overlay on Preview Frame
        status_text = (
            f"Faces: {len(faces)} | Pose: {'YES' if body_pose_detected else 'NO'} | "
            f"L-Hand: {'UP' if left_hand_raised else 'DOWN'} | R-Hand: {'UP' if right_hand_raised else 'DOWN'} | "
            f"Both: {'YES' if hands_raised else 'NO'}"
        )
        cv2.putText(annotated, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        return annotated, telemetry
