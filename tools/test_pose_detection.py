import os
import sys
import time
import cv2
import numpy as np

def main():
    print("==================================================")
    print("   STANDALONE MEDIAPIPE POSE DIAGNOSTIC SCRIPT    ")
    print("==================================================")
    print(f"[*] Python Executable: {sys.executable}")

    # Check MediaPipe availability
    try:
        import mediapipe as mp
    except ImportError:
        print("\n[CRITICAL ERROR] 'mediapipe' module was not found in this Python environment.")
        print("Please run the diagnostic script using Python 3.14:")
        print(r"C:\Users\chay2\AppData\Local\Programs\Python\Python314\python.exe tools/test_pose_detection.py")
        return

    model_path = "models/pose_landmarker_lite.task"
    if not os.path.exists(model_path):
        print(f"[ERROR] Pose model file not found at '{model_path}'.")
        return

    # Initialize MediaPipe PoseLandmarker Tasks API
    landmarker = None
    drawing_utils = None
    pose_connections = None
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        options = vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        drawing_utils = vision.drawing_utils
        pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
        print(f"[*] MediaPipe PoseLandmarker loaded successfully using '{model_path}'")
    except Exception as e:
        print(f"[ERROR] Could not initialize MediaPipe PoseLandmarker: {e}")
        return

    # Open Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Unable to open webcam at index 0.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[*] Webcam opened (640x480). Press 'Q' to exit.\n")

    last_print = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect Pose
        result = landmarker.detect(mp_img)

        pose_detected = False
        l_wrist_detected = False
        r_wrist_detected = False
        left_hand_raised = False
        right_hand_raised = False
        both_hands_raised = False

        l_wrist_y, r_wrist_y = 0.0, 0.0
        l_shoulder_y, r_shoulder_y = 0.0, 0.0
        l_ear_y, r_ear_y = 0.0, 0.0

        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            pose_detected = True

            # Draw complete pose skeleton
            if drawing_utils and pose_connections:
                drawing_utils.draw_landmarks(frame, landmarks, pose_connections)

            # Extract specific landmarks (MediaPipe landmark indices: 0=Nose, 7=L_Ear, 8=R_Ear, 11=L_Shoulder, 12=R_Shoulder, 15=L_Wrist, 16=R_Wrist)
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]
            l_ear = landmarks[7]
            r_ear = landmarks[8]

            l_shoulder_y = l_shoulder.y
            r_shoulder_y = r_shoulder.y
            l_wrist_y = l_wrist.y
            r_wrist_y = r_wrist.y
            l_ear_y = l_ear.y
            r_ear_y = r_ear.y

            l_wrist_valid = getattr(l_wrist, 'visibility', 1.0) > 0.4
            r_wrist_valid = getattr(r_wrist, 'visibility', 1.0) > 0.4
            l_shoulder_valid = getattr(l_shoulder, 'visibility', 1.0) > 0.4
            r_shoulder_valid = getattr(r_shoulder, 'visibility', 1.0) > 0.4

            l_wrist_detected = l_wrist_valid
            r_wrist_detected = r_wrist_valid

            # Hand-raised mathematical rule calculation
            # MediaPipe Y coords: 0.0 is top of image, 1.0 is bottom of image.
            # Raised if wrist Y is smaller than shoulder Y - 0.05 OR smaller than ear Y
            if l_wrist_valid and l_shoulder_valid:
                left_hand_raised = (l_wrist.y < l_shoulder.y - 0.05) or (l_wrist.y < l_ear.y)

            if r_wrist_valid and r_shoulder_valid:
                right_hand_raised = (r_wrist.y < r_shoulder.y - 0.05) or (r_wrist.y < r_ear.y)

            both_hands_raised = left_hand_raised and right_hand_raised

            # Highlight key landmarks with prominent circles & labels
            for lm, name, color in [
                (l_shoulder, "L_Shoulder", (255, 0, 0)),
                (r_shoulder, "R_Shoulder", (255, 0, 0)),
                (l_wrist, "L_Wrist", (0, 255, 0) if left_hand_raised else (0, 0, 255)),
                (r_wrist, "R_Wrist", (0, 255, 0) if right_hand_raised else (0, 0, 255)),
                (l_ear, "L_Ear", (255, 255, 0)),
                (r_ear, "R_Ear", (255, 255, 0)),
            ]:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 8, color, -1)
                cv2.putText(frame, name, (px + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw Shoulder threshold line across image (Yellow dashed/solid line)
            avg_shoulder_pixel_y = int(((l_shoulder_y + r_shoulder_y) / 2.0) * h)
            cv2.line(frame, (0, avg_shoulder_pixel_y), (w, avg_shoulder_pixel_y), (0, 255, 255), 1)
            cv2.putText(frame, "SHOULDER THRESHOLD LEVEL", (10, max(15, avg_shoulder_pixel_y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # On-screen HUD info overlay
        cv2.putText(frame, f"Pose Detected: {'YES' if pose_detected else 'NO'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if pose_detected else (0, 0, 255), 2)
        cv2.putText(frame, f"Left Wrist Detected: {'YES' if l_wrist_detected else 'NO'} | Y: {l_wrist_y:.3f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Right Wrist Detected: {'YES' if r_wrist_detected else 'NO'} | Y: {r_wrist_y:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Left Shoulder Y: {l_shoulder_y:.3f} | Right Shoulder Y: {r_shoulder_y:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        status_text = f"L_Hand Raised: {left_hand_raised} | R_Hand Raised: {right_hand_raised} | BOTH: {both_hands_raised}"
        cv2.putText(frame, status_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if both_hands_raised else (0, 165, 255), 2)

        cv2.imshow("Standalone Pose Diagnostic (Press Q to Quit)", frame)

        now = time.time()
        if now - last_print >= 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] Pose: {'YES' if pose_detected else 'NO'} | L_Wrist_Y: {l_wrist_y:.3f} | R_Wrist_Y: {r_wrist_y:.3f} | L_Shoulder_Y: {l_shoulder_y:.3f} | R_Shoulder_Y: {r_shoulder_y:.3f} | L_Raised: {left_hand_raised} | R_Raised: {right_hand_raised} | BOTH: {both_hands_raised}")
            last_print = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[*] Diagnostic script terminated.")

if __name__ == "__main__":
    main()
