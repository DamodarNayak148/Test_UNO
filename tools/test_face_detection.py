import sys
import cv2
import numpy as np
import time

def main():
    print("==================================================")
    print("   MULTI-CASCADE FACE DETECTION DIAGNOSTIC TOOL   ")
    print("==================================================")
    print(f"[*] Python Executable: {sys.executable}")
    print(f"[*] OpenCV Version: {getattr(cv2, '__version__', 'unknown')}")
    print(f"[*] Has CascadeClassifier: {hasattr(cv2, 'CascadeClassifier')}")

    if not hasattr(cv2, 'CascadeClassifier'):
        print("\n[CRITICAL ERROR] The installed 'opencv-python' package does not export 'CascadeClassifier'.")
        print("Run: python -m pip install \"opencv-python>=4.8.0,<5.0.0\"")
        return

    # Load multiple built-in OpenCV Haar Cascades
    cascades = {}
    cascade_names = [
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_profileface.xml"
    ]

    for cname in cascade_names:
        cpath = getattr(cv2.data, 'haarcascades', '') + cname
        try:
            clf = cv2.CascadeClassifier(cpath)
            if not clf.empty():
                cascades[cname] = clf
                print(f"[+] Loaded Cascade: {cname}")
            else:
                print(f"[-] Failed to load (empty): {cname}")
        except Exception as e:
            print(f"[-] Error loading {cname}: {e}")

    if not cascades:
        print("[ERROR] No OpenCV Haar Cascade files could be loaded.")
        return

    current_idx = 0
    cascade_keys = list(cascades.keys())

    # Open Webcam (index 0 with CAP_DSHOW)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] Falling back to default VideoCapture(0)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Unable to open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[*] Controls:")
    print("    - Press 'SPACEBAR' or 'C' to switch between Haar Cascades.")
    print("    - Press 'Q' to exit.\n")

    last_print = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        avg_brightness = float(np.mean(gray))

        active_cname = cascade_keys[current_idx]
        active_clf = cascades[active_cname]

        # Detect faces with active cascade using scaleFactor 1.08 / minNeighbors 3
        faces = []
        try:
            faces = active_clf.detectMultiScale(
                gray_eq,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(30, 30)
            )
        except Exception as e:
            print(f"[!] Error in detectMultiScale ({active_cname}): {e}")

        # Draw bounding boxes
        for (x, y, fw, fh) in faces:
            color = (0, 255, 0) if "alt2" in active_cname else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            cv2.putText(frame, f"FACE: {active_cname.split('_')[1]}", (x, max(10, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # On-screen HUD info overlay
        cv2.putText(frame, f"Active Cascade [{current_idx + 1}/{len(cascade_keys)}]: {active_cname}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
        cv2.putText(frame, "Press SPACE to toggle cascade | Press Q to quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if len(faces) > 0 else (0, 0, 255), 2)

        cv2.imshow("Multi-Cascade Face Diagnostic (Press SPACE to switch, Q to quit)", frame)

        now = time.time()
        if now - last_print >= 1.5:
            print(f"[{time.strftime('%H:%M:%S')}] Active: {active_cname} | Faces Detected: {len(faces)}")
            last_print = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32 or key == ord('c') or key == ord('C'): # Spacebar or C
            current_idx = (current_idx + 1) % len(cascade_keys)
            print(f"\n[==>] Switched active cascade to: {cascade_keys[current_idx]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
