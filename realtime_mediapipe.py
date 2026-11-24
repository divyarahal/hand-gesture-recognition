"""
Landmark-based gesture recognition + gesture-to-action mapping (no training needed).
Uses the MediaPipe Tasks HandLandmarker (works with current mediapipe versions).
The first run downloads a small model file (hand_landmarker.task, ~7 MB).

Gestures: FIST, ONE, PEACE, THREE, FOUR, PALM, THUMBS_UP
Edit ACTIONS to wire gestures to anything (media keys, slides, volume...).
Optional: pip install pyautogui to send real key presses.

Run:     python realtime_mediapipe.py
Record:  python realtime_mediapipe.py --record demo.mp4
Press q to quit. Show your palm / hand FRONT to the camera, fingers pointing up.
"""
import argparse
import os
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import pyautogui
except ImportError:
    pyautogui = None

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

ACTIONS = {                      # gesture -> function
    "PALM":      lambda: print("ACTION: pause/play") or (pyautogui and pyautogui.press("space")),
    "FIST":      lambda: print("ACTION: stop"),
    "ONE":       lambda: print("ACTION: next")       or (pyautogui and pyautogui.press("right")),
    "PEACE":     lambda: print("ACTION: previous")   or (pyautogui and pyautogui.press("left")),
    "THUMBS_UP": lambda: print("ACTION: volume up")  or (pyautogui and pyautogui.press("volumeup")),
}
COOLDOWN = 1.0                   # seconds between repeated actions

TIPS, PIPS = [8, 12, 16, 20], [6, 10, 14, 18]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def finger_states(lm, handed):
    """[thumb, index, middle, ring, pinky] -> 1 if extended (hand upright)."""
    up = [int(lm[t].y < lm[p].y) for t, p in zip(TIPS, PIPS)]
    # thumb: compare x of tip vs IP joint; direction depends on handedness
    thumb = int(lm[4].x < lm[3].x) if handed == "Right" else int(lm[4].x > lm[3].x)
    return [thumb] + up


def classify(f, lm):
    thumb, idx, mid, ring, pinky = f
    if f == [0, 0, 0, 0, 0]:
        return "FIST"
    if f == [1, 1, 1, 1, 1]:
        return "PALM"
    if idx and not (mid or ring or pinky):
        return "ONE"
    if idx and mid and not (ring or pinky):
        return "PEACE"
    if idx and mid and ring and not pinky:
        return "THREE"
    if idx and mid and ring and pinky and not thumb:
        return "FOUR"
    if thumb and not (idx or mid or ring or pinky) and lm[4].y < lm[3].y < lm[2].y:
        return "THUMBS_UP"
    return "UNKNOWN"


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 255, 0), -1)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one time)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise SystemExit(
                f"Could not download the model ({e}).\nDownload it manually from:\n"
                f"{MODEL_URL}\nand save it next to this script as {MODEL_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", help="save the annotated feed to this .mp4 file")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = ap.parse_args()

    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Close Zoom/Teams or try --camera 1")
    writer, buf, last_fire, t0 = None, deque(maxlen=8), 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                     # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect_for_video(mp_img, int((time.time() - t0) * 1000))

        label = "NO HAND"
        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            handed = res.handedness[0][0].category_name
            label = classify(finger_states(lm, handed), lm)
            draw_hand(frame, lm)
        buf.append(label)
        stable, n = Counter(buf).most_common(1)[0]
        if n >= 6 and stable in ACTIONS and time.time() - last_fire > COOLDOWN:
            ACTIONS[stable]()
            last_fire = time.time()

        cv2.putText(frame, stable, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        if args.record:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        cv2.imshow("Gesture control (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
