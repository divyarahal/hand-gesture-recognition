"""
Live webcam inference with the trained CNN.
Place your hand inside the green box. Keys: q quit, i invert image, b toggle blur/threshold view.

NOTE: LeapGestRecog is infrared imagery, so a plain webcam has a domain gap.
Accuracy improves a lot if you fine-tune on your own webcam frames, or use
realtime_mediapipe.py (landmark-based, works out of the box).
"""
import json
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG = 64
model = tf.keras.models.load_model("gesture_model.keras")
classes = json.load(open("classes.json"))
history = deque(maxlen=7)          # majority vote to stop flicker
invert = False

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(w * .55), int(h * .1), int(w * .95), int(h * .7)
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        roi = 255 - roi
    inp = cv2.resize(roi, (IMG, IMG))[None, ..., None].astype("float32")
    p = model.predict(inp, verbose=0)[0]
    history.append((int(p.argmax()), float(p.max())))
    label, _ = Counter(i for i, _ in history).most_common(1)[0]
    conf = np.mean([c for i, c in history if i == label])

    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(frame, f"{classes[label]} {conf:.0%}", (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
    cv2.imshow("Gesture (CNN)", frame)
    cv2.imshow("Model input", cv2.resize(roi, (200, 200)))
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("i"):
        invert = not invert
cap.release(); cv2.destroyAllWindows()
