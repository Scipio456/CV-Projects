"""
Hand Tracking Drawing Board (Educational Project)

DISCLAIMER:
This project uses a webcam ONLY for real-time processing.
No images, videos, or biometric data are stored, saved, or transmitted.

The application runs entirely on the local machine and is intended
for educational and learning purposes only.

"""

import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.interpolate import CubicSpline

# ================= CONFIG =================
DRAW_THICKNESS = 4
SPLINE_POINTS = 50
# =========================================

# Colors (BGR format for OpenCV)
COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "ERASER": (0, 0, 0)
}
current_color = COLORS["BLUE"]

# ================= MEDIAPIPE HANDS =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= CAMERA =================
# Uses default system camera.
# No frames are saved or transmitted.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

# ================= CANVAS =================
canvas = None

# ================= STROKES =================
strokes = []
redo_stack = []
current_stroke = []
paused = False

# ================= KALMAN FILTER =================
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)

kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)

kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# ================= HELPER FUNCTIONS =================
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(hand[tips[0]].x < hand[tips[0] - 1].x)  # Thumb
    for i in range(1, 5):
        fingers.append(hand[tips[i]].y < hand[tips[i] - 2].y)
    return fingers

def is_pinch(hand):
    return np.hypot(hand[4].x - hand[8].x,
                    hand[4].y - hand[8].y) < 0.03

def kalman_smooth(x, y):
    kf.predict()
    measurement = np.array([[np.float32(x)],
                            [np.float32(y)]])
    estimate = kf.correct(measurement)
    return int(estimate[0]), int(estimate[1])

def draw_spline(img, pts, color):
    if len(pts) < 4:
        return
    pts = np.array(pts)
    t = np.arange(len(pts))
    csx = CubicSpline(t, pts[:, 0])
    csy = CubicSpline(t, pts[:, 1])
    t_new = np.linspace(0, len(pts) - 1, SPLINE_POINTS)

    for i in range(len(t_new) - 1):
        p1 = (int(csx(t_new[i])), int(csy(t_new[i])))
        p2 = (int(csx(t_new[i + 1])), int(csy(t_new[i + 1])))
        cv2.line(img, p1, p2, color, DRAW_THICKNESS)

# ================= FPS =================
prev_time = time.time()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = result.multi_handedness[i].classification[0].label
            lm = hand_landmarks.landmark
            fingers = fingers_up(lm)

            # Right hand: Drawing
            if label == "Right":
                if fingers[1] and not paused:
                    x, y = int(lm[8].x * w), int(lm[8].y * h)
                    x, y = kalman_smooth(x, y)
                    current_stroke.append((x, y))
                elif current_stroke:
                    strokes.append((current_stroke.copy(), current_color))
                    current_stroke.clear()

            # Left hand: Controls
            if label == "Left":
                paused = is_pinch(lm)

                if all(fingers):
                    strokes.clear()
                    redo_stack.clear()
                    canvas[:] = 0

                if fingers == [1, 0, 0, 0, 0]:
                    current_color = COLORS["RED"]
                elif fingers == [0, 1, 1, 0, 0]:
                    current_color = COLORS["GREEN"]
                elif fingers == [0, 1, 0, 0, 1]:
                    current_color = COLORS["BLUE"]

                if fingers == [0, 0, 0, 0, 0] and strokes:
                    redo_stack.append(strokes.pop())

                if fingers == [1, 1, 0, 0, 0] and redo_stack:
                    strokes.append(redo_stack.pop())

    canvas[:] = 0
    for stroke, color in strokes:
        draw_spline(canvas, stroke, color)
    if current_stroke:
        draw_spline(canvas, current_stroke, current_color)

    fps = int(1 / (time.time() - prev_time))
    prev_time = time.time()

    output = cv2.addWeighted(frame, 0.4, canvas, 0.6, 0)
    cv2.putText(output, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(output,
                "Right: Draw | Left: Controls | ESC to Exit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Tracking Drawing Board", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
