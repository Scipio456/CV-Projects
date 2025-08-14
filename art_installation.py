import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy import interpolate
import sys
import time

# Initialize MediaPipe Hands with optimized settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Set resolution
width, height = 640, 480
canvas_np = np.zeros((height, width, 3), dtype=np.uint8)

DRAW_COLOR = (255, 255, 255)
thickness = 5
points = deque(maxlen=200)
smooth_points = deque(maxlen=7)  # Reduced for responsiveness
prev_point = None
is_paused = False
last_gesture_time = 0
gesture_debounce = 0.5

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
actual_width = int(cap.get(3))
actual_height = int(cap.get(4))
print(f"Webcam resolution: {actual_width}x{actual_height}")

# Frame rate tracking
last_time = time.time()

# Check if hand is open (five fingers raised)
def is_open_hand(hand_landmarks):
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    count = 0
    for tip, mcp in zip(tips, mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y - 0.02:
            count += 1
    if thumb_tip.x > thumb_mcp.x + 0.02:
        count += 1
    return count >= 5

# Smooth coordinates
def smooth_coordinates(x, y):
    smooth_points.append((x, y))
    if len(smooth_points) < 7:
        return x, y
    avg_x = sum(p[0] for p in smooth_points) / len(smooth_points)
    avg_y = sum(p[1] for p in smooth_points) / len(smooth_points)
    print(f"Raw: ({x}, {y}), Smoothed: ({int(avg_x)}, {int(avg_y)})")
    return int(avg_x), int(avg_y)

# Draw spline curve
def draw_spline(canvas_np, points, color, thickness):
    if len(points) < 4:
        cv2.polylines(canvas_np, [np.array(list(points), dtype=np.int32)], False, color, thickness)
        return
    x, y = zip(*list(points)[-10:])
    t = np.linspace(0, 1, len(x))
    try:
        fx = interpolate.interp1d(t, x, kind='cubic')
        fy = interpolate.interp1d(t, y, kind='cubic')
        t_new = np.linspace(0, 1, 20)  # Reduced for speed
        x_new = fx(t_new).astype(np.int32)
        y_new = fy(t_new).astype(np.int32)
        spline_points = np.vstack((x_new, y_new)).T
        cv2.polylines(canvas_np, [spline_points], False, color, thickness)
    except:
        cv2.polylines(canvas_np, [np.array(list(points)[-10:], dtype=np.int32)], False, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Flip and resize frame
    frame = cv2.flip(frame, 1)
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height))
        print(f"Resized frame to: {width}x{height}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Calculate frame rate
    current_time = time.time()
    fps = 1 / (current_time - last_time) if current_time > last_time else 0
    last_time = current_time
    print(f"FPS: {fps:.1f}")

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            index_tip = hand_landmarks.landmark[8]
            cx = int(index_tip.x * width)
            cy = int(index_tip.y * height)
            cx, cy = smooth_coordinates(cx, cy)
            print(f"Index finger position ({hand_label}): ({cx}, {cy})")

            if hand_label == 'Right':
                if not is_paused:
                    points.append((cx, cy))
                    if len(points) >= 2:
                        draw_spline(canvas_np, points, DRAW_COLOR, thickness)
                    prev_point = (cx, cy)
                else:
                    points.clear()
                    prev_point = None
                    print("Drawing paused")

            elif hand_label == 'Left':
                if is_open_hand(hand_landmarks) and (current_time - last_gesture_time) > gesture_debounce:
                    canvas_np.fill(0)
                    points.clear()
                    prev_point = None
                    last_gesture_time = current_time
                    print("Canvas cleared")

            # Draw landmarks and green dot on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    else:
        points.clear()
        prev_point = None
        print("No hand detected")

    # Display pause status on canvas
    status_text = "PAUSED" if is_paused else "DRAWING"
    cv2.putText(canvas_np, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_paused else (0, 255, 0), 2)

    # Overlay canvas on frame
    output = cv2.addWeighted(frame, 0.5, canvas_np, 0.5, 0)

    cv2.imshow("Index Finger Drawing", output)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('p') or key == ord(' '):  # Space or 'p'
        is_paused = not is_paused
        print(f"Pause toggled: {'Paused' if is_paused else 'Resumed'}")

cap.release()
cv2.destroyAllWindows()