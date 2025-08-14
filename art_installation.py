import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
import time

# Initialize MediaPipe Hands with optimized settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,  # Lighter model for faster processing
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5  # Balanced tracking confidence
)
mp_drawing = mp.solutions.drawing_utils

# Set resolution for normal window
width, height = 640, 480
canvas_np = np.zeros((height, width, 3), dtype=np.uint8)  # OpenCV-compatible canvas

DRAW_COLOR = (255, 255, 255)
thickness = 5
smooth_points = deque(maxlen=3)  # Small smoothing window for low lag
prev_point = None
is_paused = False
last_gesture_time = 0
gesture_debounce = 0.5  # Debounce time in seconds for clear gesture

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Check if hand is open (five fingers raised)
def is_open_hand(hand_landmarks):
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    mcps = [5, 9, 13, 17]  # Corresponding MCP joints
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    count = 0
    for tip, mcp in zip(tips, mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y - 0.02:
            count += 1
    if thumb_tip.x > thumb_mcp.x + 0.02:  # Thumb extended
        count += 1
    return count >= 5

# Smooth coordinates
def smooth_coordinates(x, y):
    smooth_points.append((x, y))
    if len(smooth_points) < 3:
        return x, y
    avg_x = sum(p[0] for p in smooth_points) / len(smooth_points)
    avg_y = sum(p[1] for p in smooth_points) / len(smooth_points)
    return int(avg_x), int(avg_y)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Flip and resize frame
    frame = cv2.flip(frame, 1)
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            # Get index finger tip coordinates
            index_tip = hand_landmarks.landmark[8]
            cx = int(index_tip.x * width)
            cy = int(index_tip.y * height)
            cx, cy = smooth_coordinates(cx, cy)

            if hand_label == 'Right':
                if not is_paused:
                    if prev_point is not None:
                        cv2.line(canvas_np, prev_point, (cx, cy), DRAW_COLOR, thickness)
                    prev_point = (cx, cy)
                else:
                    prev_point = None
                    print("Drawing paused")

            elif hand_label == 'Left':
                if is_open_hand(hand_landmarks) and (current_time - last_gesture_time) > gesture_debounce:
                    canvas_np.fill(0)
                    prev_point = None
                    last_gesture_time = current_time
                    print("Canvas cleared")

            # Draw landmarks and green dot
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    else:
        prev_point = None
        print("No hand detected")

    # Display pause status
    status_text = "PAUSED" if is_paused else "DRAWING"
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_paused else (0, 255, 0), 2)

    # Overlay canvas on frame
    output = cv2.addWeighted(frame, 0.5, canvas_np, 0.5, 0)

    cv2.imshow("Index Finger Drawing", output)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord(' ') or key == ord(' '):  # Press 'P' or 'p' to toggle pause
        is_paused = not is_paused
        print(f"Pause toggled: {'Paused' if is_paused else 'Resumed'}")

cap.release()
cv2.destroyAllWindows()