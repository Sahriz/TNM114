import socket
import os
import time
import tensorflow as tf
import mediapipe as mp
import cv2
import sys
import pkg_resources
from tensorflow.keras.models import load_model
import numpy as np

HOST = "127.0.0.1"
PORT = 5005
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.abspath(SCRIPT_DIR)
SCRIPT_DIR = os.path.join(SCRIPT_DIR, "hand_gesture_cnn.h5")

gesture_labels = ["up", "down", "left", "right", "leftup", "leftdown", "rightup", "rightdown", "scatter", "cluster", "target", "nonsense"]
current_gesture = "nonesense"


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,       # detect up to two hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

running = True

cap = cv2.VideoCapture(0)


model = load_model(SCRIPT_DIR)

def classify_frame(frame):
    global current_gesture

    # Preprocess the frame (resize + normalize)
    img = cv2.resize(frame, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    current_gesture = gesture_labels[class_index]

    return current_gesture

def resize_hand(frame, results):
    resized = None
    for hand_landmarks in results.multi_hand_landmarks:
        # Get bounding box around hand
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        # Add padding to make sure we capture the whole hand
        x_padding = int(0.3 * (x_max - x_min))
        y_padding = int(0.3 * (y_max - y_min))
        x_min = max(0, x_min - x_padding)
        y_min = max(0, y_min - y_padding)
        x_max = min(w, x_max + x_padding)
        y_max = min(h, y_max + y_padding)
        
        # Draw landmarks on main frame first (for debugging the full frame)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Crop the region
        cropped = frame[y_min:y_max, x_min:x_max].copy()

        if cropped.size > 0:
            # Create a copy for drawing landmarks on the cropped image
            cropped_with_landmarks = cropped.copy()
            
            # Adjust landmarks to cropped coordinates
            adjusted_landmarks = mp_hands.HandLandmark
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            
            # Draw landmarks on cropped image by adjusting coordinates
            crop_h, crop_w = cropped.shape[:2]
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_lm = hand_landmarks.landmark[start_idx]
                end_lm = hand_landmarks.landmark[end_idx]
                
                # Convert to cropped image coordinates
                start_x = int((start_lm.x * w - x_min) * crop_w / (x_max - x_min))
                start_y = int((start_lm.y * h - y_min) * crop_h / (y_max - y_min))
                end_x = int((end_lm.x * w - x_min) * crop_w / (x_max - x_min))
                end_y = int((end_lm.y * h - y_min) * crop_h / (y_max - y_min))
                
                # Draw connection line
                cv2.line(cropped_with_landmarks, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
            
            # Draw landmark points
            for lm in hand_landmarks.landmark:
                lm_x = int((lm.x * w - x_min) * crop_w / (x_max - x_min))
                lm_y = int((lm.y * h - y_min) * crop_h / (y_max - y_min))
                cv2.circle(cropped_with_landmarks, (lm_x, lm_y), 3, (0, 255, 0), -1)
            
            # Resize to 128x128 (with landmarks drawn on it)
            resized = cv2.resize(cropped_with_landmarks, (128, 128))

            # Show both frames
            cv2.imshow("Cropped Hand (128x128)", resized)
            
    return resized

def camera_feed():
    global running
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        running = False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    resized = None
    if results.multi_hand_landmarks:
        resized = resize_hand(frame, results)
        classified_gesture = classify_frame(resized)
        cv2.putText(frame, f"Gesture: {classified_gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        pass

    cv2.imshow("Camera Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        running = False

    

def speaker(last_sent_time, interval=2):
    """Send rule messages periodically."""
    current_time = time.time()
    if current_time - last_sent_time >= interval:
        msg = gesture_labels[int(current_time) % len(gesture_labels)]
        sock.sendall(msg.encode("utf-8"))
        print(f"Sent rule: {msg}")
        return current_time
    return last_sent_time

def debug_versions():
    # 1️⃣ Check TensorFlow version
    print("TensorFlow version:", tf.__version__)

    # 2️⃣ Check MediaPipe version
    print("MediaPipe version:", mp.__version__)

    # 3️⃣ Check OpenCV version
    print("OpenCV version:", cv2.__version__)

    # 4️⃣ Check protobuf version
    try:
        import google.protobuf
        print("Protobuf version:", google.protobuf.__version__)
    except ImportError:
        print("Protobuf not installed!")

    # 5️⃣ Optional: list all installed packages and their versions
    print("\nInstalled packages and versions:")
    for pkg in ["tensorflow", "mediapipe", "opencv-python", "protobuf"]:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"{pkg}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{pkg} not installed")
        #speaker()
        #print("Speaker module loaded")

def run():
    sock.connect((HOST, PORT))
    last_sent_time = 0 # Ensure immediate first send
    while running:
        camera_feed()
        last_sent_time = speaker(last_sent_time,2)
    cap.release()
    cv2.destroyAllWindows()
    sock.close()

def main(): 
   run()

main()