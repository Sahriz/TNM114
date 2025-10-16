import socket
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
import pickle

# Socket configuration
HOST = "127.0.0.1"
PORT = 5005

# Load landmark model and encoder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_gesture_landmarks.keras')
ENCODER_PATH = os.path.join(SCRIPT_DIR, 'label_encoder.pkl')
model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def preprocess_landmarks(landmarks):
    """
    Normalize landmarks: wrist-relative and scale to [0,1].
    Input: list of 21 [x,y,z] landmarks.
    """
    data = np.array(landmarks).reshape(21, 3)
    wrist = data[0]
    data = data - wrist
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    range_vals = np.where((max_vals - min_vals) == 0, 1, (max_vals - min_vals))
    data = (data - min_vals) / range_vals
    return data.flatten()[np.newaxis, :]

class Speaker:
    def __init__(self):
        self.cap = None
        self.sock = None
        self.current_gesture = "none"
        self.last_sent_time = 0

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Unable to open webcam")

    def setup_socket(self):
        """Connect to the socket server"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((HOST, PORT))
            print(f"Connected to socket server at {HOST}:{PORT}")
        except Exception as e:
            print(f"Warning: Could not connect to socket server: {e}")
            print("Continuing without socket connection...")
            self.sock = None

    def send_gesture(self, gesture, interval=0.05):
        """Send gesture via socket at specified interval"""
        if self.sock is None:
            return
        
        current_time = time.time()
        if current_time - self.last_sent_time >= interval and gesture != "none":
            try:
                self.sock.sendall(gesture.encode("utf-8"))
                print(f"Sent gesture: {gesture}")
                self.last_sent_time = current_time
            except Exception as e:
                print(f"Error sending gesture: {e}")

    def run(self):
        self.setup_camera()
        self.setup_socket()
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                landmarks = [[p.x, p.y, p.z] for p in lm.landmark]

                # Preprocess and predict
                X = preprocess_landmarks(landmarks)
                preds = model.predict(X, verbose=0)
                class_id = np.argmax(preds)
                confidence = preds[0][class_id]
                gesture = label_encoder.inverse_transform([class_id])[0]
                
                self.current_gesture = gesture
                text = f"{gesture} ({confidence*100:.1f}%)"

                # Send gesture via socket every 0.05 seconds
                self.send_gesture(gesture, interval=0.05)
            else:
                text = "No hand detected"
                self.current_gesture = "none"

            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Speaker Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        if self.sock:
            self.sock.close()
            print("Socket connection closed")

if __name__ == "__main__":
    speaker = Speaker()
    speaker.run()