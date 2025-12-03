import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import pickle

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

class SpeakerDemo:
    def __init__(self):
        self.cap = None
        self.current_gesture = "none"

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Unable to open webcam")

    def run(self):
        self.setup_camera()
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
            else:
                text = "No hand detected"
                self.current_gesture = "none"

            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Landmark Gesture Recognition Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = SpeakerDemo()
    demo.run()
