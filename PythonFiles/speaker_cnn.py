import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# Load CNN model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_gesture_cnn.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names from model training directory structure
DATA_DIR = r"C:\Project\TNM114\HagridCNN\hagrid_data\processed_images"
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Loaded {len(class_names)} gesture classes: {class_names}")

# Initialize MediaPipe hand detector for hand detection and cropping
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Image parameters (must match training)
IMG_SIZE = (64, 64)

def extract_hand_roi(frame, hand_landmarks):
    """
    Extract hand region of interest from frame using hand landmarks bounding box.
    """
    h, w, _ = frame.shape
    
    # Get all landmark coordinates
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    # Calculate bounding box with padding
    x_min = max(0, int(min(x_coords) * w) - 20)
    x_max = min(w, int(max(x_coords) * w) + 20)
    y_min = max(0, int(min(y_coords) * h) - 20)
    y_max = min(h, int(max(y_coords) * h) + 20)
    
    # Crop hand region
    hand_roi = frame[y_min:y_max, x_min:x_max]
    
    return hand_roi, (x_min, y_min, x_max, y_max)

def preprocess_image(hand_roi):
    """
    Preprocess hand ROI for CNN prediction.
    Resize to 64x64 and normalize to [0,1].
    """
    # Resize to model input size
    resized = cv2.resize(hand_roi, IMG_SIZE)
    
    # Normalize to [0,1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

class SpeakerCNN:
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
                
                # Extract hand ROI
                hand_roi, bbox = extract_hand_roi(frame, lm)
                
                if hand_roi.size > 0:
                    # Preprocess and predict
                    X = preprocess_image(hand_roi)
                    preds = model.predict(X, verbose=0)
                    class_id = np.argmax(preds)
                    confidence = preds[0][class_id]
                    gesture = class_names[class_id]
                    
                    self.current_gesture = gesture
                    text = f"{gesture} ({confidence*100:.1f}%)"
                else:
                    text = "Hand too small"
                    self.current_gesture = "none"
            else:
                text = "No hand detected"
                self.current_gesture = "none"

            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("CNN Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    speaker = SpeakerCNN()
    speaker.run()
