import socket
import time
import tensorflow as tf
import mediapipe as mp
import cv2
import sys
import pkg_resources

HOST = "127.0.0.1"
PORT = 5005

rules = ["up", "down", "left", "right", "leftup", "leftdown", "rightup", "rightdown", "scatter", "cluster", "target", "nonsense"]
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
        padding = int(0.3 * (x_max - x_min))
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
                
        # Crop the region
        cropped = frame[y_min:y_max, x_min:x_max]

        if cropped.size > 0:
            # Resize to 128x128
            resized = cv2.resize(cropped, (128, 128))

            # Show both frames
            cv2.imshow("Cropped Hand (128x128)", resized)

        # Draw landmarks on main frame for debugging
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
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
        msg = rules[int(current_time) % len(rules)]
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