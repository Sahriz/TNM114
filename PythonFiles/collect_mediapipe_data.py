import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
import pandas as pd

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "MediaPipe")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

# CSV file for storing landmark data
CSV_FILE = os.path.join(OUTPUT_DIR, "gesture_landmarks.csv")

# Number of samples to capture per gesture
SAMPLES_PER_GESTURE = 200

# Delay between captures (in seconds) 
CAPTURE_DELAY = 0.02

# Gesture labels (matching your original list)
GESTURES = [
    "Cluster",
    "Left",
    "Upleft",
    "Up",
    "UpRight",
    "Right",
    "Downright",
    "Down",
    "Downleft",
    "Scatter",
    "Target"
]

class MediaPipeDataCollector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        self.init_csv_file()
    
    def init_csv_file(self):
        """Initialize CSV file with column headers"""
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Create headers: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, gesture
                headers = []
                for i in range(21):  # 21 landmarks
                    headers.extend([f'x{i}', f'y{i}', f'z{i}'])
                headers.append('gesture')
                
                writer.writerow(headers)
            print(f"Created CSV file: {CSV_FILE}")
        else:
            print(f"Using existing CSV file: {CSV_FILE}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from MediaPipe results"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def save_landmarks(self, landmarks, gesture_name):
        """Save landmarks to CSV file"""
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            row = landmarks + [gesture_name]
            writer.writerow(row)
    
    def capture_gesture_data(self, gesture_name, gesture_index):
        """Capture landmark data for a specific gesture"""
        print(f"\n{'='*60}")
        print(f"Gesture {gesture_index + 1}/{len(GESTURES)}: {gesture_name}")
        print(f"{'='*60}")
        print(f"Press SPACE to capture landmarks (manual mode)")
        print(f"Press ENTER to start auto-capture of {SAMPLES_PER_GESTURE} samples")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        captured_count = 0
        auto_mode = False
        last_capture_time = 0
        
        while captured_count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw landmarks if hand is detected
            hand_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    hand_detected = True
            
            # Display information on frame
            mode_text = "AUTO" if auto_mode else "MANUAL"
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {mode_text}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Captured: {captured_count}/{SAMPLES_PER_GESTURE}", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if hand_detected:
                cv2.putText(frame, "Hand Detected!", (10, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if not auto_mode:
                cv2.putText(frame, "SPACE: Capture | ENTER: Auto Mode", (10, 155),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Auto capturing...", (10, 155),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("MediaPipe Data Collection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == 13 and not auto_mode:  # Enter key - start auto mode
                auto_mode = True
                print(f"Starting auto-capture mode for '{gesture_name}'...")
                last_capture_time = current_time
            elif key == ord(' ') and not auto_mode:  # Space key - manual capture
                if hand_detected and results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
                    self.save_landmarks(landmarks, gesture_name)
                    captured_count += 1
                    print(f"Manual capture {captured_count}/{SAMPLES_PER_GESTURE}")
                else:
                    print("No hand detected - capture skipped")
            
            # Auto capture logic
            if auto_mode and hand_detected and results.multi_hand_landmarks:
                if current_time - last_capture_time >= CAPTURE_DELAY:
                    landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
                    self.save_landmarks(landmarks, gesture_name)
                    captured_count += 1
                    last_capture_time = current_time
                    print(f"Auto capture {captured_count}/{SAMPLES_PER_GESTURE}")
        
        cap.release()
        
        if captured_count >= SAMPLES_PER_GESTURE:
            print(f"✓ Completed capturing {captured_count} samples for '{gesture_name}'")
            return True
        else:
            print(f"Capture incomplete: {captured_count}/{SAMPLES_PER_GESTURE} samples collected")
            return False
    
    import pandas as pd

    def display_statistics(self):
        #Display statistics about collected data using pandas
        if not os.path.exists(CSV_FILE):
            print("No data file found")
            return

        # Load entire CSV into DataFrame (auto-detect headers or no-header)
        try:
            df = pd.read_csv(CSV_FILE)
        except pd.errors.ParserError:
            # Fall back to no-header
            df = pd.read_csv(CSV_FILE, header=None)
            # Assign generic column names: x0,y0,z0,…,x20,y20,z20,label
            coords = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
            df.columns = coords + ["gesture"]
        # Now the last column is guaranteed to be 'gesture'
        gesture_col = df.columns[-1]

        total = len(df)
        counts = df[gesture_col].value_counts()
        print(f"\\n{'='*60}")
        print("DATA COLLECTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total samples collected: {total}")
        print(f"Data saved to: {CSV_FILE}\\n")
        print("Samples per gesture:")
        for gesture in GESTURES:
            cnt = counts.get(gesture, 0)
            mark = "✓" if cnt >= SAMPLES_PER_GESTURE else "⚠"
            print(f"  {mark} {gesture}: {cnt}/{SAMPLES_PER_GESTURE}")

    
    def run_collection(self):
        """Main data collection loop"""
        print("=" * 60)
        print("MEDIAPIPE GESTURE DATA COLLECTOR")
        print("=" * 60)
        print(f"Total gestures: {len(GESTURES)}")
        print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
        print(f"Auto-capture delay: {CAPTURE_DELAY}s")
        print(f"Output file: {CSV_FILE}")
        print("=" * 60)
        
        # Collect data for each gesture
        for idx, gesture in enumerate(GESTURES):
            success = self.capture_gesture_data(gesture, idx)
            if not success:
                print(f"\nData collection stopped at gesture: {gesture}")
                break
        else:
            # Only executes if loop completes without break
            print("\n" + "=" * 60)
            print("ALL GESTURES DATA COLLECTED!")
            print("=" * 60)
        
        cv2.destroyAllWindows()
        self.display_statistics()

def main():
    collector = MediaPipeDataCollector()
    collector.run_collection()

if __name__ == "__main__":
    main()