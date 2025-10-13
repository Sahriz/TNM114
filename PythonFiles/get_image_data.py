import cv2
import os
import time

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet", "Original")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

# Number of images to capture per gesture
IMAGES_PER_GESTURE = 5

# Delay between captures (in seconds)
CAPTURE_DELAY = 1.0

# Gesture labels (in order)
GESTURES = [
    "Cluster",
    "Down",
    "Left",
    "Right",
    "Scatter",
    "Target",
    "Up"
]

def create_gesture_folders():
    """Create folders for each gesture if they don't exist"""
    for gesture in GESTURES:
        gesture_path = os.path.join(OUTPUT_DIR, gesture)
        os.makedirs(gesture_path, exist_ok=True)
    print(f"Gesture folders ready at: {OUTPUT_DIR}\n")

def capture_images_for_gesture(gesture_name, gesture_index):
    """Capture images for a specific gesture"""
    gesture_path = os.path.join(OUTPUT_DIR, gesture_name)
    
    print(f"\n{'='*60}")
    print(f"Gesture {gesture_index + 1}/{len(GESTURES)}: {gesture_name}")
    print(f"{'='*60}")
    print(f"Press ENTER to start capturing {IMAGES_PER_GESTURE} images...")
    print("(Press 'q' to quit)")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Wait for user to press Enter
    waiting = True
    while waiting:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            cap.release()
            return False
        
        # Display instructions on frame
        instruction_text = f"Gesture: {gesture_name} - Press ENTER to start"
        cv2.putText(frame, instruction_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Images to capture: {IMAGES_PER_GESTURE}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Webcam - Get Image Data", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == 13:  # Enter key
            waiting = False
    
    # Start capturing images
    print(f"\nCapturing images for '{gesture_name}'...")
    
    for i in range(IMAGES_PER_GESTURE):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to grab frame {i + 1}")
            break
        
        # Display countdown and progress
        progress_text = f"Capturing: {i + 1}/{IMAGES_PER_GESTURE}"
        cv2.putText(frame, progress_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, gesture_name, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Webcam - Get Image Data", frame)
        cv2.waitKey(1)
        
        # Save image
        timestamp = int(time.time() * 1000)  # Millisecond timestamp for uniqueness
        filename = f"{gesture_name}_{timestamp}_{i+1}.jpg"
        filepath = os.path.join(gesture_path, filename)
        cv2.imwrite(filepath, frame)
        
        print(f"  Captured image {i + 1}/{IMAGES_PER_GESTURE}: {filename}")
        
        # Wait before next capture
        if i < IMAGES_PER_GESTURE - 1:  # Don't wait after the last image
            time.sleep(CAPTURE_DELAY)
    
    cap.release()
    print(f"\n✓ Completed capturing {IMAGES_PER_GESTURE} images for '{gesture_name}'")
    
    return True

def main():
    print("="*60)
    print("GESTURE IMAGE CAPTURE TOOL")
    print("="*60)
    print(f"Total gestures: {len(GESTURES)}")
    print(f"Images per gesture: {IMAGES_PER_GESTURE}")
    print(f"Delay between captures: {CAPTURE_DELAY}s")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Create gesture folders
    create_gesture_folders()
    
    # Capture images for each gesture
    for idx, gesture in enumerate(GESTURES):
        success = capture_images_for_gesture(gesture, idx)
        if not success:
            print("\nCapture process stopped.")
            break
    else:
        # Only executes if loop completes without break
        print("\n" + "="*60)
        print("ALL GESTURES CAPTURED SUCCESSFULLY!")
        print("="*60)
        print(f"Total images captured: {len(GESTURES) * IMAGES_PER_GESTURE}")
        print(f"Images saved to: {OUTPUT_DIR}")
    
    cv2.destroyAllWindows()
    print("\nExiting...")

if __name__ == "__main__":
    main()