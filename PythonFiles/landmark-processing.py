import cv2
import mediapipe as mp
import os

# Get folder where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directory with cropped images
CROPPED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet", "Cropped")
CROPPED_DIR = os.path.abspath(CROPPED_DIR)

# Output directory for processed images with landmarks
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet", "Processed")
PROCESSED_DIR = os.path.abspath(PROCESSED_DIR)

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def process_image_with_landmarks(img_path, save_path):
    """Load image, draw MediaPipe landmarks, and save."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping invalid image: {img_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        print(f"No hand detected in {img_path}")
        return False

    # Draw landmarks on the image
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green dots
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Red lines
        )

    # Save the processed image
    cv2.imwrite(save_path, img)
    print(f"Processed: {save_path}")
    return True

def main():
    # Create main Processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    
    # Iterate through labeled folders
    for label_folder in os.listdir(CROPPED_DIR):
        cropped_label_path = os.path.join(CROPPED_DIR, label_folder)

        if not os.path.isdir(cropped_label_path):
            continue

        # Create corresponding folder in Processed directory
        processed_label_path = os.path.join(PROCESSED_DIR, label_folder)
        os.makedirs(processed_label_path, exist_ok=True)
        print(f"\nProcessing folder: {label_folder}")

        # Loop through all images in the cropped label folder
        for img_name in os.listdir(cropped_label_path):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(cropped_label_path, img_name)
            save_path = os.path.join(processed_label_path, img_name)
            
            if process_image_with_landmarks(img_path, save_path):
                total_processed += 1
            else:
                total_skipped += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {total_processed}")
    print(f"Total images skipped: {total_skipped}")
    print(f"Processed images saved to: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()