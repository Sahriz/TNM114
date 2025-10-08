import cv2
import mediapipe as mp
import os

# Get folder where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Root directory where your labeled folders live
ROOT_DIR = os.path.join(SCRIPT_DIR,"..", "Data", "TemporarySet")  # Go up one folder from PythonPrograms
# Make it an absolute normalized path
ROOT_DIR = os.path.abspath(ROOT_DIR)
# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def process_image(img_path, save_path):
    """Detect hand, crop around it, resize to 128x128, and save."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping invalid image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        print(f"No hand detected in {img_path}")
        return

    h, w, _ = img.shape
    landmarks = results.multi_hand_landmarks[0].landmark
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]

    min_x, max_x = int(min(x_coords) * w), int(max(x_coords) * w)
    min_y, max_y = int(min(y_coords) * h), int(max(y_coords) * h)

    pad_x = int((max_x - min_x) * 0.2)
    pad_y = int((max_y - min_y) * 0.2)

    min_x = max(0, min_x - pad_x)
    max_x = min(w, max_x + pad_x)
    min_y = max(0, min_y - pad_y)
    max_y = min(h, max_y + pad_y)

    cropped = img[min_y:max_y, min_x:max_x]
    resized = cv2.resize(cropped, (128, 128))

    cv2.imwrite(save_path, resized)
    print(f"Saved: {save_path}")

def main():
    # Iterate through labeled folders
    for label_folder in os.listdir(ROOT_DIR):
        label_path = os.path.join(ROOT_DIR, label_folder)
        original_path = os.path.join(label_path, "Original")
        cropped_path = os.path.join(label_path, "Cropped")

        if not os.path.isdir(original_path):
            continue  # Skip if no "Original" folder exists

        os.makedirs(cropped_path, exist_ok=True)
        print(f"\nProcessing folder: {label_folder}")

        # Loop through all images in Original/
        for img_name in os.listdir(original_path):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(original_path, img_name)
            save_path = os.path.join(cropped_path, img_name)
            process_image(img_path, save_path)

    print("\nAll images processed successfully!")

main()
