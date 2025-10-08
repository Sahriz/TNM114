import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# Path to the Cropped images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet")
ROOT_DIR = os.path.abspath(ROOT_DIR)

# Augmented images will be saved here (inside each label folder)
AUGMENTED_FOLDER_NAME = "Augmented"

# Image size
IMG_SIZE = (128, 128)

# ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=20,      # rotate images by up to 20 degrees
    width_shift_range=0.1,  # shift horizontally
    height_shift_range=0.1, # shift vertically
    zoom_range=0.2,         # zoom in/out
    horizontal_flip=True,   # flip horizontally
    fill_mode='nearest'     # how to fill empty pixels
)

def augment_images():
    for label_folder in os.listdir(ROOT_DIR):
        label_path = os.path.join(ROOT_DIR, label_folder)
        cropped_path = os.path.join(label_path, "Cropped")
        augmented_path = os.path.join(label_path, AUGMENTED_FOLDER_NAME)

        if not os.path.isdir(cropped_path):
            continue  # skip if no Cropped folder

        os.makedirs(augmented_path, exist_ok=True)
        print(f"\nAugmenting images in: {label_folder}")

        for img_name in os.listdir(cropped_path):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(cropped_path, img_name)
            img = load_img(img_path, target_size=IMG_SIZE)
            x = img_to_array(img)  # Convert to numpy array
            x = np.expand_dims(x, axis=0)  # Add batch dimension

            # Generate 5 augmented images per original
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_path,
                                      save_prefix='aug', save_format='png'):
                i += 1
                if i >= 5:  # create 5 augmented images per original
                    break

        print(f"Finished augmenting {label_folder}")

if __name__ == "__main__":
    augment_images()
    print("\nAll images augmented successfully!")
