import os
import json
import cv2
import numpy as np

# --- Configuration ---
IMAGE_SIZE = (64, 64) 
DATA_DIR = 'hagrid_data' 
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_images')


def crop_and_save_hand_images(annotations_folder, raw_images_root, processed_root):
    """
    Crops the hand region based on bounding box annotations and saves the resized image.
    Handles multiple JSON files (one per gesture class).
    """
    if not os.path.exists(annotations_folder):
        print(f"Error: Annotations folder not found at {annotations_folder}")
        return

    # Find all JSON annotation files
    json_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"Error: No JSON files found in {annotations_folder}")
        return
    
    print(f"Found {len(json_files)} annotation files")
    
    # Collect all class names
    class_names = sorted([f.replace('.json', '') for f in json_files])
    
    # Store the class names for later use by the CNN
    with open(os.path.join(DATA_DIR, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"Gesture classes: {class_names}\n")
    print("Starting image cropping and resizing...")
    
    cropped_count = 0
    skipped_count = 0
    
    # Process each annotation file
    for json_file in json_files:
        gesture_class = json_file.replace('.json', '')
        annotation_path = os.path.join(annotations_folder, json_file)
        
        print(f"\nProcessing {gesture_class}...", end=' ')
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        class_count = 0
        class_skipped = 0
        
        # The images are in folders like "train_val_call", "train_val_peace", etc.
        image_folder = os.path.join(raw_images_root, f'train_val_{gesture_class}')
        
        if not os.path.exists(image_folder):
            print(f"✗ Image folder not found: {image_folder}")
            continue
        
        for img_key, data in annotations.items():
            # Image filename is just the UUID with .jpg extension
            image_path = os.path.join(image_folder, img_key + '.jpg')
            
            if not os.path.exists(image_path):
                class_skipped += 1
                continue

            image = cv2.imread(image_path)
            if image is None:
                class_skipped += 1
                continue
                
            H, W, _ = image.shape
            
            # Extract bounding box
            if 'bboxes' in data and data['bboxes'] and len(data['bboxes']) > 0:
                bbox_norm = data['bboxes'][0]
                
                # Convert normalized coordinates to pixel coordinates
                x_norm, y_norm, w_norm, h_norm = bbox_norm
                x_min = int(x_norm * W)
                y_min = int(y_norm * H)
                x_max = int(x_min + w_norm * W)
                y_max = int(y_min + h_norm * H)

                # Apply padding for better feature capture
                padding_ratio = 0.2
                pad_x = int(w_norm * W * padding_ratio)
                pad_y = int(h_norm * H * padding_ratio)
                
                x_min = max(0, x_min - pad_x)
                y_min = max(0, y_min - pad_y)
                x_max = min(W, x_max + pad_x)
                y_max = min(H, y_max + pad_y)
                
                # Crop the image
                cropped_image = image[y_min:y_max, x_min:x_max]
                
                # Skip if crop is too small
                if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                    class_skipped += 1
                    continue

                # Resize to target CNN input size
                resized_image = cv2.resize(cropped_image, IMAGE_SIZE)

                # Create destination directory for the class
                class_dir = os.path.join(processed_root, gesture_class)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save the processed image
                output_path = os.path.join(class_dir, img_key + '.png')
                cv2.imwrite(output_path, resized_image)
                cropped_count += 1
                class_count += 1
            else:
                class_skipped += 1
        
        skipped_count += class_skipped
        print(f"✓ {class_count} images processed, {class_skipped} skipped")
    
    print(f"\n" + "="*80)
    print(f"✓ Successfully processed {cropped_count} images")
    print(f"✗ Skipped {skipped_count} images (not found or unreadable)")
    print(f"📁 Saved to: {PROCESSED_DIR}")
    print("="*80)


if __name__ == '__main__':
    # Check if data already exists
    raw_images_folder = os.path.join(DATA_DIR, 'raw_images', 'subsample')
    annotations_folder = os.path.join(DATA_DIR, 'raw_annotations', 'ann_subsample')
    
    if not os.path.exists(raw_images_folder):
        print(f"✗ Error: Images folder not found at {raw_images_folder}")
        print("Please run git_download_hagrid.py first to download and organize the data.")
        exit(1)
    
    if not os.path.exists(annotations_folder):
        print(f"✗ Error: Annotations folder not found at {annotations_folder}")
        print("Please run git_download_hagrid.py first to download and organize the data.")
        exit(1)
    
    print("✓ Found raw images and annotations")
    print(f"  Images: {raw_images_folder}")
    print(f"  Annotations: {annotations_folder}\n")
    
    # Create processed images directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Process and crop images
    crop_and_save_hand_images(annotations_folder, raw_images_folder, PROCESSED_DIR)
    
    print("\n✓✓✓ Setup complete! You can now run cnn_gesture_trainer.py to train your model.")