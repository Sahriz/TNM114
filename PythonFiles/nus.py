import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Root directory for NUS Hand Posture Dataset II
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUS_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "NUS Hand Posture dataset-II", "Hand Postures")
NUS_DIR = os.path.abspath(NUS_DIR)

# Create a reorganized directory for training
ORGANIZED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "NUS_Organized")
ORGANIZED_DIR = os.path.abspath(ORGANIZED_DIR)

def organize_nus_dataset():
    """
    NUS dataset structure has images like:
    Hand Postures/
        A_1.png
        A_2.png
        B_1.png
        B_2.png
        ...
    
    This function organizes by first letter (gesture label):
    NUS_Organized/
        A/
            A_1.png
            A_2.png
        B/
            B_1.png
            B_2.png
    """
    print("Organizing NUS Hand Posture Dataset II...")
    
    # Clean up and recreate organized directory
    if os.path.exists(ORGANIZED_DIR):
        shutil.rmtree(ORGANIZED_DIR)
    os.makedirs(ORGANIZED_DIR)
    
    # Check if NUS_DIR exists
    if not os.path.exists(NUS_DIR):
        print(f"ERROR: NUS dataset not found at {NUS_DIR}")
        print("Please ensure the dataset is extracted to Data/NUS Hand Posture dataset-II/")
        return False
    
    # Dictionary to count images per gesture
    gesture_counts = {}
    total_images = 0
    
    # Process all image files in the Hand Postures directory
    for img_file in os.listdir(NUS_DIR):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Extract the first letter as the gesture label
        gesture_label = img_file[0].upper()
        
        # Create gesture directory if it doesn't exist
        gesture_dir = os.path.join(ORGANIZED_DIR, gesture_label)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
            gesture_counts[gesture_label] = 0
        
        # Copy image to the appropriate gesture folder
        src_img = os.path.join(NUS_DIR, img_file)
        dst_img = os.path.join(gesture_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        gesture_counts[gesture_label] += 1
        total_images += 1
    
    # Print summary
    print("\nGesture distribution:")
    for gesture in sorted(gesture_counts.keys()):
        print(f"  Gesture '{gesture}': {gesture_counts[gesture]} images")
    
    print(f"\nTotal: {len(gesture_counts)} gestures, {total_images} images")
    print(f"Organized dataset saved to: {ORGANIZED_DIR}\n")
    return True

# Organize the dataset first
if not organize_nus_dataset():
    exit(1)

# Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

#augmentation intensity
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,        # Reduced from 30
    width_shift_range=0.1,    # Reduced from 0.2
    height_shift_range=0.1,   # Reduced from 0.2
    zoom_range=0.1,           # Reduced from 0.2
    shear_range=0.05,         # Reduced from 0.15
    brightness_range=[0.9, 1.1],  # Reduced from [0.8, 1.2]
    fill_mode='nearest',
    horizontal_flip=False,
    validation_split=0.2
)

# No augmentation for validation
datagen_val = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create generators
train_generator = datagen_train.flow_from_directory(
    ORGANIZED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = datagen_val.flow_from_directory(
    ORGANIZED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

num_classes = train_generator.num_classes
print(f"Number of gesture classes: {num_classes}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Build the CNN model
model = Sequential([
    # First conv block
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Second conv block
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Third conv block
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Dense layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train the model
print("\nStarting training on NUS Hand Posture Dataset II...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save the trained model
MODEL_PATH = os.path.join(SCRIPT_DIR, "nus_hand_gesture_model.keras")
model.save(MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# Save class labels
import json
class_indices = train_generator.class_indices
labels = {v: k for k, v in class_indices.items()}
LABELS_PATH = os.path.join(SCRIPT_DIR, "nus_class_labels.json")
with open(LABELS_PATH, 'w') as f:
    json.dump(labels, f, indent=2)
print(f"Class labels saved to: {LABELS_PATH}")

# Print final results
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# If validation accuracy > 80%, the model and architecture are good
if history.history['val_accuracy'][-1] > 0.8:
    print("\n✓ Model works well with proper dataset!")
    print("  Your architecture is fine - the issue is your original dataset size.")
else:
    print("\n⚠ Results suggest the model needs tuning or more training data.")