import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet", "Processed")
ROOT_DIR = os.path.abspath(ROOT_DIR)

# Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50

# Use ImageDataGenerator for training and validation (20% split)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generators = []
val_generators = []

# Merge all augmented images into a single folder tree for flow_from_directory
# We'll point flow_from_directory to ROOT_DIR but it must see the class subfolders
# So your structure should be:
# ROOT_DIR/
#   follow/Augmented/
#   scatter/Augmented/
#   ...
# Real-time augmentation for training (creates variations on-the-fly)
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,           # Rotate up to 30 degrees
    width_shift_range=0.15,       # Shift horizontally
    height_shift_range=0.15,      # Shift vertically
    zoom_range=0.15,              # Zoom in/out
    fill_mode='nearest',
    horizontal_flip=False,       # Don't flip hands (gestures might change meaning)
    validation_split=0.2
)
# No augmentation for validation (only rescaling)
datagen_val = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


# Define train generator
train_generator = datagen_train.flow_from_directory(
    ROOT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,  # Explicitly enable shuffling
    seed=42  # Optional: for reproducible results
)

# Define validation generator
val_generator = datagen_val.flow_from_directory(
    ROOT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,  # Also shuffle validation for better batch diversity
    seed=42  # Use same seed for consistent split
)


num_classes = train_generator.num_classes
print(f"Number of gesture classes: {num_classes}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Calculate steps to see each image multiple times per epoch with different augmentations
AUGMENTATION_FACTOR = 20  # Each original image will be seen 20 times per epoch with different augmentations
steps_per_epoch = (train_generator.samples * AUGMENTATION_FACTOR) // BATCH_SIZE
validation_steps = val_generator.samples // BATCH_SIZE

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")


# Build the CNN model
model = Sequential([
    # First conv block
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Second conv block
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Dense layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add callbacks to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Specify how many batches per epoch
    validation_data=val_generator,
    validation_steps=validation_steps,  # Specify validation batches
    epochs=50,
    callbacks=[early_stop, reduce_lr]
)

# Save the trained model
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_gesture_cnn.keras")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"\Model saved to: {MODEL_PATH}")
