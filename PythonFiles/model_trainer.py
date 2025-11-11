import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Hardcoded paths to HagRID processed images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"C:\Project\TNM114\HagridCNN\hagrid_data\processed_images"

print(f"Data directory: {DATA_DIR}")

# Image parameters
IMG_SIZE = (64, 64)
BATCH_SIZE = 64  # Increased for large dataset - faster training
EPOCHS = 20  # Reduced - large dataset trains faster

# Real-time augmentation for training (creates variations on-the-fly)
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,           # Rotate up to 10 degrees
    width_shift_range=0.05,       # Shift horizontally
    height_shift_range=0.05,      # Shift vertically
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
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,  # Explicitly enable shuffling
    seed=42  # Optional: for reproducible results
)

# Define validation generator
val_generator = datagen_val.flow_from_directory(
    DATA_DIR,
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

# Calculate steps - no augmentation factor needed with large dataset
# Let Keras handle it automatically with steps_per_epoch=None (will use all data once per epoch)
steps_per_epoch = None  # Use all training data once per epoch
validation_steps = None  # Use all validation data

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")


# Build the CNN model
model = Sequential([
    # First conv block
    Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    Conv2D(32, (5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Second conv block
    Conv2D(64, (5,5), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Third conv block
    Conv2D(128, (5,5), activation='relu', padding='same'),
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
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# Save the trained model
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_gesture_cnn.keras")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"\Model saved to: {MODEL_PATH}")
