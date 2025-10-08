import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet")
ROOT_DIR = os.path.abspath(ROOT_DIR)

# Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

# Path to Augmented images
def get_augmented_path(label_folder):
    return os.path.join(ROOT_DIR, label_folder, "Augmented")

# Collect all label folders that have an Augmented subfolder
label_folders = [f for f in os.listdir(ROOT_DIR)
                 if os.path.isdir(os.path.join(ROOT_DIR, f, "Augmented"))]

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

# Define train generator
train_generator = datagen.flow_from_directory(
    ROOT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,  # Explicitly enable shuffling
    seed=42  # Optional: for reproducible results
)

# Define validation generator
val_generator = datagen.flow_from_directory(
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

# Build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save the trained model
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_gesture_cnn.h5")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"\Model saved to: {MODEL_PATH}")
