import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "MediaPipe")
CSV_FILE = os.path.join(DATA_DIR, "gesture_landmarks.csv")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_STATE = 42

class MediaPipeDataLoader:
    """Data loading and preprocessing for MediaPipe landmark data"""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.label_encoder = LabelEncoder()
        self.scaler_params = None
        
    def load_data(self):
        #\"\"\"Load CSV data and return features and labels, with header fallback.\"\"\"
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Data file not found: {self.csv_file}")

        # First attempt: read with header
        try:
            df = pd.read_csv(self.csv_file)
            if 'gesture' not in df.columns:
                raise KeyError("No 'gesture' column")
        except (KeyError, pd.errors.ParserError):
            # Fallback: read without header, assume last col is gesture
            df = pd.read_csv(self.csv_file, header=None)
            num_cols = df.shape[1]
            feature_cols = [f'x{i}' for i in range(21)] + \
                           [f'y{i}' for i in range(21)] + \
                           [f'z{i}' for i in range(21)]
            if num_cols != len(feature_cols) + 1:
                raise ValueError(f"Expected {len(feature_cols)+1} cols, got {num_cols}")
            df.columns = feature_cols + ['gesture']

        print(f"Loaded {len(df)} samples from {self.csv_file}")

        # Now proceed as before
        X = df.drop('gesture', axis=1).values.astype(np.float32)
        y = df['gesture'].values

        # (rest of normalization, encoding, splitting...)
        return X, y
    
    def normalize_landmarks(self, X):
        """
        Normalize landmarks using wrist-relative positioning and scaling
        This is the same preprocessing used in data collection
        """
        print("Applying landmark normalization...")
        
        # Reshape to (samples, 21, 3) for easier processing
        X_reshaped = X.reshape(-1, 21, 3)
        
        # 1. Wrist-relative normalization (landmark 0 is wrist)
        wrist = X_reshaped[:, 0:1, :]  # Keep as (samples, 1, 3)
        X_normalized = X_reshaped - wrist
        
        # 2. Scale normalization per sample
        normalized_samples = []
        for i in range(X_normalized.shape[0]):
            sample = X_normalized[i]
            
            # Find min/max for this sample
            min_vals = np.min(sample, axis=0)
            max_vals = np.max(sample, axis=0)
            range_vals = max_vals - min_vals
            
            # Avoid division by zero
            range_vals = np.where(range_vals == 0, 1, range_vals)
            
            # Scale to 0-1 range
            sample_scaled = (sample - min_vals) / range_vals
            normalized_samples.append(sample_scaled)
        
        X_normalized = np.array(normalized_samples)
        
        # Flatten back to (samples, 63)
        X_final = X_normalized.reshape(-1, 63)
        
        print(f"Normalized shape: {X_final.shape}")
        print(f"Value range: [{X_final.min():.3f}, {X_final.max():.3f}]")
        
        return X_final
    
    def encode_labels(self, y):
        """Encode string labels to integers"""
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Label mapping:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {class_name}")
        return y_encoded
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        print(f"\nSplitting data:")
        print(f"  Test split: {TEST_SPLIT}")
        print(f"  Validation split: {VALIDATION_SPLIT}")
        print(f"  Training split: {1 - TEST_SPLIT - VALIDATION_SPLIT}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, 
            stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = VALIDATION_SPLIT / (1 - TEST_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=RANDOM_STATE, stratify=y_temp
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)

class MediaPipeLandmarkModel:
    """Neural network model for MediaPipe landmark classification"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build dense neural network optimized for landmark data
        Much simpler than CNN since we're working with coordinate features
        """
        print(f"\nBuilding model for {self.num_classes} classes...")
        
        self.model = Sequential([
            # Input layer - 63 features (21 landmarks × 3 coordinates)
            Dense(128, activation='relu', input_shape=(63,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 1
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 2
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Note: sparse because we have integer labels
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model with callbacks"""
        print(f"\nStarting training...")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Max epochs: {EPOCHS}")
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test, label_encoder):
        """Evaluate model and show detailed metrics"""
        print(f"\n{'='*50}")
        print("MODEL EVALUATION")
        print(f"{'='*50}")
        
        # Basic evaluation
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Detailed classification report
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_pred_classes, 
            target_names=label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return test_accuracy, test_loss
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("MEDIAPIPE LANDMARK GESTURE RECOGNITION TRAINING")
    print("=" * 60)
    
    # 1. Load and preprocess data
    data_loader = MediaPipeDataLoader(CSV_FILE)
    
    try:
        X, y = data_loader.load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run collect_mediapipe_data.py first to generate training data.")
        return
    
    # Check if we have enough data
    if len(X) < 50:
        print(f"Warning: Only {len(X)} samples found. Consider collecting more data.")
        
    # 2. Normalize landmarks (same as preprocessing during inference)
    X_normalized = data_loader.normalize_landmarks(X)
    
    # 3. Encode labels
    y_encoded = data_loader.encode_labels(y)
    
    # 4. Split data
    (X_train, X_val, X_test), (y_train, y_val, y_test) = data_loader.split_data(
        X_normalized, y_encoded
    )
    
    # 5. Build and train model
    num_classes = len(data_loader.label_encoder.classes_)
    model = MediaPipeLandmarkModel(num_classes)
    model.build_model()
    
    # 6. Train
    history = model.train(X_train, y_train, X_val, y_val)
    
    # 7. Evaluate
    test_accuracy, test_loss = model.evaluate(X_test, y_test, data_loader.label_encoder)
    
    # 8. Plot results
    model.plot_training_history()
    
    # 9. Save model and label encoder
    MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_gesture_landmarks.keras")
    ENCODER_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")
    
    model.save_model(MODEL_PATH)
    
    # Save label encoder for inference
    import pickle
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(data_loader.label_encoder, f)
    print(f"Label encoder saved to: {ENCODER_PATH}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Label encoder saved: {ENCODER_PATH}")

if __name__ == "__main__":
    main()