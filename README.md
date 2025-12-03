# Hand Gesture Recognition for Video Game Control

A real-time hand gesture recognition system designed to control video game characters using computer vision and deep learning. This project is part of the course *Artificial Intelligence for Interactive Media - TNM114*.

## Overview

This system uses trained convolutional neural networks to identify hand gestures from a webcam feed. The recognized gestures are sent to a Godot game engine to control a swarm of characters in real-time, creating an intuitive gesture-based game controller.

The project combines:
- **Python Backend** - OpenCV and TensorFlow for real-time gesture recognition
- **Godot Frontend** - Game engine that listens for gesture inputs via WebSocket and controls character swarms accordingly

## System Architecture

### Python Component
- Real-time hand detection using MediaPipe
- CNN-based gesture classification trained on the HagRID dataset
- Two approaches:
  - **Landmark-based**: Uses hand skeleton coordinates for fast inference
  - **Image-based**: Uses 64×64 hand region images for higher accuracy
- WebSocket communication to send recognized gestures to Godot

### Godot Component
- Listens for gesture inputs from Python via WebSocket
- Interprets gestures as game commands
- Controls character swarm behavior based on recognized gestures

## Dataset

The project uses the **HagRID (Hand Gesture Recognition Image Dataset)** for training. The dataset is preprocessed and organized into labeled gesture classes stored in:
```
c:\Project\TNM114\HagridCNN\hagrid_data\processed_images
```

## Models

- **hand_gesture_landmarks.keras** - Landmark-based model (fast, real-time)
- **hand_gesture_cnn.keras** - CNN image-based model (higher accuracy)

## Usage

### Training the Model
```bash
python model_trainer.py
```

### Running Gesture Recognition Demos

Two demo scripts illustrate the different recognition approaches:

```bash
python speaker_demo.py  # Landmark-based approach with visual feedback
python speaker_cnn.py   # CNN image-based approach with higher accuracy
```

Both demos display real-time gesture recognition from your webcam and show the predicted gesture with confidence score.

### Playing the Game with Gesture Control

To use the full gesture-controlled game:

1. Launch the Godot game engine and run the game project
2. In a separate terminal, run the Python gesture recognition script:
```bash
python speaker.py  # Connects to the Godot game via WebSocket
```

The Python script will connect to the Godot game via WebSocket, sending recognized hand gestures in real-time. These gestures control the swarm of characters in the game, allowing you to steer them intuitively using hand movements captured from your webcam.

## Requirements

- Python 3.8+
- TensorFlow/Keras
- OpenCV
- MediaPipe
- NumPy
- Godot Engine 4.x (for game integration)
