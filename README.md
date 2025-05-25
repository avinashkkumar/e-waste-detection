# E-Waste Detection System

A computer vision system for real-time detection of electronic waste (e-waste) items using a camera feed. This project uses deep learning to identify e-waste components from video input with minimal hardware requirements.

## Project Overview

This project provides tools for detecting electronic waste using computer vision and machine learning. It's designed to identify various e-waste components in real-time from a camera feed, helping in sorting and proper disposal of electronic waste.

## Available Scripts

The project includes the following scripts:

- **simple_ewaste_detector.py**: A simplified detection script for basic e-waste identification
- **real_time_detection.py**: Advanced detection with additional features like heatmap visualization
- **object_detection_model.py**: Core detector implementation used by both detection scripts
- **organize_dataset.py**: Prepares image datasets for training
- **prepare_data.py**: Handles data loading and preprocessing
- **train_model.py**: Creates and trains the CNN model

## Features

- **Real-time detection** of e-waste components from camera feed
- **Two detection modes**:
  - **Advanced Mode** (real_time_detection.py): Enhanced sliding window with clustered detection and heatmap
  - **Simple Mode** (simple_ewaste_detector.py): Basic detection with FPS counter
- **Visualization options**:
  - Bounding boxes with confidence percentages
  - Optional heatmap visualization in advanced mode
- **Hardware flexibility**:
  - CPU-optimized processing 
  - Automatic GPU acceleration when available
- **Interactive controls**:
  - Adjustable confidence threshold
  - Screenshot capture

## Project Structure

```
├── object_detection_model.py   # Core detector implementation
├── simple_ewaste_detector.py   # Simple e-waste detection script
├── real_time_detection.py      # Advanced detection with heatmap visualization
├── organize_dataset.py         # Prepares the dataset 
├── prepare_data.py             # Handles data loading and preprocessing
├── train_model.py              # Creates and trains a MobileNetV2-based CNN model
├── models/                     # Contains the trained model
│   └── ewaste_detector.h5      # Pre-trained e-waste detection model
├── dataset/                    # Raw input dataset
└── organized_dataset/          # Processed dataset after organization
```

## E-Waste Detection Theory

### What is E-Waste?

Electronic waste or e-waste refers to discarded electronic devices and components. These include computers, mobile phones, televisions, circuit boards, batteries, and other electronic equipment that have reached the end of their useful life. E-waste contains valuable materials that can be recycled, as well as hazardous substances that require proper handling.

### Computer Vision Approach for E-Waste Detection

This project uses a two-stage approach for e-waste detection:

1. **Feature Learning**: A convolutional neural network (CNN) learns discriminative features from images of e-waste items. The model is based on MobileNetV2, which provides good accuracy with lower computational requirements.

2. **Sliding Window Detection**: Rather than using region proposal networks, this project uses a sliding window approach:
   - Windows of varying sizes scan across the image
   - Each window is classified by the CNN
   - Windows with high e-waste probability are flagged as detections
   - Non-maximum suppression removes overlapping detections

### Challenges in E-Waste Detection

E-waste detection poses several unique challenges:

- **Visual Diversity**: E-waste components vary widely in shape, size, and appearance
- **Occlusion**: Items may be partially visible or stacked
- **Varying Lighting Conditions**: Detection must work under different lighting
- **False Positives**: Many everyday objects can resemble e-waste components

The model addresses these challenges through:

- **Data Augmentation**: Training with rotated, scaled, and color-shifted images
- **Confidence Thresholds**: Adjustable thresholds to balance detection sensitivity
- **Non-Maximum Suppression**: Algorithms to reduce duplicate detections
- **Size Filtering**: Rejection of detections that are too small to be valid e-waste

## Usage

### Simple E-Waste Detection

The simplified detector provides basic e-waste detection with FPS counter:

```
python simple_ewaste_detector.py [options]
```

Command-line options:
- `--camera`: Camera index to use (default: 0)
- `--threshold`: Detection confidence threshold (default: 0.85)
- `--model`: Path to trained model (default: models/ewaste_detector.h5)
- `--min-size`: Minimum detection size in pixels (default: 80)

Controls:
- Press 'q' to quit
- Press 's' to save screenshot

### Advanced E-Waste Detection

For more advanced features including heatmap visualization:

```
python real_time_detection.py [options]
```

Command-line options:
- `--camera`: Camera index to use (default: 0)
- `--model`: Path to trained model (default: models/ewaste_detector.h5)
- `--threshold`: Detection confidence threshold (default: 0.75)
- `--nms-threshold`: NMS threshold for duplicate removal (default: 0.8)
- `--resolution`: Camera resolution (default: 640x480)
- `--record`: Enable video recording
- `--no-heatmap`: Disable heatmap visualization
- `--heatmap-intensity`: Set heatmap intensity (0.1-0.9, default: 0.3)
- `--fps`: Frames per second to process (default: 1.0)
- `--no-gpu`: Disable GPU acceleration

Additional keyboard controls:
- `h` - Toggle heatmap on/off
- `+` - Increase heatmap intensity
- `-` - Decrease heatmap intensity
- `c` - Increase confidence threshold
- `v` - Decrease confidence threshold
- `n` - Increase NMS threshold
- `m` - Decrease NMS threshold

## Training Your Own Model

To train your own e-waste detection model:

1. Prepare your dataset:
   ```
   python organize_dataset.py --source dataset --output organized_dataset
   ```

2. Train the model:
   ```
   python train_model.py
   ```

## Requirements

The project requires the following dependencies:

- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- NumPy
- PIL (Pillow)

Install dependencies using:
```
pip install -r requirements.txt
```

## Performance Notes

- Processing rate is adjustable through command-line options
- The model is optimized for CPU usage but can utilize GPU when available
- Detection accuracy depends on model training quality and camera resolution

## Technology Overview

This project uses the following technologies:

- **TensorFlow**: For building and training the CNN model
- **MobileNetV2**: As the base model for efficient object detection
- **OpenCV**: For camera capture and image processing
- **Sliding Window Technique**: For detecting objects at different positions
- **Non-Maximum Suppression (NMS)**: For removing duplicate detections
- **Heatmap Visualization**: Using Gaussian distribution to highlight detection confidence

## Troubleshooting

- **Camera not detected**: Try different camera indices (0, 1, 2...)
- **Low detection accuracy**: Add more diverse training images and re-train
- **Slow performance**: Lower the resolution, reduce the FPS, or try simple mode
- **Model not found**: Ensure you've trained the model before running detection
- **Multiple detections of same object**: Increase NMS threshold or switch to advanced mode 