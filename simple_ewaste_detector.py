import cv2
import numpy as np
import os
import time
import argparse
from object_detection_model import EwasteObjectDetector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple E-waste detection from camera feed')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.85, 
                       help='Detection confidence threshold (default: 0.85)')
    parser.add_argument('--model', type=str, default='models/ewaste_detector.h5',
                       help='Path to trained model (default: models/ewaste_detector.h5)')
    parser.add_argument('--min-size', type=int, default=80,
                       help='Minimum detection size in pixels (default: 80)')
    args = parser.parse_args()

    # Load the trained model
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    # Initialize detector
    try:
        detector = EwasteObjectDetector(model_path=args.model)
        detector.confidence_threshold = args.threshold
        print("Model loaded successfully")
        print(f"Using confidence threshold: {args.threshold}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera index {args.camera}")
        print("Available cameras:")
        for i in range(10):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                print(f"  - Camera index {i}")
                temp_cap.release()
        return
        
    print("Starting e-waste detection...")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    # FPS calculation variables
    frame_times = []
    prev_time = time.time()
    
    while True:
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - prev_time
        prev_time = current_time
        
        # Keep last 30 frame times for smoothing
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        
        # Calculate FPS (prevent division by zero)
        if frame_times and sum(frame_times) > 0:
            fps = len(frame_times) / sum(frame_times)
        else:
            fps = 0
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
            
        # Process frame and detect e-waste objects
        result_frame, detections = detector.detect_objects(frame)
        
        # Filter out small detections (likely false positives)
        filtered_detections = []
        for detection in detections:
            box = detection['box']
            _, _, w, h = box
            
            # Skip detections that are too small
            if w < args.min_size or h < args.min_size:
                continue
                
            # Keep good detections
            filtered_detections.append(detection)
        
        # Add FPS information to the display
        cv2.putText(
            result_frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        
        # Add detection count in smaller text
        # cv2.putText(
        #     result_frame, f"Detections: {len(filtered_detections)}", (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2
        # )
        
        # Redraw only the filtered detections
        for detection in filtered_detections:
            box = detection['box']
            x, y, w, h = box
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Create label with confidence percentage
            label = f"E-waste: {confidence * 100:.1f}%"
            
            # Draw label background
            cv2.rectangle(result_frame, (x, y - 30), (x + len(label) * 9, y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(
                result_frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
        
        # Display the resulting frame with detections
        cv2.imshow('E-waste Detection', result_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit on 'q' press
            break
        elif key == ord('s'):
            # Save screenshot on 's' press
            screenshot_path = f"ewaste_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_path, result_frame)
            print(f"Screenshot saved to {screenshot_path}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main() 