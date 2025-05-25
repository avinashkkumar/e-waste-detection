import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

class EwasteObjectDetector:
    def __init__(self, model_path='models/ewaste_detector.h5'):
        """
        Initialize the E-waste object detector
        """
        # Load the model
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.input_size = (224, 224)  # Input size expected by the model
        self.confidence_threshold = 0.65  # Minimum confidence to consider a detection valid
        
    def preprocess_image(self, image):
        """
        Preprocess an image for model input
        """
        # Resize to model's input size
        img = cv2.resize(image, self.input_size)
        img = img.astype('float32') / 255.0
        
        # Expand dimensions to create a batch of size 1
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_objects(self, frame):
        """
        Perform object detection on a frame
        Returns: original frame with detection boxes, list of detections
        """
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Initialize detections list
        detections = []
        
        # Sliding window detection for different regions
        window_sizes = [(300, 300), (400, 400)]
        overlap = 0.5  # 50% overlap between windows
        
        for win_size in window_sizes:
            win_w, win_h = win_size
            
            # Calculate step size based on overlap
            step_x = int(win_w * (1 - overlap))
            step_y = int(win_h * (1 - overlap))
            
            # Slide window across the image
            for y in range(0, h - win_h + 1, step_y):
                for x in range(0, w - win_w + 1, step_x):
                    # Extract window
                    window = frame[y:y + win_h, x:x + win_w]
                    
                    # Preprocess the window
                    processed_window = self.preprocess_image(window)
                    
                    # Get prediction
                    prediction = self.model.predict(processed_window, verbose=0)[0]
                    
                    # Class 1 is e-waste (based on our data preparation)
                    confidence = prediction[1]
                    
                    # If confidence is above threshold, add detection
                    if confidence > self.confidence_threshold:
                        # Store detection info
                        detection = {
                            'box': (x, y, win_w, win_h),
                            'confidence': float(confidence)
                        }
                        detections.append(detection)
        
        # Apply non-maximum suppression to remove overlapping boxes with stricter threshold
        final_detections = self.non_max_suppression(detections, 0.2)
        
        # Don't draw detections here, let the caller handle it 
        # for more flexibility in the main script
        
        return original_frame, final_detections
    
    def non_max_suppression(self, detections, iou_threshold=0.3):
        """
        Apply non-maximum suppression to remove overlapping bounding boxes
        """
        if len(detections) == 0:
            return []
            
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        while len(detections) > 0:
            # Take the detection with highest confidence
            best_detection = detections[0]
            final_detections.append(best_detection)
            
            # Remove the best detection from the list
            detections = detections[1:]
            
            # Filter out detections with IoU > threshold
            filtered_detections = []
            for detection in detections:
                # Skip if IoU is too high (i.e., significant overlap)
                if self.calculate_iou(best_detection['box'], detection['box']) < iou_threshold:
                    # Additional validation: reject detections with almost same position
                    # even if size is different (this helps with false positives)
                    box1 = best_detection['box']
                    box2 = detection['box']
                    
                    # Calculate center points
                    center1_x = box1[0] + box1[2] // 2
                    center1_y = box1[1] + box1[3] // 2
                    center2_x = box2[0] + box2[2] // 2
                    center2_y = box2[1] + box2[3] // 2
                    
                    # Calculate distance between centers
                    distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
                    
                    # Skip if centers are too close (relative to box sizes)
                    min_box_dim = min(box1[2], box1[3], box2[2], box2[3])
                    if distance > min_box_dim * 0.25:  # Centers should be at least 25% of box size apart
                        filtered_detections.append(detection)
                    
            detections = filtered_detections
            
            # Limit the total number of detections to prevent false positives
            if len(final_detections) >= 10:  # Cap at 10 detections maximum
                break
            
        return final_detections
    
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two boxes
        box format: (x, y, width, height)
        """
        # Convert to (x1, y1, x2, y2) format
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas of both boxes
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and confidence values on the frame
        """
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            
            x, y, w, h = box
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Create label with confidence percentage
            label = f"E-waste: {confidence * 100:.1f}%"
            
            # Draw label background
            cv2.rectangle(frame, (x, y - 30), (x + len(label) * 9, y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
        
        return frame

# Testing function
def test_detector_on_image(detector, image_path):
    """
    Test the detector on a single image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Detect objects
    result_image, detections = detector.detect_objects(image)
    
    # Display results
    print(f"Found {len(detections)} e-waste items")
    for i, detection in enumerate(detections):
        print(f"  Item {i+1}: Confidence {detection['confidence']*100:.1f}%")
    
    # Save and show result
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to {output_path}")
    
    # Display the image (if in interactive environment)
    cv2.imshow("E-waste Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Initialize detector
        detector = EwasteObjectDetector()
        
        # Test on an image if provided
        import sys
        if len(sys.argv) > 1:
            test_detector_on_image(detector, sys.argv[1])
        else:
            print("No test image provided. Usage: python object_detection_model.py <image_path>")
    except Exception as e:
        print(f"Error: {e}") 