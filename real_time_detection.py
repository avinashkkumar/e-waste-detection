import cv2
import numpy as np
import time
import argparse
import os
import tensorflow as tf
from object_detection_model import EwasteObjectDetector

# Terminal colors for better readability (Windows compatible)
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Enable GPU acceleration (works with integrated GPUs too)
def enable_gpu():
    try:
        # Configure TensorFlow to use the GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print(f"Found {len(physical_devices)} GPU(s)")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Enable mixed precision for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Enable XLA compilation for faster execution
            tf.config.optimizer.set_jit(True)
            
            print("GPU acceleration enabled")
            return True
        else:
            print("No GPU found, using CPU")
            return False
    except Exception as e:
        print(f"Error enabling GPU: {e}")
        print("Falling back to CPU")
        return False

def get_available_cameras():
    """
    Check for available camera devices
    Returns a list of available camera indices
    """
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras

def create_heatmap(frame, detections, alpha=0.3):
    """
    Create a proper heatmap overlay for detected e-waste items
    with reduced intensity to keep the original image visible
    """
    if not detections:
        return frame
    
    h, w = frame.shape[:2]
    
    # Create a heatmap layer (black background)
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Add heat to the detected regions with Gaussian distribution
    for detection in detections:
        box = detection['box']
        confidence = detection['confidence']
        
        # Extract box coordinates
        x, y, box_w, box_h = box
        
        # Create a gaussian gradient centered on the detection
        center_x, center_y = x + box_w // 2, y + box_h // 2
        
        # Make sigma proportional to the box size
        sigma_x = max(box_w // 3, 10)
        sigma_y = max(box_h // 3, 10)
        
        # Define region of influence (2.5*sigma - reduced spread)
        radius_x = 2.5 * sigma_x
        radius_y = 2.5 * sigma_y
        
        # Define the area to apply the gaussian (constrained to the detection area + margin)
        x1 = max(0, int(x - radius_x // 3))
        x2 = min(w, int(x + box_w + radius_x // 3))
        y1 = max(0, int(y - radius_y // 3))
        y2 = min(h, int(y + box_h + radius_y // 3))
        
        # Apply gaussian gradient (only to the detected area plus a small margin)
        for cy in range(y1, y2):
            for cx in range(x1, x2):
                # Calculate distance from center (normalized by sigma)
                dx = (cx - center_x) / sigma_x
                dy = (cy - center_y) / sigma_y
                dist_sq = dx*dx + dy*dy
                
                # Apply gaussian function if within reasonable distance
                if dist_sq <= 6:  # Reduced from 9 to 6 (2.5-sigma radius)
                    # Gaussian function weighted by confidence
                    heat_val = confidence * np.exp(-0.7 * dist_sq)  # Steeper falloff
                    # Maximum value when overlapping
                    heatmap[cy, cx] = max(heatmap[cy, cx], heat_val)
    
    # Normalize to 0-255 range but slightly reduce max intensity
    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap) * 200).astype(np.uint8)  # Cap at 200 instead of 255
    else:
        heatmap = np.zeros((h, w), dtype=np.uint8)
    
    # Apply colormap (COLORMAP_JET: blue->green->yellow->red)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Draw bounding boxes and confidence scores
    result = frame.copy()
    for detection in detections:
        box = detection['box']
        x, y, box_w, box_h = box
        confidence = detection['confidence']
        
        # Draw white rectangle around detection
        cv2.rectangle(result, (x, y), (x + box_w, y + box_h), (255, 255, 255), 1)
        
        # Show confidence percentage
        conf_text = f"{confidence*100:.1f}%"
        cv2.putText(result, conf_text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend the heatmap with the original frame (reduced alpha)
    result = cv2.addWeighted(result, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Add subtle heatmap indicator
    cv2.putText(result, "HEAT", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    return result

def improve_sliding_window(detector, frame, confidence_threshold=0.7, nms_iou_threshold=0.8):
    """
    Enhanced sliding window detection with sparse sampling to avoid duplicate detections
    """
    h, w = frame.shape[:2]
    detections = []
    
    # Use fewer window sizes for better coverage without excessive overlap
    window_sizes = [(300, 300), (400, 400)]
    
    # Use larger step size to reduce duplicates
    overlap = 0.5  # 50% overlap between windows
    
    for win_size in window_sizes:
        win_w, win_h = win_size
        
        # Calculate step size based on overlap
        step_x = max(50, int(win_w * (1 - overlap)))  # Significantly larger step size
        step_y = max(50, int(win_h * (1 - overlap)))
        
        windows = []
        positions = []
        
        # Extract windows with larger steps
        for y in range(0, h - win_h + 1, step_y):
            for x in range(0, w - win_w + 1, step_x):
                # Extract window
                window = frame[y:y + win_h, x:x + win_w]
                
                # Skip if window is too small
                if window.shape[0] < 50 or window.shape[1] < 50:
                    continue
                
                # Preprocess window
                processed = detector.preprocess_image(window)
                windows.append(processed[0])  # Extract from batch dimension
                positions.append((x, y, win_w, win_h))
        
        if not windows:
            continue
            
        # Convert to batch for efficient processing
        batch = np.array(windows)
        
        # Get predictions for entire batch
        predictions = detector.model.predict(batch, verbose=0)
        
        # Process predictions
        for i, prediction in enumerate(predictions):
            confidence = prediction[1]  # Class 1 is e-waste
            
            if confidence > confidence_threshold:
                detection = {
                    'box': positions[i],
                    'confidence': float(confidence)
                }
                detections.append(detection)
    
    # Apply stronger non-maximum suppression to remove duplicates
    return advanced_nms(detections, nms_iou_threshold)

def advanced_nms(detections, iou_threshold=0.8):
    """
    Advanced non-maximum suppression to ensure only one box per object
    """
    if len(detections) == 0:
        return []
        
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # First, cluster detections that significantly overlap
    clusters = []
    processed = [False] * len(detections)
    
    for i in range(len(detections)):
        if processed[i]:
            continue
            
        current_cluster = [i]
        processed[i] = True
        
        for j in range(i + 1, len(detections)):
            if not processed[j]:
                if calculate_iou(detections[i]['box'], detections[j]['box']) > 0.3:
                    # Add to current cluster if it overlaps significantly
                    current_cluster.append(j)
                    processed[j] = True
                    
        clusters.append(current_cluster)
    
    # For each cluster, select the best box
    final_detections = []
    for cluster in clusters:
        if not cluster:
            continue
            
        # Choose box with highest confidence
        best_idx = cluster[0]  # Already sorted by confidence
        best_detection = detections[best_idx]
        
        # If the cluster has multiple detections, try to find the most representative box size
        if len(cluster) > 1:
            # Calculate average size of all boxes in the cluster
            avg_width = sum(detections[idx]['box'][2] for idx in cluster) / len(cluster)
            avg_height = sum(detections[idx]['box'][3] for idx in cluster) / len(cluster)
            
            # Find the box closest to average size with high confidence
            closest_to_avg = best_idx
            min_size_diff = float('inf')
            
            for idx in cluster:
                if detections[idx]['confidence'] > 0.85 * best_detection['confidence']:
                    width = detections[idx]['box'][2]
                    height = detections[idx]['box'][3]
                    size_diff = abs(width - avg_width) + abs(height - avg_height)
                    
                    if size_diff < min_size_diff:
                        min_size_diff = size_diff
                        closest_to_avg = idx
            
            # Use the box that's closest to average size with good confidence
            best_detection = detections[closest_to_avg]
            
        final_detections.append(best_detection)
    
    return final_detections

def calculate_iou(box1, box2):
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

def show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold):
    """Display current settings in terminal"""
    # Clear several lines in terminal
    print("\033[H\033[J", end="")  # Clear screen
    
    print(f"{TermColors.HEADER}===== E-WASTE DETECTION CONTROLS ====={TermColors.ENDC}")
    print(f"{TermColors.BOLD}CURRENT SETTINGS:{TermColors.ENDC}")
    print(f"  Resolution: {args.resolution}")
    print(f"  FPS: {args.fps}")
    print(f"  Heatmap: {TermColors.GREEN + 'ON' + TermColors.ENDC if show_heatmap else TermColors.RED + 'OFF' + TermColors.ENDC}")
    if show_heatmap:
        print(f"  Heatmap Intensity: {TermColors.YELLOW}{heatmap_intensity:.1f}{TermColors.ENDC}")
    print(f"  Confidence Threshold: {TermColors.YELLOW}{confidence_threshold:.2f}{TermColors.ENDC}")
    print(f"  NMS Threshold: {nms_threshold:.2f}")
    
    print(f"\n{TermColors.BOLD}KEYBOARD CONTROLS:{TermColors.ENDC}")
    print(f"  {TermColors.BLUE}h{TermColors.ENDC} - Toggle heatmap on/off")
    print(f"  {TermColors.BLUE}+{TermColors.ENDC} - Increase heatmap intensity")
    print(f"  {TermColors.BLUE}-{TermColors.ENDC} - Decrease heatmap intensity")
    print(f"  {TermColors.BLUE}c{TermColors.ENDC} - Increase confidence threshold")
    print(f"  {TermColors.BLUE}v{TermColors.ENDC} - Decrease confidence threshold")
    print(f"  {TermColors.BLUE}n{TermColors.ENDC} - Increase NMS threshold (fewer duplicates)")
    print(f"  {TermColors.BLUE}m{TermColors.ENDC} - Decrease NMS threshold (more detections)")
    print(f"  {TermColors.BLUE}s{TermColors.ENDC} - Save screenshot")
    print(f"  {TermColors.BLUE}q{TermColors.ENDC} - Quit")
    
    print(f"\n{TermColors.HEADER}===================================={TermColors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description='Real-time E-waste detection from camera feed')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use')
    parser.add_argument('--model', type=str, default='models/ewaste_detector.h5', help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.75, help='Detection confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.8, help='NMS threshold for duplicate removal')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution (WxH)')
    parser.add_argument('--record', action='store_true', help='Record video output')
    parser.add_argument('--no-heatmap', action='store_true', help='Disable heatmap visualization')
    parser.add_argument('--heatmap-intensity', type=float, default=0.3, help='Heatmap intensity (0.1-0.9)')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to process (default: 1.0)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    args = parser.parse_args()
    
    # Enable Windows terminal colors
    os.system('color')
    
    # Enable GPU if available and not disabled
    if not args.no_gpu:
        using_gpu = enable_gpu()
    else:
        using_gpu = False
        print("GPU acceleration disabled by user")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("You need to train the model first using train_model.py")
        return
    
    # Check available cameras
    available_cameras = get_available_cameras()
    if not available_cameras:
        print("Error: No cameras detected")
        return
    
    if args.camera not in available_cameras:
        print(f"Warning: Camera index {args.camera} not available")
        camera_id = available_cameras[0]
        print(f"Using camera index {camera_id} instead")
    else:
        camera_id = args.camera
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except:
        width, height = 640, 480
        print(f"Invalid resolution format. Using default {width}x{height}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize the detector
    try:
        detector = EwasteObjectDetector(model_path=args.model)
        detector.confidence_threshold = args.threshold
    except Exception as e:
        print(f"Error initializing detector: {e}")
        cap.release()
        return
    
    # Initialize video writer if recording
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = f"ewaste_detection_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (width, height))
        print(f"Recording video to {output_path}")
    
    # Calculate frame interval based on requested FPS
    frame_interval = 1.0 / args.fps if args.fps > 0 else 1.0
    
    # Set initial values
    show_heatmap = not args.no_heatmap
    heatmap_intensity = min(0.9, max(0.1, args.heatmap_intensity))
    confidence_threshold = args.threshold
    nms_threshold = args.nms_threshold
    
    # Show initial settings
    show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
    
    # Loop for real-time detection
    last_process_time = 0
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            current_time = time.time()
            # Process frame only if enough time has elapsed
            if current_time - last_process_time >= frame_interval:
                # Use improved sliding window detection with better NMS
                detections = improve_sliding_window(detector, frame, 
                                                   confidence_threshold=confidence_threshold,
                                                   nms_iou_threshold=nms_threshold)
                
                # Draw results on a copy of the frame
                result_frame = frame.copy()
                
                # Apply heatmap if enabled
                if show_heatmap:
                    result_frame = create_heatmap(result_frame, detections, alpha=heatmap_intensity)
                else:
                    # If no heatmap, just draw bounding boxes
                    for detection in detections:
                        box = detection['box']
                        x, y, w, h = box
                        confidence = detection['confidence']
                        
                        # Draw bounding box
                        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Show confidence percentage
                        conf_text = f"{confidence*100:.1f}%"
                        cv2.putText(result_frame, conf_text, (x, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display stats on the frame
                cv2.putText(result_frame, f"FPS: {args.fps:.1f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(result_frame, f"Det: {len(detections)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(result_frame, f"Conf: {confidence_threshold:.2f}", (width - 120, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Display the resulting frame
                cv2.imshow('E-waste Detection', result_frame)
                
                # Record video if enabled
                if video_writer is not None:
                    video_writer.write(result_frame)
                
                # Update last process time
                last_process_time = current_time
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('h'):
                # Toggle heatmap
                show_heatmap = not show_heatmap
                show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
            elif key == ord('+') or key == ord('='):
                # Increase heatmap intensity
                heatmap_intensity = min(0.9, heatmap_intensity + 0.1)
                show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
            elif key == ord('-'):
                # Decrease heatmap intensity
                heatmap_intensity = max(0.1, heatmap_intensity - 0.1)
                show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
            elif key == ord('c'):
                # Increase confidence threshold
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
            elif key == ord('v'):
                # Decrease confidence threshold
                confidence_threshold = max(0.25, confidence_threshold - 0.05)
                show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
            elif key == ord('n'):
                # Increase NMS threshold (fewer duplicates)
                nms_threshold = min(0.9, nms_threshold + 0.05)
                show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
            elif key == ord('m'):
                # Decrease NMS threshold (more detections)
                nms_threshold = max(0.1, nms_threshold - 0.05)
                show_current_settings(args, show_heatmap, heatmap_intensity, confidence_threshold, nms_threshold)
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"ewaste_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, result_frame)
                print(f"Screenshot saved to {screenshot_path}")
    
    except Exception as e:
        print(f"Error during detection: {e}")
    
    finally:
        # When everything is done, release the capture
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Detection stopped")

if __name__ == "__main__":
    main() 