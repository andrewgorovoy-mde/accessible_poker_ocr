"""
Real-time Playing Cards Detection using YOLOv8 and Webcam
Based on: https://github.com/PD-Mera/Playing-Cards-Detection

This script captures video from a webcam and uses a pre-trained YOLOv8 model
to detect playing cards in real-time. It displays bounding boxes around detected
cards and shows confidence scores for each detection.

Author: Based on Playing-Cards-Detection repository
Purpose: Real-time object detection for playing cards using computer vision
"""

# Import argparse module - allows us to parse command-line arguments
# This lets users customize the script behavior when running it
import argparse

# Import os module - provides functions for interacting with the operating system
# We'll use it to check if files exist on the filesystem
import os

# Import collections module - provides Counter for counting occurrences
from collections import Counter

# Try to import cv2 (OpenCV library) - this is used for video capture and image processing
# We wrap it in a try-except block to handle the case where the library isn't installed
try:
    import cv2
except ImportError:
    # If cv2 is not installed, print an error message explaining how to fix it
    print("Error: opencv-python is not installed.")
    print("Please install it using:")
    print("  pip install opencv-python")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    # exit(1) terminates the program with an error code (1 means failure)
    exit(1)

# Try to import numpy - used for array operations in image processing
# We'll use it for contour and coordinate calculations
try:
    import numpy as np
except ImportError:
    print("Error: numpy is not installed.")
    print("Please install it using:")
    print("  pip install numpy")
    exit(1)

# Try to import YOLO from ultralytics - this is the machine learning model we'll use
# YOLO (You Only Look Once) is a popular object detection algorithm
try:
    from ultralytics import YOLO
except ImportError:
    # If ultralytics is not installed, print an error message explaining how to fix it
    print("Error: ultralytics is not installed.")
    print("Please install it using:")
    print("  pip install ultralytics")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    # exit(1) terminates the program with an error code (1 means failure)
    exit(1)


def merge_duplicate_card_detections(boxes, frame_shape):
    """
    Merges duplicate card detections that result from detecting corner elements separately.
    
    Key insight: In a standard deck of playing cards, there cannot be duplicate cards.
    If the same card class (e.g., "Ace of Spades") is detected multiple times, 
    those detections must be from the same physical card (detected at different corners).
    
    This function:
    1. Groups detections by card class
    2. If a class appears multiple times, merges all detections of that class into one
    3. Single detections (unique classes) are kept as-is
    
    Args:
        boxes: List of detection boxes from YOLO results (each box has .cls, .conf, .xyxy)
        frame_shape: Tuple of (height, width) of the frame (not used, kept for compatibility)
    
    Returns:
        merged_boxes: List of merged boxes, where each box represents a single card
        Each merged box contains: class_id, confidence, merged_bbox [x1, y1, x2, y2], source
    """
    if boxes is None or len(boxes) == 0:
        return []
    
    # Convert boxes to a list of dictionaries and group by class_id
    detections_by_class = {}
    
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        
        # Group detections by class_id
        if class_id not in detections_by_class:
            detections_by_class[class_id] = []
        
        detections_by_class[class_id].append({
            'index': i,
            'class_id': class_id,
            'confidence': confidence,
            'bbox': bbox
        })
    
    merged_results = []
    
    # Process each card class
    for class_id, detections in detections_by_class.items():
        if len(detections) == 1:
            # Single detection - keep as is (no duplicates)
            merged_results.append({
                'class_id': class_id,
                'confidence': detections[0]['confidence'],
                'bbox': detections[0]['bbox'],
                'source': 'single'
            })
        else:
            # Multiple detections of the same card class - merge them
            # In a deck, there can only be one of each card, so these must be the same card
            
            # Merge all bounding boxes: take the union (min x1,y1 and max x2,y2)
            x1_min = min(d['bbox'][0] for d in detections)
            y1_min = min(d['bbox'][1] for d in detections)
            x2_max = max(d['bbox'][2] for d in detections)
            y2_max = max(d['bbox'][3] for d in detections)
            
            # Use the highest confidence score from all detections
            max_confidence = max(d['confidence'] for d in detections)
            
            merged_results.append({
                'class_id': class_id,
                'confidence': max_confidence,
                'bbox': np.array([x1_min, y1_min, x2_max, y2_max]),
                'source': 'merged'
            })
    
    return merged_results


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2] coordinates
        bbox2: [x1, y1, x2, y2] coordinates
    
    Returns:
        IoU value between 0 and 1 (1 = perfect overlap, 0 = no overlap)
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's no intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate intersection area
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # Calculate IoU
    iou = inter_area / union_area
    return iou


def match_detections_to_tracked_cards(detections, tracked_cards, iou_threshold=0.3):
    """
    Match current frame detections to tracked cards from previous frames.
    
    Args:
        detections: List of current frame detections (with 'bbox', 'class_id', 'confidence')
        tracked_cards: List of tracked cards, each with 'id', 'predictions', 'last_bbox'
        iou_threshold: Minimum IoU to consider a match (default: 0.3)
    
    Returns:
        matches: List of (detection_index, tracked_card_index) tuples
        unmatched_detections: List of detection indices that don't match any tracked card
        unmatched_tracked: List of tracked card indices that don't match any detection
    """
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracked = list(range(len(tracked_cards)))
    
    # Calculate IoU matrix between all detections and tracked cards
    iou_matrix = []
    for i, det in enumerate(detections):
        iou_row = []
        for j, tracked in enumerate(tracked_cards):
            iou = calculate_iou(det['bbox'], tracked['last_bbox'])
            iou_row.append(iou)
        iou_matrix.append(iou_row)
    
    # Greedy matching: match highest IoU pairs first
    # Sort all potential matches by IoU (descending)
    potential_matches = []
    for i in range(len(detections)):
        for j in range(len(tracked_cards)):
            if iou_matrix[i][j] >= iou_threshold:
                potential_matches.append((iou_matrix[i][j], i, j))
    
    # Sort by IoU (highest first)
    potential_matches.sort(reverse=True, key=lambda x: x[0])
    
    # Match greedily (each detection and tracked card can only match once)
    matched_det_indices = set()
    matched_tracked_indices = set()
    
    for iou, det_idx, tracked_idx in potential_matches:
        if det_idx not in matched_det_indices and tracked_idx not in matched_tracked_indices:
            matches.append((det_idx, tracked_idx))
            matched_det_indices.add(det_idx)
            matched_tracked_indices.add(tracked_idx)
    
    # Find unmatched detections and tracked cards
    unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
    unmatched_tracked = [i for i in range(len(tracked_cards)) if i not in matched_tracked_indices]
    
    return matches, unmatched_detections, unmatched_tracked


def detect_card_outline(image, bbox, padding=10):
    """
    Detects the actual outline/contour of a card within a bounding box region.
    
    This function:
    1. Extracts the region of interest (ROI) from the bounding box
    2. Applies edge detection to find card edges
    3. Finds contours (outline shapes) in the region
    4. Filters for the best card-like contour (quadrilateral)
    5. Returns the contour points and a mask for the card
    
    Args:
        image: Full frame image (numpy array)
        bbox: Bounding box coordinates [x1, y1, x2, y2] from YOLO detection
        padding: Extra pixels around bbox to include (default: 10)
    
    Returns:
        contour: Detected card outline points (numpy array) or None if not found
        card_mask: Binary mask of the card region (numpy array) or None
    """
    # Extract coordinates from bounding box
    # bbox is typically [x1, y1, x2, y2] format
    x1, y1, x2, y2 = bbox
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Add padding and ensure coordinates are within image bounds
    # max(0, ...) ensures we don't go negative
    # min(w/h, ...) ensures we don't exceed image boundaries
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    
    # Extract the region of interest (ROI) from the original image
    # This crops the image to just the area around the detected card
    roi = image[y1:y2, x1:x2]
    
    # Check if ROI is valid (has dimensions)
    if roi.size == 0:
        return None, None
    
    # Convert to grayscale for edge detection
    # cv2.cvtColor converts color images (BGR) to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    # This smooths the image to make edge detection more reliable
    # (5, 5) is the kernel size, 0 is automatic standard deviation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    # This creates a binary image (black/white) by separating card from background
    # ADAPTIVE_THRESH_GAUSSIAN_C uses a Gaussian-weighted sum of neighborhood
    # THRESH_BINARY_INV inverts so card is white, background is black
    # 11 is block size for local threshold calculation
    # 2 is constant subtracted from mean
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Alternative: Use Canny edge detection
    # Canny finds edges in the image using gradient detection
    # 50 and 150 are the low and high thresholds for edge detection
    # Lower threshold = more edges detected, higher = fewer but stronger edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Combine both edge detection methods for better results
    # Bitwise OR combines both binary images
    combined = cv2.bitwise_or(binary, edges)
    
    # Dilate the edges to close gaps in the outline
    # This makes the card outline more continuous
    # kernel is a small matrix used for the dilation operation
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=2)
    
    # Find contours (outline shapes) in the processed image
    # RETR_EXTERNAL only finds outer contours (not holes inside)
    # CHAIN_APPROX_SIMPLE compresses the contour to save memory
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return None
    if not contours:
        return None, None
    
    # Find the largest contour (most likely to be the card)
    # Cards should be the largest object in the ROI
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate contour area
    area = cv2.contourArea(largest_contour)
    
    # Filter out very small contours (likely noise)
    # Require at least 5% of ROI area
    roi_area = (x2 - x1) * (y2 - y1)
    if area < 0.05 * roi_area:
        return None, None
    
    # Approximate the contour to a simpler polygon
    # This reduces the number of points and makes the shape more regular
    # epsilon is the maximum distance from original contour to approximation
    # True means the contour is closed
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Adjust contour coordinates back to full image coordinates
    # The contour is relative to the ROI, so we need to add (x1, y1) offset
    adjusted_contour = approx + [x1, y1]
    
    # Create a mask of the card region for optional use
    card_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(card_mask, [adjusted_contour], -1, 255, -1)
    
    return adjusted_contour, card_mask


def main():
    """
    Main function that orchestrates the entire card detection process.
    
    This function:
    1. Sets up command-line arguments so users can customize behavior
    2. Loads the YOLOv8 machine learning model
    3. Initializes the webcam
    4. Continuously captures frames, runs detection, and displays results
    5. Handles user input (quit, save frame)
    6. Cleans up resources when done
    """
    
    # Create an ArgumentParser object to handle command-line arguments
    # This allows users to pass options like --model, --camera, etc. when running the script
    parser = argparse.ArgumentParser(description='Real-time Playing Cards Detection with Webcam')
    
    # Add --model argument: specifies the path to the model file
    # type=str means it expects a text string
    # default='./yolov8s_playing_cards.pt' means if user doesn't provide this, use this default path
    # help='...' provides description text when user runs --help
    parser.add_argument('--model', type=str, default='./yolov8s_playing_cards.pt',
                        help='Path to the trained YOLOv8 model weights')
    
    # Add --camera argument: specifies which camera to use (0 = first camera, 1 = second, etc.)
    # type=int means it expects a whole number
    # default=0 means use the first camera by default
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    
    # Add --conf argument: sets the confidence threshold for detections
    # type=float means it expects a decimal number (like 0.25)
    # Only detections with confidence >= this value will be shown
    # Lower values (like 0.1) show more detections but may include false positives
    # Higher values (like 0.5) show fewer detections but are more reliable
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    
    # Add --imgsz argument: sets the image size for the model to process
    # type=int means it expects a whole number
    # Larger values (like 640) = better accuracy but slower processing
    # Smaller values (like 320) = faster processing but may miss small cards
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference (default: 640)')
    
    # Add --outline argument: enables card outline detection
    # type=bool, action='store_true' means it's a flag (on/off)
    # If user includes --outline, it will be True, otherwise False
    parser.add_argument('--outline', action='store_true',
                        help='Detect and draw card outlines (contours) instead of bounding boxes')
    
    # Parse all the command-line arguments the user provided
    # This converts the raw command-line input into a Python object (args)
    # We can then access args.model, args.camera, etc. in the code
    args = parser.parse_args()
    
    # Check if the model file exists on the filesystem before trying to load it
    # os.path.exists() returns True if the file exists, False otherwise
    # This prevents errors later when we try to load a non-existent file
    if not os.path.exists(args.model):
        # If the file doesn't exist, print an error message
        # f"...{args.model}..." is an f-string that inserts the variable value into the text
        print(f"Error: Model file not found at {args.model}")
        print("Please download the model from:")
        print("https://drive.google.com/file/d/1AqZnW6dI6flFZvGxAn6A9apDNSviXZ5f/view?usp=share_link")
        print("\nOr train your own model following the repository instructions.")
        # return exits the function early - we can't continue without a model
        return
    
    # Load the YOLOv8 model from the file path
    # This reads the pre-trained weights from the .pt file
    # The model contains the neural network architecture and learned parameters
    print(f"Loading model from {args.model}...")
    try:
        # YOLO() creates a YOLO model object and loads the weights from the file
        # The model object can then be used to make predictions on images
        model = YOLO(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        # If loading fails (corrupted file, wrong format, etc.), catch the error
        # e contains the error message
        print(f"Error loading model: {e}")
        # return exits the function - we can't continue without a loaded model
        return
    
    # Initialize webcam/camera for video capture
    # cv2.VideoCapture() creates a VideoCapture object that connects to a camera
    # args.camera is the camera index (0 = first camera, 1 = second camera, etc.)
    # On most laptops, 0 is the built-in webcam
    print(f"Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    # Check if the camera was successfully opened
    # cap.isOpened() returns True if camera is accessible, False otherwise
    # This could fail if camera is in use by another app, doesn't exist, or permissions denied
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        print("Troubleshooting:")
        print("  1. Try a different camera index: --camera 1 or --camera 2")
        print("  2. Check if camera is being used by another application")
        print("  3. On macOS: Grant camera permissions in System Preferences")
        print("  4. On Linux: Check video device permissions (/dev/video*)")
        print("  5. Try restarting the application or your computer")
        # return exits the function - we can't continue without a camera
        return
    
    # Set camera properties to optimize video quality and performance
    # These are "requests" to the camera - the camera may ignore them if it doesn't support them
    
    # Set frame width to 1280 pixels (HD width)
    # This makes the video wider and provides more detail for detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    
    # Set frame height to 720 pixels (HD height)
    # This makes the video taller and provides more detail for detection
    # Together with width, this gives us 1280x720 resolution (720p HD)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Set frames per second (FPS) to 30
    # This requests 30 frames per second capture rate
    # Higher FPS = smoother video but more processing required
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Allow camera to warm up - some cameras need time to initialize
    # Read a few frames to allow the camera to stabilize
    print("Warming up camera...")
    import time
    for i in range(5):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)
    
    # Verify we can actually read frames
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Error: Camera opened but cannot read frames")
        print("Troubleshooting:")
        print("  1. Camera may be in use by another application - close other apps using the camera")
        print("  2. Try a different camera index with --camera flag")
        print("  3. Check camera permissions in system settings")
        print("  4. Try unplugging and replugging USB cameras")
        cap.release()
        return
    
    print("Camera ready!")
    
    # Print instructions for the user
    print("\nStarting video stream...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press SPACEBAR to start/stop collecting predictions for statistical voting")
    if args.outline:
        print("Card outline detection: ENABLED")
    else:
        print("Card outline detection: DISABLED (use --outline to enable)")
    # "-" * 50 creates a line of 50 dashes for visual separation
    print("-" * 50)
    
    # Initialize counters to track statistics
    # frame_count: counts how many frames we've processed (for display)
    frame_count = 0
    # saved_count: counts how many frames we've saved (for unique filenames)
    saved_count = 0
    
    # Statistical voting system state
    collecting_predictions = False  # Whether we're currently collecting data
    tracked_cards = []  # List of tracked cards: each has 'id', 'predictions', 'last_bbox', 'frames_seen'
    next_card_id = 0  # ID counter for new tracked cards
    
    # Wrap the main loop in a try-except-finally block for proper error handling
    # try: code that might fail
    # except: what to do if an error occurs
    # finally: code that always runs, even if there's an error (cleanup)
    try:
        # Main processing loop - runs continuously until user quits
        # while True means "loop forever until break is called"
        while True:
            # Read a single frame from the camera
            # cap.read() returns two values:
            #   ret: boolean (True if frame was read successfully, False if error)
            #   frame: numpy array containing the image data (height x width x 3 colors)
            ret, frame = cap.read()
            
            # Check if we successfully captured a frame
            # If ret is False, the camera might be disconnected or there's an error
            if not ret:
                print("Error: Failed to capture frame")
                print("Camera may have disconnected or been closed by another application.")
                print("Attempting to reconnect...")
                
                # Try to reopen the camera
                cap.release()
                import time
                time.sleep(1)  # Wait a bit before retrying
                cap = cv2.VideoCapture(args.camera)
                
                if not cap.isOpened():
                    print("Failed to reconnect to camera. Exiting...")
                    break
                
                # Warm up again and get a valid frame
                ret = False
                for attempt in range(5):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print("Camera reconnected successfully!")
                        frame = test_frame
                        break
                    time.sleep(0.1)
                
                if not ret:
                    print("Could not read frames after reconnection. Exiting...")
                    break
                
                # Continue with the frame we just read (already assigned above)
            
            # Run inference (detection) on the current frame using the YOLO model
            # model.predict() analyzes the image and finds objects (playing cards)
            # It returns a list of results, with one result per image (we only have one image)
            results = model.predict(
                frame,              # The image/frame to analyze (numpy array)
                conf=args.conf,     # Confidence threshold - only show detections above this
                imgsz=args.imgsz,   # Resize image to this size before processing
                verbose=False       # Don't print model output to console (keep it clean)
            )
            
            # Create a copy of the frame for annotation
            # We'll draw on this copy, leaving the original frame unchanged
            annotated_frame = frame.copy()
            
            # Extract information about detected cards for display
            # Create an empty list to store card names and confidence scores
            detected_cards = []
            
            # Check if any boxes (detections) were found
            # results[0].boxes contains all the detected objects
            # If it's None, no cards were detected
            if results[0].boxes is not None:
                # Merge duplicate detections from corner elements into single cards
                # Key insight: If the same card is detected multiple times, they must be 
                # from the same physical card (corner elements). In a deck, there can't be duplicates.
                merged_detections = merge_duplicate_card_detections(
                    results[0].boxes, 
                    frame.shape
                )
                
                # If we're collecting predictions, track cards across frames and accumulate predictions
                if collecting_predictions:
                    # Match current detections to tracked cards from previous frames
                    matches, unmatched_dets, unmatched_tracked = match_detections_to_tracked_cards(
                        merged_detections, 
                        tracked_cards,
                        iou_threshold=0.3
                    )
                    
                    # Update matched tracked cards with new predictions
                    for det_idx, tracked_idx in matches:
                        detection = merged_detections[det_idx]
                        tracked_card = tracked_cards[tracked_idx]
                        
                        # Add prediction to this tracked card
                        tracked_card['predictions'].append(detection['class_id'])
                        tracked_card['last_bbox'] = detection['bbox']
                        tracked_card['frames_seen'] += 1
                        tracked_card['consecutive_misses'] = 0  # Reset miss counter
                    
                    # Create new tracked cards for unmatched detections
                    for det_idx in unmatched_dets:
                        detection = merged_detections[det_idx]
                        tracked_cards.append({
                            'id': next_card_id,
                            'predictions': [detection['class_id']],  # Start with first prediction
                            'last_bbox': detection['bbox'],
                            'frames_seen': 1,
                            'consecutive_misses': 0
                        })
                        next_card_id += 1
                    
                    # Handle unmatched tracked cards (cards that weren't detected this frame)
                    # Increment miss counter, but keep them for a few frames (they might be temporarily occluded)
                    for tracked_idx in unmatched_tracked:
                        tracked_cards[tracked_idx]['consecutive_misses'] += 1
                        # Keep cards that have been missed for less than 5 consecutive frames
                        # This allows for temporary occlusions or detection failures
                        if tracked_cards[tracked_idx]['consecutive_misses'] >= 5:
                            # Mark for removal if missed too many frames
                            tracked_cards[tracked_idx]['frames_seen'] = 0
                    
                    # Remove tracked cards that have been missed for too long
                    tracked_cards = [tc for tc in tracked_cards if tc['frames_seen'] > 0]
                else:
                    # Not collecting - clear tracked cards
                    tracked_cards = []
                    next_card_id = 0
                
                # Loop through each merged detection (each card found)
                for detection in merged_detections:
                    # Get the class ID (which type of card it is)
                    # detection is a dictionary from merge_duplicate_card_detections
                    class_id = detection['class_id']
                    
                    # Get the confidence score (how sure the model is)
                    # This is already a float from the merge function
                    confidence = detection['confidence']
                    
                    # Convert class ID to human-readable name (e.g., "Ace of Spades")
                    # model.names is a dictionary mapping IDs to names
                    # model.names[class_id] looks up the name for this ID
                    class_name = model.names[class_id]
                    
                    # Add this card info to our list as a formatted string
                    # f"{class_name} ({confidence:.2f})" creates text like "Ace_of_Spades (0.95)"
                    # .2f means format the float to 2 decimal places
                    # Add indicator if this was merged from two detections
                    merge_indicator = " [merged]" if detection['source'] == 'merged' else ""
                    detected_cards.append(f"{class_name} ({confidence:.2f}){merge_indicator}")
                    
                    # Get bounding box coordinates
                    # detection['bbox'] is already [x1, y1, x2, y2] as numpy array
                    bbox = detection['bbox']
                    
                    # If outline detection is enabled, detect and draw card outline
                    if args.outline:
                        # Detect the card outline/contour
                        contour, _ = detect_card_outline(frame, bbox)
                        
                        if contour is not None:
                            # Draw the detected card outline
                            # cv2.drawContours draws the contour on the image
                            # -1 means draw all contours in the list
                            # (0, 255, 0) is green color in BGR format
                            # 2 is the line thickness
                            cv2.drawContours(annotated_frame, [contour], -1, (0, 255, 0), 2)
                            
                            # Draw corner points of the card for better visibility
                            # This highlights the four corners of the detected card
                            for point in contour:
                                x, y = point[0]
                                # Draw a small circle at each corner
                                cv2.circle(annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)
                            
                            # Calculate center point of the card for label placement
                            # M is the moment, which helps calculate center
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                            else:
                                # Fallback: use bbox center if moments fail
                                cx = int((bbox[0] + bbox[2]) / 2)
                                cy = int((bbox[1] + bbox[3]) / 2)
                            
                            # Draw label with card name and confidence
                            label = f"{class_name} {confidence:.2f}"
                            # Calculate text size for background rectangle
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            # Draw semi-transparent background for better text visibility
                            overlay = annotated_frame.copy()
                            cv2.rectangle(
                                overlay,
                                (cx - text_width // 2 - 5, cy - text_height - baseline - 5),
                                (cx + text_width // 2 + 5, cy + baseline + 5),
                                (0, 0, 0),
                                -1
                            )
                            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                            # Draw the text
                            cv2.putText(
                                annotated_frame,
                                label,
                                (cx - text_width // 2, cy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
                        else:
                            # If outline detection failed, fall back to bounding box
                            # Draw bounding box
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Draw label
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
                
                # If outline detection is not enabled, draw bounding boxes for merged detections
                # This ensures we always use merged detections, not the original YOLO boxes
                if not args.outline:
                    # Draw bounding boxes for each merged detection
                    for detection in merged_detections:
                        class_id = detection['class_id']
                        confidence = detection['confidence']
                        class_name = model.names[class_id]
                        bbox = detection['bbox']
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        # Use blue color for merged detections, green for single detections
                        color = (255, 0, 0) if detection['source'] == 'merged' else (0, 255, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )
            
            # Create text to display on the screen showing detection info
            # Start with the count of detected cards
            info_text = f"Detected cards: {len(detected_cards)}"
            
            # If cards were detected, add their names to the text
            if detected_cards:
                # Join the first 3 card names with commas
                # detected_cards[:3] gets the first 3 items from the list
                # ', '.join() combines them with commas: "Card1, Card2, Card3"
                info_text += f" | {', '.join(detected_cards[:3])}"
                
                # If there are more than 3 cards, add "..." to indicate there are more
                if len(detected_cards) > 3:
                    info_text += "..."
            
            # Draw the info text on the annotated frame
            # cv2.putText() draws text on an image
            cv2.putText(
                annotated_frame,           # Image to draw on
                info_text,                 # Text to draw
                (10, 30),                  # Position (x, y) - top-left corner, 10px from left, 30px from top
                cv2.FONT_HERSHEY_SIMPLEX, # Font style (simple sans-serif font)
                0.7,                       # Font scale (size multiplier)
                (0, 255, 0),               # Color in BGR format (Blue, Green, Red) - (0,255,0) = green
                2                          # Line thickness (thickness of the text strokes)
            )
            
            # Add a frame counter display
            # Increment the frame counter each time we process a frame
            frame_count += 1
            # Create text showing the current frame number
            fps_text = f"Frame: {frame_count}"
            
            # Draw the frame counter on the image
            cv2.putText(
                annotated_frame,           # Image to draw on
                fps_text,                  # Text to draw
                (10, 60),                  # Position (x, y) - 10px from left, 60px from top (below the other text)
                cv2.FONT_HERSHEY_SIMPLEX, # Font style
                0.7,                       # Font scale
                (255, 255, 255),           # Color in BGR format - (255,255,255) = white
                2                          # Line thickness
            )
            
            # Display collection status
            if collecting_predictions:
                status_text = f"COLLECTING: {len(tracked_cards)} cards tracked"
                total_predictions = sum(len(tc['predictions']) for tc in tracked_cards)
                status_text += f" | {total_predictions} predictions"
                
                # Draw status with red background to make it visible
                (text_width, text_height), baseline = cv2.getTextSize(
                    status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    annotated_frame,
                    (10, 90),
                    (15 + text_width, 95 + text_height + baseline),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    status_text,
                    (10, 95 + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            else:
                # Draw inactive status
                status_text = "Press SPACEBAR to collect predictions"
                cv2.putText(
                    annotated_frame,
                    status_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (128, 128, 128),
                    2
                )
            
            # Display the annotated frame in a window
            # cv2.imshow() creates a window and displays the image
            # 'Playing Cards Detection - Webcam' is the window title
            # annotated_frame is the image to display
            cv2.imshow('Playing Cards Detection - Webcam', annotated_frame)
            
            # Check for keyboard input from the user
            # cv2.waitKey(1) waits for 1 millisecond for a key press
            # & 0xFF masks the result to get only the lowest 8 bits (the actual key code)
            # This is needed because waitKey() returns a 32-bit value, but we only need the key code
            key = cv2.waitKey(1) & 0xFF
            
            # Check if the user pressed 'q' (quit)
            # ord('q') converts the character 'q' to its ASCII code
            # If the pressed key matches 'q', exit the loop
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            # Check if the user pressed 's' (save frame)
            elif key == ord('s'):
                # Save current frame to a file
                # Increment the saved counter to create unique filenames
                saved_count += 1
                # Create a filename with the counter (e.g., "captured_frame_1.jpg")
                filename = f"captured_frame_{saved_count}.jpg"
                # cv2.imwrite() saves the image to a file
                # filename is the path/filename, annotated_frame is the image data
                cv2.imwrite(filename, annotated_frame)
                # Print confirmation message
                print(f"Saved frame to {filename}")
            
            # Check if the user pressed SPACEBAR (toggle prediction collection)
            elif key == ord(' '):
                if collecting_predictions:
                    # Stop collecting and calculate final predictions using majority voting
                    collecting_predictions = False
                    print("\n" + "="*60)
                    print("FINAL PREDICTIONS (Majority Voting):")
                    print("="*60)
                    
                    if len(tracked_cards) == 0:
                        print("No cards were tracked.")
                    else:
                        # Calculate final predictions for each tracked card
                        final_predictions = []
                        for tracked_card in tracked_cards:
                            predictions = tracked_card['predictions']
                            
                            # Count occurrences of each class_id
                            class_counts = Counter(predictions)
                            
                            # Find the most common class (majority vote)
                            most_common_class, count = class_counts.most_common(1)[0]
                            total_count = len(predictions)
                            confidence_percentage = (count / total_count) * 100
                            
                            # Get card name
                            card_name = model.names[most_common_class]
                            
                            final_predictions.append({
                                'class_id': most_common_class,
                                'card_name': card_name,
                                'vote_count': count,
                                'total_predictions': total_count,
                                'confidence_percentage': confidence_percentage,
                                'all_predictions': class_counts
                            })
                        
                        # Sort by confidence (highest first)
                        final_predictions.sort(key=lambda x: x['confidence_percentage'], reverse=True)
                        
                        # Print results
                        for i, pred in enumerate(final_predictions, 1):
                            print(f"\nCard {i}: {pred['card_name']}")
                            print(f"  Final Prediction: {pred['card_name']} (Class ID: {pred['class_id']})")
                            print(f"  Confidence: {pred['confidence_percentage']:.1f}% ({pred['vote_count']}/{pred['total_predictions']} votes)")
                            print(f"  All predictions: {dict(pred['all_predictions'])}")
                        
                        print("\n" + "="*60)
                    
                    # Reset tracking
                    tracked_cards = []
                    next_card_id = 0
                else:
                    # Start collecting predictions
                    collecting_predictions = True
                    tracked_cards = []
                    next_card_id = 0
                    print("\nStarted collecting predictions...")
                    print("Position cards clearly and press SPACEBAR again when ready.")
    
    # Handle the case where user presses Ctrl+C to interrupt the program
    # KeyboardInterrupt is the exception raised when Ctrl+C is pressed
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # The finally block always runs, even if there's an error or break
    # This ensures we clean up resources properly
    finally:
        # Cleanup: Release the camera so other programs can use it
        # cap.release() disconnects from the camera and frees system resources
        cap.release()
        
        # Close all OpenCV windows that were opened
        # cv2.destroyAllWindows() closes any windows created by cv2.imshow()
        cv2.destroyAllWindows()
        
        # Print goodbye message
        print("\nCamera released. Goodbye!")


# This is the entry point of the script - code that runs when script is executed directly
# __name__ is a special Python variable that equals "__main__" when script is run directly
# If someone imports this file as a module, __name__ would be "camera_detect" instead
# This check ensures main() only runs when script is executed, not when imported
if __name__ == "__main__":
    # Call the main() function to start the program
    main()

