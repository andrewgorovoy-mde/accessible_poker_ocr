"""
Poker Game Card Detection Tool using YOLOv8 and Webcam
Based on: https://github.com/PD-Mera/Playing-Cards-Detection

This script is designed for poker gameplay assistance. It captures video from a webcam
and uses a pre-trained YOLOv8 model to detect playing cards in real-time. It automatically
tracks poker betting rounds:
- FLOP: Detects and records the first 3 community cards
- TURN: Detects and records the 4th community card
- RIVER: Detects and records the 5th community card

The script displays bounding boxes around detected cards, shows confidence scores,
and stores the complete board in arrays for use in poker analysis tools.

Author: Based on Playing-Cards-Detection repository
Purpose: Real-time poker card detection and board tracking for poker game tools
"""

# Import argparse module - allows us to parse command-line arguments
# This lets users customize the script behavior when running it
import argparse

# Import os module - provides functions for interacting with the operating system
# We'll use it to check if files exist on the filesystem
import os

# Import time module - provides functions for working with time
# We'll use it to track how long we've been recording detections
import time

# Import collections module - provides specialized container datatypes
# We'll use Counter to count occurrences of each card during recording
from collections import Counter

# Import numpy module - provides numerical computing capabilities
# We'll use it for calculating bounding box distances and overlaps
import numpy as np

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
    parser = argparse.ArgumentParser(description='Poker Game Card Detection Tool - Tracks FLOP, TURN, and RIVER')
    
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
    
    # Add --record-time argument: how many seconds to record detections when new cards appear
    # type=float means it expects a decimal number (like 3.0)
    # This gives the system time to stabilize and get accurate card readings
    parser.add_argument('--record-time', type=float, default=3.0,
                        help='Seconds to record detections when cards appear (default: 3.0)')
    
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
        print("Try a different camera index with --camera flag")
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
    
    # Print instructions for the user
    print("\n" + "=" * 60)
    print("POKER GAME CARD DETECTION TOOL")
    print("=" * 60)
    print("Starting video stream...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("\nGame Flow:")
    print("  - Waiting for FLOP (3 cards)...")
    print("  - Then waiting for TURN (4th card)...")
    print("  - Then waiting for RIVER (5th card)...")
    print("-" * 60)
    
    # Initialize counters to track statistics
    # frame_count: counts how many frames we've processed (for display)
    frame_count = 0
    # saved_count: counts how many frames we've saved (for unique filenames)
    saved_count = 0
    
    # Poker game state tracking variables
    # prev_card_count: tracks how many cards were detected in the previous frame
    # This helps us detect when new cards are added to the table
    prev_card_count = 0
    
    # current_stage: tracks which poker stage we're currently in
    # "waiting" = waiting for next stage, "recording" = currently recording detections
    # "flop_done" = flop recorded, "turn_done" = turn recorded, "river_done" = all done
    current_stage = "waiting_flop"
    
    # recording_start_time: timestamp when we started recording the current stage
    # Used to determine when we've recorded for long enough
    recording_start_time = None
    
    # detection_buffer: list to store card detections during recording period
    # Each element is a list of card names detected in one frame
    # We'll use this to find the most common cards (most stable detection)
    detection_buffer = []
    
    # Store the final poker hand results
    # flop_cards: list/array of 3 cards from the flop
    flop_cards = []
    # turn_card: single card from the turn (4th card)
    turn_card = None
    # river_card: single card from the river (5th card)
    river_card = None
    # complete_board: array storing all 5 community cards [flop1, flop2, flop3, turn, river]
    # This will be populated once all cards are dealt
    complete_board = []
    
    # Get FPS to estimate recording duration in frames
    # This helps us know how many frames to record based on time
    fps = cap.get(cv2.CAP_PROP_FPS)
    # If FPS is invalid or 0, use a default of 30 FPS
    if fps <= 0:
        fps = 30.0
    # Calculate how many frames to record based on record_time
    # fps * args.record_time gives total frames needed
    frames_to_record = int(fps * args.record_time)
    
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
                # break exits the while loop
                break
            
            # Run inference (detection) on the current frame using the YOLO model
            # model.predict() analyzes the image and finds objects (playing cards)
            # It returns a list of results, with one result per image (we only have one image)
            results = model.predict(
                frame,              # The image/frame to analyze (numpy array)
                conf=args.conf,     # Confidence threshold - only show detections above this
                imgsz=args.imgsz,   # Resize image to this size before processing
                verbose=False       # Don't print model output to console (keep it clean)
            )
            
            # SIMPLE APPROACH: Use edge detection to find card boundaries
            # Then associate label detections within each card boundary
            # This ensures one card = one detection, regardless of label variability
            
            # Step 1: Find card boundaries using edge detection
            # Convert frame to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection to find edges
            # Thresholds: 50 (low), 150 (high) - adjust these if needed
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours (closed shapes) which should represent card boundaries
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find card-like shapes
            # Cards should be rectangular with reasonable size
            card_boundaries = []
            min_card_area = 5000  # Minimum area for a card (adjust based on camera resolution)
            max_card_area = 500000  # Maximum area for a card
            
            for contour in contours:
                # Calculate area of contour
                area = cv2.contourArea(contour)
                
                # Filter by size (cards should be within reasonable size range)
                if min_card_area < area < max_card_area:
                    # Get bounding rectangle of contour
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate aspect ratio (cards are roughly rectangular)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Cards typically have aspect ratio between 0.5 and 2.0
                    if 0.5 < aspect_ratio < 2.0:
                        # Store card boundary as (x1, y1, x2, y2)
                        card_boundaries.append((x, y, x + w, y + h))
            
            # Step 2: Collect all label detections from YOLO model
            label_detections = []
            
            # Check if any boxes (detections) were found
            # results[0].boxes contains all the detected objects
            if results[0].boxes is not None:
                # Loop through each detected box (each label/symbol detected)
                for box in results[0].boxes:
                    # Get bounding box coordinates
                    # box.xyxy[0] returns the box coordinates as [x1, y1, x2, y2]
                    box_coords = box.xyxy[0].tolist()
                    
                    # Get the class ID (which type of card it is)
                    class_id = int(box.cls[0])
                    
                    # Get the confidence score (how sure the model is)
                    confidence = float(box.conf[0])
                    
                    # Convert class ID to human-readable name (e.g., "Ace_of_Spades")
                    class_name = model.names[class_id]
                    
                    # Calculate center point of label detection
                    center_x = (box_coords[0] + box_coords[2]) / 2
                    center_y = (box_coords[1] + box_coords[3]) / 2
                    
                    # Store: (box_coords, class_name, confidence, center_x, center_y)
                    label_detections.append((box_coords, class_name, confidence, center_x, center_y))
            
            # Step 3: Associate label detections with card boundaries
            # For each card boundary, find all labels that fall within it
            # Then use the most common label for that card
            unique_cards = []
            detected_card_names = []
            detected_cards_display = []
            
            # Process each detected card boundary
            for card_x1, card_y1, card_x2, card_y2 in card_boundaries:
                # Find all label detections within this card boundary
                labels_in_card = []
                
                for box_coords, class_name, confidence, center_x, center_y in label_detections:
                    # Check if label center is within card boundary
                    if (card_x1 <= center_x <= card_x2 and 
                        card_y1 <= center_y <= card_y2):
                        labels_in_card.append((class_name, confidence))
                
                # If we found labels within this card, use the most common one
                if labels_in_card:
                    # Count occurrences of each card type within this card
                    card_type_counts = Counter([name for name, _ in labels_in_card])
                    # Get the most common card type (most frequently detected label)
                    most_common_card = card_type_counts.most_common(1)[0][0]
                    
                    # Get the highest confidence for this card type
                    max_confidence = max([conf for name, conf in labels_in_card if name == most_common_card])
                    
                    # Store this card with its card boundary
                    unique_cards.append(((card_x1, card_y1, card_x2, card_y2), most_common_card, max_confidence))
                    
                    # Add to lists for processing
                    detected_card_names.append(most_common_card)
                    detected_cards_display.append(f"{most_common_card} ({max_confidence:.2f})")
            
            # Get the current number of cards detected
            current_card_count = len(detected_card_names)
            
            # POKER GAME LOGIC: Detect stage transitions and record cards
            # Check if we're transitioning to a new stage (new cards detected)
            if current_stage == "waiting_flop" and current_card_count == 3 and prev_card_count != 3:
                # We just detected 3 cards for the first time - this is the FLOP!
                print(f"\n{'='*60}")
                print("FLOP DETECTED! Recording 3 cards...")
                print(f"{'='*60}")
                # Start recording state
                current_stage = "recording_flop"
                # Record the current time so we know when we started recording
                recording_start_time = time.time()
                # Clear the detection buffer to start fresh
                detection_buffer = []
            
            elif current_stage == "flop_done" and current_card_count == 4 and prev_card_count != 4:
                # We just detected 4 cards for the first time - this is the TURN!
                print(f"\n{'='*60}")
                print("TURN DETECTED! Recording 4th card...")
                print(f"{'='*60}")
                # Start recording state
                current_stage = "recording_turn"
                # Record the current time so we know when we started recording
                recording_start_time = time.time()
                # Clear the detection buffer to start fresh
                detection_buffer = []
            
            elif current_stage == "turn_done" and current_card_count == 5 and prev_card_count != 5:
                # We just detected 5 cards for the first time - this is the RIVER!
                print(f"\n{'='*60}")
                print("RIVER DETECTED! Recording 5th card...")
                print(f"{'='*60}")
                # Start recording state
                current_stage = "recording_river"
                # Record the current time so we know when we started recording
                recording_start_time = time.time()
                # Clear the detection buffer to start fresh
                detection_buffer = []
            
            # If we're in a recording state, collect detections
            if current_stage.startswith("recording"):
                # Store card names detected in this frame
                # Each card is already resolved to one detection per card boundary
                # We'll count occurrences across frames to handle variability
                frame_card_names = detected_card_names.copy()
                
                # Add this frame's card detections to the buffer
                # Each frame stores a list of card names (one per detected card boundary)
                detection_buffer.append(frame_card_names)
                
                # Check if we've recorded for long enough
                # time.time() gets current time, recording_start_time is when we started
                # Subtract to get elapsed time in seconds
                elapsed_time = time.time() - recording_start_time
                
                # If we've recorded for the specified duration, process the results
                if elapsed_time >= args.record_time:
                    # Process the detection buffer to find the most stable card identities
                    # Since we're using edge detection, each frame already has one card per boundary
                    # We just need to count occurrences across frames to handle variability
                    
                    # Collect all card names from all frames
                    all_cards = []
                    for frame_cards in detection_buffer:
                        # Add all cards from this frame to the list
                        all_cards.extend(frame_cards)
                    
                    # Count occurrences of each card type across all frames
                    # This handles model variability - if same card detected as different types,
                    # we use the most frequently detected type
                    card_counts = Counter(all_cards)
                    
                    # Get the most common cards
                    # most_common(N) returns the N most frequent cards as (card, count) tuples
                    if current_stage == "recording_flop":
                        # For flop, we need exactly 3 UNIQUE cards
                        # Since a deck has 52 unique cards, flop must have 3 different cards
                        # card_counts already contains unique resolved cards (one per spatial location)
                        # We need to select the 3 most stable cards (highest counts)
                        most_common = card_counts.most_common(10)  # Get top 10 to find 3 unique
                        
                        # Extract unique card names (ensure no duplicates)
                        # Since card_counts is already from resolved cards (one per location),
                        # each card should already be unique, but we verify anyway
                        flop_cards = []
                        seen_cards = set()
                        for card, count in most_common:
                            if card not in seen_cards:
                                flop_cards.append(card)
                                seen_cards.add(card)
                                # Stop once we have 3 unique cards
                                if len(flop_cards) == 3:
                                    break
                        
                        # If we don't have 3 unique cards, warn the user
                        if len(flop_cards) < 3:
                            print(f"\nWARNING: Only detected {len(flop_cards)} unique cards in flop!")
                            print(f"  Detected cards: {flop_cards}")
                            print(f"  All card counts: {dict(card_counts.most_common(10))}")
                        
                        # Convert to comma-separated string for display
                        flop_str = ", ".join(flop_cards)
                        print(f"\nFLOP RECORDED: {flop_str}")
                        print(f"  Flop array: {flop_cards}")
                        print(f"  Detection counts: {dict([(c, card_counts[c]) for c in flop_cards])}")
                        # Move to next stage: wait for turn
                        current_stage = "flop_done"
                        # Clear buffer for next recording
                        detection_buffer = []
                    
                    elif current_stage == "recording_turn":
                        # For turn, we need 1 NEW card (the 4th card)
                        # Since a deck has 52 unique cards, turn must be different from flop cards
                        # We need to find a card that's NOT in the flop
                        turn_card = None
                        
                        # Create a set of flop cards for fast lookup
                        flop_cards_set = set(flop_cards)
                        
                        # Loop through most common cards (sorted by frequency across frames)
                        most_common = card_counts.most_common(10)
                        
                        # Find the first card that's NOT in flop
                        for card, count in most_common:
                            if card not in flop_cards_set:
                                turn_card = card
                                break
                        
                        # If we didn't find a new card, warn the user
                        # This shouldn't happen if detection is working correctly
                        if turn_card is None:
                            print(f"\nWARNING: Could not find a new card for TURN!")
                            print(f"  Flop cards: {flop_cards}")
                            print(f"  Most common detected cards: {dict(most_common[:5])}")
                            # Try to use the most common card that's not in flop anyway
                            if most_common:
                                turn_card = most_common[0][0]
                                print(f"  Using most common card: {turn_card}")
                        
                        print(f"\nTURN RECORDED: {turn_card}")
                        if turn_card:
                            print(f"  Detection count: {card_counts[turn_card]}")
                        # Move to next stage: wait for river
                        current_stage = "turn_done"
                        # Clear buffer for next recording
                        detection_buffer = []
                    
                    elif current_stage == "recording_river":
                        # For river, we need 1 NEW card (the 5th card)
                        # Since a deck has 52 unique cards, river must be different from flop and turn
                        # We need to find a card that's NOT already recorded
                        river_card = None
                        
                        # Create a set of already recorded cards for quick lookup
                        recorded_cards = set(flop_cards)
                        if turn_card:
                            recorded_cards.add(turn_card)
                        
                        # Loop through most common cards (sorted by frequency across frames)
                        most_common = card_counts.most_common(10)
                        
                        # Find the first card that's NOT already recorded
                        for card, count in most_common:
                            if card not in recorded_cards:
                                river_card = card
                                break
                        
                        # If we didn't find a new card, warn the user
                        # This shouldn't happen if detection is working correctly
                        if river_card is None:
                            print(f"\nWARNING: Could not find a new card for RIVER!")
                            print(f"  Already recorded: {list(recorded_cards)}")
                            print(f"  Most common detected cards: {dict(most_common[:5])}")
                            # Try to use the most common card that's not already recorded anyway
                            if most_common:
                                river_card = most_common[0][0]
                                print(f"  Using most common card: {river_card}")
                        
                        print(f"\nRIVER RECORDED: {river_card}")
                        if river_card:
                            print(f"  Detection count: {card_counts[river_card]}")
                        # All cards are now recorded!
                        current_stage = "river_done"
                        # Clear buffer
                        detection_buffer = []
                        
                        # Create the complete board array with all 5 cards
                        # This stores flop + turn + river in order
                        complete_board = flop_cards.copy()  # Start with flop cards
                        if turn_card:
                            complete_board.append(turn_card)  # Add turn card
                        if river_card:
                            complete_board.append(river_card)  # Add river card
                        
                        # Print final summary
                        print(f"\n{'='*60}")
                        print("DEALING COMPLETE!")
                        print(f"{'='*60}")
                        print(f"FLOP: {', '.join(flop_cards)}")
                        print(f"TURN: {turn_card}")
                        print(f"RIVER: {river_card}")
                        print(f"\nFULL BOARD: {', '.join(complete_board)}")
                        print(f"COMPLETE BOARD ARRAY: {complete_board}")
                        print(f"{'='*60}\n")
            
            # Update previous card count for next frame comparison
            prev_card_count = current_card_count
            
            # Annotate the frame with detection results
            # Start with the original frame
            annotated_frame = frame.copy()
            
            # Draw card boundaries found by edge detection
            # These are the actual physical card outlines
            for card_x1, card_y1, card_x2, card_y2 in card_boundaries:
                # Draw rectangle around card boundary (green color)
                cv2.rectangle(annotated_frame, (card_x1, card_y1), (card_x2, card_y2), (0, 255, 0), 3)
            
            # Draw label detections and card names
            # For each unique card, draw its name on the card boundary
            for card_boundary, class_name, confidence in unique_cards:
                card_x1, card_y1, card_x2, card_y2 = card_boundary
                # Draw card name text on the card
                label_text = f"{class_name} ({confidence:.2f})"
                # Position text at top-left of card boundary
                cv2.putText(annotated_frame, label_text, (card_x1, card_y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Create text to display on the screen showing detection info
            # Start with the count of detected cards
            info_text = f"Detected cards: {current_card_count}"
            
            # If cards were detected, add their names to the text
            if detected_cards_display:
                # Join the first 3 card names with commas
                # detected_cards_display[:3] gets the first 3 items from the list
                # ', '.join() combines them with commas: "Card1, Card2, Card3"
                info_text += f" | {', '.join(detected_cards_display[:3])}"
                
                # If there are more than 3 cards, add "..." to indicate there are more
                if len(detected_cards_display) > 3:
                    info_text += "..."
            
            # Add poker stage information to the display
            # Show current stage status for user feedback
            stage_text = ""
            if current_stage == "waiting_flop":
                stage_text = "Status: Waiting for FLOP (3 cards)"
            elif current_stage == "recording_flop":
                # Show recording progress
                if recording_start_time:
                    elapsed = time.time() - recording_start_time
                    remaining = max(0, args.record_time - elapsed)
                    stage_text = f"Status: Recording FLOP... ({remaining:.1f}s remaining)"
                else:
                    stage_text = "Status: Recording FLOP..."
            elif current_stage == "flop_done":
                stage_text = f"Status: FLOP recorded! Waiting for TURN (4th card)"
                # Show recorded flop
                if flop_cards:
                    stage_text += f" | Flop: {', '.join(flop_cards)}"
            elif current_stage == "recording_turn":
                # Show recording progress
                if recording_start_time:
                    elapsed = time.time() - recording_start_time
                    remaining = max(0, args.record_time - elapsed)
                    stage_text = f"Status: Recording TURN... ({remaining:.1f}s remaining)"
                else:
                    stage_text = "Status: Recording TURN..."
            elif current_stage == "turn_done":
                stage_text = f"Status: TURN recorded! Waiting for RIVER (5th card)"
                # Show recorded flop and turn
                if flop_cards and turn_card:
                    stage_text += f" | Board: {', '.join(flop_cards)}, {turn_card}"
            elif current_stage == "recording_river":
                # Show recording progress
                if recording_start_time:
                    elapsed = time.time() - recording_start_time
                    remaining = max(0, args.record_time - elapsed)
                    stage_text = f"Status: Recording RIVER... ({remaining:.1f}s remaining)"
                else:
                    stage_text = "Status: Recording RIVER..."
            elif current_stage == "river_done":
                stage_text = "Status: DEALING COMPLETE!"
                # Show full board
                if complete_board:
                    stage_text += f" | Board: {', '.join(complete_board)}"
            
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
            
            # Draw the poker stage status text
            # This shows users what stage of the game we're in
            if stage_text:
                cv2.putText(
                    annotated_frame,           # Image to draw on
                    stage_text,                # Text to draw (poker stage info)
                    (10, 60),                  # Position (x, y) - 10px from left, 60px from top
                    cv2.FONT_HERSHEY_SIMPLEX, # Font style
                    0.7,                       # Font scale
                    (255, 255, 0),             # Color in BGR format - (255,255,0) = cyan/yellow
                    2                          # Line thickness
                )
            
            # Add a frame counter display
            # Increment the frame counter each time we process a frame
            frame_count += 1
            # Create text showing the current frame number
            fps_text = f"Frame: {frame_count}"
            
            # Draw the frame counter on the image
            # Position it below the stage text (at y=90 if stage_text exists, else y=60)
            y_position = 90 if stage_text else 60
            cv2.putText(
                annotated_frame,           # Image to draw on
                fps_text,                  # Text to draw
                (10, y_position),          # Position (x, y) - 10px from left, dynamically positioned
                cv2.FONT_HERSHEY_SIMPLEX, # Font style
                0.7,                       # Font scale
                (255, 255, 255),           # Color in BGR format - (255,255,255) = white
                2                          # Line thickness
            )
            
            # Display the annotated frame in a window
            # cv2.imshow() creates a window and displays the image
            # 'Poker Card Detection Tool' is the window title
            # annotated_frame is the image to display
            cv2.imshow('Poker Card Detection Tool', annotated_frame)
            
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

