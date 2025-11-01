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
    print("\nStarting video stream...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    # "-" * 50 creates a line of 50 dashes for visual separation
    print("-" * 50)
    
    # Initialize counters to track statistics
    # frame_count: counts how many frames we've processed (for display)
    frame_count = 0
    # saved_count: counts how many frames we've saved (for unique filenames)
    saved_count = 0
    
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
            
            # Annotate the frame with detection results
            # results[0] gets the first (and only) result from the list
            # .plot() draws bounding boxes and labels on the image automatically
            # It returns a new image (annotated_frame) with boxes drawn on it
            annotated_frame = results[0].plot()
            
            # Extract information about detected cards for display
            # Create an empty list to store card names and confidence scores
            detected_cards = []
            
            # Check if any boxes (detections) were found
            # results[0].boxes contains all the detected objects
            # If it's None, no cards were detected
            if results[0].boxes is not None:
                # Loop through each detected box (each card found)
                for box in results[0].boxes:
                    # Get the class ID (which type of card it is)
                    # box.cls[0] gets the first class ID from the box
                    # int() converts it to a whole number
                    class_id = int(box.cls[0])
                    
                    # Get the confidence score (how sure the model is)
                    # box.conf[0] gets the confidence value (0.0 to 1.0)
                    # float() converts it to a decimal number
                    confidence = float(box.conf[0])
                    
                    # Convert class ID to human-readable name (e.g., "Ace of Spades")
                    # model.names is a dictionary mapping IDs to names
                    # model.names[class_id] looks up the name for this ID
                    class_name = model.names[class_id]
                    
                    # Add this card info to our list as a formatted string
                    # f"{class_name} ({confidence:.2f})" creates text like "Ace_of_Spades (0.95)"
                    # .2f means format the float to 2 decimal places
                    detected_cards.append(f"{class_name} ({confidence:.2f})")
            
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

