"""
Playing Card Edge Detection System

This module provides robust edge detection capabilities to identify
the outline of playing cards in images using computer vision techniques.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import math


class CardEdgeDetector:
    """Robust playing card outline detection using edge detection and contour analysis."""
    
    def __init__(self, 
                 canny_low: int = 50,
                 canny_high: int = 150,
                 blur_kernel: int = 5,
                 threshold_value: int = 127,
                 min_card_area: int = 1000,
                 max_card_area: int = 1000000,
                 aspect_ratio_tolerance: float = 0.3,
                 use_threshold: bool = True):
        """
        Initialize the card edge detector.
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            blur_kernel: Kernel size for Gaussian blur preprocessing (must be odd)
            threshold_value: Threshold value for binary thresholding (0-255)
            min_card_area: Minimum area for a valid card contour
            max_card_area: Maximum area for a valid card contour
            aspect_ratio_tolerance: Tolerance for card aspect ratio (0.63 is standard playing card ratio)
            use_threshold: Whether to apply binary thresholding before edge detection
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.threshold_value = threshold_value
        self.min_card_area = min_card_area
        self.max_card_area = max_card_area
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.use_threshold = use_threshold
        self.standard_card_ratio = 0.63  # Standard playing card aspect ratio (width/height)
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image to enhance edge detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (grayscale image, preprocessed image)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        return gray, blurred
    
    def detect_edges(self, image: np.ndarray, use_adaptive: bool = False) -> np.ndarray:
        """
        Detect edges in the image using thresholding and Canny edge detection.
        
        Args:
            image: Preprocessed grayscale image
            use_adaptive: If True, use adaptive thresholding instead of binary threshold
            
        Returns:
            Binary edge image
        """
        # Apply thresholding first to reduce sensitivity
        if use_adaptive:
            # Use adaptive thresholding for varying lighting conditions
            thresholded = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif self.use_threshold:
            # Apply binary threshold to convert to black and white
            # This reduces noise and makes edges cleaner
            _, thresholded = cv2.threshold(
                image, self.threshold_value, 255, cv2.THRESH_BINARY
            )
        else:
            # No thresholding, use image directly
            thresholded = image
        
        # Apply Canny edge detection on thresholded image
        edges = cv2.Canny(thresholded, self.canny_low, self.canny_high)
        
        # Morphological operations to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the edge image.
        
        Args:
            edges: Binary edge image
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def approximate_polygon(self, contour: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
        """
        Approximate a contour to a polygon with fewer vertices.
        
        Args:
            contour: Input contour
            epsilon_factor: Approximation accuracy factor (percentage of perimeter)
            
        Returns:
            Approximated polygon
        """
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    
    def is_valid_card_shape(self, contour: np.ndarray, approx: np.ndarray) -> bool:
        """
        Check if a contour represents a valid card shape.
        
        Args:
            contour: Original contour
            approx: Approximated polygon
            
        Returns:
            True if the shape is likely a card
        """
        # Check area
        area = cv2.contourArea(contour)
        if area < self.min_card_area or area > self.max_card_area:
            return False
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) < 4 or len(approx) > 8:
            return False
        
        # Calculate bounding rectangle and check aspect ratio
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            return False
        
        # Normalize aspect ratio (handles rotation)
        aspect_ratio = min(width, height) / max(width, height)
        expected_ratio = self.standard_card_ratio
        
        # Check if aspect ratio matches playing card dimensions
        ratio_diff = abs(aspect_ratio - expected_ratio)
        if ratio_diff > self.aspect_ratio_tolerance:
            return False
        
        return True
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in a consistent order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        # Reshape if needed
        pts = pts.reshape(4, 2)
        
        # Initialize ordered points
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left has smallest sum, bottom-right has largest sum
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Top-right has smallest difference, bottom-left has largest difference
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def detect_card_outline(self, 
                           image: np.ndarray, 
                           use_adaptive: bool = False,
                           return_all: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect card outline in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            use_adaptive: Use adaptive thresholding for varying lighting
            return_all: If True, return all valid card contours, not just the best one
            
        Returns:
            If return_all=False: Tuple of (card_contour, approximated_polygon) or None
            If return_all=True: List of tuples (contour, approximated_polygon)
        """
        # Preprocess
        gray, blurred = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(blurred, use_adaptive)
        
        # Find contours
        contours = self.find_contours(edges)
        
        # Filter and validate card shapes
        valid_cards = []
        for contour in contours:
            approx = self.approximate_polygon(contour)
            if self.is_valid_card_shape(contour, approx):
                valid_cards.append((contour, approx))
        
        if not valid_cards:
            return None if not return_all else []
        
        if return_all:
            return valid_cards
        
        # Return the largest valid card (most likely to be the main card)
        valid_cards.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
        return valid_cards[0]
    
    def extract_card_region(self, 
                           image: np.ndarray, 
                           card_points: np.ndarray,
                           output_size: Tuple[int, int] = (200, 320)) -> np.ndarray:
        """
        Extract and warp the card region to a standard size.
        
        Args:
            image: Original image
            card_points: Four corner points of the card
            output_size: Desired output size (width, height)
            
        Returns:
            Warped card image
        """
        # Order the points
        rect = self.order_points(card_points.reshape(-1, 2))
        
        # Define destination points
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Warp the image
        warped = cv2.warpPerspective(image, M, output_size)
        
        return warped
    
    def visualize_detection(self, 
                           image: np.ndarray, 
                           contour: np.ndarray,
                           approx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw the detected card outline on the image.
        
        Args:
            image: Original image
            contour: Detected contour
            approx: Optional approximated polygon
            
        Returns:
            Image with drawn outline
        """
        result = image.copy()
        
        # Draw contour
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        # Draw approximated polygon if provided
        if approx is not None:
            cv2.drawContours(result, [approx], -1, (255, 0, 0), 3)
            
            # Draw corner points
            for point in approx:
                x, y = point.ravel()
                cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        return result
    
    def process_camera_feed(self,
                           camera_index: int = 0,
                           use_adaptive: bool = False,
                           show_edges: bool = False,
                           show_warped: bool = False) -> None:
        """
        Process live camera feed and detect cards in real-time.
        
        Args:
            camera_index: Camera device index (default: 0)
            use_adaptive: Use adaptive thresholding for varying lighting
            show_edges: Show edge detection visualization
            show_warped: Show warped card region when detected
        """
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera resolution (optional, adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Camera feed started. Press 'q' to quit, 's' to save frame, 'a' to toggle adaptive")
        print(f"Current settings: Threshold={self.threshold_value}, Canny({self.canny_low}, {self.canny_high}), Adaptive={use_adaptive}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect card
            result = self.detect_card_outline(frame, use_adaptive=use_adaptive)
            
            # Prepare display images
            display = frame.copy()
            edge_display = None
            
            if result:
                contour, approx = result
                # Draw detection
                display = self.visualize_detection(frame, contour, approx)
                
                # Add text overlay
                area = cv2.contourArea(contour)
                cv2.putText(display, f"Card Detected! Area: {area:.0f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Optionally show warped card
                if show_warped:
                    try:
                        warped = self.extract_card_region(frame, approx)
                        # Resize warped card for side-by-side display
                        warped_resized = cv2.resize(warped, (160, 256))
                        # Place in top-right corner
                        h, w = warped_resized.shape[:2]
                        display[10:10+h, display.shape[1]-w-10:display.shape[1]-10] = warped_resized
                    except:
                        pass
            else:
                cv2.putText(display, "No card detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show edges if requested
            if show_edges:
                gray, blurred = self.preprocess_image(frame)
                # Show thresholded image
                if use_adaptive:
                    thresholded = cv2.adaptiveThreshold(
                        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                elif self.use_threshold:
                    _, thresholded = cv2.threshold(
                        blurred, self.threshold_value, 255, cv2.THRESH_BINARY
                    )
                else:
                    thresholded = blurred
                
                edges = self.detect_edges(blurred, use_adaptive)
                # Resize for better display
                h, w = frame.shape[:2]
                display_width = 640
                scale = display_width / w
                display_height = int(h * scale)
                
                threshold_display = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                threshold_display = cv2.resize(threshold_display, (display_width, display_height))
                cv2.putText(threshold_display, "Thresholded", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edge_display = cv2.resize(edge_display, (display_width, display_height))
                cv2.putText(edge_display, "Edges", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Stack them vertically
                edge_display = np.vstack([threshold_display, edge_display])
            
            # Add frame counter and controls info
            cv2.putText(display, f"Frame: {frame_count} | Threshold: {self.threshold_value}", 
                       (10, display.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "Press 'q' to quit, 's' to save, 'a' for adaptive, 't'/'T' for threshold", 
                       (10, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"Canny: ({self.canny_low}, {self.canny_high})", 
                       (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Card Detection - Live Feed", display)
            
            if show_edges and edge_display is not None:
                cv2.imshow("Edge Detection", edge_display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"detected_card_frame_{frame_count}.jpg"
                cv2.imwrite(filename, display)
                print(f"Frame saved to {filename}")
            elif key == ord('a'):
                # Toggle adaptive thresholding
                use_adaptive = not use_adaptive
                print(f"Adaptive thresholding: {use_adaptive}")
            elif key == ord('e'):
                # Toggle edge display
                show_edges = not show_edges
                if not show_edges:
                    cv2.destroyWindow("Edge Detection")
                print(f"Edge display: {show_edges}")
            elif key == ord('w'):
                # Toggle warped card display
                show_warped = not show_warped
                print(f"Warped card display: {show_warped}")
            elif key == ord('+') or key == ord('='):
                # Increase Canny high threshold
                self.canny_high = min(255, self.canny_high + 10)
                print(f"Canny high threshold: {self.canny_high}")
            elif key == ord('-'):
                # Decrease Canny high threshold
                self.canny_high = max(50, self.canny_high - 10)
                print(f"Canny high threshold: {self.canny_high}")
            elif key == ord('t'):
                # Decrease threshold value
                self.threshold_value = max(0, self.threshold_value - 10)
                print(f"Threshold value: {self.threshold_value}")
            elif key == ord('T'):
                # Increase threshold value
                self.threshold_value = min(255, self.threshold_value + 10)
                print(f"Threshold value: {self.threshold_value}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera feed stopped.")


def detect_card_from_image(image_path: str, 
                          detector: Optional[CardEdgeDetector] = None,
                          visualize: bool = True,
                          save_output: bool = False,
                          output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to detect a card from an image file.
    
    Args:
        image_path: Path to input image
        detector: Optional CardEdgeDetector instance (uses default if None)
        visualize: Whether to display the result
        save_output: Whether to save the visualization
        output_path: Path to save the output image
        
    Returns:
        Tuple of (contour, approximated_polygon) or None if no card detected
    """
    if detector is None:
        detector = CardEdgeDetector()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Detect card
    result = detector.detect_card_outline(image)
    
    if result is None:
        print("No card detected in the image.")
        return None
    
    contour, approx = result
    
    if visualize or save_output:
        vis_image = detector.visualize_detection(image, contour, approx)
        
        if visualize:
            cv2.imshow("Card Detection", vis_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if save_output:
            output_file = output_path or image_path.replace('.', '_detected.')
            cv2.imwrite(output_file, vis_image)
            print(f"Output saved to {output_file}")
    
    return result


if __name__ == "__main__":
    """
    Example usage of the card edge detector.
    Supports both image files and live camera feed.
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect playing card outlines in images or live camera feed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process an image
  python card_edge_detector.py image.jpg
  
  # Use live camera feed (default)
  python card_edge_detector.py --camera
  
  # Use specific camera with adaptive thresholding
  python card_edge_detector.py --camera --camera-id 1 --adaptive
  
  # Live feed with edge visualization
  python card_edge_detector.py --camera --show-edges

Controls (when using camera):
  'q' - Quit
  's' - Save current frame
  'a' - Toggle adaptive thresholding
  'e' - Toggle edge display
  'w' - Toggle warped card display
  '+' - Increase Canny threshold
  '-' - Decrease Canny threshold
  't' - Decrease threshold value
  'T' - Increase threshold value
        """
    )
    parser.add_argument("image_path", nargs="?", help="Path to input image (optional if using --camera)")
    parser.add_argument("--camera", action="store_true", help="Use live camera feed")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny lower threshold")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny upper threshold")
    parser.add_argument("--threshold", type=int, default=127, help="Binary threshold value (0-255)")
    parser.add_argument("--no-threshold", action="store_true", help="Disable binary thresholding")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive thresholding")
    parser.add_argument("--show-edges", action="store_true", help="Show edge detection visualization (camera mode)")
    parser.add_argument("--show-warped", action="store_true", help="Show warped card region (camera mode)")
    parser.add_argument("--no-visualize", action="store_true", help="Don't display result (image mode)")
    parser.add_argument("--save", action="store_true", help="Save output image (image mode)")
    parser.add_argument("--output", type=str, help="Output image path (image mode)")
    
    args = parser.parse_args()
    
    # Create detector with custom parameters
    detector = CardEdgeDetector(
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        threshold_value=args.threshold,
        use_threshold=not args.no_threshold
    )
    
    # Check if using camera or image file
    # Default to camera if no image path provided or --camera flag is set
    if args.camera or args.image_path is None:
        # Use live camera feed
        print("Starting live camera feed...")
        detector.process_camera_feed(
            camera_index=args.camera_id,
            use_adaptive=args.adaptive,
            show_edges=args.show_edges,
            show_warped=args.show_warped
        )
    else:
        # Process image file
        result = detect_card_from_image(
            args.image_path,
            detector=detector,
            visualize=not args.no_visualize,
            save_output=args.save,
            output_path=args.output
        )
        
        if result:
            contour, approx = result
            area = cv2.contourArea(contour)
            print(f"Card detected! Area: {area:.2f} pixels")
            print(f"Corners: {len(approx)} points")

