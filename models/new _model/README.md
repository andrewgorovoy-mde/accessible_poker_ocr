# Playing Card Edge Detection System

A robust computer vision system for detecting playing card outlines using edge detection techniques.

## Features

- **Robust Edge Detection**: Uses Canny edge detection with preprocessing to handle various lighting conditions
- **Shape Validation**: Filters contours based on area, aspect ratio, and polygon approximation
- **Adaptive Thresholding**: Optional adaptive thresholding for varying lighting conditions
- **Perspective Correction**: Can extract and warp card regions to standard dimensions
- **Multiple Card Detection**: Can detect and return all valid cards in an image
- **Live Camera Feed**: Real-time card detection from webcam with interactive controls

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from card_edge_detector import CardEdgeDetector, detect_card_from_image
import cv2

# Load image
image = cv2.imread('path/to/card_image.jpg')

# Create detector
detector = CardEdgeDetector()

# Detect card outline
result = detector.detect_card_outline(image)

if result:
    contour, approx = result
    # Visualize detection
    vis_image = detector.visualize_detection(image, contour, approx)
    cv2.imshow("Detection", vis_image)
    cv2.waitKey(0)
```

### Command Line Usage

#### Image Processing

```bash
# Basic detection with visualization
python card_edge_detector.py path/to/image.jpg

# With custom Canny thresholds
python card_edge_detector.py path/to/image.jpg --canny-low 30 --canny-high 200

# Use adaptive thresholding for difficult lighting
python card_edge_detector.py path/to/image.jpg --adaptive

# Save output without displaying
python card_edge_detector.py path/to/image.jpg --save --output detected_card.jpg --no-visualize
```

#### Live Camera Feed

```bash
# Start live camera feed (default camera)
python card_edge_detector.py --camera

# Use specific camera device
python card_edge_detector.py --camera --camera-id 1

# Enable adaptive thresholding for varying lighting
python card_edge_detector.py --camera --adaptive

# Show edge detection visualization
python card_edge_detector.py --camera --show-edges

# Show warped card region when detected
python card_edge_detector.py --camera --show-warped

# Combine options
python card_edge_detector.py --camera --adaptive --show-edges --show-warped
```

#### Camera Controls (While Running)

- **'q'** - Quit the application
- **'s'** - Save current frame to file
- **'a'** - Toggle adaptive thresholding on/off
- **'e'** - Toggle edge detection display
- **'w'** - Toggle warped card display
- **'+'** or **'='** - Increase Canny high threshold
- **'-'** - Decrease Canny high threshold

### Advanced Usage

```python
from card_edge_detector import CardEdgeDetector
import cv2

# Custom detector configuration
detector = CardEdgeDetector(
    canny_low=30,           # Lower Canny threshold
    canny_high=150,         # Upper Canny threshold
    blur_kernel=7,          # Gaussian blur kernel size
    min_card_area=5000,     # Minimum card area in pixels
    max_card_area=500000,   # Maximum card area in pixels
    aspect_ratio_tolerance=0.2  # Aspect ratio tolerance
)

# Detect with adaptive thresholding
result = detector.detect_card_outline(image, use_adaptive=True)

# Detect all cards in image
all_cards = detector.detect_card_outline(image, return_all=True)

# Extract and warp card region
if result:
    contour, approx = result
    warped_card = detector.extract_card_region(image, approx, output_size=(200, 320))
    cv2.imwrite('warped_card.jpg', warped_card)

# Process live camera feed
detector.process_camera_feed(
    camera_index=0,      # Camera device index
    use_adaptive=True,   # Use adaptive thresholding
    show_edges=False,   # Show edge visualization
    show_warped=True    # Show warped card when detected
)
```

## API Reference

### CardEdgeDetector

Main class for card edge detection.

#### Parameters

- `canny_low` (int): Lower threshold for Canny edge detection (default: 50)
- `canny_high` (int): Upper threshold for Canny edge detection (default: 150)
- `blur_kernel` (int): Kernel size for Gaussian blur (default: 5, must be odd)
- `min_card_area` (int): Minimum area for valid card (default: 1000)
- `max_card_area` (int): Maximum area for valid card (default: 1000000)
- `aspect_ratio_tolerance` (float): Tolerance for card aspect ratio (default: 0.3)

#### Methods

- `detect_card_outline(image, use_adaptive=False, return_all=False)`: Detect card outline(s)
- `extract_card_region(image, card_points, output_size=(200, 320))`: Extract and warp card region
- `visualize_detection(image, contour, approx=None)`: Draw detection on image
- `process_camera_feed(camera_index=0, use_adaptive=False, show_edges=False, show_warped=False)`: Process live camera feed

## Algorithm Overview

1. **Preprocessing**: Convert to grayscale and apply Gaussian blur
2. **Edge Detection**: Use Canny edge detection (optionally with adaptive thresholding)
3. **Contour Detection**: Find all external contours
4. **Shape Validation**: Filter contours based on:
   - Area constraints
   - Polygon approximation (4-8 vertices)
   - Aspect ratio matching standard playing card dimensions (0.63 ratio)
5. **Corner Ordering**: Order detected corners consistently

## Tips for Best Results

1. **Lighting**: Ensure even lighting across the card
2. **Contrast**: Higher contrast between card and background improves detection
3. **Resolution**: Higher resolution images provide better edge detection
4. **Angle**: Works best with cards facing the camera, but handles slight angles
5. **Background**: Solid, contrasting backgrounds work best

## Troubleshooting

- **No card detected**: Try adjusting Canny thresholds or using `use_adaptive=True`
- **Multiple false positives**: Increase `min_card_area` or adjust `aspect_ratio_tolerance`
- **Missed corners**: Decrease `blur_kernel` or adjust `epsilon_factor` in `approximate_polygon`

