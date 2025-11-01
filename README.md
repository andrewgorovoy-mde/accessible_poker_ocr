# Playing Cards Detection - Camera Stream Version

Real-time playing cards detection using YOLOv8 and your laptop's webcam, based on [Playing-Cards-Detection](https://github.com/PD-Mera/Playing-Cards-Detection).

## Overview

This project demonstrates **real-time object detection** using computer vision and machine learning. It uses a pre-trained YOLOv8 (You Only Look Once version 8) neural network model to detect and identify playing cards from your webcam feed in real-time.

### What is Object Detection?

Object detection is a computer vision task that involves:
1. **Finding objects** in an image (locating where cards are)
2. **Classifying objects** (identifying which card it is - e.g., "Ace of Spades")
3. **Drawing bounding boxes** around detected objects with confidence scores

### How It Works

1. **Camera Capture**: The script continuously captures frames (images) from your webcam
2. **Preprocessing**: Each frame is resized and formatted for the neural network
3. **Inference**: The YOLOv8 model analyzes the frame to find playing cards
4. **Post-processing**: Detections are filtered by confidence threshold
5. **Visualization**: Bounding boxes and labels are drawn on the frame
6. **Display**: The annotated frame is shown in a window, updating in real-time

## Features

- **Real-time card detection** from webcam stream (typically 20-30 frames per second)
- **Supports all 52 playing card classes** (Ace through King in all 4 suits)
- **Visual feedback** with bounding boxes and confidence scores displayed on screen
- **Save frames** with detections to image files
- **Configurable parameters** for confidence threshold and image processing size
- **User-friendly controls** with keyboard shortcuts

## Installation

### Prerequisites

Before you begin, ensure you have:
- **Python 3.8 or higher** installed on your system
- A **webcam/camera** connected to your computer
- **pip** (Python package installer) available in your terminal

### Step 1: Install Python Dependencies

Open your terminal/command prompt and navigate to this project directory, then run:

```bash
pip install -r requirements.txt
```

This will install all required libraries:
- **ultralytics**: The YOLOv8 model framework
- **opencv-python**: Computer vision library for camera and image processing
- **numpy**: Numerical computing library (used internally by OpenCV)
- **torch**: PyTorch deep learning framework (required by YOLOv8)
- **torchvision**: Computer vision utilities for PyTorch
- **Pillow**: Image processing library

**Note**: If you're on macOS or Linux, you may need to use `pip3` instead of `pip`.

### Step 2: Download the Model Weights

The script needs a pre-trained model file to detect cards. You have two options:

#### Option A: Download Pre-trained Model (Recommended)

1. Download the model file from: https://drive.google.com/file/d/1AqZnW6dI6flFZvGxAn6A9apDNSviXZ5f/view?usp=share_link
2. Place the downloaded file `yolov8s_playing_cards.pt` in the project root directory (same folder as `camera_detect.py`)

#### Option B: Train Your Own Model

If you want to train a custom model, follow the instructions in the original repository: https://github.com/PD-Mera/Playing-Cards-Detection

## Usage

### Basic Usage (Default Settings)

Run the script with default settings:

```bash
python camera_detect.py
```

This will:
- Use the default camera (camera index 0)
- Load the model from `./yolov8s_playing_cards.pt`
- Use confidence threshold of 0.25
- Process images at 640x640 pixels

### Advanced Usage (Custom Options)

You can customize the script behavior with command-line arguments:

```bash
python camera_detect.py --model ./yolov8s_playing_cards.pt --camera 0 --conf 0.25 --imgsz 640
```

### Command-Line Arguments Explained

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | string | `./yolov8s_playing_cards.pt` | Path to the YOLOv8 model weights file (.pt file) |
| `--camera` | integer | `0` | Camera device index. 0 = first camera, 1 = second camera, etc. |
| `--conf` | float | `0.25` | Confidence threshold (0.0 to 1.0). Only detections with confidence ≥ this value are shown. Lower = more detections (may include false positives). Higher = fewer detections (more reliable). |
| `--imgsz` | integer | `640` | Image size for inference. Larger values (640, 1280) = better accuracy but slower. Smaller values (320, 416) = faster but may miss small cards. |

### Example Usage Scenarios

**Use a different camera (e.g., external USB camera):**
```bash
python camera_detect.py --camera 1
```

**Increase detection threshold to reduce false positives:**
```bash
python camera_detect.py --conf 0.5
```

**Faster processing for lower-end hardware:**
```bash
python camera_detect.py --imgsz 320 --conf 0.3
```

**Use a custom model path:**
```bash
python camera_detect.py --model /path/to/your/model.pt
```

### Interactive Controls

While the script is running:

- **Press `q`**: Quit the application and close the camera
- **Press `s`**: Save the current frame with detections to a file (saved as `captured_frame_1.jpg`, `captured_frame_2.jpg`, etc.)

## Understanding the Output

When you run the script, you'll see:

1. **Console Output**:
   - Status messages about model loading
   - Camera initialization status
   - Frame save confirmations

2. **Video Window**:
   - Live video feed from your camera
   - **Bounding boxes** (colored rectangles) around detected cards
   - **Labels** showing card name and confidence score (e.g., "Ace_of_Spades 0.95")
   - **Info overlay** at the top showing:
     - Number of detected cards
     - Names of detected cards (up to 3)
     - Frame counter

## Requirements

### Software Requirements

- **Python 3.8 or higher** (tested on Python 3.8-3.11)
- **pip** (Python package installer)

### Hardware Requirements

- **Webcam/Camera**: Any USB or built-in camera
- **CPU**: Modern multi-core processor recommended
- **RAM**: At least 4GB available memory
- **GPU** (Optional): CUDA-capable NVIDIA GPU for faster inference (will use CPU if GPU not available)

### Operating System Compatibility

- ✅ **Windows 10/11**
- ✅ **macOS 10.14+**
- ✅ **Linux** (Ubuntu 18.04+, Debian, etc.)

## Troubleshooting

### Issue: Camera Not Opening

**Symptoms**: Error message "Could not open camera {index}"

**Solutions**:
1. Try different camera indices:
   ```bash
   python camera_detect.py --camera 1
   python camera_detect.py --camera 2
   ```

2. Check if other applications are using the camera:
   - Close video conferencing apps (Zoom, Teams, etc.)
   - Close other camera applications
   - Restart your computer if needed

3. **macOS specific**: Grant camera permissions:
   - Go to System Preferences → Security & Privacy → Camera
   - Check the box next to Terminal (or your IDE)

4. **Windows specific**: Check Device Manager:
   - Open Device Manager → Imaging devices
   - Ensure your camera is listed and enabled

### Issue: Model File Not Found

**Symptoms**: Error message "Model file not found at {path}"

**Solutions**:
1. Verify the file exists:
   ```bash
   ls yolov8s_playing_cards.pt  # On macOS/Linux
   dir yolov8s_playing_cards.pt  # On Windows
   ```

2. Check the file path:
   - Ensure the file is in the same directory as `camera_detect.py`
   - Or use the `--model` flag to specify the full path:
     ```bash
     python camera_detect.py --model /full/path/to/yolov8s_playing_cards.pt
     ```

3. Re-download the model file if it's corrupted or missing

### Issue: Performance Issues (Slow/Laggy)

**Symptoms**: Low frame rate, delayed detection, choppy video

**Solutions**:
1. **Reduce image processing size** (faster but less accurate):
   ```bash
   python camera_detect.py --imgsz 320
   ```

2. **Increase confidence threshold** (processes fewer detections):
   ```bash
   python camera_detect.py --conf 0.4
   ```

3. **Close other applications** using CPU/GPU resources

4. **Use GPU acceleration** (if available):
   - Install CUDA toolkit and PyTorch with CUDA support
   - The script will automatically use GPU if available

5. **Lower camera resolution** (the script requests 1280x720, but camera may override)

### Issue: Import Errors

**Symptoms**: "ModuleNotFoundError" or "ImportError"

**Solutions**:
1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Use `pip3` instead of `pip` on macOS/Linux:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Try installing packages individually:
   ```bash
   pip install ultralytics opencv-python numpy torch torchvision Pillow
   ```

### Issue: No Cards Detected

**Symptoms**: Camera works but no cards are detected

**Solutions**:
1. **Lower the confidence threshold**:
   ```bash
   python camera_detect.py --conf 0.15
   ```

2. **Ensure good lighting**: Cards should be well-lit and visible

3. **Check card positioning**: Cards should be flat and clearly visible to the camera

4. **Increase image size** for better accuracy:
   ```bash
   python camera_detect.py --imgsz 1280
   ```

## Code Structure

For entry-level engineers wanting to understand the code:

- **`camera_detect.py`**: Main script with heavily commented code explaining every line
- **`requirements.txt`**: Lists all Python package dependencies
- **`yolov8s_playing_cards.pt`**: Pre-trained model weights (neural network parameters)

The code is extensively documented with inline comments explaining:
- What each line does
- Why it's needed
- How variables are used
- What values mean

## Model Performance

The YOLOv8s model used in this project achieves excellent performance:

- **mAP50**: 0.99498 (99.5% accuracy at 50% IoU threshold)
- **mAP50:95**: 0.95681 (average precision across multiple IoU thresholds)
- **Model Size**: 22.0MB (relatively small, fast to load)
- **Speed**: ~20-30 FPS on modern CPU, ~60+ FPS on GPU

### What These Metrics Mean

- **mAP (mean Average Precision)**: Measures how accurately the model detects objects
- **mAP50**: Accuracy when bounding boxes overlap by at least 50% with ground truth
- **mAP50:95**: Average accuracy across IoU thresholds from 50% to 95%
- **Higher values = better accuracy**

## How Object Detection Works (Conceptual)

1. **Input**: Raw image/video frame from camera
2. **Preprocessing**: Image resized and normalized for the neural network
3. **Neural Network**: YOLOv8 processes the image through multiple layers:
   - Extracts features (edges, shapes, patterns)
   - Identifies potential object locations
   - Classifies objects at each location
4. **Post-processing**: Filters results by confidence threshold
5. **Output**: List of detected objects with:
   - Bounding box coordinates (x, y, width, height)
   - Class label (which card)
   - Confidence score (0.0 to 1.0)

## Learning Resources

If you're new to computer vision and machine learning:

- **OpenCV Tutorial**: https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **Object Detection Concepts**: https://www.tensorflow.org/lite/models/object_detection/overview

## References

- **Original Repository**: https://github.com/PD-Mera/Playing-Cards-Detection
- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **OpenCV Documentation**: https://docs.opencv.org/

## License

This project is based on the Playing-Cards-Detection repository. Please refer to the original repository for license information.

## Contributing

To improve this project:
1. Report bugs by opening an issue
2. Suggest improvements or new features
3. Submit pull requests with code improvements

## Support

If you encounter issues not covered in the troubleshooting section:
1. Check the original repository issues: https://github.com/PD-Mera/Playing-Cards-Detection/issues
2. Review the code comments for detailed explanations
3. Ensure all dependencies are correctly installed

