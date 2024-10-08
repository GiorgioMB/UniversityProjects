# YOLO Webcam Object Detection

## Project Overview

This project implements a real-time object detection system using a webcam feed and the YOLO (You Only Look Once) object detection model. The model detects objects from the live video stream, drawing bounding boxes around them and labeling them with the corresponding object class and confidence score. It leverages the **YOLOv5** pre-trained model, provided by the `ultralytics` library, and processes video frames in real-time through the OpenCV library.

The script continuously captures frames from a connected webcam, processes each frame through the YOLO model, and visually displays the detection results (bounding boxes and labels) on the screen. The user can terminate the session by pressing the 'q' key.

## Libraries Used

The following libraries are required to run this project:

- **OpenCV (`cv2`)**: Used for accessing and processing video streams from the webcam. It provides functionalities for image and video analysis and manipulation, including displaying the output window with bounding boxes and labels.
  
- **Ultralytics (`YOLO`)**: This library provides a Python interface for the YOLO object detection models. In this project, we use a pre-trained YOLOv5 model for detecting objects in real-time.
  
  The model used here is `yolov5su.pt`, which is a custom or pre-trained version of YOLOv5 that suits lightweight, real-time object detection tasks.

## Installation

Before running the project, ensure you have the following installed:

1. **OpenCV**: Install via pip:
    ```bash
    pip install opencv-python
    ```

2. **Ultralytics YOLO**: Install the Ultralytics YOLO package:
    ```bash
    pip install ultralytics
    ```

## Running the Project

1. Ensure your webcam is connected and functioning properly.
2. Run the Python script. It will automatically open a webcam feed, detect objects in real-time, and display the results on the screen.
3. Press the 'q' key at any time to exit the video feed.

## How It Works

- **Webcam Capture**: The script captures frames using OpenCV from the connected webcam.
- **YOLO Inference**: Each captured frame is passed to the YOLO model for inference, which returns bounding boxes, labels, and confidence scores for detected objects.
- **Displaying Results**: The script draws a rectangle around each detected object and annotates it with the class label and detection confidence score.
- **Exit Condition**: The script can be stopped by pressing 'q'.

## Example Output

Upon running the script, you should see a window displaying the webcam feed with real-time object detection. Detected objects will be surrounded by a green bounding box with the label and confidence score displayed on top of the object.

---

Ensure you have a compatible webcam and the required Python packages installed before running the script.
