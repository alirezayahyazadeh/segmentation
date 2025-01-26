# 3D Object Detection and Depth Image Processing

## Overview
This repository contains a collection of Python scripts that combine real-time object detection, depth processing, and 3D point cloud visualization. These scripts leverage advanced libraries like YOLOv8, Open3D, PyRealSense2, and OpenCV to process depth data, detect objects, and visualize 3D point clouds.

## Features
- **YOLOv8 Object Detection**: Leverages the power of YOLOv8 for real-time object detection on video streams.
- **Depth Image Processing**: Processes depth images from Intel RealSense cameras for background subtraction and object isolation.
- **3D Point Cloud Generation**: Converts depth and color data into interactive 3D point clouds using Open3D.
- **Background Removal**: Removes the background from the color images using `rembg`.
- **Visualization**: Provides both 2D and 3D visualization for detections and processed data.

## Included Files
### Core Scripts
1. **t6.py**: Processes video streams from Intel RealSense cameras to perform real-time YOLOv8 object detection and display annotated images.
2. **t7.py**: Captures depth data, isolates objects using threshold-based masking, and visualizes results.
3. **t8.py**: Focuses on generating 3D point clouds from depth and color data, integrating object detection for specific regions.
4. **t9.py**: Integrates YOLOv8 detection results with filtered depth images for precise object segmentation.
5. **t10.py**: Applies custom filters to depth images based on YOLOv8 bounding boxes, emphasizing detected objects.
6. **t11.py**: Combines 2D object detection with depth processing to produce annotated depth masks.
7. **t12.py**: The main script for generating and visualizing 3D point clouds while applying YOLOv8 detections and removing unwanted regions.
8. **t13.py**: The primary driver for the project, showcasing full functionality, including YOLOv8 detection, depth processing, background removal, and 3D visualization.

## Requirements
- **Python**: 3.8 or higher
- **Required Libraries**:
  - OpenCV
  - NumPy
  - PyRealSense2
  - Open3D
  - Rembg
  - Ultralytics YOLO

## Installation
1. Clone the repository:
   ```bash
   git clone (https://github.com/alirezayahyazadeh/segmentation)
   
##Usage
Setup Hardware
•	Connect your Intel RealSense camera.
•	Ensure the camera drivers are installed and working.
##Run a Script
Example: Running the main file for object detection and point cloud generation
python t13.py
Explore Outputs
•	View real-time detections on the video feed.
•	Interact with 3D point clouds in the Open3D visualization window.
Key Commands
•	Press ESC to exit camera-based scripts.
•	Open3D visualization supports interactive 3D manipulation (e.g., rotate, zoom).
#Contribution
We welcome contributions to improve this project. Feel free to fork the repository, submit issues, or create pull requests.
 

