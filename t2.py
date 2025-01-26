import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

# Set up output directory
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Distance thresholds in meters
min_object_distance_in_meters = 1.0
max_object_distance_in_meters = 1.3
min_depth_threshold = int(min_object_distance_in_meters * 1000)  # Convert to millimeters
max_depth_threshold = int(max_object_distance_in_meters * 1000)

# Function to process frames
def process_frame(color_image, depth_image):
    objects_image = np.zeros_like(color_image, dtype=np.uint8)

    # Create mask for objects within depth range
    mask = (depth_image > min_depth_threshold) & (depth_image < max_depth_threshold)
    objects_image[mask] = color_image[mask]

    # Draw a bounding box (for demonstration purposes)
    if np.any(mask):
        y, x = np.where(mask)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        cv2.rectangle(objects_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return objects_image

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipeline.start(config)
    last_capture_time = time.time()  # Initialize timer

    # Initialize objects_image with a blank frame
    objects_image = None

    while True:
        # Get frames from the pipeline
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Capture only every 3 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 3:
            # Process frame
            objects_image = process_frame(color_image, depth_image)

            # Save the processed image
            output_path = os.path.join(output_dir, f"detected_object_{int(current_time)}.jpg")
            cv2.imwrite(output_path, objects_image)
            print(f"Saved image: {output_path}")

            last_capture_time = current_time  # Reset timer

        # If objects_image is None (not yet processed), display color_image
        display_image = objects_image if objects_image is not None else color_image

        # Display results
        cv2.imshow('Detected Objects', display_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
