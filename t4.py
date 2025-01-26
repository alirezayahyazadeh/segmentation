import numpy as np
from PIL import Image, ImageOps
import os
import pyrealsense2 as rs  # Import RealSense library
import time

# Directory to save output images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Distance thresholds in meters
min_object_distance_in_meters = 1.0
max_object_distance_in_meters = 1.3
min_depth_threshold = int(min_object_distance_in_meters * 1000)
max_depth_threshold = int(max_object_distance_in_meters * 1000)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Change detection: store the last frame
last_depth_frame = None

# Function to process and mask objects based on depth
def process_frame(color_image, depth_image):
    # Create a mask for the objects within the depth range
    mask_np = (depth_image > min_depth_threshold) & (depth_image < max_depth_threshold)
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))  # Binary mask

    # Apply mask to the color image
    objects_image_np = np.zeros_like(color_image)
    objects_image_np[mask_np] = color_image[mask_np]
    objects_image = Image.fromarray(objects_image_np)

    return objects_image, mask_image

# Function to display images inline using PIL
def show_images(color_image, objects_image, mask_image):
    # Create a single composite image for display
    composite = Image.new("RGB", (color_image.width * 3, color_image.height))
    composite.paste(color_image, (0, 0))
    composite.paste(objects_image, (color_image.width, 0))
    composite.paste(ImageOps.colorize(mask_image.convert("L"), black="black", white="green"), (color_image.width * 2, 0))
    composite.show()

try:
    pipeline.start(config)

    frame_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to NumPy arrays
        color_image_np = np.asanyarray(color_frame.get_data())
        depth_image_np = np.asanyarray(depth_frame.get_data())

        # Convert images to PIL format
        color_image = Image.fromarray(color_image_np)

        # Process depth and color images to isolate objects
        objects_image, mask_image = process_frame(color_image_np, depth_image_np)

        # Save the images to disk
        color_path = os.path.join(output_dir, f"frame_{frame_count}_color.jpg")
        objects_path = os.path.join(output_dir, f"frame_{frame_count}_objects.jpg")
        mask_path = os.path.join(output_dir, f"frame_{frame_count}_mask.jpg")
        color_image.save(color_path)
        objects_image.save(objects_path)
        mask_image.save(mask_path)

        print(f"Saved: {color_path}, {objects_path}, {mask_path}")

        # Show images inline
        show_images(color_image, objects_image, mask_image)

        frame_count += 1

        # Simulate processing delay (e.g., every 2 seconds)
        time.sleep(2)

finally:
    pipeline.stop()
