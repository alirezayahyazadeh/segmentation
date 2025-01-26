import numpy as np
import pyrealsense2 as rs
from PIL import Image
import open3d as o3d  # For point cloud visualization and generation

# Set up the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Depth range thresholds (in meters)
min_object_distance = 1.0
max_object_distance = 1.3
min_depth = int(min_object_distance * 1000)  # Convert meters to millimeters
max_depth = int(max_object_distance * 1000)

# Function to process and generate 3D point cloud
def generate_point_cloud(color_image, depth_image, intrinsics):
    # Flatten depth image within the valid range
    depth_mask = (depth_image > min_depth) & (depth_image < max_depth)
    valid_depth = np.where(depth_mask, depth_image, 0)

    # Generate 3D points (point cloud)
    points = []
    colors = []
    for y in range(valid_depth.shape[0]):
        for x in range(valid_depth.shape[1]):
            depth = valid_depth[y, x]
            if depth > 0:
                # Map pixel to 3D space
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                points.append(point)
                colors.append(color_image[y, x] / 255.0)  # Normalize color to [0, 1]
    return np.array(points), np.array(colors)

# Function to save and visualize the output
def visualize_and_save(points, colors, output_file):
    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save as a .ply file
    o3d.io.write_point_cloud(output_file, point_cloud)
    print(f"Point cloud saved to {output_file}")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

try:
    pipeline.start(config)
    print("Pipeline started. Capturing frame...")

    # Capture a single frame (color + depth)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise Exception("Failed to capture frames from the RealSense camera.")

    # Convert frames to NumPy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Get the camera intrinsics for depth-to-3D projection
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # Remove background, segment objects, and generate 3D points
    points, colors = generate_point_cloud(color_image, depth_image, intrinsics)

    # Save and visualize the point cloud
    output_file = "output_point_cloud.ply"
    visualize_and_save(points, colors, output_file)

    # Save the segmented 2D image (with the background removed)
    color_image[depth_image < min_depth] = [0, 0, 0]  # Remove close objects
    color_image[depth_image > max_depth] = [0, 0, 0]  # Remove far objects
    segmented_image = Image.fromarray(color_image)
    segmented_image.save("segmented_image.jpg")
    segmented_image.show()

    print("Segmented image saved as 'segmented_image.jpg'.")

finally:
    pipeline.stop()
    print("Pipeline stopped.")
