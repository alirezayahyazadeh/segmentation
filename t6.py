import numpy as np
import pyrealsense2 as rs
from PIL import Image
import open3d as o3d

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

# Function to generate 3D point cloud
def generate_point_cloud(color_image, depth_image, intrinsics):
    # Create masks based on depth thresholds
    depth_mask = (depth_image > min_depth) & (depth_image < max_depth)
    valid_depth = np.where(depth_mask, depth_image, 0)

    # Generate points and corresponding colors
    points = []
    colors = []
    for y in range(valid_depth.shape[0]):
        for x in range(valid_depth.shape[1]):
            depth = valid_depth[y, x]
            if depth > 0:
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                points.append(point)
                colors.append(color_image[y, x] / 255.0)  # Normalize the color
    return np.array(points), np.array(colors)

# Function to visualize and save the point cloud
def visualize_and_save(points, colors, output_file):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save and visualize the point cloud
    o3d.io.write_point_cloud(output_file, point_cloud)
    print(f"Point cloud saved to {output_file}")
    o3d.visualization.draw_geometries([point_cloud])

try:
    pipeline.start(config)
    print("Pipeline started. Capturing frames...")

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise Exception("Failed to capture frames from the RealSense camera.")

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    points, colors = generate_point_cloud(color_image, depth_image, intrinsics)
    output_file = "output_point_cloud.ply"
    visualize_and_save(points, colors, output_file)

    # Enhanced segmentation by removing background based on depth
    mask = (depth_image >= min_depth) & (depth_image <= max_depth)
    segmented_image = np.where(mask[:,:,None], color_image, 0)
    img = Image.fromarray(segmented_image)
    img.save("segmented_image.jpg")
    img.show()

    print("Segmented image saved as 'segmented_image.jpg'.")

finally:
    pipeline.stop()
    print("Pipeline stopped.")
