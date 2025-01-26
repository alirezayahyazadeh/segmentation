import numpy as np
import pyrealsense2 as rs
from PIL import Image
import open3d as o3d
from rembg import remove

def main():
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Obtain the stream profile and extract camera intrinsic data
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = depth_profile.get_intrinsics()

    # Setup Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window('3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Remove the background from the color image
            color_image_no_bg = remove(color_image)
            if color_image_no_bg.shape[2] != 3:
                color_image_no_bg = color_image_no_bg[..., :3]  # Ensure RGB format

            # Generate the point cloud
            pcd = depth_to_pointcloud(color_image_no_bg, depth_image, intrinsics)

            # Update the point cloud
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop and close everything properly
        pipeline.stop()
        vis.destroy_window()

def depth_to_pointcloud(color_image, depth_image, intrinsics):
    points = []
    colors = []

    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            depth = depth_image[v, u]
            if depth > 0:  # Filter out depth value of 0
                depth_point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
                points.append(depth_point)
                colors.append(color_image[v, u] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

if __name__ == "__main__":
    main()
