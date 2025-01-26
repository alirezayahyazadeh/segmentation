import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from PIL import Image
from rembg import remove
import cv2

def main():
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get camera intrinsic parameters
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = depth_profile.get_intrinsics()

    # Precompute intrinsic matrix
    intrinsic_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy],
                                  [0, 0, 1]])

    # Setup Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window('3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Validate shapes of color and depth images
            if color_image.shape[:2] != depth_image.shape:
                raise ValueError("Color and depth image dimensions do not match!")

            # Debugging: Print image shapes
            print(f"Color image shape: {color_image.shape}, Depth image shape: {depth_image.shape}")

            try:
                # Remove the background using rembg
                color_image_pil = Image.fromarray(color_image)
                color_image_no_bg = remove_background(color_image_pil)

                # Convert the processed image back to NumPy
                color_image_no_bg_np = np.array(color_image_no_bg)

                # Resize the background-removed image to match the depth image
                color_image_no_bg_np = cv2.resize(color_image_no_bg_np, (depth_image.shape[1], depth_image.shape[0]))
            except Exception as e:
                print(f"Error during background removal or resizing: {e}")
                continue

            try:
                # Generate 3D point cloud
                points, colors = generate_point_cloud(color_image_no_bg_np, depth_image, intrinsic_matrix)

                # Debugging: Check number of points
                print(f"Generated {len(points)} 3D points.")
            except Exception as e:
                print(f"Error during point cloud generation: {e}")
                continue

            if points.shape[0] == 0:
                print("No points generated in point cloud!")
                continue

            # Update point cloud geometry
            try:
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            except Exception as e:
                print(f"Error during point cloud visualization update: {e}")
                continue

            # Display the original and processed images side by side
            try:
                combined_image = np.hstack((color_image, color_image_no_bg_np))
                cv2.imshow('Original and Background Removed', combined_image)
            except Exception as e:
                print(f"Error during image display: {e}")
                continue

            if cv2.waitKey(1) == 27:  # Exit on 'ESC' key
                break

    except Exception as e:
        print(f"An error occurred in the main loop: {e}")

    finally:
        # Stop and close everything properly
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


def remove_background(image):
    """
    Use rembg to remove the background from an image.
    """
    try:
        return remove(image)
    except Exception as e:
        print(f"Background removal failed: {e}")
        return image


def generate_point_cloud(color_image, depth_image, intrinsic_matrix):
    """
    Generate a point cloud using vectorized depth-to-3D transformation.
    """
    try:
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        uv = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)

        depth_flat = depth_image.flatten() / 1000.0  # Convert depth to meters
        valid = (depth_flat > 0) & (depth_flat < 10)  # Remove invalid depth points

        # Debugging: Print number of valid depth points
        print(f"Number of valid depth points: {np.sum(valid)}")
        print(f"Depth flat shape: {depth_flat.shape}, UV shape: {uv.shape}, Valid mask shape: {valid.shape}")

        # Apply the valid mask to depth and color data
        uv_valid = uv[valid]
        depth_valid = depth_flat[valid]

        # Reshape the color image and apply the mask along the first dimension
        color_flat = color_image.reshape(-1, 3)
        colors_valid = color_flat[valid]

        # Debugging: Ensure color_flat and valid have compatible shapes
        print(f"Color flat shape: {color_flat.shape}, Colors valid shape: {colors_valid.shape}")

        if depth_valid.shape[0] == 0:
            print("No valid depth points found!")
            return np.empty((0, 3)), np.empty((0, 3))

        # Transform pixel coordinates to 3D points
        intrinsic_inv = np.linalg.inv(intrinsic_matrix)
        points_3d = depth_valid[:, None] * (uv_valid @ intrinsic_inv.T)

        # Normalize colors to [0, 1] range
        colors = colors_valid / 255.0

        return points_3d, colors
    except Exception as e:
        print(f"Error during point cloud generation: {e}")
        return np.empty((0, 3)), np.empty((0, 3))

if __name__ == "__main__":
    main()
