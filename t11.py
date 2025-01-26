import numpy as np
import pyrealsense2 as rs
from PIL import Image
import open3d as o3d
from rembg import remove
import torch
import cv2


def main():
    # Initialize YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # YOLOv5 small model
    model.eval()

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

            # Run YOLOv5 detection
            results = model(color_image)
            detections = results.xyxy[0].numpy()  # Get bounding boxes (xmin, ymin, xmax, ymax, confidence, class)

            # Remove the background from the color image
            color_image_no_bg = remove(color_image)
            if color_image_no_bg.shape[2] != 3:
                color_image_no_bg = color_image_no_bg[..., :3]  # Ensure RGB format

            # Filter depth and color data based on YOLOv5 detections
            filtered_color_image = filter_image_with_detections(color_image_no_bg, detections)
            filtered_depth_image = filter_depth_with_detections(depth_image, detections)

            # Generate the point cloud
            pcd = depth_to_pointcloud(filtered_color_image, filtered_depth_image, intrinsics)

            # Update the point cloud
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Display detections on the original color image
            annotated_image = annotate_detections(color_image, detections)
            cv2.imshow("Detections", annotated_image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop and close everything properly
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


def filter_image_with_detections(image, detections):
    """
    Filters the color image by zeroing out areas outside the detected bounding boxes.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for xmin, ymin, xmax, ymax, _, _ in detections:
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        mask[ymin:ymax, xmin:xmax] = 255

    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    return filtered_image


def filter_depth_with_detections(depth_image, detections):
    """
    Filters the depth image by zeroing out areas outside the detected bounding boxes.
    """
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    for xmin, ymin, xmax, ymax, _, _ in detections:
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        mask[ymin:ymax, xmin:xmax] = 255

    filtered_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    return filtered_depth


def annotate_detections(image, detections):
    """
    Annotates the image with bounding boxes and class labels from YOLOv5 detections.
    """
    annotated_image = image.copy()
    for xmin, ymin, xmax, ymax, conf, cls in detections:
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        label = f"{int(cls)} {conf:.2f}"
        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(annotated_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image


def depth_to_pointcloud(color_image, depth_image, intrinsics):
    """
    Converts depth and color images to a 3D point cloud.
    """
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
