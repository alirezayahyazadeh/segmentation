import cv2
import numpy as np
from PIL import Image
import random
import os
import pyrealsense2 as rs  # Import RealSense library

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

min_object_distance_in_meters = 1.0
max_object_distance_in_meters = 1.3
min_depth_threshold = int(min_object_distance_in_meters * 1000)
max_depth_threshold = int(max_object_distance_in_meters * 1000)

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def generate_random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

def process_frame(color_image, depth_image):
    segmented_image = color_image.copy()
    objects_image = np.zeros_like(color_image, dtype=np.uint8)

    for _ in range(3):
        color = generate_random_color()
        mask_np = (depth_image > min_depth_threshold) & (depth_image < max_depth_threshold)

        objects_image[mask_np] = color_image[mask_np]

        x1, y1, x2, y2 = np.random.randint(0, 640, size=2).tolist() + np.random.randint(0, 480, size=2).tolist()
        cv2.rectangle(segmented_image, (x1, y1), (x2, y2), color, 2)

        label_text = f"object {np.random.randint(1, 100)}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x, text_y = x1, max(y1 - 10, 10)

        cv2.rectangle(
            segmented_image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y),
            color, -1
        )
        cv2.putText(segmented_image, label_text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return segmented_image, objects_image

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipeline.start(config)
    frame_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        segmented_image, objects_image = process_frame(color_image, depth_image)

        segmented_image_path = os.path.join(output_dir, f"frame_{frame_count}_segmented.jpg")
        objects_image_path = os.path.join(output_dir, f"frame_{frame_count}_objects.jpg")
        cv2.imwrite(segmented_image_path, segmented_image)
        cv2.imwrite(objects_image_path, objects_image)

        print(f"Saved: {segmented_image_path} and {objects_image_path}")
        frame_count += 1

        cv2.imshow('Segmented Image', segmented_image)
        cv2.imshow('Objects Only', objects_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
