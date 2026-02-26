import cv2
import random
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

yolo = YOLO("yolov8s.pt")
track_history = defaultdict(lambda: [])
def getColors(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/Mallard.mp4"


videoCap = cv2.VideoCapture(0)
if not videoCap.isOpened():
    print("Error: Cannot open video capture")
    exit()
frame_width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('output.mp4', fourcc, 30, (frame_width, frame_height))

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if not ret:
        break
    results = yolo.track(frame, persist=True)
    result = results[0]
    class_names = result.names

        # Get the boxes and track IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        names = result.names
        track_ids = result.boxes.id.int().cpu().tolist()
            # Visualize the result on the frame
        frame = result.plot()
            # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 100:  # retain for 60 frames
                    track.pop(0)
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                color = getColors(track_id)
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
    cv2.imshow('Camera',frame)
    out.write(frame)
    cv2.waitKey(1)
videoCap.release()
out.release()
cv2.destroyAllWindows()