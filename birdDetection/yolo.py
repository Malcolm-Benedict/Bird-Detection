import cv2
import random
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import torch

#model = YOLO("yolo26n.pt")
#model.export(format="engine")  # dla:0 or dla:1 corresponds to the DLA cores
yolo = YOLO("yolo26n.engine")

track_history = defaultdict(lambda: [])
def getColors(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "videos/Mallard.mp4"
videoCap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter.fourcc(*'MPEG')  # Codec
out = cv2.VideoWriter('output.mp4', fourcc, 24, (1280, 720))

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if not ret:
        break
    results = yolo.track(frame, persist=True)
    result = results[0]
    class_names = result.names

        # Get the boxes and track IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh
        names = result.names
        track_ids = result.boxes.id.int().tolist()
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
    out.write(frame)
videoCap.release()
out.release()
cv2.destroyAllWindows()