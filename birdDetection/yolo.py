import cv2
import random
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import torch

#model = YOLO("yolo26n.pt")
#model.export(format="engine")  # dla:0 or dla:1 corresponds to the DLA cores
yolo = YOLO("yolo26n.engine", task="detect")

track_history = defaultdict(lambda: [])
def getColors(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor_id=0 ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1, format=NV12 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d ! "
        "nvvidconv !"
        #"videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        "appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


videoCap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not videoCap.isOpened():
    print("Error: Cannot open video capture using GStreamer")
    exit()
frame_width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('output.mp4', fourcc, 30, (frame_width, frame_height))

while True:
    ret, frame = videoCap.read()
    if cv2.waitKey(1) == ord('q'):
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
    cv2.imshow('Camera', frame)
    out.write(frame)
videoCap.release()
out.release()
cv2.destroyAllWindows()