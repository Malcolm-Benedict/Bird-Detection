import cv2
import random
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

class YoloTracker:
    def __init__(self,model):
        self.model = YOLO(model)
        self.track_history = defaultdict(lambda: [])
     
    def get_colors(self,num):
        """Generate unique colors for each class ID"""
        random.seed(num)
        return tuple(random.randint(0, 255) for _ in range(3))

    def track_frame(self,frame):
        results = self.model.track(frame, persist=True)
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
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 100:  # retain for 60 frames
                        track.pop(0)
                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    color = self.get_colors(track_id)
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
        return frame
