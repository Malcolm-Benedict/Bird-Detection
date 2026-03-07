import cv2
import random
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

class YoloTracker:
    def __init__(self,model):
        self.model = YOLO(model, task="detect")
        self.track_history = defaultdict(lambda: [])

    def get_results(self,frame):
        results = self.model.track(frame, persist=True)
        return results
    def make_engine(self):
        self.model.export(format="engine")