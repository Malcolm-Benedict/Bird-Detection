import cv2
import random
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

class YoloTracker:
    def __init__(self,model_name,do_verbose):
        """Construct the model

        Args:
            model_name (str): model name
        Returns:
            model: A YOLO model instance
        """
        self.model = YOLO(model_name, task="track", verbose=do_verbose)
        self.track_history = defaultdict(lambda: [])

    def get_results(self,frame,do_verbose,conf):
        """Get the results for a frame.

        Args:
            frame (image): An image object

        Returns:
            results (ultralytics results object): the results for this frame
        """
        results = self.model.track(frame, stream=True, persist=True, tracker="bytetrack.yaml",verbose=do_verbose, conf=conf)
        return results