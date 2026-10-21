"""Stationary-camera motion segmentation and centroid tracking."""
from collections import defaultdict, deque

import cv2
import numpy as np
from norfair import Detection, Tracker


class BlobTracker:
    def __init__(self, config=None, track_length=20):
        cfg = config or {}
        self.min_area = float(cfg.get('min_area', 12))
        self.max_area = float(cfg.get('max_area', 20000))
        self.blur_size = int(cfg.get('blur_size', 3))
        self.open_size = int(cfg.get('open_size', 0))
        self.close_size = int(cfg.get('close_size', 3))
        self.warmup_frames = int(cfg.get('warmup_frames', 30))
        self.learning_rate = float(cfg.get('learning_rate', -1))
        if not 0 < self.min_area <= self.max_area:
            raise ValueError('Require 0 < min_area <= max_area')
        for size in (self.blur_size, self.open_size, self.close_size):
            if size < 0 or (size and size % 2 == 0):
                raise ValueError('Kernel sizes must be zero or positive odd integers')
        if track_length < 4 or self.warmup_frames < 0:
            raise ValueError('track_length must be >= 4 and warmup_frames >= 0')
        if self.learning_rate != -1 and not 0 <= self.learning_rate <= 1:
            raise ValueError('learning_rate must be -1 or between 0 and 1')
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.get('history', 500)),
            varThreshold=float(cfg.get('var_threshold', 16)), detectShadows=False)
        self.tracker = Tracker(
            distance_function='euclidean',
            distance_threshold=float(cfg.get('distance_threshold', 50)),
            hit_counter_max=int(cfg.get('hit_counter_max', 15)),
            initialization_delay=int(cfg.get('initialization_delay', 3)))
        self.track_history = defaultdict(lambda: deque(maxlen=track_length))
        self.frame_index = 0
        self.mask = None

    def update(self, frame):
        self.frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_size > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        self.mask = self.background.apply(gray, learningRate=self.learning_rate)
        for size, operation in ((self.open_size, cv2.MORPH_OPEN),
                                (self.close_size, cv2.MORPH_CLOSE)):
            if size > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                self.mask = cv2.morphologyEx(self.mask, operation, kernel)
        detections = []
        if self.frame_index > self.warmup_frames:
            count, _, stats, centers = cv2.connectedComponentsWithStats(self.mask)
            for label in range(1, count):
                x, y, w, h, area = map(int, stats[label])
                if self.min_area <= area <= self.max_area:
                    detections.append(Detection(
                        points=np.asarray([centers[label]], dtype=float),
                        data={'frame': self.frame_index, 'box': (x, y, w, h), 'area': area}))
        objects = self.tracker.update(detections=detections)
        alive_ids = {obj.id for obj in objects}
        for track_id in list(self.track_history):
            if track_id not in alive_ids:
                del self.track_history[track_id]
        observed = []
        for obj in objects:
            # Predictions bridge gaps internally, but aren't collision evidence.
            detection = obj.last_detection
            if detection.data['frame'] != self.frame_index:
                continue
            center = tuple(map(float, detection.points[0]))
            self.track_history[obj.id].append(center)
            observed.append({'id': obj.id, 'center': center, **detection.data})
        return observed
