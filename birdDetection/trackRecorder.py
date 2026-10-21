"""Stream one annotated MP4 per confirmed track, including brief detection gaps."""
from datetime import datetime
import math
from pathlib import Path
import random

import cv2
import imageio_ffmpeg
import numpy as np


def draw_track(frame, track_id, track, box=None):
    """Annotate a single candidate in place; omit stale boxes on missed frames."""
    color = tuple(random.Random(track_id).sample(range(64, 256), 3))
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    elif track:
        x, y = map(int, track[-1])
    else:
        return
    label = f'Candidate {track_id}'
    if box is None:
        label += ' (temporarily lost)'
    cv2.putText(frame, label, (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    points = np.asarray(track, dtype=np.int32).reshape((-1, 1, 2))
    if len(points) > 1:
        cv2.polylines(frame, [points], False, color, 2)


class _TrackVideo:
    def __init__(self, path, frame, fps, timestamp):
        height, width = frame.shape[:2]
        self.path = path
        self.fps = fps
        self.start = timestamp
        self.last_timestamp = timestamp
        self.frame_count = 0
        self.previous = None
        self.writer = imageio_ffmpeg.write_frames(
            str(path), (width, height), fps=fps,
            pix_fmt_in='bgr24', pix_fmt_out='yuv420p', codec='libx264',
            macro_block_size=2,
            output_params=['-threads', '1', '-movflags', '+faststart'])
        self.writer.send(None)

    def add_frame(self, frame, timestamp):
        # Sample on a fixed timeline so slow live processing preserves duration.
        elapsed = timestamp - self.start
        while self.previous is not None and self.frame_count / self.fps < elapsed - 1e-8:
            self.writer.send(self.previous)
            self.frame_count += 1
        self.previous = frame.tobytes()
        self.last_timestamp = timestamp
        if self.frame_count / self.fps <= elapsed + 1e-8:
            self.writer.send(self.previous)
            self.frame_count += 1

    def close(self):
        try:
            # Include the last sample, even when live timestamps fall between ticks.
            count = math.ceil((self.last_timestamp - self.start) * self.fps - 1e-8) + 1
            while self.previous is not None and self.frame_count < count:
                self.writer.send(self.previous)
                self.frame_count += 1
        finally:
            self.writer.close()
        print(f'Saved track clip: {self.path}')


class TrackRecorder:
    def __init__(self, output_path, fps):
        self.output_path = Path(output_path) / 'tracks'
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.session = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.active = {}

    def add_frame(self, frame, timestamp, observations, track_history):
        # Histories contain all confirmed live IDs, even during detection gaps.
        for track_id in list(self.active):
            if track_id not in track_history:
                self.active.pop(track_id).close()
        observed = {obj['id']: obj for obj in observations}
        for track_id, history in track_history.items():
            obj = observed.get(track_id)
            if track_id not in self.active and obj is None:
                continue
            annotated = frame.copy()
            draw_track(annotated, track_id, list(history), obj['box'] if obj else None)
            # H.264 yuv420p needs even dimensions; pad without resizing the scene.
            height, width = annotated.shape[:2]
            if height % 2 or width % 2:
                annotated = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2,
                                               cv2.BORDER_CONSTANT)
            if track_id not in self.active:
                path = self.output_path / f'track-{self.session}-id-{track_id}.mp4'
                self.active[track_id] = _TrackVideo(path, annotated, self.fps, timestamp)
            self.active[track_id].add_frame(annotated, timestamp)

    def close(self):
        error = None
        while self.active:
            _, video = self.active.popitem()
            try:
                video.close()
            except Exception as exc:
                error = exc
        if error is not None:
            raise error
