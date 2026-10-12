"""Keep a rolling second of footage and save a clip for each collision."""

from bisect import bisect_right
from collections import deque
from datetime import datetime
from pathlib import Path
import math

import imageio_ffmpeg


class CollisionRecorder:
    def __init__(self, output_path, fps):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.buffer = deque()
        self.pending = []

    def add_frame(self, frame, timestamp, collisions=0):
        # Copies protect footage from subsequent drawing or capture reuse.
        sample = (timestamp, frame.copy())
        self.buffer.append(sample)
        # Retain the frame spanning the start of the one-second window.
        while len(self.buffer) > 1 and self.buffer[1][0] <= timestamp - 1:
            self.buffer.popleft()
        for clip in self.pending:
            clip['frames'].append(sample)
        for _ in range(collisions):
            self.pending.append({
                'start': max(timestamp - 1, self.buffer[0][0]),
                'end': timestamp + 1,
                'frames': list(self.buffer),
            })
        for clip in self.pending[:]:
            if timestamp >= clip['end']:
                self._save(clip, clip['end'])
                self.pending.remove(clip)

    def _save(self, clip, end):
        frames = clip['frames']
        height, width = frames[0][1].shape[:2]
        path = self.output_path / f"collision-{datetime.now():%Y%m%d-%H%M%S-%f}.mp4"
        # H.264 with yuv420p supports browser and desktop video players.
        writer = imageio_ffmpeg.write_frames(
            str(path), (width, height), fps=self.fps,
            pix_fmt_in='bgr24', pix_fmt_out='yuv420p', codec='libx264',
            macro_block_size=2, output_params=['-movflags', '+faststart'])
        times = [sample[0] for sample in frames]
        try:
            writer.send(None)
            # Resample elapsed time so slow inference doesn't shorten live clips.
            count = max(1, math.ceil((end - clip['start']) * self.fps))
            for index in range(count):
                timestamp = clip['start'] + index / self.fps
                source = max(0, bisect_right(times, timestamp) - 1)
                writer.send(frames[source][1].tobytes())
        finally:
            writer.close()
        print(f"Saved collision clip: {path}")

    def close(self):
        # Save available footage if capture stops before the post-event second.
        while self.pending:
            clip = self.pending.pop(0)
            self._save(clip, min(clip['end'], clip['frames'][-1][0]))
