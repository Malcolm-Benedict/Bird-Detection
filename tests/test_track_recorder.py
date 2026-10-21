from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import yaml

from birdDetection.trackRecorder import TrackRecorder


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise AssertionError(f'Cannot play {path}')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


class TrackRecorderTests(unittest.TestCase):
    def test_overlapping_tracks_gaps_and_expiry(self):
        with TemporaryDirectory() as directory:
            recorder = TrackRecorder(directory, 10)
            frame = np.zeros((121, 201, 3), np.uint8)
            histories = {1: [(30, 40)], 2: [(130, 90)]}
            observations = [{'id': 1, 'box': (25, 35, 10, 10)},
                            {'id': 2, 'box': (125, 85, 10, 10)}]
            try:
                recorder.add_frame(frame, 0, observations, histories)
                recorder.add_frame(frame, 0.1, observations[:1], histories)
                recorder.add_frame(frame, 0.2, observations[:1], {1: histories[1]})
                self.assertEqual(set(recorder.active), {1})
                recorder.add_frame(frame, 0.3, [], {})
                self.assertFalse(recorder.active)
            finally:
                recorder.close()
            self.assertFalse(frame.any(), 'Recording should not draw on the source frame')
            videos = list(Path(directory).rglob('*.mp4'))
            self.assertEqual(len(videos), 2)
            first = read_frames(next(p for p in videos if p.stem.endswith('id-1')))
            second = read_frames(next(p for p in videos if p.stem.endswith('id-2')))
            self.assertEqual((len(first), len(second)), (3, 2))
            self.assertEqual(first[0].shape, (122, 202, 3))
            # Each full-frame clip should show only its own box.
            self.assertGreater(first[0][35:47, 25:37].mean(), 10)
            self.assertLess(first[0][85:97, 125:137].mean(), 5)
            self.assertGreater(second[0][85:97, 125:137].mean(), 10)
            self.assertLess(second[0][35:47, 25:37].mean(), 5)

    def test_slow_live_capture_duration_and_shutdown(self):
        with TemporaryDirectory() as directory:
            recorder = TrackRecorder(directory, 10)
            try:
                for timestamp, shade in [(0, 0), (0.25, 60), (0.5, 120)]:
                    frame = np.full((120, 200, 3), shade, np.uint8)
                    recorder.add_frame(frame, timestamp, [{'id': 1, 'box': (25, 35, 10, 10)}],
                                       {1: [(30, 40)]})
            finally:
                recorder.close()
            recorder.close()
            frames = read_frames(next(Path(directory).rglob('*.mp4')))
            self.assertEqual(len(frames), 6)
            for frame, shade in zip(frames, [0, 0, 0, 60, 60, 120]):
                self.assertAlmostEqual(float(frame[-10:, -10:].mean()), shade, delta=6)

    def test_command_line_saves_track_videos_without_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.avi'
            output = root / 'output'
            video = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (200, 120))
            self.assertTrue(video.isOpened())
            try:
                for index in range(45):
                    frame = np.zeros((120, 200, 3), np.uint8)
                    if index >= 10:
                        cv2.circle(frame, (20 + 2 * (index - 10), 30), 4, (255, 255, 255), -1)
                        cv2.circle(frame, (170 - 2 * (index - 10), 90), 4, (255, 255, 255), -1)
                    video.write(frame)
            finally:
                video.release()
            repo = Path(__file__).resolve().parents[1]
            cfg = yaml.safe_load((repo / 'config/laptop.yaml').read_text())
            cfg['general'].update(input_method='video', show_output=False, save_output=False, save_tracks=True)
            cfg['paths'].update(video_path=str(root) + '/', output_path=str(output))
            cfg['video'] = source.name
            cfg['blob'].update(warmup_frames=5, blur_size=0, close_size=0)
            config = root / 'test.yaml'
            config.write_text(yaml.safe_dump(cfg))
            subprocess.run([sys.executable, '-m', 'birdDetection.main', str(config)],
                           cwd=repo, check=True, capture_output=True, text=True, timeout=45)
            clips = list(output.rglob('*.mp4'))
            self.assertEqual(len(clips), 2)
            self.assertFalse(list(output.rglob('*.csv')))
            for clip in clips:
                self.assertGreater(len(read_frames(clip)), 20)
            # Disabling track saving should produce no files when collisions are also disabled.
            cfg['general']['save_tracks'] = False
            disabled_output = root / 'disabled'
            cfg['paths']['output_path'] = str(disabled_output)
            config.write_text(yaml.safe_dump(cfg))
            subprocess.run([sys.executable, '-m', 'birdDetection.main', str(config)],
                           cwd=repo, check=True, capture_output=True, text=True, timeout=45)
            self.assertFalse(disabled_output.exists())


if __name__ == '__main__':
    unittest.main()
