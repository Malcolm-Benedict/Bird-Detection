import unittest
import cv2
import numpy as np
from birdDetection.tracker import BlobTracker


class BlobPipelineTests(unittest.TestCase):
    def make_tracker(self):
        tracker = BlobTracker({'warmup_frames': 5, 'blur_size': 0, 'close_size': 0,
                               'min_area': 10, 'max_area': 500,
                               'initialization_delay': 0, 'distance_threshold': 25}, 8)
        for _ in range(5):
            self.assertEqual(tracker.update(np.zeros((120, 200, 3), np.uint8)), [])
        tracker.learning_rate = 0
        return tracker

    def test_two_objects_keep_ids_and_missing_frames_do_not_extend_history(self):
        tracker = self.make_tracker()
        identities = []
        for index in range(8):
            frame = np.zeros((120, 200, 3), np.uint8)
            cv2.circle(frame, (25 + 4 * index, 30), 4, (255, 255, 255), -1)
            cv2.circle(frame, (150 - 3 * index, 90), 4, (255, 255, 255), -1)
            objects = tracker.update(frame)
            self.assertEqual(len(objects), 2)
            identities.append([o['id'] for o in sorted(objects, key=lambda o: o['center'][1])])
        self.assertTrue(all(ids == identities[0] for ids in identities))
        lengths = {key: len(value) for key, value in tracker.track_history.items()}
        self.assertEqual(tracker.update(np.zeros((120, 200, 3), np.uint8)), [])
        self.assertEqual(lengths, {key: len(value) for key, value in tracker.track_history.items()})
        for _ in range(20):
            tracker.update(np.zeros((120, 200, 3), np.uint8))
        self.assertFalse(tracker.track_history)

    def test_area_filter(self):
        tracker = self.make_tracker()
        frame = np.zeros((120, 200, 3), np.uint8)
        frame[10, 10] = 255
        frame[30:70, 30:70] = 255
        self.assertEqual(tracker.update(frame), [])

    def test_invalid_kernel(self):
        with self.assertRaises(ValueError):
            BlobTracker({'blur_size': 2})


if __name__ == '__main__':
    unittest.main()
