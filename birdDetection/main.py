"""Run motion-blob tracking and the existing collision recorder."""
import argparse
import time

import cv2
import numpy as np

if __package__:
    from .tracker import BlobTracker
    from .argumentParser import Args
    from .collisionRecorder import CollisionRecorder
    from .trackRecorder import TrackRecorder, draw_track
else:
    from tracker import BlobTracker
    from argumentParser import Args
    from collisionRecorder import CollisionRecorder
    from trackRecorder import TrackRecorder, draw_track


def main():
    parser = argparse.ArgumentParser(description='Track moving bird candidates with OpenCV and Norfair')
    parser.add_argument('configPath', help='Path to YAML configuration')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print observed track counts')
    cmd = parser.parse_args()
    args = Args(cmd.configPath)
    detector = args.make_detector()
    tracker = BlobTracker(args.config.get('blob', {}), args.track_length)
    capture = args.make_video_source()
    recorder = None
    track_recorder = None
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open video capture')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        if args.SAVE_OUTPUT:
            recorder = CollisionRecorder(args.output_path, fps)
        if args.config['general'].get('save_tracks', True):
            track_recorder = TrackRecorder(args.output_path, fps)
        frame_index = 0
        started = time.monotonic()
        ignore_until = {}
        override_until = 0
        print('Starting capture (stationary camera; blobs are unclassified bird candidates).')
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps if args.video_source == 'video' else time.monotonic() - started
            observations = tracker.update(frame)
            if track_recorder is not None:
                track_recorder.add_frame(frame, timestamp, observations, tracker.track_history)
            collisions = 0
            ignore_until = {key: expiry for key, expiry in ignore_until.items() if expiry > frame_index}
            for obj in observations:
                track_id = obj['id']
                cx, cy = obj['center']
                track = list(tracker.track_history[track_id])
                draw_track(frame, track_id, track, obj['box'])
                if (len(track) >= 4 and track_id not in ignore_until
                        and frame_index >= override_until and detector.detect(track)):
                    collisions += 1
                    ignore_until[track_id] = frame_index + args.refractory_frames
                    override_until = frame_index + args.override_frames
                    cv2.circle(frame, (int(cx), int(cy)), 12, (0, 0, 255), -1)
                    print(f'Collision candidate: track {track_id}')
            if cmd.verbose:
                print(f'Frame {frame_index}: {len(observations)} observed tracks')
            if recorder is not None:
                recorder.add_frame(frame, timestamp, collisions)
            frame_index += 1
            if args.SHOW_OUTPUT:
                cv2.imshow('Bird candidates', frame)
                if args.config['general'].get('show_mask', False):
                    cv2.imshow('Motion mask', tracker.mask)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    finally:
        capture.release()
        try:
            if recorder is not None:
                recorder.close()
        finally:
            try:
                if track_recorder is not None:
                    track_recorder.close()
            finally:
                if args.SHOW_OUTPUT:
                    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
