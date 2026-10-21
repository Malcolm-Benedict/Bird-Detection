# Bird detection

Stationary-camera pipeline: OpenCV MOG2 background subtraction → mask cleanup → connected components and area filtering → Norfair centroid
tracking → annotated trails, separate track videos, and collision clips.

From the repository root:

```sh
python -m pip install -r requirements.txt
python -m birdDetection.main config/laptop.yaml
```

Use `config/jetson.yaml` for the configured GStreamer camera. On Jetson, use an
OpenCV build with GStreamer support (the standard pip wheel may lack it).
Press Q or Escape to stop when the preview is enabled. For headless processing,
set `general.show_output: false`. Set `general.input_method: video` and the
`paths.video_path` / `video` fields to process a recording.

The first `blob.warmup_frames` frames learn the background without reporting
tracks. Adjust `min_area` and `max_area` in square pixels; all coordinates refer
to the original frame. `open_size: 0` preserves tiny birds; larger odd kernels
remove specks but may erase distant targets. `show_mask` displays segmentation
alongside the preview. `distance_threshold` is the maximum centroid matching
distance in pixels. `initialization_delay` suppresses short-lived tracks, and
`hit_counter_max` controls how long established tracks can survive missed detections.

`save_tracks: true` saves a separate full-frame H.264 MP4 for every confirmed
track in `paths.output_path/tracks/`. Filenames include the session time and track
ID. Each video highlights only its own candidate, with its box, ID, and trail;
other objects may still be visible in the scene. Recording starts at the first
confirmed observation (after `initialization_delay`) and ends when Norfair drops
the track or capture stops. Brief detection gaps stay in the same clip, with a
"temporarily lost" label and no stale box. Live footage is resampled to preserve
elapsed time, and clips stream to disk instead of accumulating in memory.

No CSV positions are written. Set `save_tracks: false` to disable track videos.
`save_output` separately controls the existing one-second-before/after collision
clips. Predicted positions aren't added to collision history.

Blobs are **unclassified candidates**, not confirmed birds. YOLO weights and
class-confidence filters are no longer used by this pipeline. Camera movement,
foliage, insects, lighting changes, and overlapping objects can cause false
tracks or identity switches. The existing geometric collision heuristic is
unchanged and its events are candidates, not verified collisions.

Run synthetic segmentation/tracking checks:

```sh
python -m unittest discover -s tests -v
```
