import cv2
import numpy as np
from tracker import YoloTracker
from detector import GeometryMethod
from argumentParser import Args
import yaml
import datetime
import argparse
import os
import atexit
import random

parser = argparse.ArgumentParser(prog='birdDetection', description='detects birds', epilog='todo')
parser.add_argument('configPath')
cmd_args = parser.parse_args()
configPath = cmd_args.configPath

args = Args(configPath)

model_path = args.model_path
output_path = args.output_path
video_source = args.video_source
SAVE_OUTPUT = args.SAVE_OUTPUT
track_length = args.track_length
videoCap = args.make_video_source()
detector = args.make_detector()
model = args.model

if SAVE_OUTPUT:
    DESTROY_OUTPUT = False
else:
    DESTROY_OUTPUT = True
    
def exit_handler():
    videoCap.release()
    out.release()
    cv2.destroyAllWindows()
    if not DESTROY_OUTPUT: # This isn't exactly elegant, but it prevents warnings.
        try:
            out.release()  
        except:
            print("Error, unable to save output")
    if DESTROY_OUTPUT: # This isn't exactly elegant, but it prevents warnings.
        try:
            os.remove(saveName)
        except:
            print("Error, unable to delete output")
atexit.register(exit_handler)

if not videoCap.isOpened():
        print("Error: Cannot open video capture")
        exit()
        
frameWidth = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourCC = cv2.VideoWriter.fourcc(*'mp4v')  # Codec
current_time = str(datetime.datetime.now().isoformat())
saveName = output_path+'output-'+current_time+'.mp4'
out = cv2.VideoWriter(saveName, fourCC, 30, (frameWidth, frameHeight))
tracker = YoloTracker(model_path + model)

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if not ret:
        break
    results = tracker.get_results(frame)
    for result in results:
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh
            class_ids = result.boxes.cls
            track_ids = result.boxes.id.int().tolist()
            frame = result.plot()
            for box, cls, track_id in zip(boxes, class_ids, track_ids):
                name = result.names[int(cls)]
                x, y, w, h = box
                track = tracker.track_history[track_id]
                track.append((float(x), float(y)))  
                if len(track) > 20:  
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                if True and (len(track) >= 2): # check cls
                    COLLISION_DETECTED = detector.detect(track)
                    if COLLISION_DETECTED:
                        print("\033[31mCOLLISION!\033[0m")
                random.seed(track_id)
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.imshow('Camera',frame)
    out.write(frame)
    cv2.waitKey(1)