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
import operator

parser = argparse.ArgumentParser(prog='birdDetection', description='detects birds', epilog='todo')
parser.add_argument('configPath')
cmd_args = parser.parse_args()
configPath = cmd_args.configPath

args = Args(configPath)
videoCap = args.make_video_source()
detector = args.make_detector()
model = args.model

if args.SAVE_OUTPUT:
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
saveName = args.output_path+'output-'+current_time+'.mp4'
out = cv2.VideoWriter(saveName, fourCC, 30, (frameWidth, frameHeight))
tracker = YoloTracker(args.model_path + model)
ignore_list = []

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
            probs = result.boxes.conf.tolist()
            frame = result.plot()
            for box, cls, track_id, prob in zip(boxes, class_ids, track_ids, probs):
                name = result.names[int(cls)]
                #print(str(name) + ": " + str(prob))
                x, y, w, h = box
                track = tracker.track_history[track_id]
                track.append((float(x), float(y)))  
                if len(track) > args.track_length:  
                    track.pop(0)
                if args.target_name:
                    valid_name = args.target_name
                else:
                    valid_name = name
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                if (name == valid_name) and (prob > args.confidence_threshold) and (track_id not in map(operator.itemgetter(0),ignore_list)) and (len(track) >= 2):
                    COLLISION_DETECTED = detector.detect(track)
                    if COLLISION_DETECTED:
                        print("\033[31mCOLLISION!\033[0m")
                        ignore_list.append([track_id, args.refractory_frames])
                random.seed(track_id)
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
    for i in ignore_list:
        i[1] = i[1] - 1
        if i[1] < 1: ignore_list.remove(i)
    print(ignore_list)

    cv2.imshow('Camera',frame)
    out.write(frame)
    cv2.waitKey(1)