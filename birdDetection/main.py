import cv2
import numpy as np
from tracker import YoloTracker
from detector import GeometryMethod
import datetime
import argparse
import yaml
import os
import atexit
import random

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
    
MODEL_PATH = 'models/'
OUTPUT_PATH = 'outputs/'
VIDEO_PATH = 'videos/'
CONFIG_PATH = 'config/'

current_time = str(datetime.datetime.now().isoformat())

parser = argparse.ArgumentParser(
                    prog='birdDetection',
                    description='detects birds',
                    epilog='todo')
parser.add_argument('model')
parser.add_argument('-s','--save', action="store_true")
subparsers = parser.add_subparsers(dest='source', help='Video sources')

webcamSP = subparsers.add_parser("webcam")
videoSP = subparsers.add_parser("video")
videoSP.add_argument("video_path")
gsSP = subparsers.add_parser("gstreamer")
gsSP.add_argument("config_path")
args = parser.parse_args()
model = args.model
source = args.source

if args.save:
    DESTROY_OUTPUT = False
else:
    DESTROY_OUTPUT = True
if source == "webcam":
    videoCap = cv2.VideoCapture(0)
elif source == "video":
    videoCap = cv2.VideoCapture(VIDEO_PATH+args.video_path)
elif source == "gstreamer": # incorrect syntax
    with open(CONFIG_PATH+args.config_path, 'r') as file:
        cfg = yaml.safe_load(file)
        pipe = gstreamer_pipeline()
        print(pipe)
        videoCap = cv2.VideoCapture(gstreamer_pipeline(),cv2.CAP_GSTREAMER)
else:
    print("Please specify video source!")
    exit()


def exit_handler():
    videoCap.release()
    out.release()
    cv2.destroyAllWindows()
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
saveName = OUTPUT_PATH+'output-'+current_time+'.mp4'
out = cv2.VideoWriter(saveName, fourCC, 30, (frameWidth, frameHeight))
tracker = YoloTracker(MODEL_PATH+str(model))
detector = GeometryMethod(45)

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