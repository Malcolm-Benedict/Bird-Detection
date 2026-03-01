import cv2
import numpy as np
from tracker import YoloTracker
import datetime
import argparse
import yaml

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
    SAVE_OUTPUT = True
else:
    SAVE_OUTPUT = False

if source == "webcam":
    videoCap = cv2.VideoCapture(0)
elif source == "video":
    videoCap = cv2.VideoCapture(VIDEO_PATH+args.video_path)
elif source == "gstreamer": # incorrect syntax
    with open(CONFIG_PATH+args.config_path, 'r') as file:
        cfg = yaml.safe_load(file)
        videoCap = cv2.VideoCapture(cfg,cv2.CAP_GSTREAMER)
else:
    print("Please specify video source!")
    exit()

if not videoCap.isOpened():
        print("Error: Cannot open video capture")
        exit()
frameWidth = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourCC = cv2.VideoWriter.fourcc(*'mp4v')  # Codec
if SAVE_OUTPUT:
    out = cv2.VideoWriter(OUTPUT_PATH+'output-'+current_time+'.mp4', fourCC, 30, (frameWidth, frameHeight))
tracker = YoloTracker(MODEL_PATH+str(model))

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if not ret:
        break
    frame = tracker.track_frame(frame)
    cv2.imshow('Camera',frame)
    if SAVE_OUTPUT:
        out.write(frame)
    cv2.waitKey(1)
videoCap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()