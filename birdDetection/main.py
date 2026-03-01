import cv2
import numpy as np
from tracker import YoloTracker
import datetime
import argparse

MODEL_PATH = 'models/'
OUTPUT_PATH = 'outputs/'

current_time = str(datetime.datetime.now().isoformat())

parser = argparse.ArgumentParser(
                    prog='birdDetection',
                    description='detects birds',
                    epilog='todo')
parser.add_argument('model')
parser.add_argument('videoSource')
parser.add_argument('-s','--save', action="store_true")
args = parser.parse_args()
model = args.model
if args.save:
    SAVE_OUTPUT = True
else:
    SAVE_OUTPUT = False


videoCap = cv2.VideoCapture(0)
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