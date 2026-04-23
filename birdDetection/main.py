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

# Command line arguments
parser = argparse.ArgumentParser(prog='birdDetection', description='A program to detect birds in frame, using a YOLO backbone')
parser.add_argument('configPath', help='Path to the config. file. File must be a valid yaml file')
parser.add_argument('-v', '--verbose', help='Display the YOLO models output info for every frame', action='store_true')
cmd_args = parser.parse_args()
configPath = cmd_args.configPath

# Load params. from yaml file
args = Args(configPath)
videoCap = args.make_video_source()
detector = args.make_detector()
model = args.model
conf = args.model_conf
    
def exit_handler():
    videoCap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
        print("Saving!")  
    except:
        _ = 0
    print("Exiting.")
atexit.register(exit_handler)

# Throw an error if unable to open the video cap.
if not videoCap.isOpened():
        print("Error: Cannot open video capture")
        exit()
        
# Params to save video
if args.SAVE_OUTPUT:
    frameWidth = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourCC = cv2.VideoWriter.fourcc(*'mp4v')  # Codec
    current_time = str(datetime.datetime.now().isoformat())
    saveName = args.output_path+'output-'+current_time+'.mp4'
    out = cv2.VideoWriter(saveName, fourCC, 30, (frameWidth, frameHeight))

# Initialize the tracker
tracker = YoloTracker(args.model_path + model,cmd_args.verbose)
ignore_list = []
override_frames = 0
global diamond
# Loop to process incoming frames
while videoCap.isOpened(): 
    ret, frame = videoCap.read()
    if not ret:
        break
    
    # Call the tracker
    results = tracker.get_results(frame, cmd_args.verbose,conf)
    for result in results:
        if result.boxes and result.boxes.is_track:
            
            # Plot result frame
            boxes = result.boxes.xywh
            class_ids = result.boxes.cls
            track_ids = result.boxes.id.int().tolist() # This line will throw a warning, its fine I believe it is the result of a vague class output from ultralytics
            probs = result.boxes.conf.tolist()
            frame = result.plot()
            
            for box, cls, track_id, prob in zip(boxes, class_ids, track_ids, probs): # Iterate over all results
                name = result.names[int(cls)]
                #print(str(name) + ": " + str(prob)) #Unused display argument
                x, y, w, h = box
                
                # Add point to track and discard oldest point if > max length
                track = tracker.track_history[track_id]
                track.append((float(x), float(y)))  
                if len(track) > args.track_length:  
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                
                # If filtering detector by object names, only run it on tracks with that name
                if args.target_name:
                    valid_name = args.target_name
                else:
                    valid_name = name # If no name is specified, set the name valid name equal to the current one to process all tracks
                
                # Only run the detector for valid tracks    
                if (name == valid_name) and (prob > args.confidence_threshold) and (track_id not in map(operator.itemgetter(0),ignore_list)) and (len(track) >= 4) and (override_frames < 1):
                    COLLISION_DETECTED = detector.detect(track) # The COLLISION_DETECTED flag is used for modularity
                    if COLLISION_DETECTED:
                        print("\033[31mCOLLISION!\033[0m")
                        diamond = np.array([[int(x),int(y-10)],[int(x+10),int(y)],[int(x),int(y+10)],[int(x-10),int(y)]])
                        ignore_list.append([track_id, args.refractory_frames]) # Add the box to the ignore list with a timer value
                        override_frames += args.override_frames
                        
                # Make randomly colored boxes for display
                random.seed(track_id)
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
    
    # Reduce the timer for all tracks on the ignore list            
    for i in ignore_list:
        try:
            RED = (0, 0, 255)
            cv2.fillPoly(frame, [diamond], RED)
        except:
            _ = 0
        i[1] = i[1] - 1
        if i[1] < 1: ignore_list.remove(i) # Remove tracks that have been on the list for more than a given amount of time

    if override_frames > 0: override_frames -= 1
    
    # Display and save output.
    if args.SHOW_OUTPUT:
        try:
            cv2.imshow('Camera',frame)
            cv2.waitKey(1)
        except:
            _ = 0
    
    try:
        out.write(frame)
    except:
        _ = 0
    