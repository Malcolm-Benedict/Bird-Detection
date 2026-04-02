import yaml
from detector import GeometryMethod
import cv2
class Args():
    """
    Load arguments from the specified yaml, and process some of them. The yaml is designed to be human readable, 
    """
    def __init__(self, yaml_path):  
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_path = self.config["paths"]["model_path"]
        self.output_path = self.config["paths"]["output_path"]
        
        self.video_source = self.config["general"]["input_method"]
        self.SAVE_OUTPUT = self.config["general"]["save_output"]
        self.track_length = self.config["general"]["track_length"]
        self.model = self.config["model"]
          
    def make_gstreamer_pipeline(self):
        pipe = self.config["gstreamer"]
        return (
            pipe["source"]+" ! "
            +pipe["file"]+"(memory:"+pipe["memory"]+"), "
            "width="+str(pipe["input_width"])+", height="+str(pipe["input_height"])+", "
            "format="+pipe["input_format"]+", framerate="+str(pipe["framerate"])+"/1 ! "
            "nvvidconv flip-method="+str(pipe["flip_method"])+" ! "
            +pipe["file"]+", width="+str(pipe["output_width"])+", height="+str(pipe["input_height"])+", format="+pipe["output_format"]+" ! "
            "videoconvert ! "
            +pipe["file"]+", format="+pipe["sink_format"]+" ! "+pipe["sink"]
        )

    def make_video_source(self):
        if self.video_source == "webcam":
            capture = cv2.VideoCapture(0)
        elif self.video_source == "video":
            capture = cv2.VideoCapture(self.config["paths"]["video_path"] + self.config["video"])
        elif self.video_source == "gstreamer":
            capture = cv2.VideoCapture(self.make_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            print("Please specify video source!")
            exit()
        return capture

    def make_detector(self):
        if self.config["detector"] == "GeometryMethod":
            detector = GeometryMethod(self.config["geometry_prams"]["threshold"])
        else:
            print("Please specify detector!")
            exit()
        return detector