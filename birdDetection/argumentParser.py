import yaml
if __package__:
    from .detector import GeometryMethod
else:
    from detector import GeometryMethod
import cv2
class Args():
    """
    Load arguments from the specified yaml. General params :
    
    **model_path**   : File path to the model
    
    **output_path**  : File path to the output
    
    **video_source** : Input method
    
    **SAVE_OUTPUT**  : Flag to save the output
    
    **track_length** : How many points are kept in the track. Older points are discarded
    
    **model**        : Model name
    """
    def __init__(self, yaml_path):  
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_path = self.config["paths"].get("model_path", "models/")
        self.output_path = self.config["paths"]["output_path"]
        
        self.video_source = self.config["general"]["input_method"]
        self.SAVE_OUTPUT = self.config["general"]["save_output"]
        self.track_length = self.config["general"]["track_length"]
        self.SHOW_OUTPUT = self.config["general"]["show_output"]

          
    def make_gstreamer_pipeline(self):
        """
        Construct the gstreamer pipeline.

        Returns:
            Out:_A gstreamer video pipeline_ 
        """
        pipe = self.config["gstreamer"]
        print("Building gstreamer pipeline..")
        return (str(
            pipe["source"]+" sensor-id="+str(pipe.get("id", 0))+" ! "
            +pipe["file"]+"(memory:"+pipe["memory"]+"), "
            "width="+str(pipe["input_width"])+", height="+str(pipe["input_height"])+", "
            "format="+pipe["input_format"]+", framerate="+str(pipe["framerate"])+"/1 ! "
            "nvvidconv flip-method="+str(pipe["flip_method"])+" ! "
            +pipe["file"]+", width="+str(pipe["output_width"])+", height="+str(pipe["output_height"])+", format="+pipe["output_format"]+" ! "
            "videoconvert ! "
            +pipe["file"]+", format="+pipe["sink_format"]+" ! "+pipe["sink"])
        )

    def make_video_source(self):
        """ Builds the cv2  video source

        Returns:
            capture: _A cv2 video source_
        """
        print("Getting video source...")
        if self.video_source == "webcam":
            capture = cv2.VideoCapture(0)
        elif self.video_source == "video":
            capture = cv2.VideoCapture(self.config["paths"]["video_path"] + self.config["video"])
        elif self.video_source == "gstreamer":
            #print(self.make_gstreamer_pipeline())
            capture = cv2.VideoCapture(self.make_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            print("Please specify video source!")
            exit()
        return capture

    def make_detector(self):
        """Builds the detector and sets params.

        Returns:
            Detector: _Detector class instance_
        """
        if self.config["detector"]['type'] == "GeometryMethod":
            detector = GeometryMethod(self.config["detector"]['threshold'])
            self.target_name = self.config["detector"].get("target_name", None)
            self.confidence_threshold = self.config["detector"].get("confidence_threshold", 0)
            self.refractory_frames = self.config["detector"]["refractory_frames"]
            self.override_frames = self.config["detector"].get("override_frames", 100)
        else:
            print("Please specify detector!")
            exit()
        return detector