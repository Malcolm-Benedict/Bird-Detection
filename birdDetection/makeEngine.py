from ultralytics import YOLO
import argparse

MODEL_PATH = 'models/'

parser = argparse.ArgumentParser(
                    prog='makeEngine',
                    description='converts a model to tensorrt engine',
                    epilog='todo')
parser.add_argument('model')
args = parser.parse_args()
model_name = args.model
model = YOLO(MODEL_PATH+model_name, task="track",)
model.export(format="engine")