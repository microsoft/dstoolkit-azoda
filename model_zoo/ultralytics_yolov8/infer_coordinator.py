import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, help="Images for inference")
parser.add_argument("--weights", type=str, help="Path to model weights")
parser.add_argument("--conf", type=str, help="Confidence threshold")
args = parser.parse_args()

# project must be set to outputs, since the AML saves results in the outputs directory
print("Start inferencing")
os.chdir("yolov8/")
os.system(
    f"yolo detect predict \
          conf={args.conf} \
          project=../outputs/ \
          save_json=True \
          source={args.images} \
          model={args.weights}"
)
print("Inferencing complete")
