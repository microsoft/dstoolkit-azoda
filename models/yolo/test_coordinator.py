import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--weights", type=str, help="Path to model weights")
parser.add_argument("--conf", type=str, help="Confidence threshold")
parser.add_argument("--iou", type=str, help="Bounding box IOU threshold")
args = parser.parse_args()

# project must be set to outputs, since the AML saves results in the outputs directory
print('Start testing')
os.system(f"python yolov5/val.py \
          --conf-thres {args.conf} \
          --data {args.dataset}.yaml \
          --iou-thres {args.iou} \
          --project outputs/ \
          --save-conf \
          --save-txt \
          --task val \
          --verbose \
          --weights {args.weights}")
print('Testing complete')
