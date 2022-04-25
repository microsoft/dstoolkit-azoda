import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--cfg", type=str, help="Model yaml config")
parser.add_argument("--batch-size", type=int, help="Batch size")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--input_ref", type=str, help="Dataset mount reference")
args = parser.parse_args()

# Creates the config file on the 
lines = f'path: {args.input_ref}/yolo\n' \
        f'train: images/train_{args.dataset}_220406080415\n' \
        f'val: images/test_{args.dataset}_220406080415\n' \
         'nc: 1\n' \
         'names: [\'circle\']\n'

with open(f'{args.dataset}.yaml', 'w') as f:
    f.writelines(lines)

# project must be set to outputs, since the AML saves results in the outputs directory
os.system(f"python yolov5/train.py --data {args.dataset}.yaml \
          --cfg {args.cfg} \
          --batch-size {args.batch_size} \
          --epochs {args.epochs} \
          --project outputs/")
