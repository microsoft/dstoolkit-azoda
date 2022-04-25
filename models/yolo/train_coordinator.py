import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--cfg", type=str, help="Model yaml config")
parser.add_argument("--batch-size", type=int, help="Batch size")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--input_ref", type=str, help="Dataset mount reference")
args = parser.parse_args()

print('Editing config')
# Updates the dataset location
with open(f'{args.dataset}.yaml') as f:
    lines = f.readlines()
print(lines)
# lines[0] = f'path: {args.input_ref}/yolo\n'
lines[0] = f'path: {args.dataset}/yolo\n'
print('After edit:')
print(lines)
with open(f'{args.dataset}.yaml', 'w') as f:
    f.writelines(lines)

# project must be set to outputs, since the AML saves results in the outputs directory
print('Start training')
os.system(f"python yolov5/train.py --data {args.dataset}.yaml \
          --cfg {args.cfg} \
          --batch-size {args.batch_size} \
          --epochs {args.epochs} \
          --project outputs/")
