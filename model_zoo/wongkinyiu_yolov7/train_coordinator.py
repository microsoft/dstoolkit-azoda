import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--cfg", type=str, help="Model yaml config")
parser.add_argument("--batch-size", type=int, help="Batch size")
parser.add_argument("--epochs", type=int, help="Number of epochs")
args = parser.parse_args()

print("Editing config")
# Updates the dataset location
with open(f"{args.dataset}.yaml") as f:
    lines = f.readlines()
print(lines)
lines[0] = f"\n"
print("After edit:")
print(lines)
with open(f"{args.dataset}.yaml", "w") as f:
    f.writelines(lines)

# project must be set to outputs, since the AML saves results in the outputs directory
print("Start training")
os.system(
    f"python yolov5/train.py \
          --batch-size {args.batch_size} \
          --cfg {args.cfg} \
          --data {args.dataset}.yaml \
          --epochs {args.epochs} \
          --project outputs/"
)
