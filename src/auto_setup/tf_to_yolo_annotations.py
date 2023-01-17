# Script to convert current annotation style to the yolo annotation convention

# Imports
import argparse
from PIL import Image
import os
import pandas as pd
import shutil
import util

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Name of dataset/project")
parser.add_argument("-m", "--model", help="Name of model", default="yolo")
args = parser.parse_args()
dataset_name = args.dataset
model_name = args.model
base_dir = f"{dataset_name}"
datasets_dir = os.path.join(base_dir, "datasets/")
os.makedirs(datasets_dir, exist_ok=True)
train_path = util.get_lastest_iteration(datasets_dir, req_prefix="train")
test_path = util.get_lastest_iteration(datasets_dir, req_prefix="test")
dataset_paths = []
if train_path:
    dataset_paths.append(train_path)
if test_path:
    dataset_paths.append(test_path)
class_id_dict = dict()

# Set paths
for annotation_path in dataset_paths:
    images_path = os.path.join(base_dir, "images")
    yolo_annotations_path = os.path.join(
        base_dir, "yolo", "labels", f"{os.path.basename(annotation_path)[:-4]}/"
    )
    os.makedirs(os.path.join(base_dir, "yolo", "labels"), exist_ok=True)
    df = pd.read_csv(annotation_path)

    # Set class name to id mapping
    for class_int, class_name in enumerate(sorted(list(set(df["class"])))):
        class_id_dict[class_name] = class_int

    # Build up annotation dictionary
    file_annotations = dict()
    for i in range(df.shape[0]):
        image_name = df["filename"][i]
        class_name = df["class"][i]
        image_path = os.path.join(images_path, image_name)
        with Image.open(image_path) as img:
            width, height = img.size
            x1 = float(df["xmin"][i])
            x2 = float(df["xmax"][i])
            y1 = float(df["ymin"][i])
            y2 = float(df["ymax"][i])
            x_center = 0.5 * (x1 + x2) / width
            y_center = 0.5 * (y1 + y2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            for val in [x_center, y_center, box_width, box_height]:
                if not 0 <= val <= 1:
                    print("ERROR, invalid annotation location")
                    print(image_name)
                    print(x1, x2, y1, y2, width, height)
                    print([x_center, y_center, box_width, box_height])
            if (
                x_center - box_width / 2 < 0
                or x_center + box_width / 2 > 1
                or y_center - box_height / 2 < 0
                or y_center + box_height / 2 > 1
            ):
                print("ERROR, invalid annotation location")
                print(image_name)
                print(x1, x2, y1, y2, width, height)
                print([x_center, y_center, box_width, box_height])
            if class_name not in class_id_dict:
                print("ERROR, unexpected class name:", class_name)
                class_id = -1
            else:
                class_id = class_id_dict[class_name]
            annotation_line = " ".join(
                [
                    str(class_id),
                    str(x_center),
                    str(y_center),
                    str(box_width),
                    str(box_height),
                ]
            )
            if image_name not in file_annotations:
                file_annotations[image_name] = []
            file_annotations[image_name].append(annotation_line)

    # Create directory for yolo annotations
    if not os.path.exists(yolo_annotations_path):
        os.mkdir(yolo_annotations_path)

    # Write to file
    for image_name in file_annotations:
        annotation_file_path = yolo_annotations_path + image_name[:-4] + ".txt"
        f = open(annotation_file_path, "w+")
        for annotation_line in file_annotations[image_name]:
            f.write(annotation_line + "\n")
        f.close()

    print("Complete")

# Move images
src_loc = os.path.join(base_dir, "images")
for annotation_path in dataset_paths:
    images_path = os.path.join(base_dir, "yolo", "images")
    os.makedirs(images_path, exist_ok=True)
    annotation_base_name = os.path.basename(annotation_path)[:-4]
    dst_loc = os.path.join(base_dir, "yolo", f"images/{annotation_base_name}")
    os.makedirs(dst_loc, exist_ok=True)
    df = pd.read_csv(annotation_path)
    filenames = sorted(list(set(df["filename"])))
    for filename in filenames:
        shutil.copyfile(
            os.path.join(src_loc, filename), os.path.join(dst_loc, filename)
        )

# Make yaml
class_list = str(sorted(list(class_id_dict)))
if model_name.startswith("yolo"):
    lines = (
        f"train: ../../{dataset_name}/yolo/images/{os.path.basename(train_path)[:-4]}\n"
        f"val: ../../{dataset_name}/yolo/images/{os.path.basename(test_path)[:-4]}\n"
        f"nc: {len(list(class_id_dict))}\n"
        f"names: {class_list}\n"
    )
else:
    lines = (
        f"path: ../../{dataset_name}/yolo\n"
        f"train: images/{os.path.basename(train_path)[:-4]}\n"
        f"val: images/{os.path.basename(test_path)[:-4]}\n"
        f"nc: {len(list(class_id_dict))}\n"
        f"names: {class_list}\n"
    )

with open(f"{dataset_name}.yaml", "w") as f:
    f.writelines(lines)
