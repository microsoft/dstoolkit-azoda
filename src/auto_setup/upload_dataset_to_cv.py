# %%
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
    Region,
)
from datetime import datetime
from msrest.authentication import ApiKeyCredentials
import argparse
import os
import pandas as pd
import util

# %%
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--subscription_id", type=str, help="Subscription ID")
parser.add_argument("--resource_group", type=str, help="Resource Group")
parser.add_argument(
    "--key", type=str, help="Key for the specified custom vision deployment"
)
parser.add_argument("-d", "--dataset", help="Name of project/dataset")
parser.add_argument("--cv_name", type=str, help="Name of Custom Vision resource")
args = parser.parse_args()

# %%
id_to_filename_dict = dict()
tagged_images_with_regions = []
ENDPOINT = "https://westeurope.api.cognitive.microsoft.com/"
credentials = ApiKeyCredentials(in_headers={"Training-key": args.key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
publish_iteration_name = "detectModel"
obj_detection_domain = next(
    domain
    for domain in trainer.get_domains()
    if domain.type == "ObjectDetection" and domain.name == "General"
)

print("Creating project...")
project = trainer.create_project(args.dataset, domain_id=obj_detection_domain.id)
print(project.name)
# %%
base_data_location = args.dataset

if os.path.exists(f"../../{args.dataset}"):
    base_directory = f"../../{args.dataset}"
else:
    base_directory = f"../../model_zoo/ultralytics_yolov5/{args.dataset}"
base_yolo_directory = os.path.join(base_directory, "yolo")

# Load images from folder
image_groups_directory = os.path.join(base_yolo_directory, "images")
label_groups_directory = os.path.join(base_yolo_directory, "labels")
all_images_directory = os.path.join(base_directory, "images")

# If no annotations exist, just add the images to Custom Vision
if not os.path.exists(image_groups_directory):
    regions = []
    for image_filename in os.listdir(all_images_directory):
        full_filename = os.path.join(all_images_directory, image_filename)
        with open(full_filename, mode="rb") as image_contents:
            tagged_images_with_regions.append(
                ImageFileCreateEntry(
                    name=image_filename,
                    contents=image_contents.read(),
                    regions=[],
                )
            )
        upload_result = trainer.create_images_from_files(
            project.id,
            ImageFileCreateBatch(images=tagged_images_with_regions),
        )
        if upload_result.images[-1].image:
            id_to_filename_dict[upload_result.images[-1].image.id] = image_filename
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
            exit(-1)
        tagged_images_with_regions = []
else:
    image_directories = [filename for filename in os.listdir(image_groups_directory)]
    # Set get class names
    csv_datasets_dir = os.path.join(base_directory, "datasets")
    train_path = util.get_lastest_iteration(csv_datasets_dir, req_prefix="train")
    test_path = util.get_lastest_iteration(csv_datasets_dir, req_prefix="test")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    labels = sorted(
        list(set(list(df_train["class"].values) + list(df_test["class"].values)))
    )
    print("labels:", labels)
    label_tags = []
    for label in labels:
        label_tags.append(trainer.create_tag(project.id, label))

    for image_directory in image_directories:
        for image_filename in os.listdir(
            os.path.join(image_groups_directory, image_directory)
        ):
            print("filename", image_filename)
            label_path = os.path.join(
                label_groups_directory, image_directory, image_filename[:-4] + ".txt"
            )
            regions = []
            if os.path.exists(label_path):
                with open(label_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        line_vals = line.replace("\n", "").split(" ")
                        print(line_vals)
                        class_val = int(line_vals[0])
                        x_center = float(line_vals[1])
                        y_center = float(line_vals[2])
                        width = float(line_vals[3])
                        height = float(line_vals[4])
                        left = x_center - 0.5 * width
                        top = y_center - 0.5 * height
                        regions.append(
                            Region(
                                tag_id=label_tags[class_val].id,
                                left=left,
                                top=top,
                                width=width,
                                height=height,
                            )
                        )
            else:
                print(label_path + " does not exist")
            print("-")
            full_filename = os.path.join(
                image_groups_directory, image_directory, image_filename
            )
            with open(full_filename, mode="rb") as image_contents:
                tagged_images_with_regions.append(
                    ImageFileCreateEntry(
                        name=image_filename.replace("_", "-"),
                        contents=image_contents.read(),
                        regions=regions,
                    )
                )
            upload_result = trainer.create_images_from_files(
                project.id,
                ImageFileCreateBatch(images=tagged_images_with_regions),
            )
            if upload_result.images[-1].image:
                id_to_filename_dict[upload_result.images[-1].image.id] = image_filename
            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status)
                exit(-1)
            tagged_images_with_regions = []

# %%
for id in id_to_filename_dict:
    print(id, ":", id_to_filename_dict[id])

# %%
time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
df = pd.DataFrame.from_dict(id_to_filename_dict, orient="index")
df.to_csv(f"ids_to_filenames_{time_stamp}.csv")
