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
from msrest.authentication import ApiKeyCredentials
import argparse
import os
import uuid


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

ENDPOINT = f"https://{args.cv_name}.cognitiveservices.azure.com/"
prediction_key = args.key
prediction_resource_id = f"/subscriptions/{args.subscription_id}/resourceGroups/{args.resource_group}/providers/Microsoft.CognitiveServices/accounts/{args.cv_name}"

credentials = ApiKeyCredentials(in_headers={"Training-key": args.key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key}
)
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

publish_iteration_name = "detectModel"
obj_detection_domain = next(
    domain
    for domain in trainer.get_domains()
    if domain.type == "ObjectDetection" and domain.name == "General"
)
print("Creating project...")
project = trainer.create_project(str(uuid.uuid4()), domain_id=obj_detection_domain.id)

print(project.name)

labels = ["circle"]
label_tags = []
for label in labels:
    label_tags.append(trainer.create_tag(project.id, label))

base_data_location = args.dataset

if os.path.exists(f"../../{args.dataset}"):
    base_data_directory = f"../../{args.dataset}/yolo"
else:
    base_data_directory = "../../model_zoo/ultralytics_yolov5/synthetic_dataset/yolo"

# Load images from folder
image_groups_directory = os.path.join(base_data_directory, "images")
label_groups_directory = os.path.join(base_data_directory, "labels")
image_directories = [filename for filename in os.listdir(image_groups_directory)]
print(image_directories)
tagged_images_with_regions = []
batch_size = 32
for image_directory in image_directories:
    for image_filename in os.listdir(
        os.path.join(image_groups_directory, image_directory)
    ):
        print("filename", image_filename)
        with open(
            os.path.join(
                label_groups_directory, image_directory, image_filename[:-4] + ".txt"
            )
        ) as f:
            lines = f.readlines()
            regions = []
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
                        # filename=image_filename,
                    )
                )
        print("-")
        full_filename = os.path.join(
            image_groups_directory, image_directory, image_filename
        )
        with open(full_filename, mode="rb") as image_contents:
            tagged_images_with_regions.append(
                ImageFileCreateEntry(
                    name=image_filename.replace("_", "-"),
                    # name="testname",
                    contents=image_contents.read(),
                    regions=regions,
                    # tagIds=["testmeta"]
                    # tag_ids=["tagtest"], breaking
                )
            )
        upload_result = trainer.create_images_from_files(
            project.id,
            ImageFileCreateBatch(images=tagged_images_with_regions),
            # tag_ids=[image_filename.replace("_", "-")],
        )
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
            exit(-1)
        tagged_images_with_regions = []
