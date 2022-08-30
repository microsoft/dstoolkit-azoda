# pip install azure-cognitiveservices-vision-customvision
# tqdm pascal_voc_writer opencv-python-headless
# Tick responsible AI Notice - https://ms.portal.azure.com/#create/Microsoft.CognitiveServicesComputerVision

from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from msrest.authentication import ApiKeyCredentials
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import os
import requests

# Manually making a project at https://www.customvision.ai/projects

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--key", type=str, help="Key for the specified custom vision deployment"
)
parser.add_argument("--cv_name", type=str, help="Name of Custom Vision resource")
parser.add_argument("--output_dir", type=str, help="Download location")
args = parser.parse_args()

# Replace with valid values
cv_project_name = "azoda_example_dataset"
key = args.key
output_dir = args.output_dir
cv_name = args.cv_name
ENDPOINT = f"https://{cv_name}.cognitiveservices.azure.com/"

# Create authenticate client based on the given values
credentials = ApiKeyCredentials(in_headers={"Training-key": key})
client = CustomVisionTrainingClient(ENDPOINT, credentials)
projects = client.get_projects()
project = [project for project in projects if project.name == cv_project_name][0]
project_id = project.id

# Test that it works. Print number of images in the project.
print("Number of images in project:", client.get_tagged_image_count(project_id))

# Project tags
tags = client.get_tags(project_id)
defect_classes = []

for tag in tags:
    defect_classes.append(tag.name)
    print("Images with", tag.name, tag.image_count)

# Sort defect classes and print list
defect_classes.sort()
print("\nDefect Classes:\n", sorted(defect_classes))

# Define folder where to save imported data
images_dir = os.path.join(output_dir, "images/")
annotations_dir = os.path.join(output_dir, "annotations/")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Check that save folder exists
if not os.path.isdir(images_dir):
    print(
        "Output folder:",
        images_dir,
        "not found",
    )
else:
    print("Save images folder found!")

if not os.path.isdir(annotations_dir):
    print(
        "Output folder:",
        annotations_dir,
        "not found",
    )
else:
    print("Save annotations folder found!")


def get_image_name(img_data):
    # If image is downloaded to Custom Vision manually, then it does not have name. In this case use ID as name.
    try:
        print(img_data.metadata.get("name"))
    except:
        print("No name in metadata")
    try:
        print(img_data.id)
    except:
        print("No id in img data")
    if img_data.metadata is not None:
        img_name = img_data.metadata.get("name")
    else:
        img_name = img_data.id + ".jpg"
    return img_name


def download_and_save(image_metadata):
    # If image height is over 900 pixels, custom vision resizes images. Fix this with scale_factor
    # scale = 1

    # Get image data.
    img_name = get_image_name(image_metadata)

    # Download and save image file
    # image = requests.get(img_data.resized_image_uri).content
    image = requests.get(img_data.original_image_uri).content
    print("img_data.original_image_uri", img_data.original_image_uri)
    print(img_data.as_dict())

    # Define image name and save image
    img_fname = os.path.join(images_dir, img_name)
    with open(img_fname, "wb") as f:
        f.write(image)
        f.close()

    # Get true image shape
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img_height = decoded.shape[0]
    img_width = decoded.shape[1]

    # There can be multiple defects in single image. Loop through defects.
    detections = image_metadata.regions
    lines = []
    for idx, bb in enumerate(detections):
        tag_name = bb.tag_name
        x_center = bb.left + 0.5 * bb.width
        y_center = bb.left + 0.5 * bb.height
        lines.append(f"{tag_name} {x_center} {y_center} {bb.width} {bb.height}\n")

    with open(f"{output_dir}/annotations/{img_name[:-4]}.txt", "w") as f:
        f.writelines(lines)


# Get number of images in project
count = client.get_tagged_image_count(project_id)
print("Found:", count, "tagged images from Custom Vision project")

downloaded = 0
while count > 0:
    # Get number of images to download. Custom Vision has max number of 256.
    count_to_export = min(count, 256)

    # Print images currently downloading
    print("\nImages left to download:", count)
    print("Now downloading", count_to_export, "images and their metadata")

    # Get Custom Vision project image metadata
    images_data = client.get_tagged_images(
        project_id, take=count_to_export, skip=downloaded
    )

    # Loop through images data and save image + metadata
    for img_data in tqdm(images_data):
        # Download and save image and metadata
        download_and_save(img_data)
    # Increment variables
    downloaded += count_to_export
    count -= count_to_export

# Print results
print("\nDownloaded and saved", downloaded, "images and their metadata\n")
