import argparse
import datetime
import os
import pandas as pd
import pixie as px
from typing import List
import yaml

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--api', help='API url of the pixie deployment')
parser.add_argument('-k', '--key', help='Key for the specified pixie deployment')
parser.add_argument('-d', '--dataset', help='Name of project/dataset')
args = parser.parse_args()

# Set up Pixie connection
# Authentication information
config_dict = {'pixie': {'deploy': {'api': args.api, 'key': args.key}}}
with open('config.yaml', 'w') as file:
    documents = yaml.dump(config_dict, file)
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)["pixie"]["deploy"]

# Create Pixie client
px_client = px.PixieClient(config["api"], api_key=config.get("key"))

# Create a pixie project
project_name = args.dataset
project = px_client.create_project(px.Project(project_name, 'Dataset import test'), exist_ok=True)

# Configure directories
image_directory = f'../../{project_name}/images/'
annotations_directory = f'../../{project_name}/datasets/'

# Build up annotation dictionary
file_annotations = dict()
class_names = []
os.makedirs(annotations_directory, exist_ok=True)
for filename in os.listdir(annotations_directory):
    if not filename.endswith('.csv'):
        continue
    print(filename)
    df = pd.read_csv(annotations_directory+filename)
    for index, row in df.iterrows():
        class_name = row['class']
        image_filename = row['filename']
        x_min = row['xmin']
        y_min = row['ymin']
        bb_width = row['xmax']-x_min
        bb_height = row['ymax']-y_min
        if image_filename not in file_annotations:
            file_annotations[image_filename] = []
        if class_name not in class_names:
            class_names.append(class_name)
        file_annotations[image_filename].append({'x_min': x_min,
                                                 'y_min': y_min,
                                                 'bb_width': bb_width,
                                                 'bb_height': bb_height,
                                                 'class_name': class_name})
# Create a Pixie dataset
din = px.DatasetIn(project_id=project.id,
                   name=project_name,
                   data=px.ObjectDetectionDataset(classes=class_names),
                   description="")
dataset = px_client.create_dataset(din)

# Upload the local images to Pixie
time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
for image_filename in os.listdir(image_directory):
    if image_filename.endswith('.jpg'):
        image_path = os.path.join(image_directory, image_filename)
        f: px.File = px_client.upload_image(image=image_path,
                                            project_id=project.id,
                                            verify_file=True,
                                            tags=[time_stamp],
                                            properties={"filename": image_filename})
# Fetch files which were just uploaded
files: List[px.File] = px_client.get_files(project_id=project.id, where=f"f.type='image' and \
                                                                          f.info.projectId='{project.id}' and \
                                                                          f.tags[0]='{time_stamp}'")
# Label the files
label_ins: List[px.LabelIn] = []
for file in files:
    image_filename = file.properties.additional_properties['filename']
    dims = file.additional_properties['image']
    img_width, img_height = dims['width'], dims['height']
    if image_filename in file_annotations:
        boxes: List[px.ObjectDetectionAnnotatedBox] = [px.ObjectDetectionAnnotatedBox(
                class_=box['class_name'],
                left=box['x_min']/img_width,
                top=box['y_min']/img_height,
                width=box['bb_width']/img_width,
                height=box['bb_height']/img_height) for box in file_annotations[image_filename]
            ]
        label_in = px.LabelIn(dataset.id, px.ObjectDetectionLabel(file_id=file.id, boxes=boxes))
        label_ins.append(label_in)
labels: List[px.Label] = px_client.create_label(label_ins)
