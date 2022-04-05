import argparse
import pandas as pd
import pixie as px
from typing import List
import yaml

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--api', help='API url of the pixie deployment')
parser.add_argument('-k', '--key', help='Key for the specified pixie deployment')
parser.add_argument('-d', '--dataset', help='Name of dataset/project')
parser.add_argument('-i', '--project_id', help='ID for the project')
parser.add_argument('-m', '--model_id', help='ID for the model', default='')
parser.add_argument('--export_labels', action='store_true', help='Activates label saving')
parser.add_argument('--export_inferences', action='store_true', help='Activates inference saving')
args = parser.parse_args()
project_id = args.project_id
dataset_name = args.dataset
model_id = args.model_id

# Set up Pixie connection
config_dict = {'pixie': {'deploy': {'api': args.api, 'key': args.key}}}
with open('config.yaml', 'w') as file:
    documents = yaml.dump(config_dict, file)
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)["pixie"]["deploy"]
px_client = px.PixieClient(config["api"], api_key=config.get("key"))

# Connect to input project
project = px_client.get_project(project_id)
print('Available datasets:')
for dataset_i in px_client.get_datasets(project.id):
    print(dataset_i.name)

# Connect to target dataset
dataset = px_client.get_dataset(f'{project_id}_{dataset_name}')
files: List[px.File] = px_client.get_files(project_id=project.id, where=f"f.type='image' and \
                                                                          f.info.projectId='{project.id}'")

# Save labels
if args.export_labels:
    row_data = []
    for file in files:
        img_width, img_height = file.image.width, file.image.height
        filename = file.properties['filename']
        print(filename)
        if dataset_name not in file.additional_properties['labels']:
            continue
        label_id = file.additional_properties['labels'][dataset_name]['labelId']
        label = px_client.get_labels(label_id=label_id)
        if not hasattr(label, 'data'):
            continue
        for box in label.data.boxes:
            print(box)
            bb_w = box.width*img_width
            bb_h = box.height*img_height
            x_min = box.left*img_width
            y_min = box.top*img_height
            x_max = x_min + bb_w
            y_max = y_min + bb_h
            class_name = box.class_
            print(file.labels)
            row_data.append([filename, int(x_min), int(x_max), int(y_min), int(y_max), class_name])

    if len(row_data) > 0:
        export_path = 'labels_pixie.csv'
        df = pd.DataFrame(row_data)
        column_names = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class']
        df.columns = column_names
        df.to_csv(export_path, index=False)
    else:
        print('No labels found')

# Save inferences
if args.export_inferences:
    row_data = []
    for file in files:
        img_width, img_height = file.image.width, file.image.height
        filename = file.properties['filename']
        print(filename)
        if model_id not in file.inferences.additional_properties:
            continue
        inference_id = file.inferences.additional_properties[model_id].additional_properties['']
        inference = px_client.get_inference(inference_id)
        for box in inference.data.boxes:
            bb_w = box.width*img_width
            bb_h = box.height*img_height
            x_min = box.left*img_width
            y_min = box.top*img_height
            x_max = x_min + bb_w
            y_max = y_min + bb_h
            conf = box.score
            class_name = box.class_
            if conf >= 0.5:
                print(file.inferences)
                row_data.append([filename, int(x_min), int(x_max), int(y_min), int(y_max), class_name, conf])

    if len(row_data) > 0:
        export_path = 'inferences_pixie.csv'
        df = pd.DataFrame(row_data)
        column_names = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class', 'Confidence']
        df.columns = column_names
        df.to_csv(export_path, index=False)
    else:
        print('No model inferences found')
