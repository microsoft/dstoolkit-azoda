from datetime import datetime
from PIL import Image
import argparse
import json
import os
import pandas as pd
import PIL
import requests

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--endpoint', help='AML deployment endpoint for inference')
parser.add_argument('-k', '--key', help='API key for the AML deployment')
parser.add_argument('-d', '--dataset', help='Name of dataset/project')
args = parser.parse_args()
dataset = args.dataset

image_directory = f'../../{dataset}/test_images/'
inferences_directory = f'../../{dataset}/test_inferences/'
os.makedirs(inferences_directory, exist_ok=True)
enable_auth = False
row_data = []
counter = 0
for image_filename in os.listdir(image_directory):
    if image_filename.endswith('.jpg'):
        print(image_filename)
        image_PIL = PIL.Image.open(image_directory+image_filename)
        width, height = image_PIL.size
        img = open(image_directory+image_filename, 'rb').read()
        headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + args.key)}
        resp = requests.post(f'{args.endpoint}?prob=0.5', img, headers=headers)
        results = resp.text
        detections = json.loads(results)
        for detection in detections:
            box = detection['bounding_box']
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
            row_data.append([image_filename,
                             int(xmin*width),
                             int(xmax*width),
                             int(ymin*height),
                             int(ymax*height),
                             detection['label'],
                             detection['confidence']])

df = pd.DataFrame(row_data)
column_names = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class', 'Confidence']
df.columns = column_names
time_stamp = datetime.now().strftime('%y%m%d%H%m%S')
df.to_csv(f'{inferences_directory}/labels_{time_stamp}.csv', index=False)
print("Complete")
