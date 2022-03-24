import json
import os
import pandas as pd
from PIL import Image
import PIL
import requests

endpoint = 'http://278dd78f-fb02-4321-8dfe-8bae23dcb3f9.westeurope.azurecontainer.io/score?prob=0.5'
api_key = 'znixQQu5lqsKCWhulYNJzNUsf41ykbPV'
image_directory = 'synthetic_dataset/images/'
enable_auth = False
row_data = []
counter = 0
for image_filename in os.listdir(image_directory):
    if image_filename.endswith('.jpg'):
        if counter >= 5:
            break
        counter += 1
        print(image_filename)
        image_PIL = PIL.Image.open(image_directory+image_filename)
        width, height = image_PIL.size
        img = open(image_directory+image_filename, 'rb').read()
        headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
        resp = requests.post(endpoint, img, headers=headers)
        results = resp.text
        detections = json.loads(results)
        for detect in detections:
            box = detect['bounding_box']
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
            row_data.append([image_filename,
                             int(xmin*width),
                             int(xmax*width),
                             int(ymin*height),
                             int(ymax*height),
                             detect['label'],
                             detect['confidence']])

df = pd.DataFrame(row_data)
column_names = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class', 'Confidence']
df.columns = column_names
df.to_csv('labels.csv', index=False)
print("Complete")
