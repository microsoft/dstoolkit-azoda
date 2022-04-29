# %%
from PIL import Image
import ast
import base64
import cv2
import json
import requests

sample_image_path = 'synthetic_dataset/images/img_2.jpg'
# Get this information from your AML endpoint under the Consume tab
endpoint = 'your-rest-endpoint'
api_key = 'your-key'
headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
with open(sample_image_path, mode='rb') as file:
    img = file.read()
data = dict()
data['img'] = base64.encodebytes(img).decode('utf-8')
json_string = json.dumps(data)
response = requests.post(endpoint, data=json_string, headers=headers)
print(response.json())
bbs = ast.literal_eval(response.json().replace('tensor', ''))
image = cv2.imread(sample_image_path)

# Confidence threshold in percent
conf_thres = 50
for bb in bbs:
    x1 = int(bb[0])
    y1 = int(bb[1])
    x2 = int(bb[2])
    y2 = int(bb[3])
    class_id = int(bb[5])
    conf = int(bb[4]*100)
    if conf < conf_thres:
        continue
    label = f'class: {class_id}, conf: {conf}%'
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

Image.fromarray(image)
# %%
