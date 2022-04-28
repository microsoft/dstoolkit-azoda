from io import BytesIO
from PIL import Image
import json
import base64
import torch
import sys
import os


def init():
    global model
    model = torch.hub.load('yolo/yolov5', 'custom', path='yolo/last.pt', source='local', force_reload=True, device='cpu')
    print(sys.version)
    print('ls .')
    os.listdir('.')


def run(request):
    json_load = json.loads(request)
    decoded_img = base64.b64decode(json_load['img'])
    stream = BytesIO(decoded_img)
    img = Image.open(stream).convert("RGBA")
    results = model([img])
    return f"test is {results.xyxy[0]}"
