from io import BytesIO
from PIL import Image
import json
import base64
import torch
import os


def init():
    global model
    model = torch.hub.load(
        "ultralytics_yolov5/yolov5",
        "custom",
        path="ultralytics_yolov5/best.pt",
        source="local",
        force_reload=True,
    )


def run(request):
    json_load = json.loads(request)
    decoded_img = base64.b64decode(json_load["img"])
    stream = BytesIO(decoded_img)
    img = Image.open(stream).convert("RGBA")
    results = model([img])
    return f"{results.xyxy[0]}"
