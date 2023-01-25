from io import BytesIO
from PIL import Image
import json
import base64
import torch
import os


def init():
    global model
    model = torch.hub.load(
        "wongkinyiu_yolov7/yolov7",
        "custom",
        path_or_model="wongkinyiu_yolov7/best.pt",
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
