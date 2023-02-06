from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import json
import base64


def init():
    global model
    model = YOLO("ultralytics_yolov8/best.pt")


def run(request):
    json_load = json.loads(request)
    decoded_img = base64.b64decode(json_load["img"])
    stream = BytesIO(decoded_img)
    img = Image.open(stream).convert("RGB")
    results = model([img])
    return f"{results[0].boxes}"
