'''
Model scoring script

Loads a specified model from AML model registry
and score the input image from service call

'''

import os
import time
from io import BytesIO

import numpy as np
import tensorflow as tf
from azureml.core.model import Model
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from PIL import Image
from utils import label_map_util

MODEL_NAME = '__REPLACE_MODEL_NAME__'
LABEL_MAP_NAME = 'tf_label_map.pbtxt'
MODEL_FILE_NAME = 'saved_model'


def init():
    global model
    global blob_service
    model = load_model()


def load_model():
    model_dir = Model.get_model_path(MODEL_NAME)

    label_map_file = os.path.join(model_dir, LABEL_MAP_NAME)
    label_map_dict = label_map_util.get_label_map_dict(label_map_file)
    num_classes = len(label_map_dict)
    label_map = label_map_util.load_labelmap(label_map_file)

    categories = label_map_util.convert_label_map_to_categories(
                                        label_map,
                                        max_num_classes=num_classes,
                                        use_display_name=True)

    category_index = label_map_util.create_category_index(categories)

    model_path = os.path.join(model_dir, MODEL_FILE_NAME)

    # New model load approach
    detection_fn = tf.saved_model.load(model_path)

    return {'detection_fn': detection_fn,
            'category_index': category_index}


@rawhttp
def run(request):
    try:
        if request.method == 'POST':
            reqBody = request.get_data(False)
            THRESHOLD = float(request.args.get('prob'))
            print(THRESHOLD)
            response = inference(reqBody, THRESHOLD)
            return response
        if request.method == 'GET':
            respBody = str.encode("GET is not supported")
            return AMLResponse(respBody, 405)
    except Exception as e:
        response = str(e)
    return response


def inference(raw_data, THRESHOLD):

    start_time = time.time()

    image = Image.open(BytesIO(raw_data))
    image_np = np.array(image.convert('RGB'))

    latency = time.time() - start_time
    print("Time to convert the image: {}".format(latency))

    start_time = time.time()

    # The input needs to be a tensor, convert it using
    # `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with
    # `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform the detections by running the model
    output_dict = model['detection_fn'](input_tensor)

    latency = time.time() - start_time
    print("Time to score: {}".format(latency))

    num_detections = int(output_dict.pop('num_detections'))

    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}

    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = (output_dict['detection_classes']
                                        .astype(np.int64))

    result = []

    for idx, score in enumerate(output_dict['detection_scores']):
        if score > THRESHOLD:
            result.append(
                {'class': int(output_dict['detection_classes'][idx]),
                 'label': (model['category_index']
                                [output_dict['detection_classes']
                                            [idx]]
                                ['name']),
                 'confidence': float(output_dict['detection_scores'][idx]),
                 'bounding_box': (output_dict['detection_boxes'][idx]).tolist()
                 })
        else:
            print('idx {} detection score too low {}'.format(idx, score))

    return result
