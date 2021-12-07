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
INCLUDE_MASK = 'mask' in MODEL_NAME    # set global variable indicate whether to include masks in output dict
LABEL_MAP_NAME = 'tf_label_map.pbtxt'
MODEL_FILE_NAME = 'frozen_inference_graph.pb'


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

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}

        # extend detection key list if required including mask
        detection_keys = ['num_detections', 'detection_classes', 'detection_boxes', 'detection_masks', 'detection_scores'] if INCLUDE_MASK \
            else ['num_detections', 'detection_classes', 'detection_boxes', 'detection_scores']

        for key in detection_keys:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = (tf.get_default_graph()
                                      .get_tensor_by_name(tensor_name))

        image_tensor = (tf.get_default_graph()
                          .get_tensor_by_name('image_tensor:0'))

        sess = tf.Session(graph=detection_graph)

    return {'session': sess,
            'image_tensor': image_tensor,
            'tensor_dict': tensor_dict,
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

    image_np_expanded = np.expand_dims(image_np, axis=0)

    start_time = time.time()
    output_dict = model['session'].run(model['tensor_dict'],
                                       feed_dict={model['image_tensor']:
                                                  image_np_expanded})

    latency = time.time() - start_time
    print("Time to score: {}".format(latency))

    output_dict['num_detections'] = int(output_dict['num_detections'][0])

    output_dict['detection_classes'] = (output_dict['detection_classes'][0]
                                        .astype(np.uint8).tolist())

    output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()

    output_dict['detection_scores'] = (output_dict['detection_scores'][0]
                                       .tolist())

    if INCLUDE_MASK:
        output_dict['detection_masks'] = output_dict['detection_masks'][0].tolist()

    result = []

    for idx, score in enumerate(output_dict['detection_scores']):
        idx_dict = {
            'class': output_dict['detection_classes'][idx],
            'label': (model['category_index'][output_dict['detection_classes'][idx]]['name']),
            'confidence': output_dict['detection_scores'][idx],
            'bounding_box': output_dict['detection_boxes'][idx]
        }
        if INCLUDE_MASK:
            idx_dict['mask'] = output_dict['detection_masks'][idx]

        if score > THRESHOLD:
            result.append(idx_dict)
        else:
            print('idx {} detection score too low {}'.format(idx, score))

    return result
