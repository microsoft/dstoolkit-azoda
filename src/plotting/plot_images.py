"""
Standalone script to score a model with a set of images and plot there results
"""


import argparse
import json
import os
from io import StringIO
from random import sample

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from azure.storage.blob import BlobServiceClient
from azureml.core import Experiment, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.run import get_run
from PIL import Image


config = tf.ConfigProto()
sess = tf.Session(config=config)


def read_config(config_path):
    """
    Reads a JSON configuration file

    :cfg_path: Absolute path to JSON file
    :returns: Configuration as JSON object
    """
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config


def get_configs(__here__):
    """
    Given a relative path, retrieves a JSON file

    :returns: Configurations as JSON object
    """
    plot_cfg_path = os.path.join(__here__, 'plot_config.json')
    env_cfg_path = os.path.join(__here__, '..', '..', 'configuration',
                                'dev_config.json')
    plot_cfg = read_config(plot_cfg_path)
    env_cfg = read_config(env_cfg_path)
    return env_cfg, plot_cfg


def get_workspace(env_cfg):
    """
    Gets the Azure Machine Learning Workspace

    :aml_tenant: AML Tenant ID str
    :aml_sub_id: AML Subscription ID str
    :aml_rg_group: AML Resource Group str
    :aml_ws: AML Workspace str
    :returns: AML Workspace object
    """
    interactive_auth = (InteractiveLoginAuthentication
                        (tenant_id=env_cfg['AML_TENANT_ID']))
    ws = Workspace(subscription_id=env_cfg['AML_SUBSCRIPTION_ID'],
                   resource_group=env_cfg['AML_RESOURCE_GROUP'],
                   workspace_name=env_cfg['AML_WORKSPACE_NAME'],
                   auth=interactive_auth)
    return ws


def rehydrate_run(ws, plt_cfg):
    """
    Rehydrate the experiment run id from AML to get information.

    :ws: AML Workspace object
    :plt_cfg: plotting config JSON
    :retruns: AML Run object
    """
    exp = Experiment(workspace=ws, name=plt_cfg['EXPERIMENT_NAME'])
    run = get_run(exp, plt_cfg['RUN_ID'], rehydrate=True)
    return run


def load_version_file(run, env_cfg, plt_cfg):
    """
    Load Version Files. Loads the .csv version file to pandas DF.

    :run: AML run object
    :env_cfg: Environment config JSON
    :plt_cfg: Plotting config JSON
    :returns: pandas DataFrame
    """
    file_name = run.tags['test set']
    blob_client = get_blobserviceclient(env_cfg)
    con_client = blob_client.get_container_client(plt_cfg['DATA_MOUNT'])
    blob_path = os.path.join(plt_cfg['EXPERIMENT_NAME'], 'datasets', file_name)
    blob_path = blob_path.replace('\\', '/')
    stream = con_client.download_blob(blob_path).content_as_text()
    df_vf = pd.read_csv(StringIO(stream))
    return df_vf


def get_blobserviceclient(env_cfg):
    """
    Get Blob Service Client.

    :env_cfg: Environment config dict
    :returns: Blob Service Client Object
    """
    # TODO - Add your way of conntecting to either blob or file share.
    # TODO - update with better auth method - e.g. token
    bs = BlobServiceClient(env_cfg['STORAGE_ACCOUNT_URL_SAS'])
    return bs


def download_run_artifacts(run, plt_cfg):
    base_path = os.path.join('outputs', 'final_model', 'model')
    model_path = os.path.join(base_path, 'frozen_inference_graph.pb')
    map_file_path = os.path.join(base_path, 'tf_label_map.pbtxt')
    output_dir = plt_cfg['OUTPUT_DIR']
    print('INFO: Downloading model to output directory.')
    run.download_file(name=model_path, output_file_path=output_dir)
    print('INFO: Downloading label mapping file to output directory.')
    run.download_file(name=map_file_path, output_file_path=output_dir)


def download_images(plt_cfg, env_cfg, df_vf):
    download_base_path = plt_cfg['OUTPUT_DIR']
    list_img = df_vf['filename'].unique().tolist()
    list_plot = sample(list_img, plt_cfg['NUM_IMAGES'])
    df_sample = df_vf[df_vf['filename'].isin(list_plot)]
    blob_client = get_blobserviceclient(env_cfg)
    con_client = blob_client.get_container_client(plt_cfg['DATA_MOUNT'])
    blob_base_path = os.path.join(plt_cfg['EXPERIMENT_NAME'], 'images')

    for img_i in list_plot:
        img_blob_path = os.path.join(blob_base_path, img_i)
        img_local_path = os.path.join(download_base_path, img_i)
        download_blob_locally(con_client, img_blob_path, img_local_path)

    return df_sample


def download_blob_locally(con_client, img_blob_path, img_local_path):
    download_stream = con_client.download_blob(img_blob_path)
    with open(img_local_path, "wb") as download_file:
        download_file.write(download_stream.readall())


def load_model(plt_cfg):
    detection_graph = tf.Graph()
    model_path = os.path.join(plt_cfg['OUTPUT_DIR'],
                              'frozen_inference_graph.pb')
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

        for key in ['num_detections',
                    'detection_boxes',
                    'detection_scores',
                    'detection_classes']:

            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = (tf.get_default_graph()
                                      .get_tensor_by_name(tensor_name))

        image_tensor = (tf.get_default_graph()
                          .get_tensor_by_name('image_tensor:0'))

        sess = tf.Session(graph=detection_graph)

    return {'session': sess,
            'image_tensor': image_tensor,
            'tensor_dict': tensor_dict}


def load_map_file(plt_cfg):
    map_file = os.path.join(plt_cfg['OUTPUT_DIR'], 'tf_label_map.pbtxt')
    map_dict = {}

    with open(map_file, 'r') as f:
        lines = [line.strip() for line in f]

        for line in lines:
            if 'id' in line:
                map_id = int(line.split(':')[-1])

            if 'name' in line:
                map_name = line.split(':')[-1]
                map_name = map_name.replace("'", "")
                map_dict[map_id] = map_name

    return map_dict


def score_images(model, map_dict, plt_cfg, df_plot, conf_thres):
    image_list = df_plot['filename'].unique().tolist()
    output_dir = plt_cfg['OUTPUT_DIR']

    for image_file in image_list:
        all_results = {}
        image_file = os.path.join(output_dir, image_file)
        image = Image.open(image_file)
        output_dict = infer_image(model, image)
        image.close()

        results = []

        for idx, score in enumerate(output_dict['detection_scores']):
            if score > conf_thres:
                results.append(
                    {'class': output_dict['detection_classes'][idx],
                     'confidence': output_dict['detection_scores'][idx],
                     'bounding_box': output_dict['detection_boxes'][idx]})
        all_results[image_file] = results

        plot_images(all_results, map_dict, output_dir, df_plot)


def infer_image(model, image):
    (im_width, im_height) = image.size

    image_np = (np.array(image.getdata())
                  .reshape((im_height, im_width, 3))
                  .astype(np.uint8))

    image_np_expanded = np.expand_dims(image_np, axis=0)

    output_dict = model['session'].run(model['tensor_dict'],
                                       feed_dict={model['image_tensor']:
                                                  image_np_expanded})

    output_dict['num_detections'] = int(output_dict['num_detections'][0])

    output_dict['detection_classes'] = (output_dict['detection_classes'][0]
                                        .astype(np.uint8).tolist())

    output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()

    output_dict['detection_scores'] = (output_dict['detection_scores'][0]
                                       .tolist())

    return output_dict


def plot_images(all_results, map_dict, out_dir, df_plot):
    images = list(all_results.keys())

    for image in images:
        image_name = image.split('/')[-1]
        print('INFO: Plotting predictions and GT on {}.'.format(image_name))
        img = load_mplt_image(image)
        detections = list(all_results.get(image))
        gt_frame = df_plot[df_plot['filename'].isin([image_name])]
        plot_bounding_boxes(img, detections, gt_frame, map_dict, out_dir,
                            image_name)


def load_mplt_image(image_path):
    img_np = mpimg.imread(image_path)
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    return img


def plot_bounding_boxes(img,
                        detections,
                        gt_frame,
                        map_dict,
                        out_dir,
                        image_name):
    img_width, img_height = img.size
    fig, ax = plt.subplots()
    ax.imshow(img)

    # NOTE - Adjust to your version file.
    if gt_frame['class'].notnull().any():
        for _, gt_row in gt_frame.iterrows():
            label_text = 'GT_' + gt_row['class']
            xmin, ymin = gt_row['xmin'], gt_row['ymin']
            xmax, ymax = gt_row['xmax'], gt_row['ymax']

            left_x, top_y, bottom_y = xmin, ymin, ymax
            gt_box_width, gt_box_height = xmax - xmin, ymax - ymin

            color = 'white'
            rect = patches.Rectangle((left_x, top_y),
                                     gt_box_width,
                                     gt_box_height,
                                     linewidth=0.5,
                                     edgecolor=color,
                                     facecolor='none')

            ax.add_patch(rect)
            plt.text((left_x+2), (bottom_y-2), label_text, color=color)

    for det_i in detections:
        label = map_dict.get(det_i['class'])
        conf = round(det_i['confidence'], 2)
        label_text = '{} {}'.format(label, conf)

        box = det_i['bounding_box']
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        left_x, top_y = img_width * xmin,  img_height * ymin
        width, height = img_width * (xmax - xmin), img_height * (ymax - ymin)

        cmap = mpl.cm.get_cmap("viridis", len(map_dict.keys())).colors
        rect = patches.Rectangle((left_x, top_y),
                                 width,
                                 height,
                                 linewidth=0.5,
                                 edgecolor=cmap[det_i['class']-1],
                                 facecolor='none')
        ax.add_patch(rect)
        plt.text(left_x+2, top_y-2, label_text, color=cmap[det_i['class']-1])

    out_path = os.path.join(out_dir, image_name)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close('all')


def plt_params():
    RC_DICT = {
        "font.size": 3,
    }
    mpl.rcParams.update(RC_DICT)


def get_arguments():
    """
    Get arguments if parsed when executing script.

    :returns: Argument parser object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name',
                        help='Experiment name in AML.',
                        required=True),
    parser.add_argument('--run_id',
                        help='Run ID for experiment.',
                        required=True),
    FLAGS = parser.parse_args()
    return FLAGS


def main():
    # FLAGS = get_arguments()
    __here__ = os.path.dirname(__file__)
    env_cfg, plt_cfg = get_configs(__here__)
    plt_params()
    ws = get_workspace(env_cfg)
    run = rehydrate_run(ws, plt_cfg)
    df_vf = load_version_file(run, env_cfg, plt_cfg)
    download_run_artifacts(run, plt_cfg)
    df_plot = download_images(plt_cfg, env_cfg, df_vf)
    model = load_model(plt_cfg)
    map_dict = load_map_file(plt_cfg)

    score_images(model, map_dict, plt_cfg, df_plot, 0.5)


if __name__ == "__main__":
    main()
