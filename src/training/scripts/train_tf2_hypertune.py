"""
Example script for hypertuning based on FasterRCNN Inception base model

Arg parse and hparams need to be updated to match your base model if different
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from azureml.core import Run

from tfod_utils.evaluation_tf2 import Scoring
from tfod_utils.evaluation_tf2 import ImageMetrics
from tfod_utils.evaluation_tf2 import DetectionMetrics
from tfod_utils.training_tf2 import TF2ODRun
from tfod_utils.tfrecords import TFRecord

run = Run.get_context()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--desc',
                        help='Description of experiment',
                        required=True),

    parser.add_argument('--data_dir',
                        help='mnt with images and label sets',
                        required=True),

    parser.add_argument('--image_type',
                        help='Image type either thermal or plate',
                        required=True),

    parser.add_argument('--train_csv',
                        help='CSV file containing the training data',
                        default="latest"),

    parser.add_argument('--test_csv',
                        help='CSV file containing the test images and labels',
                        default="latest"),

    parser.add_argument('--base_model',
                        help='Dir name of base model in mnt',
                        required=False),

    parser.add_argument('--steps',
                        help='Number of Steps',
                        default=20000),

    parser.add_argument('--batch_size',
                        help='Batch Size',
                        default=1),

    parser.add_argument('--build_id',
                        help='Batch Size',
                        default=None),

    parser.add_argument('--eval_conf',
                        help='Evaluation Conf Threshold',
                        default=0.5),

    # Hyper Params
    parser.add_argument('--fs_nms_iou',
                        help='First stage IoU',
                        default=0.5,
                        type=float),

    parser.add_argument('--fs_nms_score',
                        help='First stage nms score threshold',
                        default=0.0,
                        type=float),

    parser.add_argument('--fs_max_prop',
                        help='First stage maximum proposals',
                        default=1,
                        type=int),

    parser.add_argument('--fs_loc_loss',
                        help='First stage localization loss weight',
                        default=100,
                        type=int),

    parser.add_argument('--fs_obj_loss',
                        help='First stage objectness loss weight',
                        default=1.0,
                        type=float)

    FLAGS = parser.parse_args()
    return FLAGS


def main():

    # parse arguments
    FLAGS = get_arguments()

    # Set base model dir
    base_model_dir = os.path.join(FLAGS.data_dir, 'models')
    image_dir = 'images'
    label_dir = 'datasets'

    # create tensorflow run object
    train_run = TF2ODRun(run, FLAGS, base_model_dir, image_dir, label_dir)

    # create tf records needed for training
    tfrecords = TFRecord(train_run.base_path,
                         train_run.image_dir,
                         train_run.train_df,
                         train_run.test_df,
                         include_masks=train_run.base_model.startswith('mask_rcnn'))

    # log the details of the configured run object to AML
    train_run.log_details()

    # Kwargs are Model hyperparams to update at runtime
    # keys are paths in pipeline.config file and are updated with the values
    kwargs = {
        'model.faster_rcnn.first_stage_nms_iou_threshold': FLAGS.fs_nms_iou, # noqa
        'model.faster_rcnn.first_stage_nms_score_threshold': FLAGS.fs_nms_score, # noqa
        'model.faster_rcnn.first_stage_max_proposals': FLAGS.fs_max_prop, # noqa
        'model.faster_rcnn.first_stage_localization_loss_weight': FLAGS.fs_loc_loss, # noqa
        'model.faster_rcnn.first_stage_objectness_loss_weight': FLAGS.fs_obj_loss # noqa
        }

    # in addition to tfrecords parse any hyparams required as kwargs
    train_run.update_pipeline_config(tfrecords,
                                     **kwargs)

    # Train
    train_run.train_model()

    # Model Saving Step
    checkpoint_prefix = train_run.base_model_dir
    model_dir = train_run.export_model(checkpoint_prefix)
    saved_model_path = os.path.join(model_dir, 'saved_model')

    # image and test csv path
    img_dir = os.path.join(train_run.base_path, train_run.image_dir)

    test_csv = os.path.join(train_run.base_path,
                            train_run.label_dir,
                            train_run.test_csv)

    # create eval_tf2 object
    dets = Scoring(saved_model_path,
                   train_run.mapping_file,
                   test_csv,
                   img_dir,
                   conf=FLAGS.eval_conf)

    # Calulate image levele metrics
    img_metrics = ImageMetrics(dets)
    # Calculate detection metrics
    det_metrics = DetectionMetrics(dets)

    # AML log img and detection metrics
    img_metrics.log_AML(train_run)
    det_metrics.log_AML(train_run)

    # if triggered from devops we automatically register with the build_id
    if FLAGS.build_id is not None:

        run_id = run.get_details()['runId']

        tags = {'run_id': run_id,
                'build_id': FLAGS.build_id}

        run.register_model(model_name=FLAGS.image_type,
                           model_path=model_dir,
                           tags=tags)


if __name__ == '__main__':
    main()
