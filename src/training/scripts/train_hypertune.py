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

from tfod_utils.training import TFODRun
from tfod_utils.evaluation import TFEval
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
    base_model_dir = os.path.join(FLAGS.data_dir,
                                  'models')
    image_dir = 'images'
    label_dir = 'datasets'

    # create tensorflow run object
    train_run = TFODRun(run, FLAGS, base_model_dir, image_dir, label_dir)

    # create tf records needed for training
    tfrecords = TFRecord(train_run.base_path,
                         train_run.image_dir,
                         train_run.train_df,
                         train_run.test_df,
                         include_masks=train_run.base_model.startswith('mask_rcnn'))

    # log the details of the configured run object to AML
    train_run.log_details()

    if train_run.base_model.startswith("ssd") or\
       train_run.base_model.startswith("faster_rcnn") or\
       train_run.base_model.startswith('mask_rcnn'):

        hparams_1 = train_run.set_run_params(tfrecords)

        # update model specific params as key value pairs
        # where key is the config tree path
        hparams_2 = {
            'model.faster_rcnn.first_stage_nms_iou_threshold': FLAGS.fs_nms_iou, # noqa
            'model.faster_rcnn.first_stage_nms_score_threshold': FLAGS.fs_nms_score, # noqa
            'model.faster_rcnn.first_stage_max_proposals': FLAGS.fs_max_prop, # noqa
            'model.faster_rcnn.first_stage_localization_loss_weight': FLAGS.fs_loc_loss, # noqa
            'model.faster_rcnn.first_stage_objectness_loss_weight': FLAGS.fs_obj_loss # noqa
            }

        train_run.update_pipeline_config(hparams_1, hparams_2)

    else:
        raise ValueError('unknown base model {}, \
                          we can only handle ssd, faster_rcnn or mask_rcnn'
                         .format(FLAGS.base_model))

    # Train
    train_run.train_model()

    # Model Saving Step
    checkpoint_prefix = os.path.join(train_run.checkpoint_dir,
                                     'model.ckpt-' + str(train_run.steps))

    model_path = train_run.export_model(checkpoint_prefix)

    img_dir = os.path.join(train_run.base_path, train_run.image_dir)

    test_csv = os.path.join(train_run.base_path,
                            train_run.label_dir,
                            train_run.test_csv)

    # create eval object
    eval_run = TFEval(model_path,
                      train_run.mapping_file,
                      test_csv,
                      img_dir,
                      train_run,
                      conf=FLAGS.eval_conf)

    # run tf built in eval and save checkpoint
    final_map = eval_run.eval_and_log()

    # Run Final Model Evaluation
    binary_dict = eval_run.run_evaluation()

    eval_run.create_summary(final_map,
                            binary_dict)

    # if triggered from devops we automatically register with the build_id
    if FLAGS.build_id is not None:

        run_id = run.get_details()['runId']

        tags = {'run_id': run_id,
                'build_id': FLAGS.build_id}

        model_path = 'outputs/final_model/model/'
        run.register_model(model_name=FLAGS.image_type,
                           model_path=model_path,
                           tags=tags)


if __name__ == '__main__':
    main()
