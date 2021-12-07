import os
import ast
import time
import pprint

import pandas as pd
import shutil
from datetime import datetime

import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
from object_detection import exporter_lib_v2
from object_detection.utils import config_util

from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import variables_helper

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)


class TF2ODRun():
    def __init__(self, run, FLAGS, base_model_dir, image_dir, label_dir):
        """
        Tensorflow Object Detection Class
        This class groups all TF object detection functions and runs
        on the compute within the run
        :param run: the AML run object
        :param FLAGS: argument flags from AML for run configuration
        :param base_model_dir: path to base models
        :param image_dir: path to images
        :param label_dir: path to label version files
        """
        self.run = run
        self.desc = FLAGS.desc
        self.steps = FLAGS.steps
        self.batch_size = FLAGS.batch_size
        self.train_csv = FLAGS.train_csv
        self.test_csv = FLAGS.test_csv
        self.data_dir = FLAGS.data_dir
        self.image_type = FLAGS.image_type
        self.base_model_dir = base_model_dir
        self.base_model = FLAGS.base_model
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.base_path = os.path.join(self.data_dir,
                                      self.image_type)

        if self.train_csv == "latest":
            self.get_latest_dataset()

        self.set_outdirs()
        self.load_dataset(label_dir)

    def log_details(self):
        """
        Log details
        Logs the basic run details like run description
        """
        self.run.tag('description', self.desc)
        self.run.tag('train set', self.train_csv)
        self.run.tag('test set', self.test_csv)
        self.run.tag('base model', self.base_model)
        self.run.tag('steps', self.steps)

        count = len(self.train_df.filename.unique())
        self.run.log('Number of Training Images', count)

        count = len(self.test_df.filename.unique())
        self.run.log('Number of Test Images', count)

    def load_dataset(self, sub_dir):
        """
        Load Dataset
        load the traina nd test files to pandas dataframe
        :param sub_dir: the sub directory in the store that contains the
        version files
        """
        train_file = os.path.join(sub_dir,
                                  self.train_csv)

        test_file = os.path.join(sub_dir,
                                 self.test_csv)

        if self.base_model.startswith("mask_rcnn"):
            self.train_df = pd.read_csv(os.path.join(self.base_path, train_file),
                                        converters={"segmentation": ast.literal_eval})
            self.test_df = pd.read_csv(os.path.join(self.base_path, test_file),
                                       converters={"segmentation": ast.literal_eval})
            # For the moment support only segmentation based on polygons (object instance)
            self.train_df["iscrowd"] = 0
            self.test_df["iscrowd"] = 0
        else:
            self.train_df = pd.read_csv(os.path.join(self.base_path, train_file))
            self.test_df = pd.read_csv(os.path.join(self.base_path, test_file))

    def set_outdirs(self):
        """
        Set the output directories
        Sets the output directories for models and logs to write too.
        """
        self.log_dir = os.path.join('.', 'outputs', 'logging')
        self.output_dir = os.path.join('.', 'outputs', 'model')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_latest_dataset(self):
        """
        Get Latest Dataset
        Gets the lastest version files based on the current date and
        file name creation date format
        """
        version_files = os.path.join(self.data_dir,
                                     self.image_type,
                                     self.label_dir)
        first = True
        latest = ''
        for f in os.listdir(version_files):
            if os.path.isfile(os.path.join(version_files, f)):
                date = datetime.strptime((f.split('_')[-1].split('.')[0]),
                                         '%y%m%d%H%M%S')
                if first:
                    latest = date
                    first = False
                else:
                    if date > latest:
                        latest = date
        latest = latest.strftime('%y%m%d%H%M%S')
        self.train_csv = 'train_{}_{}.csv'.format(self.image_type, latest)
        self.test_csv = 'test_{}_{}.csv'.format(self.image_type, latest)

    def update_pipeline_config(self,
                               tfrecords,
                               **kwargs):

        # copy the model to the outputs location of the node
        # to make it avliable in AML stuido
        new_model_dir = os.path.join(self.output_dir, self.base_model)
        base_model = os.path.join(self.base_model_dir, self.base_model)
        shutil.copytree(base_model, new_model_dir)
        self.base_model_dir = self.output_dir

        org_pipeline_config_file = os.path.join(self.base_model_dir,
                                                self.base_model,
                                                'pipeline.config')

        model_checkpoint = os.path.join(self.base_model_dir,
                                        self.base_model,
                                        'checkpoint',
                                        'ckpt-0')

        self.mapping_file = tfrecords.mapping_file
        self.tfr_train = tfrecords.train_tfrecords
        self.tfr_test = tfrecords.test_tfrecords
        num_classes = len(tfrecords.classes)

        # TODO(FREDRIK) - Add number of classes
        kwargs.update({
            'num_classes': num_classes,
            'train_config.fine_tune_checkpoint': model_checkpoint,
            'train_config.fine_tune_checkpoint_type': 'detection',
            'train_steps': self.steps,
            'label_map_path': self.mapping_file,
            'train_input_path': self.tfr_train,
            'eval_input_path': self.tfr_test,
            'batch_size': int(self.batch_size),
        })

        cfg = (config_util
               .get_configs_from_pipeline_file(org_pipeline_config_file))

        configs = config_util.merge_external_params_with_configs(
            cfg, None, kwargs_dict=kwargs)

        # set local config location
        config_dir = os.path.dirname(os.path.realpath(__file__))
        self.pipeline_config_file = os.path.join(config_dir,
                                                 'pipeline.config')

        self.pipeline_config = (config_util
                                .create_pipeline_proto_from_configs(configs))

        config_util.save_pipeline_config(self.pipeline_config, config_dir)

    def train_model(self, num_gpus=None):
        """
        Train Model
        Execute the model training based on the run specs
        """
        if num_gpus is None:
            num_gpus = len(tf.config.list_physical_devices('GPU'))

        if num_gpus > 1:
            print(f'INFO: Using multiple GPUs, count = {num_gpus}.')
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            if num_gpus == 0:
                print('INFO: Using CPU.')
            else:
                print('INFO: Using single GPU.')

            strategy = tf.compat.v2.distribute.MirroredStrategy()

        with strategy.scope():
            self.train_loop(
                pipeline_config_path=self.pipeline_config_file,
                model_dir=self.base_model_dir,
                train_steps=self.steps)

    def export_model(self,
                     checkpoint_prefix):
        """
        Export Model
        Export the trained model to file
        :param checkpoint_prefix: the model checkpoint prefix to export
        """
        export_dir = os.path.join('.',
                                  'outputs',
                                  'final_model',
                                  'model')

        if (os.path.exists(export_dir)):  # no overwrite option for exporting
            shutil.rmtree(export_dir)

        exporter_lib_v2.export_inference_graph('image_tensor',
                                               self.pipeline_config,
                                               checkpoint_prefix,
                                               export_dir)

        # copy mapping file to model output
        map_file = os.path.basename(self.mapping_file)
        out_map_file = os.path.join(export_dir, map_file)
        shutil.copy(self.mapping_file, out_map_file)

        return export_dir

    # function from tensorflow model lib v2 with small change for AML logging
    def train_loop(self,  # noqa: C901
                   pipeline_config_path,
                   model_dir,
                   train_steps,
                   checkpoint_every_n=1000,
                   checkpoint_max_to_keep=7,
                   record_summaries=True,
                   num_steps_per_iteration=100):

        print("\n========== TRAINING STARTING ==========\n")
        train_steps = int(train_steps)

        run = self.run

        MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP
        steps_per_sec_list = []

        configs = config_util.get_configs_from_pipeline_file(
            pipeline_config_path)

        model_config = configs['model']
        train_config = configs['train_config']
        train_input_config = configs['train_input_config']

        unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
        add_regularization_loss = train_config.add_regularization_loss
        clip_gradients_value = None
        if train_config.gradient_clipping_by_norm > 0:
            clip_gradients_value = train_config.gradient_clipping_by_norm

        if train_config.load_all_detection_checkpoint_vars:
            raise ValueError('train_pb2.load_all_detection_checkpoint_vars '
                             'unsupported in TF2')

        fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type
        fine_tune_checkpoint_version = train_config.fine_tune_checkpoint_version

        # Build the model, optimizer, and training input
        strategy = tf.compat.v2.distribute.get_strategy()
        with strategy.scope():
            detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
                model_config=model_config, is_training=True,
                add_summaries=record_summaries)

            def train_dataset_fn(input_context):
                """Callable to create train input."""
                # Create the inputs.
                train_input = inputs.train_input(
                    train_config=train_config,
                    train_input_config=train_input_config,
                    model_config=model_config,
                    model=detection_model,
                    input_context=input_context)
                train_input = train_input.repeat()
                return train_input

            train_input = strategy.experimental_distribute_datasets_from_function(
                train_dataset_fn)

            global_step = tf.Variable(
                0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
                aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)
            optimizer, (learning_rate,) = optimizer_builder.build(
                train_config.optimizer, global_step=global_step)

            # We run the detection_model on dummy inputs in order to ensure that the
            # model and all its variables have been properly constructed. Specifically,
            # this is currently necessary prior to (potentially) creating shadow copies
            # of the model variables for the EMA optimizer.
            if train_config.optimizer.use_moving_average:
                model_lib_v2.model_ensure_model_is_built(
                    detection_model,
                    train_input,
                    unpad_groundtruth_tensors)
                optimizer.shadow_copy(detection_model)

            if callable(learning_rate):
                learning_rate_fn = learning_rate
            else:
                def learning_rate_fn(): return learning_rate  # noqa

        # Train the model
        # Get the appropriate filepath (temporary or not) based on whether the worker
        # is the chief.
        summary_writer_filepath = model_lib_v2.get_filepath(strategy, os.path.join(model_dir, 'train'))

        summary_writer = tf.compat.v2.summary.create_file_writer(
            summary_writer_filepath)

        with summary_writer.as_default():
            with strategy.scope():
                with tf.compat.v2.summary.record_if(lambda: global_step % num_steps_per_iteration == 0):

                    # Load a fine-tuning checkpoint.
                    if train_config.fine_tune_checkpoint:
                        variables_helper.ensure_checkpoint_supported(
                            train_config.fine_tune_checkpoint,
                            fine_tune_checkpoint_type,
                            model_dir)

                        model_lib_v2.load_fine_tune_checkpoint(
                            detection_model, train_config.fine_tune_checkpoint,
                            fine_tune_checkpoint_type, fine_tune_checkpoint_version,
                            train_config.run_fine_tune_checkpoint_dummy_computation,
                            train_input, unpad_groundtruth_tensors)

                    ckpt = tf.compat.v2.train.Checkpoint(
                        step=global_step, model=detection_model, optimizer=optimizer)

                    manager_dir = model_lib_v2.get_filepath(strategy, model_dir)
                    if not strategy.extended.should_checkpoint:
                        checkpoint_max_to_keep = 1
                    manager = tf.compat.v2.train.CheckpointManager(
                        ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep)

                    # We use the following instead of manager.latest_checkpoint because
                    # manager_dir does not point to the model directory when we are running
                    # in a worker.
                    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
                    ckpt.restore(latest_checkpoint)

                    def train_step_fn(features, labels):
                        """Single train step."""
                        if record_summaries:
                            tf.compat.v2.summary.image(
                                name='train_input_images',
                                step=global_step,
                                data=features[fields.InputDataFields.image],
                                max_outputs=3)
                        losses_dict = model_lib_v2.eager_train_step(
                            detection_model,
                            features,
                            labels,
                            unpad_groundtruth_tensors,
                            optimizer,
                            add_regularization_loss=add_regularization_loss,
                            clip_gradients_value=clip_gradients_value,
                            num_replicas=strategy.num_replicas_in_sync)
                        global_step.assign_add(1)
                        return losses_dict

                    def _sample_and_train(strategy, train_step_fn, data_iterator):
                        features, labels = data_iterator.next()
                        if hasattr(tf.distribute.Strategy, 'run'):
                            per_replica_losses_dict = strategy.run(
                                train_step_fn, args=(features, labels))
                        else:
                            per_replica_losses_dict = (
                                strategy.experimental_run_v2(
                                    train_step_fn, args=(features, labels)))

                        return model_lib_v2.reduce_dict(
                            strategy, per_replica_losses_dict, tf.distribute.ReduceOp.SUM)

                    @tf.function
                    def _dist_train_step(data_iterator):
                        """A distributed train step."""

                        if num_steps_per_iteration > 1:
                            for _ in tf.range(num_steps_per_iteration - 1):
                                with tf.name_scope(''):
                                    _sample_and_train(strategy,
                                                      train_step_fn,
                                                      data_iterator)

                        return _sample_and_train(strategy, train_step_fn,
                                                 data_iterator)

                    train_input_iter = iter(train_input)

                    if int(global_step.value()) == 0:
                        manager.save()

                    checkpointed_step = int(global_step.value())
                    logged_step = global_step.value()

                    last_step_time = time.time()
                    for _ in range(global_step.value(), train_steps,
                                   num_steps_per_iteration):

                        losses_dict = _dist_train_step(train_input_iter)

                        time_taken = time.time() - last_step_time
                        last_step_time = time.time()
                        steps_per_sec = num_steps_per_iteration * 1.0 / time_taken

                        tf.compat.v2.summary.scalar(
                            'steps_per_sec', steps_per_sec, step=global_step)

                        steps_per_sec_list.append(steps_per_sec)

                        logged_dict = losses_dict.copy()
                        logged_dict['learning_rate'] = learning_rate_fn()

                        for key, val in logged_dict.items():
                            tf.compat.v2.summary.scalar(key, val, step=global_step)

                        if global_step.value() - logged_step >= 100:
                            logged_dict_np = {name: value.numpy() for name, value in
                                              logged_dict.items()}

                            tf.compat.v1.logging.info(
                                'Step {} per-step time {:.3f}s'.format(
                                    global_step.value(), time_taken / num_steps_per_iteration))

                            tf.compat.v1.logging.info(pprint.pformat(logged_dict_np, width=40))
                            logged_step = global_step.value()

                            for k, v in logged_dict_np.items():
                                run.log(k, v)

                        if ((int(global_step.value()) - checkpointed_step) >= checkpoint_every_n):
                            manager.save()
                            checkpointed_step = int(global_step.value())

        # Remove the checkpoint directories of the non-chief workers that
        # MultiWorkerMirroredStrategy forces us to save during sync distributed
        # training.
        model_lib_v2.clean_temporary_directories(strategy, manager_dir)
        model_lib_v2.clean_temporary_directories(strategy, summary_writer_filepath)
        print("\n========== TRAINING FINISHED ==========\n")
