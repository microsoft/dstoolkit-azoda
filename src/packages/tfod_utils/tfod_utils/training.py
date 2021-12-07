import os
import ast

import pandas as pd
import shutil
from datetime import datetime

import tensorflow.compat.v1 as tf

from object_detection import model_hparams
from object_detection import model_lib
from object_detection import exporter
from object_detection.utils import config_util
from object_detection.utils import label_map_util

# used for hparams
from tensorflow.contrib import training as contrib_training

class Logging():
    def __init__(self, run, base_model, log_interval=100):
        self.run = run
        self.base_model = base_model
        self.log_interval = log_interval
        self.total_loss = []

        if self.base_model.startswith("faster_rcnn"):
            self.bccl_loss = []
            self.bcll_loss = []
            self.rpnll_loss = []
            self.rpnol_loss = []

    def append_step_metrics(self,
                            total_loss,
                            bccl_loss=None,
                            bcll_loss=None,
                            rpnll_loss=None,
                            rpnol_loss=None):

        self.total_loss.append(total_loss)

        if self.base_model.startswith("faster_rcnn"):
            self.bccl_loss.append(bccl_loss)
            self.bcll_loss.append(bcll_loss)
            self.rpnll_loss.append(rpnll_loss)
            self.rpnol_loss.append(rpnol_loss)

    def aml_log_average(self):

        window = self.log_interval
        neg_window = window * -1

        total_loss = sum(self.total_loss[neg_window:])/window
        self.run.log('Train - Total Training Loss', total_loss)

        if self.base_model.startswith("faster_rcnn"):
            bccl_loss = sum(self.bccl_loss[neg_window:])/window
            self.run.log('Train - Box Classifier Classification Loss',
                         bccl_loss)

            bcll_loss = sum(self.bcll_loss[neg_window:])/window
            self.run.log('Train - Box Classifier Localization Loss',
                         bcll_loss)
            rpnll_loss = sum(self.rpnll_loss[neg_window:])/window
            self.run.log('Train - RPN Localization Loss', rpnll_loss)

            rpnol_loss = sum(self.rpnol_loss[neg_window:])/window
            self.run.log('Train - RPN Objectness Loss', rpnol_loss)


class TFODRun():
    def __init__(self, run, FLAGS, base_model_dir, image_dir, label_dir,
                 log_interval=None):
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
        self.log_interval = log_interval
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
        self.run.tag('Number of Training Images', count)

        count = len(self.test_df.filename.unique())
        self.run.tag('Number of Test Images', count)

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
            self.train_df = pd.read_csv(os.path.join(self.base_path, train_file), converters={"segmentation": ast.literal_eval})
            self.test_df = pd.read_csv(os.path.join(self.base_path, test_file), converters={"segmentation": ast.literal_eval})
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
        self.output_dir = os.path.join('.', 'outputs', 'final_model')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def set_run_params(self, tfrecords):
        """
        Update run Hparams using tf contrib
        Takes the tfrecords and base run params, creates the tf param object
        and returns it
        :param tfrecords: the path to where the tfrecords are stored
        :returns params: tf contrib hparams object
        """
        self.mapping_file = tfrecords.mapping_file
        self.tfr_train = tfrecords.train_tfrecords
        self.tfr_test = tfrecords.test_tfrecords
        params = contrib_training.HParams(train_steps=self.steps,
                                          label_map_path=self.mapping_file,
                                          train_input_path=self.tfr_train,
                                          eval_input_path=self.tfr_test,
                                          batch_size=int(self.batch_size))
        return params

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
                               hparams_1,
                               hparams_2=None):
        """
        Update Pipline Config
        Updates the base model pipeline config from model zoo with run params
        :param hparams_1: hparams group 1 - default run params
        :param hparams_2: hparams group 2 - model specific hyper params
        provided as a dict, default is None
        """
        org_pipeline_config_file = os.path.join(self.base_model_dir,
                                                self.base_model,
                                                'pipeline.config')

        model = os.path.join(self.base_model_dir,
                             self.base_model,
                             'model.ckpt')

        cfg = (config_util
               .get_configs_from_pipeline_file(org_pipeline_config_file))

        if self.base_model.startswith("ssd"):
            model_cfg = cfg['model'].ssd
        elif self.base_model.startswith("faster_rcnn"):
            model_cfg = cfg['model'].faster_rcnn
        elif self.base_model.startswith("mask_rcnn"):
            model_cfg = cfg['model'].faster_rcnn
        else:
            raise ValueError('unknown base model {}, \
                             we can only handle ssd, faster_rcnn or mask_rcnn'
                             .format(self.base_model))

        label_map_dict = label_map_util.get_label_map_dict(self.mapping_file)
        num_classes = len(label_map_dict)
        model_cfg.num_classes = num_classes

        train_cfg = cfg['train_config']
        train_cfg.fine_tune_checkpoint = model

        if hparams_2:
            cfg = config_util.merge_external_params_with_configs(cfg,
                                                                 hparams_1,
                                                                 hparams_2)
        else:
            cfg = config_util.merge_external_params_with_configs(cfg,
                                                                 hparams_1)

        self.pipeline_config = (config_util
                                .create_pipeline_proto_from_configs(cfg))

        # base model folder will contain base config
        config_dir = os.path.dirname(os.path.realpath(__file__))
        self.pipeline_config_file = os.path.join(config_dir,
                                                 'pipeline.config')

        config_util.save_pipeline_config(self.pipeline_config, config_dir)

    # flake8: noqa: C901
    def create_train_and_eval_specs(self,
                                    train_input_fn,
                                    eval_input_fns,
                                    eval_on_train_input_fn,
                                    predict_input_fn,
                                    train_steps,
                                    eval_on_train_data=False,
                                    final_exporter_name='Servo',
                                    eval_spec_names=None):

        """
        Create the model train and eval specs
        creates the train and eval spec to use at model training run and
        sets the AML logging hooks for the graph
        :param train_input_fn: train_and_eval_dict train input
        :param eval_inputs_fn: train_and_eval_dict eval input
        :param eval_on_train_input_fn: train_and_eval_dict eval on train input
        :param predict_input_fn: train_and_eval_dict predict input
        :param train_steps: train_and_eval_dict train_steps input
        :param eval_on_train_data: option to eval on train data, defaul False
        :param final_exporter_name: final exporter name, default Servo
        :param eval_spec_names: option for different eval spec name,
        default None
        """
        run = self.run

        log_interval = self.log_interval
        if log_interval is None:
            log_interval = 100
        log = Logging(run, self.base_model, log_interval)

        if self.base_model.startswith("ssd"):

            class TrainLoggingHook(tf.estimator.SessionRunHook):
                def after_create_session(self, session, coord):
                    step = session.graph.get_tensor_by_name('global_step:0')

                    tensor = 'total_loss:0'
                    total_loss = session.graph.get_tensor_by_name(tensor)

                    lr = session.graph.get_tensor_by_name('learning_rate:0')

                    self.args = [step, total_loss, lr]

                def before_run(self, run_context):
                    return tf.train.SessionRunArgs(self.args)

                def after_run(self, run_context, run_values):
                    step = run_values.results[0]
                    total_loss = run_values.results[1]
                    lr = run_values.results[2]

                    log.append_step_metrics(total_loss)

                    if step % log_interval == 0 and step > 0:
                        log.aml_log_average()
                        run.log('Train - Learning Rate', lr)

        elif self.base_model.startswith("faster_rcnn") or self.base_model.startswith("mask_rcnn"):

            class TrainLoggingHook(tf.estimator.SessionRunHook):
                def after_create_session(self, session, coord):
                    step = session.graph.get_tensor_by_name('global_step:0')

                    tensor = 'total_loss:0'
                    total_loss = session.graph.get_tensor_by_name(tensor)

                    tensor = 'Loss/BoxClassifierLoss/classification_loss:0'
                    bccl_loss = session.graph.get_tensor_by_name(tensor)

                    tensor = 'Loss/BoxClassifierLoss/localization_loss:0'
                    bcll_loss = session.graph.get_tensor_by_name(tensor)

                    tensor = 'Loss/RPNLoss/localization_loss:0'
                    rpnll_loss = session.graph.get_tensor_by_name(tensor)

                    tensor = 'Loss/RPNLoss/objectness_loss:0'
                    rpnol_loss = session.graph.get_tensor_by_name(tensor)

                    lr = (session.graph.get_tensor_by_name('learning_rate:0'))

                    self.args = [step,
                                 total_loss,
                                 bccl_loss,
                                 bcll_loss,
                                 rpnll_loss,
                                 rpnol_loss,
                                 lr]

                def before_run(self, run_context):
                    return tf.train.SessionRunArgs(self.args)

                def after_run(self, run_context, run_values):
                    step = run_values.results[0]
                    total_loss = run_values.results[1]
                    bccl_loss = run_values.results[2]
                    bcll_loss = run_values.results[3]
                    rpnll_loss = run_values.results[4]
                    rpnol_loss = run_values.results[5]
                    lr = run_values.results[6]

                    log.append_step_metrics(total_loss,
                                            bccl_loss,
                                            bcll_loss,
                                            rpnll_loss,
                                            rpnol_loss)

                    if step % log_interval == 0 and step > 0:
                        log.aml_log_average()
                        run.log('Train - Learning Rate', lr)

        train_spec = tf.estimator.TrainSpec(
                input_fn=train_input_fn,
                max_steps=train_steps,
                hooks=[TrainLoggingHook()])

        # TODO handle different eval specs other than coco
        if eval_spec_names is None:
            eval_spec_names = [str(i) for i in range(len(eval_input_fns))]

        eval_specs = []
        for index, (eval_spec_name,
                    eval_input_fn) in enumerate(zip(eval_spec_names,
                                                    eval_input_fns)):
            if index == 0:
                exporter_name = final_exporter_name
            else:
                exporter_name = '{}_{}'.format(final_exporter_name,
                                               eval_spec_name)

            exporter = tf.estimator.FinalExporter(
                name=exporter_name, serving_input_receiver_fn=predict_input_fn)
            eval_specs.append(
                tf.estimator.EvalSpec(
                    name=eval_spec_name,
                    input_fn=eval_input_fn,
                    steps=None,
                    exporters=exporter))

        if eval_on_train_data:
            eval_specs.append(
                tf.estimator.EvalSpec(name='eval_on_train',
                                      input_fn=eval_on_train_input_fn,
                                      steps=None))

        return train_spec, eval_specs

    def get_spec(self,
                 model_dir,
                 hparams_overrides=None,
                 num_train_steps=None,
                 sample_1_of_n_eval_examples=1,
                 sample_1_of_n_eval_on_train_examples=5):

        """
        Get train and eval spec
        Takes the base model and hparams and creates the estimator and the
        train/eval specs
        :param model_dir: base model directory
        :param hparams_overrides: hparam overides for training
        :param num_train_steps: number of training steps
        :param sample_1_of_n_eval_examples: sample rate of eval on eval
        examples, defaults to 1
        :param sample_1_of_n_eval_on_train_examples: sample rate of
        eval on train examples, defaults to 5
        :return estimator: tensorflow estimator obejct for training
        :return train_spec: tensorflow training spec
        :return eval_spec: tensorflow eval spec
        :return eval_inputs_fns: eval inpust fns object
        """

        tf.logging.set_verbosity(tf.logging.INFO)

        config = tf.estimator.RunConfig(model_dir=model_dir)

        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            run_config=config,
            hparams=model_hparams.create_hparams(hparams_overrides),
            pipeline_config_path=self.pipeline_config_file,
            train_steps=num_train_steps,
            sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(
                sample_1_of_n_eval_on_train_examples))

        estimator = train_and_eval_dict['estimator']
        train_input_fn = train_and_eval_dict['train_input_fn']
        eval_input_fns = train_and_eval_dict['eval_input_fns']
        eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        predict_input_fn = train_and_eval_dict['predict_input_fn']
        train_steps = train_and_eval_dict['train_steps']

        train_spec, \
            eval_specs = (self.create_train_and_eval_specs
                          (train_input_fn,
                           eval_input_fns,
                           eval_on_train_input_fn,
                           predict_input_fn,
                           train_steps,
                           eval_on_train_data=False))

        return estimator, train_spec, eval_specs[0], eval_input_fns

    def train_model(self):
        """
        Train Model
        Execute the model training based on the run specs
        """
        self.checkpoint_dir = os.path.join(self.log_dir, 'model')
        if (os.path.exists(self.checkpoint_dir)):
            shutil.rmtree(self.checkpoint_dir)

        self.estimator, \
            train_spec, \
            eval_specs, \
            self.eval_input_fns = self.get_spec(self.checkpoint_dir)

        tf.estimator.train_and_evaluate(self.estimator,
                                        train_spec,
                                        eval_specs)

    def export_model(self,
                     checkpoint_prefix):
        """
        Export Model
        Export the trained model to file
        :param checkpoint_prefix: the model checkpoint prefix to export
        """
        export_dir = os.path.join(self.output_dir,
                                  'model')

        if (os.path.exists(export_dir)):  # no overwrite option for exporting
            shutil.rmtree(export_dir)

        exporter.export_inference_graph('image_tensor',
                                        self.pipeline_config,
                                        checkpoint_prefix,
                                        export_dir)

        # copy mapping file to model output
        map_file = os.path.basename(self.mapping_file)
        out_map_file = os.path.join(export_dir, map_file)
        shutil.copy(self.mapping_file, out_map_file)

        return export_dir
