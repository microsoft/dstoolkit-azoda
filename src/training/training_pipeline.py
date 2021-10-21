''' training_pipeline.py

Script to run and publish AML pipelines with the following options
 - execute an AML pipeline run
 - publish an AML pipeline
 - get and update and existing AML pipeline endpoint with new pipeline
'''

import os
from datetime import datetime
from uuid import uuid4

from azureml.core import Datastore, Experiment, RunConfiguration
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineEndpoint, PipelineParameter

from azure_utils.experiment import AMLExperiment
from azure_utils.azure import load_config

__here__ = os.path.dirname(__file__)


def get_configs(__here__):
    exp_config = os.path.join(__here__, 'pipeline_config.json')
    exp_config = load_config(exp_config)
    return exp_config


def set_aml_runconfig(aml_exp, docker_image):
    env = aml_exp.get_env(docker_image)
    aml_run_config = RunConfiguration()
    aml_run_config.environment = env
    return aml_run_config


# --- define params ---
pipe_cfg = get_configs(__here__)

store_name = pipe_cfg['DATA']['DATASTORE_NAME']
cluster_name = pipe_cfg['COMPUTE']['NAME']
nodes = pipe_cfg['COMPUTE']['NODES']
training_script = pipe_cfg['TRAIN_SCRIPT']

docker_image = pipe_cfg['COMPUTE']['DOCKER_IMAGE_NAME']

img_type = PipelineParameter(
  name="img_type",
  default_value=pipe_cfg['DATA']['IMAGE_TYPE'])

train_csv = PipelineParameter(
  name="train_csv",
  default_value=pipe_cfg['DATA']['TRAIN_CSV'])

test_csv = PipelineParameter(
  name="test_csv",
  default_value=pipe_cfg['DATA']['TEST_CSV'])

base_model = PipelineParameter(
  name="base_model",
  default_value=pipe_cfg['BASE_MODEL'])

steps = PipelineParameter(
  name="steps",
  default_value=pipe_cfg['STEPS'])

build_id = PipelineParameter(
  name="build_id",
  default_value='None')

# initialise experiment
aml_exp = AMLExperiment(pipe_cfg['EXP_NAME'],
                        config_file=pipe_cfg['ENV_CONFIG_FILE'])

# set datamount
datastore = Datastore.get(aml_exp.ws, datastore_name=store_name)
data_mount = datastore.as_mount()
input_data = DataReference(datastore=datastore,
                           data_reference_name=store_name)

# set compute target
aml_exp.set_compute(cluster_name, node_count=nodes)

# create estimator
__here__ = os.path.dirname(__file__)
scripts = os.path.join(__here__, 'scripts')

script_args = ["--desc", 'pipline training run',
               "--data_dir", input_data,
               "--image_type", img_type,
               "--train_csv", train_csv,
               "--test_csv", test_csv,
               "--base_model", base_model,
               "--steps", steps,
               "--build_id", build_id]

# --- define pipeline ---
aml_run_config = set_aml_runconfig(aml_exp, docker_image)

ps_step = PythonScriptStep(script_name=training_script,
                           source_directory=scripts,
                           compute_target=aml_exp.compute,
                           runconfig=aml_run_config,
                           arguments=script_args,
                           inputs=[input_data],
                           allow_reuse=False)

pipeline = Pipeline(workspace=aml_exp.ws, steps=[ps_step])


# --- pipline publish params ---
pipeline_name = pipe_cfg['PIPELINE']['NAME']
pipeline_description = pipe_cfg['PIPELINE']['DESCRIPTION']

# --- get existing endpoint ---

# get the existing endpoint if exists and versions the new pipeline,
# else it creates the new pipeline
if pipe_cfg['PIPELINE']['PUBLISH']:
    update_endpoint = True
    try:
        pipeline_endpoint = PipelineEndpoint.get(workspace=aml_exp.ws,
                                                 name=pipeline_name)

        print('INFO: Pipeline Name - {}, Pipeline ID - {}'
              .format(pipeline_endpoint.name, pipeline_endpoint.id))

        date_time = datetime.now().strftime('%Y%m-%d%H-%M%S-')
        date_time_uid = date_time + str(uuid4())
        new_pipeline_name = pipeline_name + '_' + date_time_uid
    except Exception:
        # TODO - find right execption
        new_pipeline_name = pipeline_name
        update_endpoint = False
        pass
else:
    update_endpoint = False

if pipe_cfg['PIPELINE']['RUNFIRST']:

    print('INFO: running pipeline')

    pipeline_run = Experiment(aml_exp.ws,
                              pipe_cfg['EXP_NAME']).submit(pipeline)

    if pipe_cfg['PIPELINE']['WAIT']:
        pipeline_run.wait_for_completion()

    if pipe_cfg['PIPELINE']['PUBLISH']:

        published_pipeline = pipeline_run.publish_pipeline(
            name=pipeline_name,
            description=pipeline_description,
            version='test')

        print('INFO: published pipeline - {}'.format(published_pipeline))

elif pipe_cfg['PIPELINE']['PUBLISH']:

    print('INFO: deploying pipeline without running')

    # --- pulbish pipline from pipline definition ---
    published_pipeline = PipelineEndpoint.publish(
                            workspace=aml_exp.ws,
                            name=new_pipeline_name,
                            pipeline=pipeline,
                            description=pipeline_description)

    print('INFO: published pipeline - {}'.format(published_pipeline))

# --- update existing endpoint ---
# BUG - Waiting on AML PG to response on error.
# if update_endpoint is True:
#     pipeline_endpoint.add(published_pipeline)
#     pipeline_endpoint.set_default(published_pipeline)
