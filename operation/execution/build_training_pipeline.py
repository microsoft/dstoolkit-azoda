# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azureml.core import Datastore, Environment
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration

from utils import config, workspace, compute, pipeline


def main(train_script,
         base_model,
         image_type,
         datastore_name,
         train_csv,
         test_csv,
         model_steps,
         pipeline_name,
         compute_name,
         environment_path,
         pipeline_version):

    # Retrieve workspace
    ws = workspace.retrieve_workspace()

    # Training setup
    compute_target = compute.get_compute_target(ws, compute_name)
    env = Environment.load_from_directory(path=environment_path)
    run_config = RunConfiguration()
    run_config.environment = env

    # set datamount
    datastore = Datastore.get(ws, datastore_name=datastore_name)
    input_data = DataReference(datastore=datastore,
                               data_reference_name=datastore_name)

    script_args = ["--desc", 'pipline training run',
                   "--data_dir", input_data,
                   "--image_type", image_type,
                   "--train_csv", train_csv,
                   "--test_csv", test_csv,
                   "--base_model", base_model,
                   "--steps", model_steps,
                   "--build_id", pipeline_version]

    # --- define pipeline ---
    train_step = PythonScriptStep(script_name=train_script,
                                  source_directory="src",
                                  compute_target=compute_target,
                                  runconfig=run_config,
                                  arguments=script_args,
                                  inputs=[input_data],
                                  allow_reuse=False)

    # Publish training pipeline
    published_pipeline = pipeline.publish_pipeline(
        ws,
        name=pipeline_name,
        steps=[train_step],
        description="Model training/retraining pipeline",
        version=pipeline_version
    )

    print(f"Published pipeline {published_pipeline.name}\
            version {published_pipeline.version}")


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == "__main__":
    args = parse_args()

    # Get argurments from environment
    # (these variables are defined in the yml file)

    main(
        train_script=config.get_env_var("TRAIN_SCRIPT"),
        base_model=config.get_env_var("BASE_MODEL"),
        image_type=config.get_env_var("MODEL_IMAGE_TYPE"),
        datastore_name=config.get_env_var("DATASTORE_NAME"),
        train_csv=config.get_env_var("MODEL_TRAIN_CSV"),
        test_csv=config.get_env_var("MODEL_TEST_CSV"),
        model_steps=config.get_env_var("MODEL_STEPS"),
        pipeline_name=config.get_env_var("AML_TRAINING_PIPELINE"),
        compute_name=config.get_env_var("AML_TRAINING_COMPUTE"),
        environment_path=config.get_env_var("AML_TRAINING_ENV_PATH"),
        pipeline_version=args.version
    )
