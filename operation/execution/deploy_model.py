# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse

from azureml.core import Model

from utils import workspace, deployment


def update_score_script(score_file, model_name):
    """
    Update Scoring Script
    Updates the scoring script with the specific model refernce
    :param score_file: base scoring file
    :param model_name: model name
    :return: path to new scoreing file
    """
    new_score_file = 'src/deployment/model_score.py'

    # Update score.py with params
    with open(score_file, "rt") as score:
        with open(new_score_file, "wt") as new_score:
            for line in score:
                new_score.write(
                    line
                    .replace('__REPLACE_MODEL_NAME__',
                             model_name))
    return new_score_file


def main(model_name, service_name, compute_config_file, environment_path, aks_target_name=None):

    ws = workspace.retrieve_workspace()
    model = Model(ws, name=model_name)

    score_file = os.path.join('src', 'deployment', 'score.py')
    _ = update_score_script(score_file, model_name)

    script_dir = os.path.join('src', 'deployment')

    # Deployment configuration
    deployment_params = deployment.build_deployment_params(
        ws,
        script_dir=script_dir,
        script_file='model_score.py',
        environment_path=environment_path,
        compute_config_file=compute_config_file,
        aks_target_name=aks_target_name
    )

    service = deployment.launch_deployment(
        ws,
        service_name=service_name,
        models=[model],
        deployment_params=deployment_params
    )
    print(f'Waiting for deployment of {service.name} to finish...')
    service.wait_for_deployment(show_output=True)


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--env-path', type=str, required=True)
    parser.add_argument('--service-name', type=str, default='webservice')
    parser.add_argument('--aks-target-name', type=str, default=None)
    return parser.parse_args(args_list)


if __name__ == "__main__":
    args = parse_args()

    main(
        model_name=args.model_name,
        service_name=args.service_name,
        compute_config_file=args.config_path,
        environment_path=args.env_path,
        aks_target_name=args.aks_target_name
    )
