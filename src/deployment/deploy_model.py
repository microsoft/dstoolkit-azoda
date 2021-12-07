'''Deploy Model

Takes in a experiment run and registers the model in the AML registry.
Builds a docker image based on the model and scoring script.
Deploys the model to either an existing ACI or AKS endpoint or creates a new
one.

If the experiment has already registered the model and built and image it will
use the existing image for deployment
'''

import os
import sys

from azure_utils.azure import load_config
from azure_utils.deployment import AMLDeploy


__here__ = os.path.dirname(os.path.realpath(__file__))


def get_config():
    deploy_config = os.path.join(__here__, 'deploy_config.json')
    deploy_config = load_config(deploy_config)
    return deploy_config


def main():

    # load deployment configs and set variables
    deploy_config = get_config()

    model_name = deploy_config['IMAGE_TYPE']
    webservice_name = model_name.lower().replace("_", '')

    if not deploy_config['USE_ACI']:
        webservice_name = webservice_name + '-aks'
        compute_target_name = deploy_config['COMPUTE_TARGET_NAME'] + '-aks'

    # initialse deployment class with paramters
    deployment = AMLDeploy(deploy_config['RUN_ID'],
                           deploy_config['EXPERIMENT'],
                           webservice_name,
                           model_name,
                           deploy_config['IMAGE_TYPE'],
                           config_file=deploy_config['ENV_CONFIG_FILE'])

    # register model if first deployment
    if deploy_config['REG_MODEL']:
        model = deployment.register_run_model()
    else:
        tags = ['run_id', ['run_id', deploy_config['RUN_ID']]]
        model = deployment.find_model(model_name, tags)
        if model is None:
            print("ERROR: the model you provided is not registered")
            print("ERROR: Either set reg_model as True, or check spelling")
            sys.exit()

    # update scoring file
    score_filename, env_filename = ('score_tf2.py', 'conda_env_tf2.yml') if deploy_config['TF_VERSION'] == 2 \
        else ('score.py', 'conda_env.yml')

    score_file = os.path.join(__here__, score_filename)
    print("INFO: src dir is: {}".format(__here__))

    inference_config = deployment.create_inference_config(score_file,
                                                          __here__,
                                                          env_filename)

    # check for an update existing webservice else create new
    if deployment.webservice_exists(deployment.webservice_name):
        print("INFO: Updating deployed Service")
        deployment.update_existing_webservice(model, inference_config)
    else:
        if deploy_config['USE_ACI']:
            target, config = deployment.create_aci()
        else:
            target, config = deployment.create_aks(compute_target_name)

        print("INFO: Service dosent exist! Creating new service")
        deployment.deploy_new_webservice(model,
                                         inference_config,
                                         config,
                                         target)


if __name__ == '__main__':
    main()
