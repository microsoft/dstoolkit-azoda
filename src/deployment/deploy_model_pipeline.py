'''Deploy Model

Takes in a experiment run and registers the model in the AML registry.
Builds a docker image based on the model and scoring script.
Deploys the model to either an existing ACI or AKS endpoint or creates a new
one.

If the experiment has already registered the model and built and image it will
use the existing image for deployment
'''

import argparse
import os
import sys

from azure_utils.deployment import AMLDeploy

__here__ = os.path.dirname(os.path.realpath(__file__))


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--aml_sub_id',
                        help='AML Subscription ID',
                        required=True),

    parser.add_argument('--aml_rg_group',
                        help='AML Resourcegroup',
                        required=True),

    parser.add_argument('--aml_ws',
                        help='AML workspace Name',
                        required=True)

    parser.add_argument('--aml_tenant',
                        help='AML Tenenant ID',
                        required=True),

    parser.add_argument('--az_cli',
                        help='If already authed with AZ Cli',
                        action='store_true'),

    parser.add_argument('--exp',
                        help='Experiment group run exists in',
                        required=True),

    parser.add_argument('--img_type',
                        help='Image type',
                        required=True),

    parser.add_argument('--build_id',
                        help='Devops Build ID',
                        required=True),

    parser.add_argument('--tf_version',
                        help='TF OD version',
                        required=True),

    parser.add_argument('--compute',
                        help='Compute Target Name',
                        required=True),

    parser.add_argument('--use_aci',
                        help='if flag given be use aci else use AKS',
                        action='store_true'),

    FLAGS = parser.parse_args()
    return FLAGS


def main():

    FLAGS = get_arguments()

    model_name = FLAGS.img_type,
    webservice_name = model_name.lower().replace("_", '')

    if not FLAGS.use_aci:
        webservice_name = webservice_name + '-aks'
        compute_target_name = FLAGS.compute + '-aks'

    # initialse deployment class with paramters
    deployment = AMLDeploy(FLAGS.build_id,
                           FLAGS.exp,
                           webservice_name,
                           model_name,
                           FLAGS.img_type,
                           ws=FLAGS.aml_ws,
                           sub=FLAGS.aml_sub_id,
                           rg=FLAGS.aml_rg_group,
                           tenant=FLAGS.aml_tenant,
                           cli_auth=FLAGS.az_cli)

    # register model if first deployment
    tags = ['build_id', ['build_id', FLAGS.build_id]]
    model = deployment.find_model(model_name, tags=tags)

    if model is None:
        print("the model you provided is not registered")
        print("Either set reg_model as True, or check the model spelling")
        sys.exit()

    # update scoring file
    score_filename, env_filename = ('score_tf2.py', 'conda_env_tf2.yml') if FLAGS.tf_version == 2 \
        else ('score.py', 'conda_env.yml')

    score_file = os.path.join(__here__, score_filename)
    print("Src dir is: {}".format(__here__))

    inference_config = deployment.create_inference_config(score_file,
                                                          __here__,
                                                          env_filename)

    # check for an update existing webservice else create new
    if deployment.webservice_exists(deployment.webservice_name):
        print("INFO: Updating deployed Service")
        deployment.update_existing_webservice(model, inference_config)
    else:
        if FLAGS.use_aci:
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
