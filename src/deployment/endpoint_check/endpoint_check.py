'''Endpoint test

Takes in a workspace and service name/type and runs a test imaghe through
the endpoint to test service is working correctly

Used as part of the deployment pipline
'''

import argparse
import json
import requests
import time

from azureml.core import Workspace
from azureml.core.webservice import AksWebservice, AciWebservice


def test_web_service(FLAGS, test_image):

    ws = Workspace(subscription_id=FLAGS.aml_sub_id,
                   resource_group=FLAGS.aml_rg_group,
                   workspace_name=FLAGS.aml_ws)

    service_name = FLAGS.service_name
    service_name = service_name.lower().replace("_", '')

    print("INFO: Getting Service {}".format(service_name))
    if FLAGS.service_type == "ACI":
        service = AciWebservice(ws, service_name)
    else:
        service_name = service_name + "-aks"
        service = AksWebservice(ws, service_name)

    headers = {'Content-Type': 'application/json'}

    print("INFO: Checking for auth and getting keys")
    if service.auth_enabled:
        service_keys = service.get_keys()
        headers['Authorization'] = 'Bearer ' + service_keys[0]

    print("INFO: Loading Test Image")

    img = open(test_image, 'rb').read()

    start_time = time.time()
    data = {"file": img}
    input_data = json.dumps(data)

    print("INFO: Calling Service")
    resp = requests.post(service.scoring_uri + '?prob=0.5',
                         input_data, headers=headers)

    end_time = time.time()
    time_taken = (end_time - start_time)

    print("INFO: Time taken to call service {} seconds".format(time_taken))

    return resp


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

    parser.add_argument("--service_type",
                        type=str,
                        choices=["AKS", "ACI"],
                        required=True,
                        help="type of service to be tested"),

    parser.add_argument("--service_name",
                        type=str,
                        required=True,
                        help="Name of the service to test")

    FLAGS = parser.parse_args()
    return FLAGS


def main():

    FLAGS = get_arguments()
    test_image = "./test_image.JPG"

    resp = test_web_service(FLAGS, test_image)

    print("INFO: Checking Service Response")

    assert resp.status_code == 200

    print("INFO: Service endpoint test successful.")


if __name__ == '__main__':
    main()
