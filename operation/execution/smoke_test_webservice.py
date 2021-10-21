# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
from utils import workspace, webservice

# TODO - update to use a test image instead of JSON for smoke test


def main(webservice_name):
    ws = workspace.retrieve_workspace()
    aml_webservice = webservice.get_webservice(ws, webservice_name)
    if aml_webservice is not None:
        scoring_uri = aml_webservice.scoring_uri
        key = webservice.retrieve_authentication_key(aml_webservice)

        # TODO(mishar) add example image to repo
        sample_file_path = 'operation/tests/data_validation/smoke_test_ws_sample_data.json'
        resp = webservice.call_webservice(scoring_uri+'?prob=0.5',
                                          sample_file_path, key)
        print(resp)


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--webservice-name", type=str, required=True)
    return parser.parse_args(args_list)


if __name__ == "__main__":
    args = parse_args()

    main(
        webservice_name=args.webservice_name
    )
