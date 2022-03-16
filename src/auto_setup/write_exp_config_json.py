import json
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--experiment_name', help='Name of AML experiment')
parser.add_argument('-r', '--acr', help='Name of ACR')
args = parser.parse_args()

data = {
    "ENV_CONFIG_FILE": "dev_config.json",
    "DESCRIPTION": "description place holder",
    "EXP_NAME": args.experiment_name,
    "COMPUTE_NAME": "gpu-1",
    "DOCKER_IMAGE_NAME": f"{args.acr}.azurecr.io/tensorflowobjectdetection",
    "TF_VERSION": 2,
    "DATA_MOUNT": "workspaceblobstore",
    "IMAGE_TYPE": args.experiment_name,
    "TRAIN_CSV": "latest",
    "TEST_CSV": "latest",
    "EVAL_CONF": 0.5,
    "RUN_PARAMS": {
        "STEPS": 100000
    },
    "MODEL_PARAMS": {
        "BASE_MODEL": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8",
        "FIRST_STAGE": {
            "STRIDE": 16,
            "NMS_IOU_THRESHOLD": 0.5,
            "NMS_SCORE_THRESHOLD": 0.0,
            "MAX_PROPOSALS": 200,
            "LOCALIZATION_LOSS_WEIGHT": 2.0,
            "OBJECTNESS_LOSS_WEIGHT": 1.0
        }
    },
    "HYPERTUNE": False,
    "NODES": 3
}

json_string = json.dumps(data, indent=4)
print(json_string)

# with open('exp_config.json', 'w') as outfile:
#    json.dump(json_string, outfile)

with open('exp_config.json', 'w') as outfile:
    outfile.write(json_string)
