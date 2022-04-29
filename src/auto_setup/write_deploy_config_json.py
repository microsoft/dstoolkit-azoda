import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment_name', help='Name of AML experiment')
parser.add_argument('-r', '--run_id', help='ID of the target run')
args = parser.parse_args()

data = {
    "ENV_CONFIG_FILE": "dev_config.json",
    "EXPERIMENT": args.experiment_name,
    "RUN_ID": args.run_id,
    "TF_VERSION": 2,
    "REG_MODEL": True,
    "IMAGE_TYPE": args.experiment_name,
    "ACI_PARAMS": {
        "USE_ACI": True,
        "ACI_AUTH": True
    },
    "AKS_PARAMS": {
        "USE_AKS": False,
        "VM_TYPE": "Standard_NC6",
        "COMPUTE_TARGET_NAME": "gpu-1"
    }
}

json_string = json.dumps(data, indent=4)
print(json_string)

with open('deploy_config.json', 'w') as outfile:
    outfile.write(json_string)
