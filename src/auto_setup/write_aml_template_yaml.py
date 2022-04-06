import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subscription', help='Your Azure subscription')
args = parser.parse_args()

container_reg = f'/subscriptions/{args.subscription}/resourceGroups/tfod-dev-rg/' \
                f'providers/Microsoft.ContainerRegistry/registries/tfod-dev-acr-{args.subscription}'

config_dict = {'$schema': 'https://azuremlschemas.azureedge.net/latest/workspace.schema.json',
               'name': 'tfod-dev-amlw',
               'location': 'westeurope',
               'display_name': 'Basic workspace',
               'description': '',
               'container_registry': container_reg,
               'hbi_workspace': False,
               'tags': {'purpose': 'setup'}}
with open('aml_config.yaml', 'w') as file:
    documents = yaml.dump(config_dict, file)
