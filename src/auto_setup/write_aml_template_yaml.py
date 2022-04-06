import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subscription', help='Your Azure subscription')
parser.add_argument('-r', '--resource_group', help='Your Azure resource group')
parser.add_argument('-w', '--workspace', help='Your AML workspace')
args = parser.parse_args()

container_reg = f'/subscriptions/{args.subscription}/resourceGroups/{args.resource_group}/' \
                f'providers/Microsoft.ContainerRegistry/registries/tfod-dev-acr-{args.subscription}'

config_dict = {'$schema': 'https://azuremlschemas.azureedge.net/latest/workspace.schema.json',
               'name': args.workspace,
               'location': 'westeurope',
               'display_name': 'Basic workspace',
               'description': '',
               'container_registry': container_reg,
               'hbi_workspace': False,
               'tags': {'purpose': 'setup'}}
with open('aml_config.yaml', 'w') as file:
    documents = yaml.dump(config_dict, file)
