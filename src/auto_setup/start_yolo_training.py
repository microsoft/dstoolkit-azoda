from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--tenant_id", type=str, help="Tenant ID")
parser.add_argument("--service_principal_id", type=str, help="Service Principal ID")
parser.add_argument("--service_principal_password", type=str, help="Serice Principal Password")
parser.add_argument("--subscription_id", type=str, help="Subscription ID")
parser.add_argument("--resource_group", type=str, help="Resource Group")
parser.add_argument("--workspace_name", type=str, help="Workspace Name")
args = parser.parse_args()

sp = ServicePrincipalAuthentication(tenant_id=args.tenant_id,
                                    service_principal_id=args.service_principal_id,
                                    service_principal_password=args.service_principal_password)
ws = Workspace.get(subscription_id=args.subscription_id,
                   resource_group=args.resource_group,
                   name=args.workspace_name,
                   auth=sp)
time_stamp = datetime.now().strftime('%y%m%d_%H%m%S')
env = Environment.get(workspace=ws, name="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu")
env = env.clone("yolo_env")
datastore = ws.get_default_datastore()
dataset_name = args.dataset
dataset = Dataset.File.from_files(path=(datastore, f'{dataset_name}/'))

src = ScriptRunConfig(source_directory='models/yolo/',
                      script='train_coordinator.py',
                      compute_target='gpu-cluster',
                      environment=env.from_pip_requirements('myenv', 'yolo_requirements.txt'),
                      arguments=['--dataset', dataset_name,
                                 '--cfg', 'yolov5s.yaml',
                                 '--batch-size', '16',
                                 '--epochs', '5'])
print('Starting run')
run = Experiment(workspace=ws, name='tfod_exp').submit(config=src)
run.wait_for_completion(show_output=True)
run.download_files('outputs')
datastore.upload(src_dir='outputs/', target_path=f'yolov5_models/{time_stamp}/', overwrite=False)
