from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment, DEFAULT_GPU_IMAGE
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
parser.add_argument("--mode", type=str, help="train/infer")
parser.add_argument("--epochs", type=str, default='5', help="Number of training epochs")
parser.add_argument("--model", type=str, default='yolov5s', help="Number of training epochs")

args = parser.parse_args()
sp = ServicePrincipalAuthentication(tenant_id=args.tenant_id,
                                    service_principal_id=args.service_principal_id,
                                    service_principal_password=args.service_principal_password)
ws = Workspace.get(subscription_id=args.subscription_id,
                   resource_group=args.resource_group,
                   name=args.workspace_name,
                   auth=sp)
time_stamp = datetime.now().strftime('%y%m%d_%H%M%S')
env = Environment.from_conda_specification(name="myenv", file_path="model_zoo/ultralytics_yolov5/myenv.yml")
env.docker.base_image = DEFAULT_GPU_IMAGE
datastore = ws.get_default_datastore()
if args.mode == 'train':
    src = ScriptRunConfig(source_directory='model_zoo/ultralytics_yolov5/',
                          script='train_coordinator.py',
                          compute_target='gpu-cluster',
                          environment=env,
                          arguments=['--dataset', args.dataset,
                                     '--cfg', f'{args.model}.yaml',
                                     '--batch-size', '16',
                                     '--epochs', args.epochs])
    print('Starting run')
    run = Experiment(workspace=ws, name='azoda_train').submit(config=src)
    run.wait_for_completion(show_output=True)
    run.download_files('outputs')
    datastore.upload(src_dir='outputs/', target_path=f'yolov5_models/{time_stamp}/', overwrite=False)
elif args.mode == 'infer':
    src = ScriptRunConfig(source_directory='model_zoo/ultralytics_yolov5/',
                          script='infer_coordinator.py',
                          compute_target='gpu-cluster',
                          environment=env.from_pip_requirements('myenv',
                                                                'model_zoo/ultralytics_yolov5/yolov5/requirements.txt'),
                          arguments=['--images', f'{args.dataset}/images/',
                                     '--weights', 'loaded_weights/weights/best.pt',
                                     '--conf', '0.5'])
    print('Starting run')
    run = Experiment(workspace=ws, name='azoda_infer').submit(config=src)
    run.wait_for_completion(show_output=True)
    run.download_files('outputs')
    datastore.upload(src_dir='outputs/', target_path=f'yolov5_inferences/{time_stamp}/', overwrite=False)
elif args.mode == 'test':
    src = ScriptRunConfig(source_directory='model_zoo/ultralytics_yolov5/',
                          script='test_coordinator.py',
                          compute_target='gpu-cluster',
                          environment=env.from_pip_requirements('myenv',
                                                                'model_zoo/ultralytics_yolov5/yolov5/requirements.txt'),
                          arguments=['--dataset', args.dataset,
                                     '--weights', 'loaded_weights/weights/best.pt',
                                     '--conf', '0.5',
                                     '--iou', '0.5'])
    print('Starting run')
    run = Experiment(workspace=ws, name='azoda_test').submit(config=src)
    run.wait_for_completion(show_output=True)
    run.download_files('outputs')
    datastore.upload(src_dir='outputs/', target_path=f'yolov5_tests/{time_stamp}/', overwrite=False)
else:
    print('Invalid argument for mode. Please choose from [\'train\', \'infer\', \'test\']')
