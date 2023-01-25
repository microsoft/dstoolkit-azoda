from azureml.core import Dataset, Experiment, ScriptRunConfig, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment, DEFAULT_GPU_IMAGE
from azureml.data.datapath import DataPath
from datetime import datetime
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--tenant_id", type=str, help="Tenant ID")
parser.add_argument("--service_principal_id", type=str, help="Service Principal ID")
parser.add_argument(
    "--service_principal_password", type=str, help="Serice Principal Password"
)
parser.add_argument("--subscription_id", type=str, help="Subscription ID")
parser.add_argument("--resource_group", type=str, help="Resource Group")
parser.add_argument("--workspace_name", type=str, help="Workspace Name")
parser.add_argument("--mode", type=str, help="train/infer/test")
parser.add_argument("--epochs", type=str, default="5", help="Number of training epochs")
parser.add_argument(
    "--model", type=str, default="yolov5s", help="Name of pretrained model"
)
parser.add_argument(
    "--model_source",
    type=str,
    default="wongkinyiu_yolov7",
    help="Choice of model from model zoo",
)

args = parser.parse_args()
sp = ServicePrincipalAuthentication(
    tenant_id=args.tenant_id,
    service_principal_id=args.service_principal_id,
    service_principal_password=args.service_principal_password,
)
ws = Workspace.get(
    subscription_id=args.subscription_id,
    resource_group=args.resource_group,
    name=args.workspace_name,
    auth=sp,
)
time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
datastore = ws.get_default_datastore()

if args.model_source == "ultralytics_yolov5":
    model_src_dir = "model_zoo/ultralytics_yolov5/"
    env = Environment.from_conda_specification(
        name="myenv", file_path="model_zoo/ultralytics_yolov5/myenv.yml"
    )
    env.docker.base_image = DEFAULT_GPU_IMAGE
    upload_base_dir = "yolov5"
    images_dir = f"{args.dataset}/images/"
    weights_dir = "loaded_weights/weights/best.pt"
elif args.model_source == "wongkinyiu_yolov7":
    model_src_dir = "model_zoo/wongkinyiu_yolov7/"
    env = Environment.from_conda_specification(
        "myenv", file_path="model_zoo/wongkinyiu_yolov7/myenv.yml"
    )
    env.docker.base_image = DEFAULT_GPU_IMAGE
    upload_base_dir = "yolov7"
    images_dir = f"../{args.dataset}/images/"
    weights_dir = "../loaded_weights/best.pt"
else:
    raise ValueError("Invalid model class. Please check the model_source argument")

dockerfile = r"""
FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
"""
env.docker.base_image = None
env.docker.base_dockerfile = dockerfile

if args.mode == "train":
    src = ScriptRunConfig(
        source_directory=model_src_dir,
        script="train_coordinator.py",
        compute_target="gpu-1",
        environment=env,
        arguments=[
            "--dataset",
            args.dataset,
            "--cfg",
            f"{args.model}.yaml",
            "--batch-size",
            "16",
            "--epochs",
            args.epochs,
        ],
    )
    print("Starting run")
    run = Experiment(workspace=ws, name="azoda_train").submit(config=src)
    run.wait_for_completion(show_output=True)
    run.download_files("outputs")
    os.system("echo After download:")
    os.system("pwd")
    os.system("ls -l")
    Dataset.File.upload_directory(
        src_dir="outputs/",
        target=DataPath(datastore, f"{upload_base_dir}_models/{time_stamp}/"),
    )
elif args.mode == "infer":
    src = ScriptRunConfig(
        source_directory=model_src_dir,
        script="infer_coordinator.py",
        compute_target="gpu-1",
        environment=env,
        arguments=[
            "--images",
            images_dir,
            "--weights",
            weights_dir,
            "--conf",
            "0.5",
        ],
    )
    print("Starting run")
    run = Experiment(workspace=ws, name="azoda_infer").submit(config=src)
    run.wait_for_completion(show_output=True)
    run.download_files("outputs")
    Dataset.File.upload_directory(
        src_dir="outputs/",
        target=DataPath(datastore, f"{upload_base_dir}_inferences/{time_stamp}/"),
    )
elif args.mode == "test":
    src = ScriptRunConfig(
        source_directory=model_src_dir,
        script="test_coordinator.py",
        compute_target="gpu-1",
        environment=env,
        arguments=[
            "--dataset",
            args.dataset,
            "--weights",
            weights_dir,
            "--conf",
            "0.5",
            "--iou",
            "0.5",
        ],
    )
    print("Starting run")
    run = Experiment(workspace=ws, name="azoda_test").submit(config=src)
    run.wait_for_completion(show_output=True)
    run.download_files("outputs/")
    Dataset.File.upload_directory(
        src_dir="outputs/",
        target=DataPath(datastore, f"{upload_base_dir}_tests/{time_stamp}/"),
    )
else:
    print("Invalid argument for mode. Please choose from ['train', 'infer', 'test']")
