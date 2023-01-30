import argparse
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.environment import Environment, DEFAULT_GPU_IMAGE
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AksWebservice
from azureml.exceptions import ComputeTargetException
from datetime import datetime
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--tenant_id", type=str, help="Tenant ID")
parser.add_argument("--service_principal_id", type=str, help="Service Principal ID")
parser.add_argument(
    "--service_principal_password", type=str, help="Service Principal Password"
)
parser.add_argument("--subscription_id", type=str, help="Subscription ID")
parser.add_argument("--resource_group", type=str, help="Resource Group")
parser.add_argument("--workspace_name", type=str, help="Workspace Name")
parser.add_argument("--dataset", type=str, help="Dataset")
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
aks_name = "aks-auto"

# Check to see if the cluster already exists
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print("Found existing compute target")
except ComputeTargetException:
    print("Creating a new compute target...")
    # Provision AKS cluster with GPU machine
    prov_config = AksCompute.provisioning_configuration(
        vm_size="Standard_NC6", agent_count=1
    )
    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace=ws, name=aks_name, provisioning_configuration=prov_config
    )
    aks_target.wait_for_completion(show_output=True)

gpu_aks_config = AksWebservice.deploy_configuration(
    autoscale_enabled=True, cpu_cores=2, memory_gb=6
)

myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
myenv.docker.base_image = DEFAULT_GPU_IMAGE
inference_config = InferenceConfig(
    entry_script="score_gpu.py", source_directory=".", environment=myenv
)

# get the registered model
model = Model.register(ws, model_name="yolov5model", model_path="best.pt")

# Name of the web service that is deployed
aks_service_name = "aks-" + str(uuid.uuid4())[:4]

# Deploy the model
aks_service = Model.deploy(
    ws,
    models=[model],
    inference_config=inference_config,
    deployment_config=gpu_aks_config,
    deployment_target=aks_target,
    name=aks_service_name,
)

aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)
print(aks_service.scoring_uri)
