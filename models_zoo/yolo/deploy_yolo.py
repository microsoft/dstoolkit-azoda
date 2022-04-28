# %%
# create environment for the deploy
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice
import types
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from datetime import datetime
from azureml.core.webservice import LocalWebservice
from azureml.core.webservice import Webservice
import uuid
from azureml.core.model import InferenceConfig
from azureml.core.model import Model

args = types.SimpleNamespace()
args.tenant_id = '***REMOVED***'
args.service_principal_id = '***REMOVED***'
args.service_principal_password = '***REMOVED***'
args.subscription_id = '***REMOVED***'
args.resource_group = 'tfod-dev-rg-demo'
args.workspace_name = 'tfod-dev-amlw'
args.dataset = 'synthetic_dataset'
# %%
sp = ServicePrincipalAuthentication(tenant_id=args.tenant_id,
                                    service_principal_id=args.service_principal_id,
                                    service_principal_password=args.service_principal_password)
ws = Workspace.get(subscription_id=args.subscription_id,
                   resource_group=args.resource_group,
                   name=args.workspace_name,
                   auth=sp)
time_stamp = datetime.now().strftime('%y%m%d_%H%M%S')
# env = Environment.get(workspace=ws, name="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu")
env = Environment.get(workspace=ws, name="AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu")
env = env.clone("yolo_env")
datastore = ws.get_default_datastore()
# %%

# create deployment config i.e. compute resources
aciconfig = AciWebservice.deploy_configuration(
    cpu_cores=3,
    memory_gb=7,
    tags={"timestamp": "time_stamp"},
    description="Yolov5 inference webservice",
)
#aciconfig = LocalWebservice.deploy_configuration(port=6809)
# %%
# get the registered model
model = Model.register(ws, model_name='yolov5model', model_path='best.pt')

# create an inference config i.e. the scoring script and environment
#env = Environment(name="project_environment")
inference_config = InferenceConfig(entry_script="score.py", source_directory=".", environment=env.from_pip_requirements('myenv', '../../yolo_requirements.txt'))

myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
inference_config = InferenceConfig(entry_script="score.py", source_directory=".", environment=myenv)
# %%
# deploy the service
service_name = "yolov5-infer-" + str(uuid.uuid4())[:4]
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=aciconfig,
)

service.wait_for_deployment(show_output=True)

# %%
service.get_logs()
# %%
service.scoring_uri

# %%
service = Webservice(name=service_name, workspace=ws)
# %%

# %%
service.update(inference_config=inference_config)
service.wait_for_deployment(show_output=True)
print(service.state)
print(service.scoring_uri)
print(service.get_logs())
# %%
import torch
# %%
