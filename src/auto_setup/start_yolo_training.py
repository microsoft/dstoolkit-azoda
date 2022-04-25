# %%
from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig, Dataset
from datetime import datetime
ws = Workspace.from_config()
time_stamp = datetime.now().strftime('%y%m%d_%H%m%S')
env = Environment.get(workspace=ws, name="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu")
env = env.clone("yolo_env")
datastore = ws.get_default_datastore()
dataset = Dataset.File.from_files(path=(datastore, 'synthetic_dataset/'))

src = ScriptRunConfig(source_directory='models/yolo/',
                      script='train_coordinator.py',
                      compute_target='gpu-cluster',
                      environment=env.from_pip_requirements('myenv', 'freeze_alt.txt'),
                      arguments=['--dataset', 'synthetic_dataset',
                                 '--cfg', 'yolov5s.yaml',
                                 '--batch-size', '16',
                                 '--epochs', '5',
                                 '--input_ref', dataset.as_mount()])

run = Experiment(workspace=ws, name='tfod_exp').submit(config=src)
run.wait_for_completion(show_output=True)
run.download_files('outputs')
datastore.upload(src_dir='outputs/', target_path=f'yolov5_models/{time_stamp}/', overwrite=False)
