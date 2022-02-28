# Quickstart

## Introduction

This section explains the quickstart process to setup the solution and create your first model.

The steps are summarised below:

1. Deploy Azure Resources using ARM templates
2. Build and push docker training image
3. Create dataset version files of labelled data
4. Pull desired base model for transfer learning
5. Register datastore
6. Populate configuration files
7. Trigger model training
8. Trigger model deployment


### Python and Anaconda
This repo is built in python 3. It is recommended in addition to having python installed to also use Anaconda to create virtual environments to isolate dependencies and reduce conflicts. Below defines the steps to install and setup a conda virtual environment in python.

1. Install Anaconda on you machine. Navigate to the latest version from [Anaconda Website](https://www.anaconda.com/download/).

2. Create conda environment with Python 3.7, type:
    ```bash
    C:\Repo_Name> conda create -n {yourenvname} python=3.7 anaconda
    ```

3. Activate your environment:
    ```bash
    C:\Repo_Name> conda activate {yourenvname}
    ```

4. Pull a copy of this repo using Git clone and change into the directory.

5. pip install requirements and dependencies:
    ```bash
    (yourenvname) C:\Repo_Name> pip install -r requirements.txt
    ```

### Visual Studio Code
Any python IDE is suitable for working with this repo but a recommendation is the open source IDE [Visual Studio Code](https://code.visualstudio.com/). This tool is free to use and has the added benefit of several available plugins to enhance working with Azure resources used in this project.

## Steps

### 1. Azure Resource Deployment

#### Resource List
Below lists the primary Azure resources used in this project:

- **Azure Machine Learning** - Used for model training experimentation and deployment of models as services.

- **Azure Storage Account** - Used for storage in the labeling workflow and created datasets for model training. Either Fileshare or blob storage can be used for storing data and base models.

#### ARM Templates

The Azure resources required for this project are avaliable as Azure resource templates for easy deployment. The ARM templates used can be obtained in the `azure-pipelines\templates\arm-templates` folder.

Details on how to deploy ARM templates can be found [here](https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/quickstart-create-templates-use-the-portal)

Alternatively, you can make use of the ``PIPELINE-setup.yml`` through Azure DevOps to deploy required resources provided you have a service principle configured.

You can also deploy the Azure Machine Learning and Azure Storage via the Azure Portal.

#### Storage Configuration

Note that within the storage you need to either create a blob container or fileshare to hold the data and base models. As each project is different we have not specified this process and expect it to be setup as part of your data pipelines for managing the labeling flow. Guidance on this can be found in the Labeling section in this wiki. The remainder of this quickstart will expect you to have either the blob container or fileshare configured.

### 2. Build and Push Docker Training Image

To create and register the base docker image for training locally follow the steps below:

1. Ensure you have docker installed on your machine. Desktop Docker can be found [here](https://www.docker.com/products/docker-desktop)

2. Ensure you have created the Azure container registry as part of the environment setup

3. Run the following commands replacing your_image_id and your_acr with the correct values

    ``` bash
    (yourenvname) C:\Repo_Name> docker build . -f docker/tf_2/Dockerfile --no-cache
    (yourenvname) C:\Repo_Name> docker tag your_image_id your_acr.azurecr.io/tensorflowobjectdetection:latest
    (yourenvname) C:\Repo_Name> docker push your_acr.azurecr.io/tensorflowobjectdetection:latest
    ```

### 3. Create datasets

#### Dataset Version Files
Due to different labelling setups, and diversity in different customer data the solution is based of working with a simplified labelling format.

Inline with tensorflows object detection repo the solution uses two datasets a training and an evaluation set. Each dataset is defined as a .csv with the followign column format as shown below.

![datasets.PNG](/docs/.attachments/datasets.PNG)

For examples of converting labelling tool outputs to the .csv format please refer to the example adapter scripts under src/data_orchestration.

Datasets can be saved with any naming convention and referenced in the config file, however to use the built in logic to train on the latest data automatically the following naming convention should be followed:

``` python
train_{usecase}_{YYMMDDHHmmss}
test_{usecase}_{YYMMDDHHmmss}
```
e.g. **train_pothole_210218141224**

#### Data store Setup

The solution expects the following folder setup within the datastore for training.

``` python
{usecase_name}/images
{usecase_name}/datasets
```

This allows for the same setup to be used for multiple models, by createing multiple usecases in the same datastore and grouping experiments within AML.

Your storage should look like the below screenshot

![datasets.PNG](/docs/.attachments/test_data_dir.PNG)

With the datset files looking like below

![dataset_files.PNG](/docs/.attachments/dataset_files.PNG)

### 4. Setup base model

The solution leverages transferlearning from pretrained models provided by tensorflow via there model zoo. The base models are documented in detail in this wiki under "Model Support and Data Augmentation". The base model zoo can be found [here.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

The model comes in ``.tar.gz`` and need to be un-tar using the command documented in the model zoo. This model then needs to be uploaded to the same storage as the datasets, and at the same level as the different usecases, in a folder called models as shown below:

![model_store.PNG](/docs/.attachments/model_store.PNG)

Within that directory each model is stored in a folder and that name is used within the model training config:

![models.PNG](/docs/.attachments/models.PNG)

**Note!** - some models are known to have issues with there base config file, check the Github for issues.


### 5. Register Datastore

Once all the required data and base models are stored within the Azure storage you need to register this storage to AML for the jobs to have access. This can be done in two ways as descibed below.

#### Method 1 - Studio

1. Navigate to the AML studio - https://ml.azure.com/

2. In the right hand menu select datastore

3. At the top of the datastore tab select "new datastore"

4. Fill out the new pain, the name you give of the datastore can then be used in experiments to mount data to the compute cluster.

**Note!** - "Save credentials with the datastore for data access" needs to be set to "Yes" and the account key or SAS token needs to be provided.


#### Method 2 - Code
The Azure utils provides functions to register datastores. Refer to the function add_datastore in auzre_utils/azure.py for the inputs.


### 6. Populate Configuration Files

All paramters for the solution are defined within cofiguration files in .json format. Below provides each config required and where they need to be created within the repo.

#### Environment Config

The environment config stores the information to connect to the AML workspace. This file should be saved the repo root folder configuration/{environment}_config.json. By default the repo has empty dev, test and prod configs. This config name is provided to the other configs for training and deployment and can be named as desired.

Below provides the example config content. In other projects storage accounts and other environment variables have been added to this.

``` json
{
    "AML_TENANT_ID": "",
    "AML_WORKSPACE_NAME" : "",
    "AML_SUBSCRIPTION_ID" : "",
    "AML_RESOURCE_GROUP" : ""
}
```

#### Experiment Config

The experiment config stores all the model training run paramters. It needs to be created under the training directory called exp_config.json and fill with the following parameters. An example file is stored in the required location called exp_config_sample.json, the actual config is set to be ignored by git. A more detailed breakdown of each paramter can be found under model training.

``` json
{
    "ENV_CONFIG_FILE": "dev_config.json",
    "DESCRIPTION": "test training",
    "EXP_NAME" : "pothole",
    "COMPUTE_NAME" : "gpu-1",
    "DOCKER_IMAGE" : "dockerimageacr.azurecr.io/tfod_tf1:dev",
    "TF_VERSION": 2,
    "DATA_MOUNT" : "registered_datastore",
    "IMAGE_TYPE" : "pothole",
    "TRAIN_CSV" : "latest",
    "TEST_CSV" : "latest",
    "EVAL_CONF" : 0.5,
    "RUN_PARAMS": {
        "STEPS": 40000
    },
    "MODEL_PARAMS": {
        "BASE_MODEL": "faster_rcnn_resnet101_coco_2018_01_28",
        "FIRST_STAGE" : {
            "STRIDE": 16,
            "NMS_IOU_THRESHOLD": 0.5,
            "NMS_SCORE_THRESHOLD": 0.0,
            "MAX_PROPOSALS": 200,
            "LOCALIZATION_LOSS_WEIGHT": 2.0,
            "OBJECTNESS_LOSS_WEIGHT": 1.0
        }
    },
    "HYPERTUNE": false,
    "NODES":3
}
```

#### Deployment Config

The deployment config stores the deployment parameters and points to the desired training run to deploy the model from. This config needs to be created under the deployment directory within src called deploy_config.json. An example config is provided already in this location called deploy_config_sample.json. A more detailed breakdown of each paramter can be found under model deployment cloud.

```json
{
    "ENV_CONFIG_FILE": "dev_config.json",
    "EXPERIMENT" : "",
    "RUN_ID" : "",
    "TF_VERSION": 2,
    "REG_MODEL" : true,
    "IMAGE_TYPE": "",
    "ACI_PARAMS": {
        "USE_ACI": true,
        "ACI_AUTH": true
    },
    "AKS_PARAMS": {
        "USE_AKS": false,
        "VM_TYPE": "Standard_NC6",
        "COMPUTE_TARGET_NAME" : ""
    }
}
```

### 7. Trigger Model Training

Once you have setup the environment and configs you can run the model training by following the steps below:

1. Activate your conda environment either in a cmd window or inside VS code.

2. Either press run in VScode window or navigate to the script folder and run the following command

    ``` bash
    (yourenvname) C:\Repo_Name\src\training\> python submit_training.py
    ```

3. If you have not you will be prompted to log in. Follow the login details that open in your browser. If you have signed in recently this step will be skipped.

4. To Monitor your run navigate to the Azure ML portal and to your experiment, there you will see you run in progress and can monitor.


### 8. Trigger Model Deployment

Once you have a completed model training and have updated the deployment config you can execute the deployment as shown below:

1. Activate your conda environment either in a cmd window or inside VS code.

2. Either press run in VScode window or navigate to the script folder and run the following command

    ``` bash
    (yourenvname) C:\Repo_Name\deployment\> python deploy_model.py
    ```

3. If you have not you will be prompted to log in. Follow the login details that open in your browser. If you have signed in recently this step will be skipped.

4. Wait for the deployment process to complete, this can take some time as the docker image needs to be built and registered, additionally if the compute does not exist this needs to be created. On completion you will be able to see the deployed services in the Azure ML Portal.