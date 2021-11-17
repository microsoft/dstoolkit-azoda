[[_TOC_]]

# Model Training Process
---

## Introduction
---
This section describes the process to train a model using the Azure ML.

## Training Stages
---
Below illustrates the stages executed during model training and its inputs and artifacts:

![image.png](/docs/.attachments/image-8703edf3-ac31-4614-aec3-58c45ac44f3c.png)

## Training Parameters
---
When executing there are several parameters to be set. In the training directory a file should be created called ``exp_config.json`` with the following content:

``` json
{
    "ENV_CONFIG_FILE": "dev_config.json",
    "DESCRIPTION": "Test training workflow",
    "EXP_NAME" : "pothole",
    "COMPUTE_NAME" : "test",
    "DOCKER_IMAGE" : "csaddevamlacr.azurecr.io/tfod_tf2:latest",
    "DATA_MOUNT" : "test_data",
    "IMAGE_TYPE" : "pothole",
    "TRAIN_CSV" : "latest",
    "TEST_CSV" : "latest",
    "EVAL_CONF" : 0.5,
    "RUN_PARAMS": {
        "STEPS": 1000
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

Below explains each parameter:

- ``ENV_CONFIG_FILE`` - point to the configuration file stored within the configuration section of the repo to point to the desired AML environment.

- ``DESCRIPTION`` - Brief explanation of the runs purpose.

- ``EXP_NAME`` - the experiment group or "use case" being run. Should match existing experiment groups that already exist and it is case sensitive.

- ``COMPUTE_NAME`` - The name of the compute you wish to use. If this does not exist it will create a new compute based on the nodes parameter using NC6 VMs. A list of your created compute can be found in the Azure ML Portal.

- ``DOCKER_IMAGE`` - Base docker image used on the remote compute for training. This is stored in the Azure container registry.

- ``DATA_MOUNT`` - Azure ML datastore reference name set when registering the storage in the portal.

- ``IMAGE_TYPE`` - The type of image or "use case" name as defined in the storage as the use case parent folder.

- ``TRAIN_CSV`` - The Train version file to use for training available in the storage. Set to "latest" to automatically find the latest version file.

- ``TEST_CSV`` - The Train version file to use for testing available in the storage (this should be the same date as the train to ensure no repeat images between sets). Set to "latest" to automatically find the latest version file.

- `EVAL_CONF` - Confidence threshold used to filter out low probability predictions during evaluation.

- ``RUN_PARAMS`` - Group of parameters used for the run. Includes number of steps to execute for a given run.

- ``MODEL_PARAMS`` - Base model and hyper parameters to use for the model training. Details on these can be found based on the existing research papers.

- ``HYPERTUNE`` - Set to true to run intelligent hyper tuning of model parameters.

- ``NODES`` - Number of nodes for the compute target and also used to set the number of hyper tune runs.


## Running Model Training
---
Below explains how to execute model training and observe training progress. This assumes you have already followed the setup process documented in the QuickStart section.

1. Update the ``exp_config.json`` file to the desired experiment configuration

2. Activate your conda environment either in a cmd window or inside VS code.

3. Either run in VScode window or navigate to the script folder and run the following command

    ``` bash
    (yourenvname) C:\<Repo_Name>\src\training\> python submit_training.py
    ```

4. If you are not already you will be prompted to log in. Follow the login details that open in your browser. If you have signed in recently this step will be skipped.

5. To Monitor your run navigate to the Azure ML portal and to your experiment, there you will see your run in progress.


## Experiment Logging
---

### Metrics
Below defines the metrics logged as part of an AML run:
- **Loss** - The error between predicted and actual during model training. For object detection this is broken down into total loss and loss for each stage of the network.
- **mAP** - The mean average precision across all classes. This is a localisation metric and the industry benchmark for object detection.
- **Binary Classification** - Binary Metrics are metrics based on image level classification of clean or defective.

## Tags:
Below defines the tags logged as part of an AML run, these are all based on the values set in the config. They are tagged to allow easy experiment reproduction.
- Description
- Train Set
- Test Set
- Base model
- Steps

### Unlogged Metrics

In addition to the logged metrics there are also some additional metrics that have not yet been integrated into the logging solution.

To access any additional information from the model run and also to support in debugging you can use the logs tab in the AML experiment.

This section records all printouts from the compute job and will catch any additional information not formally parsed to the logging and is recorded and available as the model runs to allow inspection of progress.

During training the model conducts evaluation to see how the model is improving against the test set at regular intervals. This provides the same localisation metrics as the final evaluation but on model checkpoints. Below shows the print block that captures this evaluation.

![image.png](/docs/.attachments/image-f34faaf9-2e1e-40f9-bd0e-a56f5ee8a7b1.png)

The evaluation metrics is based on the evaluation approach, in most cases this is the COCO metrics. More information on these metrics can be found [here](http://cocodataset.org/#detection-eval)

## Tools
---

### Register Datastore
If recreating the Azure environment from scratch you will have created a new datastore. In order to reference this in AML experiments you first need to register the datastore to the AML workspace. Below explains the steps to register a data store:

#### Method - Studio

1. Navigate to the AML studio - https://ml.azure.com/

2. In the right hand menu select datastore

3. At the top of the datastore tab select "new datastore"

4. Fill out the new pain, the name you give of the datastore can then be used in experiments to mount data to the compute cluster.

#### Method 2 - Code
The Azure utils provides functions to register datastores. Refer to the function add_datastore in auzre_utils/azure.py for the inputs.

### Create Training Docker Image
Training is based on docker images to encapsulate dependencies needed during training. By default the devops build pipeline will create a new docker image on each pull request and register it under your_acr.azurecr.io/tensorflowobjectdetection:latest.

To create and register the base docker image for training locally follow the steps below:

1. Ensure you have docker installed on your machine. Desktop Docker can be found [here](https://www.docker.com/products/docker-desktop)

2. Ensure you have created the Azure container registry as part of the environment setup

3. Run the following commands replacing your_image_id and your_acr with the correct values
    ``` bash
    (yourenvname) C:\Repo_Name> docker build . -f docker/tf_2/Dockerfile --no-cache
    (yourenvname) C:\Repo_Name> docker tag your_image_id your_acr.azurecr.io/tensorflowobjectdetection:latest
    (yourenvname) C:\Repo_Name> docker push your_acr.azurecr.io/tensorflowobjectdetection:latest
    ```