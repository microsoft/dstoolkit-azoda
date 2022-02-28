# Model Deployment Process

## Introduction

This section describes the process to take a trained model experiment and deploy as a webservice to be consumed by applications. Note, with model training if using the submit training during deployment you will need to set the `REG_MODEL` to `true` as this process does not automatically register the model. If using the AML pipeline this step is included.

## Deployment Stages

Below illustrates the deployment script stages.

![image.png](/docs/.attachments/image-1afe45e5-5f4b-4db7-898e-1e057b117568.png)

## Deployment Parameters

When deploying a model there are several parameters to be set. In order to make this simple they are set via a config file. In the deployment directory there is a config file called `deploy_config.json` with the following content:

``` json
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

Below a breakdown of each parameter can be obtained:

- `ENV_CONFIG_FILE` - point to the configuration file stored within the configuration section of the repo to point to the desired AML environment

- `EXPERIMENT` - this refers to the experiment group you run sits within in AML

- `RUN_ID` - This is the run ID from AML of the model you wish to deploy. This can be found from the Experiment view in AML

- `TF_VERSION` - This is the tensorflow version (1 or 2), needed to specify different scoring script

- `REG_MODEL` - If the model needs to be registered or not from the run_id. If set to false the deployment will check the model registry for a registered model with a runid tag matching that of your set RUN_ID

- `IMAGE_TYPE` - The type of image or "use case" name as defined in the storage as the use case parent folder.

- `ACI_PARAMS.USE_ACI` - Set this to true if you wish to deploy a model to ACI (this is for testing purposes only)

- `ACI_PARAMS.ACI_AUTH` - Whether to enable key-based authentication when using ACI, default value is true

- `AKS_PARAMS.USE_AKS` - Set this to true if you wish to deploy a model to AKS, this is more suitable for production usage

- `AKS_PARAMS.COMPUTE_TARGET_NAME` - This will be the name of your deployed service and compute. If this already exists it will be updated

- `AKS_PARAMS.VM_TYPE` - Machine type of AKS cluster, default value is "Standard_NC6"



## Running Model Deployment

Below explains how to execute the model deployment process.

1. Update the deploy_config.json file to the desired experiment configuration

2. Activate your conda environment either in a cmd window or inside VS code.

3. Either press run in VScode window or navigate to the script folder and run the following command

    ``` bash
    (yourenvname) C:\Repo_Name\src\deployment\> python deploy_model.py
    ```

4. If you have not you will be prompted to log in. Follow the login details that open in your browser. If you have signed in recently this step will be skipped.

5. Wait for the deployment process to complete, this can take some time as the docker image needs to be built and registered, additionally if the compute does not exist this needs to be created. On completion you will be able to see the deployed services in the Azure ML Portal.