[[_TOC_]]

# MLOps
---

## Introduction
---

This repo is built on the MLOps Solution Accelerator (link will be added) and follows the same repository structure and process for setting up the CI/CD process. Detailed documentation is avaliable in the MLOps accelerator repository.

The repository is built on Azure DevOps, if you use a different DevOps/MLOps platform you can still make use of the src code and training scripts for the training process in Azure Machine Learning.

This section provides a summary of the MLOps process and where to find the key components, configuration files and pipelines. Before setting up the full MLOps process it is recommended to setup and test the manual train and deploy process as described in the QuickStart section. This will ensure you have the required Azure Resources, Dataset and Base models.

## Structure
---

For MLOps there are 3 directories within the repo of relevance:
- **Azure Pipelines**
- **Configuration**
- **Operation**

### Azure Pipelines
Contains the templates and pipelines for Azure DevOps. The pipelines require the creation of a service principal to allow Azure DevOps to provision resources, as well as a container registry principle once resources are deployed to allow registration of docker images.

There are 3 pipelines for the overall solution:
- **`PIPELINE-setup.yml`** - infrastructure pipeline that deploys the resource group and required resources within Azure as well as provisioning compute.
- **`PIPELINE-build.yml`** - CI pipeline setup to trigger on pull requests and check for code quality, execute unit tests and update the docker image in Azure ML.
- **`PIPELINE-modelling.yml`** - Scheduled model retraining and deployment pipeline including manual approval gate for release.

### Configuration
Configuration contains config and variable files used for different areas of the solution.

There are 4 key groups of variables:
- **compute** - defines the compute spec for the MLOps pipelines
- **environments** - defines the environments and packages for the modelling pipeline
- **`.variable.yml` files** - defines the service principles and model paramters for the pipelines
- **`_config.json`** - local config files containing workspace connection paramters for executing notebooks and local scripts - !Note these do not get committed


### Operation
Contains the python code for deploying and running the Azure ML pipelines and deployment of models to endpoints as part of the MLOps workflow.

Most of the scripts in this section are fixed and do not need modification as you develop and tune your models. The only file likely to need changing is `build_training_pipeline.py`, which sets the script arguments for retraining. Based on the model parameters explored this will need to be updated to match your best model.
