![banner](/docs/.attachments/banner.jpg)

# Object Detection Solution Accelerator

This repository contains all the code for training TensorFlow object detection models within Azure Machine Learning (AML) with setups for training on Azure compute, experiment monitoring and endpoint deployment as a webservice. It is built on the MLOps Accelerator and provides end to end training and deployment pipelines allowing quick and easy setup of CI/CD pipelines for your projects.

For detailed documentation please refer to the docs section of the repo containing the solution wiki.

## Prerequisites

In order to successfully setup your solution you will need to have access to and or provisioned the following:

- Access to an Azure subscription
- Access to an Azure DevOps subscription
- Annotated Images

## Getting Started

In order to get started with setup and running your first model training experiment and deployment please refer to the QuickStart page within the docs section of this repo. This will provide the quickest path to environment setup and experiment running.

[QuickStart](/docs/DetectionSolutionAccelerator/Quickstart.md)

For the full MLOps setup please refer to the documentation within the MLOps page of the docs section of this repo. As this is built on the MLOps accelerator further documentation can be found in that repo.


## Repository Structure

This section breaks down the structure of the repo and how the code is organised into an end to end workflow.

```
├───azure-pipelines
│   ├───arm-templates
│   └───pipelines
├───configuration
├───docker
│   ├───tf_1
│   └───tf_2
├───docs
├───notebooks
├───operation
└───src
    ├───data_orchestration
    ├───deployment
    │   ├───endpoint_check
    │   ├───test
    │   └───utils
    ├───packages
    │   ├───azure_utils
    │   └───tfod_utils
    ├───plotting
    └───training
        └───scripts
```

- azure-pipelines - Azure resource templates (ARM) for the project as well as the devops pipelines
- configuration - Environment configuration files
- docker - Base docker images to run the solution
- docs - contains solution documentation wiki in .md format. Designed to be imported as an Azure DevOps wiki
- notebooks - example notebooks on model training and deployment
- operation - python code for MLOps retraining pipelines
- src - contains the python solution source code and packages, including experiment submission scripts

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.