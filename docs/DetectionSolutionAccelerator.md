# Object Detection Solution Accelerator 
--- 

Object detection solution accelerator provides a pre-packaged solution to train. deploy and monitor custom object detection models using the tensorflow object detection framework within AzureML. The aim is to bring SOTA object detection models quickly into production scenarios particualrly around the use of defect detection as seen in many quality control scenarios. 

This wiki contains the technical documentation for the solution accelerator. The sections cover the following: 

Architecure - high level solution architecture

Quickstart - setting up local environment and setup of the azure resources

Labelling - tools and adapter scripts for the solution dataset structure

Model Training - AML model training workflow

Model Support and Data Augmentation - supported models and deeper details of the tensorflow solution

Evaluation - built in model evaluation process and results plotting

Model Deployment Cloud - process to deploy models ot ACI and AKS endpoints with AML

Model Deployment Edge - process to deploy models to edge device

Model Pipelines - MLOps workflows for model retraining, automated deployment and setting up release gates for endpoints.