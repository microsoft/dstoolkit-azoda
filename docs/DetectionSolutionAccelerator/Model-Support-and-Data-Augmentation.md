[[_TOC_]]

# Model Framework
---

## Introduction
---

This repo is built using the TensorFlow Object Detection API. This is a framework to train object detection model using TensorFlow. The base code can be found [here](https://github.com/tensorflow/models/tree/master/research/object_detection). This repo integrates this into an Azure machine learning workflow that provides quick cloud based model training, tracking and deployment.

The repo supports both TensorFlow 1 & 2. Recommendation is to make use of TensorFlow 2 which benefits from the latest model architectures whereas TensorFlow 1 has limited support as outlined below.

TensorFlow versions are set in two key places. The first is the base docker image. The second is at deployment in the `environment.yml` file. Please note both versions need to be the same to ensure stability. The `tfod_utils` package in this repo contains code that is specific to either TensorFlow 1 or 2, you will see `_tf2` for the TensorFlow 2 support.


## TensorFlow 1 Supported Models
---
Below is a list of supported base models. These are grouped into 2 supported architectures, SSD and Faster RCNN. To use these you will need to download them from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). You will then need to extract them from there .tar format and store them in the base model folder in the azure storage.

**Note!** - Some supported models have issues in the config files that they are provided with. If an error occurs check the GitHub page for the required fix as these can vary.

### Model List
| **Model**                                                |
| -------------------------------------------------------- |
| ssd_mobilenet_v1_coco                                    |
| ssd_mobilenet_v1_0.75_depth_coco                         |
| ssd_mobilenet_v1_quantized_coco                          |
| ssd_mobilenet_v1_0.75_depth_quantized_coco               |
| ssd_mobilenet_v1_ppn_coco                                |
| ssd_mobilenet_v1_fpn_coco                                |
| ssd_resnet_50_fpn_coco                                   |
| ssd_mobilenet_v2_coco                                    |
| ssd_mobilenet_v2_quantized_coco                          |
| ssdlite_mobilenet_v2_coco                                |
| ssd_inception_v2_coco                                    |
| faster_rcnn_inception_v2_coco                            |
| faster_rcnn_resnet50_coco                                |
| faster_rcnn_resnet50_lowproposals_coco                   |
| rfcn_resnet101_coco                                      |
| faster_rcnn_resnet101_coco                               |
| faster_rcnn_resnet101_lowproposals_coco                  |
| faster_rcnn_inception_resnet_v2_atrous_cocoxes           |
| faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco |
| faster_rcnn_nas                                          |
| faster_rcnn_nas_lowproposals_coco                        |

When running model training the support between the models types is slightly different in the solutions current state.

### Support Level
Faster RCNN based models have the deepest integration. They provide in depth logging of both total loss but also each network stages loss. SSD has support for logging but only logs the total loss. Both have the same level of support for evaluation logging.

Evaluation logging is tied to the base dataset used for training. This repo has full support for the most common base image set of COCO. The code is setup to allow the addition of other base datasets but this will need to be handled by updating the docker to make use of the latest TensorFlow Object Detection API.

## TensorFlow 2 Supported Models
---

For TensorFlow 2 this repo supports all object detection models using the boxes input/output format. For keypoint and mask based models updates need to be made to the TFRecords and Evaluation sections of the TFOD_utils to include the support. The model zoo for base models can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

## Model Tuning Support
---
Model tuning is dependent on the combination of model architecture (SSD/FasterRCNN/CenterNet etc) and base model architecture (Inception, resnet, mobilenet etc). To enable hypertuning you need to have a variation of train.py that has setup both the argparsing options and the hparams updates setup for your model type,. In addition you will also need to add support for your variations in the submit_training.py.

Model hypertuning examples are provided in both the notebooks section and the training section of src.

## Data Augmentation
---
As part of model training there is also the concept of data augmentation. This is the process of applying random changes to the input images to assist in building a more robust model. The TensorFlow object detection framework supports data augmentation on the fly through the pipeline.config.

The current Object Detection Solution Accelerator solution does not support per run variations to the data augmentation options. You are still able to explore and use data augmentation through modification to the base pipeline.config by manually updating them inside the base model directory. The following [link](https://stackoverflow.com/questions/44906317/what-are-possible-values-for-data-augmentation-options-in-the-tensorflow-object) explains the format on updating the config, and this [link](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) explains the current options available for data augmentation. Note some augmentation options are only supported in certain TF1 versions or TF2.

**Note!** - Data augmentation support can vary depending on model. Check each augmentation before applying to the config.

**Note!** - Once you update a pipeline.config this augmentation will apply to all future model training with that base model including the models used in retraining. Its recommended to either make copies of the models with augmentation based configs or make sure to remove the options if augmentation has a negative effect of model performance to the main model config.