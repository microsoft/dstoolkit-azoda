[[_TOC_]]

# Labeling
---

## Introduction
---
This section describes the process and tooling for annotating images. It is important to consider that these are used for both model training and evaluation and bad labels can cause a good model to look poor.

If there are different classes that visually looks similar or have the same features it is sometimes diffucult for the AI model to learn. Consider for your use case if these classes can be grouped to a parent class (e.g. metal scratch and glass scratch with similar features could be aggretaged to scratch).

## Labeling Tools
---
### Image Tagging

Object detection requires image annotations that provide both the image coordinates for the region of interest (bounding box), as well as the class or label for that object. There are a number of closed and open source tools that provide this support.

This repository uses an input dataset in .csv format to store annotations. This allows for any annotation tool to be used and then converted to this simple format.

### Recommended Tools

Microsoft offers two options for labelling tools, VOTT an open source visual object tagging tool as well as a labelling tool provided as part of Azure Machine Learning. Below provides the links to those tools and their documentations:

- [VOTT GitHub](https://github.com/Microsoft/VoTT)
- [VOTT Web Tool](https://vott.z22.web.core.windows.net/#/)
- [Azure ML Labelling Feature](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-labeling-projects)

### Datasets and Adapter Scripts

Adapter scripts are used to adapt different label tool outputs/formats to the supported .csv format, used to define datasets and the train/eval split.

An example adapter script can be found in src/data_orchestration/testdata_format.py for processing outputs from labelling tools using .xml file format.

## Labeling Guidance
---

It is important to create quality labels in order to have a performant model. Many use cases fail by focusing on the model tuning without considering the data quality. When setting up a labelling process we recommend to consider the follow points:

- Define a set of instructions for those labelling, including clear definition of each class/label
- Take care to ensure bounding boxes contain all features of the object but do not include unessary background information
- Where possible, using multiple labellers, set up a review process or use a consensus approach to labels (consensus is important when using crowdsourcing)

Below provides a flow for generating quality data:

![data_quality_process2.jpg](/docs/.attachments/data_quality_process2.jpg)

### Consensus Approach

Below illustrates a simple multi-labeller team consensus approach. The images/boxes that are removed or do not reach consensus are important to review as these often identify the confusing cases. These will be the cases a model typically will struggle with.

![image.png](/docs/.attachments/consensus_analysis_flow.png)

### EXIF Meta Data

EXIF is a meta data format associated with an image, it captures important information regarding how the image was taken and additional parameters that may be important to your use case. In the case of mobile images it can contain rotation information due to the devices accelerometer. This means that the raw binary may be stored in a different orientation to the capture orientation. Different browsers will treat this differently and when using different labellers can lead to mismatching results. Ensure you check the orientation of the image at the time of labelling and consider tracking this information in your processing pipelines.
