
[[_TOC_]]

# Evaluation Process
---

## Introduction
---

Evaluation is incorporated as part of the training jobs, all metrics calculated are logged to Azure Machine Learning. Figure below summarizes the flow of evaluation.

![image-867a4779-f73e-4f62-8acf-6b9707c237a7.png](/docs/.attachments/image-867a4779-f73e-4f62-8acf-6b9707c237a7.png)

## How Performance is Measured
---
Section below summerizes the calculation for provided metrics.

### Position vs. Binary Evaluations
- **Position** – Metrics calculated considering the exact location of a defect
- **Binary** – Metrics calculated summarizing the overall quality regardless of position

In the Binary case we measure performance on a Image Level and on Class Level.

### Confusion Matrix
Metrics calculated is based on the confusion matrix:

![image-2bb949ae-d4be-4439-9539-44ceb0da5eeb.png](/docs/.attachments/image-2bb949ae-d4be-4439-9539-44ceb0da5eeb.png)

Example below:
- **True Positive (TP)** – Case we predicted that the product has a defect, and the product has a defect
- **False Positive (FP)** – Case we predict that the product has a defect, but the product do not has any defects
- **False Negative (FN)** – Case we predict that the product does not have any defects, but the product has a defects
- **True Negative (TN)** – Case we predict that the product does not have any defects, and the product does

### Metrics
Based on the values from the confusion matrix the following metrics is calculated.
- **Precision** - How many of the predictions we are doing are right. Positive predicted value.
$Precision = \frac{TP}{TP+FP}$
- **Recall** - How many of the defect do we find. True positive rate.
$Recall = \frac{TP}{TP+FN}$
- **Accuracy** - Closeness of the predictions true value.
$Accuracy = \frac{TP+TN}{TP+FP+FN+TN}$
- **Average Precision (AP)** - The average precision value for recall value over 0 to 1.

#### Average Precision and Mean Average Precision for Object Detection

These metrics are position based metrics and are aligned to the computer vision industry standards.

The below figure is the example of caluclating the Average Precision.

![AP_calculation.jpeg](/docs/.attachments/AP_calculation.jpeg)

Further information about how the AP and mAP is calculated can be ontained in the following article - [link](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173).

##### IoU - Defining When Position is Close Enough
Intersection of Union (IoU) is the metrics used evaluating if a prediction is close enough. If IoU is above a specific threshold the prediction is defined as TP otherwise FP. The IoU is used evaluating a models predictions on a positional level. This is used as a based in the mAP calculations.

![image.png](/docs/.attachments/image-7f99f0d7-9ec5-44c4-ab8b-75432aa23b5a.png)