## Introduction

In general, this repo supports the training, deployment and evaluation of `mask-rcnn` based models to enable
`instance segmentation`, on both TF1 and TF2 versions. In particular, this touches the following models:


|  **TensorFlow 1 Supported Models**                       |
| -------------------------------------------------------- |
| mask_rcnn_inception_resnet_v2_atrous_coco                |
| mask_rcnn_inception_v2_coco                              |
| mask_rcnn_resnet101_atrous_coco                          |
| mask_rcnn_resnet50_atrous_coco                           |


|  **TensorFlow 1 Supported Models**                       |
| -------------------------------------------------------- |
| mask R-CNN Inception ResNet V2 1024x1024                 |


## Training
Analog to the other models, you need to download the mask_rcnn based models from tf model zoo and save them in the 
datastore registered. Some known issues so far regarding the model pipeline config files:

- **TF1**: You need to specify the `mask_type` as PNG_MASKS and set `load_instance_masks` to True in both train_input and 
evaluation_input readers. This is not included from the raw configs:
```
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
  }
	load_instance_masks: true
	mask_type: PNG_MASKS
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
  }
	load_instance_masks: true
	mask_type: PNG_MASKS
}
```

- **TF2**: In addition to the settings above, you need to specify the `fine_tune_checkpoint_type` to detection, and 
set `fine_tune_checkpoint_version` to V2 in the train_config snippet:
```
train_config: {
  batch_size: 16
  num_steps: 200000
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 0.008
          total_steps: 200000
          warmup_learning_rate: 0.0
          warmup_steps: 5000
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED"
  fine_tune_checkpoint_type: "detection"
  fine_tune_checkpoint_version: V2
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}
```

Once mask_rcnn based model is submitted to training, the tf_record generator will look for the `segmentation` values in 
your training and test datasets to create tf_records. That is to say, you also need to patch the polygon coordinates 
`[x1, y1, x2, y2, ...]` of the instance as a string of list into the csv files:

| filename | xmin |  ymin | xmax | ymax | segmentation | class  |
| ------ | ------ |------ | -----| ------- |------- |-----------|
| xxx.jpg | 422 |  426 | 424 | 428 | "[[423, 426, 422, 427, 424, 428]]" | name |


## Evaluation 
To be updated


## Deployment & Scoring

Please note that to depoly a mask_rcnn based model, your `IMAGE_TYPE` name specified in the deployment config should contain
the substring of `mask`, for instance `{usecase_name}_mask` so that the mask values can be extracted in the scoring file:
```json
{
    "ENV_CONFIG_FILE": "dev_config.json",
    "EXPERIMENT" : "",
    "RUN_ID" : "",
    "TF_VERSION": 2,
    "USE_ACI" : true,
    "IMAGE_TYPE": "usecase_mask",
    "COMPUTE_TARGET_NAME" : "",
    "REG_MODEL" : false
}
```

In addition to class, label, confidence and bounding box, the mask_rcnn based models also return one or more `mask` of 
fixed sizes on a given input image. An unmold function is needed to resize the mask to its original size
and put it back to the correct location. More details of the inferencing and visualization can be found in the 
`notebooks/test_deployment.ipynb`.