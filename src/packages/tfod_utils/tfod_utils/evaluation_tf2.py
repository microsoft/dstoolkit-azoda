''' Eval Utils
Eval module to bundle local and AML run evaluation steps
'''

import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import per_image_evaluation
from object_detection.utils import metrics
from object_detection.utils import ops as utils_ops


class Scoring():
    def __init__(self,
                 path_model,
                 path_label_map,
                 path_test_csv,
                 img_dir,
                 conf=0.5):
        """
        Base Evaluation Class
        Takes a model and test set and scores the model.
        Generates the binary and class classification metrics for the model
        :param path_model: path to model file, output from training run
        :param path_label_map: path to mapping file of int encoding to
        class name, output from training run
        :param path_test_csv: csv test file containing image names
        and ground truth labels. Based on version file format
        :param img_dir: directory to test set images
        :param conf: Optional confidence threshold for what is considered
        a true detection, defaults 0.5
        """
        self.path_model = path_model
        self.path_label_map = path_label_map
        self.path_test_csv = path_test_csv
        self.img_dir = img_dir
        self.num_classes = 100
        self.conf_threshold = float(conf)
        self.df_gts = pd.read_csv(self.path_test_csv)

        self.detection_fn = self.load_model()
        self.category_index = self.load_category_index()
        self.id_to_class, self.class_to_id = self.format_category_index()
        self.dets, self.gts = self.make_predictions()

    def load_model(self):
        """
        Load model
        Loads the model file into a tensorflow session to be used for inference
        :return: tf detection function
        """
        detection_fn = tf.saved_model.load(self.path_model)
        return detection_fn

    def load_category_index(self):
        """
        Load catergory index
        Creates the catergory index mapping from the label map file.
        :return: Category index
        """
        label_map = label_map_util.load_labelmap(self.path_label_map)

        categories = (label_map_util
                      .convert_label_map_to_categories
                      (label_map,
                       max_num_classes=self.num_classes,
                       use_display_name=True))

        category_index = (label_map_util
                          .create_category_index(categories))
        return category_index

    def format_category_index(self):
        """
        """
        id_cat = {}
        cat_id = {}
        for _, cat_dict in self.category_index.items():
            id_cat[cat_dict['id']] = cat_dict['name']
            cat_id[cat_dict['name']] = cat_dict['id']
        return id_cat, cat_id

    def make_predictions(self):
        """
        Make Predictions
        Loads and scores the model against the test set
        :return: a predictions pandas dataframe
        """
        all_images = self.df_gts.filename.unique().tolist()
        per_img_dets = {}
        per_img_gts = {}

        print("\n========== SCORING IMAGES STARTING ==========\n")

        for image_i in all_images:

            image_path_i = os.path.join(self.img_dir, image_i)
            image = Image.open(image_path_i).convert('RGB')
            width, height = image.size
            print('INFO: Processing image - {}'.format(image_i))

            # Convert image to numpy array
            image_np = np.array(image)
            # Converting to input tensors
            input_tensor = tf.convert_to_tensor(image_np)
            # adding new axis for batches of images
            input_tensor = input_tensor[tf.newaxis, ...]

            # Perform the detections by running the model
            detections = self.detection_fn(input_tensor)

            # Convert to numpy arrays, and take index [0] to remove the batch
            # dimension
            num_detections = int(detections.pop('num_detections'))

            need_detection_key = ['detection_classes', 'detection_boxes', 'detection_masks', 'detection_scores']
            detections = {key: detections[key][0, :num_detections].numpy()
                           for key in need_detection_key}

            detections['num_detections'] = num_detections

            # Converting detection classes to int
            detections['detection_classes'] = (detections['detection_classes']
                                               .astype(np.int64))

            # # Handle models with masks:
            # if 'detection_masks' in detections:
            #     # Reframe the the bbox mask to the image size.
            #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            #         tf.convert_to_tensor(detections['detection_masks']), detections['detection_boxes'],
            #         image_np.shape[0], image_np.shape[1])
            #     detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
            #                                        tf.uint8)
            #     detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

            # Format detections to TFODAPI
            per_img_dets[image_i] = {
                'boxes': detections['detection_boxes'],
                'scores': detections['detection_scores'],
                'labels': detections['detection_classes'],
                # 'masks': detections['detection_masks_reframed']
            }

            # Format gts to TFODAPI
            gt_boxes, gt_labels = self.format_gts(image_i, width, height)

            per_img_gts[image_i] = {
                'gt_boxes': gt_boxes,
                'gt_labels': gt_labels
            }

        print("\n========== SCORING IMAGES FINISHED ==========\n")

        return per_img_dets, per_img_gts

    def format_gts(self, img_id, width, height):
        """
        """
        df = self.df_gts[self.df_gts.filename == img_id]
        gt_boxes = []
        gt_labels = []
        # gt_masks = []
        for _, row_val in df.iterrows():
            xmin = float(row_val.xmin/width)
            ymin = float(row_val.ymin/height)
            xmax = float(row_val.xmax/width)
            ymax = float(row_val.ymax/height)
            boxes_i = [xmin, ymin, xmax, ymax]
            gt_boxes.append(boxes_i)
            gt_labels.append(self.class_to_id[row_val['class']])
        return gt_boxes, gt_labels


class ImageMetrics():
    def __init__(self, dets):
        self.dets = dets
        self.conf_thres = self.dets.conf_threshold
        self.category_index = self.dets.category_index
        self.per_img_bin_metrics = self.compute_per_img_binary_metrics()
        self.bin_img_metrics = self.calculate_binary_metrics()
        self.confusion_matrix = self.format_confusion_matrix()

    def compute_per_img_binary_metrics(self):
        per_img_bin_metrics = {}
        print('\n========== CALCULATING BINARY IMAGE METRICS ==========\n')
        for img_i in self.dets.gts:
            dets_i = self.dets.dets[img_i]
            num_dets = self.get_num_dets(dets_i)
            gts_i = self.dets.gts[img_i]
            if num_dets == 0 and not gts_i['gt_labels']:
                per_img_bin_metrics[img_i] = 'TN'
            if num_dets > 0 and gts_i['gt_labels']:
                per_img_bin_metrics[img_i] = 'TP'
            if num_dets > 0 and not gts_i['gt_labels']:
                per_img_bin_metrics[img_i] = 'FP'
            if num_dets == 0 and gts_i['gt_labels']:
                per_img_bin_metrics[img_i] = 'FN'
        return per_img_bin_metrics

    def get_num_dets(self, dets_i):
        bool_list = np.where(dets_i['scores'] >= self.conf_thres, True, False)
        num_dets = dets_i['labels'][bool_list].size
        return num_dets

    def calculate_binary_metrics(self):
        sum_TP = sum(map(('TP').__eq__, self.per_img_bin_metrics.values()))
        sum_FP = sum(map(('FP').__eq__, self.per_img_bin_metrics.values()))
        sum_TN = sum(map(('TN').__eq__, self.per_img_bin_metrics.values()))
        sum_FN = sum(map(('FN').__eq__, self.per_img_bin_metrics.values()))

        bin_prec = sum_TP/sum([sum_TP, sum_FP])
        bin_rec = sum_TP/sum([sum_TP, sum_FN])
        bin_acc = (sum_TP+sum_TN)/sum([sum_TP, sum_FP, sum_TN, sum_FN])

        bin_img_metrics = {
            'TP': sum_TP,
            'FP': sum_FP,
            'FN': sum_FN,
            'TN': sum_TN,
            'Precision': bin_prec,
            'Recall': bin_rec,
            'Accuracy': bin_acc
        }
        print('\n========== BINARY IMAGE METRICS CALCULATED ==========\n')
        return bin_img_metrics

    def format_confusion_matrix(self):
        """
        Log Confusion Matrix
        Takes the predictions from model evaluation class and
        forms a confusion matrix to log
        :param dt: binary results dictionary
        """
        class_labels = ['Defect', 'Clean']
        tp = int(self.bin_img_metrics['TP'])
        fp = int(self.bin_img_metrics['FP'])
        fn = int(self.bin_img_metrics['FN'])
        tn = int(self.bin_img_metrics['TN'])

        con_mat = [[tp, fn],
                   [fp, tn]]

        con_mat_json = {
            "schema_type": "confusion_matrix",
            "schema_version": "v1",
            "data": {
                "class_labels": class_labels,
                "matrix": con_mat
            }
        }

        return con_mat_json

    def log_AML(self, train_run):
        train_run.run.log_table(
            'Binary - Metric Table',
            self.bin_img_metrics
        )
        train_run.run.log_confusion_matrix(
            'Binary - Confusion Matrix',
            self.confusion_matrix
        )


class DetectionMetrics():
    def __init__(self, dets, iou_thres=0.5):
        self.dets = dets
        self.per_image_detections = self.dets.dets
        self.per_image_gts = self.dets.gts
        self.matching_iou_threshold = iou_thres
        self.category_index = self.dets.category_index
        self.num_gt_classes = len(self.category_index)

        self.per_cat_metrics = self.compute_precision_recall_bbox()
        self.format_cat_metrics = self.format_per_cat_metrics()
        self.mAP = self.calculate_mAP()

    def compute_precision_recall_bbox(self):
        """
        Compute the precision and recall at each confidence level for detection
        results of various classes.
        Args:
            per_image_detections: dict, image_id (str) => dict with fields
                'boxes': array-like, shape [N, 4], type float, each row is
                    [ymin, xmin, ymax, xmax] in normalized coordinates
                'scores': array-like, shape [N], float
                'labels': array-like, shape [N], integers in
                [1, num_gt_classes]
            per_image_gts: dic, image_id (str) => dict with fields
                'gt_boxes': array-like, shape [M, 4], type float, each row is
                    [ymin, xmin, ymax, xmax] in normalized coordinates
                'gt_labels': array-like, shape [M], integers in
                [1, num_gt_classes] num_gt_classes: int, number of classes in
                the ground truth labels matching_iou_threshold: float, IoU
                above which a detected and a ground truth box are considered
                overlapping
        Returns: dict, per-class metrics, keys are integers in
        [1, num_gt_classes]and 'one_class' which considers all classes.
        Each value is a dict with fields
        ['precision', 'recall', 'average_precision', ...]
        """
        # REVIEW: NMS iou threshold that is applied
        per_image_eval = per_image_evaluation.PerImageEvaluation(
            num_groundtruth_classes=self.num_gt_classes,
            matching_iou_threshold=self.matching_iou_threshold,
            nms_iou_threshold=1.0,
            nms_max_output_boxes=10000)

        # keys are categories (int)
        detection_tp_fp = defaultdict(list)  # in each list, 1 is tp, 0 is fp
        detection_scores = defaultdict(list)
        num_total_gt = defaultdict(int)

        print('\n========== CALCULATING DETECTION METRICS ==========\n')
        print('INFO: Running per-object analysis...', flush=True)
        for image_id, dets in self.per_image_detections.items():
            # we force *_boxes to have shape [N, 4], even in case that N = 0
            detected_boxes = np.asarray(
                dets['boxes'], dtype=np.float32).reshape(-1, 4)
            detected_scores = np.asarray(dets['scores'])
            # labels input to compute_object_detection_metrics() needs to
            # start at 0, not 1 start at 0
            detected_labels = np.asarray(dets['labels'], dtype=np.int) - 1
            # num_detections = len(dets['boxes'])

            gts = self.per_image_gts[image_id]
            gt_boxes = (np.asarray(gts['gt_boxes'], dtype=np.float32)
                        .reshape(-1, 4))
            # start at 0
            gt_labels = np.asarray(gts['gt_labels'], dtype=np.int) - 1
            num_gts = len(gts['gt_boxes'])

            # place holders - we don't have these
            groundtruth_is_difficult_list = np.zeros(num_gts, dtype=bool)
            groundtruth_is_group_of_list = np.zeros(num_gts, dtype=bool)

            results = per_image_eval.compute_object_detection_metrics(
                detected_boxes=detected_boxes,
                detected_scores=detected_scores,
                detected_class_labels=detected_labels,
                groundtruth_boxes=gt_boxes,
                groundtruth_class_labels=gt_labels,
                groundtruth_is_difficult_list=groundtruth_is_difficult_list,
                groundtruth_is_group_of_list=groundtruth_is_group_of_list)
            scores, tp_fp_labels, is_class_correctly_detected_in_image = results  # noqa

            for i, tp_fp_labels_cat in enumerate(tp_fp_labels):
                # true positives < gt of that category
                assert sum(tp_fp_labels_cat) <= sum(gt_labels == i)

                cat = i + 1  # categories start at 1
                detection_tp_fp[cat].append(tp_fp_labels_cat)
                detection_scores[cat].append(scores[i])
                # gt_labels start at 0
                num_total_gt[cat] += sum(gt_labels == i)

        all_scores = []
        all_tp_fp = []

        print('INFO: Computing precision recall for each category...')
        per_cat_metrics = {}
        for i in range(self.num_gt_classes):
            cat = i + 1
            scores_cat = np.concatenate(detection_scores[cat])
            tp_fp_cat = np.concatenate(detection_tp_fp[cat]).astype(np.bool)
            all_scores.append(scores_cat)
            all_tp_fp.append(tp_fp_cat)

            precision, recall = metrics.compute_precision_recall(
                scores_cat, tp_fp_cat, num_total_gt[cat])
            average_precision = metrics.compute_average_precision(precision,
                                                                  recall)

            cat_name = self.dets.id_to_class[cat]

            per_cat_metrics[cat_name] = {
                'category': cat_name,
                'precision': precision,
                'recall': recall,
                'average_precision': average_precision,
                'scores': scores_cat,
                'tp_fp': tp_fp_cat,
                'num_gt': num_total_gt[cat]
            }
            print(f'INFO: Number of ground truth in category \
                 {cat_name}: {num_total_gt[cat]}')

        # compute one-class precision/recall/average precision (if every box
        # is just of an object class)
        all_scores = np.concatenate(all_scores)
        all_tp_fp = np.concatenate(all_tp_fp)
        overall_gt_count = sum(num_total_gt.values())

        class_avg_prec, class_avg_recall = metrics.compute_precision_recall(
            all_scores, all_tp_fp, overall_gt_count)
        class_average_precision = metrics.compute_average_precision(
            class_avg_prec, class_avg_recall)

        per_cat_metrics['class_avg'] = {
            'category': 'class_avg',
            'precision': class_avg_prec,
            'recall': class_avg_recall,
            'average_precision': class_average_precision,
            'scores': all_scores,
            'tp_fp': all_tp_fp,
            'num_gt': overall_gt_count
        }
        print('\n========== DETECTION METRICS CALCULATED ==========\n')

        return per_cat_metrics

    def format_per_cat_metrics(self):
        """
        """
        category = []
        average_precision = []
        num_gts = []
        for _, val_i in self.per_cat_metrics.items():
            category.append(val_i['category'])
            average_precision.append(val_i['average_precision'])
            num_gts.append(val_i['num_gt'])

        format_cat_met = {
            'category': category,
            'average_precision': average_precision,
            'number_of_gts': num_gts
        }

        return format_cat_met

    def calculate_mAP(self):
        """
        Mean average precision, the mean of the average precision for each category
        Args:
            per_cat_metrics: dict, result of compute_precision_recall()
        Returns: float, mAP for this set of detection results
        """
        # minus the 'one_class' set of metrics
        self.num_gt_classes = len(self.per_cat_metrics) - 1

        mAP_from_cats = sum(
            v['average_precision']
            if k != 'class_avg' and not math.isnan(v['average_precision']) else 0
            for k, v in self.per_cat_metrics.items()
        ) / self.num_gt_classes
        print(f'INFO: mAP: {mAP_from_cats}')
        return mAP_from_cats

    def log_AML(self, train_run):
        train_run.run.log_table('Detection - Metric Table',
                                self.format_cat_metrics)
        train_run.run.log('Detection - mAP', self.mAP)
