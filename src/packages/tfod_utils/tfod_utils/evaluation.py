''' Eval Utils
Eval module to bundle local and AML run evaluation steps
'''

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from PIL import Image

from object_detection.utils import label_map_util


class Eval():
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
        self.num_classes = 20
        self.conf_threshold = float(conf)

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

    def load_model(self):
        """
        Load model
        Loads the model file into a tensorflow session to be used for inference
        :return: detection graph and tf session
        """
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
        return detection_graph, sess

    def get_coordinates(self, boxes):
        """
        Get box coordinates
        maps each box for an image to a list of xmin, ymin, xmax, ymax lists
        used for formatting the results for metric calculation
        :param boxes: detection boxes from tensrflow model
        :return: xmin, ymin, xmax, ymax
        """
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for box_i in boxes:
            xmin.append(box_i[0])
            ymin.append(box_i[1])
            xmax.append(box_i[2])
            ymax.append(box_i[3])
        return xmin, ymin, xmax, ymax

    def get_class_name(self, class_ids, category_index):
        """
        Get Class Name
        Takes the class mapping and list of ids and returns list of names
        :param class_ids: list of class ids
        :param category_index: category index
        :return: list of names
        """
        list_class_name = []
        for class_i in class_ids:
            list_class_name.append(category_index[class_i]['name'])
        return list_class_name

    def get_class_id(self, classes_names, category_index):
        """
        Get Class ID
        Takes a list of class names and the catergory index
        and returns list of class IDs
        :param class_names: list of class names
        :param category_index: category index
        :return: list of class ids
        """
        list_class_id = []
        for class_i in classes_names:
            for key_i in category_index.keys():
                class_name = category_index[key_i]['name']
                if class_name == class_i:
                    list_class_id.append(category_index[key_i]['id'])
        return list_class_id

    def get_gt_coordinates(self, width, height):
        """
        Get Ground truth coordinates
        Takes the image width and height and returns the ground truth
        coordinates as a numpy array
        :param width: image width
        :param height: image height
        :return: numpy array of ground truth coordinates
        """
        xmin_norm = (self.df_gt.xmin/width).tolist()
        xmax_norm = (self.df_gt.xmax/width).tolist()
        ymin_norm = (self.df_gt.ymax/height).tolist()
        ymax_norm = (self.df_gt.ymin/height).tolist()
        coor_mat = np.array([ymin_norm, xmin_norm, ymax_norm, xmax_norm])
        coor_mat_T = coor_mat.T
        return coor_mat_T

    def get_gt(self, width, height):
        """
        Get Ground truth
        Takes the image width and height and return gt coordinates scores
        and classes
        :param width: image width
        :param height: image height
        :return: gt_boxes, gt_scores, gt_classes
        """
        gt_classes = np.array(self.get_class_id(self.df_gt['class'],
                                                self.category_index))
        # gt_scores = np.array([None for i in range(0, len(df_gt.label))])
        gt_scores = None
        gt_boxes = self.get_gt_coordinates(self.df_gt, width, height)
        return gt_boxes, gt_scores, gt_classes

    def format_predictions(self,
                           image_i,
                           boxes,
                           scores,
                           classes):
        """
        Format Predictions
        Takes the predictions for an image from the model and formats them into
        a pandas dataframe
        :param image_i: current image name
        :param boxes: detected boxes
        :param scores: box probability scores
        :param classes: box classes
        :return: prediction pandas dataframe
        """
        if len(scores[scores >= self.conf_threshold]) > 0:
            boxed_filtered = boxes[scores >= self.conf_threshold]
            xmin, ymin, xmax, ymax = self.get_coordinates(boxed_filtered)
            scores_filtered = scores[scores >= self.conf_threshold]
            classes_filterd = classes[scores >= self.conf_threshold]
            class_name_filtered = self.get_class_name(classes_filterd,
                                                      self.category_index)
            filename_filtered = image_i

            df_pred = pd.DataFrame({
                'filename': filename_filtered,
                'xmin_pred': xmin,
                'ymin_pred': ymin,
                'xmax_pred': xmax,
                'ymax_pred': ymax,
                'class_pred': class_name_filtered,
                'score_pred': scores_filtered
            })
        else:
            df_pred = None

        return df_pred

    def make_predictions(self):
        """
        Make Predictions
        Loads and scores the model against the test set
        :return: a predictions pandas dataframe
        """
        df_pred = None

        # Define input and output tensors for the object detection classifier
        image_tensor = (self.detection_graph
                        .get_tensor_by_name('image_tensor:0'))
        detection_boxes = (self.detection_graph
                           .get_tensor_by_name('detection_boxes:0'))
        detection_scores = (self.detection_graph
                            .get_tensor_by_name('detection_scores:0'))
        detection_classes = (self.detection_graph
                             .get_tensor_by_name('detection_classes:0'))
        num_detections = (self.detection_graph
                          .get_tensor_by_name('num_detections:0'))

        list_test_images = self.df_gt.filename.unique().tolist()

        all_images = np.array(list_test_images)

        for image_i in all_images:

            image_path_i = os.path.join(self.img_dir, image_i)

            image = Image.open(image_path_i).convert('RGB')

            width, height = image.size
            image_expanded = np.expand_dims(image, axis=0)

            print('INFO: Processing image - {}'.format(image_i))

            # Perform the detections by running the model
            (boxes, scores, classes, num) = self.sess.run(
                [detection_boxes,
                 detection_scores,
                 detection_classes,
                 num_detections],
                feed_dict={image_tensor: image_expanded})

            if df_pred is None:
                df_pred = self.format_predictions(image_i,
                                                  boxes,
                                                  scores,
                                                  classes)
            else:
                df_i = self.format_predictions(image_i,
                                               boxes,
                                               scores,
                                               classes)

                if df_i is not None:
                    df_pred = pd.concat([df_pred, df_i],
                                        ignore_index=True)
        return df_pred

    def classify_bin_class_level(self, df):
        """
        Classify Binary Class Level
        Takes the grouped merged pandas dataframe of ground truth and labels
        and returns dataframe with binary class column containing TP,TN,FP,FN
        :param df: Grouped pandas dataframe containing groundtruth
        and predictions
        :return: input dataframe with binary class column
        """
        TP = (df.freq_gt > 0) & (df.freq_pred > 0)
        FP = (df.freq_gt == 0) & (df.freq_pred > 0)
        FN = (df.freq_gt > 0) & (df.freq_pred == 0)
        TN = (df.freq_gt == 0) & (df.freq_pred == 0)
        df['binary_classification_class'] = (
            np.where(TP,
                     "TP",
                     np.where(FP,
                              "FP",
                              np.where(FN,
                                       "FN",
                                       np.where(TN,
                                                "TN",
                                                np.nan)))))
        return df

    def binary_classification_image(self):
        """
        Binary Classification Image Level
        Takes the ground truth and predictions dataframes
        :return: a dictionary of binary image class metrics containing
        TP, FP, FN, TN, Precision, Recall and Accuracy
        """
        pos_gt = self.df_gt[self.df_gt['class'].notnull()]
        gt_image = pos_gt.filename.unique()
        neg_gt = self.df_gt[self.df_gt['class'].isnull()]
        gt_neg_image = neg_gt.filename.unique()
        pred_images = self.df_pred.filename.unique()

        TP = len(np.intersect1d(pred_images, gt_image))
        FP = len(np.setdiff1d(pred_images, gt_image))
        FN = len(np.setdiff1d(gt_image, pred_images))
        TN = len(np.setdiff1d(gt_neg_image, pred_images))

        with np.errstate(invalid='ignore'):
            prec = np.divide(TP, (TP + FP))
            rec = np.divide(TP, (TP + FN))
            acc = np.divide((TP + TN), (TP + FP + FN + TN))

        dict_img_metrics = {
            "TP": [TP],
            "FP": [FP],
            "FN": [FN],
            "TN": [TN],
            "Precision": [prec],
            "Recall": [rec],
            "Accuracy": [acc]
        }
        return dict_img_metrics

    def binary_classification_class(self):
        """
        Binary Classification Class
        Takes the ground truth df and the predictions dataframe formats and
        calls the class classification function
        :return: class metric dataframe
        """
        df_gt = self.df_gt[self.df_gt['class'].notnull()]

        df_gt_grouped = (df_gt
                         .groupby(by=['filename',
                                      'class'])
                         .count()
                         .reset_index()
                         [['filename',
                           'class',
                           'xmin']]
                         .rename(columns={
                             'class': 'class_name',
                             'xmin': 'freq_gt'
                         }))

        df_pred_grouped = (self.df_pred
                           .groupby(by=['filename',
                                        'class_pred'])
                           .count()
                           .reset_index()
                           [['filename',
                             'class_pred',
                             'xmin_pred']]
                           .rename(columns={
                               'class_pred': 'class_name',
                               'xmin_pred': 'freq_pred'
                           }))

        df_merge = (pd.merge(df_gt_grouped,
                             df_pred_grouped,
                             on=['filename', 'class_name'],
                             how='outer',
                             indicator='indicator_column')
                    .fillna({'freq_pred': 0, 'freq_gt': 0}))

        df_class_metric = (df_merge
                           .pipe(self.classify_bin_class_level)
                           .groupby(by=['class_name',
                                        'binary_classification_class'])
                           .count()
                           .reset_index()
                           [['class_name',
                             'binary_classification_class',
                             'filename']]
                           .rename(columns={'filename': 'freq'}))

        return df_class_metric

    def print_class_classification(self,
                                   df_class_metric,
                                   pos_img,
                                   neg_img):
        """
        Print class classification
        Takes the df class metric dataframe and formats the results for
        printing and logging
        :param df_class_metric: Class metric dataframe
        :param pos_img: count of positive images
        :param neg_img: count of negative images
        :return: a dictionary of metrics
        """
        classes = df_class_metric.class_name.unique()
        metrics = {'class': [],
                   'TP':  [],
                   'FP':  [],
                   'FN':  [],
                   'TN':  [],
                   'Precison':  [],
                   'Recall':  [],
                   'Accuarcy':  []}

        print('\n-- Class Classification Level --')
        for class_i in classes:
            print('\n- {} -'.format(class_i))
            df_print = (df_class_metric
                        .query("class_name=='{}'".format(class_i)))
            TP = (df_print[df_print.binary_classification_class == 'TP']
                  .freq)
            if TP.empty is True:
                TP = 0
            else:
                TP = TP.iat[0]
            FP = (df_print[df_print.binary_classification_class == 'FP']
                  .freq)
            if FP.empty is True:
                FP = 0
            else:
                FP = FP.iat[0]
            FN = (df_print[df_print.binary_classification_class == 'FN']
                  .freq)
            if FN.empty is True:
                FN = 0
            else:
                FN = FN.iat[0]
            TN = pos_img + neg_img - (TP + FP + FN)

            with np.errstate(invalid='ignore'):
                prec = np.divide(TP, (TP + FP))
                rec = np.divide(TP, (TP + FN))
                acc = np.divide((TP + TN), (TP + FP + FN + TN))

            print('TP: \t{:d}\n'.format(TP),
                  'FP: \t{:d}\n'.format(FP),
                  'FN: \t{:d}\n'.format(FN),
                  'TN: \t{:d}\n\n'.format(TN),
                  'Precicion: \t{:4.2f} %\n'.format(prec*100),
                  'Recall: \t{:4.2f} %\n'.format(rec*100),
                  'Accuracy \t{:4.2f} %\n'.format(acc*100))

            metrics['class'].append(class_i)
            metrics['TP'].append(TP)
            metrics['FP'].append(FP)
            metrics['FN'].append(FN)
            metrics['TN'].append(TN)
            metrics['Precison'].append(prec)
            metrics['Recall'].append(rec)
            metrics['Accuarcy'].append(acc)

        return metrics

    def count_images(self):
        """
        Count Images
        Gets the count of postive and negative images
        from the groudn truth dataframe
        :return: pos_count and neg_count
        """
        pos_gt = self.df_gt[self.df_gt['class'].notnull()]
        neg_gt = self.df_gt[self.df_gt['class'].isnull()]
        pos_count = len(pos_gt.filename.unique())
        neg_count = len(neg_gt.filename.unique())
        return pos_count, neg_count

    def eval_model(self):
        """
        Eval Model
        Takes the data and model and generates the summary performance metrics
        :return: img_metrics, cl_metrics, neg_count, pos_count
        """
        self.df_gt = pd.read_csv(self.path_test_csv)

        self.category_index = self.load_category_index()

        self.detection_graph, self.sess = self.load_model()

        self.df_pred = self.make_predictions()

        pos_count, neg_count = self.count_images()

        # check if no predictions
        if self.df_pred is None:
            print('INFO: Model Unable to make any predictions')
            print('INFO: Try training for more steps')
            cl_metrics = {'class': ['class'],
                          'TP': [0],
                          'FP': [0],
                          'TN': [neg_count],
                          'FN': [pos_count],
                          'Precision': [0],
                          'Recall': [0],
                          'Accuracy': [0]}

            img_metrics = {'TP': [0],
                           'FP': [0],
                           'TN': [neg_count],
                           'FN': [pos_count],
                           'Precision': [0],
                           'Recall': [0],
                           'Accuracy': [0]}
        else:
            df_class_metric = self.binary_classification_class()

            img_metrics = self.binary_classification_image()

            cl_metrics = self.print_class_classification(df_class_metric,
                                                         pos_count,
                                                         neg_count)

            self.print_image_classification(img_metrics)

        return img_metrics, cl_metrics, neg_count, pos_count

    def print_image_classification(self, img_metrics):
        """
        Print Image Classification
        Takes the image metrics dictionary and prints the summary
        :param img_metrics: image metrics dictionary
        """
        print('\n-- Image Classification Level --')

        prec = img_metrics['Precision'][0]*100

        print('TP: \t{:d}\n'.format(img_metrics['TP'][0]),
              'FP: \t{:d}\n'.format(img_metrics['FP'][0]),
              'FN: \t{:d}\n'.format(img_metrics['FN'][0]),
              'TN: \t{:d}\n\n'.format(img_metrics['TN'][0]),
              'Precicion: \t{:4.2f} %\n'.format(prec),
              'Recall: \t{:4.2f} %\n'.format(img_metrics['Recall'][0]*100),
              'Accuracy \t{:4.2f} %\n'.format(img_metrics['Accuracy'][0]*100))


class TFEval(Eval):
    def __init__(self,
                 path_model,
                 path_label_map,
                 path_test_csv,
                 img_dir,
                 train_run,
                 conf=0.5):
        """
        Tensorflow Evaluation Class
        Inherits the base eval class and adds functionality to log within AML
        runs
        :param path_model: path to model file, output from training run
        :param path_label_map: path to mapping file of int encoding to
        class name, output from training run
        :param path_test_csv: csv test file containing image names
        and ground truth labels. Based on version file format
        :param img_dir: directory to test set images
        :param train_run: the AML run object to log against
        """

        super().__init__(path_model,
                         path_label_map,
                         path_test_csv,
                         img_dir,
                         conf)

        self.train_run = train_run

    def run_evaluation(self):
        """
        Run Evaluation
        Execute tensorflow model evalaution and log results to the run
        :return: binary dictionary
        """
        # TODO need to create tf2 path
        self.path_model = os.path.join(self.path_model,
                                       'frozen_inference_graph.pb')

        binary_dict,\
            class_dict,\
            num_neg,\
            num_pos = self.eval_model()

        run = self.train_run.run
        self.log_con_mat(binary_dict)
        run.log_table('Class - Classification Table', class_dict)
        run.log_table('Image - Classification Table', binary_dict)
        run.log('Eval - Binary TP', binary_dict.get('TP')[0])
        run.log('Eval - Binary TN', binary_dict.get('TN')[0])
        run.log('Eval - Binary FP', binary_dict.get('FP')[0])
        run.log('Eval - Binary FN', binary_dict.get('FN')[0])
        run.log('Eval - Binary Precision', binary_dict.get('Precision')[0])
        run.log('Eval - Binary Recall', binary_dict.get('Recall')[0])
        run.log('Eval - Binary Accuracy', binary_dict.get('Accuracy')[0])
        run.tag('Eval - Num Defective Eval Images', num_pos)
        run.tag('Eval - Num Positive Eval Images', num_neg)
        return binary_dict

    def eval_and_log(self):
        """
        Evaluate and log
        Checks for dataset variation and runs default aml eval
        :return: the final mAP
        """
        if 'coco' in self.train_run.base_model:
            final_map = self.eval_and_log_coco()
        elif 'kitti' in self.train_run.base_model:
            final_map = self.eval_and_log_kitti()
        else:
            final_map = self.eval_and_log_basic()
        return final_map

    def eval_and_log_basic(self):
        """
        Evaluation and Logging basic
        Runs the evaluation for model not specifically supported.
        Prints general summary instead of logging to AML
        """
        input_fn = self.eval_input_fns[0]

        metric = (self.train_run
                  .estimator
                  .evaluate(input_fn,
                            steps=None,
                            checkpoint_path=tf.train.latest_checkpoint(
                                self.train_run.checkpoint_dir)))

        print(metric)
        return "Eval Unsupported"

    def eval_and_log_coco(self):
        """
        Evaluation and Logging COCO Base Model
        Runs the evaluation for base models trained on COCO dataset
        logs the COCO metrics to the AML run
        :return: final_map
        """

        input_fn = self.train_run.eval_input_fns[0]

        metric = (self.train_run
                  .estimator
                  .evaluate(input_fn,
                            steps=None,
                            checkpoint_path=(tf
                                             .train
                                             .latest_checkpoint(
                                                 self.train_run
                                                 .checkpoint_dir))))

        final_map = metric.get('DetectionBoxes_Precision/mAP@.50IOU')

        run = self.train_run.run

        # Log standard fixed thresholds mAP
        run.log('Final mAP',
                final_map)
        run.log('Eval - IoU=0.5 mAP',
                final_map)
        run.log('Eval - IoU=0.75 mAP',
                metric.get('DetectionBoxes_Precision/mAP@.75IOU'))

        # Log mAP for average IoU range and box size
        run.log('Eval - IoU=[0.5:0.95] mAP',
                metric.get('DetectionBoxes_Precision/mAP'))
        run.log('Eval - IoU=[0.5:0.95] mAP (Small)',
                metric.get('DetectionBoxes_Precision/mAP (small)'))
        run.log('Eval - IoU=[0.5:0.95] mAP (Medium)',
                metric.get('DetectionBoxes_Precision/mAP (Medium)'))
        run.log('Eval - IoU=[0.5:0.95] mAP (Large)',
                metric.get('DetectionBoxes_Precision/mAP (Large)'))

        # Log Recall for average of IoU range and box size
        run.log('Eval - IoU=[0.5:0.95] Average Recall',
                metric.get('DetectionBoxes_Recall/AR@100'))
        run.log('Eval - IoU=[0.5:0.95] Average Recall (Small)',
                metric.get('DetectionBoxes_Recall/AR@100 (small)'))
        run.log('Eval - IoU=[0.5:0.95] Average Recall (Medium)',
                metric.get('DetectionBoxes_Recall/AR@100 (medium)'))
        run.log('Eval - IoU=[0.5:0.95] Average Recall (Large)',
                metric.get('DetectionBoxes_Recall/AR@100 (large)'))

        # Log Eval Losses
        run.log('Eval - Total Training Loss',
                metric.get('Loss/total_loss'))
        run.log('Eval - Box Classifier Classification Loss',
                metric.get('Loss/BoxClassifierLoss/classification_loss'))
        run.log('Eval - Box Classifier Localization Loss',
                metric.get('Loss/BoxClassifierLoss/localization_loss'))
        run.log('Eval - RPN Localization Loss',
                metric.get('Loss/RPNLoss/localization_loss'))
        run.log('Eval - RPN Objectness Loss',
                metric.get('Loss/RPNLoss/objectness_loss'))

        return final_map

    # TODO - add support for kitti trained model
    def eval_and_log_kitti(self):
        """
        Evaluation and Logging Kitti Base Model
        Runs the evaluation for base models trained on Kitti dataset
        Warning - not yet implimented so does not log to AML and just prints
        :return: final_map
        """
        input_fn = self.eval_input_fns[0]

        metric = (self
                  .estimator
                  .evaluate(input_fn,
                            steps=None,
                            checkpoint_path=tf.train.latest_checkpoint(
                                self.train_run.checkpoint_dir)))
        print('-----Metrics------')
        print(metric)

        final_map = 0
        return final_map

    def log_con_mat(self, dt):
        """
        Log Confusion Matrix
        Takes the predictions from model evaluation class and
        forms a confusion matrix to log
        :param dt: binary results dictionary
        """
        class_labels = ['Defect', 'Clean']
        tp = int(dt.get('TP')[0])
        fp = int(dt.get('FP')[0])
        fn = int(dt.get('FN')[0])
        tn = int(dt.get('TN')[0])

        con_mat = [[tp, fn],
                   [fp, tn]]

        print(con_mat)

        con_mat_json = self.format_json(class_labels, con_mat)

        print(con_mat_json)

        run = self.train_run.run
        run.log_confusion_matrix('Image - Confusion Matrix', con_mat_json)

    def format_json(self, class_labels, con_mat):
        """
        Format Json for AML confusion matrix
        :param class_labels: confusion matrix class labels
        :param con_mat: confusion matrix
        :return: confusion matrix json
        """
        con_mat_json = {
            "schema_type": "confusion_matrix",
            "schema_version": "v1",
            "data": {
                "class_labels": class_labels,
                "matrix": con_mat
            }
        }
        return con_mat_json

    def create_summary(self,
                       final_map,
                       binary_dict):
        """
        Create Summary
        Creates a summary file in fileshare of the experiment
        for the reporting flow to PowerBI
        :param final_map: final map of model
        :param binary_dict: binary results dictionary
        """

        date = datetime.today().strftime('%y%m%d%H%M%S')
        exp_id = self.train_run.run.get_details().get('runId')

        train = self.train_run.train_csv
        test = self.train_run.test_csv

        TP = binary_dict.get('TP')
        TN = binary_dict.get('TN')
        FP = binary_dict.get('FP')
        FN = binary_dict.get('FN')
        prec = binary_dict.get('Precision')
        rec = binary_dict.get('Recall')
        acc = binary_dict.get('Accuracy')

        file_name = 'summary_{}.csv'.format(exp_id)

        file_dir = os.path.join(self.train_run.data_dir,
                                'summaries',
                                self.train_run.image_type)

        os.makedirs(file_dir, exist_ok=True)

        file_path = os.path.join(file_dir, file_name)

        df = pd.DataFrame({'Usecase': self.train_run.image_type,
                           'ExperimentID': exp_id,
                           'Date': date,
                           'Train': train,
                           'Test': test,
                           'mAP': final_map,
                           'TP': TP,
                           'TN': TN,
                           'FP': FP,
                           'FN': FN,
                           'Accuracy': acc,
                           'Precision': prec,
                           'Recall': rec}, index=[0])

        df.to_csv(file_path, index=False, sep=',')
