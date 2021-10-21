'''Test Training

Collection of unit tests using pytest to check for code issues
and ensure code quality. These test will be automatically executed
on creation of a pull reuqest and will have to pass before
any merges to master.
'''

import os
import sys
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from tfod_utils.evaluation import Eval # noqa

__here__ = os.path.dirname(__file__)
parentdir = os.path.dirname(__here__)
sys.path.insert(0, parentdir)


class TestEvalMetrics(unittest.TestCase):
    def setUp(self):
        df_gt = os.path.join(
            __here__,
            'test_data',
            'labels.csv'
        )
        self.df_gt = pd.read_csv(df_gt)

        df_pred = os.path.join(
            __here__,
            'test_data',
            'predictions.csv'
        )
        self.df_pred = pd.read_csv(df_pred)

        class_metrics = os.path.join(
            __here__,
            'test_data',
            'class_metrics.csv'
        )
        self.class_metrics = pd.read_csv(class_metrics)

        # intialise empty class, we will add as needed for each test
        self.eval_run = Eval("test",
                             "test",
                             "test",
                             "test")

        # intialise the predictions and ground truth
        self.eval_run.df_gt = self.df_gt
        self.eval_run.df_pred = self.df_pred

    def test_classification_binary(self):
        expected_results = {'TP': [3],
                            'FP': [2],
                            'FN': [2],
                            'TN': [3],
                            'Precision': [0.6],
                            'Recall': [0.6],
                            'Accuracy': [0.6]}

        results = self.eval_run.binary_classification_image()

        self.assertDictEqual(results, expected_results)

    def test_classification_class(self):
        results = self.eval_run.binary_classification_class()

        results = results.sort_values(by=['class_name',
                                          'binary_classification_class'])
        print(type(results))

        class_metrics = (self.class_metrics
                             .sort_values(by=['class_name',
                                              'binary_classification_class']))
        print(type(class_metrics))
        assert_frame_equal(results,
                           class_metrics,
                           check_like=True)
