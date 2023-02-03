import unittest
from unittest import mock
import os
import tempfile
import argparse
import yaml

from create_synthetic_dataset import make_directories, generate_dataset
from create_dataset_splits import create_test_train_split, parse_arguments


class TestCreateDatasetSplit(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryFile()

    def tearDown(self):
        self.test_dir.close()

    @mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            dataset="azoda-dataset",
        ),
    )
    def test_parse_arguments(self, _):
        """Test the parse_arguments function."""
        args = parse_arguments()

        self.assertEqual(args.dataset, "azoda-dataset")

    def test_parse_arguments_no_args(self):
        """Test the parse_arguments function with no arguments."""
        with self.assertRaises(SystemExit):
            parse_arguments()

    def test_create_test_train_split(self):
        """Test the generate_aml_config function."""

        dataset = "azoda-dataset"

        create_test_train_split(
            dataset,
        )


if __name__ == "__main__":
    unittest.main()
