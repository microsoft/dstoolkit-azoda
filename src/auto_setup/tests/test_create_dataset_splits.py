import unittest
from unittest import mock
import os
import tempfile
import argparse
import pandas as pd

from create_dataset_splits import create_test_train_split, parse_arguments


class TestCreateDatasetSplit(unittest.TestCase):
    @mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            dataset="azoda-dataset",
            split=0.5,
        ),
    )
    def test_parse_arguments(self, _):
        """Test the parse_arguments function."""
        args = parse_arguments()

        self.assertEqual(args.dataset, "azoda-dataset")
        self.assertEqual(args.split, 0.5)

    def test_parse_arguments_no_args(self):
        """Test the parse_arguments function with no arguments."""
        with self.assertRaises(SystemExit) as cm:
            parse_arguments()

        self.assertEqual(cm.exception.code, 2)

    def test_create_test_train_split(self):
        """Test the create_test_train_split function."""

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a mock csv
            num_mock_records = 100
            test_df = pd.DataFrame(
                {
                    "filename": list(range(num_mock_records)),
                }
            )

            mock_dataset_name = "mock_dataset"
            test_df.to_csv(
                os.path.join(tmpdirname, f"{mock_dataset_name}.csv"), index=False
            )

            mock_train_test_split = 0.6
            create_test_train_split(
                mock_dataset_name,
                train_test_split=mock_train_test_split,
                req_prefix="",
                directory_name=tmpdirname,
            )

            # Open up the resulting files and check the split

            test_data = pd.read_csv(
                os.path.join(
                    tmpdirname,
                    f"test_{mock_dataset_name}.csv",
                )
            )

            train_data = pd.read_csv(
                os.path.join(
                    tmpdirname,
                    f"train_{mock_dataset_name}.csv",
                )
            )

            # Check we split correctly
            self.assertEqual(
                test_data.shape[0],
                round(num_mock_records * (1 - mock_train_test_split)),
            )

            self.assertEqual(
                train_data.shape[0], round(num_mock_records * mock_train_test_split)
            )

    def test_create_test_train_split_with_invalid_test_train_split(self):
        """Test the create_test_train_split function with an invalid test train split."""

        with self.assertRaises(ValueError):
            create_test_train_split("test_dataset", train_test_split=1.1)

    def test_create_test_train_split_with_prefix(self):
        """Test the create_test_train_split function."""

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a mock csv
            num_mock_records = 100
            test_df = pd.DataFrame(
                {
                    "filename": list(range(num_mock_records)),
                }
            )

            mock_dataset_name = "mock_dataset"
            test_df.to_csv(
                os.path.join(tmpdirname, f"all_{mock_dataset_name}.csv"), index=False
            )

            mock_train_test_split = 0.7
            create_test_train_split(
                mock_dataset_name,
                train_test_split=mock_train_test_split,
                req_prefix="all_",
                directory_name=tmpdirname,
            )

            # Open up the resulting files and check the split

            test_data = pd.read_csv(
                os.path.join(
                    tmpdirname,
                    f"test_{mock_dataset_name}.csv",
                )
            )

            train_data = pd.read_csv(
                os.path.join(
                    tmpdirname,
                    f"train_{mock_dataset_name}.csv",
                )
            )

            # Check we split correctly
            self.assertEqual(
                test_data.shape[0],
                round(num_mock_records * (1 - mock_train_test_split)),
            )

            self.assertEqual(
                train_data.shape[0], round(num_mock_records * mock_train_test_split)
            )


if __name__ == "__main__":
    unittest.main()
