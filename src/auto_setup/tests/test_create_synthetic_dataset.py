import unittest
import os
import tempfile
import pandas as pd
from math import ceil

from create_synthetic_dataset import make_directories, generate_dataset
from util import get_lastest_iteration


class TestCreateSyntheticDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryFile()

    def tearDown(self):
        self.test_dir.close()

    def test_make_directories(self):
        """Test the make_directories function."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            directory_name = tmpdirname + "/unit_test_make_directories"
            _, _, _ = make_directories(directory_name)

            self.assertTrue(os.path.exists(directory_name))
            self.assertTrue(os.path.exists(os.path.join(directory_name, "images")))
            self.assertTrue(os.path.exists(os.path.join(directory_name, "datasets")))

    def test_generate_dataset(self):
        """Test the generate_dataset function."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            directory_name = os.path.join(tmpdirname, "unit_test_generate_dataset")

            test_image_file_count = 10
            test_train_split = 0.8
            generate_dataset(
                test_image_file_count, test_train_split, 100, 100, directory_name
            )

            # Check how many images were created
            _, _, image_files = next(os.walk(os.path.join(directory_name, "images")))
            image_file_count = len(image_files)

            self.assertEqual(image_file_count, test_image_file_count)

            # Check records in the test and train datasets. The exact number is not known but train should be greater than test
            datasets_directory = os.path.join(directory_name, "datasets")

            test_data = pd.read_csv(
                os.path.join(
                    datasets_directory,
                    get_lastest_iteration(datasets_directory, req_prefix="test"),
                )
            )

            train_data = pd.read_csv(
                os.path.join(
                    datasets_directory,
                    get_lastest_iteration(datasets_directory, req_prefix="train"),
                )
            )

            self.assertLessEqual(
                test_data.shape[0],
                2 * ceil(test_image_file_count * (1 - test_train_split)),
            )

            self.assertLessEqual(
                train_data.shape[0], 2 * ceil(test_image_file_count * test_train_split)
            )

    def test_generate_dataset_invalid_img_count(self):
        """Test the generate_dataset function with an invalid image count."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = os.path.join(
                    tmpdirname, "unit_test_generate_dataset_invalid_img_count"
                )
                generate_dataset(-5, 0.8, 100, 100, directory_name)

    def test_generate_dataset_invalid_train_split(self):
        """Test the generate_dataset function with an invalid train split."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = os.path.join(
                    tmpdirname, "unit_test_generate_dataset_train_split"
                )
                generate_dataset(5, 1.8, 100, 100, directory_name)

    def test_generate_dataset_invalid_width(self):
        """Test the generate_dataset function with an invalid width."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = os.path.join(
                    tmpdirname, "unit_test_generate_dataset_invalid_width"
                )
                generate_dataset(5, 0.8, -100, 100, directory_name)

    def test_generate_dataset_invalid_height(self):
        """Test the generate_dataset function with an invalid height."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = os.path.join(
                    tmpdirname, "unit_test_generate_dataset_invalid_height"
                )
                generate_dataset(5, 0.8, 100, -100, directory_name)


if __name__ == "__main__":
    unittest.main()
