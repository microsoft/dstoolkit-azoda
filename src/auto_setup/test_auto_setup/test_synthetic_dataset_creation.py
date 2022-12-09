import unittest
import os
import tempfile
import pandas as pd
from math import ceil

from synthetic_dataset_creation import make_directories, generate_dataset


class TestSyntheticDatasetCreation(unittest.TestCase):
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
            self.assertTrue(os.path.exists(directory_name + "/images"))
            self.assertTrue(os.path.exists(directory_name + "/datasets"))

    def test_generate_dataset(self):
        """Test the generate_dataset function."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            directory_name = tmpdirname + "/unit_test_generate_dataset"

            test_image_file_count = 10
            test_train_split = 0.8
            generate_dataset(
                test_image_file_count, test_train_split, 100, 100, directory_name
            )

            # Check how many images were created
            _, _, image_files = next(os.walk(directory_name + "/images"))
            image_file_count = len(image_files)

            self.assertEqual(image_file_count, test_image_file_count)

            # Check records in the test and train datasets. The exact number is not known but train should be greater than test
            _, _, dataset_files = next(os.walk(directory_name + "/datasets"))

            test_data = pd.read_csv(
                "{}/datasets/{}".format(directory_name, dataset_files[0])
            )

            train_data = pd.read_csv(
                "{}/datasets/{}".format(directory_name, dataset_files[1])
            )

            self.assertGreater(
                test_data.shape[0], ceil(test_image_file_count * (1 - test_train_split))
            )

            self.assertGreater(
                train_data.shape[0], ceil(test_image_file_count * test_train_split)
            )

            self.assertGreater(train_data.shape[0], test_data.shape[0])

    def test_generate_dataset_invalid_img_count(self):
        """Test the generate_dataset function with an invalid image count."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = (
                    tmpdirname + "/unit_test_generate_dataset_invalid_img_count"
                )
                generate_dataset(-5, 0.8, 100, 100, directory_name)

    def test_generate_dataset_invalid_train_split(self):
        """Test the generate_dataset function with an invalid train split."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = tmpdirname + "/unit_test_generate_dataset_train_split"
                generate_dataset(5, 1.8, 100, 100, directory_name)

    def test_generate_dataset_invalid_width(self):
        """Test the generate_dataset function with an invalid width."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = (
                    tmpdirname + "/unit_test_generate_dataset_invalid_width"
                )
                generate_dataset(5, 0.8, -100, 100, directory_name)

    def test_generate_dataset_invalid_height(self):
        """Test the generate_dataset function with an invalid height."""
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                directory_name = (
                    tmpdirname + "/unit_test_generate_dataset_invalid_height"
                )
                generate_dataset(5, 0.8, 100, -100, directory_name)


if __name__ == "__main__":
    unittest.main()
