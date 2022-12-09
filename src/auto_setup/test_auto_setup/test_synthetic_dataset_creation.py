import unittest
import os
import tempfile

from synthetic_dataset_creation import make_directories, generate_dataset


class TestSyntheticDatasetCreation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryFile()

    def tearDown(self):
        self.test_dir.close()

    def test_make_directories(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp_name = tmpdirname + "/unit_test_make_directories"
            _, _, _ = make_directories(exp_name)

            self.assertTrue(os.path.exists(exp_name))
            self.assertTrue(os.path.exists(exp_name + "/images"))
            self.assertTrue(os.path.exists(exp_name + "/datasets"))

    def test_generate_dataset(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp_name = tmpdirname + "/unit_test_generate_dataset"

            generate_dataset(10, 0.8, 100, 100, exp_name)

            # Check how many images were created
            _, _, image_files = next(os.walk(exp_name + "/images"))
            image_file_count = len(image_files)

            self.assertEqual(image_file_count, 10)


if __name__ == "__main__":
    unittest.main()
