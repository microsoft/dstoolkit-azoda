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
            exp_name = tmpdirname + "/unit_test"
            _, _, _ = make_directories(exp_name)

            self.assertTrue(os.path.exists(exp_name))
            self.assertTrue(os.path.exists(exp_name + "/images"))
            self.assertTrue(os.path.exists(exp_name + "/datasets"))


if __name__ == "__main__":
    unittest.main()
