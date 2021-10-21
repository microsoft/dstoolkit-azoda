'''-Evaluates image-

Evaluates both positive and negative images and
returning the result from evaluation.

input:
    path_neg_img (str): Path to negative images
    path_test_csv (str): Path to positive image
    path_model (str): Path to trained model
    path_label_map (str): Path to label mapping file
    path_results (str): Path to results directory \
                        always set to None.

Output:
    df_class_metric (pd df): Dataframe containing results \
                        from evaluation
    num_images (str): Sentence with number of images
'''

import argparse
import os
import json

from tfod_utils.evaluation import Eval


def read_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_share',
                        help='File Share Path including usecase',
                        required=True),

    parser.add_argument('--test_file',
                        help='Test version file name in version files store',
                        required=True),

    parser.add_argument('--model_path',
                        help='Path to frozen model',
                        required=True),

    parser.add_argument('--label_map_file',
                        help='Path to label map file',
                        required=True)

    FLAGS = parser.parse_args()
    return FLAGS


def main():

    # parse arguments
    FLAGS = get_arguments()

    # get paths from args
    file_share = FLAGS.file_share
    test_file = FLAGS.test_file
    model_path = FLAGS.model_path
    label_map_path = FLAGS.label_map_file

    # Define paths - file share

    img_dir = os.path.join(file_share, 'images')

    test_csv = os.path.join(file_share, 'version_files', test_file)

    # Run Script
    Eval(img_dir, test_csv, model_path, label_map_path)


if __name__ == '__main__':
    main()
