import argparse
import os
import pandas as pd
import shutil

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='Name of dataset/project')
args = parser.parse_args()

project_name = f'../../{args.dataset}'
datasets_path = f'{project_name}/datasets/'
dataset_names = [f for f in os.listdir(datasets_path)
                 if os.path.isfile(os.path.join(datasets_path, f))
                 and f.startswith('test')]
latest_test_dataset = sorted(dataset_names)[-1]
df = pd.read_csv(os.path.join(datasets_path, latest_test_dataset))
filenames = set(df['filename'])
src_loc = os.path.join(project_name, 'images')
dst_loc = os.path.join(project_name, 'test_images')
os.makedirs(dst_loc)
for filename in filenames:
    shutil.copyfile(os.path.join(f'{src_loc}/{filename}'), os.path.join(f'{dst_loc}/{filename}'))
