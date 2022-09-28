# Script to partition a csv annotation file into train and test splits

# Imports
import argparse
import pandas as pd
import util

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Name of dataset/project")
args = parser.parse_args()
dataset_name = args.dataset

dataset_path = util.get_lastest_iteration(".", req_prefix=f"all_{dataset_name}")
df = pd.read_csv(dataset_path)
train_test_split = 0.8
df_train = df[: round(train_test_split * df.shape[0])]
df_test = df[round(train_test_split * df.shape[0]) :]

df_train.to_csv(
    dataset_path.replace(f"all_{dataset_name}", f"train_{dataset_name}"), index=False
)
df_test.to_csv(
    dataset_path.replace(f"all_{dataset_name}", f"test_{dataset_name}"), index=False
)
