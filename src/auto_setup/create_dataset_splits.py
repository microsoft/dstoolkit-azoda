# Script to convert current annotation style to the yolo annotation convention
# %%
# Imports
import argparse
from PIL import Image
import os
import pandas as pd
import shutil
import util

# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--dataset", help="Name of dataset/project")
# args = parser.parse_args()
# dataset_name = args.dataset
dataset_name = "synthetic_dataset"
datasets_dir = "."
dataset_path = util.get_lastest_iteration(
    datasets_dir, req_prefix=f"all_{dataset_name}"
)

# %%
print(dataset_path)
# %%
df = pd.read_csv(dataset_path)

# %%
train_test_split = 0.8
# %%
df_train = df[: round(train_test_split * df.shape[0])]
df_test = df[round(train_test_split * df.shape[0]) :]
# %%
df_train
# %%
df_test
# %%
df_train.to_csv(
    dataset_path.replace(f"all_{dataset_name}", f"train_{dataset_name}"), index=False
)
df_test.to_csv(
    dataset_path.replace(f"all_{dataset_name}", f"test_{dataset_name}"), index=False
)
# %%
dataset_path
# %%
