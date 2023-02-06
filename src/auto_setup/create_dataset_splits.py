# Script to partition a csv annotation file into train and test splits

import argparse
import pandas as pd
import util


def create_test_train_split(
    dataset_name: str,
    train_test_split: float = 0.8,
    req_prefix: str = "all_",
    directory_name: str = ".",
) -> None:
    """Create a test and train split of the dataset.

    Args:
        dataset_name (str): Name of the dataset.
        train_test_split (float, optional): Percentage of the dataset to use for training. Defaults to 0.8.
        req_prefix (str, optional): Prefix of the dataset. Defaults to "all_".
        directory_name (str, optional): Directory to read the dataset from. Defaults to ".".
    """

    if train_test_split > 1 or train_test_split < 0:
        raise ValueError("The train_test_split must be between 0 and 1.")

    # Find the latest version of the dataset
    dataset_path = util.get_lastest_iteration(
        directory_name, req_prefix=f"{req_prefix}{dataset_name}"
    )

    dataset = pd.read_csv(dataset_path)

    num_rows = dataset.shape[0]

    dataset_train = dataset[: round(train_test_split * num_rows)]
    dataset_test = dataset[round(train_test_split * num_rows) :]

    dataset_train.to_csv(
        dataset_path.replace(f"{req_prefix}{dataset_name}", f"train_{dataset_name}"),
        index=False,
    )
    dataset_test.to_csv(
        dataset_path.replace(f"{req_prefix}{dataset_name}", f"test_{dataset_name}"),
        index=False,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # Parse arguments
    parser.add_argument(
        "-d", "--dataset", help="Name of dataset/project", required=True, type=str
    )

    parser.add_argument(
        "-s",
        "--split",
        help="Split to use for the test train split.",
        type=float,
        default=0.8,
        nargs="?",
        const=0.8,
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Get the args
    args = parse_arguments()

    # Create the test train split
    create_test_train_split(args.dataset, args.split)
