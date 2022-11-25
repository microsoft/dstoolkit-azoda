from datetime import datetime
import cv2
import numpy as np
import os
import pandas as pd


def add_rect(img: np.array):
    """Add a rectangle to the image in place.

    Args:
        img (np.array): Array containing the image to add the rectangle to."""

    # Generate two points
    pt1_x = np.random.randint(low=0, high=width)
    pt1_y = np.random.randint(low=0, high=height)
    pt2_x = np.random.randint(low=0, high=width)
    pt2_y = np.random.randint(low=0, high=height)

    # Generate a random color
    clr1 = np.random.randint(low=0, high=255)
    clr2 = np.random.randint(low=0, high=255)
    clr3 = np.random.randint(low=0, high=255)
    rnd_clr = (clr1, clr2, clr3)

    # Create a random thickness
    rnd_thickness = np.random.randint(low=2, high=10)

    # Add the rectangle onto the image
    _ = cv2.rectangle(
        img,
        pt1=(pt1_x, pt1_y),
        pt2=(pt2_x, pt2_y),
        color=rnd_clr,
        thickness=rnd_thickness,
    )


def add_circle(img: np.array, thickness="random", central=True) -> list:
    """Add a circle to the image in place.

    Args:
        img (np.array): Array containing the image to add the circle to.
        thickness (str, optional): Thickness of the circle. Defaults is "random".
        central (bool, optional): Whether the circle should be central. Defaults is True.

    Returns:
        list: List containing a bounding box for the circle."""

    min_dim = min(width, height)

    if central:
        # Calculate the center of the image
        lower_bound_width = width // 4
        upper_bound_width = 3 * width // 4
        lower_bound_height = height // 4
        upper_bound_height = 3 * height // 4
        radius = np.random.randint(low=5, high=min_dim // 4)
    else:
        lower_bound_width = 0
        upper_bound_width = width
        lower_bound_height = 0
        upper_bound_height = height

        # Generate a random radius
        radius = np.random.randint(low=5, high=min_dim)

    # Generate random point
    pt1_x = np.random.randint(low=lower_bound_width, high=upper_bound_width)
    pt1_y = np.random.randint(low=lower_bound_height, high=upper_bound_height)

    # Generate random colour
    clr1 = np.random.randint(low=0, high=255)
    clr2 = np.random.randint(low=0, high=255)
    clr3 = np.random.randint(low=0, high=255)
    rnd_clr = (clr1, clr2, clr3)

    if thickness == -1:
        rnd_thickness = -1
    else:
        rnd_thickness = np.random.randint(low=2, high=10)

    # Add the circle onto the image
    _ = cv2.circle(
        img,
        center=(pt1_x, pt1_y),
        radius=radius,
        color=rnd_clr,
        thickness=rnd_thickness,
    )

    # Return the bounding box for the circle
    x_min = pt1_x - radius
    x_max = pt1_x + radius
    y_min = pt1_y - radius
    y_max = pt1_y + radius

    return [x_min, x_max, y_min, y_max]


def add_line(img: np.array):
    """Add a line to the image in place.

    Args:
        img (np.array): Array containing the image to add the line to."""

    # Generate two points
    pt1_x = np.random.randint(low=0, high=width)
    pt1_y = np.random.randint(low=0, high=height)
    pt2_x = np.random.randint(low=0, high=width)
    pt2_y = np.random.randint(low=0, high=height)

    # Generate a random color
    clr1 = np.random.randint(low=0, high=255)
    clr2 = np.random.randint(low=0, high=255)
    clr3 = np.random.randint(low=0, high=255)
    rnd_clr = (clr1, clr2, clr3)

    # Create a random thickness
    rnd_thickness = np.random.randint(low=2, high=10)

    # Add the line onto the image
    _ = cv2.line(
        img,
        pt1=(pt1_x, pt1_y),
        pt2=(pt2_x, pt2_y),
        color=rnd_clr,
        thickness=rnd_thickness,
    )


def add_blur(img: np.array) -> np.array:
    """Add a blur to the image.

    Args:
        img (np.array): Array containing the image to add the blur to.

    Returns:
        np.array: Array containing the blurred image."""

    ksize = (101, 101)
    return cv2.blur(img, ksize, cv2.BORDER_DEFAULT)


def make_directories(exp_name: str):
    """Generate the output directories for the dataset.

    Args:
        exp_name (str): Name of the experiment.

    Returns:
        str: Path to the directory to put the dataset in.
        str: Path to the directory to put the images in.
        str: Path to the directory to put the labels in."""

    output_base = f"{exp_name}/"
    output_images = output_base + "images/"
    output_datasets = output_base + "datasets/"
    os.makedirs(output_datasets)
    os.makedirs(output_images)

    return output_base, output_images, output_datasets


def generate_dataset(
    img_count: int, train_split: float, width: int, height: int, exp_name: str
):
    """Generate the dataset.

    Args:
        img_count (int): Total quantity of images to generate.
        train_split (float): Split of images to use for training.
        width (int): Width of the image.
        height (int): Height of the image.
        exp_name (str): Name of the experiment."""

    # Generate the output directories
    output_base, output_images, _ = make_directories(exp_name)

    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    export_path_train = output_base + f"datasets/train_{exp_name}_{time_stamp}.csv"
    export_path_test = output_base + f"datasets/test_{exp_name}_{time_stamp}.csv"
    row_data_train = []
    row_data_test = []
    np.random.seed(2022)

    for img_number in range(img_count):
        filename = f"img_{img_number}.jpg"
        print(filename)

        # Create a blank image
        img = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

        # Add 20 circles to the image and blur it
        for _ in range(20):
            _ = add_circle(img, thickness=-1, central=False)
            img = add_blur(img)

        for _ in range(2):
            if np.random.rand() > 0.2:
                bounding_box = add_circle(img)

                # If this is a training image, add the bounding box to the training data or the test data accordingly
                if img_number < (train_split * img_count):
                    row_data_train.append(
                        [
                            filename,
                            bounding_box[0],
                            bounding_box[1],
                            bounding_box[2],
                            bounding_box[3],
                            "circle",
                        ]
                    )
                else:
                    row_data_test.append(
                        [
                            filename,
                            bounding_box[0],
                            bounding_box[1],
                            bounding_box[2],
                            bounding_box[3],
                            "circle",
                        ]
                    )

            # Add lines and rectangles with randomly probability
            if np.random.rand() > 0.2:
                add_rect(img)
            if np.random.rand() > 0.2:
                add_line(img)
            if np.random.rand() > 0.2:
                add_rect(img)
            if np.random.rand() > 0.2:
                add_line(img)

        cv2.imwrite(output_images + filename, img)

    # Save the training data to a csv file
    column_names = ["filename", "xmin", "xmax", "ymin", "ymax", "class"]
    df_train = pd.DataFrame(row_data_train)
    df_train.columns = column_names
    df_train.to_csv(export_path_train, index=False)

    # Save the test data to a csv file
    df_test = pd.DataFrame(row_data_test)
    df_test.columns = column_names
    df_test.to_csv(export_path_test, index=False)


if __name__ == "__main__":
    generate_dataset(100, 0.9, 640, 320, "synthetic_dataset")
