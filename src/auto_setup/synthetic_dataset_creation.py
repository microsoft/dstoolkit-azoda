# %%
import cv2
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# %%


def add_rect(img):
    pt1_x = np.random.randint(low=0, high=width)
    pt1_y = np.random.randint(low=0, high=height)
    pt2_x = np.random.randint(low=0, high=width)
    pt2_y = np.random.randint(low=0, high=height)
    clr1 = np.random.randint(low=0, high=255)
    clr2 = np.random.randint(low=0, high=255)
    clr3 = np.random.randint(low=0, high=255)
    rnd_clr = (clr1, clr2, clr3)
    rnd_thickness = np.random.randint(low=2, high=10)
    _ = cv2.rectangle(img,
                      pt1=(pt1_x, pt1_y),
                      pt2=(pt2_x, pt2_y),
                      color=rnd_clr,
                      thickness=rnd_thickness)


def add_circle(img, thickness='random', central=True):
    min_dim = min(width, height)
    if central:
        lower_bound_width = width//4
        upper_bound_width = 3*width//4
        lower_bound_height = height//4
        upper_bound_height = 3*height//4
        radius = np.random.randint(low=5, high=min_dim//4)
    else:
        lower_bound_width = 0
        upper_bound_width = width
        lower_bound_height = 0
        upper_bound_height = height
        radius = np.random.randint(low=5, high=min_dim)
    pt1_x = np.random.randint(low=lower_bound_width, high=upper_bound_width)
    pt1_y = np.random.randint(low=lower_bound_height, high=upper_bound_height)
    clr1 = np.random.randint(low=0, high=255)
    clr2 = np.random.randint(low=0, high=255)
    clr3 = np.random.randint(low=0, high=255)
    rnd_clr = (clr1, clr2, clr3)
    if thickness == -1:
        rnd_thickness = -1
    else:
        rnd_thickness = np.random.randint(low=2, high=10)
    _ = cv2.circle(img,
                   center=(pt1_x, pt1_y),
                   radius=radius,
                   color=rnd_clr,
                   thickness=rnd_thickness)
    x_min = pt1_x-radius
    x_max = pt1_x+radius
    y_min = pt1_y-radius
    y_max = pt1_y+radius
    return [x_min, x_max, y_min, y_max]


def add_line(img):
    pt1_x = np.random.randint(low=0, high=width)
    pt1_y = np.random.randint(low=0, high=height)
    pt2_x = np.random.randint(low=0, high=width)
    pt2_y = np.random.randint(low=0, high=height)
    clr1 = np.random.randint(low=0, high=255)
    clr2 = np.random.randint(low=0, high=255)
    clr3 = np.random.randint(low=0, high=255)
    rnd_clr = (clr1, clr2, clr3)
    rnd_thickness = np.random.randint(low=2, high=10)
    _ = cv2.line(img,
                 pt1=(pt1_x, pt1_y),
                 pt2=(pt2_x, pt2_y),
                 color=rnd_clr,
                 thickness=rnd_thickness)


def add_blur(img):
    ksize = (101, 101)
    return cv2.blur(img, ksize, cv2.BORDER_DEFAULT)


# %%
width, height = 640, 320
output_base = '../../../data/'
output_images = output_base + 'images/'
export_path = output_base + 'labels.csv'
row_data = []
for i in range(10):
    filename = f'img_{i}.jpg'
    print(filename)
    img = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

    for i in range(20):
        _ = add_circle(img, thickness=-1, central=False)
        img = add_blur(img)

    for i in range(2):
        if np.random.rand() > 0.2:
            bb = add_circle(img)
            row_data.append([filename,
                             bb[0],
                             bb[1],
                             bb[2],
                             bb[3],
                             'circle'])
        if np.random.rand() > 0.2:
            add_rect(img)
        if np.random.rand() > 0.2:
            add_line(img)
        if np.random.rand() > 0.2:
            add_rect(img)
        if np.random.rand() > 0.2:
            add_line(img)
    cv2.imwrite(output_images+filename, img)
# %%
df = pd.DataFrame(row_data)
column_names = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class']
df.columns = column_names
df.to_csv(export_path, index=False)

# %%

# %%
