"""
Example script to convert raw mask images into the csv row based training/test datasets
Containing polygon lists used for segmentation
"""

import cv2
from datetime import datetime
import os, csv
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.image as mpimg


def name_files(use_case):
    """
    Name Files
    Naming the files with date stamp.
    :return: Names for version, train and test files
    """
    todays_date = datetime.today().strftime('%y%m%d%H%M%S')
    train_name = 'train_' + use_case + '_' + todays_date + '.csv'
    test_name = 'test_' + use_case + '_' + todays_date + '.csv'
    return train_name, test_name


def visualize_polygons(img_path, polygons, output_path):
    img = Image.open(img_path)
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    for polygon in polygons:
        draw.polygon(tuple(polygon), fill=200, outline=None)
    img3 = Image.blend(img, img2, 0.5)
    img3.save(output_path)


def visualize_bboxes(img_path, bboxes, output_path, color='red'):
    img = Image.open(img_path)
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    for bounding_box in bboxes:
        [l, t, w, h] = bounding_box  # [l, t, weight, height]
        [l, t, r, b] = [l, t, l + w, t + h]  # [l, t, right, bottom]
        xy = [(l, b), (r, t)]   # Two points to identify a rectangle
        draw.rectangle(xy=xy, outline=color, width=5, fill=None)
    img3 = Image.blend(img, img2, 0.5)
    img3.save(output_path)


def mask2polygons(mask):
    """ Convert a mask image into list of polygons """
    mask_np = mpimg.imread(mask)
    contours, _ = cv2.findContours((mask_np).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []
    for contour in contours:
        if contour.size > 6:  # Need at least 6 (3 points) to build a valid polygon
            polygon = contour.flatten().tolist()  # Extract polygon list
            bbox = list(cv2.boundingRect(contour))  # Extract bbox list [l, t, w, h]
            annotations.append([bbox, polygon])
    return annotations


def write_annotation_csv(img_name, annotations, class_name, csv_path):
    """ Generate csv files used for mask-rcnn based model training """
    with open(csv_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        if os.stat(csv_path).st_size == 0:   # Write header on newly created file
            writer.writerow(['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'segmentation', 'class'])

        for annotation in annotations:
            bbox, polygon = annotation[0], annotation[-1]
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            writer.writerow([img_name, xmin, ymin, xmax, ymax, [polygon], class_name])


def prepare_dataset(img_labels, class_name, out_file_path, out_visualization=None):
    print('Directory size ' + str(len(os.listdir(img_labels))))

    acc =0
    for f in os.listdir(img_labels):
        annotations = mask2polygons(mask=os.path.join(img_labels, f))
        write_annotation_csv(img_name=f, class_name=class_name, annotations=annotations, csv_path=out_file_path)

        if out_visualization:
            bboxes = [annotation[0] for annotation in annotations]
            polygons = [annotation[-1] for annotation in annotations]

            visualize_bboxes(img_path=os.path.join(img_labels, f),
                             bboxes=bboxes,
                             output_path=os.path.join(out_visualization, f))

            visualize_polygons(img_path=os.path.join(out_visualization, f),
                               polygons=polygons,
                               output_path=os.path.join(out_visualization, f))
        acc += 1
        print(f'Create new record for image {f}, at position {acc}')



if __name__ == '__main__':
    use_case = 'usecase'
    train_name, test_name = name_files(use_case)
    out_file_train = f"outpath/{train_name}"
    out_file_test = f"outpath/{test_name}"

    img_labels_train = "path"  # directory containing training mask images
    img_labels_test = "path"  # directory containing test mask images
    out_visualization = 'path' # if set, annotations are visualized for later labelling check

    # generate training dataset
    prepare_dataset(img_labels=img_labels_train,
                    class_name=use_case,
                    out_file_path=out_file_train,
                    out_visualization=out_visualization)

    # generate test dataset
    prepare_dataset(img_labels=img_labels_test,
                    class_name=use_case,
                    out_file_path=out_file_test,
                    out_visualization=out_visualization)