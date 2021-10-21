'''
Script to convert test label data stored in XML into the csv row based
structure the repo expects
'''

import os
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split


def create_label_frame(img_labels):

    data_labels = []

    # loop images
    for img in os.listdir(img_labels):

        img_fp = os.path.join(img_labels, img)

        ext = os.path.splitext(img)[-1].lower()
        if ext == ".xml":
            root = ET.parse(img_fp).getroot()

            filename = root.find('filename').text

            # load xml file
            for type_tag in root.findall('object'):
                name = type_tag.find('name').text
                xmin = int(type_tag.find('bndbox/xmin').text)
                ymin = int(type_tag.find('bndbox/ymin').text)
                xmax = int(type_tag.find('bndbox/xmax').text)
                ymax = int(type_tag.find('bndbox/ymax').text)

                data_labels.append([filename, xmin, ymin, xmax, ymax, name])

        # convert each defect to image row in df
    column_names = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    df = pd.DataFrame(data_labels, columns=column_names)
    return df


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


def main(img_labels, out_file_train, out_file_test):

    label_df = create_label_frame(img_labels)

    images = label_df.filename.unique()

    train, test = train_test_split(images,
                                   train_size=0.8,
                                   test_size=0.2,
                                   shuffle=True)

    train_df = label_df[label_df['filename'].isin(train)]
    train_df.to_csv(out_file_train, index=False)

    test_df = label_df[label_df['filename'].isin(test)]
    test_df.to_csv(out_file_test, index=False)


if __name__ == '__main__':
    use_case = 'usecase'
    train_name, test_name = name_files(use_case)
    img_labels = "path"
    out_file_train = f"outpath/{train_name}"
    out_file_test = f"outpath/{test_name}"
    main(img_labels, out_file_train, out_file_test)
