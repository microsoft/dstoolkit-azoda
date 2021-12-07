from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import tensorflow.compat.v1 as tf
import pandas as pd
import json
import numpy as np

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
from pycocotools import mask


# Helper Funtions
def read_json(path_json):
    """
    Read Json File
    :param path_json: path to json
    :returns mapping_json: json file contents as dict
    """
    with open(path_json) as json_file:
        mapping_json = json.load(json_file)
    return mapping_json


def split(df, group):
    """
    Split Dataframe to group list
    Takes the pandas dataframe and groups columsn contents to filename as a
    list
    :param df: train or test datafame to group
    :param group: column to group on
    :returns list_group: list of grouped objects
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    list_group = [data(filename, gb.get_group(x)) for filename,
                  x in zip(gb.groups.keys(), gb.groups)]
    return list_group


class TFRecord():
    def __init__(self, base_path, image_dir, train_df, test_df, include_masks):
        """
        TF Record
        Creates a set of tfrecords from version files
        :param base_path: base directory path to image type,
        avaliable from the TFOD run object
        :param image_dir: directory path of images
        :param train_df: train pandas dataframe from version file
        :param test_df: test pandas dataframe from version file
        :param include_masks: Whether to include instance segmentations masks (PNG encoded) in the result
        """
        self.base_path = base_path
        self.image_dir = image_dir
        self.df_train = train_df
        self.df_test = test_df
        self.include_masks = include_masks
        self.create_records()

    def create_records(self):
        """
        Create records
        Takes the version file and images and creates tfrecord files and
        a mapping file
        """
        train_df = self.df_train
        test_df = self.df_test
        train_grouped = split(train_df, 'filename')
        test_grouped = split(test_df, 'filename')

        self.mapping_file = self.write_mapping_files()
        self.train_tfrecords = self.write_tfrecords(train_grouped, 'train')
        self.test_tfrecords = self.write_tfrecords(test_grouped, 'test')

    def write_mapping_files(self):
        """
        Write class mapping to file
        Takes the list of classes from the train and test frames and generates
        the class map file
        :returns map_file_path: the path to the mapping file
        """
        self.classes = (pd.concat([self.df_train,
                                   self.df_test],
                                  ignore_index=True)
                        ['class']
                        .unique()
                        .tolist())

        dir_path = os.path.join(os.getcwd(), 'label_files')

        map_file_path = os.path.join(dir_path, 'tf_label_map.pbtxt')

        class_map_file = os.path.join(dir_path, 'class_map.json')

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        dict_classes = {}

        with open(map_file_path, 'w') as map_file:
            for class_index, class_name in enumerate(self.classes, 1):
                if class_name is np.nan:
                    continue
                map_file.write('item {\n')
                map_file.write(' id:{}\n'.format(class_index))
                map_file.write(' name:\'{}\'\n'.format(class_name))
                if class_index is len(self.classes):
                    map_file.write('}')
                else:
                    map_file.write('}\n')
                dict_classes[class_name] = class_index

        with open(class_map_file, 'w') as json_file:
            json.dump(dict_classes, json_file)

        print('Successfully created the class mapping .json: {}'
              .format(class_map_file))
        print('Successfully created the tf mapping .pbtxt: {}'
              .format(map_file_path))
        return map_file_path

    def write_tfrecords(self, grouped, file_type):
        """
        Write tfrecords to file
        Takes the grouped list of bboxes and file type and creates the tfrecord
        :param grouped: list of bboxes grouped by filename
        :param file_type: prefix of train or test for naming record set
        :returns output_path: path to records file
        """
        file_name = file_type + '.record'

        output_path = os.path.join(os.getcwd(),
                                   'label_files',
                                   file_name)

        dir_name = os.path.dirname(output_path)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        writer = tf.python_io.TFRecordWriter(output_path)

        input_path = os.path.join(self.base_path,
                                  self.image_dir)

        json_mapping_path = os.path.join(os.getcwd(),
                                         'label_files',
                                         'class_map.json')

        dict_mapping = read_json(json_mapping_path)

        for group in grouped:
            tf_example = self.create_tf_example(group,
                                                input_path,
                                                dict_mapping)

            writer.write(tf_example.SerializeToString())

        print('Successfully created the TFRecords: {}'.format(output_path))
        return output_path

    def create_tf_example(self, group, path, dict_mapping):
        """
        Create tf record example
        takes the group of labels and creates a tf example
        :param group: group object from list of groups containing bboxes
        for an image
        :param path: path to image for group example
        :param dict_mapping: class to int mapping dict
        :returns tf_example: tf_example object
        """
        write_path = os.path.join(path, '{}'.format(group.filename))
        with tf.gfile.GFile(write_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        c_text = []
        classes = []
        area = []
        is_crowd = []
        encoded_mask_png = []

        for index, row in group.object.iterrows():
            if row['class'] is np.nan:
                continue
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            c_text.append(row['class'].encode('utf8'))
            classes.append(dict_mapping[row['class']])

            # ToDo: @Hong conduct polygon validity check
            if self.include_masks:
                is_crowd.append(row['iscrowd'])
                run_len_encoding = mask.frPyObjects(row['segmentation'], height, width)
                area.append(mask.area(run_len_encoding))
                binary_mask = mask.decode(run_len_encoding)
                if not row['iscrowd']:
                    binary_mask = np.amax(binary_mask, axis=2)
                binary_mask = (binary_mask * 255).astype(np.uint8)
                pil_image = Image.fromarray(binary_mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format='PNG')
                encoded_mask_png.append(output_io.getvalue())

        feature_dict = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(c_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),  # noqa
        }

        if self.include_masks:
            feature_dict['image/object/is_crowd'] = dataset_util.int64_list_feature(is_crowd)
            feature_dict['image/object/area'] = dataset_util.float_list_feature(area)
            feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png))

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return tf_example

    def inspect_record(self, record, out_path):
        """
        Writes the contents of a TFRecord to file for inspection
        :param record: tf record file to inspect
        :param out_path: file output path to write plain text record details
        """
        for i, example in enumerate(tf.python_io
                                    .tf_record_iterator(record), 1):
            result = tf.train.SequenceExample.FromString(example)
            path = os.path.join(out_path, "objec_{}.txt".format(i))
            with open(path, 'w') as f:
                f.write(str(result))
