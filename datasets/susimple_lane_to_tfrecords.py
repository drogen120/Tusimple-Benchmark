import os
import sys
import random

import numpy as np
import tensorflow as tf
import cv2
import json
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

DATASET_PATH='./train_set/'
LABEL_FILE_NAME='label_data_0531.json'
SAMPLES_PER_FILES = 200
def _process_image(directory, name):
    filename = directory+name
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    shape = list(cv2.imread(filename).shape)
    return image_data, shape


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def _convert_to_example(image_data, gt_lanes, y_samples, shape):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    if len(gt_lanes) < 5:
        padding_list = [-2]*56
        padding = 5 - len(gt_lanes)
        for i in range(padding):
            gt_lanes.append(padding_list)

    image_format = b'JPG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/line1': float_feature(gt_lanes[0]),
            'image/line2': float_feature(gt_lanes[1]),
            'image/line3': float_feature(gt_lanes[2]),
            'image/line4': float_feature(gt_lanes[3]),
            'image/line5': float_feature(gt_lanes[4]),
            'image/ysamples': float_feature(y_samples),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, gt_lanes, y_samples, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape = _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, gt_lanes, y_samples, shape)
    tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir, output_dir, name='tusimple_lane', shuffling=False):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    json_gt = [json.loads(line) for line in open(dataset_dir+LABEL_FILE_NAME).readlines()]

    i = 0
    j = 0
    fidx = 3
    while i < len(json_gt):
        
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            while i < len(json_gt) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1,
                                                                  len(json_gt)))
                sys.stdout.flush()
                gt_lanes = json_gt[i]['lanes']
                y_samples = json_gt[i]['h_samples']
                raw_file = json_gt[i]['raw_file']
                _add_to_tfrecord(dataset_dir, raw_file, gt_lanes, y_samples, tfrecord_writer)
                i += 1
                j += 1
            j = 0
            fidx += 1

    print('\nFinished converting the dataset!')



