# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for reading the WingNet training data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import cv2

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the WingNet data set.
NUM_CLASSES = 4
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 57060
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000


def read_wingnet(filename_queue):
    """Reads and parses examples from WingNet training data files.

    Arguments
        1. filename_queue: A queue of strings with the filenames to read from.

    Returns
        1. An object representing a single example, with the following fields:
            * rows:     number of rows in the image
            * cols:     number of columns in the image
            * depth:    number of color channels in the image
            * key:      a scalar string Tensor describing the filename &
                        record number for this example.
            * label:    an int32 Tensor with the label in the range 0 or 1.
            * image:    a [rows, cols, depth] uint8 Tensor with the image data
    """

    # empty object to store image object...
    class WingNetRecord(object):
        pass

    result = WingNetRecord()

    # Dimensions of the images in the WingNet training dataset
    result.rows = 32
    result.cols = 32
    result.depth = 1

    # setup the reader to read a whole PNG training file
    #reader = tf.WholeFileReader()
    #result.key, value = reader.read(filename_queue[0])
    value = tf.read_file(filename_queue[0])

    # Convert from a string to a vector of uint8 that is record_bytes long.
    # decode the png from the filename_queue in greyscale.
    # contents passed is the PNG file represented as a string format?
    result.image = tf.image.decode_png(contents = value,
                                       channels = 1,
                                       dtype = tf.uint8)

    result.label = filename_queue[1]
    return(result)


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Arguments
        1. image: 3-D tensor of [rows, cols, 1] of type.float32.
        2. label: 1-D tensor of type.int32
        3. min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        4. batch_size: Number of images per batch.
        5. shuffle: boolean indicating whether to use a shuffling queue.

    Returns
        1. images: Images. 4-D tensor of [batch_size, rows, cols, 1] size.
        2. labels: Labels. 1-D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 10
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size=batch_size,
                                                     num_threads=num_preprocess_threads,
                                                     capacity=min_queue_examples + 3 * batch_size,
                                                     min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch([image, label],
                                             batch_size=batch_size,
                                             num_threads=num_preprocess_threads,
                                             capacity=min_queue_examples + 3 * batch_size)
    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(labels_and_files, data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Arguments
        1. labels_and_files: numpy array of the files and their corresponding
           labels
        2. data_dir: Path to the WingNet training data directory.
        3. batch_size: Number of images per batch.

    Returns
        1. images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        2. labels: Labels. 1D tensor of [batch_size] size.
    """
    # extract just the filename and join with the data directory
    filenames = [os.path.join(data_dir, fname)
                 for fname in labels_and_files[:, 1]]
    labels = [label for label in labels_and_files[:, 0]]

    # check that the file exists
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    #filename_queue = tf.train.string_input_producer(filenames)
    filename_queue = tf.train.slice_input_producer([filenames, labels])

    # Read examples from files in the filename queue.
    read_input = read_wingnet(filename_queue)
    reshaped_image = tf.cast(read_input.image, tf.float32)
    distorted_image = tf.identity(reshaped_image)
    #read_input.label = tf.reshape(read_input.label, [1])

    rows = IMAGE_SIZE
    cols = IMAGE_SIZE

    distorted_image = tf.expand_dims(distorted_image, 0)
    # distorted_image = tf.image.resize_bicubic(distorted_image,
    #                                           size=[32, 32],
    #                                           align_corners=True)
    distorted_image = tf.reshape(distorted_image, shape=[rows, cols, 1])

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    with tf.variable_scope("data_augmentation"):
        with tf.variable_scope("random_lighting"):
            bern = tf.random_uniform(shape = [1], minval = 0, maxval = 1)
            if tf.greater(bern, tf.constant(0.5, shape = [1])) is not None:
                distorted_image = tf.image.random_brightness(distorted_image,
                                                             max_delta = 0.35)
                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower = 0.2,
                                                           upper = 1.8)
            else:
                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower = 0.2,
                                                           upper = 1.8)
                distorted_image = tf.image.random_brightness(distorted_image,
                                                             max_delta = 0.35)
        with tf.variable_scope("random_rotations"):
            # Randomly flip the image horizontally or vertically
            distorted_image = tf.image.random_flip_left_right(distorted_image)
            distorted_image = tf.image.random_flip_up_down(distorted_image)

            # Randomly rotate the image 90 deg
            num_rotations = tf.floor(tf.random_uniform(shape = [1],
                                                       minval = 0,
                                                       maxval = 4))
            num_rotations = tf.to_int32(num_rotations)
            distorted_image = tf.image.rot90(distorted_image, k = num_rotations[0])

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
    read_input.label = tf.cast(tf.string_to_number(read_input.label, tf.int32), tf.int32)

    # Set the shapes of tensors.
    float_image.set_shape([rows, cols, 1])
    # read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d WingNet training images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=True)


def inputs(eval_data, labels_and_files, data_dir, batch_size):
    """Construct input for WingNet evaluation using the Reader ops.

    Arguments
        1. eval_data: bool, indicating if one should use the train or eval data set.
        2. labels_and_files: numpy array of the files and their corresponding
           labels
        3. data_dir: Path to the WingNet data directory.
        4. batch_size: Number of images per batch.

    Returns
        1. images: Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        2. labels: Labels. 1-D tensor of [batch_size] size.
    """

    # extract just the filename and join with the data directory
    filenames = [os.path.join(data_dir, fname)
                 for fname in labels_and_files[:, 1]]
    labels = [label for label in labels_and_files[:, 0]]

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.slice_input_producer([filenames, labels])
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN if not eval_data \
        else NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # check that the filenames exist
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Read examples from files in the filename queue.
    read_input = read_wingnet(filename_queue)
    reshaped_image = tf.cast(read_input.image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width,
                                                           height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # convert the label to int32
    read_input.label = tf.cast(tf.string_to_number(read_input.label, tf.int32),
                               tf.int32)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 1])
    # read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=False)
