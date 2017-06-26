
"""Builds the WingNet network, based on the CIFAR10 network built in the
TensorFlow tutorial. # TODO: Add website reference.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

import wingnet_input

# Basic model parameters.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './data/',
                           """Path to the WingNet data directory.""")
tf.app.flags.DEFINE_string('train_data', './data/',
                           """Path to the WingNet label directory.""")
tf.app.flags.DEFINE_string('train_labels', './data/training_labels.csv',
                           """Path to the WingNet training labels file.""")
tf.app.flags.DEFINE_string('test_labels', './data/testing_labels.csv',
                           """Path to the WingNet testing labels file.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")



# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = wingnet_input.IMAGE_SIZE
NUM_CLASSES = wingnet_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = wingnet_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = wingnet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1000000.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE =1.5e-3       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Arguments
        1. x: Tensor

    Returns
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Arguments
        1. name: name of the variable
        2. shape: list of ints
        3. initializer: initializer for Variable

    Returns:
        1. Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name,
                              shape,
                              initializer = initializer,
                              dtype = dtype)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Arguments
        1. name: name of the variable
        2. shape: list of ints
        3. stddev: standard deviation of a truncated Gaussian
        4. wd: add L2Loss weight decay multiplied by this float. If None, weight
               decay is not added for this Variable.

    Returns
        1. Variable Tensor
    """

    var = _variable_on_cpu(name,
                           shape,
                           tf.truncated_normal_initializer(stddev=stddev,
                                                           dtype=tf.float32))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def _randomize_data_inputs_7_labels(batch_size, labels_and_files):
    zero_idx = np.concatenate((np.where(labels_and_files[:, 0] == "0")[0],
                               np.where(labels_and_files[:, 0] == "2")[0]))

    one_idx = np.concatenate((np.where(labels_and_files[:, 0] == "1")[0],
                              np.where(labels_and_files[:, 0] == "3")[0]))


    labels_0 = labels_and_files[zero_idx, :]
    labels_1 = labels_and_files[one_idx, :]


    # counts and results
    total = labels_and_files.shape[0]
    zero = zero_idx.shape[0]
    per_zero = round(float(zero) / float(total), 2)
    ones = one_idx.shape[0]
    per_ones = 1.0 - per_zero

    # get batch counts
    per_batch_0 = int(batch_size * per_zero)
    per_batch_1 = batch_size - per_batch_0

    # loop until the arrays are gone
    # TODO: Add seed functionality so an experiment can be repeated
    batches = []
    while labels_0.shape[0] > 0 or labels_1.shape[0] > 0:
            temp_batch = []
            range_labels_0 = labels_0.shape[0]
            range_labels_1 = labels_1.shape[0]

            # do we have any zero labels remaining?
            if range_labels_0 > 0:

                    # do we have enough labels to fill a batch?
                    if range_labels_0 > per_batch_0:
                            random_idx_0 = np.random.choice(a = range_labels_0, size = per_batch_0)

                    else:
                            # if not, grab the remaining labels
                            random_idx_0 = np.arange(range_labels_0)

                    # put those labels into the temp_batch list
                    temp_batch.append(labels_0[random_idx_0, :])

                    # delete those elements from the labels_0 array
                    labels_0 = np.delete(arr = labels_0, obj = random_idx_0, axis = 0)


            # do we have any one labels remaining?
            if range_labels_1 > 0:

                    # do we have enough labels to fill a batch?
                    if range_labels_1 > per_batch_1:
                            random_idx_1 = np.random.choice(a = range_labels_1, size = per_batch_1)

                    else:
                            # if not, grab the remaining labels
                            random_idx_1 = np.arange(range_labels_1)

                    # put those labels into the temp_batch list
                    temp_batch.append(labels_1[random_idx_1, :])

                    # delete those elements from the labels_1 array
                    labels_1 = np.delete(arr = labels_1, obj = random_idx_1, axis = 0)

            # convert temp_batch to a numpy array and shuffle rows
            temp_batch = np.vstack(temp_batch)
            np.random.shuffle(temp_batch)

            # add to the main list of batches
            batches.append(temp_batch)

    # convert to numpy array and return
    batches = np.vstack(batches)
    return(batches)

def _randomize_data_inputs(batch_size, labels_and_files):
    labels_0 = labels_and_files[np.where(labels_and_files[:, 0] == "0")[0], :]
    labels_1 = labels_and_files[np.where(labels_and_files[:, 0] == "1")[0], :]

    # counts and results
    total = labels_and_files.shape[0]
    zero = labels_0.shape[0]
    per_zero = round(float(zero) / float(total), 2)
    ones = labels_1.shape[0]
    per_ones = 1.0 - per_zero

    # get batch counts
    per_batch_0 = int(batch_size * per_zero)
    per_batch_1 = batch_size - per_batch_0

    # loop until the arrays are gone
    # TODO: Add seed functionality so an experiment can be repeated
    batches = []
    while labels_0.shape[0] > 0 or labels_1.shape[0] > 0:
            temp_batch = []
            range_labels_0 = labels_0.shape[0]
            range_labels_1 = labels_1.shape[0]

            # do we have any zero labels remaining?
            if range_labels_0 > 0:

                    # do we have enough labels to fill a batch?
                    if range_labels_0 > per_batch_0:
                            random_idx_0 = np.random.choice(a = range_labels_0, size = per_batch_0)

                    else:
                            # if not, grab the remaining labels
                            random_idx_0 = np.arange(range_labels_0)

                    # put those labels into the temp_batch list
                    temp_batch.append(labels_0[random_idx_0, :])

                    # delete those elements from the labels_0 array
                    labels_0 = np.delete(arr = labels_0, obj = random_idx_0, axis = 0)


            # do we have any one labels remaining?
            if range_labels_1 > 0:

                    # do we have enough labels to fill a batch?
                    if range_labels_1 > per_batch_1:
                            random_idx_1 = np.random.choice(a = range_labels_1, size = per_batch_1)

                    else:
                            # if not, grab the remaining labels
                            random_idx_1 = np.arange(range_labels_1)

                    # put those labels into the temp_batch list
                    temp_batch.append(labels_1[random_idx_1, :])

                    # delete those elements from the labels_1 array
                    labels_1 = np.delete(arr = labels_1, obj = random_idx_1, axis = 0)

            # convert temp_batch to a numpy array and shuffle rows
            temp_batch = np.vstack(temp_batch)
            np.random.shuffle(temp_batch)

            # add to the main list of batches
            batches.append(temp_batch)

    # convert to numpy array and return
    batches = np.vstack(batches)
    return(batches)


def distorted_inputs():
    """Construct distorted input for WingNet training.

    Returns
        1. images: Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        2. labels: Labels. 1-D tensor of [batch_size] size.

    Raises:
        ValueError: If no train_data
    """

    if not FLAGS.train_data:
        raise ValueError('Please supply a train_data directory')

    # set the image data directory; load the labels;
    data_dir = os.path.join(FLAGS.train_data, 'train_images')
    labels_and_files = np.genfromtxt(FLAGS.train_labels,
                                     delimiter=",",
                                     dtype=str)

    if NUM_CLASSES == 2:
        zero_idx = np.concatenate((np.where(labels_and_files[:, 0] == "0")[0],
                                   np.where(labels_and_files[:, 0] == "2")[0],
                                   np.where(labels_and_files[:, 0] == "4")[0],
                                   np.where(labels_and_files[:, 0] == "6")[0]))

        one_idx = np.concatenate((np.where(labels_and_files[:, 0] == "1")[0],
                                  np.where(labels_and_files[:, 0] == "3")[0],
                                  np.where(labels_and_files[:, 0] == "5")[0]))


        labels_and_files[zero_idx, 0] = "0"
        labels_and_files[one_idx, 0]  = "1"
        labels_and_files = _randomize_data_inputs(batch_size=FLAGS.batch_size,
                                                  labels_and_files=labels_and_files)
    else:
        labels_and_files = _randomize_data_inputs_7_labels(batch_size=FLAGS.batch_size,
                                                           labels_and_files=labels_and_files)

    # get the distorted inputs from the reader
    images, labels = wingnet_input.distorted_inputs(labels_and_files,
                                                    data_dir = data_dir,
                                                    batch_size = FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images, labels


def inputs(eval_data):
    """Construct input for WingNet evaluation using the Reader ops.

    Arguments
        1. eval_data: bool, indicating if one should use the train or eval data set.

    Returns
        1. images: Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        2. labels: Labels. 1-D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    # TODO: Generalize the FLAGs so the user inputs training / testing
    # information separately.
    data_dir = os.path.join(FLAGS.data_dir, 'eval_images')
    labels_and_files = np.loadtxt(FLAGS.test_labels,
                                  delimiter=",",
                                  dtype=str)
    images, labels = wingnet_input.inputs(eval_data=eval_data,
                                          labels_and_files=labels_and_files,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images, labels


def inference(images):
    """Build the WingNet model.  Currently based on the CIFAR-10 model.

    Arguments
    1. images: Images returned from distorted_inputs() or inputs().

    Returns
    1. Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # Conv 1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 1, 64],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool1,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')

        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # CONV 3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 384],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool2,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

    # pool3
    pool3 = tf.nn.max_pool(conv3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # CONV 4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                              shape=[3, 3, 384, 384],
                                              stddev=0.04,
                                              wd=0.004)
        conv = tf.nn.conv2d(pool3,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')

        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4)

    # pool4
    pool4 = tf.nn.max_pool(conv4,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # CONV 5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights',
                                              shape=[1, 1, 384, 192],
                                              stddev=0.04,
                                              wd=0.004)
        conv = tf.nn.conv2d(pool4,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')

        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')


    # CONV 6 -- LAST LAYER
    with tf.variable_scope('conv6') as scope:
        kernel = _variable_with_weight_decay('weights',
                                              shape=[1, 1, 192, NUM_CLASSES],
                                              stddev=0.04,
                                              wd=0.004)
        conv = tf.nn.conv2d(pool5,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')

        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        softmax_conv = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(softmax_conv)

    # reshape to 1 x NUM_CLASSES and return
    softmax_conv = tf.reshape(softmax_conv, shape=[-1, NUM_CLASSES])
    return softmax_conv


def loss(logits, labels):
    """Add L2-Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Arguments
        1. logits: Logits from inference().
        2. labels: Labels from distorted_inputs or inputs(). 1-D tensor
                   of shape [batch_size]

    Returns
        1. Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses',
                         cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in WingNet model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Arguments
        1. total_loss: Total loss from loss().

    Returns
        1. loss_averages_op: op for generating moving averages of losses.
    """

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9,
                                                      name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train WingNet model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Arguments
        1. total_loss: Total loss from loss().
        2. global_step: Integer Variable counting the number of training steps
                        processed.
    Returns
        1. train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,
        global_step
    )
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
