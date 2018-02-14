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

"""Builds the cifar-100 network.
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

import argparse
import os
import re
import sys
import tarfile
import random # just debugging tasks

from six.moves import urllib
import tensorflow as tf

import cifar100_input

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar100_data',
                    help='Path to the cifar-100 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()

# Global constants describing the cifar-100 data set.
IMAGE_SIZE = cifar100_input.IMAGE_SIZE
NUM_CLASSES = cifar100_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar100_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar100_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-100-binary')
    images, labels = cifar100_input.distorted_inputs(data_dir=data_dir,
                                                     batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-100-batches-bin')
    images, labels = cifar100_input.inputs(eval_data=eval_data,
                                           data_dir=data_dir,
                                           batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images):
    """Build the cifar-100 model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    
    ### INPUT A ###
    # conv1
    with tf.variable_scope('conv1A') as scope:
        kernel = _variable_with_weight_decay('Conv1weightsA',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('Conv1biasesA', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1A = tf.nn.relu(pre_activation, name=scope.name)

    ### INPUT B ###
    # conv1
    with tf.variable_scope('conv1B') as scope:
        kernel = _variable_with_weight_decay('Conv1weightsB',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('Conv1biasesB', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1B = tf.nn.relu(pre_activation, name=scope.name)
 
    # pool1
    pool1A = tf.nn.max_pool(conv1A, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1A')
    pool1B = tf.nn.max_pool(conv1B, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1B')
    # norm1
    norm1A = tf.nn.lrn(pool1A, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1A')
    norm1B = tf.nn.lrn(pool1B, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1B')
    _activation_summary(norm1A)
    _activation_summary(norm1B)
    

    ### CONV 2 ###
    Conv2kernel = _variable_with_weight_decay('Conv2weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
    Conv2biases = _variable_on_cpu('Conv2biases', [64], tf.constant_initializer(0.1))
    
    Conv2convA = tf.nn.conv2d(norm1A, Conv2kernel, [1, 1, 1, 1], padding='SAME')
    Conv2convB = tf.nn.conv2d(norm1B, Conv2kernel, [1, 1, 1, 1], padding='SAME')
    Conv2pre_activationA = tf.nn.bias_add(Conv2convA, Conv2biases)
    Conv2pre_activationB = tf.nn.bias_add(Conv2convB, Conv2biases)
    conv2A = tf.nn.relu(Conv2pre_activationA, name='conv2A')
    conv2B = tf.nn.relu(Conv2pre_activationB, name='conv2B')
    _activation_summary(conv2A)
    _activation_summary(conv2B)
    
    # norm2
    norm2A = tf.nn.lrn(conv2A, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2A')
    norm2B = tf.nn.lrn(conv2B, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2B')
    # pool2
    pool2A = tf.nn.max_pool(norm2A, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2A')
    pool2B = tf.nn.max_pool(norm2B, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2B')
    
    
    # local3
    Local3reshapeA = tf.reshape(pool2A, [FLAGS.batch_size, -1])
    Local3reshapeB = tf.reshape(pool2B, [FLAGS.batch_size, -1])
    Local3dim = Local3reshapeA.get_shape()[1].value
    Local3weights = _variable_with_weight_decay('Local3weights', shape=[Local3dim, 384], stddev=0.04, wd=0.004)
    Local3biases = _variable_on_cpu('Local3biases', [384], tf.constant_initializer(0.1))
    local3A = tf.nn.relu(tf.matmul(Local3reshapeA, Local3weights) + Local3biases, name='local3A')
    local3B = tf.nn.relu(tf.matmul(Local3reshapeB, Local3weights) + Local3biases, name='local3B')
    _activation_summary(local3A)
    _activation_summary(local3B)
    

    ### HIDDEN LAYERS ###
    Hidden1weights = _variable_with_weight_decay('Hidden1weights', shape=[384, 384], stddev=0.04, wd=0.004)
    Hidden1biases = _variable_on_cpu('Hidden1biases', [384], tf.constant_initializer(0.1))
    hidden1A = tf.nn.relu(tf.matmul(local3A, Hidden1weights) + Hidden1biases, name='hidden1A')
    hidden1B = tf.nn.relu(tf.matmul(local3B, Hidden1weights) + Hidden1biases, name='hidden1B')
    _activation_summary(hidden1A)
    _activation_summary(hidden1B)
    
    Hidden2weights = _variable_with_weight_decay('Hidden2weights', shape=[384, 384], stddev=0.04, wd=0.004)
    Hidden2biases = _variable_on_cpu('Hidden2biases', [384], tf.constant_initializer(0.1))
    hidden2A = tf.nn.relu(tf.matmul(hidden1A, Hidden2weights) + Hidden2biases, name='hidden2A')
    hidden2B = tf.nn.relu(tf.matmul(hidden1B, Hidden2weights) + Hidden2biases, name='hidden2B')
    _activation_summary(hidden2A)
    _activation_summary(hidden2B)
    

    # local4
    Local4weights = _variable_with_weight_decay('Local4weights', shape=[384, 192], stddev=0.04, wd=0.004)
    Local4biases = _variable_on_cpu('Local4biases', [192], tf.constant_initializer(0.1))
    local4A = tf.nn.relu(tf.matmul(hidden2A, Local4weights) + Local4biases, name='local4A')
    local4B = tf.nn.relu(tf.matmul(hidden2B, Local4weights) + Local4biases, name='local4B')
    _activation_summary(local4A)
    _activation_summary(local4B)
    
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.


    ### OUTPUT A ###
    with tf.variable_scope('softmax_linearA') as scope:
        weights = _variable_with_weight_decay('SoftmaxWeightsA', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('SoftmaxBiasesA', [NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linearA = tf.add(
            tf.matmul(local4A, weights), biases, name=scope.name)
        _activation_summary(softmax_linearA)
    
    ### OUTPUT B ###
    with tf.variable_scope('softmax_linearB') as scope:
        weights = _variable_with_weight_decay('SoftmaxWeightsB', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('SoftmaxBiasesB', [NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linearB = tf.add(
            tf.matmul(local4B, weights), biases, name=scope.name)
        _activation_summary(softmax_linearB)
        

    return (softmax_linearA, softmax_linearB)
    
    
def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in cifar-100 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
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
    """Train cifar-100 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
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
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-100-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
