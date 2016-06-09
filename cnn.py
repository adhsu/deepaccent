import tensorflow as tf
import numpy as np
import math 
from datetime import datetime
import time
import os
import re
import sys
from utils import _variable, conv2d, _activation_summary, BatchNorm, log
import cnn_input
from config import config
FLAGS = tf.app.flags.FLAGS

# CONSTANTS
tf.app.flags.DEFINE_string('env', 'dev', """either string 'dev' or 'prod'""")
filepath = os.path.dirname(os.path.abspath(__file__))
if FLAGS.env=='dev':
  DATA_DIR = os.path.join(filepath, 'tmp/data')
elif FLAGS.env=='prod':
  DATA_DIR = '/mnt/deepaccent-data'

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
BATCH_SIZE = config.batch_size # minibatch size


# Constants describing the training process.
MOVING_AVERAGE_DECAY = config.moving_average_decay    # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = config.n_epochs_per_decay      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = config.lr_decay_factor  # Learning rate decay factor.
INITIAL_LEARNING_RATE = config.lr_initial       # Initial learning rate.


DIM_TIME = config.example_height
DIM_FREQ = config.example_width

CONV1_FILTERS = config.conv1_filters
CONV1_HEIGHT = 9 # time
CONV1_WIDTH = 9 # freq

POOL1_HEIGHT = 3 # time
POOL1_WIDTH = 3 # freq
POOL1_STRIDE_HEIGHT = 3
POOL1_STRIDE_WIDTH = 3

CONV2_FILTERS = config.conv2_filters
CONV2_HEIGHT = 3 # time
CONV2_WIDTH = 4 # freq

POOL2_HEIGHT = 2 # time
POOL2_WIDTH = 2 # freq
POOL2_STRIDE_HEIGHT = 2
POOL2_STRIDE_WIDTH = 2

FC3_SIZE = config.all_fc_size
FC4_SIZE = config.all_fc_size
FC5_SIZE = config.all_fc_size
FC6_SIZE = config.all_fc_size
NUM_CLASSES = config.num_classes

# NETWORK
def inputs(data_type='train'):
  return cnn_input.inputs(data_type=data_type, data_dir=DATA_DIR, batch_size=BATCH_SIZE)

def inference(examples):

  # CONV1
  with tf.variable_scope('conv1') as scope:
    # conv weights [filter_height, filter_width, filter_depth, num_filters]
    kernel = _variable('weights', [CONV1_HEIGHT, CONV1_WIDTH, 1, CONV1_FILTERS], tf.contrib.layers.xavier_initializer_conv2d())
    biases = _variable('biases', [CONV1_FILTERS], tf.constant_initializer(0.1))

    conv = conv2d(examples, kernel)
    conv1 = tf.nn.relu(conv + biases, name=scope.name)
    _activation_summary(conv1)

  # pool1 dim: [n, time, freq after pooling, num_filters]
  pool1 = tf.nn.max_pool(conv1, ksize=[1, POOL1_HEIGHT, POOL1_WIDTH, 1], 
    strides=[1, POOL1_STRIDE_HEIGHT, POOL1_STRIDE_WIDTH, 1], padding='SAME', name='pool1')

  ## TODO: add batch norm 1 here
  batch_norm1_object = BatchNorm(name='batch_norm1', shape=[CONV1_FILTERS])
  batch_norm1 = batch_norm1_object(pool1)

  # CONV2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable('weights', [CONV2_HEIGHT, CONV2_WIDTH, CONV1_FILTERS, CONV2_FILTERS], tf.contrib.layers.xavier_initializer_conv2d())
    biases = _variable('biases', [CONV2_FILTERS], tf.constant_initializer(0.1))

    conv = conv2d(batch_norm1, kernel)
    conv2 = tf.nn.relu(conv + biases, name=scope.name)
    _activation_summary(conv2)

  # POOL2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, POOL2_HEIGHT, POOL2_WIDTH, 1], 
    strides=[1, POOL2_STRIDE_HEIGHT, POOL2_STRIDE_WIDTH, 1], padding='SAME', name='pool2')

  ## TODO: add batch norm 2 here
  batch_norm2_object = BatchNorm(name='batch_norm2', shape=[CONV2_FILTERS])
  batch_norm2 = batch_norm2_object(pool2)

  # FC3
  with tf.variable_scope('fc3') as scope:
    reshape = tf.reshape(batch_norm2, [BATCH_SIZE, -1])
    dim = (DIM_TIME/POOL1_HEIGHT/POOL2_HEIGHT) * (DIM_FREQ/POOL1_WIDTH/POOL2_WIDTH) * CONV2_FILTERS
    weights = _variable('weights', [dim, FC3_SIZE], tf.contrib.layers.xavier_initializer(), wd=config.fc_wd)
    biases = _variable('biases', [FC3_SIZE], tf.constant_initializer(0.1))

    fc3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc3)

  # FC4
  with tf.variable_scope('fc4') as scope:
    weights = _variable('weights', [FC3_SIZE, FC4_SIZE], tf.contrib.layers.xavier_initializer(), wd=config.fc_wd)
    biases = _variable('biases', [FC4_SIZE], tf.constant_initializer(0.1))

    fc4 = tf.nn.relu(tf.matmul(fc3, weights) + biases, name=scope.name)
    _activation_summary(fc4)

  # # FC5
  # with tf.variable_scope('fc5') as scope:
  #   weights = _variable('weights', [FC4_SIZE, FC5_SIZE], tf.contrib.layers.xavier_initializer(), wd=config.fc_wd)
  #   biases = _variable('biases', [FC5_SIZE], tf.constant_initializer(0.1))

  #   fc5 = tf.nn.relu(tf.matmul(fc4, weights) + biases, name=scope.name)
  #   _activation_summary(fc5)

  # # FC6
  # with tf.variable_scope('fc6') as scope:
  #   weights = _variable('weights', [FC5_SIZE, FC6_SIZE], tf.contrib.layers.xavier_initializer(), wd=config.fc_wd)
  #   biases = _variable('biases', [FC6_SIZE], tf.constant_initializer(0.1))

  #   fc6 = tf.nn.relu(tf.matmul(fc5, weights) + biases, name=scope.name)
  #   _activation_summary(fc6)

  # softmax
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable('weights', [FC6_SIZE, NUM_CLASSES], tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    # shape of y_conv is (N,3)
    softmax_linear = tf.add(tf.matmul(fc4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
  return softmax_linear

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64) # required int64
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  
  tf.add_to_collection('losses', cross_entropy_mean)

  # tf.scalar_summary('cross_entropy_loss', cross_entropy_mean)

  # total_loss = cross_entropy loss + weight decay L2 loss
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return total_loss, tf.get_collection('losses')

def accuracy(logits, labels):
  logits_argmax = tf.cast(tf.argmax(logits,1), tf.int32)
  correct_prediction = tf.equal(logits_argmax, labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary("accuracy", accuracy)  
  return accuracy


def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op



def train(total_loss, global_step):
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

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
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      update = grad*lr
      tf.histogram_summary(var.op.name + '/updates', update)
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  # with tf.control_dependencies([variables_averages_op]):
  #   train_op = tf.train.AdamOptimizer(lr, name='train').minimize(total_loss)

  return train_op

def main(_):
  if not FLAGS.env:
    msg = ('env flag must be specified. Either dev or prod.')
    print(msg)
    return -1

  train()

if __name__ == '__main__':
  tf.app.run()