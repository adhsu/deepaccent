import tensorflow as tf
import numpy as np
import math 
from datetime import datetime
import time
import os
import re
import sys
from utils import _variable, weight_variable, bias_variable, conv2d, read_data_sets, _activation_summary, BatchNorm
import cnn_input
from config import config
FLAGS = tf.app.flags.FLAGS

# CONSTANTS
tf.app.flags.DEFINE_string('env', 'dev', """either string 'dev' or 'prod'""")

if FLAGS.env=='dev':
  DATA_DIR = 'tmp/cnn/data'
elif FLAGS.env=='prod':
  DATA_DIR = '/mnt/deepaccent-data'

BATCH_SIZE = config.batch_size # minibatch size

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

# LOGGING
def log():
  log_str = ('BATCH_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, FC3_SIZE %d, FC4_SIZE %d, FC5_SIZE %d, FC6_SIZE %d') % (BATCH_SIZE, CONV1_FILTERS, CONV2_FILTERS, FC3_SIZE, FC4_SIZE, FC5_SIZE, FC6_SIZE)
  return log_str

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
    weights = _variable('weights', [dim, FC3_SIZE], tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [FC3_SIZE], tf.constant_initializer(0.1))

    fc3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc3)

  # FC4
  with tf.variable_scope('fc4') as scope:
    weights = _variable('weights', [FC3_SIZE, FC4_SIZE], tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [FC4_SIZE], tf.constant_initializer(0.1))

    fc4 = tf.nn.relu(tf.matmul(fc3, weights) + biases, name=scope.name)
    _activation_summary(fc4)

  # FC5
  with tf.variable_scope('fc5') as scope:
    weights = _variable('weights', [FC4_SIZE, FC5_SIZE], tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [FC5_SIZE], tf.constant_initializer(0.1))

    fc5 = tf.nn.relu(tf.matmul(fc4, weights) + biases, name=scope.name)
    _activation_summary(fc5)

  # FC6
  with tf.variable_scope('fc6') as scope:
    weights = _variable('weights', [FC5_SIZE, FC6_SIZE], tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [FC6_SIZE], tf.constant_initializer(0.1))

    fc6 = tf.nn.relu(tf.matmul(fc5, weights) + biases, name=scope.name)
    _activation_summary(fc6)

  # softmax
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable('weights', [FC6_SIZE, NUM_CLASSES], tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    # shape of y_conv is (N,3)
    softmax_linear = tf.add(tf.matmul(fc6, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
  return softmax_linear

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64) # required int64
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.scalar_summary('cross_entropy_loss', cross_entropy_mean)
  return cross_entropy_mean

def accuracy(logits, labels):
  logits_argmax = tf.cast(tf.argmax(logits,1), tf.int32)
  correct_prediction = tf.equal(logits_argmax, labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary("accuracy", accuracy)  
  return accuracy


def train(loss):
  # returns train_op
  lr = config.lr
  train_op = tf.train.AdamOptimizer(lr).minimize(loss)
  return train_op

def main(_):
  if not FLAGS.env:
    msg = ('env flag must be specified. Either dev or prod.')
    logging.error(msg)
    print(msg)
    return -1

  train()

if __name__ == '__main__':
  tf.app.run()