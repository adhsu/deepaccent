import tensorflow as tf
import numpy as np
import math 
import datetime 
import os
import re
import sys
import logging
import moving_averages
from config import config
TRAIN_DIR = config.train_dir

# start logging
if not tf.gfile.Exists(TRAIN_DIR):
  tf.gfile.MakeDirs(TRAIN_DIR)
logging.basicConfig(filename=os.path.join(TRAIN_DIR, 'train.log'), level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
  # print
  print(msg)
  # log to file
  logger.info(msg)


def _variable(name, shape, initializer, wd=None):
  var = tf.get_variable(name, shape, initializer=initializer)

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1], padding='SAME')

def _activation_summary(x):
  # create summaries for activations
  # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

# BatchNorm object from Cinjon
# If your convnet has N filter maps, then you'd make this with shape=[N].
# It would then expect the convnet output as the "audio" input.
# The two given values for decay and epsilon are fairly aggressive. If you're going to tune them, try going in the .99 and .01 direction first.
class BatchNorm(object):
  def __init__(self, name, shape):
    self.name = name
    self.decay = 0.999
    self.epsilon = 0.001
    with tf.name_scope(name + '/vars'):
      self.beta = tf.Variable(tf.zeros(shape), name='beta')
      self.moving_mean = tf.Variable(tf.zeros(shape), name='moving_mean',
                                     trainable=False)
      self.moving_variance = tf.Variable(tf.ones(shape),
                                         name='moving_variance',
                                         trainable=False)

  def __call__(self, audio, is_training=True):
    with tf.name_scope(self.name) as scope:
      control_inputs = []
      if is_training:
        mean, variance = tf.nn.moments(audio, [0, 1, 2])
        update_moving_mean = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.decay)
        update_moving_variance = moving_averages.assign_moving_average(
            self.moving_variance, variance, self.decay)
        control_inputs = [update_moving_mean, update_moving_variance]
      else:
        mean = self.moving_mean
        variance = self.moving_variance
      with tf.control_dependencies(control_inputs):
        return tf.nn.batch_normalization(
            audio, mean=mean, variance=variance, offset=self.beta,
            scale=None, variance_epsilon=self.epsilon, name=scope)


