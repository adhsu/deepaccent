import tensorflow as tf
import numpy as np
import math 
import datetime 
import os
import re
import sys
import moving_averages

total_num_params = 0

def _variable(name, shape, initializer, wd=None):
  var = tf.get_variable(name, shape, initializer=initializer)

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def weight_variable(shape):

  global total_num_params
  num_params = 1
  for n in shape:
    num_params *= n
  total_num_params += num_params
  print "Total params: {:,}".format(total_num_params)


  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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



def read_data_sets():

  class DataSets(object):
    pass

  data_sets = DataSets()

  def load_and_shape_data(data_filename):
    # load data from file and shape properly
    data = np.load(data_filename)
    return data.reshape(data.shape[0], data.shape[1] * data.shape[2])

  # am_images = load_and_shape_data('data/am.npy') # (N, 200 * 32), class 0
  # br_images = load_and_shape_data('data/br.npy') # (N, 200 * 32), class 1

  am_images = np.zeros((5000,300*42)) # (N,12600)
  br_images = np.zeros((5000,300*42))

  print am_images.shape[0], br_images.shape[0]
  
  # Make one-hot labels
  am_labels = np.zeros((am_images.shape[0],2))
  am_labels[:,0] = 1
  br_labels = np.zeros((br_images.shape[0],2))
  br_labels[:,1] = 1

  # combine images and labels 
  full_dataset = [np.append(am_images, br_images, axis=0), 
    np.append(am_labels, br_labels, axis=0)]

  def shuffle_and_split(data_set):

    split = (70,20,10) # % train, validation, test
    data_length = data_set[0].shape[0]
    
    # Shuffle
    random_idx = np.arange(data_length)
    np.random.shuffle(random_idx)
    data_set[0] = data_set[0][random_idx]
    data_set[1] = data_set[1][random_idx]

    training_section_end = int(math.floor(data_length * .7))
    validate_section_end = int(math.floor(data_length * .9))
    test_section_end = data_length

    split_sets = {}

    split_sets['train'] = [data_set[0][:training_section_end, :], data_set[1][:training_section_end, :]]
    split_sets['validate'] = [data_set[0][training_section_end:validate_section_end, :], data_set[1][training_section_end:validate_section_end, :]]
    split_sets['test'] = [data_set[0][validate_section_end:test_section_end, :], data_set[1][validate_section_end:test_section_end, :]]

    return split_sets

  
  split_sets = shuffle_and_split(full_dataset)

  data_sets.train = DataSet(split_sets['train'][0], split_sets['train'][1])
  print 'train 0 shape', split_sets['train'][0].shape
  data_sets.validate = DataSet(split_sets['validate'][0], split_sets['validate'][1])
  data_sets.test = DataSet(split_sets['test'][0], split_sets['test'][1])


  return data_sets 


class DataSet(object):

  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
    self.num_examples = images.shape[0]
    self.images = images 
    self.labels = labels
    self.index_in_epoch = 0
    self.epochs_completed = 0

  # @property
  # def labels(self):
  #     return self.labels
  
  # @property
  # def images(self):
  #     return self.images
  


  def next_batch(self, batch_size):
    
    # return dataset[0][0:n], dataset[1][0:n]

    start = self.index_in_epoch
    self.index_in_epoch += batch_size
    if self.index_in_epoch > self.num_examples:
      # epoch finished
      self.epochs_completed += 1
      # shuffle data
      perm = np.arange(self.num_examples)
      np.random.shuffle(perm)
      # shuffle that shit
      self.images = self.images[perm]
      self.labels = self.labels[perm]
      # start next epoch
      start = 0
      self.index_in_epoch = batch_size
      assert batch_size <= self.num_examples

    end = self.index_in_epoch

    # print INDEX_IN_EPOCH, EPOCHS_COMPLETED, start, end
    return self.images[start:end], self.labels[start:end]


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




def check_shape(tensor, name, shape):
  batch = data.train.next_batch(BATCH_SIZE)
  tensor_shape = sess.run(tf.shape(tensor), feed_dict={x: batch[0], y_: batch[1]})
  tensor_shape_value = tensor_shape.tolist()
  print name, ":", tensor_shape_value
  assert tensor_shape_value == shape
  if tensor_shape_value == shape:
    print '...correct'
  return tensor_shape_value

  # CHECK SHAPES
  # if CHECK_SHAPES:
  #   check_shape(x_image, 'x_image', [BATCH_SIZE, DIM_TIME, DIM_FREQ, 1])
  #   check_shape(conv1, 'conv1', [BATCH_SIZE, DIM_TIME, DIM_FREQ, CONV1_FILTERS])
  #   check_shape(pool1, 'pool1', [BATCH_SIZE, DIM_TIME/POOL1_HEIGHT, DIM_FREQ/POOL1_WIDTH, CONV1_FILTERS]) # height cut in half due to MP
  #   check_shape(conv2, 'conv2', [BATCH_SIZE, DIM_TIME/POOL1_HEIGHT, DIM_FREQ/POOL1_WIDTH, CONV2_FILTERS])
  #   check_shape(pool2, 'pool2', [BATCH_SIZE, DIM_TIME/POOL1_HEIGHT/POOL2_HEIGHT, DIM_FREQ/POOL1_WIDTH/POOL2_WIDTH, CONV2_FILTERS])
  #   check_shape(fc3, 'fc3', [BATCH_SIZE, FC3_SIZE])
  #   check_shape(fc4, 'fc4', [BATCH_SIZE, FC4_SIZE])
  #   check_shape(fc5, 'fc5', [BATCH_SIZE, FC5_SIZE])
  #   check_shape(fc6, 'fc6', [BATCH_SIZE, FC6_SIZE])
  #   check_shape(y_conv, 'y_conv', [BATCH_SIZE, NUM_CLASSES])