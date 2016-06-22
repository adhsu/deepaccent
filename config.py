import tensorflow as tf
import sys
import os
import glob
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('env', 'dev', """either string 'dev' or 'prod'""")

if not FLAGS.env:
  msg = ("""env flag must be specified. Either 'dev' or 'prod'.""")
  print(msg)
  sys.exit(1)
elif FLAGS.env=='dev':
  DATA_DIR = './tmp/data'
  TRAIN_DIR_ROOT = './tmp'
  EVAL_DIR_ROOT = TRAIN_DIR_ROOT
  CHECKPOINT_DIR = './tmp/checkpoints'
elif FLAGS.env=='prod':
  DATA_DIR = '/home/connor/mnt/deepaccent-data'
  TRAIN_DIR_ROOT = '/home/connor/mnt/deepaccent-results'
  EVAL_DIR_ROOT = TRAIN_DIR_ROOT
  CHECKPOINT_DIR = '/home/connor/checkpoints'

class Config(object):
  def __init__(self):
    # nowTimeStr = datetime.strftime(datetime.now(), '%Y-%m-%d--%H-%M-%S')

    # Model options.
    self.summary_every_n_steps = 1
    self.ckpt_every_n_steps = 500 # checkpoint and validate

    #### DATA
    self.train_bins = [0] # leave empty to use all .bins
    self.test_bins = [1]
    self.num_classes = 2
    self.example_height = 300 # time/frames
    self.example_width = 42 # frequency

    #### MODEL ARCHITECTURE
    self.conv1_filters = 128
    self.conv2_filters = 256
    self.all_fc_size = 1024
    
    #### TRAINING
    self.batch_size = 64
    self.fc_wd = 0.00 # fc layer weight decay
    self.lr_initial = 1e-3
    self.lr_decay_factor = 0.0
    self.n_epochs_per_decay = 3.0
    self.max_steps = 1000000
    self.moving_average_decay = 0.9999
    
    #### EVAL
    self.eval_interval_secs = 60*.5 # how often to run eval

    self.name = 'overfit-c{}c{}fc{}-7'.format(self.conv1_filters, self.conv2_filters, self.all_fc_size)
    self.data_dir = DATA_DIR
    self.train_dir = os.path.join(TRAIN_DIR_ROOT, self.name, 'train')
    self.eval_dir = os.path.join(EVAL_DIR_ROOT, self.name, 'eval')
    self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, self.name)

    # calculate number of training examples
    if len(self.train_bins) > 0: # if bins are specified
      self.filenames_train = [os.path.join(self.data_dir, 'train_%d.bin' % i) for i in self.train_bins]
    else: # bins not specified, use all
      self.filenames_train = glob.glob(os.path.join(self.data_dir, 'train_*.bin'))

    # calculate number of test examples
    if len(self.test_bins) > 0: # if bins are specified
      self.filenames_test = [os.path.join(self.data_dir, 'test_%d.bin' % i) for i in self.test_bins]
    else: # bins not specified, use all
      self.filenames_test = glob.glob(os.path.join(self.data_dir, 'test_*.bin'))

    self.num_examples_train = sum([int(os.path.getsize(f)/50404) for f in self.filenames_train])
    self.num_examples_test = sum([int(os.path.getsize(f)/50404) for f in self.filenames_test])

    # make train and eval dirs if they don't exist
    if not tf.gfile.Exists(self.train_dir):
      tf.gfile.MakeDirs(self.train_dir)
    if not tf.gfile.Exists(self.eval_dir):
      tf.gfile.MakeDirs(self.eval_dir)
    if not tf.gfile.Exists(self.checkpoint_dir):
      tf.gfile.MakeDirs(self.checkpoint_dir)

    # print('name is ' + self.name)


config = Config()
