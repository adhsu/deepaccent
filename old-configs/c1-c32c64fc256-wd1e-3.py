import os

class Config(object):
  def __init__(self, d):
    self.__dict__ = d


config = Config({
  'name': 'c1-c32c64fc256-wd1e-3',
  'summary_every_n_steps': 1,
  'ckpt_every_n_steps': 100,

  #### MODEL
  'num_classes': 2,
  'example_height': 300, # time/frames
  'example_width': 42, # frequency
  'conv1_filters': 32,
  'conv2_filters': 64,
  'all_fc_size': 256,
  'fc_wd': 1e-3, # fc layer weight decay
  
  #### TRAINING
  'batch_size': 128,
  'lr_initial': 1e-3,
  'lr_decay_factor': 0.1,
  'n_epochs_per_decay': 3.0,
  'max_steps': 1000000,
  'num_examples_per_epoch_train': 50000,
  'moving_average_decay': 0.9999,
  
  #### EVAL
  'eval_interval_secs': 60*.5, # how often to run eval
  'num_examples_per_epoch_eval': 5000,
  'run_once': False
})

