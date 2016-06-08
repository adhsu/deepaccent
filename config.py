import os

class Config(object):
  def __init__(self, d):
    self.__dict__ = d


config = Config({
  'name': '1_c32c64fc256_wd1e-3',
  'summary_every_n_steps': 1,
  'ckpt_every_n_steps': 3, # checkpoint and validate

  #### MODEL
  'num_classes': 2,
  'example_height': 300, # time/frames
  'example_width': 42, # frequency
  'conv1_filters': 16,
  'conv2_filters': 16,
  'all_fc_size': 128,
  'fc_wd': 1e-3, # fc layer weight decay
  
  #### TRAINING
  'batch_size': 128, # 26.5k steps/epoch at 128 batch_size
  'lr_initial': 1e-3,
  'lr_decay_factor': 0.1,
  'n_epochs_per_decay': 3.0,
  'max_steps': 1000000,
  'num_examples_per_epoch_train': 3390000, # 678 bins * 5000/bin
  'moving_average_decay': 0.9999,
  
  #### EVAL
  'eval_interval_secs': 60*.5, # how often to run eval
  'num_examples_per_epoch_eval': 110000 # 22 bins * 5000/bin
})


