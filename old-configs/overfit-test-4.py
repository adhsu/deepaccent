import os
import pprint
pp = pprint.PrettyPrinter(indent=2)

class Config(object):
  def __init__(self, d):
    self.__dict__ = d

  def __str__(self):
    pp_details = pp.pformat(self.__dict__)
    return pp_details

config_dict = {
  'name': 'overfit-test-4',
  'summary_every_n_steps': 1,
  'ckpt_every_n_steps': 100, # checkpoint and validate

  #### DATA
  'use_train_bins': [126],
  'use_test_bins': [128],

  #### MODEL
  'num_classes': 2,
  'example_height': 300, # time/frames
  'example_width': 42, # frequency
  'conv1_filters': 128,
  'conv2_filters': 128,
  'all_fc_size': 768,
  'fc_wd': 0.00, # fc layer weight decay
  
  #### TRAINING
  'batch_size': 128, # 26.5k steps/epoch at 128 batch_size
  'lr_initial': 1e-3,
  'lr_decay_factor': 0.1,
  'n_epochs_per_decay': 3.0,
  'max_steps': 1000000,
  # 'num_examples_per_epoch_train': 5000, # 678 bins * 5000/bin
  'moving_average_decay': 0.9999,
  
  #### EVAL
  'eval_interval_secs': 60*.5, # how often to run eval
  # 'num_examples_per_epoch_eval': 5000 # 22 bins * 5000/bin
}

filepath = os.path.dirname(os.path.abspath(__file__))
config_dict['train_dir'] = os.path.join(filepath, 'tmp/train', config_dict['name'])
config_dict['eval_dir'] = os.path.join(filepath, 'tmp/eval', config_dict['name'])
config_dict['checkpoint_dir'] = os.path.join(filepath, 'tmp/train', config_dict['name']) # read model checkpoints from here

config = Config(config_dict)
