class Config(object):
  def __init__(self, d):
    self.__dict__ = d


config = Config({
  'name': 'c32c64fc512_noreg',

  #### DATA
  'train_dir': 'tmp/cnn/train', # summaries + ckpts
  'eval_dir': 'tmp/cnn/eval', # eval summaries

  #### MODEL
  'num_classes': 2,
  'example_height': 300, # time/frames
  'example_width': 42, # frequency
  'conv1_filters': 32,
  'conv2_filters': 64,
  'all_fc_size': 512,
  
  #### TRAINING
  'batch_size': 128,
  'lr': 1e-3,
  'max_steps': 1000000,
  'num_examples_per_epoch_train': 50000,
  
  #### EVAL
  'eval_interval_secs': 60*.5, # how often to run eval
  'num_examples_per_epoch_eval': 5000,
  'run_once': False
})

