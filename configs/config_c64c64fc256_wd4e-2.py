class Config(object):
  def __init__(self, d):
    self.__dict__ = d


config = Config({
  'name': 'c128c128fc512_wd4e-2',
  'summary_every_n_steps': 1,
  'ckpt_every_n_steps': 100,


  #### DATA
  'train_dir': 'tmp/cnn/train', # summaries + ckpts
  'eval_dir': 'tmp/cnn/eval', # eval summaries

  #### MODEL
  'num_classes': 2,
  'example_height': 300, # time/frames
  'example_width': 42, # frequency
  'conv1_filters': 64,
  'conv2_filters': 64,
  'all_fc_size': 256,
  'fc_wd': 0.04, # fc layer weight decay
  
  #### TRAINING
  'batch_size': 128,
  'lr': 1e-3,
  'max_steps': 1000000,
  'num_examples_per_epoch_train': 50000,
  'moving_average_decay': 0.9999,
  
  #### EVAL
  'eval_interval_secs': 60*.5, # how often to run eval
  'num_examples_per_epoch_eval': 5000,
  'run_once': False
})

