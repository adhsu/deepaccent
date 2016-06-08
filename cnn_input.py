import os
import tensorflow as tf
import glob
from config import config
from utils import log

# constants
NUM_CLASSES = config.num_classes
EXAMPLE_HEIGHT = config.example_height
EXAMPLE_WIDTH = config.example_width
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = config.num_examples_per_epoch_train
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = config.num_examples_per_epoch_eval

def read_cnn(filename_queue):
  
  class CNNRecord(object):
    pass
  result = CNNRecord()

  # each example in the file is ('megabatch') is label + (300,42) data in float32 (4 bytes per value)
  # (1+12600)*4 = 50404 bytes per example
  bytes_per_value = 4 # float32
  label_bytes = 1 * bytes_per_value
  result.height = EXAMPLE_HEIGHT
  result.width = EXAMPLE_WIDTH
  result.depth = 1
  example_bytes = result.height * result.width * result.depth * bytes_per_value
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + example_bytes
  # print 'record_bytes number is ', record_bytes # should be 4 + 300*42*4 = 50404
  # # Read a record, getting filenames from the filename_queue.  No
  # header or footer, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value_bytes = reader.read(filename_queue)

  # Convert from a string to a vector of float32 that is record_bytes long.
  value = tf.decode_raw(value_bytes, tf.float32)

  
  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.slice(value, [0], [label_bytes/bytes_per_value])
  # result.label = tf.cast(result_label_raw, tf.int32)
  
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  result.example = tf.slice(value, [label_bytes/bytes_per_value], [example_bytes/bytes_per_value])

  return result


def _generate_example_and_label_batch(example, label, min_queue_examples,
                                    batch_size, shuffle):
 
  num_preprocess_threads = 16
  if shuffle:
    examples, label_batch = tf.train.shuffle_batch(
        [example, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    examples, label_batch = tf.train.batch(
        [example, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training examples/images in the visualizer.
  tf.image_summary('examples', examples)

  return examples, tf.reshape(label_batch, [batch_size])

def inputs(data_type, data_dir, batch_size):
  
  if data_type == 'train':
    
    if len(config.use_train_bins)>0:
      log('train_bins specified in config...')
      filenames = [os.path.join(data_dir, 'train_%d.bin' % i)
                     for i in config.use_train_bins]
    else:
      filenames = glob.glob(data_dir + '/train_*.bin')

    # num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  
  elif data_type == 'test':
    if len(config.use_test_bins)>0:
      log('test_bins specified in config...')
      filenames = [os.path.join(data_dir, 'test_%d.bin' % i)
                     for i in config.use_test_bins]
    else:
      filenames = glob.glob(data_dir + '/test_*.bin')
    # num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  
  else:
    raise ValueError('inputs data_type not valid')

  total_examples = 0
  log('loading {} data...'.format(data_type))
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

    num_examples_f = os.path.getsize(f)/50404
    total_examples += num_examples_f
    log("{}: {} examples".format(f.split('/')[-1], num_examples_f))
  
  log("TOTAL: {} examples, ~{:.3f} hours of data \n---".format(total_examples, total_examples*3./3600.))
  num_examples_per_epoch = total_examples

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cnn(filename_queue)
  reshaped_example = tf.cast(tf.reshape(read_input.example, [EXAMPLE_HEIGHT, EXAMPLE_WIDTH, 1]), tf.float32)

  # Subtract off the mean and divide by the variance of the pixels.
  whitened_example = tf.image.per_image_whitening(reshaped_example)
  
  # cast labels to int32
  read_input.label = tf.cast(read_input.label, tf.int32)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4 # 135,600 examples for train, 4,400 for eval
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  log('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_example_and_label_batch(whitened_example, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
