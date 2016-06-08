from datetime import datetime
import os.path
import math
import time
import numpy as np
import tensorflow as tf
import cnn
from config import config
filepath = os.path.dirname(os.path.abspath(__file__))


BATCH_SIZE = config.batch_size
EVAL_DIR = os.path.join(filepath, 'tmp/eval', config.name)
CHECKPOINT_DIR = os.path.join(filepath, 'tmp/train', config.name) # read model checkpoints from here
EVAL_INTERVAL_SECS = config.eval_interval_secs # how often to run eval
NUM_EXAMPLES = config.num_examples_per_epoch_eval

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('checkpoint found, global step is ' + str(global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(NUM_EXAMPLES / BATCH_SIZE)) # ~39
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * BATCH_SIZE # 5k
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op]) # (128,)
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = float(true_count) / float(total_sample_count)
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(run_once=False):
  with tf.Graph().as_default() as graph:
    
    examples, labels = cnn.inputs(data_type='test')
    logits = cnn.inference(examples)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(EVAL_DIR, graph)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if run_once==True:
        break
      time.sleep(EVAL_INTERVAL_SECS)


def main(_):
  # if tf.gfile.Exists(EVAL_DIR):
  #   tf.gfile.DeleteRecursively(EVAL_DIR)
  # tf.gfile.MakeDirs(EVAL_DIR)
  
  evaluate()

if __name__ == '__main__':
  tf.app.run()
