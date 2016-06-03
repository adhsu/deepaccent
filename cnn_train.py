from datetime import datetime
import os.path
import time
import logging
import numpy as np
import tensorflow as tf

import cnn
from config import config

# constants
BATCH_SIZE = config.batch_size
# TRAIN_DIR = os.path.join(config.train_dir, config.name+'-'+str(time.time()))
TRAIN_DIR = os.path.join(config.train_dir, config.name)
# VALIDATE_DIR = 'tmp/cnn/validate' # storing validation summaries
MAX_STEPS = config.max_steps

logging.basicConfig(filename='train.log', level=logging.INFO)


def train():
  with tf.Graph().as_default():
    # start logging
    log_str_0 = '===== start training run: ' + str(datetime.now()) + '====='
    logging.info(log_str_0)
    logging.info(cnn.log())

    global_step = tf.Variable(0, trainable=False)
    
    # get examples and labels
    examples, labels = cnn.inputs(data_type='train')

    # build graph to compute logits
    logits = cnn.inference(examples)

    # compute loss
    loss = cnn.loss(logits, labels)
    accuracy = cnn.accuracy(logits, labels)

    # train model with one batch of examples
    train_op = cnn.train(loss)

    # create saver
    saver = tf.train.Saver(tf.all_variables())
  
    # build summary and init op
    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

    # start session
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()
    sess.run(init_op)
    
    # start queue runners
    tf.train.start_queue_runners(sess=sess)

    # set up summary writers
    train_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)
    # validation_writer = tf.train.SummaryWriter(VALIDATE_DIR)
    
    # every 1 step: run train_step, add training summaries
    # every 10 steps: measure validation accuracy, write validation summaries
    for step in xrange(MAX_STEPS):
      
      start_time = time.time()
      summary, loss_value, accuracy_value, _ = sess.run([summary_op, loss, accuracy, train_op])
      duration = time.time() - start_time

      if step % 1 == 0: # summaries
        
        examples_per_sec = BATCH_SIZE / duration
        sec_per_batch = float(duration)
        
        train_writer.add_summary(summary, step)

        log_str_1 = ('step %d, loss = %.3f (%.2f examples/sec; %.3f sec/batch), accuracy %.3f') % (step, loss_value,
                             examples_per_sec, sec_per_batch, accuracy_value)

        print(log_str_1)
        logging.info(log_str_1)

      if (step % 10 == 0) and (step>0): # save weights to file
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')

        saver.save(sess, checkpoint_path, global_step=step)
        print "Checkpoint saved at step %d" % step


def main(_):
  # if tf.gfile.Exists(TRAIN_DIR):
  #   tf.gfile.DeleteRecursively(TRAIN_DIR)
  # tf.gfile.MakeDirs(TRAIN_DIR)
  
  train()

if __name__ == '__main__':
  tf.app.run()
