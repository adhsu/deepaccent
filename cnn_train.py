import resource
import os.path
from datetime import datetime
import time
import numpy as np
import tensorflow as tf

import cnn
from config import config
from utils import log

def train():
  with tf.Graph().as_default():
    
    log('===== START TRAIN RUN: ' + str(datetime.now()) + '=====')
    
    global_step = tf.Variable(0, trainable=False)
    
    # get examples and labels
    examples, labels = cnn.inputs(data_type='train')

    # build graph to compute logits
    logits = cnn.inference(examples)

    # compute loss
    loss, losses_collection = cnn.loss(logits, labels)
    accuracy = cnn.accuracy(logits, labels)

    # train model with one batch of examples
    train_op = cnn.train(loss, global_step)

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
    train_writer = tf.train.SummaryWriter(config.train_dir, sess.graph)
    
    for step in xrange(config.max_steps):
      
      start_time = time.time()
      summary, loss_value, accuracy_value, _ = sess.run([summary_op, loss, accuracy, train_op])

      loss_breakdown = [(str(l.op.name), sess.run(l)) for l in losses_collection]
        
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % config.summary_every_n_steps == 0: # summaries
        
        examples_per_sec = config.batch_size / duration
        sec_per_batch = float(duration)
        
        train_writer.add_summary(summary, step)

        log_str_1 = ('%s: step %d, loss = %.3f (%.2f examples/sec; %.3f sec/batch), accuracy %.3f   ') % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch, accuracy_value)
        # log_str_1 += str(loss_breakdown) # print loss breakdown
        log(log_str_1)

        log("memory usage: {} Mb".format(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000.0))
        

      if (step % config.ckpt_every_n_steps == 0) and (step>0): # save weights to file & validate
        checkpoint_path = os.path.join(config.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        log("Checkpoint saved at step %d" % step)


def main(_):  
  train()

if __name__ == '__main__':
  tf.app.run()
