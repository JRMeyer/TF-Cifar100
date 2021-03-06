# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-100 using a single GPU.
Accuracy:
cifar100_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar100_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-100
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar100

PARSER = cifar100.parser

PARSER.add_argument('--train_dir', type=str, default='/tmp/cifar100_train',
                    help='Directory where to write event logs and checkpoint.')

PARSER.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')

PARSER.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

PARSER.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')


def train():
    """Train CIFAR-100 for a number of steps."""
    output = open('output_data/output_' + str(time.time()) + '.txt', 'w')
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-100.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
          images, labels = cifar100.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        
        logitsA,logitsB = cifar100.inference(images)

        # Calculate loss.
        lossA = cifar100.loss(logitsA, labels)
        lossB = cifar100.loss(logitsB, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_opA = cifar100.train(lossA, global_step)
        train_opB = cifar100.train(lossB, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(lossA)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))
                    print((str(self._step) + '\t' +
                           str(loss_value) + '\n'), file=output)

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(lossA),
                   tf.train.NanTensorHook(lossB),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            
            file_writer = tf.summary.FileWriter('tb-logs/', mon_sess.graph)

            while not mon_sess.should_stop():
                print("stepA")
                mon_sess.run(train_opA)
                print("stepB")
                mon_sess.run(train_opB)
        output.close()


def main(argv=None):  # pylint: disable=unused-argument
    cifar100.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    FLAGS = PARSER.parse_args()
    tf.app.run()
