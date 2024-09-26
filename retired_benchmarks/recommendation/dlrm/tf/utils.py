# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for configuring training DLRM training script."""

from absl import flags
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

# Metric names.
ACC_KEY = "accuracy"
AUC_KEY = "auc"
PRECISION_KEY = "precision"
RECALL_KEY = "recall"


def create_scalar_summary(name, simple_value):
  return tf.Summary(
      value=[tf.Summary.Value(tag=name, simple_value=simple_value)])


def train_loop_iters():

  def _ceil(n, d):
    return (n + d - 1) // d

  return _ceil(FLAGS.train_steps, FLAGS.steps_between_evals)


def lr_fn(params, global_step):
  """Calculates adjusted LR based on global step.

  Linear warmup and polynomial decay.

  Args:
    params: Params dict for the model.
    global_step: Variable representing the current step.

  Returns:
    New learning rate tensor (float32).
  """
  decay_exp = 2
  base_learning_rate = params["learning_rate"]
  global_step = tf.cast(global_step, tf.float32)
  lr_warmup_steps = tf.constant(params["lr_warmup_steps"], tf.float32)
  decay_steps_float = tf.constant(params["decay_steps"], tf.float32)
  decay_start_step_float = tf.constant(params["decay_start_step"], tf.float32)
  global_batch_size = params["batch_size"]
  scaling_factor = global_batch_size / 2048.0
  adjusted_lr = base_learning_rate * scaling_factor
  adjusted_lr = tf.constant(adjusted_lr, tf.float32)
  if not params["lr_warmup_steps"]:
    return adjusted_lr

  change_rate = adjusted_lr / lr_warmup_steps
  warmup_lr = adjusted_lr - (lr_warmup_steps - global_step) * change_rate

  steps_since_decay_start_float = global_step - decay_start_step_float
  already_decayed_steps = tf.minimum(steps_since_decay_start_float,
                                     decay_steps_float)
  decay_lr = adjusted_lr * ((decay_steps_float - already_decayed_steps) /
                            decay_steps_float)**decay_exp
  decay_lr = tf.maximum(decay_lr, tf.constant(0.0000001))

  is_warmup_step = tf.cast(global_step < lr_warmup_steps, tf.float32)
  is_decay_step = tf.cast(global_step > decay_start_step_float, tf.float32)
  is_middle_step = tf.cast(
      tf.equal(is_warmup_step + is_decay_step, 0.0), tf.float32)

  lr = (is_warmup_step * warmup_lr + is_middle_step * adjusted_lr +
        is_decay_step * decay_lr)
  return lr
