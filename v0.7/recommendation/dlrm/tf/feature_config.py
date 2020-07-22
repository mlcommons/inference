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
"""Configuration information for sparse features and tables."""

import math

import tensorflow.compat.v1 as tf

from tensorflow.python.tpu import tpu_embedding


LABEL_FEATURE = "clicked"
INT_FEATURES = ["int-feature-%d" % x for x in range(1, 14)]
CATEGORICAL_FEATURES = [
    "categorical-feature-%d" % x for x in range(14, 40)
]
FAKE_DATA_VOCAB_SIZE = 1000
FAKE_DATA_INT_MAX = 10.0


class FeatureConfig(object):
  """Configure dense and sparse features.

  The embedding tables can be placed either in EmbeddingCore or .
  In practice, large embedding tables are placed in EmbeddingCore while small
  ones are in . FeatureConfig sorts the embedding table by its size
  and stores the metadata such that input (dataloader.py) and model (dlrm.py)
  see a consistent view of the placement of all embedding tables.

  """

  def __init__(self, params):
    """Init method."""
    # Hyperparameters
    self._batch_size = params["batch_size"]
    self._learning_rate = params["learning_rate"]
    self._lr_warmup_steps = params["lr_warmup_steps"]
    self._optimizer = params["optimizer"]
    self._decay_steps = params["decay_steps"]
    self._decay_start_steps = params["decay_start_step"]

    self._vse = params["vocab_sizes"]
    self._de = params["dim_embed"]
    self._num_dense_features = params["num_dense_features"]
    self._num_sparse_features = len(self._vse)

    self._num_tables_in_ec = params["num_tables_in_ec"]
    # Get the sorted table size in a descending order.
    self._table_size_sorted = sorted(self._vse)[::-1]
    # The following is equivalent to np.argsort in a descending order.
    self._table_idx_ordered_by_size = sorted(
        range(len(self._vse)), key=self._vse.__getitem__)[::-1]
    tf.logging.info(self._table_size_sorted)
    tf.logging.info(self._table_idx_ordered_by_size)

  def get_sorted_table_size(self):
    return self._table_size_sorted

  def get_table_idx_orderd_by_size(self):
    return self._table_idx_ordered_by_size

  def get_num_tables_in_ec(self):
    return self._num_tables_in_ec

  def get_num_dense_features(self):
    return self._num_dense_features

  def get_num_sparse_features(self):
    return self._num_sparse_features

  def get_feature_tbl_config(self):
    """Creates table configuration data structures.

    For all tables, vocab size and width are given by params.

    Table setup:
    tbl0 - categorical-feature-14
    tbl1 - categorical-feature-15
    ..

    Feature setup:
    categorical-feature-14 -- tbl0 (first sparse feature)
    categorical-feature-15 -- tbl1 (second sparse feature)

    Returns:
      A tuple of dicts, one for feature_to_config and one for table_to_config.
    """

    def lr_fn(global_step):
      """Learning function for the embeddings. Linear warmup and poly decay."""
      decay_exp = 2
      scal = self._batch_size / 2048
      adj_lr = self._learning_rate * scal
      if self._lr_warmup_steps == 0:
        return adj_lr
      if self._optimizer == "adagrad":
        return self._learning_rate
      warmup_lr = tf.cast(
          global_step, dtype=tf.float32) / self._lr_warmup_steps * adj_lr

      global_step = tf.cast(global_step, tf.float32)
      decay_steps = tf.cast(self._decay_steps, tf.float32)
      decay_start_step = tf.cast(self._decay_start_steps, tf.float32)
      steps_since_decay_start = global_step - decay_start_step
      already_decayed_steps = tf.minimum(steps_since_decay_start, decay_steps)
      decay_lr = adj_lr * (
          (decay_steps - already_decayed_steps) / decay_steps)**decay_exp
      decay_lr = tf.maximum(0.0000001, decay_lr)

      lr = tf.where(
          global_step < self._lr_warmup_steps, warmup_lr,
          tf.where(
              tf.logical_and(decay_steps > 0, global_step > decay_start_step),
              decay_lr, adj_lr))

      return lr

    table_to_config_dict = {}
    for i in range(self._num_tables_in_ec):
      vocab_size = self._table_size_sorted[i]
      table_to_config_dict["tbl%02d" % i] = tpu_embedding.TableConfig(
          vocabulary_size=vocab_size,
          dimension=self._de,
          # NOTE: Default weight initializer uses trunc_normal,
          #       stddv=1/dimension. This is changed to match the mlperf
          #       reference model.
          initializer=tf.random_uniform_initializer(
              minval=-1 / math.sqrt(vocab_size),
              maxval=1 / math.sqrt(vocab_size)),
          combiner=None,
          learning_rate_fn=lr_fn,

          # TODO(tayo): Using the utils lr_fn leads to problems with embedding
          # table size. The embedding table stops being able to fit.
          # learning_rate_fn=functools.partial(utils.lr_fn, params)
      )

    # Use an offset to allow the categorical feature numbering to be subsequent
    # to the integer feature numbering.
    offset = 1 + self._num_dense_features
    feature_to_config_dict = {}
    feature_to_config_dict.update([
        ("categorical-feature-%02d" % i,
         tpu_embedding.FeatureConfig(table_id="tbl%02d" % (i - offset)))
        for i in range(offset, offset + self._num_tables_in_ec)
    ])

    return feature_to_config_dict, table_to_config_dict
