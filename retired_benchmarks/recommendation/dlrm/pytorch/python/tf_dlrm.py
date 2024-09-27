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
"""Model definition for the DLRM model."""

import math
import numpy as np

# import tensorflow.compat.v1 as tf  # ti
import tensorflow as tf

from tensorflow.compiler.tf2xla.python import xla
from tensorflow.contrib import  layers as contrib_layers
# import utils  # ti

import sys # ti

def rand_features(batch_size):
  """Emits random input features, used for testing."""
  features = {}
  pos_size = batch_size // 2
  neg_size = batch_size - pos_size
  features["clicked"] = tf.concat([
      tf.ones([pos_size, 1], dtype=tf.float32),
      tf.zeros([neg_size, 1], dtype=tf.float32)
  ], axis=0)
  features["int-features"] = tf.random.uniform(
      shape=(batch_size, 13),
      maxval=100)
  features["cat-features"] = tf.random.uniform(
      shape=(batch_size, 26),
      maxval=100,
      dtype=tf.int32)
  return features

def rand_features_np(batch_size, num_d, num_s, minsize):
  """Emits random input features, used for testing."""
  # features = {}
  # pos_size = batch_size // 2
  # neg_size = batch_size - pos_size
  # features["clicked"] = tf.concat([
  #     tf.ones([pos_size, 1], dtype=tf.float32),
  #     tf.zeros([neg_size, 1], dtype=tf.float32)
  # ], axis=0)
  features_int_np = np.random.randint(100,
      size=(batch_size, num_d)
  )
  features_cat_np = np.random.randint(minsize,
      size=(batch_size, num_s),
      dtype=np.int32
  )
  return features_int_np, features_cat_np


def dot_interact(concat_features, params=None):
  """Performs feature interaction operation between dense and sparse.

  Input tensors represent dense and sparse features.
  Pre-condition: The tensors have been stacked along dimension 1.

  Args:
    concat_features: Tensor of features with shape [B, n_features, feature_dim].
    params: Model params.

  Returns:
    activations: Tensor representing interacted features.
  """
  # batch_size = concat_features.shape[0]
  if not params:
    params = {}

  # Interact features, select lower-triangular portion, and re-shape.
  xactions = tf.matmul(concat_features, concat_features, transpose_b=True)
  tf.logging.info("Model_FN: xactions shape: %s", xactions.get_shape())
  ones = tf.ones_like(xactions)
  upper_tri_mask = tf.linalg.band_part(ones, 0, -1)
  feature_dim = xactions.shape[-1]

  if params["opt_skip_gather"]:
    upper_tri_bool = tf.cast(upper_tri_mask, tf.bool)
    activations = tf.where(
        condition=upper_tri_bool, x=tf.zeros_like(xactions), y=xactions)
    tf.logging.info("Model_FN: activations shape: %s", activations.get_shape())
    out_dim = feature_dim * feature_dim
  else:
    lower_tri_mask = ones - upper_tri_mask
    activations = tf.boolean_mask(xactions, lower_tri_mask)
    tf.logging.info("Model_FN: activations shape: %s", activations.get_shape())
    out_dim = feature_dim * (feature_dim - 1) // 2

  activations = tf.reshape(activations, (-1, out_dim))
  return activations


def logits_fn(features_int, features_cat, params):
  """Calculate predictions."""
  # tf.logging.info("Model_FN: Number of input features: %d", len(features))
  # for ft in sorted(features.keys()):
  #   tf.logging.info("Model_FN: Feature %s -- shape %s", ft,
  #                   features[ft].get_shape())

  reuse = False if params["is_training"] else True

  bot_mlp_input = features_int
  tf.logging.info("Model_FN: Bottom MLP input (int features) shape: %s",
                  bot_mlp_input.get_shape())
  mlp_dims_bottom = params["mlp_bottom"]

  for layer_idx in range(len(mlp_dims_bottom)):
    bot_mlp_input = tf.layers.dense(
        bot_mlp_input,
        mlp_dims_bottom[layer_idx],
        activation="relu",
        # ti: modules dont exist
        # kernel_initializer=tf.compat.v2.initializers.GlorotNormal(),
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
        # bias_initializer=tf.compat.v2.initializers.RandomNormal(
        #     mean=0.0, stddev=math.sqrt(1.0 / mlp_dims_bottom[layer_idx])),
        bias_initializer=tf.compat.v1.random_normal_initializer(
          mean=0.0, stddev=math.sqrt(1.0 / mlp_dims_bottom[layer_idx])),
        name="bottom_mlp_layer_%d" % layer_idx,
        reuse=reuse
        )
  bot_mlp_output = bot_mlp_input

  cat_features = []
  emb_tables = []
  # for feature_name, value in sorted(features.items()):
  #   if "categorical-feature" in feature_name:
  #     cat_features.append(value)
  num_s = params["num_sparse_features"]
  for j in range(num_s):
    emb_init = tf.random_uniform([params["vocab_sizes"][j], params["dim_embed"]], -1.0, 1.0)
    emb_matrix = tf.get_variable(
        name="embedding_table%02d" % j,
        dtype=tf.float32,
        trainable=True,
        initializer=emb_init)
    emb_tables.append(emb_matrix)

  for j in range(num_s):
    col = tf.slice(features_cat, [0, j], [-1, 1])
    ecol = tf.nn.embedding_lookup(emb_tables[j], col)
    ecol = tf.reshape(ecol, [-1, params["dim_embed"]])
    cat_features.append(ecol)

  # tc_features = []
  # if "tc-features" in features:
  #   # Compute offsets for single concatenated table.
  #   batch_size = features_tc.shape[0]
  #   num_tc_features = features_tc.shape[1]
  #   num_tables_in_ec = params["num_tables_in_ec"]
  #   tc_table_sizes = feature_config.get_sorted_table_size()[num_tables_in_ec:]  # params["vocab_sizes"]
  #   total_tbl_size = sum(tc_table_sizes)
  #   idx_offsets = [0] + list(np.cumsum(tc_table_sizes[:-1]))
  #   idx_offsets = tf.broadcast_to(
  #       tf.constant(idx_offsets), (batch_size, num_tc_features))
  #   idxs = idx_offsets + features_tc

  #   def _create_init_table():
  #     """Table initialization varies depending on the vocab size."""
  #     full_tbl = np.zeros(
  #         shape=(total_tbl_size, params["dim_embed"]), dtype=np.float32)
  #     start_idx = 0
  #     for idx, tbl_size in enumerate(tc_table_sizes):
  #       end_idx = start_idx + tc_table_sizes[idx]
  #       cur_tbl_init = np.random.uniform(
  #           low=-1 / np.sqrt(tbl_size),
  #           high=1 / np.sqrt(tbl_size),
  #           size=(tbl_size, params["dim_embed"])).astype(np.float32)
  #       full_tbl[start_idx:end_idx, :] = cur_tbl_init
  #       start_idx += tc_table_sizes[idx]
  #     return tf.constant(full_tbl)

  #   tc_embedding_table = tf.get_variable(
  #       name="tc_embedding_table",
  #       dtype=tf.float32,
  #       trainable=True,
  #       # pylint: disable=unnecessary-lambda
  #       initializer=lambda: _create_init_table())
  #   tc_features = tf.gather(tc_embedding_table, idxs)
  #   tf.logging.info("TC features shape: {}".format(tc_features.get_shape()))

  # Dot feature interaction
  # Concat and reshape, instead of stack. Better for performance.
  # batch_size = bot_mlp_output.shape[0]
  feature_stack = tf.concat([bot_mlp_output] + cat_features, axis=-1)
  feature_stack = tf.reshape(feature_stack,
                             [-1, params["num_sparse_features"] + 1, params["dim_embed"]])

  # if "tc-features" in features:
  #   feature_stack = tf.concat([feature_stack, tc_features], axis=1)
  tf.logging.info("Model_FN: concated feature shape: %s",
                  feature_stack.get_shape())
  dot_interact_output = dot_interact(
      concat_features=feature_stack, params=params)
  top_mlp_input = tf.concat([bot_mlp_output, dot_interact_output], axis=1)
  tf.logging.info("Model_FN: Top MLP input (full features) shape: %s",
                  top_mlp_input.get_shape())

  # Capture original MLP fan-in for proper kernel initialization.
  num_fts = len(cat_features) + 1
  orig_top_mlp_dim = (num_fts * (num_fts - 1)) / 2 + params["dim_embed"]
  tf.logging.info("Model_FN: Original feature len: {}".format(orig_top_mlp_dim))

  # Top MLP
  # NOTE: For the top MLP, the last layer is a sigmoid. The loss function should
  #       therefore take [0,1] probability values as inputs, instead of logits.
  mlp_dims_top = params["mlp_top"]
  num_layers_top = len(mlp_dims_top)
  sigmoid_layer_top = num_layers_top - 1
  for layer_idx in range(num_layers_top):
    fan_in = orig_top_mlp_dim if layer_idx == 0 else mlp_dims_top[layer_idx - 1]
    fan_out = mlp_dims_top[layer_idx]
    tf.logging.info("  layer {}: fan_in={} fan_out={}".format(
        layer_idx, fan_in, fan_out))
    top_mlp_input = tf.layers.dense(
        top_mlp_input,
        mlp_dims_top[layer_idx],
        activation="sigmoid" if layer_idx == sigmoid_layer_top else "relu",
        # NOTE: We would usually use GlorotNormal() initializer here. But due to
        # the skip_gather optimization, the GlorotNormal would result in a
        # mathematical error, as Glorot is a function of the fan-in.
        # The fan-in will be larger for skip-gather activations since we also
        # pass in the zeros. Therefore we explicitly set the kernel intializer
        # to RandomNormal(0, sqrt(2/(fan_in+fan_out))

        # ti: see above
        # kernel_initializer=tf.compat.v2.initializers.RandomNormal(
        #     mean=0.0, stddev=math.sqrt(2.0 / (fan_in + fan_out))),
        # bias_initializer=tf.compat.v2.initializers.RandomNormal(
        #     mean=0.0, stddev=math.sqrt(1.0 / mlp_dims_top[layer_idx])),
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
        bias_initializer=tf.compat.v1.random_normal_initializer(
          mean=0.0, stddev=math.sqrt(1.0 / mlp_dims_top[layer_idx])),
        name="top_mlp_layer_%d" % layer_idx,
        reuse=reuse
        )
  predictions = top_mlp_input
  return predictions
