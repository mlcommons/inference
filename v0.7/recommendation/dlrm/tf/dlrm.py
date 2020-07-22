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

import tensorflow.compat.v1 as tf

from tensorflow.compiler.tf2xla.python import xla
from tensorflow.contrib import  layers as contrib_layers
import utils


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
  batch_size = concat_features.shape[0]
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

  activations = tf.reshape(activations, (batch_size, out_dim))
  return activations


def logits_fn(features, params, feature_config):
  """Calculate predictions."""
  tf.logging.info("Model_FN: Number of input features: %d", len(features))
  for ft in sorted(features.keys()):
    tf.logging.info("Model_FN: Feature %s -- shape %s", ft,
                    features[ft].get_shape())

  bot_mlp_input = features["int-features"]
  tf.logging.info("Model_FN: Bottom MLP input (int features) shape: %s",
                  bot_mlp_input.get_shape())
  mlp_dims_bottom = params["mlp_bottom"]

  for layer_idx in range(len(mlp_dims_bottom)):
    bot_mlp_input = tf.layers.dense(
        bot_mlp_input,
        mlp_dims_bottom[layer_idx],
        activation="relu",
        kernel_initializer=tf.compat.v2.initializers.GlorotNormal(),
        bias_initializer=tf.compat.v2.initializers.RandomNormal(
            mean=0.0, stddev=math.sqrt(1.0 / mlp_dims_bottom[layer_idx])),
        name="bottom_mlp_layer_%d" % layer_idx)
  bot_mlp_output = bot_mlp_input

  cat_features = []
  for feature_name, value in sorted(features.items()):
    if "categorical-feature" in feature_name:
      cat_features.append(value)

  tc_features = []
  if "tc-features" in features:
    # Compute offsets for single concatenated table.
    batch_size = features["tc-features"].shape[0]
    num_tc_features = features["tc-features"].shape[1]
    num_tables_in_ec = params["num_tables_in_ec"]
    tc_table_sizes = feature_config.get_sorted_table_size()[num_tables_in_ec:]
    total_tbl_size = sum(tc_table_sizes)
    idx_offsets = [0] + list(np.cumsum(tc_table_sizes[:-1]))
    idx_offsets = tf.broadcast_to(
        tf.constant(idx_offsets), (batch_size, num_tc_features))
    idxs = idx_offsets + features["tc-features"]

    def _create_init_table():
      """Table initialization varies depending on the vocab size."""
      full_tbl = np.zeros(
          shape=(total_tbl_size, params["dim_embed"]), dtype=np.float32)
      start_idx = 0
      for idx, tbl_size in enumerate(tc_table_sizes):
        end_idx = start_idx + tc_table_sizes[idx]
        cur_tbl_init = np.random.uniform(
            low=-1 / np.sqrt(tbl_size),
            high=1 / np.sqrt(tbl_size),
            size=(tbl_size, params["dim_embed"])).astype(np.float32)
        full_tbl[start_idx:end_idx, :] = cur_tbl_init
        start_idx += tc_table_sizes[idx]
      return tf.constant(full_tbl)

    tc_embedding_table = tf.get_variable(
        name="tc_embedding_table",
        dtype=tf.float32,
        trainable=True,
        # pylint: disable=unnecessary-lambda
        initializer=lambda: _create_init_table())
    tc_features = tf.gather(tc_embedding_table, idxs)
    tf.logging.info("TC features shape: {}".format(tc_features.get_shape()))

  # Dot feature interaction
  # Concat and reshape, instead of stack. Better for performance.
  batch_size = bot_mlp_output.shape[0]
  feature_stack = tf.concat([bot_mlp_output] + cat_features, axis=-1)
  feature_stack = tf.reshape(feature_stack,
                             [batch_size, -1, params["dim_embed"]])
  if "tc-features" in features:
    feature_stack = tf.concat([feature_stack, tc_features], axis=1)
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
        kernel_initializer=tf.compat.v2.initializers.RandomNormal(
            mean=0.0, stddev=math.sqrt(2.0 / (fan_in + fan_out))),
        bias_initializer=tf.compat.v2.initializers.RandomNormal(
            mean=0.0, stddev=math.sqrt(1.0 / mlp_dims_top[layer_idx])),
        name="top_mlp_layer_%d" % layer_idx)
  predictions = top_mlp_input
  return predictions, None


def create_model_fn():
  """Creates the model_fn to be used by the TPUEstimator."""

  def _dlrm_model_fn(features, mode, params):
    """Model function definition for DLRM.

    Args:
      features: List of feature tensors used in model.
      mode: Usage mode of the model function, e.g. train, eval, etc.
      params: Hparams for the model.

    Returns:
      TPUEstimatorSpec providing the train_op and loss operators.

    Raises:
      NotImplementedError for unsupported execution modes.
    """

    preds, host_call_fn = logits_fn(features, params, None)
    tf.logging.info("Model_FN: Shape of predictions: %s", preds.get_shape())
    labels = features["clicked"]
    tf.logging.info("Model_FN: Shape of labels: %s", labels.get_shape())

    if mode == tf.estimator.ModeKeys.EVAL:
      labels = tf.reshape(labels, [-1])
      preds = tf.reshape(preds, [-1])
      bce_func = tf.keras.losses.BinaryCrossentropy(
          from_logits=False, reduction=tf.compat.v2.keras.losses.Reduction.NONE)
      eval_loss = tf.reduce_mean(bce_func(labels, preds))

      def metric_fn(labels, predictions):
        label_weights = tf.ones_like(labels, dtype=tf.float32)
        prediction_labels = tf.round(predictions)

        return {
            utils.ACC_KEY:
                tf.metrics.accuracy(
                    labels=labels,
                    predictions=prediction_labels,
                    weights=label_weights),
            utils.AUC_KEY:
                tf.metrics.auc(
                    labels=labels,
                    predictions=predictions,
                    weights=label_weights,
                    num_thresholds=1000,
                    curve="ROC"),
        }

      eval_metrics = (metric_fn, [labels, preds])

      tf.logging.info(
          "Model_FN EVAL: Metrics have been set up. Now returning..")

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=eval_loss,
          host_call=host_call_fn,
          eval_metrics=eval_metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:

      bce_func = tf.keras.losses.BinaryCrossentropy(
          from_logits=False, reduction=tf.compat.v2.keras.losses.Reduction.NONE)
      loss = tf.reduce_mean(bce_func(labels, preds))

      global_step = tf.train.get_global_step()
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=params["learning_rate"])
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)
      train_op = optimizer.minimize(loss, global_step=global_step)

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          host_call=host_call_fn,
      )

    else:
      raise NotImplementedError(
          "Only TRAIN and EVAL modes are supported. Got: %s" % (mode))

  return _dlrm_model_fn


class ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, params, lr, global_step):
    self._params = params
    self._global_step = global_step
    self._bfloat16_grads_all_reduce = params["bfloat16_grads_all_reduce"]
    self._lr = lr
    self._opt = tf.train.GradientDescentOptimizer(
        learning_rate=utils.lr_fn(self._params, self._global_step))
    if params["optimizer"] == "adagrad":
      self._opt = tf.train.AdagradOptimizer(
          learning_rate=params["learning_rate"],
          initial_accumulator_value=params["adagrad_init_accum"])

  def _cast_like(self, x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    if x.dtype.base_dtype == y.dtype.base_dtype:
      return x
    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
      tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'",
                         x.name, x.device, cast_x.device)
    return cast_x

  # pylint: disable=arguments-differ
  def compute_gradients(self, loss, var_list=None, **kwargs):
    gradients = self._opt.compute_gradients(loss, var_list, **kwargs)

    def cast_grad(g, v):
      if v is not None and g is not None:
        g = self._cast_like(g, v)
      return (g, v)

    gradients = [cast_grad(g, v) for g, v in gradients]
    if self._bfloat16_grads_all_reduce:
      gradients = [(tf.cast(g, tf.bfloat16), v) for g, v in gradients]
    return gradients

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # Cast the gradients back to float32 for weight updates.
    if self._bfloat16_grads_all_reduce:
      grads_and_vars = [(tf.cast(g, tf.float32), v) for g, v in grads_and_vars]
    return self._opt.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)


# TODO(tayo): Clean this up and merge with estimator invocation.
def dlrm_llr_model_fn(params,
                      feature_config,
                      features,
                      labels,
                      is_training,
                      eval_step_num=None,
                      predictions=None):
  """Model fn.

  Args:
    params: Params dict for the model.
    feature_config: Configuration of features.
    features: Features dict for the model.
    labels: Labels tensor. Not used for this model.
    is_training: Boolean, True if training.
    eval_step_num: Int tensor, representing the batch number during eval.
    predictions: [num_batches, batch_size, 2] tensor holding all predictions.

  Returns:
    [train_op, predictions]
  """
  assert labels is None, "Labels should be None. Reconfigure."
  labels = features["clicked"]
  preds, _ = logits_fn(features, params, feature_config)
  global_step = tf.train.get_or_create_global_step()

  if is_training:
    bce_func = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.compat.v2.keras.losses.Reduction.NONE)
    loss = tf.reduce_mean(bce_func(labels, preds))
    learning_rate = utils.lr_fn(params, global_step)
    optimizer = ConditionalOptimizer(params, learning_rate, global_step)
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
    train_op = contrib_layers.optimize_loss(
        name="training",
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=optimizer,
        colocate_gradients_with_ops=True)
    return train_op, None
  else:
    # TODO(tayo): Consider adding a local key-value sort.
    new_preds = tf.concat([preds, tf.cast(labels, tf.float32)], axis=1)

    predictions = xla.dynamic_update_slice(
        predictions, tf.expand_dims(new_preds, axis=0),
        tf.stack([eval_step_num, tf.constant(0),
                  tf.constant(0)]))

    return None, dict(results=predictions)
