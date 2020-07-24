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
"""Lightweight embedding augmentation for TrainAndEvalRunner.

This module adds embedding hooks for train-and-eval. The
intent is to be as lightweight as possible, in the spirit of the
TrainAndEvalRunner. For simplicity, this module is built using the tpu_embedding
"mid-level" API.

  Typical user model usage example:
  embedding = tpu_embedding.TPUEmbedding(
      <table and feature params>
  )
  runner = DLRMEmbeddingRunner(
      <TrainAndEvalRunner args>,
      sparse_features="sparse-features",
      embedding=embedding
  )
  runner.initialize(
      <Input Fns and Model Fn>
  )
  runner.train_and_eval()

Limitations:
  This module does not attempt to offer support for all types and configurations
  of embeddings.  For example, the only embedding batch enqueue API exposed is
  enqueue_tpu_embedding_integer_batch().
"""
from absl import flags
import tensorflow.compat.v1 as tf

from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
import train_and_eval_runner as tr

FLAGS = flags.FLAGS


class DLRMEmbeddingRunner(tr.TrainAndEvalRunner):
  """Augmentation of the TrainAndEvalRunner with embedding support.

  This class uses the TPUEmbedding library as an API for organizing embedding
  metadata for:
  1. Configuration
  2. Building infeed ops
  3. Buidling embedding table load/restore ops
  4. Building an embedding update/train op.

  Attributes:
    sparse_features_key: String key used for all embedding features. This class
      requires all embedding features to be keyed under this string. This is
      necessary for the runner to properly strip away only those features and
      enqueue them properly.
    embedding: TPUEmbedding object representing the table and feature config.
      This attribute is required.
    **kwargs: See TrainAndEvalRunner.
  """

  def __init__(self, sparse_features_key, embedding, **kwargs):
    """Initializes the runner."""
    super(DLRMEmbeddingRunner, self).__init__(**kwargs, do_initialize=False)
    self.embedding = embedding
    self.embedding_config = embedding.config_proto
    self.features_key = sparse_features_key
    self.embed_vars_and_ops = None
    self.retrieve_ops = None
    self.enqueue_datas_list = {True: [], False: []}
    self.dummy_variables = None
    self.dummy_variables_init = None
    self.num_outfeeds = 1

    with self.graph.as_default():
      self.embed_vars_and_ops = self.embedding.create_variables_and_ops()
      self.dummy_variables, self.dummy_variables_init = (
          tpu_embedding_gradient.create_dummy_table_variables(self.embedding))
    self.device_topology = tf.Session(
        self.master, config=self.config).run(
            tpu.initialize_system(embedding_config=self.embedding_config))

  def eval_step(self, step_num, preds):
    """One evaluation step."""
    inp = self.infeed_op[False].generate_dequeue_op()
    flatten_structure = tf.nest.flatten(self.feature_structure[False])
    inp = [
        tf.slice(i, [0] * i.shape.ndims, j.shape)
        for i, j in zip(inp, flatten_structure)
    ]
    eval_has_labels = False
    if eval_has_labels:
      features, labels = tf.nest.pack_sequence_as(self.feature_structure[False],
                                                  inp)
    else:
      features = tf.nest.pack_sequence_as(self.feature_structure[False], inp)
      labels = None
    self.maybe_add_embedding_features(features, False)
    _, self.predict_output = self.model_fn(features, labels, False, step_num,
                                           preds)
    for _ in self.predict_output:
      self.dequeue_ops.append([])
    with tf.device(tr.device_for_tpu_core(self.get_host(0))):
      return step_num + 1, self.predict_output["results"]

  @tpu_function.on_device_training_loop
  def eval_loop(self):
    per_replica_eval_batch_size = self.eval_batch_size // self.num_replicas
    tf.get_variable_scope().reuse_variables()
    predictions = tf.zeros([self.eval_steps, per_replica_eval_batch_size, 2])
    _, predictions = training_loop.repeat(
        int(self.eval_steps), self.eval_step, [tf.constant(0), predictions])
    with tf.control_dependencies([tpu_ops.outfeed_enqueue_tuple([predictions])
                                 ]):
      return tf.no_op()

  def maybe_capture_embedding_inputs(self, inputs, is_training):
    """Removes sparse inputs and stores them.

    Args:
      inputs: Dict of input features, resulting from iterator.get_next().
      is_training: Boolean that is True for training and False otherwise.
    """
    sparse_inputs = inputs.pop(self.features_key)
    sparse_inputs = tf.split(sparse_inputs, sparse_inputs.shape[-1], axis=1)
    sparse_inputs = [tf.squeeze(x) for x in sparse_inputs]
    self.enqueue_datas_list[is_training].append(sparse_inputs)

  def maybe_add_embedding_enqueue_ops_int(self, is_training, enqueue_ops):
    """Adds embedding input enqueue ops.

    Args:
      is_training: Boolean that is True for training and False otherwise.
      enqueue_ops: List of existing enqueue ops used by the runner.
    """
    sparse_enqueue_ops = []
    for i, batch_data in enumerate(self.enqueue_datas_list[is_training]):
      enqueue_op = tpu_ops.enqueue_tpu_embedding_integer_batch(
          batch=batch_data,
          device_ordinal=i % FLAGS.replicas_per_host,
          mode_override="inference" if not is_training else None)
      sparse_enqueue_ops.append(enqueue_op)
    enqueue_ops.extend(sparse_enqueue_ops)
    # Clear sparse input list for this host.
    del self.enqueue_datas_list[is_training][:]

  def maybe_get_embedding_train_op(self):
    """Builds embedding table update op.

    Returns:
      An op which computes gradients and updates tables.
    """
    with tf.device(tr.device_for_tpu_core(self.get_host(0))):
      sparse_grads = (
          tpu_embedding_gradient.get_gradients_through_dummy_table_variables(
              self.embedding))
      embedding_train_op = self.embedding.generate_send_gradients_op(
          sparse_grads, tf.compat.v1.train.get_global_step())
      return embedding_train_op

  def maybe_add_embedding_features(self, features, hook_dummy_variables):
    """Adds sparse activations to feature list.

    Args:
      features: Dict of features, used by the model_fn.
      hook_dummy_variables: Boolean telling whether to back-propagate through
        embedding activations. Set to true when training and desiring backprop
        to extend to the embedding tables.
    """
    if hook_dummy_variables:
      with tf.device(tr.device_for_tpu_core(self.get_host(0))):
        embedding_activations = self.embedding.get_activations()
        new_embedding_activations = tpu_embedding_gradient.hook_dummy_table_variables_to_activations(
            self.embedding, embedding_activations, self.dummy_variables)
        features.update(new_embedding_activations)
    else:
      embedding_activations = self.embedding.get_activations()
      features.update(embedding_activations)

  def maybe_load_embedding_vars(self):
    """Loads tables into accelerator device memory."""
    self.sess.run(self.dummy_variables_init)
    self.sess.run(self.embed_vars_and_ops.load_ops())
    self.retrieve_ops = self.embed_vars_and_ops.retrieve_ops()

  def retrieve_embedding_vars(self):
    self.sess.run(self.retrieve_ops)
