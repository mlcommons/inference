# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Bypass TPUEstimator for ResNet-50 Train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import operator
import os
import threading
import time
from absl import flags
import tensorflow.compat.v1 as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.tpu import device_assignment
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops

FLAGS = flags.FLAGS
_IS_PADDED = "is_padded"

flags.DEFINE_string(
    "master",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_integer(
    "replicas_per_host", default=8, help=("Number of replicas per host."))

flags.DEFINE_bool("enable_summary", default=False, help=("Enable summary"))

flags.DEFINE_string(
    "model_dir",
    default=None,
    help=("The directory where the model and summaries are stored."))

flags.DEFINE_bool("save_checkpoint", default=False, help=("Save checkpoint"))

flags.DEFINE_bool(
    "restore_checkpoint", default=False, help=("Restore checkpoint"))

flags.DEFINE_integer(
    "sleep_after_init", default=60, help=("Sleep for N seconds after init."))

flags.DEFINE_bool(
    "enable_mlir_bridge", default=False, help=("Enable TF/XLA MLIR bridge"))

flags.DEFINE_bool(
    "enable_profiling",
    default=False,
    help=("Get xprof traces at"
          "the start and middle of the train loops"))

_NUM_CORES_TO_COMPUTATION_SHAPE = {
    1: [1, 1, 1, 1],
    2: [1, 1, 1, 2],
    4: [1, 2, 1, 2],
    8: [2, 2, 1, 2],
    16: [4, 2, 1, 2],
}


def _profiler_callback(comment, session_id):
  if session_id is None:
    tf.logging.info("Profiling failed for %s", comment)
  else:
    tf.logging.info("Profiling succeeded for %s. Overview page url:", comment)


# Decorator function for tpu computation func that was passed to tpu.rewrite()
# if there are embedded train and eval loops in this func, trace tools will
# generate step markers for each iteration.
def on_device_train_and_eval_loops(func):
  # Value for this attribute is from xla.DebugOptions.StepMarkerLocation.
  setattr(func, "step_marker_location", "STEP_MARK_AT_SECOND_LEVEL_WHILE_LOOP")
  return func


def device_for_tpu_core(host_name, core=0):
  return host_name + "/device:TPU_REPLICATED_CORE:%d" % core


def device_for_host(host_name):
  return host_name + "/device:CPU:0"


class TrainAndEvalRunner(object):
  """Remove init overheads in TPU Estimator via direct session.run calls."""

  def __init__(self,
               iterations_per_loop,
               train_steps,
               eval_steps,
               num_replicas,
               eval_dataset_repeats=True,
               do_initialize=True):
    self.feature_structure = {}
    self.infeed_op = {}
    self.num_replicas = num_replicas
    self.eval_dataset_repeats = eval_dataset_repeats
    # Set number of input graphs to number of hosts up to a maximum of 32.
    self.num_input_graphs = min(32,
                                self.num_replicas // FLAGS.replicas_per_host)
    # Following data has separated copies for training and eval, thus
    # represented as a map from is_train(boolean) to actual data
    self.dataset_initializer = {True: [], False: []}
    self.input_graph = {True: [], False: []}
    self.input_sess = {True: [], False: []}
    self.enqueue_ops = {True: [], False: []}
    for _ in range(self.num_input_graphs):
      self.input_graph[True].append(tf.Graph())
      self.input_graph[False].append(tf.Graph())
      self.dataset_initializer[True].append([])
      self.dataset_initializer[False].append([])
      self.enqueue_ops[True].append([])
      self.enqueue_ops[False].append([])
      self.input_sess[True].append([])
      self.input_sess[False].append([])
    # dequeue_ops is only for eval
    self.dequeue_ops = []
    self.iterations_per_loop = iterations_per_loop
    self.sess = None
    self.output_sess = None
    self.train_eval_thread = None
    self.graph = tf.Graph()
    if iterations_per_loop != 0 and train_steps % iterations_per_loop != 0:
      train_steps = iterations_per_loop * int(
          math.ceil(train_steps / iterations_per_loop))
    self.train_steps = train_steps
    if iterations_per_loop == 0:
      self.max_train_iterations = 1
    else:
      self.max_train_iterations = train_steps // iterations_per_loop
    self.eval_steps = int(eval_steps)
    self.train_batch_size = 0
    self.eval_batch_size = 0
    self.eval_has_labels = 0
    self.model_fn = None
    self.num_outfeeds = self.eval_steps
    self.config = tf.ConfigProto(
        operation_timeout_in_ms=600 * 60 * 1000,
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True)

    if FLAGS.enable_mlir_bridge:
      self.config.experimental.enable_mlir_bridge = True

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.master, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project,
        job_name="tpu_worker")
    self.master = tpu_cluster_resolver.get_master()
    self.job_name = tpu_cluster_resolver.get_job_name() or "tpu_worker"
    self.embedding_config = None
    self.device_topology = None
    if do_initialize:
      self.device_topology = tf.Session(
          self.master, config=self.config).run(
              tpu.initialize_system())

  def maybe_capture_embedding_inputs(self, inputs, is_training):
    pass

  def maybe_add_embedding_enqueue_ops_int(self, is_training, enqueue_ops):
    pass

  def maybe_get_embedding_train_op(self):
    return tf.no_op()

  def maybe_add_embedding_features(self, features, hook_dummy_variables):
    pass

  def maybe_load_embedding_vars(self):
    pass

  def get_host(self, host_id):
    if self.master in ("", "local"):
      return "/replica:0/task:0"
    return "/job:%s/task:%d" % (self.job_name, host_id)

  def build_enqueue_ops(self, input_fn, is_training, input_partition_dims,
                        params):
    """Build enqueue operations for the input pipeline in a given host.

    Args:
      input_fn: dataset input graph generation function
      is_training: boolean indicates if it is training
      input_partition_dims: list of integers to partition input
      params: hyper parameters
    """

    def _tpu_ordinal_fn(shard_index_in_host):
      replica_id = self.device_assignment.lookup_replicas(
          host_id, logical_core=0)[shard_index_in_host]
      return self.device_assignment.tpu_ordinal(
          replica=replica_id, logical_core=0)

    host_id = params["dataset_index"]
    gindex = host_id % self.num_input_graphs
    with self.input_graph[is_training][gindex].as_default():
      with tf.device(device_for_host(self.get_host(host_id))):
        dataset = input_fn(params)
        if not is_training and self.eval_dataset_repeats:
          dataset = dataset.cache().repeat()
        iterator = dataset.make_initializable_iterator()
        self.dataset_initializer[is_training][gindex].append(
            iterator.initializer)

        def enqueue_ops_fn(idx):
          """Generate the infeed enqueue ops graph."""

          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(FLAGS.replicas_per_host):
            with tf.control_dependencies(control_deps):
              self.feature_structure[is_training] = iterator.get_next()
            self.maybe_capture_embedding_inputs(
                self.feature_structure[is_training], is_training)
            flattened_inputs = tf.nest.flatten(
                self.feature_structure[is_training])
            control_deps.extend(flattened_inputs)
            if input_partition_dims:
              padded_inputs = []
              for inp in flattened_inputs:
                if inp.shape.ndims < len(input_partition_dims):
                  padded_inputs.append(inp)
                  continue
                paddings = []
                for i, j in enumerate(input_partition_dims):
                  r = inp.shape.as_list()[i] % j
                  if r > 0:
                    paddings.append([0, j - r])
                  else:
                    paddings.append([0, 0])
                for i in range(inp.shape.ndims - len(input_partition_dims)):
                  paddings.append([0, 0])
                padded_inputs.append(tf.pad(inp, paddings))
              per_host_sharded_inputs.append(padded_inputs)
            else:
              per_host_sharded_inputs.append(flattened_inputs)

          if input_partition_dims:
            flattened_input_dims = []
            for i in per_host_sharded_inputs[0]:
              if i.shape.ndims == len(input_partition_dims):
                flattened_input_dims.append(input_partition_dims)
              elif i.shape.ndims > len(input_partition_dims):
                flattened_input_dims.append(
                    input_partition_dims + [1] *
                    (i.shape.ndims - len(input_partition_dims)))
              else:
                flattened_input_dims.append([1] * i.shape.ndims)
            # pylint: disable=protected-access
            self.infeed_op[is_training] = tpu_feed._PartitionedInfeedQueue(
                number_of_tuple_elements=len(per_host_sharded_inputs[0]),
                host_id=host_id,
                input_partition_dims=flattened_input_dims,
                device_assignment=self.device_assignment)
            with tf.control_dependencies(
                self.infeed_op[is_training].generate_enqueue_ops(
                    per_host_sharded_inputs)):
              return idx + 1
          else:
            self.infeed_op[is_training] = tpu_feed.InfeedQueue(
                number_of_tuple_elements=len(per_host_sharded_inputs[0]))
            per_host_enqueue_ops = (
                self.infeed_op[is_training].generate_enqueue_ops(
                    per_host_sharded_inputs,
                    tpu_ordinal_function=_tpu_ordinal_fn))

          self.maybe_add_embedding_enqueue_ops_int(
              is_training, per_host_enqueue_ops)
          with tf.control_dependencies(per_host_enqueue_ops):
            return idx + 1

        iterations = self.iterations_per_loop if is_training else self.eval_steps
        self.enqueue_ops[is_training][gindex].append(
            tf.while_loop(
                lambda i: tf.less(i, iterations),
                enqueue_ops_fn, [tf.constant(0)],
                parallel_iterations=1))

  def launch_profiler(self):
    """Launches a profiling session to collect a trace from worker-0."""
    if result == profiler_client.PROFILED_IN_NEW_THREAD:
      tf.logging.info("A profiler session launched in a new thread.")
    else:
      tf.logging.info("profiler.collect() failed.")

  def eval_step(self):
    """One evaluation step."""
    inp = self.infeed_op[False].generate_dequeue_op()
    flatten_structure = tf.nest.flatten(self.feature_structure[False])
    inp = [
        tf.slice(i, [0] * i.shape.ndims, j.shape)
        for i, j in zip(inp, flatten_structure)
    ]
    if self.eval_has_labels:
      features, labels = tf.nest.pack_sequence_as(
          self.feature_structure[False], inp)
    else:
      features = tf.nest.pack_sequence_as(self.feature_structure[False], inp)
      labels = None
    self.maybe_add_embedding_features(features, False)
    _, self.predict_output = self.model_fn(features, labels, False)
    for _ in self.predict_output:
      self.dequeue_ops.append([])
    with tf.device(device_for_tpu_core(self.get_host(0))):
      return [
          tpu_ops.outfeed_enqueue_tuple(tf.nest.flatten(self.predict_output))
      ]

  @tpu_function.on_device_training_loop
  def eval_loop(self):
    tf.get_variable_scope().reuse_variables()
    return training_loop.repeat(int(self.eval_steps), self.eval_step)

  def initialize(self,
                 train_input_fn,
                 eval_input_fn,
                 model_fn,
                 train_batch_size,
                 eval_batch_size,
                 input_partition_dims=None,
                 init_fn=None,
                 train_has_labels=True,
                 eval_has_labels=True,
                 params=None,
                 num_partitions=None):
    """Build graphs for the TPU device and the input pipelines."""
    num_cores_per_replica = 1
    num_cores_per_replica = functools.reduce(
        operator.mul, input_partition_dims
    ) if input_partition_dims else num_partitions if num_partitions else 1

    self.device_assignment = device_assignment.device_assignment(
        topology=self.device_topology,
        computation_shape=_NUM_CORES_TO_COMPUTATION_SHAPE[
            num_cores_per_replica],
        num_replicas=self.num_replicas)
    self.train_batch_size = train_batch_size
    self.eval_batch_size = eval_batch_size
    self.eval_has_labels = eval_has_labels
    self.model_fn = model_fn

    if params is None:
      params = {}
    params["dataset_num_shards"] = self.num_replicas // FLAGS.replicas_per_host
    per_replica_train_batch_size = train_batch_size // self.num_replicas
    per_replica_eval_batch_size = eval_batch_size // self.num_replicas
    for i in range(self.num_replicas // FLAGS.replicas_per_host):
      params["dataset_index"] = i
      params["batch_size"] = per_replica_train_batch_size
      self.build_enqueue_ops(train_input_fn, True, input_partition_dims, params)
      if self.eval_steps > 0:
        params["batch_size"] = per_replica_eval_batch_size
        self.build_enqueue_ops(eval_input_fn, False, input_partition_dims,
                               params)

    def train_step(_):
      """One train step."""
      inp = self.infeed_op[True].generate_dequeue_op()
      flatten_structure = tf.nest.flatten(self.feature_structure[True])
      inp = [
          tf.slice(i, [0] * i.shape.ndims, j.shape)
          for i, j in zip(inp, flatten_structure)
      ]
      if train_has_labels:
        features, labels = tf.nest.pack_sequence_as(
            self.feature_structure[True], inp)
      else:
        features = tf.nest.pack_sequence_as(self.feature_structure[True], inp)
        labels = None
      self.maybe_add_embedding_features(features, True)
      train_op, _ = model_fn(features, labels, True)
      embedding_train_op = self.maybe_get_embedding_train_op()
      with tf.device(device_for_tpu_core(self.get_host(0))):
        with tf.control_dependencies([train_op, embedding_train_op]):
          return tf.constant(0)

    @tpu_function.on_device_training_loop
    def train_loop():
      return training_loop.repeat(self.iterations_per_loop, train_step,
                                  tf.constant(0))

    def train_eval_step():
      with tf.control_dependencies(train_loop()):
        if self.eval_steps > 0:
          return self.eval_loop()
        else:
          return tf.no_op()

    @on_device_train_and_eval_loops
    def train_eval_loop():
      return training_loop.repeat(self.max_train_iterations, train_eval_step)

    with self.graph.as_default():
      (self.train_eval_op,) = tpu.shard(
          train_eval_loop,
          inputs=[],
          num_shards=self.num_replicas,
          outputs_from_all_shards=False,
          device_assignment=self.device_assignment)
      if FLAGS.model_dir:
        tf.io.write_graph(self.graph, FLAGS.model_dir, "graph.pbtxt")

    output_graph = tf.Graph()
    if self.eval_steps > 0:
      with output_graph.as_default():
        flatten_output = tf.nest.flatten(self.predict_output)
        self.dequeue_ops = [[] for _ in flatten_output]
        tensor_dtypes = [v.dtype for v in flatten_output]
        tensor_shapes = [v.shape for v in flatten_output]
        is_padded_index = flatten_output.index(
            self.predict_output[_IS_PADDED]
        ) if _IS_PADDED in self.predict_output else -1
        for i in range(self.num_replicas // FLAGS.replicas_per_host):
          with tf.device(device_for_host(self.get_host(i))):
            host_dequeue_ops = [[] for _ in flatten_output]
            for j in range(FLAGS.replicas_per_host):
              replica_id = self.device_assignment.lookup_replicas(i, 0)[j]
              ordinal = self.device_assignment.tpu_ordinal(
                  replica=replica_id, logical_core=0)
              dequeue_ops = tpu_ops.outfeed_dequeue_tuple(
                  dtypes=tensor_dtypes,
                  shapes=tensor_shapes,
                  device_ordinal=ordinal)
              if is_padded_index >= 0:
                num_non_pad = tf.shape(
                    dequeue_ops[is_padded_index])[0] - tf.reduce_sum(
                        tf.cast(dequeue_ops[is_padded_index], tf.int32))
                dequeue_ops = [
                    tf.slice(k, [0] * k.shape.ndims,
                             [num_non_pad] + [-1] * (k.shape.ndims - 1))
                    for k in dequeue_ops
                ]
              for k, item in enumerate(dequeue_ops):
                host_dequeue_ops[k].append(item)
            for k in range(len(self.predict_output)):
              self.dequeue_ops[k].append(tf.concat(host_dequeue_ops[k], axis=0))

    self.sess = tf.Session(self.master, graph=self.graph, config=self.config)
    for is_training in [True, False]:
      if is_training or self.eval_steps > 0:
        for i in range(self.num_input_graphs):
          with self.input_graph[is_training][i].as_default():
            self.input_sess[is_training][i] = tf.Session(
                self.master,
                graph=self.input_graph[is_training][i],
                config=self.config)
            self.input_sess[is_training][i].run(
                self.dataset_initializer[is_training][i])
    self.output_sess = tf.Session(
        self.master, graph=output_graph, config=self.config)

    with self.graph.as_default():
      _ = tf.train.get_or_create_global_step()
      if init_fn:
        init_fn()
      checkpoint_path = tf.train.latest_checkpoint(
          FLAGS.model_dir) if FLAGS.model_dir else None
      if FLAGS.restore_checkpoint and checkpoint_path:
        tf.train.Saver().restore(self.sess, checkpoint_path)
      else:
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
      self.maybe_load_embedding_vars()
      self.global_step = self.sess.run(tf.train.get_global_step(self.graph))

    def train_eval_thread_fn(sess, train_eval_op):
      sess.run([train_eval_op])

    # Start the just in time compilation of the model function
    self.train_eval_thread = threading.Thread(
        target=train_eval_thread_fn, args=(self.sess, self.train_eval_op))
    self.train_eval_thread.start()

    # Sleep for JTC to finish
    time.sleep(FLAGS.sleep_after_init)

  def train_and_eval(self,
                     eval_init_fn=None,
                     eval_finish_fn=None,
                     run_finish_fn=None):
    """Run the Train steps on the TPU device."""
    if FLAGS.enable_summary:
      output_dir = os.path.join(FLAGS.model_dir, "eval")
      tf.gfile.MakeDirs(output_dir)
      summary_writer = tf.summary.FileWriter(output_dir)
    else:
      summary_writer = None

    def infeed_thread_fn(thread_index):
      # Wait for condition
      """Build and infeed session.run calls in a background thread."""
      for _ in range(self.max_train_iterations):
        self.input_sess[True][thread_index].run(
            [self.enqueue_ops[True][thread_index]])
        if self.eval_steps > 0:
          if not self.eval_dataset_repeats:
            self.input_sess[False][thread_index].run(
                self.dataset_initializer[False][thread_index])
          self.input_sess[False][thread_index].run(
              [self.enqueue_ops[False][thread_index]])

    infeed_threads = []
    for i in range(self.num_input_graphs):
      thread = threading.Thread(target=infeed_thread_fn, args=([i]))
      thread.start()
      infeed_threads.append(thread)

    global_step = self.global_step

    if self.eval_steps > 0:
      enable_tracing = FLAGS.enable_profiling
      if enable_tracing:
        self.launch_profiler()

      success = False
      step_range = [global_step] if self.iterations_per_loop == 0 else range(
          global_step, global_step + self.train_steps, self.iterations_per_loop)
      for cur_step in step_range:
        if not success and eval_init_fn:
          eval_init_fn(cur_step)
        eval_output = [[] for _ in self.dequeue_ops]
        for _ in range(self.num_outfeeds):
          for i, t in enumerate(self.output_sess.run(self.dequeue_ops)):
            eval_output[i] += list(t)
        eval_output = tf.nest.pack_sequence_as(self.predict_output, eval_output)
        if eval_finish_fn and not success and eval_finish_fn(
            cur_step, eval_output, summary_writer):
          success = True
        if enable_tracing and cur_step > self.train_steps // 4:
          self.launch_profiler()
          enable_tracing = False

      if run_finish_fn:
        run_finish_fn(success)

    if FLAGS.save_checkpoint:
      with self.graph.as_default():
        self.global_step = self.sess.run(tf.train.get_global_step(self.graph))
        checkpoint_path = FLAGS.model_dir + "/model.ckpt-%d" % self.global_step
        tf.train.Saver().save(self.sess, checkpoint_path)
        tf.logging.info("Checkpoint saved to %s", checkpoint_path)

    if FLAGS.enable_summary:
      summary_writer.close()

    self.train_eval_thread.join()
    for i in range(self.num_input_graphs):
      infeed_threads[i].join()
    self.sess.close()
