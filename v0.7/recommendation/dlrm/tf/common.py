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
"""Flags and common definitions for all modules in the DLRM module."""
 
import collections
from typing import Dict, Any
from absl import flags
 
 
FLAGS = flags.FLAGS
 
PARAMS = collections.defaultdict(
    lambda: None,  # Set default value to None.
    default_batch_size=32,
 
    # TPU-specific parameters
    use_tpu=True,
)
 
FAKE_DATA_VOCAB_SIZE = 1000

def get_params() -> Dict[str, Any]:
  """Provides param dict and sets defaults.
 
  Returns:
    A dict representing the params for the model execution.
  Raises:
    ValueError: If parameters violate model architecture constraints.
  """
 
  params = PARAMS.copy()
 
  params["data_dir"] = FLAGS.data_dir
  params["model_dir"] = FLAGS.model_dir
  params["summary_every_n_steps"] = FLAGS.summary_every_n_steps
  params["batch_size"] = FLAGS.batch_size
  params["eval_batch_size"] = FLAGS.eval_batch_size
  params["dim_embed"] = FLAGS.dim_embed
  params["vocab_sizes"] = [int(x) for x in FLAGS.vocab_sizes_embed]
  params["num_dense_features"] = FLAGS.num_dense_features
  params["num_tables_in_ec"] = FLAGS.num_tables_in_ec
  params["mlp_bottom"] = [int(x) for x in FLAGS.mlp_bottom]
  params["mlp_top"] = [int(x) for x in FLAGS.mlp_top]
  params["learning_rate"] = FLAGS.learning_rate
  params["lr_warmup_steps"] = FLAGS.lr_warmup_steps
  params["decay_steps"] = FLAGS.decay_steps
  params["decay_start_step"] = FLAGS.decay_start_step
  params["optimizer"] = FLAGS.optimizer
  params["adagrad_init_accum"] = FLAGS.adagrad_init_accum
  params["num_shards"] = FLAGS.num_tpu_shards
  params["eval_steps"] = FLAGS.eval_steps
  params["replicas_per_host"] = FLAGS.replicas_per_host
  params["bfloat16_grads_all_reduce"] = FLAGS.bfloat16_grads_all_reduce
  # Dataset
  params["terabyte"] = FLAGS.terabyte
  params["use_synthetic_data"] = FLAGS.use_synthetic_data
  params["use_cached_data"] = FLAGS.use_cached_data
  if params["use_synthetic_data"]:
    params["vocab_sizes"] = [FAKE_DATA_VOCAB_SIZE for _ in FLAGS.vse]
  # Optimization
  params["opt_skip_gather"] = True
 
  if params["dim_embed"] != params["mlp_bottom"][-1]:
    raise ValueError("Dimensionality of latent features (embedding dim) " +
                     "must be equal to size of last layer of the bottom MLP.")
  if params["batch_size"] % params["num_shards"]:
    raise ValueError("Training batch size {} must be a multiple of num_cores {}"
                     .format(params["batch_size"], params["num_shards"]))
  if params["eval_batch_size"] % params["num_shards"]:
    raise ValueError("Eval batch size {} must be a multiple of num_cores {}"
                     .format(params["eval_batch_size"], params["num_shards"]))
  return params
 
 
def define_dlrm_flags() -> None:
  """Flags for running dlrm_main."""
 
  # TODO(tayo): Merge flags with the low level runner.
  flags.DEFINE_string(
      "data_dir",
      default=None,
      help="Path to the data directory.")
  flags.DEFINE_integer(
      "batch_size",
      default=32,
      help="Batch size for training.")
  flags.DEFINE_bool(
      "use_synthetic_data",
      default=False,
      help="If true, uses synthetic data.")
  flags.DEFINE_enum(
      "optimizer",
      default="sgd",
      enum_values=["sgd", "adagrad"],
      help="Optimizer to use for parameter updates.")
  flags.DEFINE_float(
      name="adagrad_init_accum",
      default=0.01,
      help="Adagrad initial accumulator values.")
  flags.DEFINE_integer(
      name="lr_warmup_steps",
      default=0,
      help="Number of warmup steps in learning rate.")
  flags.DEFINE_integer(
      name="decay_steps",
      default=0,
      help="Number of decay steps used in polynomial decay.")
  flags.DEFINE_integer(
      name="decay_start_step",
      default=0,
      help="Step to begin decay, if decay_steps > 0.")
  flags.DEFINE_integer(
      name="num_tpu_shards",
      default=8,
      help="Number of shards (cores).")
  flags.DEFINE_integer(
      name="eval_batch_size",
      short_name="ebs",
      default=16384,
      help="Global batch size to use during eval.")
  flags.DEFINE_float(
      name="learning_rate",
      short_name="lr",
      default=0.01,
      help="The learning rate.")
  flags.DEFINE_integer(
      name="train_steps",
      short_name="ts",
      default=1000,
      help="The number of steps used to train.")
  flags.DEFINE_integer(
      name="eval_steps",
      short_name="es",
      default=5440,
      help="The number of steps used to eval.")
  flags.DEFINE_integer(
      name="steps_between_evals",
      short_name="sbe",
      default=100,
      help="The Number of training steps to run between evaluations. This is "
           "used if --train_steps is defined.")
  flags.DEFINE_integer(
      name="summary_every_n_steps",
      default=100,
      help="Number of training steps to run before communicating with host to "
           "send summaries.")
  flags.DEFINE_bool(
      name="use_cached_data",
      default=False,
      help="If true, take a few samples and repeat.")
  flags.DEFINE_string(
      name="mode",
      default="train",
      help="mode: train or eval")
  flags.DEFINE_bool(
      name="use_ctl",
      default=False,
      help="Whether the model runs with custom training loop.")
  flags.DEFINE_bool(
      name="terabyte",
      default=True,
      help="If true, data paths use terabyte format. Else kaggle.")
  # System params.
  flags.DEFINE_bool(
      name="pipeline_execution",
      default=False,
      help="If true, pipeline embedding execution with TensorCore.")
  flags.DEFINE_bool(
      name="use_batched_tfrecords",
      default=False,
      help="If true, use dataset of batched TFRecords, instead of csv.")
  flags.DEFINE_bool(
      name="bfloat16_grads_all_reduce",
      default=False,
      help="If true, use bfloat16 for all-reduce computation.")
  flags.DEFINE_string(
      name="data_cell",
      default="mb",
      help="Data cell. Path to data directory is determined dynamically.")
  flags.DEFINE_enum(
      "partition_strategy",
      default="div",
      enum_values=["div", "mod"],
      help="Partition strategy for the embeddings.")
  # Model architecture params.
  flags.DEFINE_integer(
      name="dim_embed",
      short_name="de",
      default=4,
      help="Embedding dimension.")
  flags.DEFINE_list(
      name="mlp_bottom",
      default="8, 4",
      help="Hidden layers for the bottom MLP. "
           "To specify different sizes of MLP layers: --layers=32,16,8,4")
  flags.DEFINE_list(
      name="mlp_top",
      default="128, 64, 1",
      help="The sizes of hidden layers for MLP. "
           "To specify different sizes of MLP layers: --layers=32,16,8,4")
  flags.DEFINE_list(
      name="vocab_sizes_embed",
      short_name="vse",
      default="8, 8, 8, 8",
      help="Vocab sizes for each of the sparse features. The order agrees with "
           "the order of the input data.")
  flags.DEFINE_integer(
      name="num_dense_features",
      short_name="ndf",
      default=3,
      help="Number of dense features.")
  flags.DEFINE_integer(
      name="num_tables_in_ec",
      default=26,
      help="Number of embedding tables in the embedding core.")

