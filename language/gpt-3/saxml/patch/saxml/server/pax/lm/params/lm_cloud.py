# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serving model parameters for lm_cloud."""

import os
from typing import List, cast

from jax import numpy as jnp
from paxml import base_experiment
from paxml import tasks_lib
from paxml.tasks.lm.params import lm_cloud
from praxis import base_input
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import activations
from saxml.server import servable_model_registry
from saxml.server.pax import quantization
from saxml.server.pax.lm.layers import LLaMARotaryEmbedding
from saxml.server.pax.lm.layers import ParallelTransformer
from saxml.server.pax.lm.params import template


@template.make_servable()
class BaseLLaMA(base_experiment.BaseExperiment):
  """Base LLaMA Transformer LM configuration."""

  SPM_MODEL = 'gs://sax-data/pax-llama/tokenizer.model'
  SOS_ID = 1
  EOS_ID = 2

  # architecture related
  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16

  ACTIVATION_CLS = activations.SiLU
  USE_GATED_ACTIVATION = True
  RMS_NORM_EPSILON = 1.0e-05

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 1, 1]
  DCN_MESH_SHAPE = None
  DECODE_MESH_TRANSPOSE = None

  BATCH_SIZE = 1
  NUM_SAMPLES = 1
  ENABLE_GENERATE_STREAM = True
  STREAM_INTERVAL_STEPS = 16
  FPROP_FOR_PREFIX = True
  INPUT_SEQ_LEN = 4096
  BUCKET_KEYS = [128, 1024, 4096]
  MAX_DECODE_STEPS = [128, 512, 1024]
  EXTRA_INPUTS = {
      'temperature': 0.5,
      'per_example_max_decode_steps': 128,
      'per_example_top_k': 200,
      'per_example_top_p': 0.95,
  }

  def datasets(self) -> List[base_input.BaseInput.HParams]:
    return []

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = tasks_lib.SingleTask.HParams(name='xformer_task')
    task_p.model = pax_fiddle.Config(layers.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl.packed_input = False
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    model_p.lm_tpl.position_emb_tpl = None
    model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(
        layers.FullSoftmax,
        name='output',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
    model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
        layers.Embedding,
        name='tok_embeddings',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    ln_tpl = pax_fiddle.Config(
        layers.RmsNorm,
        name='norm',
        direct_scale=True,
        epsilon=self.RMS_NORM_EPSILON,
    )
    model_p.lm_tpl.final_ln_tpl = ln_tpl.clone()

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    transformer_layer_p = cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.norm_policy = 'pre'
    transformer_layer_p.ln_tpl = ln_tpl.clone()
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        pax_fiddle.Config(LLaMARotaryEmbedding)
    )
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    transformer_layer_p.tr_fflayer_tpl.has_bias = False
    transformer_layer_p.tr_fflayer_tpl.ln_tpl = ln_tpl.clone()
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION
    )

    model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    model_p.fprop_dtype = self.FPROP_DTYPE
    model_p.dtype = self.MODEL_DTYPE

    # Set sharding
    task_p = template.set_decoding_sharding_hparams(
        task_p,
        mesh_shape=self.ICI_MESH_SHAPE,
        decode_mesh_transpose=self.DECODE_MESH_TRANSPOSE,
    )
    # Unused.
    lp = task_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.ShardedSgd.HParams(
        learning_rate=1e-3, lr_schedule=schedules.Constant.HParams()
    )
    return task_p


@servable_model_registry.register
class LLaMA7BFP16(BaseLLaMA):
  """7B model on a A100-40GB.

  Checkpoint:
  gs://sax-data/pax-llama/7B

  April 14, 2023
  Latency = 3.619s with 128 decoded tokens. 27ms per output token
  """

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008

  BATCH_SIZE = 1
  NUM_SAMPLES = 1
  INPUT_SEQ_LEN = 128
  BUCKET_KEYS = None
  MAX_DECODE_STEPS = 32
  ENABLE_GENERATE_STREAM = False

  ICI_MESH_SHAPE = [1, 1, 1]


@servable_model_registry.register
class LLaMA7BFP16TPUv4(LLaMA7BFP16):
  """7B model on TPU v4-8.

  April 14, 2023
  Latency = 0.688s with 128 decoded tokens. 5ms per output token
  """

  ICI_MESH_SHAPE = [1, 1, 4]

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class LLaMA7B(BaseLLaMA):
  """7B model on a A100-40GB.

  April 12, 2023
  Latency = 2.337s with 128 decoded tokens. 17ms per output token.
  """

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008
  ICI_MESH_SHAPE = [1, 1, 1]

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class LLaMA13B(BaseLLaMA):
  """13B model on a A100-40GB.

  April 12, 2023
  Latency = 5.06s with 128 decoded tokens. 38ms per output token.
  """

  NUM_LAYERS = 40
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824
  ICI_MESH_SHAPE = [1, 1, 1]

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class LLaMA33B(BaseLLaMA):
  """33B model on TPU v4-8.

  April 12, 2023
  Latency = 3.35s with 128 decoded tokens. 25ms per output token.
  """

  NUM_LAYERS = 60
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 52
  MODEL_DIMS = 6656
  HIDDEN_DIMS = 17920
  ICI_MESH_SHAPE = [1, 1, 4]

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class LLaMA65B(BaseLLaMA):
  """65B model on TPUv4-8.

  April 12, 2023
  Latency = 5.9s with 128 decoded tokens. 45ms per output token.
  """

  NUM_LAYERS = 80
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 64
  MODEL_DIMS = 8192
  HIDDEN_DIMS = 22016
  ICI_MESH_SHAPE = [1, 1, 4]

  @property
  def test_mode(self) -> bool:
    return True


# GPT-J/NeoX family
@template.make_servable()
class BaseNeoX(base_experiment.BaseExperiment):
  """Base GPTJ/NeoX Transformer LM configuration."""

  SPM_MODEL = '/cns/mf-d/home/huangyp/ulm/pax-gptj/tokenizer.model'
  SOS_ID = 0
  EOS_ID = 2

  # architecture related
  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  NORM_POLICY = 'pre-hybrid'
  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16

  ACTIVATION_CLS = activations.GELU
  RMS_NORM_EPSILON = 1.0e-05

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 1, 1]
  DCN_MESH_SHAPE = None
  DECODE_MESH_TRANSPOSE = None

  BATCH_SIZE = 1
  NUM_SAMPLES = 1
  ENABLE_GENERATE_STREAM = True
  STREAM_INTERVAL_STEPS = 16
  FPROP_FOR_PREFIX = True
  INPUT_SEQ_LEN = 4096
  BUCKET_KEYS = [128, 1024, 4096]
  MAX_DECODE_STEPS = [128, 512, 1024]
  EXTRA_INPUTS = {
      'temperature': 0.5,
      'per_example_max_decode_steps': 128,
      'per_example_top_k': 200,
      'per_example_top_p': 0.95,
  }

  def datasets(self) -> List[base_input.BaseInput.HParams]:
    return []

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = tasks_lib.SingleTask.HParams(name='xformer_task')
    task_p.model = pax_fiddle.Config(layers.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl.packed_input = False
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    model_p.lm_tpl.position_emb_tpl = None
    model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(
        layers.FullSoftmax,
        name='output',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
    model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
        layers.Embedding,
        name='tok_embeddings',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    model_p.lm_tpl.final_ln_tpl.epsilon = self.RMS_NORM_EPSILON

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    transformer_layer_p = cast(
        pax_fiddle.Config[ParallelTransformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.ln_tpl.epsilon = self.RMS_NORM_EPSILON
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = True
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        pax_fiddle.Config(LLaMARotaryEmbedding)
    )
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    transformer_layer_p.tr_fflayer_tpl.has_bias = True
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = False

    model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    model_p.fprop_dtype = self.FPROP_DTYPE
    model_p.dtype = self.MODEL_DTYPE

    # Set sharding
    task_p = template.set_decoding_sharding_hparams(
        task_p,
        mesh_shape=self.ICI_MESH_SHAPE,
        decode_mesh_transpose=self.DECODE_MESH_TRANSPOSE,
    )
    # Unused.
    lp = task_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.ShardedSgd.HParams(
        learning_rate=1e-3, lr_schedule=schedules.Constant.HParams()
    )
    return task_p


@servable_model_registry.register
@template.make_servable()
class LmCloudSpmd2B(lm_cloud.LmCloudSpmd2B):
  # pylint: disable=line-too-long
  """Servable config on 1x1x4.

  Checkpoint:
  gs://sax-data/lm_cloud_2b_mesh_3/1/checkpoints/checkpoint_00000000
  """
  # pylint: enable=line-too-long

  SPM_MODEL = os.path.join(os.path.dirname(__file__), 'test_model.model')
  ICI_MESH_SHAPE = [1, 1, 4]
  FPROP_FOR_PREFIX = True
  BATCH_SIZE = 1
  TRAINING_OPTIMIZED_SHARDING = False
  USE_REPEATED_LAYER = True

  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    task_p = template.set_decoding_sharding_hparams(
        task_p,
        mesh_shape=self.ICI_MESH_SHAPE,
    )
    return task_p


@servable_model_registry.register
class LmCloudSpmd2BTest(LmCloudSpmd2B):
  """2B Servable config on 1x1x1 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 1]
  TOKENIZED = True

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
class LmCloudSpmd2B4Test(LmCloudSpmd2BTest):
  """2B Servable config on 1x1x4 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 4]


@servable_model_registry.register
class LmCloudSpmd2B8Test(LmCloudSpmd2BTest):
  """2B Servable config on 1x1x8 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 8]


@servable_model_registry.register
class LmCloudSpmd2B16Test(LmCloudSpmd2BTest):
  """2B Servable config on 1x1x16 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 16]


@servable_model_registry.register
class LmCloudSpmd2B32Test(LmCloudSpmd2BTest):
  """2B Servable config on 1x1x32 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 32]


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class LmCloudSpmd175B(LmCloudSpmd2B):
  """175B on TPU v4-32.

  April 14, 2023
  Latency = 2.337s with 128 decoded tokens. 17ms per output token
  """

  NUM_LAYERS = 96
  MODEL_DIMS = 12288
  NUM_HEADS = 96
  DIMS_PER_HEAD = 128
  HIDDEN_DIMS = MODEL_DIMS * 4
  ICI_MESH_SHAPE = [1, 1, 16]

  BATCH_SIZE = 1
  NUM_SAMPLES = 1
  FPROP_FOR_PREFIX = True
  INPUT_SEQ_LEN = 128  # 4096
  BUCKET_KEYS = None  # [128, 1024, 4096]
  MAX_DECODE_STEPS = 128  # [128, 512, 1024]
  EXTRA_INPUTS = {
      'temperature': 0.5,
      'per_example_max_decode_steps': 128,
      'per_example_top_k': 200,
      'per_example_top_p': 0.95,
  }


@servable_model_registry.register
class LmCloudSpmd175BTest(LmCloudSpmd175B):
  """175B on TPU v4-32 in test mode."""

  TOKENIZED = True

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
class LmCloudSpmd175B32Test(LmCloudSpmd175BTest):
  """175B Servable config on 1x1x32 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 32]


@servable_model_registry.register
class LmCloudSpmd175B64Test(LmCloudSpmd175BTest):
  """175B Servable config on 1x1x64 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 64]


@servable_model_registry.register
class LmCloudSpmd175B128Test(LmCloudSpmd175BTest):
  """175B Servable config on 1x1x128 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 128]


@servable_model_registry.register
class LmCloudSpmd175B256Test(LmCloudSpmd175BTest):
  """175B Servable config on 1x1x256 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 256]