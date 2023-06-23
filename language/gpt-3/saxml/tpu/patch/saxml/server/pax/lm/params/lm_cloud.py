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

from paxml import tasks_lib
from paxml.tasks.lm.params import c4
from paxml.tasks.lm.params import lm_cloud
from praxis import pax_fiddle
from saxml.server import servable_model_registry
from saxml.server.pax import quantization
from saxml.server.pax.lm.params import template


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

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
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
  """175B in test mode."""

  TOKENIZED = True

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
class LmCloudSpmd175B8Test(LmCloudSpmd175BTest):
  """175B Servable config on 1x1x8 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 8]


@servable_model_registry.register
class LmCloudSpmd175B16Test(LmCloudSpmd175BTest):
  """175B Servable config on 1x1x16 in test mode."""

  ICI_MESH_SHAPE = [1, 1, 16]


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


@template.make_servable()
class C4SpmdGpt3AdamOrgHP(c4.C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config."""

  # tokenization
  TOKENIZED = True
  SPM_MODEL = "gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model"

  FPROP_FOR_PREFIX = True
  BATCH_SIZE = 4
  TRAINING_OPTIMIZED_SHARDING = False
  USE_REPEATED_LAYER = True

  TRAINABLE_POSITION_EMB = True
  EMBEDDING_LOOKUP_STYLE = "index"

  # decoding
  INPUT_SEQ_LEN = 2048
  MAX_DECODE_STEPS = 128
  NUM_SAMPLES = 1
  USE_BEAM_SEARCH = True
  BEAM_SIZE = 4


@servable_model_registry.register
class C4SpmdGpt3AdamOrgHP8(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x8."""

  ICI_MESH_SHAPE = [1, 1, 8]


@servable_model_registry.register
class C4SpmdGpt3AdamOrgHP16(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x16."""

  ICI_MESH_SHAPE = [1, 1, 16]


@servable_model_registry.register
class C4SpmdGpt3AdamOrgHP32(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x32."""

  ICI_MESH_SHAPE = [1, 1, 32]


@servable_model_registry.register
class C4SpmdGpt3AdamOrgHP64(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x32."""

  ICI_MESH_SHAPE = [1, 1, 64]


@servable_model_registry.register
class C4SpmdGpt3AdamOrgHP128(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x128."""

  ICI_MESH_SHAPE = [1, 1, 128]


@servable_model_registry.register
class C4SpmdGpt3AdamOrgHP256(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x256."""

  ICI_MESH_SHAPE = [1, 1, 256]
