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
"""Language Model configurations on the T5/C4 dataset."""

from paxml.tasks.lm.params import c4
from saxml.server import servable_model_registry
from saxml.server.pax.lm.params import template


@servable_model_registry.register
@template.make_servable()
class C4SpmdGpt3AdamOrgHP(c4.C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x64."""

  # tokenization
  TOKENIZED = True
  SPM_MODEL = "gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model"

  ICI_MESH_SHAPE = [1, 1, 64]
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


class C4SpmdGpt3AdamOrgHP32(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x32."""

  ICI_MESH_SHAPE = [1, 1, 32]

class C4SpmdGpt3AdamOrgHP128(C4SpmdGpt3AdamOrgHP):
  """175B GPT-3 Servable config on 1x1x128."""

  ICI_MESH_SHAPE = [1, 1, 128]
