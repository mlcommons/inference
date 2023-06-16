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
"""Serving template params."""

from typing import cast, Dict, Optional, Sequence, Type

import numpy as np
from paxml import base_task
from paxml import tasks_lib
from praxis import base_layer
from praxis import decoder_hparams
from praxis import pax_fiddle
from praxis import py_utils
from praxis.layers import attentions
from praxis.layers import multi_query_attention
from praxis.layers import transformer_models
from praxis.layers import transformers
from saxml.server import servable_model_registry
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_model

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

# pytype: disable=attribute-error


class CommonServingTemplate:
  """Common Serving template for language models."""

  ICI_MESH_SHAPE = [1, 1, 8]
  USE_BEAM_SEARCH = False
  BATCH_SIZE = 1
  BATCH_WAIT_SECS = None
  INPUT_SEQ_LEN = 256
  SUFFIX_SEQ_LEN = 0  # Deprecating this attribute.
  MAX_DECODE_STEPS = 32
  NUM_SAMPLES = 2
  TOP_K = 40
  TOP_K_RECALL_TARGET = 1.0  # When < 1.0, use tpu optimized approx_max_k
  USE_TOP_K_FOR_LOGPROBS = False
  BEAM_SIZE = 4
  FPROP_FOR_PREFIX = False
  GLOBAL_NORMALIZE = False
  VOCAB_SIZE = 32000
  LENGTH_NORM_ALPHA = 0.8
  SCORE_ONLY = False
  GENERATE_ONLY = False
  SPM_MODEL = None
  TOKENIZED = False  # When the input and output are in tokens instead of text
  SOS_ID = 0
  EOS_ID = 1
  STOP_TOKEN_IDS = None
  SLICE_LEFT = True
  EXTRA_INPUTS = {'temperature': 0.1}
  EXTRA_INPUTS_DTYPES = {}
  SCORE_EXTRA_INPUTS = {}
  BUCKET_KEYS = None
  INCLUDE_PREFIX_IN_RESULT = False
  MAX_LIVE_BATCHES = 4
  ENABLE_GENERATE = True
  ENABLE_GENERATE_STREAM = False
  STREAM_INTERVAL_STEPS = 1
  DECODE_MESH_TRANSPOSE = None
  # Remove this after MultipQueryAttention supports lazy prefix.
  SUPPORT_LAZY_PREFIX_BROADCAST = True
  EMB_LOOKUP_STYLE = 'index'
  FETCH_PREFIX_LENGTHS_FROM_INPUTS = False
  POLYMORPHIC_SEQ_LEN_EXCLUSION = None
  SORT_SAMPLES = True
  TEXT_TO_EMBEDDING_SERVING_BATCH_SIZE = 1
  TEXT_TO_EMBEDDING_MAX_LIVE_BATCHES = 1
  TEXT_TO_EMBEDDING_OUTPUT_EMBEDDING_NAME = None
  TEXT_TO_EMBEDDING_MODEL_METHOD_NAME = None
  GENERATION_USE_GEOMEAN_PROB_SCORE = False
  SCORING_USE_GEOMEAN_PROB_SCORE = False
  SCORING_INCLUDE_EOS_SCORE = False

  def input_for_model_init(self) -> py_utils.NestedMap:
    batch_size = self.BATCH_SIZE
    if isinstance(batch_size, (list, tuple)):
      batch_size = batch_size[0]
    seq_len = self.INPUT_SEQ_LEN
    targets = np.ones([batch_size, seq_len], dtype=np.int32)
    input_batch = py_utils.NestedMap()
    input_batch.ids = targets
    input_batch.paddings = np.zeros_like(targets)
    input_batch.inputs_indicator = np.ones_like(targets)
    input_batch.weights = np.ones_like(targets)
    input_batch.labels = targets
    input_batch.segment_ids = targets
    input_batch.segment_pos = np.tile(
        np.arange(0, seq_len)[np.newaxis, :], [batch_size, 1]
    )
    return input_batch

  @classmethod
  def serving_mesh_shape(cls):
    return cls.ICI_MESH_SHAPE


class ServingTemplate(
    CommonServingTemplate, servable_lm_model.ServableLMModelParams
):
  """Template servable config.

  Language model parameters can be found at go/sax-lm-decode-params.
  """

  def score(self) -> Optional[servable_lm_model.ScoreHParams]:
    if self.GENERATE_ONLY:
      return None
    input_seq_len = self.INPUT_SEQ_LEN
    suffix_seq_len = self.SUFFIX_SEQ_LEN
    return servable_lm_model.ScoreHParams(
        batch_size=self.BATCH_SIZE,
        polymorphic_seq_len_exclusion=self.POLYMORPHIC_SEQ_LEN_EXCLUSION,
        max_input_seq_len=input_seq_len,
        max_suffix_seq_len=suffix_seq_len,
        bucket_keys=self.BUCKET_KEYS,
        extra_inputs=self.SCORE_EXTRA_INPUTS,
        include_eos_score=self.SCORING_INCLUDE_EOS_SCORE,
        extra_inputs_dtypes=self.EXTRA_INPUTS_DTYPES,
        fetch_prefix_lengths_from_inputs=self.FETCH_PREFIX_LENGTHS_FROM_INPUTS,
        output_geometric_mean_prob_score=self.SCORING_USE_GEOMEAN_PROB_SCORE,
    )

  def serving_tokenizer(self):
    spm_model = (
        self.SPM_MODEL
        if self.SPM_MODEL is not None
        else self._dataset_train().input.tokenizer.spm_model
    )

    return pax_fiddle.Config(
        lm_tokenizer.LMTokenizer,
        spm_model=spm_model,
        target_sos_id=self.SOS_ID,
        target_eos_id=self.EOS_ID,
        slice_left=self.SLICE_LEFT,
        tokenized=self.TOKENIZED,
    )

  def generate(self) -> Optional[servable_lm_model.DecodeHParams]:
    max_decode_steps = (
        max(self.MAX_DECODE_STEPS)
        if isinstance(self.MAX_DECODE_STEPS, list)
        else self.MAX_DECODE_STEPS
    )
    stop_token_ids = (
        self.STOP_TOKEN_IDS if self.STOP_TOKEN_IDS else [self.EOS_ID]
    )
    if not self.ENABLE_GENERATE:
      return None

    if self.SCORE_ONLY:
      return None

    if self.USE_BEAM_SEARCH:
      generate_hparams = decoder_hparams.BeamSearchHParams(
          fprop_for_prefix=True,
          max_decode_steps=self.MAX_DECODE_STEPS,
          seqlen=self.INPUT_SEQ_LEN + max_decode_steps,
          beam_size=self.BEAM_SIZE,
          eos_id=stop_token_ids,
          length_norm_alpha=self.LENGTH_NORM_ALPHA,
          decode_loop_mesh_axes_transpose=self.DECODE_MESH_TRANSPOSE,
          emb_lookup_style=self.EMB_LOOKUP_STYLE,
      )
    elif self.NUM_SAMPLES == 1 and self.TOP_K == 1:
      generate_hparams = decoder_hparams.GreedyDecoderHParams(
          fprop_for_prefix=self.FPROP_FOR_PREFIX,
          max_decode_steps=self.MAX_DECODE_STEPS,
          seqlen=self.INPUT_SEQ_LEN + max_decode_steps,
          eos_id=stop_token_ids,
          decode_loop_mesh_axes_transpose=self.DECODE_MESH_TRANSPOSE,
          emb_lookup_style=self.EMB_LOOKUP_STYLE,
      )
    else:
      generate_hparams = decoder_hparams.SampleDecoderHParams(
          fprop_for_prefix=self.FPROP_FOR_PREFIX,
          # Use LPB for whenever FPROP_FOR_PREFIX is enabled.
          lazy_prefix_broadcast=self.FPROP_FOR_PREFIX
          and self.NUM_SAMPLES > 1
          and self.SUPPORT_LAZY_PREFIX_BROADCAST,
          max_decode_steps=self.MAX_DECODE_STEPS,
          seqlen=self.INPUT_SEQ_LEN + max_decode_steps,
          num_samples=self.NUM_SAMPLES,
          temperature=0.0,
          eos_id=stop_token_ids,
          k=self.TOP_K,
          top_k_recall_target=self.TOP_K_RECALL_TARGET,
          use_top_k_for_logprobs=self.USE_TOP_K_FOR_LOGPROBS,
          global_normalize=self.GLOBAL_NORMALIZE,
          decode_loop_mesh_axes_transpose=self.DECODE_MESH_TRANSPOSE,
          emb_lookup_style=self.EMB_LOOKUP_STYLE,
          sort_samples=self.SORT_SAMPLES,
      )
    return servable_lm_model.DecodeHParams(
        batch_size=self.BATCH_SIZE,
        polymorphic_seq_len_exclusion=self.POLYMORPHIC_SEQ_LEN_EXCLUSION,
        max_input_seq_len=self.INPUT_SEQ_LEN,
        bucket_keys=self.BUCKET_KEYS,
        decoder=generate_hparams,
        include_prefix_in_result=self.INCLUDE_PREFIX_IN_RESULT,
        max_live_batches=self.MAX_LIVE_BATCHES,
        batching_wait_secs=self.BATCH_WAIT_SECS,
        extra_inputs=self.EXTRA_INPUTS,
        extra_inputs_dtypes=self.EXTRA_INPUTS_DTYPES,
        fetch_prefix_lengths_from_inputs=self.FETCH_PREFIX_LENGTHS_FROM_INPUTS,
        output_geometric_mean_prob_score=self.GENERATION_USE_GEOMEAN_PROB_SCORE,
    )

  def generate_stream(self) -> Optional[servable_lm_model.DecodeHParams]:
    max_decode_steps = (
        max(self.MAX_DECODE_STEPS)
        if isinstance(self.MAX_DECODE_STEPS, list)
        else self.MAX_DECODE_STEPS
    )
    stop_token_ids = (
        self.STOP_TOKEN_IDS if self.STOP_TOKEN_IDS else [self.EOS_ID]
    )
    if not self.ENABLE_GENERATE_STREAM:
      return None

    if self.SCORE_ONLY:
      return None

    if self.USE_BEAM_SEARCH:
      return None

    generate_hparams = decoder_hparams.SampleDecoderHParams(
        fprop_for_prefix=self.FPROP_FOR_PREFIX,
        # Use LPB for whenever FPROP_FOR_PREFIX is enabled.
        lazy_prefix_broadcast=self.FPROP_FOR_PREFIX
        and self.NUM_SAMPLES > 1
        and self.SUPPORT_LAZY_PREFIX_BROADCAST,
        max_decode_steps=self.MAX_DECODE_STEPS,
        seqlen=self.INPUT_SEQ_LEN + max_decode_steps,
        num_samples=self.NUM_SAMPLES,
        temperature=None,
        eos_id=stop_token_ids,
        k=self.TOP_K,
        emb_lookup_style=self.EMB_LOOKUP_STYLE,
        sort_samples=self.SORT_SAMPLES,
    )

    return servable_lm_model.DecodeHParams(
        batch_size=self.BATCH_SIZE,
        polymorphic_seq_len_exclusion=self.POLYMORPHIC_SEQ_LEN_EXCLUSION,
        max_input_seq_len=self.INPUT_SEQ_LEN,
        bucket_keys=self.BUCKET_KEYS,
        decoder=generate_hparams,
        include_prefix_in_result=self.INCLUDE_PREFIX_IN_RESULT,
        max_live_batches=self.MAX_LIVE_BATCHES,
        extra_inputs=self.EXTRA_INPUTS,
        extra_inputs_dtypes=self.EXTRA_INPUTS_DTYPES,
        stream_interval_steps=self.STREAM_INTERVAL_STEPS,
        fetch_prefix_lengths_from_inputs=self.FETCH_PREFIX_LENGTHS_FROM_INPUTS,
    )

  def text_to_embedding(
      self,
  ) -> Optional[servable_lm_model.TextToEmbeddingHParams]:
    if (
        self.GENERATE_ONLY
        or self.TEXT_TO_EMBEDDING_MODEL_METHOD_NAME is None
        or self.TEXT_TO_EMBEDDING_OUTPUT_EMBEDDING_NAME is None
    ):
      return None

    assert (
        self.POLYMORPHIC_SEQ_LEN_EXCLUSION is None
    ), 'This needs to be understood and supported if needed.'

    input_seq_len = self.INPUT_SEQ_LEN - 1
    return servable_lm_model.TextToEmbeddingHParams(
        batch_size=self.TEXT_TO_EMBEDDING_SERVING_BATCH_SIZE,
        max_input_seq_len=input_seq_len,
        max_live_batches=self.TEXT_TO_EMBEDDING_MAX_LIVE_BATCHES,
        output_embedding_name=self.TEXT_TO_EMBEDDING_OUTPUT_EMBEDDING_NAME,
        model_method_name=self.TEXT_TO_EMBEDDING_MODEL_METHOD_NAME,
        extra_inputs_dtypes=self.EXTRA_INPUTS_DTYPES,
    )


class ServingWithGradientTemplate(ServingTemplate):
  """Template servable config with gradient method."""

  INCLUDE_EOS_SCORE = False
  GRADIENT_WRT_INPUT_TENSOR_NAMES = None
  GRADIENT_WRT_MDL_VAR_TENSOR_NAMES = None

  def gradient(self) -> Optional[servable_lm_model.GradientHParams]:
    if self.GRADIENT_WRT_MDL_VAR_TENSOR_NAMES:
      raise ValueError(
          'Running graident method with gradients to model is not '
          'supported since it is undefined '
          'how to introduce the batch dims for export signatures.'
      )

    input_seq_len = self.INPUT_SEQ_LEN
    suffix_seq_len = self.SUFFIX_SEQ_LEN
    return servable_lm_model.GradientHParams(
        batch_size=self.BATCH_SIZE,
        polymorphic_seq_len_exclusion=self.POLYMORPHIC_SEQ_LEN_EXCLUSION,
        max_input_seq_len=input_seq_len,
        max_suffix_seq_len=suffix_seq_len,
        bucket_keys=self.BUCKET_KEYS,
        include_eos_score=self.INCLUDE_EOS_SCORE,
        inputs_tensor_names=self.GRADIENT_WRT_INPUT_TENSOR_NAMES,
        mdl_vars_tensor_names=None,
        extra_inputs=self.SCORE_EXTRA_INPUTS,
        extra_inputs_dtypes=self.EXTRA_INPUTS_DTYPES,
    )


def set_lazy_prefix_broadcast_params(lm_tpl: LayerTpl) -> None:
  """Set params to enable lazy prefix broadcast for attention."""
  xformer = lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
  if xformer.cls == transformers.StackedTransformerRepeated:
    xformer = xformer.block
  assert xformer.cls == transformers.StackedTransformer
  layer_p = xformer.transformer_layer_params_tpl
  lbp_tr_atten_tpl = pax_fiddle.Config(
      attentions.DotProductAttentionWithLPB,
  )
  mqa_cls = multi_query_attention.MultiQueryDotProductAttention
  mqs_lpb_cls = multi_query_attention.MultiQueryDotProductAttentionLPB
  lbp_multi_query_atten_tpl = pax_fiddle.Config(
      mqs_lpb_cls,
  )
  if issubclass(layer_p.tr_atten_tpl.cls, attentions.DotProductAttention):
    lbp_tr_atten_tpl.copy_fields_from(layer_p.tr_atten_tpl)
    layer_p.tr_atten_tpl = lbp_tr_atten_tpl
  elif issubclass(layer_p.tr_atten_tpl.cls, mqa_cls):
    lbp_multi_query_atten_tpl.copy_fields_from(layer_p.tr_atten_tpl)
    layer_p.tr_atten_tpl = lbp_multi_query_atten_tpl
  else:
    assert layer_p.tr_atten_tpl.cls == lbp_tr_atten_tpl.cls, (
        'Attention layer does not support lazy prefix broadcast '
        f'{layer_p.tr_atten_tpl.cls}.'
    )


def make_servable(servable_class=ServingTemplate):
  """Returns a class decorator that wraps a PAX experiment to a servable.

  It is a tool to create multi-inheritance and common serving params overrides
  on Praxis LanguageModel:
    - Disable packed inputs.
    - Efficient multi-sample decoding with lazy prefix broadcast.

  If you don't need the overrides, you can use multi-inheritance directly
  without this decorator.

  One benefit of using this decorator is that you only need to apply it to a
  base class. Subclasses of the wrapped base class do not need to be wrapped
  again, and they can directly override members in the servable_class like
  BATCH_SIZE.

  Args:
    servable_class: The class the implements ServableLMModelParams.

  Returns:
    A decorator that generates a subclass that inherits both a Pax experiment
    and servable_class.
  """

  def _decorator(pax_exp_class):
    # pax_exp_class comes before servable_class so that overrides in
    # pax_exp_class are used.

    class Wrapped(pax_exp_class, servable_class):
      """A wrapper that uses the template and overrides some common LM configs."""

      @classmethod
      def sax_registration_name(cls) -> Optional[str]:
        # Do not change the registration path for servable_class.
        if cls == Wrapped:
          return servable_model_registry.full_registration_name(pax_exp_class)
        # cls is a subclass of sax_registration_name defined somewhere else.
        return None

      def task(self) -> pax_fiddle.Config[base_task.BaseTask]:
        task_p = super().task()

        if not hasattr(task_p, 'model'):
          return task_p

        if not hasattr(task_p.model, 'lm_tpl'):
          return task_p
        # Disable packed input for online inference.
        task_p.model.lm_tpl.packed_input = False
        # Override attention with lazy prefix broadcast.
        lazy_prefix_broadcast = False
        decode_params = self.generate()
        if (
            hasattr(task_p.model.lm_tpl, 'softmax_tpl')
            and hasattr(task_p.model.lm_tpl.softmax_tpl, 'lookup_style')
            and hasattr(decode_params, 'decoder')
        ):
          task_p.model.lm_tpl.softmax_tpl.lookup_style = (
              decode_params.decoder.emb_lookup_style
          )
        if decode_params is not None:
          if decode_params.decoder.lazy_prefix_broadcast:
            assert decode_params.decoder.num_samples > 1  # pytype: disable=attribute-error
            lazy_prefix_broadcast = True

        if lazy_prefix_broadcast:
          set_lazy_prefix_broadcast_params(task_p.model.lm_tpl)  # pytype: disable=attribute-error  # enable-nested-classes
        return task_p

    return Wrapped

  return _decorator


def set_decoding_sharding_hparams(
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
    mesh_shape: Sequence[int],
    decode_mesh_transpose: Optional[Dict[str, str]] = None,
    mqa_kv_state_batch_sharding: Optional[bool] = False,
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Set spmd sharding params that works for decoding.

  Args:
    task_p: The task params for ULM.
    mesh_shape: The 3D mesh shape (replica, data/mdl2, mdl)
    decode_mesh_transpose: If not None, 2 extra dims (fprop_data, fprop_mdl)
      will the mesh shape will be added to the mesh dim, and the model will use
      fprop/training-optimized sharding except for the prefix, and switch to
      decoding-optimized sharding for the decoding loop.
    mqa_kv_state_batch_sharding: Shard along batch dim for MQA k,v activations.

  Returns:
    The updated task_p.
  """
  model_p = task_p.model
  replica_axis = 'replica'
  # We repurpose `data_mdl2` as the secondary model-parallel dim, but attention
  # state blnh's b dim is also fully sharded on both `data_mdl2` and `replica`.
  # Other activations' batch dim are not sharded by `data_mdl2`.
  data_mdl2_axis = 'data_mdl2'
  mdl_axis = 'mdl'
  mesh_axis_names = [replica_axis, data_mdl2_axis, mdl_axis]
  assert len(mesh_shape) == 3
  if decode_mesh_transpose:
    # Mesh axes that combines fprop and decode.
    mesh_axis_names += ['fprop_data', 'fprop_mdl']
    # Mesh shape is set up for fprop.
    mesh_shape = [mesh_shape[0], 1, 1] + list(mesh_shape[1:])
  model_p.ici_mesh_shape = mesh_shape
  model_p.dcn_mesh_shape = None
  model_p.mesh_axis_names = mesh_axis_names

  def maybe_fprop_data(*axes):
    if not decode_mesh_transpose:
      return axes
    return axes + ('fprop_data',)

  def maybe_fprop_mdl(*axes):
    if not decode_mesh_transpose:
      return axes
    return axes + ('fprop_mdl',)

  # ffn0 weight
  w_df = [maybe_fprop_data(data_mdl2_axis), maybe_fprop_mdl(mdl_axis)]
  # Attention weight.
  w_dnh = [maybe_fprop_data(data_mdl2_axis), maybe_fprop_mdl(mdl_axis), None]
  # embedding weight
  w_vd = [maybe_fprop_mdl(mdl_axis), maybe_fprop_data(data_mdl2_axis)]
  # Activations.
  a_bld = [
      maybe_fprop_data(replica_axis),
      None,
      maybe_fprop_mdl(data_mdl2_axis),
  ]
  a_blf = [maybe_fprop_data(replica_axis), None, maybe_fprop_mdl(mdl_axis)]
  # Attention state shards batch also by `data_mdl2`.
  a_blnh = [
      maybe_fprop_data(replica_axis, data_mdl2_axis),
      None,
      maybe_fprop_mdl(mdl_axis),
      None,
  ]
  if mqa_kv_state_batch_sharding and not decode_mesh_transpose:
    # KV Attention state for multi-query shards batch by `mdl_axis`.
    a_blh = [(replica_axis, mdl_axis), None, data_mdl2_axis]
  else:
    a_blh = None
  a_blv = [maybe_fprop_data(replica_axis), None, maybe_fprop_mdl(mdl_axis)]

  lm_cls = cast(
      Type[transformer_models.TransformerLm],
      pax_fiddle.get_callable(model_p.lm_tpl),
  )
  model_p.lm_tpl = lm_cls.set_custom_sharding_params(
      model_p.lm_tpl,
      ici_mesh_shape=mesh_shape,
      dcn_mesh_shape=None,
      mesh_axis_names=mesh_axis_names,
      w_df=w_df,
      w_dnh=w_dnh,
      w_vd=w_vd,
      a_bld=a_bld,
      a_blf=a_blf,
      a_blh=a_blh,
      a_blnh=a_blnh,
      a_blv=a_blv,
  )
  # Transpose the mesh axes for the decoding loop.
  model_p.decoder_tpl.decode_loop_mesh_axes_transpose = decode_mesh_transpose

  # Set input sharding params.
  task_p.train.inputs_split_mapping = py_utils.NestedMap(
      map_1d=(maybe_fprop_data(replica_axis),),
      map_2d=(maybe_fprop_data(replica_axis), None),
  )

  return task_p