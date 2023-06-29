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
"""Wraps a model with LMService APIs."""

import abc
import functools
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import host_callback as hcb
import numpy as np
from praxis import base_layer
from praxis import base_model
from praxis import decoder_hparams
from praxis import decoder_utils
from praxis import py_utils
from praxis import pytypes
from saxml.server.jax import np_tf_sess_wrapper
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_common
from saxml.server.services import lm_service
import tensorflow as tf

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedNpOrTfTensor = Union[NestedNpTensor, NestedTfTensor]
LMMethodName = lm_service.LMMethodName
HostTensors = servable_model.HostTensors
ShapesAndDtypes = servable_model.ShapesAndDtypes
InputShapeInfo = servable_lm_common.InputShapeInfo

decode_tf_post_processing = servable_lm_common.decode_tf_post_processing


class ScoreHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM score method.

  Attributes:
    max_input_seq_len: static prefix sequence length dimension size.
    max_suffix_seq_len: static suffix sequence length dimension size. Defaults
      to be equal to `max_input_seq_len` if not set. Inputs are padded or
      truncated to (max_input_seq_len + max_suffix_seq_len) size.
    include_eos_score: whether to add EOS score to the result.
  """

  max_input_seq_len: int = 0
  max_suffix_seq_len: int = 0
  include_eos_score: bool = False
  fetch_prefix_lengths_from_inputs: bool = False


class DecodeHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM sample decode method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    decoder: decoder params.
    include_prefix_in_result: whether to include the input prefix in the result.
    encoder_decoder_model: whether this is an encoder decoder model.
    t5_model: whether this is a T5 flaxformer based model.
  """

  max_input_seq_len: int = 0
  decoder: decoder_hparams.DecoderHParams = decoder_hparams.DecoderHParams()
  include_prefix_in_result: bool = False
  encoder_decoder_model: bool = False
  t5_model: bool = False
  stream_interval_steps: int = 1
  fetch_prefix_lengths_from_inputs: bool = False


class TextToEmbeddingHParams(servable_model_params.ServableMethodParams):
  """HParameters for TextToEmbedding method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    output_embedding_name: The name of the embedding to use from the model's
      outputs.  Required.
    model_method_name: The name of the method to call to extract embeddings from
      an input image.  Required.
  """

  max_input_seq_len: int = 0
  output_embedding_name: Optional[str] = None
  model_method_name: Optional[str] = None


class GradientHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM gradient method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    include_eos_score: whether to add EOS score to the result.
    inputs_tensor_names: tensors to take gradients with respect to in inputs.
    mdl_vars_tensors_names: tensors to take gradients with respect to in
      mdl_vars.
  """

  max_input_seq_len: int = 0
  max_suffix_seq_len: int = 0
  include_eos_score: bool = False
  inputs_tensor_names: Optional[List[str]] = None
  mdl_vars_tensor_names: Optional[List[str]] = None


class ServableLMModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta
):
  """A base class that each LM model config needs to implement for serving."""

  @abc.abstractmethod
  def serving_tokenizer(self) -> lm_tokenizer.LMTokenizer.HParams:
    """Tokenizer params used by serving."""

  def methods(self) -> Dict[str, servable_model_params.ServableMethodParams]:
    methods = {}
    score = self.score()  # pylint: disable=assignment-from-none
    if score is not None:
      methods[LMMethodName.SCORE] = score
    generate = self.generate()  # pylint: disable=assignment-from-none
    if generate is not None:
      methods[LMMethodName.GENERATE] = generate
    generate_stream = self.generate_stream()  # pylint: disable=assignment-from-none
    if generate_stream is not None:
      methods[LMMethodName.GENERATE_STREAM] = generate_stream
    text_to_embedding = self.text_to_embedding()  # pylint: disable=assignment-from-none
    if text_to_embedding is not None:
      methods[LMMethodName.EMBED] = text_to_embedding
    gradient = self.gradient()  # pylint: disable=assignment-from-none
    if gradient is not None:
      methods[LMMethodName.GRADIENT] = gradient
    return methods

  def score(self) -> Optional[ScoreHParams]:
    """Returns the params for the score method."""
    return None

  def generate(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def gradient(self) -> Optional[GradientHParams]:
    """Returns the params for the gradient method."""
    return None

  def generate_stream(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def text_to_embedding(self) -> Optional[TextToEmbeddingHParams]:
    return None

  def create_model(self, primary_process_id: int) -> 'ServableLMModel':
    return ServableLMModel(
        self,
        primary_process_id,
        self.get_checkpoint_type(),
        test_mode=self.test_mode,
    )


class ServableLMMethod(servable_model.ServableMethod):
  """Implements common method of LM."""

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  @property
  def sorted_seq_lens(self) -> List[int]:
    """A list of sorted supported (ascending order) sequence lengths."""
    return sorted(self._bucket_keys) if self._bucket_keys else [-1]

  def get_sorted_input_shapes(self) -> List[InputShapeInfo]:
    result = []
    for batch_size in self._sorted_batch_sizes:
      for seq_len in self.sorted_seq_lens:
        result.append(InputShapeInfo(batch_size, seq_len))
    return result

  def deserialize_input_shape(self, unpadded_shape_str: str) -> InputShapeInfo:
    """Deserialize input shape from a str."""
    return servable_lm_common.deserialize_input_shape(
        unpadded_shape_str, self._dummy_bucket_key
    )

  def get_unpadded_shape(
      self, unpadded_batch_size, inputs: HostTensors
  ) -> InputShapeInfo:
    return InputShapeInfo(
        unpadded_batch_size,
        servable_lm_common.get_max_seq_len_in_batch(
            inputs, self._dummy_bucket_key, self._bucket_keys
        ),
    )

  def get_padded_input_shape(
      self, unpadded_shape: InputShapeInfo
  ) -> InputShapeInfo:
    """Get padded input shape.

    Args:
      unpadded_shape: Unpadded shape information contains batch size or sequence
        length.

    Returns:
      Padded input shape.
    Raises:
      ValueError if unpadded batch size or sequence length too large.
    """
    padded_shape = super().get_padded_input_shape(unpadded_shape)
    if self._bucket_keys is None:
      return InputShapeInfo(padded_shape.batch_size)
    padded_seq_len = servable_lm_common.get_padded_input_seq_len(
        unpadded_shape.seq_len, self.sorted_seq_lens
    )
    return InputShapeInfo(padded_shape.batch_size, padded_seq_len)

  def get_dummy_inputs(self, input_shape: InputShapeInfo) -> HostTensors:
    """Returns host tensors with dummy data at a batch size."""
    batched_input = self.pre_processing(
        [self._dummy_input_sample] * input_shape.batch_size
    )

    return servable_lm_common.handle_host_input_with_input_shape(
        batched_input, input_shape
    )

  def resize_host_array(
      self,
      x: np.ndarray,
      global_input_shape_dtype: ShapesAndDtypes,
      unpadded_input_shape: InputShapeInfo,
  ):
    """Resizes x to the desired shape.

    Args:
      x: Host tensor.
      global_input_shape_dtype: Global input shape and dtype for this tensor.
      unpadded_input_shape: Unpadded input shape.

    Returns:
      host array after padding or slice of x.
    """
    x = servable_lm_common.resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )

    # Let the parent class handle the batch dim.
    x = super().resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )
    return x

  def _get_longest_seqlen(self, inputs: NestedNpTensor) -> int:
    """Gets the longest sequence length in a batch."""
    if 'paddings' in inputs:
      prefix_lengths = np.sum(1.0 - inputs['paddings'], axis=-1).astype(
          np.int32
      )  # pytype: disable=attribute-error
      return np.max(prefix_lengths).item()
    return inputs['ids'].shape[1]

  def get_unpadded_branch_key(self, inputs: NestedNpTensor) -> int:
    return self._get_longest_seqlen(inputs)

  def get_branch_inputs(
      self, inputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Returns the inputs for a branch key.

    Args:
      inputs: inputs with padded sequence lengths.
      branch_key: branch_key is seqlen.

    Returns:
      Tensors sliced at sequence length dimension.
    """
    seqlen = branch_key

    def _slice_fn(x):
      """The function to slice at sequence dimension."""
      if not isinstance(x, JTensor):
        return x
      if len(x.shape) == 2 and x.shape[1] >= seqlen:
        return jax.lax.slice(x, [0, 0], [x.shape[0], seqlen])
      return x

    return jax.tree_util.tree_map(_slice_fn, inputs)

  def get_maxlen(self) -> int:
    """Gets the max input sequence lengths."""
    raise NotImplementedError('get_maxlen not implemented')

  def output_seq_dim(self) -> int:
    """Gets the sequence dim in the output result."""
    raise NotImplementedError('output_seq_dim not implemented')

  def extra_pad_result(
      self, result: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Special paddings for some tensors."""
    return result

  def pad_result(
      self, result: NestedJTensor, pad_len: int, seq_dim: int
  ) -> NestedJTensor:
    """Pads the result at sequence dimension."""

    def _pad_fn(x):
      if not isinstance(x, JTensor) or len(x.shape) < seq_dim + 1:
        return x
      paddings = [[0, 0]] * len(x.shape)
      paddings[seq_dim] = [0, max(0, pad_len)]
      padded = jnp.pad(x, paddings)
      return padded

    return jax.tree_map(_pad_fn, result)

  def post_process_branch_outputs(
      self, outputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Post process branch outputs."""
    seqlen = branch_key
    maxlen = self.get_maxlen()
    result, state = outputs
    padded_result = self.pad_result(
        result, maxlen - seqlen, self.output_seq_dim()
    )
    padded_result = self.extra_pad_result(padded_result, branch_key)
    padded_state = self.pad_result(state, maxlen - seqlen, 1)
    return padded_result, padded_state

  @property
  def model_fn_input_polymorphic_shape(self) -> pytypes.Nested[str]:
    """Returns a batch polymorphic shape for jax2tf."""
    batched_host_dummy = self.get_dummy_inputs(InputShapeInfo(self.batch_size))
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        self.batch_size,
        [self.default_extra_inputs] * self.batch_size,
    )

    batch_pattern = 'b' if len(self.sorted_batch_sizes) > 1 else '_'
    if len(self.sorted_seq_lens) > 1:
      seq_pattern = f'{batch_pattern}, t'
    else:
      seq_pattern = f'{batch_pattern}, _'
    shape_patterns = jax.tree_util.tree_map(
        lambda x: seq_pattern if len(x.shape) == 2 else f'{batch_pattern}, ...',
        batched_host_dummy,
    )
    # apply seq len polymorphism exclusion.
    if self.method_params.polymorphic_seq_len_exclusion:
      for key in self.method_params.polymorphic_seq_len_exclusion:
        shape_patterns[key] = f'{batch_pattern}, ...'

    return shape_patterns


class LMScoreMethod(ServableLMMethod):
  """Implements the score method of an LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      score_params: ScoreHParams,
      tokenizer_p: Any,
      exportable: bool = False,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._score_params = score_params
    dummy_input_sample = ('1', ['1'])
    logging.info('Using np_tf_sess_wrapper on LMScoreMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    super().__init__(
        model,
        'compute_predictions',
        model_state,
        score_params,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
    )

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    if 'scores' in model_fn_outputs[0]:
      # Custom scores.
      return model_fn_outputs[0]['scores']
    # per_token_xent or per_example_xnent is -logprobs. We return the negative
    # value so that higher score is better.
    if 'per_token_xent' not in model_fn_outputs[0]:
      assert 'per_example_xent' in model_fn_outputs[0]
      assert model_fn_outputs[0].per_example_xent.ndim == 1  # pytype: disable=attribute-error  # jax-ndarray
      return -model_fn_outputs[0].per_example_xent  # pytype: disable=attribute-error  # jax-ndarray
    assert len(model_fn_outputs[0].per_token_xent.shape) > 1  # pytype: disable=attribute-error  # jax-ndarray
    xnent_len = model_fn_outputs[0].per_token_xent.shape[1]  # pytype: disable=attribute-error  # jax-ndarray
    assert xnent_len == model_fn_inputs.ids.shape[1]  # pytype: disable=attribute-error  # jax-ndarray
    per_token_logprobs = -model_fn_outputs[0].per_token_xent  # pytype: disable=attribute-error  # jax-ndarray
    non_paddings = 1.0 - model_fn_inputs.paddings  # pytype: disable=attribute-error  # jax-ndarray
    if (
        not self._score_params.include_eos_score
        and self._tokenizer.hparams.append_eos
    ):
      non_paddings = jnp.pad(
          # TODO(b/263808957): change back to non_paddings[:, 1:] once the bug
          # is fixed.
          jax.lax.dynamic_slice_in_dim(
              non_paddings, 1, non_paddings.shape[1] - 1, axis=1
          ),
          [[0, 0], [0, 1]],
      )
    return jnp.sum(
        per_token_logprobs * model_fn_inputs.score_masks * non_paddings,  # pytype: disable=attribute-error  # jax-ndarray
        axis=-1,
        keepdims=True,
    )

  def get_maxlen(self) -> int:
    return (
        self._score_params.max_input_seq_len
        + self._score_params.max_suffix_seq_len
    )

  def output_seq_dim(self) -> int:
    return 1

  def pre_processing(
      self, raw_inputs: List[Tuple[str, List[str]]]
  ) -> NestedNpTensor:
    prefixes = np.array([prefix for prefix, _ in raw_inputs])
    for _, suffix in raw_inputs:
      assert len(suffix) <= 1, 'Only one suffix score is supported in lm.score'
    suffixes = np.array([suffix[0] for _, suffix in raw_inputs])
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[float]:
    assert isinstance(compute_outputs, pytypes.NpTensor)
    scores = list(compute_outputs.astype(float))
    return scores

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
      extra_inputs: Optional[Mapping[str, Any]] = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    (ids, labels, paddings, weights, score_masks, inputs_indicator) = (
        servable_lm_common.score_tf_tokenize_inputs(
            prefixes,
            suffixes,
            self._tokenizer,
            self._score_params.max_input_seq_len,
            self._score_params.max_suffix_seq_len,
            self._score_params.include_eos_score,
        )
    )

    preprocessed = py_utils.NestedMap(
        ids=ids,
        labels=labels,
        paddings=paddings,
        weights=weights,
        score_masks=score_masks,
        inputs_indicator=inputs_indicator,
    )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens, preprocessed
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor
  ) -> NestedNpOrTfTensor:
    """Implements `ExportableToSavedModel.tf_post_processing`."""
    return {'scores': compute_outputs}

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[tf.TensorSpec, tf.TensorSpec, Mapping[str, tf.TensorSpec]]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='prefixes'),
        tf.TensorSpec([batch_size], dtype=tf.string, name='suffixes'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs, batch_size
        ),
    )

  @property
  def extra_trackables(self) -> Any:
    """Implements `ExportableToSavedModel.extra_trackables`."""
    return None


class LMDecodeMethod(ServableLMMethod):
  """Base decode method of an LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      method_hparams: DecodeHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      streamable: bool = False,
      load: bool = True,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._method_hparams = method_hparams
    dummy_input_sample = '1'
    if isinstance(method_hparams, DecodeHParams):
      self._include_prefix_in_result = method_hparams.include_prefix_in_result
    logging.info('Using np_tf_sess_wrapper on LMDecodeMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    logging.info(
        'Using np_tf_sess_wrapper on LMDecodeMethod.tf_post_processing'
    )
    self._tf_sess_post_processing = np_tf_sess_wrapper.wrap_tf_session(
        # pylint: disable=g-long-lambda
        lambda *args: decode_tf_post_processing(
            *args,
            tokenizer=self._tokenizer,
            t5_model=self._method_hparams.t5_model,
            include_prefix_in_result=self._include_prefix_in_result,
        ),
        False,
    )
    self._streamable = streamable
    logging.info('Initialize LMDecodeMethod to be streamable=%s.', streamable)

    def _init_stream_and_decode(new_ids):
      batch_size = tf.shape(new_ids)[:-1]
      return self._tokenizer.DecodeOnStream(
          new_ids, self._tokenizer.InitStream(batch_size)
      )

    self._tf_sess_first_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        _init_stream_and_decode, False
    )
    self._tf_sess_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.DecodeOnStream, False
    )
    self._tf_sess_stream_finish = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.FinishStream, False
    )

    super().__init__(
        model,
        'decode',
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
        load=load,
    )

  def call_model_function(self, inputs, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    outputs = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        method=self._model.decode_with_params,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return outputs

  @property
  def streamable(self) -> bool:
    return self._streamable

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    return servable_lm_common.decode_fetch_output(
        model_fn_outputs,
        model_fn_inputs,
        self._method_hparams.t5_model,
        self._method_hparams.fetch_prefix_lengths_from_inputs,
    )

  def pre_processing(self, raw_inputs: List[str]) -> NestedNpTensor:
    texts = np.array(raw_inputs)
    return self._tf_sess_pre_processing(texts)

  def get_maxlen(self) -> int:
    return self._method_hparams.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 2

  def extra_pad_result(
      self, result: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Extra pad result from decoding."""
    seqlen = branch_key

    def _pad_fn(sub_result):
      paddings = [[0, 0], [0, self.get_maxlen() - seqlen]]
      for key in {'paddings', 'weights', 'ids'}:
        if key in sub_result:
          sub_result[key] = jnp.pad(sub_result[key], paddings)
      return sub_result

    return tuple([_pad_fn(sub_result) for sub_result in result])

  def post_processing(
      self, compute_outputs: NestedNpTensor
  ) -> List[Tuple[List[str], List[float]]]:
    # A list of results for the inputs. Each element has multiple samples from
    # the decoding algorithm, which has a list of strings and a list of scores.
    post_processed = self._tf_sess_post_processing(compute_outputs)
    # post_processed = self.tf_post_processing(compute_outputs)
    batched_decoded = post_processed['topk_decoded']
    batched_scores = post_processed['topk_scores']
    return [
        ([d.decode() for d in decoded], list(scores))
        for decoded, scores in zip(batched_decoded, batched_scores)
    ]

  def post_processing_stream(
      self,
      compute_outputs: Optional[NestedNpTensor] = None,
      stream_state: Optional[Any] = None,
  ) -> Tuple[List[Tuple[List[str], List[float]]], Optional[Any]]:
    if compute_outputs is None and stream_state is None:
      raise ValueError('compute_outputs and stream_state cannot both be None')

    if compute_outputs is None:
      batch_decoded = self._tf_sess_stream_finish(stream_state)
      stream_state = None
      scores = np.zeros(batch_decoded.shape)
    elif stream_state is None:
      batch_decoded, stream_state = self._tf_sess_first_stream_step(
          compute_outputs['output_ids']
      )
      scores = compute_outputs['scores']
    else:
      batch_decoded, stream_state = self._tf_sess_stream_step(
          compute_outputs['output_ids'], stream_state
      )
      scores = compute_outputs['scores']

    return [(d, s) for (d, s) in zip(batch_decoded, scores)], stream_state

  def get_scores(self, result: NestedMap, host=False):
    """Get scores from decoding results."""
    if self._method_hparams.t5_model:
      return result.logprobs

    if hasattr(result, 'scores'):
      return result.scores

    np_op = np if host else jnp

    if 'suffix_prompt_lengths' in result and 'suffix_lengths' in result:
      # Get scores for suffix rating ids.
      is_valid_output = np_op.logical_and(
          np_op.arange(result.output_ids.shape[-1])
          >= result.decode_lengths[:, :, None]
          + result.suffix_prompt_lengths[:, :, None]
          - 1,
          np_op.arange(result.output_ids.shape[-1])
          < result.decode_lengths[:, :, None]
          + result.suffix_lengths[:, :, None]
          - 1,
      )
    else:
      is_valid_output = np_op.logical_and(
          np_op.arange(result.output_ids.shape[-1])
          >= result.prefix_lengths[:, None, None],
          np_op.arange(result.output_ids.shape[-1])
          < result.decode_lengths[:, :, None],
      )
    # [batch_size, num_samples, seqlen]
    scores = np_op.where(
        is_valid_output, result.logprobs, np_op.zeros_like(result.logprobs)
    )
    # Scores are computed by excluding the prefix and padding.
    # [batch_size, num_samples]
    return np_op.sum(scores, axis=-1)

  def tf_pre_processing(
      self,
      texts: NestedNpOrTfTensor,
      extra_inputs: Optional[Mapping[str, Any]] = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `texts` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`. If extra
    inputs are provided in the input signature, the exported
    method will take a batched tensor too. See also the `input_signature` method
    of this class.

    Args:
      texts: the input text of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    ids, paddings, prefix_lengths, weights = (
        servable_lm_common.decode_tf_tokenize_inputs(
            texts,
            self._tokenizer,
            self._method_hparams.max_input_seq_len,
            self._method_hparams.t5_model,
        )
    )

    batch_size = tf.shape(ids)[0]
    if self._method_hparams.t5_model:
      target_length = self._method_hparams.decoder.seqlen
      preprocessed = py_utils.NestedMap(
          encoder_input_tokens=ids,
          decoder_input_tokens=tf.ones((batch_size, target_length)),
      )
    elif self._method_hparams.encoder_decoder_model:
      src = py_utils.NestedMap(
          ids=tf.cast(ids, tf.int32),
          paddings=paddings,
      )
      tgt = py_utils.NestedMap(
          ids=tf.zeros((batch_size, 1), dtype=tf.int32),
          paddings=tf.zeros((batch_size, 1)),
      )
      preprocessed = py_utils.NestedMap(
          src=src,
          tgt=tgt,
          prefix_lengths=tf.ones((batch_size), tf.int32),
      )
    else:
      preprocessed = py_utils.NestedMap(
          ids=ids,
          paddings=paddings,
          prefix_lengths=tf.cast(prefix_lengths, tf.int32),
          weights=weights,
      )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens, preprocessed
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor
  ) -> NestedNpOrTfTensor:
    """Post-process the outputs using TF ops.

    This also implements `ExportableToSavedModel.tf_post_processing`.

    Args:
      compute_outputs: the outputs of the model function.

    Returns:
      A mapping that contains the decoded tensors, scores and ids of the topk
      results.
    """
    return decode_tf_post_processing(
        compute_outputs,
        tokenizer=self._tokenizer,
        t5_model=self._method_hparams.t5_model,
        include_prefix_in_result=self._include_prefix_in_result,
    )

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[tf.TensorSpec, Mapping[str, tf.TensorSpec]]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='text'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs, batch_size
        ),
    )

  @property
  def extra_trackables(self) -> Any:
    """Implements `ExportableToSavedModel.extra_trackables`."""
    return None


class TextToEmbedding(servable_model.ServableMethod):
  """Implements text embedding method."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_fn_name: str,
      model_state: servable_model.ServableModelState,
      method_hparams: TextToEmbeddingHParams,
      prng_key: PRNGKey,
      dummy_input_sample: Any,
      model_config: Any,
  ):
    self._model_config = model_config
    self._model_config.init_for_serving()
    self._max_length = method_hparams.max_input_seq_len
    self._embedding_name = method_hparams.output_embedding_name
    super().__init__(
        model,
        model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
    )

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    return py_utils.NestedMap(
        text_embedding=model_fn_outputs[0][self._embedding_name],
    )

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    ids, labels, weights, paddings = self._model_config.tokenize(
        np.array(raw_inputs), self._max_length
    )
    return py_utils.NestedMap(
        ids=np.array(ids),
        labels=np.array(labels),
        weights=np.array(weights),
        paddings=np.array(paddings),
    )

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    return list(compute_outputs['text_embedding'])


class LMGradientMethod(ServableLMMethod):
  """Implements the gradient method of LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      gradient_params: GradientHParams,
      tokenizer_p: Any,
      exportable: bool = False,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._gradient_params = gradient_params
    self._delimiter = '/'
    dummy_input_sample = ('1', '1')
    logging.info(
        'Using np_tf_sess_wrapper on LMGradientMethod.tf_pre_processing'
    )
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    super().__init__(
        model,
        '__call__',
        model_state,
        gradient_params,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
    )

  def call_model_function(
      self, inputs: NestedJTensor, mdl_vars: NestedJTensor, prng_key: PRNGKey
  ) -> NestedJTensor:
    tensors_to_take_gradients = {
        'inputs': {},
        'mdl_vars': {},
    }
    inputs_tensor_names = (
        self._gradient_params.inputs_tensor_names
        if self._gradient_params.inputs_tensor_names is not None
        else {}
    )
    mdl_vars_tensor_names = (
        self._gradient_params.mdl_vars_tensor_names
        if self._gradient_params.mdl_vars_tensor_names is not None
        else {}
    )
    split_inputs_tensor_names = {
        name: name.split(self._delimiter) for name in inputs_tensor_names
    }
    split_mdl_vars_tensor_names = {
        name: name.split(self._delimiter) for name in mdl_vars_tensor_names
    }

    def fetch(tree, keys):
      for key in keys:
        tree = tree[key]
      return tree

    def insert(tree, keys, x):
      for key in keys[:-1]:
        tree = tree[key]
      tree[keys[-1]] = x
      return x

    for k, v in split_inputs_tensor_names.items():
      try:
        tensors_to_take_gradients['inputs'][k] = fetch(inputs, v)
      except Exception as e:
        raise ValueError(f'Failed to find tensor {k} from inputs') from e
    for k, v in split_mdl_vars_tensor_names.items():
      try:
        tensors_to_take_gradients['mdl_vars'][k] = fetch(mdl_vars, v)
      except Exception as e:
        raise ValueError(f'Failed to find tensor {k} from mdl_vars') from e

    call_fn = super().call_model_function

    def forward_fn(tensors_to_take_gradients, inputs_no_grad, mdl_vars_no_grad):
      for k, v in tensors_to_take_gradients['inputs'].items():
        insert(inputs_no_grad, split_inputs_tensor_names[k], v)
      for k, v in tensors_to_take_gradients['mdl_vars'].items():
        insert(mdl_vars_no_grad, split_mdl_vars_tensor_names[k], v)
      outputs = call_fn(inputs_no_grad, mdl_vars_no_grad, prng_key)
      return outputs[0][0]['total_loss'][0], outputs

    compute_gradient_fn = jax.value_and_grad(forward_fn, has_aux=True)
    (_, outputs), grads = compute_gradient_fn(
        tensors_to_take_gradients, inputs, mdl_vars
    )
    outputs = (outputs[0], outputs[1])  # 1 is for mutable.
    outputs[0][0]['gradients'] = grads
    return outputs

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    # fetch loss and gradients from the model output
    metrics, per_example_output = model_fn_outputs[0]
    output = dict(scores=per_example_output['scores'])

    for grads_type, grads_dict in metrics['gradients'].items():
      for tensor_name, grads in grads_dict.items():
        output[f'gradients/{grads_type}/{tensor_name}'] = grads

    return output

  def get_maxlen(self) -> int:
    return self._gradient_params.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 1

  def pre_processing(self, raw_inputs: List[Tuple[str, str]]) -> NestedNpTensor:
    prefixes = np.array([prefix for prefix, _ in raw_inputs])
    suffixes = np.array([suffix for _, suffix in raw_inputs])
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def post_processing(
      self, compute_outputs: NestedNpTensor
  ) -> List[Dict[str, List[float]]]:
    flattened_outputs = jax.tree_util.tree_map(
        lambda x: x.flatten().tolist(), compute_outputs
    )

    return [flattened_outputs]  # The extra list is to just conform to base api.

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
      extra_inputs: Optional[Mapping[str, Any]] = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    (ids, labels, paddings, weights, score_masks, inputs_indicator) = (
        servable_lm_common.score_tf_tokenize_inputs(
            prefixes,
            suffixes,
            self._tokenizer,
            self._gradient_params.max_input_seq_len,
            self._gradient_params.max_suffix_seq_len,
            self._gradient_params.include_eos_score,
        )
    )

    preprocessed = py_utils.NestedMap(
        ids=ids,
        labels=labels,
        paddings=paddings,
        weights=weights,
        score_masks=score_masks,
        inputs_indicator=inputs_indicator,
    )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens, preprocessed
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(self, outputs: NestedTfTensor) -> NestedTfTensor:
    if self._gradient_params.mdl_vars_tensor_names:
      raise ValueError(
          'Exporting graident method with gradients to model '
          'variables is not supported since it is undefined '
          'how to introduce the batch dims for export signatures.'
      )

    return outputs

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[tf.TensorSpec, tf.TensorSpec, Mapping[str, tf.TensorSpec]]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='prefixes'),
        tf.TensorSpec([batch_size], dtype=tf.string, name='suffixes'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs, batch_size
        ),
    )

  @property
  def extra_trackables(self) -> Any:
    """Implements `ExportableToSavedModel.extra_trackables`."""
    return None


class ServableLMModel(servable_model.ServableModel):
  """Represents an implementation for the LM service, backed by a model.

  This class is responsible for model loading, batch padding, etc.
  """

  def init_method(
      self,
      method: str,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      method_params: servable_model_params.ServableMethodParams,
      prng_key: PRNGKey,
  ) -> servable_model.ServableMethod:
    assert isinstance(self.model_config, ServableLMModelParams)
    tokenizer_p = self.model_config.serving_tokenizer()
    if method == LMMethodName.SCORE:
      assert isinstance(method_params, ScoreHParams)
      return LMScoreMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
      )
    elif method == LMMethodName.GENERATE:
      assert isinstance(method_params, DecodeHParams)
      return LMDecodeMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
      )
    elif method == LMMethodName.GENERATE_STREAM:
      assert isinstance(method_params, DecodeHParams)
      return LMDecodeMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=False,
          streamable=True,
      )
    elif method == LMMethodName.EMBED:
      assert isinstance(method_params, TextToEmbeddingHParams)
      assert method_params.output_embedding_name is not None
      if method_params.model_method_name is None:
        raise ValueError(
            'Must specify `model_method_name` in TextToEmbeddingHParams.'
        )
      return TextToEmbedding(
          model,
          method_params.model_method_name,
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample='test',
          model_config=self.model_config,
      )
    elif method == LMMethodName.GRADIENT:
      assert isinstance(method_params, GradientHParams)
      assert (
          method_params.inputs_tensor_names is not None
          or method_params.mdl_vars_tensor_names is not None
      )
      return LMGradientMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
      )
    else:
      raise NotImplementedError(f'method {method} not implemented')

  def supports_dummy_compute_on_primary(self) -> bool:
    if self.methods is None or not isinstance(self.methods, Dict):
      return True
    for method in list(self.methods.values()):
      has_multiple_seq_lens = (
          hasattr(method, 'sorted_seq_lens')
          and method.sorted_seq_lens is not None
          and len(method.sorted_seq_lens) > 1
      )
      if has_multiple_seq_lens:
        return False
    return True