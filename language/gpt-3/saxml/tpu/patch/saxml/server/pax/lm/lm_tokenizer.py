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
"""Tokenizer for language models."""
from __future__ import annotations

import dataclasses
from typing import Any, List, Tuple

from praxis import base_hyperparams
import seqio
import tensorflow as tf

StreamState = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


class LMTokenizer(base_hyperparams.FiddleBaseParameterizable):
  """Tokenizer for language models.

  Attributes:
    append_eos: Whether to append </s> at the end and treat it as a non-padded
      label, always set to True.
    spm_model: File name for a sentencepiece model.
    target_sos_id: Start of sentence id.
    target_eos_id: End of sentence id.
    slice_left: Slice the left part of the sequence if it is too long.
      Otherwise, slice the right part of the sequence.
    streaming_whitespace_preserving_prefix: A prefix added to each non-SOS
      streaming decoding step to prevent the leading whitespace from being
      removed by sentencepiece; after decoding the step, it will be removed from
      the string result. It must be a regular token in the vocabulary.
  """

  append_eos: bool = True
  spm_model: str = None
  target_sos_id: int = 0
  target_eos_id: int = 1
  slice_left: bool = True
  streaming_whitespace_preserving_prefix: str = 'a'
  tokenized: bool = False

  _vocab: seqio.SentencePieceVocabulary = dataclasses.field(
      init=False, repr=False
  )

  def __post_init__(self):
    assert self.append_eos
    self._vocab = seqio.SentencePieceVocabulary(self.hparams.spm_model, 0)

  def StringsToIds(
      self, strs: tf.Tensor, max_length: int
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Tokenizes strs into vocab ids.

    Args:
      strs: A 1D tensor of strings.
      max_length: An int providing the max_length for strs.
      unused_args: Some not used arguments from base class.

    Returns:
      A tuple (ids, labels, paddings) with the same shape [batch, maxlen].

      - ids[i, j] is the input token id of i-th sample for j-th step.
      - labels[i, j] is the target token id of i-th sample for j-th step.
      - paddings[i, j] is 1 iff i-th sample's j-th step is padded.
    """
    p = self.hparams
    assert p.spm_model
    assert max_length is not None
    batch = tf.shape(strs)[0]
    # labels is a ragged Tensor.
    if p.tokenized:
      labels_in_str = tf.strings.split(strs, sep=',', maxsplit=-1)
      labels = tf.strings.to_number(labels_in_str, out_type=tf.int32)
    else:
      labels = self._vocab.tf_tokenizer.tokenize(strs)
    if p.slice_left:
      labels = labels[:, : max_length - 1]
    else:
      labels = labels[:, -(max_length - 1) :]

    sos_ids = tf.fill([batch, 1], tf.constant(p.target_sos_id, dtype=tf.int32))
    ids = tf.concat([sos_ids, labels], axis=1)
    eos_ids = tf.fill([batch, 1], tf.constant(p.target_eos_id, dtype=tf.int32))
    labels = tf.concat([labels, eos_ids], axis=1)
    # Convert raggedtensor to padded tensor.
    ids = ids.to_tensor()
    labels = labels.to_tensor()

    def _pad(x: tf.Tensor, shape: List[int]) -> tf.Tensor:
      """Helper function to pad tensor to the desired shape."""
      pad = shape - tf.minimum(tf.shape(x), shape)
      zeros = tf.zeros_like(pad)
      # If dim_i is less than shape[i], pads after contents.
      paddings = tf.stack([zeros, pad], axis=1)
      # If dim_i is larger than shape[i], we slice [0:shape[i]] for dim_i.
      slice_begin = zeros
      x = tf.pad(x, paddings)
      x = tf.slice(x, slice_begin, shape)

      return tf.reshape(x, shape)

    # Pad ids and labels to the desired shape.
    shape = [batch, max_length]
    ids = _pad(ids, shape)
    labels = _pad(labels, shape)

    # Calculate paddings for each example based on eos_id locations.
    eos_indices = tf.argmax(
        tf.equal(labels, p.target_eos_id), axis=1, output_type=tf.int32
    )
    eos_indices = tf.stack([eos_indices] * max_length, axis=1)
    indices = tf.repeat([tf.range(max_length)], batch, axis=0)
    paddings = tf.where(
        indices <= eos_indices,
        tf.zeros_like(indices, tf.float32),
        tf.ones_like(indices, tf.float32),
    )

    return ids, labels, paddings

  def IdsToStrings(self, ids: tf.Tensor, *unused_args: Any) -> tf.Tensor:
    """Converts ids back to strings.

    Decoding stops at padding or eos token.

    Args:
      ids: A matrix of shape [batch, seqlen]. ids[i, :] is the i-th sample's
        ids.
      unused_args: Some not used arguments for API use.

    Returns:
      sequences - A vector of shape [batch]. The converted string sequence.
    """
    p = self.hparams

    if p.tokenized:
      ids_as_string = tf.strings.as_string(ids)
      reduced_ids_as_string = tf.strings.reduce_join(
          ids_as_string, separator=',', axis=-1
      )
      return reduced_ids_as_string
    assert p.spm_model
    return self._vocab.tf_tokenizer.detokenize(ids)

  def InitStream(self, batch_size: tf.Tensor) -> StreamState:
    """Create the initial state for streaming.

    Args:
      batch_size: A scalar or 1D tensor representing stream formation, usually
        batch or [batch, num_samples].

    Returns:

      A tuple of 3 elements:
        - A tensor of shape [np.prod(batch_size), seqlen], containing
          unprocessed prefix IDs. Left-aligned if they have differen lengths.
        - A tensor of shape [np.prod(batch_size)], indicating the valid length
          in the unprocessed prefix IDs.
        - A boolean tensor of shape batch_size indicating if any prefix strings
          have been generated.
    """
    nrows = tf.math.reduce_prod(batch_size)
    started = tf.fill(batch_size, False)
    return (
        tf.zeros([nrows, 0], dtype=tf.int32),
        tf.zeros([nrows], dtype=tf.int32),
        started,
    )

  def DecodeOnStream(
      self, new_ids: tf.Tensor, stream_state: StreamState
  ) -> Tuple[tf.Tensor, StreamState]:
    """Converts new chunks of IDs on decoding streams.

    Args:
      new_ids: A matrix of shape [batch, ..., new_chunk_len] containing IDs
        newly generated from streaming.
      stream_state: Stream state. See description in InitStream.

    Returns:
      A tuple of (newly decoded strings, updated stream state).
    """
    p = self.hparams
    assert p.spm_model

    # Merge all leading N - 1 dimensions of new_ids and started to match the
    # flattened ragged tensor.
    unprocessed_prefix_ids, unprocessed_prefix_len, started = stream_state
    batch_size = tf.shape(started)
    nrows = tf.math.reduce_prod(batch_size)
    new_ids = tf.reshape(new_ids, [nrows, -1])
    started = tf.reshape(started, [nrows])

    # Hint the rank of prefix ids/len for `tf.RaggedTensor.from_tensor`, which
    # is useful when TF can't identify their ranks on its own.
    unprocessed_prefix_ids = tf.ensure_shape(
        unprocessed_prefix_ids, [None, None]
    )
    unprocessed_prefix_len = tf.ensure_shape(unprocessed_prefix_len, [None])

    # Find the byte-encoded IDs.
    new_ids_shape = tf.shape(new_ids)
    b, new_seqlen = new_ids_shape[0], new_ids_shape[1]
    new_pieces = self._vocab.tf_tokenizer.id_to_string(new_ids)
    is_byte = tf.strings.regex_full_match(new_pieces, '<0x[0-9,A-F][0-9,A-F]>$')
    # Remove trailing bytes.
    trailing_byte_count = tf.reduce_sum(
        tf.cast(
            tf.equal(
                tf.cumsum(1 - tf.cast(is_byte, tf.int32), axis=1, reverse=True),
                0,
            ),
            tf.int32,
        ),
        axis=1,
    )
    without_trailing_bytes = tf.RaggedTensor.from_tensor(
        new_ids, new_seqlen - trailing_byte_count
    )
    is_all_bytes = tf.equal(trailing_byte_count, new_seqlen)

    # Add a fake prefix to preserve leading whitespace if earlier prefix was
    # generated.
    fake_prefix_str = p.streaming_whitespace_preserving_prefix
    fake_prefix = self._vocab.tf_tokenizer.tokenize(
        [fake_prefix_str]
    ).to_tensor()
    fake_prefix = tf.repeat(fake_prefix, b, axis=0)
    fake_prefix_len = tf.fill([b], tf.shape(fake_prefix)[1])
    fake_prefix_str_len = tf.fill(
        [b], tf.constant(len(fake_prefix_str), dtype=tf.int32)
    )
    fake_prefix_len = tf.where(started, fake_prefix_len, 0)
    fake_prefix_str_len = tf.where(started, fake_prefix_str_len, 0)
    fake_prefix = tf.RaggedTensor.from_tensor(fake_prefix, fake_prefix_len)

    # Decode with prefix.
    unprocessed_prefix_ids_ragged = tf.RaggedTensor.from_tensor(
        unprocessed_prefix_ids, unprocessed_prefix_len
    )
    to_process = tf.concat(
        [fake_prefix, unprocessed_prefix_ids_ragged, without_trailing_bytes],
        axis=1,
    )
    new_strs = self._vocab.tf_tokenizer.detokenize(to_process)
    # Remove fake prefix.
    new_strs = tf.strings.substr(
        new_strs, fake_prefix_str_len, tf.fill([b], -1)
    )
    new_strs = tf.where(is_all_bytes, tf.constant(''), new_strs)

    new_started = tf.logical_or(started, tf.strings.length(new_strs) > 0)

    trailing_bytes = tf.RaggedTensor.from_tensor(
        tf.reverse(new_ids, axis=[1]), trailing_byte_count
    )
    trailing_bytes = tf.reverse(trailing_bytes, axis=[1])

    remaining_prefix_len = tf.where(is_all_bytes, unprocessed_prefix_len, 0)
    remaining_prefix = tf.RaggedTensor.from_tensor(
        unprocessed_prefix_ids, remaining_prefix_len
    )
    new_unprocessed_prefix_ids = tf.concat(
        [remaining_prefix, trailing_bytes], axis=1
    )

    # Reshape output back to [batch_size, ...] before returning.
    new_strs = tf.reshape(new_strs, batch_size)
    new_started = tf.reshape(new_started, batch_size)
    return new_strs, (
        new_unprocessed_prefix_ids.to_tensor(),
        tf.cast(new_unprocessed_prefix_ids.row_lengths(), dtype=tf.int32),
        new_started,
    )

  def FinishStream(self, stream_state: StreamState) -> tf.Tensor:
    """Finishes the streams by decoding any remaining tokens."""
    p = self.hparams
    assert p.spm_model
    _, _, started = stream_state
    batch_size = tf.shape(started)

    eos_ids = tf.fill(batch_size, tf.constant(p.target_eos_id, dtype=tf.int32))
    eos_ids = tf.expand_dims(eos_ids, axis=-1)
    new_strs, _ = self.DecodeOnStream(eos_ids, stream_state)
    return new_strs