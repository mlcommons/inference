# Lint as: python3
# Copyright 2020 Google. All Rights Reserved.
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
"""Data loader for pre-processed Criteo data."""

import tensorflow.compat.v1 as tf
import feature_config as fc


def rand_features(batch_size):
  """Emits random input features, used for testing."""
  features = {}
  pos_size = batch_size // 2
  neg_size = batch_size - pos_size
  features[fc.LABEL_FEATURE] = tf.concat([
      tf.ones([pos_size, 1], dtype=tf.float32),
      tf.zeros([neg_size, 1], dtype=tf.float32)
  ],
                                         axis=0)
  features["int-features"] = tf.random.uniform(
      shape=(batch_size, len(fc.INT_FEATURES)),
      maxval=fc.FAKE_DATA_INT_MAX)
  features["cat-features"] = tf.random.uniform(
      shape=(batch_size, len(fc.CATEGORICAL_FEATURES)),
      maxval=fc.FAKE_DATA_VOCAB_SIZE,
      dtype=tf.int32)
  return features


class CriteoTFRecordReader(object):
  """Input reader fn for TFRecords that have been serialized in batched form."""

  def __init__(self,
               file_path=None,
               feature_config=None,
               is_training=True,
               use_cached_data=False,
               use_synthetic_data=False,
               params=None):
    self._file_path = file_path
    self._feature_config = feature_config
    self._is_training = is_training
    self._use_cached_data = use_cached_data
    self._use_synthetic_data = use_synthetic_data
    self._params = params

  def __call__(self, params):

    batch_size = params["batch_size"]
    if self._use_synthetic_data:
      ds = tf.data.Dataset.from_tensor_slices(rand_features(batch_size))
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.take(1).cache().repeat()
      return ds

    def _get_feature_spec():
      feature_spec = {}
      feature_spec[fc.LABEL_FEATURE] = tf.FixedLenFeature([
          batch_size,
      ],
                                                          dtype=tf.float32)
      for int_ft in fc.INT_FEATURES:
        feature_spec[int_ft] = tf.FixedLenFeature([
            batch_size,
        ],
                                                  dtype=tf.float32)
      for cat_ft in fc.CATEGORICAL_FEATURES:
        feature_spec[cat_ft] = tf.FixedLenFeature([], dtype=tf.string)
      return feature_spec

    def _parse_fn(serialized_example):
      feature_spec = _get_feature_spec()
      p_features = tf.parse_single_example(serialized_example, feature_spec)

      features = {}
      features[fc.LABEL_FEATURE] = tf.reshape(p_features[fc.LABEL_FEATURE],
                                              (batch_size, 1))

      int_features = []
      for int_ft in fc.INT_FEATURES:
        cur_feature = tf.reshape(p_features[int_ft], (batch_size, 1))
        int_features.append(cur_feature)
      features["int-features"] = tf.concat(int_features, axis=-1)
      cat_features = []
      tc_features = []

      tbl_idxs_sorted = self._feature_config.get_table_idx_orderd_by_size()
      for idx in range(len(fc.CATEGORICAL_FEATURES)):
        # Add features from largest-vocab to smallest-vocab.
        raw_tbl_idx = tbl_idxs_sorted[idx]
        cat_feature_idx = raw_tbl_idx + 14
        cat_feature = "categorical-feature-%d" % cat_feature_idx

        # Decode from bytes to int32.
        cat_ft_int32 = tf.io.decode_raw(p_features[cat_feature], tf.int32)
        cat_ft_int32 = tf.reshape(cat_ft_int32, (batch_size, 1))
        if idx < self._feature_config.get_num_tables_in_ec():
          cat_features.append(cat_ft_int32)
        else:
          tc_features.append(cat_ft_int32)
      features["cat-features"] = tf.concat(cat_features, axis=-1)
      if tc_features:
        features["tc-features"] = tf.concat(tc_features, axis=-1)

      return features

    ds = tf.data.Dataset.list_files(self._file_path, shuffle=False)
    ds = ds.shard(params["dataset_num_shards"],
                  params["dataset_index"])

    if self._is_training:
      ds = ds.shuffle(
          tf.to_int64(
              max(256, params["dataset_num_shards"]) /
              params["dataset_num_shards"]))
      ds = ds.repeat()

    ds = tf.data.TFRecordDataset(
        ds, buffer_size=64 * 1024 * 1024, num_parallel_reads=8)
    ds = ds.map(_parse_fn, num_parallel_calls=8)

    if not self._is_training:
      num_dataset_samples = self._params["eval_steps"] * (
          self._params["eval_batch_size"] // params["dataset_num_shards"])
      num_dataset_batches = num_dataset_samples // batch_size
      def _mark_as_padding(features):
        """Padding will be denoted with a label value of -1."""
        features[fc.LABEL_FEATURE] = -1 * tf.ones(
            (batch_size, 1), dtype=tf.float32)
        return features
      # 100 steps worth of padding.
      padding_ds = ds.take(self._params["replicas_per_host"])
      padding_ds = padding_ds.map(_mark_as_padding).repeat(100)
      ds = ds.concatenate(padding_ds).take(num_dataset_batches)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    if self._use_cached_data:
      ds = ds.take(100).cache().repeat()
    return ds


class CriteoTsvReader(object):
  """Input reader fn for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.
  """

  def __init__(self,
               file_path=None,
               feature_config=None,
               is_training=True,
               distributed_eval=False,
               parallelism=1,
               use_cached_data=False,
               use_synthetic_data=False):
    self._file_path = file_path
    self._feature_config = feature_config
    self._is_training = is_training
    self._distributed_eval = distributed_eval
    self._parallelism = parallelism
    self._use_cached_data = use_cached_data
    self._use_synthetic_data = use_synthetic_data

  def __call__(self, params):
    batch_size = params["batch_size"]
    if self._use_synthetic_data:
      ds = tf.data.Dataset.from_tensor_slices(rand_features(batch_size))
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.take(1).cache().repeat()
      return ds

    @tf.function
    def _parse_example_fn(example):
      """Parser function for pre-processed Criteo TSV records."""
      label_defaults = [[0.0]]
      int_defaults = [
          [0.0] for _ in range(self._feature_config.get_num_dense_features())
      ]
      categorical_defaults = [
          [0] for _ in range(self._feature_config.get_num_sparse_features())
      ]
      record_defaults = label_defaults + int_defaults + categorical_defaults
      fields = tf.decode_csv(
          example, record_defaults, field_delim="\t", na_value="-1")

      num_labels = 1
      num_dense = len(int_defaults)
      features = {}
      features[fc.LABEL_FEATURE] = tf.reshape(fields[0], [batch_size, 1])

      int_features = []
      for idx in range(num_dense):
        int_features.append(fields[idx + num_labels])
      features["int-features"] = tf.stack(int_features, axis=1)

      cat_features = []
      tc_features = []
      # Features for tables in EmbeddingCore is in cat_features; features for
      # tables in  is in tc_features. The order of the input data
      # follows the order of FLAG.vocab_sizes_embed, so we reorder the input
      # data with resepct to the table sizes.
      for idx, idx_by_size in enumerate(
          self._feature_config.get_table_idx_orderd_by_size()):
        if idx < self._feature_config.get_num_tables_in_ec():
          cat_features.append(
              tf.cast(
                  fields[idx_by_size + num_dense + num_labels], dtype=tf.int32))
        else:
          tc_features.append(
              tf.cast(
                  fields[idx_by_size + num_dense + num_labels], dtype=tf.int32))
      features["cat-features"] = tf.stack(cat_features, axis=1)
      if tc_features:
        features["tc-features"] = tf.stack(tc_features, axis=1)

      return features

    filenames = tf.data.Dataset.list_files(self._file_path, shuffle=False)
    filenames = filenames.shard(params["dataset_num_shards"],
                                params["dataset_index"])

    def make_dataset(ds_index):
      ds = filenames.shard(self._parallelism, ds_index)
      ds = ds.repeat(2)
      ds = ds.interleave(
          tf.data.TextLineDataset,
          cycle_length=16,
          block_length=batch_size // 8,
          num_parallel_calls=8,
          deterministic=False)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.map(_parse_example_fn, num_parallel_calls=16)
      return ds

    ds_indices = tf.data.Dataset.range(self._parallelism)
    ds = ds_indices.interleave(
        make_dataset,
        cycle_length=self._parallelism,
        block_length=1,
        num_parallel_calls=self._parallelism,
        deterministic=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    if self._use_cached_data:
      ds = ds.take(100).cache().repeat()

    return ds
