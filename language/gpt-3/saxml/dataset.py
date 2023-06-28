from typing import Sequence, List, Optional, Iterable, Mapping, Union, Tuple

import numpy as np
import seqio
import tensorflow as tf


class Dataset():

  def __init__(
    self,
    dataset_path: Union[str, List[str]],
    total_count_override: int = None,
    perf_count_override: int = None,
    ):

    self._dataset_path = dataset_path
    self._dataset = tf.data.TFRecordDataset(self._dataset_path)

    self.inputs, self.inputs_str, self.inputs_pretokenized, self.targets, self.targets_pretokenized, self.count = self.process_dataset()

    if total_count_override:
      self.count = min(total_count_override, self.count)
    self.perf_count = self.count
    if perf_count_override:
      self.perf_count = min(perf_count_override, self.count)

  def process_dataset(self):

    count = 0
    inputs = []
    inputs_str = []
    inputs_pretokenized = []
    targets = []
    targets_pretokenized = []

    for record in self._dataset:

      example_proto = tf.train.Example()
      example_proto.ParseFromString(record.numpy())
      for key, feature in example_proto.features.feature.items():
        if key == "inputs":
          feature_value = feature.int64_list.value
          inputs.append(feature_value)
          feature_value_str = ','.join([str(i) for i in list(feature_value)])
          inputs_str.append(feature_value_str)
        if key == "targets":
          feature_value = feature.int64_list.value
          targets.append(feature_value)
        if key == "inputs_pretokenized":
          feature_value = feature.bytes_list.value[0].decode("utf-8")
          inputs_pretokenized.append(feature_value)
        if key == "targets_pretokenized":
          feature_value = feature.bytes_list.value[0].decode("utf-8")
          targets_pretokenized.append(feature_value)
      count += 1

    return inputs, inputs_str, inputs_pretokenized, targets, targets_pretokenized, count

  def LoadSamplesToRam(self, sample_list):
      pass

  def UnloadSamplesFromRam(self, sample_list):
      pass

  def __del__(self):
      print("Finished destroying QSL.")
