from typing import Sequence, List, Optional, Iterable, Mapping, Union, Tuple

import numpy as np
import seqio
import tensorflow as tf

class Dataset():

  def __init__(
    self,
    dataset_path: Union[str, List[str]],
    spm_path: str,
    add_eos: bool = False,
    total_count_override: int = None,
    perf_count_override: int = None,
    ):

    self._spm_path = spm_path
    self._vocabulary = seqio.SentencePieceVocabulary(self._spm_path)

    self._dataset = tf.data.TFRecordDataset(self.dataset_path)

    self.inputs, self.tokenized_inputs, self.count = self.process_dataset()

    if total_count_override:
      self.count = min(total_count_override, self.count)
    self.perf_count = self.count
    if perf_count_override:
      self.perf_count = min(perf_count_override, self.count)

  @properties
  def vocabulary(self):
    return self._vocabulary

  def process_dataset(self):

    count = 0
    inputs = []
    tokenized_inputs = []
    for record in self._dataset:

      example_proto = tf.train.Example()
      example_proto.ParseFromString(record)

      for key, feature in example_proto.features.feature.items():
        if key == "article":
          kind = feature.WhichOneof("kind")
          feature_value = getattr(feature, kind).value[0]
          article = feature_value.decode("utf-8")]
          prompt = f"summarize: {article}"
          tokenized_prompt = self.vocabulary.tf_tokenizer.tokenize(prompt)
          inputs.append(prompt)
          tokenized_inputs.append(tokenized_prompt)

      count += 1

    return inputs, tokenized_inputs, count

  def LoadSamplesToRam(self, sample_list):
      pass

  def UnloadSamplesFromRam(self, sample_list):
      pass

  def __del__(self):
      print("Finished destroying QSL.")
