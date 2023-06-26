from absl import app
from absl import flags
from absl import logging

import time
from typing import List, Optional

import functools

import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf
import numpy as np


_DATADIR = flags.DEFINE_string(
    'data_dir',
    'gs://cnn_dailymail_public/mlperf',
    '',
)
_TOKENIZED_DATADIR = flags.DEFINE_string(
    'tokenized_data_dir',
    'gs://cnn_dailymail_public/mlperf/tokenized_cnn_dailymail_3.0.0',
    '',
)
_SPM_PATH = flags.DEFINE_string(
    'spm_path',
    'gs://cnn_dailymail_public/mlperf/vocab/c4_en_301_5Mexp2_spm.model',
    '',
)
_SPLIT_SET = flags.DEFINE_string(
    'split_set',
    'validation',
    '',
)
_MAX_LENGTH = flags.DEFINE_integer(
    'max_length',
    2048,
    '',
)
_TARGET_LENGTH = flags.DEFINE_integer(
    'target_length',
    256,
    '',
)
_ADD_EOS = flags.DEFINE_boolean(
    'add_eos',
    False,
    '',
)


def main(argv):

    spm_vocabulary = seqio.SentencePieceVocabulary(_SPM_PATH.value)
    output_features = {
        'inputs': t5.data.Feature(vocabulary=spm_vocabulary, add_eos=_ADD_EOS.value),
        'targets': t5.data.Feature(vocabulary=spm_vocabulary, add_eos=_ADD_EOS.value)
    }

    split_set = _SPLIT_SET.value
    data_dir = _DATADIR.value
    tokenized_data_dir = _TOKENIZED_DATADIR.value

    saved_task_dataset_filename = f'{tokenized_data_dir}/cnn_dailymail-{split_set}.tfrecord-00000-of-00001'

    max_length = _MAX_LENGTH.value
    target_length = _TARGET_LENGTH.value

    task_feature_lengths = {
        'inputs': max_length,
        'targets': target_length
    }

    logging.info(f'spm_vocabulary: {spm_vocabulary}')
    logging.info(f'split_set: {split_set}')
    logging.info(f'data_dir: {data_dir}')
    logging.info(f'tokenized_data_dir: {tokenized_data_dir}')
    logging.info(f'saved_task_dataset_filename: {saved_task_dataset_filename}')
    logging.info(f'_MAX_LENGTH: {max_length}')
    logging.info(f'_TARGET_LENGTH: {target_length}')

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_list_feature(value):
        """Returns an int64_list from a bool / eum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    class TaskRegistry(t5.data.TaskRegistry):
        """Task registry with extra tracking."""

        TASK_NAMES = []

        @classmethod
        def add_versioned_tfds_task(cls,
                                    name: str,
                                    *,
                                    versions: List[str],
                                    pinned_version: Optional[str] = None,
                                    tfds_name: str,
                                    tfds_data_dir: Optional[str] = None,
                                    **kwargs) -> List[seqio.Task]:
            tasks = []
            for version in versions:
                tasks.append(
                    cls.add(
                        f'{name}_{version}',
                        seqio.Task,
                        source=seqio.TfdsDataSource(
                            tfds_name=f'{tfds_name}:{version}',
                            tfds_data_dir=tfds_data_dir,
                        ),
                        **kwargs,
                    ))
                if pinned_version is not None:
                    tasks.append(
                        cls.add(
                            name,
                            seqio.Task,
                            source=seqio.TfdsDataSource(
                                tfds_name=f'{tfds_name}:{pinned_version}',
                                tfds_data_dir=tfds_data_dir,
                            ),
                            **kwargs,
                        ))
            return tasks

    for name in list(TaskRegistry.names()):
        TaskRegistry.remove(name)

    TaskRegistry.add_versioned_tfds_task(
        'cnn_dailymail',
        versions=['3.0.0'],
        pinned_version='3.0.0',
        tfds_name='cnn_dailymail',
        tfds_data_dir=data_dir,
        preprocessors=[
            functools.partial(
                t5_preprocessors.summarize,
                article_key='article',
                summary_key='highlights',
            ),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=output_features,
        metric_fns=[],
    )
    logging.info(f"seqio.TaskRegistry.names(): {seqio.TaskRegistry.names()}")

    task = seqio.TaskRegistry.get("cnn_dailymail_3.0.0")

    task_dataset = task.get_dataset(sequence_length=task_feature_lengths, split=split_set, shuffle=False)

    def serialize_example(inputs_pretokenized,
                        inputs,
                        targets_pretokenized,
                        targets):
        feature = {
            'inputs_pretokenized': _bytes_feature(inputs_pretokenized),
            'inputs': _int64_list_feature(inputs),
            'targets_pretokenized': _bytes_feature(targets_pretokenized),
            'targets': _int64_list_feature(targets),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize_example(inputs_pretokenized, inputs, targets_pretokenized, targets):
        tf_string = tf.py_function(
            serialize_example,
            (inputs_pretokenized, inputs, targets_pretokenized, targets),
            tf.string
        )
        return tf.reshape(tf_string, ())

    def generator():
        for features in task_dataset:
            yield serialize_example(
                features['inputs_pretokenized'],
                features['inputs'],
                features['targets_pretokenized'],
                features['targets']
        )

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())
    serialized_features_dataset

    writer = tf.data.experimental.TFRecordWriter(saved_task_dataset_filename)
    writer.write(serialized_features_dataset)


if __name__ == '__main__':
  app.run(main)
