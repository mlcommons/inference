# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

import array
import json
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "TensorFlow", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import modeling
import numpy as np
import tensorflow as tf
from squad_QSL import get_squad_QSL

# Allow TF to increase GPU memory usage dynamically to prevent cuBLAS init problems.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class BERT_TF_ESTIMATOR_SUT():
    def __init__(self, batch_size=8):
        print("Loading TF model...")
        bert_config = modeling.BertConfig.from_json_file("bert_config.json")

        model_fn = self.model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.ckpt-5474"))

        self.estimator = tf.estimator.Estimator(model_fn=model_fn)
        self.batch_size = batch_size

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL()

    def issue_queries(self, query_samples):
        input_ids = np.zeros((len(query_samples), 1, 384), dtype=np.int32)
        input_mask = np.zeros((len(query_samples), 1, 384), dtype=np.int32)
        segment_ids = np.zeros((len(query_samples), 1, 384), dtype=np.int32)
        for sample_idx in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[sample_idx].index)
            input_ids[sample_idx, ...] = np.array(eval_features.input_ids)
            input_mask[sample_idx, ...] = np.array(eval_features.input_mask)
            segment_ids[sample_idx, ...] = np.array(eval_features.segment_ids)

        def input_fn():
            inputs = {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids
            }
            return tf.data.Dataset.from_tensor_slices(inputs)

        for i, result in enumerate(self.estimator.predict(input_fn)):
            logits = [float(x) for x in result["logits"].flat]
            response_array = array.array("B", np.array(logits).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

    def create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        """Creates a classification model."""
        model = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
                compute_type=tf.float32)

        final_hidden = model.get_sequence_output()

        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        output_weights = tf.get_variable(
                "cls/squad/output_weights", [2, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
                "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [batch_size, seq_length, 2])
        return logits

        # logits = tf.transpose(logits, [2, 0, 1])

        # unstacked_logits = tf.unstack(logits, axis=0, name='unstack')

        # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        # return (start_logits, end_logits)

    def model_fn_builder(self, bert_config, init_checkpoint, use_one_hot_embeddings=False):
        """Returns `model_fn` closure for Estimator."""

        def model_fn(features, labels):  # pylint: disable=unused-argument
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]

            logits = self.create_model(
                bert_config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

            tvars = tf.compat.v1.trainable_variables()

            initialized_variable_names = {}
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

            predictions = {
                "logits": logits
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)

            return output_spec

        return model_fn


def get_tf_estimator_sut():
    return BERT_TF_ESTIMATOR_SUT()
