# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The HuggingFace Inc. team.
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

import collections
import json
import math
import os
import random
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "TensorFlow", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering
import tensorflow as tf

def load_from_tf(config, tf_path):
    model = BertForQuestionAnswering(config)
    model.classifier = model.qa_outputs

    # This part is copied from HuggingFace Transformers with a fix to bypass an error
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier") # This line is causing the issue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    model.qa_outputs = model.classifier
    del model.classifier
    return model

def save_to_onnx(model):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model.eval()

    dummy_input = torch.ones((1, 384), dtype=torch.int64)
    torch.onnx.export(
        model,
        (dummy_input, dummy_input, dummy_input),
        "build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx",
        verbose=True,
        input_names = ["input_ids", "input_mask", "segment_ids"],
        output_names = ["output_start_logits", "output_end_logits"],
        opset_version=11,
        dynamic_axes=({"input_ids": {0: "batch_size"}, "input_mask": {0: "batch_size"}, "segment_ids": {0: "batch_size"},
            "output_start_logits": {0: "batch_size"}, "output_end_logits": {0: "batch_size"}})
    )

def main():
    with open("build/data/bert_tf_v1_1_large_fp32_384_v2/bert_config.json") as f:
        config_json = json.load(f)

    config = BertConfig(
        attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
        hidden_act=config_json["hidden_act"],
        hidden_dropout_prob=config_json["hidden_dropout_prob"],
        hidden_size=config_json["hidden_size"],
        initializer_range=config_json["initializer_range"],
        intermediate_size=config_json["intermediate_size"],
        max_position_embeddings=config_json["max_position_embeddings"],
        num_attention_heads=config_json["num_attention_heads"],
        num_hidden_layers=config_json["num_hidden_layers"],
        type_vocab_size=config_json["type_vocab_size"],
        vocab_size=config_json["vocab_size"])

    model = load_from_tf(config, "build/data/bert_tf_v1_1_large_fp32_384_v2/model.ckpt-5474")
    torch.save(model.state_dict(), "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
    save_to_onnx(model)

if __name__ == "__main__":
    main()
