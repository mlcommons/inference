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
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import torch
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL

class BERT_PyTorch_SUT():
    def __init__(self):
        print("Loading BERT configs...")
        with open("bert_config.json") as f:
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

        print("Loading PyTorch model...")
        self.model = BertForQuestionAnswering(config)
        self.model.eval()
        self.model.cuda()
        self.model.load_state_dict(torch.load("build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"))

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL()

    def issue_queries(self, query_samples):
        with torch.no_grad():
            for i in range(len(query_samples)):
                eval_features = self.qsl.get_features(query_samples[i].index)
                start_scores, end_scores = self.model.forward(input_ids=torch.LongTensor(eval_features.input_ids).unsqueeze(0).cuda(),
                    attention_mask=torch.LongTensor(eval_features.input_mask).unsqueeze(0).cuda(),
                    token_type_ids=torch.LongTensor(eval_features.segment_ids).unsqueeze(0).cuda())
                output = torch.stack([start_scores, end_scores], axis=-1).squeeze(0).cpu().numpy()

                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_pytorch_sut():
    return BERT_PyTorch_SUT()
