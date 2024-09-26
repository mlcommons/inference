# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
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
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import torch
import torch_tensorrt
import transformers
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL

import ray
from ray.util.actor_pool import ActorPool


# Adjustable Parameters
BATCH_SIZE = 16    # Note. num_samples (called "test_query_count" in CM) must be a multiple of batch_size

@ray.remote(num_cpus=1,num_gpus=1)
class TorchPredictor:
    def __init__(self, config_json, model_file, batch_size):
        print("init", os.getpid(), torch.cuda.device_count())
        self.pid = os.getpid()
        self.dev_cnt = torch.cuda.device_count()

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

        self.dev = torch.device("cuda")
        self.model = BertForQuestionAnswering(config)
        self.model.to(self.dev)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_file), strict=False)
        # tensor rt
        batch_input_ids = torch.LongTensor(np.zeros((batch_size, 384))).to(self.dev)
        traced_mlm_model = torch.jit.trace(self.model, [batch_input_ids, batch_input_ids, batch_input_ids], strict=False)
        self.trt_model = torch_tensorrt.compile(traced_mlm_model,
            inputs=[
                torch_tensorrt.Input(shape=[batch_size, 384], dtype=torch.int32),
                torch_tensorrt.Input(shape=[batch_size, 384], dtype=torch.int32),
                torch_tensorrt.Input(shape=[batch_size, 384], dtype=torch.int32),
        ],
        enabled_precisions= {torch.float32, torch.float16},
        workspace_size=2000000000,
        truncate_long_and_double=True)

        print("done loading")

    # Logic for inference on 1 batch of data.
    def forward(self, batch):
        input_ids=torch.from_numpy(batch["input_ids"]).to(self.dev)
        attention_mask=torch.from_numpy(batch["attention_mask"]).to(self.dev)
        token_type_ids=torch.from_numpy(batch["token_type_ids"]).to(self.dev)
        with torch.inference_mode():
            # pytorch
            # model_output = self.model.forward(input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     token_type_ids=token_type_ids)
            # start_scores = model_output.start_logits
            # end_scores = model_output.end_logits
            
            # tensor rt
            trt_output = self.trt_model(input_ids, attention_mask, token_type_ids)
            start_scores = trt_output["start_logits"]
            end_scores = trt_output["end_logits"]
            
            batch_ret = torch.stack([start_scores, end_scores], axis=-1).cpu().numpy()
            return {
                "output": batch_ret
            }
    
    def ready(self):
        pass

class BERT_Ray_SUT():
    def __init__(self, args):
        with open("bert_config.json") as f:
            config_json = json.load(f)
        model_file = os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")
        self.qsl = get_squad_QSL(args.max_examples)

        try:
            ray.init(address="auto")
        except:
            print("WARN: Cannot connect to existing Ray cluster.")
            print("We are going to start a new RAY cluster, but pay attention that")
            print("the cluster contains only one node.")
            print("If you want to use multiple nodes, please start the cluster manually via:")
            print("\tOn the head node, run `ray start --head`")
            print("\tOn other nodes, run `ray start --address=<head node IP>:6379`")
            ray.init()
            
        self.batch_size = BATCH_SIZE
        resources = ray.cluster_resources()
        num_gpus = int(resources.get('GPU', 0))
        
        print(f"The cluster has {num_gpus} GPUs.")
        
        self.actor_list = [TorchPredictor.remote(config_json, model_file, self.batch_size) for _ in range(num_gpus)]
        self.pool = ActorPool(self.actor_list)

        samples = []
        for i in range(self.qsl.count):
            sample = {}
            eval_features = self.qsl.get_features(i)
            sample["input_ids"] = np.array(eval_features.input_ids).astype(np.int32)
            sample["attention_mask"] = np.array(eval_features.input_mask).astype(np.int32)
            sample["token_type_ids"] = np.array(eval_features.segment_ids).astype(np.int32)
            samples.append(sample)
        self.samples = samples
        
        print("Waiting Actors init")
        for actor in self.actor_list:
            ray.get(actor.ready.remote())
        print("BERT_Ray_SUT construct complete")

    def issue_queries(self, query_samples):
        if len(query_samples) % self.batch_size != 0:
            print("ERROR: batch size must be a multiple of the number of samples")
            sys.exit(1)
            
        batch_samples = []
        i = 0
        while i < len(query_samples):
            batch_sample = {
                "input_ids": np.array([
                    self.samples[query_sample.index]["input_ids"]
                    for query_sample in query_samples[i:i+self.batch_size]]),
                "attention_mask": np.array([
                    self.samples[query_sample.index]["attention_mask"]
                    for query_sample in query_samples[i:i+self.batch_size]]),
                "token_type_ids": np.array([
                    self.samples[query_sample.index]["token_type_ids"]
                    for query_sample in query_samples[i:i+self.batch_size]]),
            }
            batch_samples.append(batch_sample)
            i = i + self.batch_size

        # print("samples len", len(batch_samples))
        batch_inference_results = list(self.pool.map_unordered(lambda a, v: a.forward.remote(v), batch_samples))

        cur_query_index = 0
        for batch_inference_result in batch_inference_results:
            batch_inference_result = batch_inference_result["output"]
            for inference_result in batch_inference_result:
                response_array = array.array("B", inference_result.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[cur_query_index].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])
                cur_query_index += 1

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_ray_sut(args):
    return BERT_Ray_SUT(args)
