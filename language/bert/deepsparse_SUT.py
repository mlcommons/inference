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
from deepsparse import compile_model, Scheduler
from deepsparse.utils import generate_random_inputs
from squad_QSL import get_squad_QSL


def batched_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        if (i+n) <= len(lst): # last batch will be dealt with outside
            yield lst[i:i + n]

def scenario_to_scheduler(scenario):
    if scenario == "SingleStream":
        return Scheduler.single_stream
    elif scenario == "Offline":
        return Scheduler.single_stream
    elif scenario == "Server":
        return Scheduler.multi_stream
    else:
        raise Exception(scenario)

class BERT_DeepSparse_SUT():
    def __init__(self, args):
        self.profile = args.profile
        self.model_path = args.model_path
        self.batch_size = args.batch_size
        self.scenario = args.scenario
        self.responses_sent = 0
        self.scheduler = scenario_to_scheduler(args.scenario)

        print("Loading ONNX model...", self.model_path)
        self.sess = compile_model(self.model_path, batch_size=self.batch_size, scheduler=self.scheduler)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

        print("Warming up engine...")
        warmup_inputs = generate_random_inputs(self.model_path, self.batch_size)
        for i in range(10):
            self.sess.run(warmup_inputs)

    def pad_to_batch(self, x):
        x_pad = np.zeros((self.batch_size, x.shape[1]), dtype=np.int64)
        x_pad[:x.shape[0], :x.shape[1]] = x
        return x_pad

    def process_batch(self, batched_features):
        pad_func = lambda x: self.pad_to_batch(x) if len(batched_features) != self.batch_size else x
        fd = [
            pad_func(np.stack(
                np.asarray([f.input_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :,
                         :]),
            pad_func(np.stack(
                np.asarray([f.input_mask for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :,
                              :]),
            pad_func(np.stack(
                np.asarray([f.segment_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0,
                              :, ])
        ]
        return fd

    def issue_queries(self, query_samples):
        if self.scenario == "SingleStream" or self.scenario == "Server":
            for i in range(len(query_samples)):
                eval_features = self.qsl.get_features(query_samples[i].index)
                fd = [np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :],
                    np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :],
                    np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :]]
                scores = self.sess.run(fd)
                output = np.stack(scores, axis=-1)[0]

                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])

        elif self.scenario == "Offline":
            #  Extracting features to be split into batches
            eval_features = [self.qsl.get_features(query_samples[i].index) for i in range(len(query_samples))]
            for batch_ind, batched_features in enumerate(batched_list(eval_features, self.batch_size)):
                fd = self.process_batch(batched_features)
                scores = self.sess.run(fd)
                output = np.stack(scores, axis=-1)

                # sending responses individually
                for sample in range(self.batch_size):
                    response_array = array.array("B", output[sample].tobytes())
                    bi = response_array.buffer_info()
                    response = lg.QuerySampleResponse(query_samples[self.responses_sent].id, bi[0], bi[1])
                    self.responses_sent += 1
                    lg.QuerySamplesComplete([response])

            # batch remainder in sync
            last_ind = (batch_ind + 1) * self.batch_size
            if last_ind < len(eval_features) - 1:
                batched_features = eval_features[last_ind:]
                fd = self.process_batch(batched_features)
                scores = self.sess.run(fd)
                output = np.stack(scores, axis=-1)[:len(batched_features)]
                for sample in range(len(output)):
                    response_array = array.array("B", output[sample].tobytes())
                    bi = response_array.buffer_info()
                    response = lg.QuerySampleResponse(query_samples[self.responses_sent].id, bi[0],
                                                        bi[1])
                    self.responses_sent += 1
                    lg.QuerySamplesComplete([response])

        else:
            raise Exception("Unknown scenario", scenario)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_deepsparse_sut(args):
    return BERT_DeepSparse_SUT(args)
