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

import threading
import requests
import array
import time
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from transformers import BertTokenizer
from create_squad_data import read_squad_examples, convert_examples_to_features
from absl import app
from absl import flags

import mlperf_loadgen as lg
import squad_QSL


# To support feature cache.
import pickle

max_seq_length = 384
max_query_length = 64
doc_stride = 128


class bert_onnxruntime_QDL:
    """QDL acting as a proxy to the SUT.
    This QDL communicates with the SUT via HTTP.
    It uses two endpoints to communicate with the SUT:
    - /predict/ : Send a query to the SUT and get a response.
    - /getname/ : Get the name of the SUT. Send a getname to the SUT and get a response.
    """

    def __init__(self, qsl: squad_QSL.SQuAD_v1_QSL, sut_server_addr: list):
        """
        Constructor for the QDL.
        Args:
            qsl: The QSL to use.
            sut_server_addr: A list of addresses of the SUT.
        """
        self.qsl = qsl
        self.quantized = False
 
        # Construct QDL from the python binding
        self.qdl = lg.ConstructQDL(
            self.issue_query, self.flush_queries, self.client_get_name)
        self.sut_server_addr = sut_server_addr
        self.num_nodes = len(sut_server_addr)

        # For round robin between the SUTs:
        self.next_sut_id = 0
        self.lock = threading.Lock()

    def issue_query(self, query_samples):
        """Process the query to send to the SUT"""
        threading.Thread(target=self.process_query_async,
                         args=[query_samples]).start()

    def flush_queries(self):
        """Flush the queries. Dummy implementation."""
        pass

    def process_query_async(self, query_samples):
        """
        This function is called by the Loadgen in a separate thread.
        It is responsible for
            1. Creating a query for the SUT, by reading the features from the QSL.
            2. Sending the query to the SUT.
            3. Waiting for the response from the SUT.
            4. Deserializing the response.
            5. Calling mlperf_loadgen.QuerySamplesComplete(query_samples, response)
        Args:
            query_samples: A list of QuerySample objects.
        """

        responses = []
        for i in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[i].index)
            if self.quantized:
                fd = {
                    "input_ids": np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :].tolist(),
                    "attention_mask": np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :].tolist(),
                    "token_type_ids": np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :].tolist()
                }
            else:
                fd = {
                    "input_ids": np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :].tolist(),
                    "input_mask": np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :].tolist(),
                    "segment_ids": np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :].tolist()
                }
            encoded_eval_features = fd

            output = self.client_predict(encoded_eval_features, query_samples[i].index)
            output = np.array(output).astype(np.float32)
            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()

            responses.append(lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1]))
        lg.QuerySamplesComplete(responses)

    def get_sut_id_round_robin(self):
        """Get the SUT id in round robin."""
        with self.lock:
            res = self.next_sut_id
            self.next_sut_id = (self.next_sut_id + 1) % self.num_nodes
        return res

    def client_predict(self, query, id):
        """Serialize the query, send it to the SUT in round robin, and return the deserialized response."""
        url = '{}/predict/'.format(self.sut_server_addr[self.get_sut_id_round_robin()])
        #print(query)
        response = requests.post(url, json={'query': query, id: id})
        return response.json()['result']

    def client_get_name(self):
        """Get the name of the SUT from ALL the SUTS."""
        if len(self.sut_server_addr) == 1:
            return requests.post(f'{self.sut_server_addr[0]}/getname/').json()['name']
    
        sut_names = [requests.post(f'{addr}/getname/').json()['name'] for addr in self.sut_server_addr]
        return "Multi-node SUT: " + ', '.join(sut_names)

    def __del__(self):
        lg.DestroyQDL(self.qdl)


