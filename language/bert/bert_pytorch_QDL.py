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

from bert_base_QDL import bert_base_QDL

class bert_pytorch_QDL(bert_base_QDL):

    def __init__(self, qsl: squad_QSL.SQuAD_v1_QSL, sut_server_addr: list):
        """
        Constructor for the QDL.
        Args:
            qsl: The QSL to use.
            sut_server_addr: A list of addresses of the SUT.
        """
        super().__init__(qsl, sut_server_addr)


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
            encoded_eval_features = {
                    "input_ids": eval_features.input_ids,
                    "input_mask": eval_features.input_mask,
                    "segment_ids": eval_features.segment_ids
                    }
            output = self.client_predict(encoded_eval_features, query_samples[i].index)
            output = np.array(output).astype(np.float32)
            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()

            responses.append(lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1]))
        lg.QuerySamplesComplete(responses)

