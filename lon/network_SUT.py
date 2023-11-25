# Copyright 2023 MLCommons. All Rights Reserved.
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
# =============================================================================

import array
import json
import os
import sys
sys.path.insert(0, os.getcwd())


import argparse
from flask import Flask, request, jsonify


app = Flask(__name__)


node = ""

def set_backend(b):
    global backend
    backend = b

def preprocess(query):
    """[SUT Node] A dummy preprocess."""
    # Here may come for example batching, tokenization, resizing, normalization, etc.
    response = query
    return response


def dnn_model(query):
    # Here may come for example a call to a dnn model such as resnet, bert, etc.
    response = backend.process_sample(query)
    return response


def postprocess(query):
    """[SUT Node] A dummy postprocess."""
    # Here may come for example a postprocessing call, e.g., NMS, detokenization, etc.
    response = query
    return response


@app.route('/predict/', methods=['POST'])
def predict():
    """Receives a query (e.g., a text) runs inference, and returns a prediction."""
    query = request.get_json(force=True)['query']
    result = postprocess(dnn_model(preprocess(query)))
    return jsonify(result=result)


@app.route('/getname/', methods=['POST', 'GET'])
def getname():
    """Returns the name of the SUT."""
    return jsonify(name=f'Demo SUT (Network SUT) node' + (' ' + node) if node else '')
