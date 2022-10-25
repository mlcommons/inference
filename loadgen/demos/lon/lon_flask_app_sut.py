# Copyright 2019 The MLPerf Authors. All Rights Reserved.
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


"""Python demo showing how to use the MLPerf Inference load generator bindings over the network.
This part of the demo runs the `server side` of the test.
A corresponding `client side` dummy implemented in py_demo_server_lon.py.

The server side is implemented using Flask.
The SUT server supports two endpoints:
- /predict/ : Receives a query (e.g., a text) runs inference, and returns a prediction.
- /getname/ : Get the name of the SUT.

The current implementation is a dummy implementation, which does not use 
a real DNN model, batching, or pre/postprocessing code,
but rather just returns subset of the input query as a response,
Yet, it illustrates the basic structure of a SUT server.
"""

import argparse
from flask import Flask, request, jsonify


app = Flask(__name__)


def preprocess(query):
    # A dummy preprocessing function.
    # Here may come for example batching, tokenization, resizing, normalization, etc.
    return query


def dnn_model(query):
    # A dummy dnn model call.
    # Here may come for example a call to a dnn model such as resnet, bert, etc.
    return query


def postprocess(query):
    # A dummy postprocess.
    # Here may come for example a postprocessing call, e.g., NMS, detokenization, etc.

    # This current dummy implementation just returns part of the input query as a response:
    # what_is_my_dummy_feature_{i}? --> dummy_feature_{i}
    return query.replace("what_is_my_", "").replace("?", "")


@app.route('/predict/', methods=['POST'])
def predict():
    query = request.get_json(force=True)['query']
    result = postprocess(dnn_model(preprocess(query)))
    return jsonify(result=result)


@app.route('/getname/', methods=['POST', 'GET'])
def getname():
    return jsonify(name='Dummy SUT (Network SUT)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
