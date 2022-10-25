import argparse
from flask import Flask, request, jsonify

# A simple app, illustrating a dummy SUT server over-the-network.
# The SUT server supports two endpoints:
# - /predict/ : Receives a query (e.g., a text) runs inference, and returns a prediction.
# - /getname/ : Get the name of the SUT.
#
# The current implementation is a dummy implementation, which does not use 
# a real DNN model, batching, or pre/postprocessing code,
# but rather just returns subset of the input query as a response,
# Yet, it illustrates the basic structure of a SUT server.
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

@app.route('/getname/', methods=['POST','GET'])
def getname():
    return jsonify(name='Dummy SUT (Network SUT)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
