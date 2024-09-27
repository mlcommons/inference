from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

node = ""

model_mem_size = None
use_gpu = False
free_mem = None

# setting semaphore
def set_semaphore(lockVar):
    global semaphore
    semaphore = threading.Semaphore(lockVar)


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
    response = backend.inference_call(query)
    print(f"dnn model response:{response}")
    return response


def postprocess(query):
    """[SUT Node] A dummy postprocess."""
    # Here may come for example a postprocessing call, e.g., NMS, detokenization, etc.
    response = query
    return response


@app.route('/predict/', methods=['POST'])
def predict():
    """Receives a query (e.g., a text) runs inference, and returns a prediction."""
    semaphore.acquire() # wait to acquire semaphore    
    try:
        query = request.get_json(force=True)['query']
        result = postprocess(dnn_model(preprocess(query)))
        print(result)
        return jsonify(result=result)
    finally:
        # semaphore is released after processing
        semaphore.release()


@app.route('/getname/', methods=['POST', 'GET'])
def getname():
    """Returns the name of the SUT."""
    return jsonify(name=f'Demo SUT (Network SUT) node' + (' ' + node) if node else '')