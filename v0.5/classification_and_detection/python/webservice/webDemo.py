from flask import Flask
import sys
sys.path.append("..")
from main_noargs import load_model, inference_model


app = Flask(__name__)
#app.logger.disabled = True
# get the args and load model, do it only once time
load_model()
inference_model()

@app.route('/gpu')
def ssd_mobile_net():
    inference_model()
    return "good"

@app.route('/')
def index():
    return 'Hello World'

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,threaded=True)
