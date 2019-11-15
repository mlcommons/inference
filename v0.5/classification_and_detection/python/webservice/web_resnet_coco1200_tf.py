from flask import Flask
from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
import sys
sys.path.append("..")
from service_resnet_coco1200_tf import load_model, inference_model


app = Flask(__name__)
#app.logger.disabled = True
# get the args and load model, do it only once time
load_model()
inference_model()
@app.route('/gpu')
def ssd_mobile_net():
    inference_model()
    return "ok"

@app.route('/gpu1')
def index():
    return 'Hello World'

if __name__ == '__main__':
    #app.run(host="0.0.0.0",port=8888,threaded=True,processes=True)
    WSGIServer(("0.0.0.0", 5000), app).serve_forever()
