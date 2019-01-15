"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring

import tensorflow as tf
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
import backend


class BackendTflite(backend.Backend):
    def __init__(self):
        super(BackendTflite, self).__init__()
        self.sess = None

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tflie"

    def image_format(self):
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        self.sess = interpreter_wrapper.Interpreter(model_path=model_path)
        self.sess.allocate_tensors()
        self.input2index = {i["name"]: i["index"] for i in self.sess.get_input_details()}
        self.output2index = {i["name"]: i["index"] for i in self.sess.get_output_details()}
        self.inputs = list(self.input2index.keys())
        self.oututs = list(self.output2index.keys())
        return self

    def predict(self, feed):
        # TODO: don't think this is thread safe so on one invoke() at a time is safe.
        for k, v in self.input2index.items():
            self.sess.set_tensor(v, feed[k])
        self.sess.invoke()
        result = []
        for k, v in self.output2index.items():
            result.append(self.sess.get_tensor(v))
        return result
