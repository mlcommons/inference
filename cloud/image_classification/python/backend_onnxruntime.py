"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring

import onnxruntime as rt

import backend


class BackendOnnxruntime(backend.Backend):
    def __init__(self):
        super(BackendOnnxruntime, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        return "onnxruntime"

    def load(self, model_path, inputs=None, outputs=None):
        self.sess = rt.InferenceSession(model_path)
        # get inpput and output names
        self.inputs = [meta.name for meta in self.sess.get_inputs()]
        self.outputs = [meta.name for meta in self.sess.get_outputs()]
        return self

    def predict(self, feed):
        return self.sess.run(self.outputs, feed)
