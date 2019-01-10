"""
null backend
"""

# pylint: disable=unused-argument,missing-docstring

import time
import backend


class BackendNull(backend.Backend):
    def __init__(self):
        super(BackendNull, self).__init__()

    def version(self):
        return "-"

    def name(self):
        return "null"

    def load(self, model_path, inputs=None, outputs=None):
        if outputs:
            self.outputs = outputs
        if inputs:
            self.inputs = inputs
        return self

    def predict(self, feed):
        # yield to give the thread that feeds our queue a chance to run
        time.sleep(0)
        # return something fake
        return [[0]]
