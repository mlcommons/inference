"""
pytoch/caffe2 backend via onnx
https://pytorch.org/docs/stable/onnx.html

TODO: this currently does not work for our resnet50.onnx model
because caffe2 seems to have issues with the onnx pad operator.
ONNX FATAL: [enforce fail at backend.cc:811] . Caffe2 only supports padding 2D Tensor,
whereas padding is [0, 3, 3, 0, 0, 3, 3, 0, ]
"""

# pylint: disable=unused-argument,missing-docstring

import onnx
import torch
import caffe2.python.onnx.backend as pt_backend
import backend


class BackendPytorch(backend.Backend):
    def __init__(self):
        super(BackendPytorch, self).__init__()
        self.sess = None
        self.model = None

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch/caffe2"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)

        # find inputs from the model if not passed in by config
        if not inputs:
            self.inputs = []
            initializers = set()
            for i in model.graph.initializer:
                initializers.add(i.name)
            for i in model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)
        # find outputs from the model if not passed in by config
        if not outputs:
            self.outputs = []
            for i in model.graph.output:
                self.outputs.append(i.name)

        # prepare the backend
        device = "CUDA:0" if torch.cuda.is_available() else "CPU"
        self.sess = pt_backend.prepare(self.model, device)
        return self

    def predict(self, feed):
        return rep.run(feed)
