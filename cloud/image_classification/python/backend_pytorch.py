"""
pytoch/caffe2 backend via onnx
https://pytorch.org/docs/stable/onnx.html

FIXME: this currently does not work for our resnet50.onnx model

caffe2 complains about the following:

[E ../caffe2/core/operator_schema.cc:64] Input index 3
(resnet_model/batch_normalization/moving_mean:0) and output idx 1
(resnet_model/batch_normalization/FusedBatchNorm:3) are not in-place but should be as required by op SpatialBN
schema->Verify(operator_def). Operator def did not pass schema checking: input: "resnet_model/conv2d/Conv2D:0"
input: "resnet_model/batch_normalization/gamma:0"
input: "resnet_model/batch_normalization/beta:0"
input: "resnet_model/batch_normalization/moving_mean:0"
input: "resnet_model/batch_normalization/moving_variance:0"
...
name: "resnet_model/batch_normalization/FusedBatchNorm"
type: "SpatialBN" args{name: "epsilon" f: 1.001e-05} device_option {device_type: 0 device_id: 0}

and

WARNING:caffe2.python.workspace:Original python traceback for operator `2` in network `tf2onnx_init`
ERROR:main:execute_parallel thread: [enforce fail at ../caffe2/core/workspace.cc:229] .
I respectfully refuse to overwrite an existing net of the same name "tf2onnx_init", unless you specify overwrite=true.
"""

# pylint: disable=unused-argument,missing-docstring,,useless-super-delegation

import caffe2.python.onnx.backend as pt_backend
import onnx
import torch  # needed to get version and cuda setup

import backend


class BackendPytorch(backend.Backend):
    def __init__(self):
        super(BackendPytorch, self).__init__()
        self.sess = None
        self.model = None

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = onnx.load(model_path)

        # find inputs from the model if not passed in by config
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)
        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)

        # prepare the backend
        device = "CUDA:0" if torch.cuda.is_available() else "CPU"
        self.sess = pt_backend.prepare(self.model, device)
        return self

    def predict(self, feed):
        return self.sess.run(feed)
