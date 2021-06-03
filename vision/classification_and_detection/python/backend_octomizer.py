"""
OctoML Octomizer backend (https://octoml.ai)
"""

import onnx
import onnxruntime as rt

import backend

import tvm
from tvm import relay

import numpy as np

from threading import Lock

import re
import os

import importlib

octomizer_prefix = 'Octomizer MLPerf backend: '

class BackendOctomizer(backend.Backend):
    def __init__(self):
        super(BackendOctomizer, self).__init__()
        self.sess = None
        self.lock = Lock()

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "tvm"

    def image_format(self):
        """image_format."""
        # We use ONNX format, which is always NCHW.
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""

        model_package_name=os.environ.get('CK_ENV_OCTOMIZER_WHEEL_PYTHON_MODULE','')
        if model_package_name=='':
            raise ValueError(octomizer_prefix + "CK_ENV_OCTOMIZER_WHEEL_PYTHON_MODULE env from CK package is not defined")
            exit(1)

        print ('')
        print (octomizer_prefix + 'importing Octomizer model ...')

        ck_octomizer_model=importlib.import_module(model_package_name)

        self.sess=ck_octomizer_model.model()

        print ('')
        print (octomizer_prefix + 'model ready ...')
        print ('')

        if not inputs:
            self.inputs = ['octomizer_input']
        if not outputs:
            self.outputs = ['octomizer_output']

        return self


    def predict(self, feed):
        """Run the prediction."""

        self.lock.acquire()

        sess = self.sess

        input_data = []

        # Prepare Octomizer inputs (only 1 at the moment)
        for iname, data in feed.items():
            max_batchsize = self.max_batchsize
            batch_size = len(data)

            # Fill in batch if less than max batch size (emulate dynamic batching)
            if batch_size <  max_batchsize:
                data_extra = np.stack([data[0]] * (max_batchsize-batch_size))
                data = np.vstack((data, data_extra))
            elif batch_size > max_batchsize:
                raise ValueError("Internal MLPerf error: dynamic batch size > max batch size")

            input_data = data

            # Only one input in Octomizer?
            break

        # Run TVM inference
        tvm_output = [sess(input_data)[:batch_size]]

        self.lock.release()

        return tvm_output
