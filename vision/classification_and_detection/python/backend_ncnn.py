import ncnn
import numpy as np
import backend
from ncnn_models import *

class BackendNCNN(backend.Backend):
    def __init__(self):
        super(BackendNCNN, self).__init__()

    def version(self):
        return ncnn.__version__

    def name(self):
        return "ncnn"

    def image_format(self):
        """image_format. For ncnn it is NCHW."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        param_file, bin_file = f"{model_path}.param", f"{model_path}.bin"
        if param_file.endswith("resnet50_v1.param"):
            # download model files if doesn't
            self.net = Resnet50(param_file, bin_file)
        else:
            import sys
            print("please add your ncnn model .param and .bin files to dir named 'resnet'")
            sys.exit()
        
        if not inputs:
            self.inputs = [self.net.input_name]
        else:
            self.inputs = inputs
        if not outputs:
            self.outputs = [self.net.output_name]
        else:
            self.outputs = outputs
        return self

    def predict(self, feed):
        return self.net(feed[self.net.input_name][0])
