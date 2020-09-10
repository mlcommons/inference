"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import numpy as np
import backend
import torch


class BackendOnnxruntime(backend.Backend):
    def __init__(self, m_spa, ln_emb, ln_bot, ln_top, use_gpu=False, mini_batch_size=1):
        super(BackendOnnxruntime, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        opt = rt.SessionOptions()
        # enable level 3 optimizations
        # FIXME: enable below once onnxruntime 0.5 is released
        # opt.set_graph_optimization_level(3)
        # print("onnx load", model_path, inputs, outputs)
        self.sess = rt.InferenceSession(model_path, opt)
        # get input and output names
        if True: #not inputs:
            self.inputs = [meta.name for meta in self.sess.get_inputs()]
        else:
            self.inputs = inputs
        if True: #not outputs:
            self.outputs = [meta.name for meta in self.sess.get_outputs()]
        else:
            self.outputs = outputs
        return self

    def predict(self, batch_dense_X, batch_lS_o, batch_lS_i):
        """Run the prediction."""
        # print("onnx predict")
        # print(self.inputs)
        # print(self.outputs)

        # force list conversion
        # if torch.is_tensor(lS_o_onnx):
        #    lS_o_onnx = [lS_o_onnx[j] for j in range(len(lS_o_onnx))]
        # if torch.is_tensor(lS_i_onnx):
        #    lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
        # force tensor conversion
        # if isinstance(lS_o_onnx, list):
        #     lS_o_onnx = torch.stack(lS_o_onnx)
        # if isinstance(lS_i_onnx, list):
        #     lS_i_onnx = torch.stack(lS_i_onnx)

        dict_inputs = {}
        dict_inputs["dense_x"] = batch_dense_X.numpy().astype(np.float32)
        if torch.is_tensor(batch_lS_o):
            dict_inputs["offsets"] = batch_lS_o.numpy().astype(np.int64)
        else:  # list
            for i in range(len(batch_lS_o)):
                dict_inputs["offsets_"+str(i)] = batch_lS_o[i].numpy().astype(np.int64)
        if torch.is_tensor(batch_lS_i):
            dict_inputs["indices"] = batch_lS_i.numpy().astype(np.int64)
        else:  # list
            for i in range(len(batch_lS_i)):
                dict_inputs["indices_"+str(i)] = batch_lS_i[i].numpy().astype(np.int64)

        # predict and return output
        # print("dict_inputs", dict_inputs)
        output = self.sess.run(output_names=self.outputs, input_feed=dict_inputs)
        output = torch.tensor(output, requires_grad=False).view(-1, 1)
        # print("output", output)
        # print("output.shape", output.shape)

        return output
