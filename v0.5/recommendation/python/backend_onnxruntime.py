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

        '''
        incoming_bs = batch_dense_X.shape[0]
        model_saved_bs = 2048
        if (incoming_bs != model_saved_bs):
            print("WARNING: mismatch beween incoming " + str(incoming_bs) + " and model saved " + str(model_saved_bs) + " mini-batch size")
            fake_output = torch.zeros(size=(incoming_bs,1), dtype=torch.float32)
            return fake_output
        '''

        dict_inputs = {}

        # Dmitriy's approach to build dictionaries
        ind = 0
        for i in self.inputs:

            if "input.1" == i:
                dict_inputs[i] = batch_dense_X.numpy().astype(np.float32)
            
            elif "lS_o" == i:
                dict_inputs[i] = batch_lS_o.numpy().astype(np.int64)

            else:
                dict_inputs[i] = batch_lS_i[ind].numpy().astype(np.int64)
                ind = ind + 1
        '''
        # Maxim's approach to build dictionaries
        dict_inputs[self.inputs[0]] = batch_dense_X.numpy().astype(np.float32)
        dict_inputs[self.inputs[1]] = batch_lS_o.numpy().astype(np.int64)
        if False: #torch.is_tensor(batch_lS_i): # approach 1: tensor
            dict_inputs[self.inputs[2]] = batch_lS_i.numpy().astype(np.int64)
        else: # approach 2: list
            for j in range(26): # 26 sparse features
                dict_inputs[self.inputs[j+2]] = batch_lS_i[j].numpy().astype(np.int64)
        '''
        # predict and return output
        # print(dict_inputs)
        output = self.sess.run(output_names=self.outputs, input_feed=dict_inputs)
        output = torch.tensor(output, requires_grad=False).view(-1, 1)
        # print("output", output)
        # print("output.shape", output.shape)

        return output
