"""
pytoch native backend for dlrm
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend
# from dlrm_s_pytorch import DLRM_Net
import tensorflow as tf
from tf_dlrm import logits_fn, rand_features_np
import numpy as np
import collections
from typing import Dict, Any
import sys

class BackendTF(backend.Backend):
    def __init__(self, dim_embed, vocab_sizes, mlp_bottom, mlp_top):
        super(BackendTF, self).__init__()
        self.sess = None
        self.model = None
        self.params = collections.defaultdict()

        self.params["dim_embed"] = dim_embed
        self.params["vocab_sizes"] = vocab_sizes.tolist()

        self.params["mlp_bottom"] = mlp_bottom.tolist()
        self.params["mlp_top"] = mlp_top.tolist()

        self.params["num_dense_features"] = self.params["mlp_bottom"][0]
        self.params["num_sparse_features"] = len(self.params["vocab_sizes"])
        self.params["num_tables_in_ec"] = 26

        self.params["learning_rate"] = 0.01
        self.params["opt_skip_gather"] = True

        self.params["is_training"] = True

    def version(self):
        return tf.__version__

    def name(self):
        return "tf-dlrm"

    def load(self, model_path, inputs=None, outputs=None):
        # debug prints
        # print(model_path, inputs, outputs)

        self.model_path = model_path

        num_d = self.params["num_dense_features"]
        num_s = self.params["num_sparse_features"]
        minsize = min(self.params["vocab_sizes"])
        print("stat: ", num_d, num_s, minsize)

        self.graph = tf.Graph()

        with self.graph.as_default():

            features_int_np, features_cat_np = rand_features_np(1, num_d, num_s, minsize)

            features_int = tf.placeholder(tf.float32, [None, num_d], name="ph_1")
            features_cat = tf.placeholder(tf.int32, [None, num_s], name="ph_2")

            preds = logits_fn(features_int, features_cat, self.params)
            preds = tf.identity(preds, name="preds")

            init_op = tf.compat.v1.global_variables_initializer()

            self.sess = tf.compat.v1.Session(graph=self.graph)

            self.sess.run(init_op)
            self.sess.run(preds, feed_dict = {features_int : features_int_np, features_cat : features_cat_np} )

        self.params["is_training"] = False

        print("load() finished ...")

        return self

    def predict(self, batch_dense_X, batch_lS_o, batch_lS_i):

        # features from input to this function
        # torch -> numpy -> tf -> numpy -> torch

        # dense features
        pytorch_tensor = batch_dense_X.detach().cpu()
        np_tensor_int = pytorch_tensor.numpy()

        # sparse features
        pytorch_tensor2 = batch_lS_i.detach().cpu()
        np_tensor2 = pytorch_tensor2.numpy()
        np_tensor_cat = np.transpose(np_tensor2)

        # print_op_preds = tf.print(estim.predictions, output_stream=sys.stdout)

        out_operation = self.graph.get_operation_by_name('preds')

        ph_1 = self.graph.get_tensor_by_name('ph_1:0')
        ph_2 = self.graph.get_tensor_by_name('ph_2:0')

        np_tensor_out = out_operation.outputs[0].eval(session=self.sess, feed_dict = {ph_1 : np_tensor_int, ph_2 : np_tensor_cat})
        # print("1st output element: ", np_tensor_out[:1])

        output = torch.from_numpy(np_tensor_out)
        return output
