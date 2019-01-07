import mxnet as mx
import numpy as np

params_mxnet=mx.nd.load('seq2cnn_model-0000.params')
params_numpy={}
for key in params_mxnet.keys():
    params_numpy[key]=params_mxnet[key].asnumpy()

np.save('params_numpy',params_numpy)



