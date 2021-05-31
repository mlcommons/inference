"""
TVM backend (https://github.com/apache/tvm)
"""

import onnx
import onnxruntime as rt

import backend

import tvm
from tvm import relay
from tvm.contrib import graph_executor

import numpy as np

class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "TVM"

    def image_format(self):
        """image_format."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None, max_batchsize=None):
        """Load model and find input/outputs from the model file."""

        # First attempt to detect input and output names via ONNX run time.
        # See backend_onnxruntime.py
        #
        # Even if inputs/outputs can be defined by MLPerf
        # TVM will need extra info about shapes to be properly initialized!

        self.inputs = inputs
        self.outputs = outputs

        opt = rt.SessionOptions()
        tmp_sess = rt.InferenceSession(model_path, opt)

        if not inputs:
            self.inputs = [meta.name for meta in tmp_sess.get_inputs()]
        if not outputs:
            self.outputs = [meta.name for meta in tmp_sess.get_outputs()]

        # Detect shapes and set max batch size.
        # If batch size is < max batch size, fill in with empty ones
        # In the future, we should support dynamic batch sizes in TVM
        shape_dict = {}
        bsize_dict = {}
        dtype_dict = {}

        for meta in tmp_sess.get_inputs():
            input_name = meta.name
            input_type = meta.type
            input_shape = meta.shape

            if input_type == 'tensor(float)':
                dtype_dict[input_name] = 'float32'

            # We expect that input_shape[0] == batch_size
            input_shape[0] = max_batchsize
            shape_dict[input_name] = tuple(input_shape)

            bsize_dict[input_name] = max_batchsize

        print ('')
        print ('TVM: input shape(s): '+str(shape_dict))
        print ('TVM: input type: '+str(dtype_dict))
        self.input_shapes = shape_dict
        self.input_batch_sizes = bsize_dict

        # We do not need ONNX runtime anymore
        del tmp_sess

        # Load model via ONNX to be used with TVM
        onnx_model = onnx.load(model_path)

        # Init model for different batch sizes
        ctx = tvm.cpu(0)

        mod_layout = 'NCHW'
        build_conf={'relay.backend.use_auto_scheduler': False}
        opt_lvl = 3
        target='llvm -mcpu=znver3'
        target_host=None
        params={}

        print ('')
        print ('TVM: import model ...')
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

        print ('')
        print ('TVM: transform to static ...')
        mod = relay.transform.DynamicToStatic()(mod)

        print ('')
        print ('TVM: build model ...')
        with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
            graph_module = relay.build(mod,
                                       target=target,
                                       target_host=target_host,
                                       params=params)
        lib = graph_module

        print ('')
        print ('TVM: init graph ...')

        self.sess = graph_executor.GraphModule(lib['default'](ctx))

        print ('')
        print ('TVM: model ready ...')

        return self


    def predict(self, feed):
        """Run the prediction."""

        sess = self.sess

        for iname, data in feed.items():
            max_batchsize = self.input_batch_sizes[iname]
            batch_size = len(data)

            if batch_size <  max_batchsize:
                data_extra = np.stack([data[0]] * (max_batchsize-batch_size))
                data = np.vstack((data, data_extra))
            elif batch_size > max_batchsize:
                raise ValueError("Internal MLPerf error: dynamic batch size > max batch size")

            sess.set_input(iname, tvm.nd.array(data))

        sess.run()

        tvm_output = []
        for i in range(sess.get_num_outputs()):
            # Take only the output of batch size for dynamic batches
            tvm_output.append(sess.get_output(i).asnumpy()[:batch_size])

        return tvm_output
