"""
TVM backend (https://github.com/apache/tvm)
"""

import onnx
import onnxruntime as rt

import backend

import tvm
from tvm import relay
from tvm.contrib import graph_executor

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

        # Detect shapes and set batch size.
        # Shape will be for the batch size 1 and for the max batch size (can be 1 too).
        # This is needed to set up static batch size in TVM.
        # We will add support for dynamic batch sizes later.
        shape_dict = {1:{}}
        dtype_dict = {}

        for meta in tmp_sess.get_inputs():
            input_name = meta.name
            input_type = meta.type
            input_shape = meta.shape

            if input_type == 'tensor(float)':
                dtype_dict[input_name] = 'float32'

            if len(input_shape)>0:
                input_shape[0]=1
                shape_dict[1][input_name] = tuple(input_shape)
                if max_batchsize and max_batchsize>1:
                    input_shape[0] = max_batchsize
                    shape_dict[max_batchsize]={input_name: tuple(input_shape)}

        print ('Input shape: '+str(shape_dict))
        print ('Input type: '+str(dtype_dict))

        # We do not need ONNX runtime anymore
        del tmp_sess

        # Load model via ONNX to be used with TVM
        onnx_model = onnx.load(model_path)


        # Init model for different batch sizes
        m={}

        ctx = tvm.cpu(0)

        mod_layout = 'NCHW'
        build_conf={'relay.backend.use_auto_scheduler': False}
        opt_lvl = 3
        target='llvm -mcpu=znver3'
        target_host=None
        params={}

        for batch_size in shape_dict:

            shape=shape_dict[batch_size]

            mod, params = relay.frontend.from_onnx(onnx_model, shape, freeze_params=True)

            mod = relay.transform.DynamicToStatic()(mod)

            with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                graph_module = relay.build(mod,
                                           target=target,
                                           target_host=target_host,
                                           params=params)
            lib = graph_module

            m[batch_size] = graph_executor.GraphModule(lib['default'](ctx))

        self.sess = m


#        import numpy as np
#        shape_dict = {'input_tensor:0': (1, 3, 224, 224)}
#        dtype_dict = {'input_tensor:0': 'float32'}
#        np.random.seed(0)
#        for iname, ishape in shape_dict.items():
#            np_data = (100 * np.random.uniform(size=ishape)).astype(dtype_dict[iname])
#            m.set_input(iname, tvm.nd.array(np_data))
#
#        m.run()
#
#        tvm_output = []

        return self

    def predict(self, feed):
        """Run the prediction."""

        batch_size = len(feed['input_tensor:0'])

        if batch_size not in self.sess:
            raise ValueError("TBD: TVM was not initialized with the dynamic batch size ("+str(batch_size)+')')

        sess = self.sess[batch_size]

        for iname, data in feed.items():
            sess.set_input(iname, tvm.nd.array(data))


#        import numpy as np
#
#        shape_dict = {'input_tensor:0': (1, 3, 224, 224)}
#        dtype_dict = {'input_tensor:0': 'float32'}
#        np.random.seed(0)
#        for iname, ishape in shape_dict.items():
#            np_data = (100 * np.random.uniform(size=ishape)).astype(dtype_dict[iname])
#            self.sess.set_input(iname, tvm.nd.array(np_data))

        sess.run()

        tvm_output = []
        for i in range(sess.get_num_outputs()):
            tvm_output.append(sess.get_output(i).asnumpy())


        return tvm_output


#input_tensor:0 [[[[54.88135   71.518936  60.276337  ... 87.26507   27.354204
#    79.80468  ]
#   [18.563595  95.27917   68.748825  ... 34.851936  81.49665
#    98.54914  ]
#   [96.89717   90.494835  29.655626  ... 63.91869   39.916115
#    43.176014 ]

#{'input_tensor:0': <tvm.nd.NDArray shape=(1, 3, 224, 224), cpu(0)>
#array([[[[54.88135  , 71.518936 , 60.276337 , ..., 87.26507  ,
#          27.354204 , 79.80468  ],
#         [18.563595 , 95.27917  , 68.748825 , ..., 34.851936 ,
#          81.49665  , 98.54914  ],
#         [96.89717  , 90.494835 , 29.655626 , ..., 63.91869  ,
#          39.916115 , 43.176014 ],
#         ...,
#         [14.080519 , 23.348122 , 86.754555 , ..., 58.712563 ,
#          77.75878  , 24.598848 ],
#         [50.2949   , 54.871693 , 40.29856  , ..., 86.43582  ,
#          82.24137  , 85.11621  ],
#         [83.33574  , 19.46625  , 27.752804 , ..., 17.514502 ,
#          76.58444  , 14.449131 ]],
