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

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""

        onnx_model = onnx.load(model_path)

        # TBD
        shape_dict = {'input_tensor:0': (1, 3, 224, 224)}

        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

        mod = relay.transform.DynamicToStatic()(mod)

        dtype_dict = {'input_tensor:0': 'float32'}
        mod_layout = 'NCHW'

        ctx = tvm.cpu(0)

        inputs = [k for k in shape_dict]
        outputs=['ArgMax:0'] # TBD

        build_conf={'relay.backend.use_auto_scheduler': False}
        opt_lvl = 3

        target='llvm -mcpu=znver3'
        target_host=None
        params={}

        with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
            graph_module = relay.build(mod,
                                       target=target,
                                       target_host=target_host,
                                       params=params)
        lib = graph_module

        m = graph_executor.GraphModule(lib['default'](ctx))

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
#        print (m.get_output(0))
#        print (m.get_output(1))
#        input('xyz9999')


#        opt = rt.SessionOptions()
        # enable level 3 optimizations
        # FIXME: enable below once onnxruntime 0.5 is released
        # opt.set_graph_optimization_level(3)
#        self.sess = rt.InferenceSession(model_path, opt)
        # get input and output names
        if not inputs:
            # From ONNX: TBD
            self.inputs = [meta.name for meta in self.sess.get_inputs()]
        else:
            self.inputs = inputs
        if not outputs:
            # From ONNX: TBD
            self.outputs = [meta.name for meta in self.sess.get_outputs()]
        else:
            self.outputs = outputs
        return self

    def predict(self, feed):
        """Run the prediction."""

        for iname, data in feed.items():
            self.sess.set_input(iname, tvm.nd.array(data))


#        import numpy as np
#
#        shape_dict = {'input_tensor:0': (1, 3, 224, 224)}
#        dtype_dict = {'input_tensor:0': 'float32'}
#        np.random.seed(0)
#        for iname, ishape in shape_dict.items():
#            np_data = (100 * np.random.uniform(size=ishape)).astype(dtype_dict[iname])
#            self.sess.set_input(iname, tvm.nd.array(np_data))

        self.sess.run()

        tvm_output = []
#        print (self.sess.get_num_outputs())
#        input('xyz999')
        for i in range(self.sess.get_num_outputs()):
#            print (i, self.sess.get_output(i))
            tvm_output.append(self.sess.get_output(i).asnumpy())

        #[array([66], dtype=int64)]


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
