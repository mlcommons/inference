"""
TVM backend for MLPerf inference vision benchmark
Developers: Alexander Peskov, Thierry Moreau, Grigori Fursin
"""

import backend

import tvm
from tvm import auto_scheduler
from tvm.contrib import graph_executor

import numpy as np

import re
import os
import multiprocessing

g_graph = None


class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()
        self.arena_num = 1
        self.arena_size = multiprocessing.cpu_count()
        self.lib = None
        self.graph = None
        self.executor_type = None
        self.max_batchsize = None
        self.pool = None

    def version(self):
        return "N/A : TODO"

    def name(self):
        """Name of the runtime."""
        return "tvm"

    def image_format(self):
        """Requested image_format. Use a more popular layout NCHW"""
        return "NCHW"

    def create_omp_args(self, arena_idx):
        idx_start = self.arena_size * arena_idx
        cur_arena_size = min(multiprocessing.cpu_count() - idx_start, self.arena_size)
        # idx_end = idx_start + cur_arena_size

        # OMP_PLACES="{N},{N+1},{N+2},...,{N+SZ}"
        # arena_places_str = "{" + "},{".join(str(i) for i in range(idx_start, idx_end)) + "}"

        return {
                "TVM_NUM_THREADS": str(cur_arena_size),
                "OMP_NUM_THREADS": str(cur_arena_size),
                # "OMP_PLACES": arena_places_str,
                # "OMP_PROC_BIND": "true"
        }

    @staticmethod
    def set_omp_envs(omp_args):
        for env_arg in omp_args:
            os.environ[env_arg[0]] = env_arg[1]

    def load_impl(self, model_path, inputs, outputs, max_batchsize):

        # Check inputs and outputs
        # Normally should be specified by MLPerf, by the command line
        # By default taken from CK packages meta to ensure reproducibility and extensibility
        x = os.environ.get('ML_MODEL_INPUT_LAYERS','').strip()
        if x != '':
           inputs = x.split(',')

        x = os.environ.get('ML_MODEL_OUTPUT_LAYERS','').strip()
        if x != '':
           outputs = x.split(',')

        self.inputs = inputs
        self.outputs = outputs

        self.output_order=None
        tmp=os.environ.get('MLPERF_TVM_OUTPUT_ORDER','')
        if tmp!='':
            import json
            self.output_order=json.loads('['+tmp+']')

        # Detect working/tmp directory to store and retreive compiled models
        work_dir = os.environ.get('MLPERF_TMP_DIR','')
        if work_dir == '':
           work_dir = os.environ.get('CK_PROGRAM_TMP_DIR','')
        if work_dir == '':
           import tempfile
           work_dir = tempfile.gettempdir()
        if work_dir == '':
           work_dir = '/tmp'

        # Check if load precompiled model
        compiled_model = os.path.join(work_dir, 'model-tvm.so')
        if model_path.endswith('.so') or model_path.endswith('.dylib'):
           compiled_model = model_path

           if not os.path.isfile(compiled_model):
               print ('')
               raise Exception("Error: Model file {} not found!".format(compiled_model))

        if os.environ.get('MLPERF_DELETE_COMPILED_MODEL','').strip().lower()=='yes' and \
           os.path.isfile(compiled_model):
              os.remove(compiled_model)

        # TODO(@peskov): who specify that?? Only outside? Looks like TVM specific WA
        # Max batch size should be passed from main function
        self.max_batchsize = max_batchsize

        # Select target (default: cpu)
        # TBD(@gfursin): need to provide better customization
        # of a target via external variables that can be passed
        # from CK workflows
        if os.environ.get('MLPERF_DEVICE','')=='gpu':
           ctx = tvm.cuda(0)
        else:
           ctx = tvm.cpu(0)

        # If precompiled model found, load it directly
        if os.path.isfile(compiled_model):
           print ('TVM: loading model '+compiled_model)
           self.lib = tvm.runtime.load_module(compiled_model)

        else:
           ############################################################################
           # Import model to TVM
           from tvm import relay

           input_shapes = os.environ.get('ML_MODEL_INPUT_SHAPES','').strip()
           if input_shapes == '':
               print ('')
               raise Exception("Error: ML_MODEL_INPUT_SHAPES environment variable is not defined!")

           input_shapes = input_shapes.replace('BATCH_SIZE', str(max_batchsize))

           print ('TVM model: '+model_path)

           build_conf = {}
           params = {}

           if model_path.endswith('.pt'):
              import torch
              from tvm.relay.build_module import bind_params_by_name

              shape_list = eval('[' + input_shapes + ']')

              print ('TVM shape list: '+str(shape_list))

              x=os.environ.get('MLPERF_TVM_TORCH_QUANTIZED_ENGINE','')
              if x!='':
                 torch.backends.quantized.engine = x
              pytorch_model = torch.jit.load(model_path)
              pytorch_model.eval()

              mod, params = relay.frontend.from_pytorch(pytorch_model, shape_list)

              mod["main"] = bind_params_by_name(mod["main"], params)

              # Some optimizations
              mod = relay.transform.FoldConstant()(mod)

              if os.environ.get('MLPERF_TVM_USE_DNNL','').strip().lower() == 'yes':
                 from tvm.relay.op.contrib.dnnl import partition_for_dnnl
                 from tvm.driver.tvmc.common import convert_graph_layout

                 #  move to NHWC layout, prerequisite for DNNL partitioning
                 mod = convert_graph_layout(mod, "NHWC")
                 mod = relay.transform.FoldConstant()(mod)

                 mod = partition_for_dnnl(mod)

           elif model_path.endswith('.onnx'):
              import onnx

              shape_dict = eval('{' + input_shapes + '}')

              print ('TVM shape dict: '+str(shape_dict))

              onnx_model = onnx.load(model_path)

              mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

              # Some optimizations
              mod = relay.transform.DynamicToStatic()(mod)
              #mod = relay.transform.FoldExplicitPadding()(mod)

              if os.environ.get('MLPERF_TVM_TRANSFORM_LAYOUT','').strip().lower() == 'yes':
                 kernel_layout='NHWC'

                 desired_layouts = {
                     'qnn.conv2d': [kernel_layout, 'default'],
                     'nn.conv2d': [kernel_layout, 'default'],
                     'nn.conv2d_transpose': [kernel_layout, 'default'],
                     'nn.depthwise_conv2d': [kernel_layout, 'default'],
                     'nn.conv3d': [kernel_layout, 'default'],
                     'nn.conv3d_transpose': [kernel_layout, 'default'],
                 }

                 seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                                 relay.transform.FoldConstant(),
                                                 relay.transform.ConvertLayout(desired_layouts),
                                                 ])

                 with tvm.transform.PassContext(opt_level=3):
                     mod = seq(mod)

           elif model_path.endswith('.tflite'):
              # Grigori used https://tvm.apache.org/docs/tutorials/frontend/deploy_prequantized_tflite.html

              import tflite

              shape_dict = eval('{' + input_shapes + '}')

              print ('TVM shape dict: '+str(shape_dict))

              tflite_model_buf = open(model_path, "rb").read()
              tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

              mod, params = relay.frontend.from_tflite(tflite_model, shape_dict)

           else:
              print ('')
              raise Exception("Error: model extension is not supported in TVM backend ({})!".format(model_path))

           # Build model
           # TBD! Apply autotuning history!
           opt_lvl = int(os.environ.get('MLPERF_TVM_OPT_LEVEL', 3))

           target = os.environ.get('MLPERF_TVM_TARGET', 'llvm')

           target_host=None

           # New target API
           tvm_target = tvm.target.Target(target, host=target_host)

           # Check if apply history
           tvm_history_json_file = os.environ.get('MLPERF_TVM_APPLY_HISTORY','').strip()
           if tvm_history_json_file!='':
              if not os.path.isfile(tvm_history_json_file):
                 print ('')
                 raise Exception("Error: TVM history file {} not found!".format(tvm_history_json_file))

              build_conf['relay.backend.use_auto_scheduler']=True

              with auto_scheduler.ApplyHistoryBest(tvm_history_json_file):
                 with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                    self.lib=relay.build(mod, target=tvm_target, params=params)
           else:
              with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                 self.lib=relay.build(mod, target=tvm_target, params=params)

           self.lib.export_library(compiled_model)

           print ('TVM compiled model: '+compiled_model)

        # Init graph
        self.graph = graph_executor.GraphModule(self.lib['default'](ctx))

        # TODO(@apekov): Check if provided inputs/outputs match with presented in model
        # TODO(@apekov): Is there function to get names of inputs/outputs? meanwhile fill it with fake names
        if not inputs:
            inputs = [str(idx) for idx in range(self.graph.get_num_outputs())]
        if not outputs:
            outputs = [str(idx) for idx in range(self.graph.get_num_outputs())]

        # Check executors. Need vm/vm-stateful for SSD object detection models
        self.executor_type = os.environ.get('MLPERF_TVM_EXECUTOR', 'graph')

        if self.executor_type in ("graph", "debug"):
            pass
        elif self.executor_type in ("vm", "vm-stateful"):
            raise Exception("VM mode is UNSUPPORTED ...")

        self.inputs = inputs
        self.outputs = outputs

    def predict_impl(self, feed):
        if self.executor_type in ("vm", "vm-stateful"):
            raise Exception("VM mode is UNSUPPORTED ...")
        else:
            max_batch_size = self.max_batchsize
            batch_size = max_batch_size
            for iname, data in feed.items():
                batch_size = len(data)
                if batch_size < max_batch_size:
                    # Fill in with the first tensor
                    data_extra = np.stack([data[0]] * (max_batch_size-batch_size))
                    data = np.vstack((data, data_extra))
                elif batch_size > max_batch_size:
                    raise ValueError("Internal MLPerf error: dynamic batch size > max batch size")

                input_idx = self.inputs.index(iname)
                self.graph.set_input(input_idx, tvm.nd.array(data))

            # Run TVM inference
            self.graph.run()

            # Process TVM outputs
            tvm_output = []
            output_order = range(self.graph.get_num_outputs()) if self.output_order == None else self.output_order
            for i in output_order:
                # Take only the output of batch size for dynamic batches
                tvm_output.append(self.graph.get_output(i).asnumpy()[:batch_size])

        return tvm_output

    @staticmethod
    def _worker_initializer(model_path, inputs, outputs, max_batchsize, omp_envs):
        BackendTVM.set_omp_envs(omp_envs)
        global g_graph
        g_graph = BackendTVM()
        g_graph.arena_num = 1
        g_graph.load_impl(model_path, inputs, outputs, max_batchsize)

    @staticmethod
    def _worker_handler(feed):
        global g_graph
        return g_graph.predict(feed)

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        self.load_impl(model_path, inputs, outputs, self.max_batchsize)

        if self.arena_num > 1:
            self.pool = multiprocessing.Pool(self.arena_num,
                                             initializer=self._worker_initializer,
                                             initargs=(model_path, inputs, outputs, self.max_batchsize,
                                                       self.create_omp_args(0))
                                             )

        # TODO(@apeskov): do we really have to return self ??
        return self

    def predict(self, feed):
        """Run the prediction."""
        if self.arena_num > 1:
            resp = self.pool.apply_async(self._worker_handler, args=(feed,))
            return resp.get()
        else:
            return self.predict_impl(feed)
