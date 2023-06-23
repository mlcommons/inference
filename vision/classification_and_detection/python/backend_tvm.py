"""
TVM backend for MLPerf inference vision benchmark
Developers: Alexander Peskov, Thierry Moreau, Grigori Fursin
"""

import backend

import tvm
from tvm import auto_scheduler
from tvm.contrib import graph_executor
from tvm.runtime import vm as runtime_vm

import numpy as np

import os
import multiprocessing

global_executor = None


class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()
        self.arena_num = 1
        self.arena_size = multiprocessing.cpu_count()
        self.executor_type = None
        self.max_batchsize = None
        self.pool = None

    def version(self):
        return tvm.__version__

    def name(self):
        """Name of the runtime."""
        return "tvm"

    def image_format(self):
        """Requested image_format. Use a more popular layout NCHW"""
        return "NCHW"

    def create_omp_args(self, arena_idx):
        idx_start = self.arena_size * arena_idx
        cur_arena_size = min(multiprocessing.cpu_count() -
                             idx_start, self.arena_size)
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

        self.max_batchsize = max_batchsize

        work_dir = os.path.dirname(model_path)
        compiled_model = os.path.join(work_dir, 'model-tvm.so')

        if compiled_model.endswith('.so') or compiled_model.endswith('.dylib'):
            if not os.path.isfile(compiled_model):
                print()
                raise RuntimeError(
                    f"Error: Model file {compiled_model} not found!"
                )
        else:
            raise RuntimeError(
                f"Error: The specified path ({model_path}) does not match path to the compiled model!"
            )

        print('TVM: loading model ' + compiled_model)

        mod = tvm.runtime.load_module(compiled_model)
        device = tvm.device("llvm", 0)

        if os.path.isfile(os.path.join(work_dir, "vm_exec_code.ro")):
            self.executor_type = "virtual_machine"

            with open(os.path.join(work_dir, "vm_exec_code.ro"), "rb") as file:
                vm_bytes = file.read()

            vm_exec = tvm.runtime.vm.Executable.load_exec(vm_bytes, mod)

            for sub_dir in next(os.walk(work_dir))[1]:
                if sub_dir.endswith("-tvm-tmp"):
                    path_consts = os.path.join(
                        work_dir, sub_dir + "/consts")
                    break

            vm_exec.mod["load_late_bound_consts"](path_consts)

            self.executor = runtime_vm.VirtualMachine(vm_exec, device)
        else:
            self.executor_type = "graph_executor"
            self.executor = graph_executor.GraphModule(
                mod['default'](device))

        if not inputs:
            if self.executor_type == "virtual_machine":
                inputs = [str(idx) for idx in range(
                    self.executor.module["get_num_outputs"]())]
            else:
                inputs = [str(idx) for idx in range(
                    self.executor.get_num_outputs())]
        if not outputs:
            if self.executor_type == "virtual_machine":
                outputs = [str(idx) for idx in range(
                    self.executor.module["get_num_outputs"]())]
            else:
                outputs = [str(idx) for idx in range(
                    self.executor.get_num_outputs())]

        self.inputs = inputs
        self.outputs = outputs

    def predict_impl(self, feed):
        iname, item = list(feed.items())[0]
        batch_size = len(item)
        if batch_size < self.max_batchsize:
            # Fill in with the first tensor
            item_extra = np.stack(
                [item[0]] * (self.max_batchsize - batch_size))
            item = np.vstack((item, item_extra))
        elif batch_size > self.max_batchsize:
            raise ValueError(
                "Internal MLPerf error: dynamic batch size > max batch size")
        input_idx = self.inputs.index(iname)
        if self.executor_type == "virtual_machine":
            self.executor.set_input(
                "main", **{"input_tensor:0": tvm.nd.array(item)})
            result = self.executor.run()

            return [result[0].asnumpy()[:batch_size], result[1].asnumpy()[:batch_size]]
        else:
            self.executor.set_input(input_idx, tvm.nd.array(item))
            self.executor.run()
            return [self.executor.get_output(0).asnumpy()[:batch_size],
                    self.executor.get_output(1).asnumpy()[:batch_size]]

    @staticmethod
    def _worker_initializer(model_path, inputs, outputs, max_batchsize, omp_envs):
        BackendTVM.set_omp_envs(omp_envs)
        global global_executor
        global_executor = BackendTVM()
        global_executor.arena_num = 1
        global_executor.load_impl(model_path, inputs, outputs, max_batchsize)

    @staticmethod
    def _worker_handler(feed):
        global global_executor
        return global_executor.predict(feed)

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        self.load_impl(model_path, inputs, outputs, self.max_batchsize)

        if self.arena_num > 1:
            multiprocessing.set_start_method(os.getenv("PYTHON_MP_START_METHOD", "fork"))
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
