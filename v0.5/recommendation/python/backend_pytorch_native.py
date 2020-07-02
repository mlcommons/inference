"""
pytoch native backend for dlrm
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend
from dlrm_s_pytorch import DLRM_Net
import numpy as np

class BackendPytorchNative(backend.Backend):
    def __init__(self, m_spa, ln_emb, ln_bot, ln_top, use_gpu=False, mini_batch_size=1):
        super(BackendPytorchNative, self).__init__()
        self.sess = None
        self.model = None

        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.ln_bot = ln_bot
        self.ln_top = ln_top

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.use_gpu else "cpu"

        ngpus = torch.cuda.device_count() if self.use_gpu else -1
        self.ndevices = min(ngpus, mini_batch_size, ln_emb.size)
        if self.use_gpu:
            print("Using {} GPU(s)...".format(ngpus))
        else:
            print("Using CPU...")

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def load(self, model_path, inputs=None, outputs=None):
        # debug prints
        # print(model_path, inputs, outputs)

        dlrm = DLRM_Net(
            self.m_spa,
            self.ln_emb,
            self.ln_bot,
            self.ln_top,
            arch_interaction_op="dot",
            arch_interaction_itself=False,
            sigmoid_bot=-1,
            sigmoid_top=self.ln_top.size - 2,
            sync_dense_params=True,
            loss_threshold=0.0,
            ndevices=self.ndevices,
            qr_flag=False,
            qr_operation=None,
            qr_collisions=None,
            qr_threshold=None,
            md_flag=False,
            md_threshold=None,
        )
        if self.use_gpu:
            dlrm = dlrm.to(self.device)  # .cuda()
            if dlrm.ndevices > 1:
                dlrm.emb_l = dlrm.create_emb(self.m_spa, self.ln_emb)

        if self.use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(model_path)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    model_path,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(model_path, map_location=torch.device('cpu'))
        # debug print
        # print(ld_model)
        dlrm.load_state_dict(ld_model["state_dict"])
        self.model = dlrm

        # find inputs from the model if not passed in by config
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)

        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)

        # debug print
        # for e in self.ln_emb:
        #     print('Embedding', type(e), e)
        return self

    def predict(self, batch_dense_X, batch_lS_o, batch_lS_i):

        if self.use_gpu:
            batch_dense_X = batch_dense_X.to(self.device)

            batch_lS_i = [S_i.to(self.device) for S_i in batch_lS_i] if isinstance(batch_lS_i, list) \
                else batch_lS_i.to(self.device)


            batch_lS_o = [S_o.to(self.device) for S_o in batch_lS_o] if isinstance(batch_lS_o, list) \
                else batch_lS_o.to(self.device)

        with torch.no_grad():
             output = self.model(dense_x=batch_dense_X, lS_o=batch_lS_o, lS_i=batch_lS_i)
        return output
