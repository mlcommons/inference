"""
pytoch native backend for dlrm
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend
from dlrm_s_pytorch import DLRM_Net

class BackendPytorchNativeDLRM(backend.Backend):
    def __init__(self, m_spa, ln_emb, ln_bot, ln_top):
        super(BackendPytorchNativeDLRM, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.ln_bot = ln_bot
        self.ln_top = ln_top

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def load(self, model_path, inputs=None, outputs=None):
        #debug prints
        print(model_path, inputs, outputs)
        #print("Loading saved model {}".format(args.load_model))

        self.model = DLRM_Net(
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
            ndevices=-1, #ndevices,
            qr_flag=False,
            qr_operation=None,
            qr_collisions=None,
            qr_threshold=None,
            md_flag=False,
            md_threshold=None,
        )
        ld_model = torch.load(model_path, map_location=torch.device('cpu')) #use self.device
        self.model.load_state_dict(ld_model["state_dict"])

        ###self.model = torch.load(model_path,map_location=lambda storage, loc: storage)
        ###self.model.eval()                                                                                                                                                                                                                                 
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

        # prepare the backend
        ###self.model = self.model.to(self.device)
        return self


    def predict(self, feed):
        key=[key for key in feed.keys()][0]
        feed[key] = torch.tensor(feed[key]).float().to(self.device)
        with torch.no_grad():
            output = self.model(feed[key])
        return output
