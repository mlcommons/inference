"""
pytoch native backend 
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend
from models.resnext50.retinanet import retinanet_resnext50_32x4d_fpn


class BackendPytorchNative(backend.Backend):
    def __init__(self, model_name):
        super(BackendPytorchNative, self).__init__()
        self.sess = None
        self.model = None
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native"

    def image_format(self):
        return "NCHW"

    def build_model_architecture(self):
        # Build the appropiate model architecture for each model name.
        # This is only needed when loading a model from a checkpoint file.
        # Currently only ssd-resnext50 is supported
        if self.model_name == "ssd-resnext50":
            return retinanet_resnext50_32x4d_fpn(
                num_classes=91, pretrained=None, image_size=[800, 800]
            )
        return None

    def load(self, model_path, inputs=None, outputs=None):
        cached = torch.load(model_path, map_location=lambda storage, loc: storage)
        if isinstance(cached, dict):
            # If the file located at model_path contains a checkpoint, it is
            # necessary to build the model architecture and then load the
            # weights of the model
            checkpoint = cached
            self.model = self.build_model_architecture()
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model = cached
        self.model.eval()
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
        self.model = self.model.to(self.device)
        return self

    def predict(self, feed):
        key = [key for key in feed.keys()][0]
        feed[key] = torch.tensor(feed[key]).float().to(self.device)
        with torch.no_grad():
            output = self.model(feed[key])
        return output
