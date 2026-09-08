"""
PyTorch TorchScript backend for MLPerf inference.

Loads TorchScript models (.pt) exported via torch.jit.trace or torch.jit.script.
Unlike backend_pytorch_native which expects raw state dicts, this backend works
directly with serialized TorchScript modules.
"""

# pylint: disable=unused-argument,missing-docstring
import torch
import backend


class BackendPytorchTorchScript(backend.Backend):
    def __init__(self):
        super(BackendPytorchTorchScript, self).__init__()
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-torchscript"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.inputs = inputs or ["image"]
        self.outputs = outputs or ["output"]
        return self

    def predict(self, feed):
        key = [key for key in feed.keys()][0]
        feed[key] = torch.tensor(feed[key]).float().to(self.device)
        with torch.no_grad():
            output = self.model(feed[key])
        if isinstance(output, torch.Tensor):
            return [output.cpu().numpy()]
        # handle tuple/list outputs
        return [o.cpu().numpy() for o in output]
