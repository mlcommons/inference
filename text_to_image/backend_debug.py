import torch
import backend


class BackendDebug(backend.Backend):
    def __init__(self, image_size=[3, 1024, 1024], **kwargs):
        super(BackendDebug, self).__init__()
        self.image_size = image_size

    def version(self):
        return torch.__version__

    def name(self):
        return "debug-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        return self

    def predict(self, prompts):
        images = []
        with torch.no_grad():
            for prompt in prompts:
                image = torch.randn(self.image_size)
                images.append(image)
        return images
