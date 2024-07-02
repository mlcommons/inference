import torch
import torchvision.models as models


def main(args):
    model_path = str(args[0])
    output_path = str(args[1])

    net = models.resnet50()
    net.load_state_dict(torch.load(model_path))
    torch.save(net, output_path)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
