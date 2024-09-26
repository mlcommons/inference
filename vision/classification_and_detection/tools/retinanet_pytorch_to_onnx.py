from model.retinanet import retinanet_from_backbone
import argparse
import numpy as np
import torch
import torchvision


def get_args():
    """
    Args used for converting PyTorch/TorchScript to ONNX model
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--weights",
        default="model_10.pth",
        help="Path to the PyTorch model weights",
    )
    parser.add_argument(
        "--output",
        default="resnext50_32x4d_fpn.onnx",
        help="Output file of the model",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Declare general model params
    args = get_args()
    backbone = "resnext50_32x4d"
    num_classes = 264
    image_size = [800, 800]

    model = retinanet_from_backbone(backbone, num_classes, image_size=image_size)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sample_input = torch.randn(1, 3, 800, 800)
    torch.onnx.export(
        model,
        sample_input,
        args.output,
        export_params=True,
        opset_version=13,
        output_names=["boxes", "scores", "labels"],
    )
