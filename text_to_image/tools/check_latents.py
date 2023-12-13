import torch
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-input", type=str, default="latents.pt")
    parser.add_argument("--numpy-input", type=str, default="latents.npy")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    torch_latents = torch.load(args.torch_input)
    numpy_latents = torch.Tensor(np.load(args.numpy_input))
    print(f"Torch Latents: {torch_latents}\nShape: {torch_latents.shape}")
    print(f"Numpy Latents: {numpy_latents}\nShape: {numpy_latents.shape}")
    assert torch_latents.shape == numpy_latents.shape
    assert (numpy_latents == torch_latents).all().item()
    print("All tests passed!")
