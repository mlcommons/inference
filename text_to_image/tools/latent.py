import argparse
import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size of the latent"
    )
    parser.add_argument(
        "--num-channels-latents", type=int, default=4, help="Batch size of the latent"
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="Type of the latent",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to make the latent deterministic",
    )
    parser.add_argument(
        "--vae-scale-factor",
        type=int,
        default=8,
        help="Variational Autoencoder scale factor, obtainded from model inspection",
    )
    parser.add_argument("--output-type", type=str, default="pt", choices=["pt", "np"])
    args = parser.parse_args()
    return args


def create_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    generator,
    vae_scale_factor,
) -> torch.Tensor:
    shape = (
        batch_size,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, dtype=dtype)
    return latents


if __name__ == "__main__":
    args = get_args()
    batch_size = args.batch_size
    num_channels_latents = args.num_channels_latents
    height = args.height
    width = args.width
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Dtype: {args.dtype} is not supported")
    seed = args.seed
    generator = torch.Generator().manual_seed(seed)
    vae_scale_factor = args.vae_scale_factor
    if args.output_type == "pt":
        save_path = "latents.pt"
    elif args.output_type == "np":
        save_path = "latents.npy"
    else:
        raise ValueError(f"Output Type: {args.output_type} is not supported")
    latents = create_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator,
        vae_scale_factor,
    )
    if args.output_type == "pt":
        torch.save(latents, save_path)
    elif args.output_type == "np":
        np.save(save_path, latents.detach().cpu().numpy())
    else:
        raise ValueError(f"Output Type: {args.output_type} is not supported")
