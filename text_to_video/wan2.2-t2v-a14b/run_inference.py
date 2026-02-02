#!/usr/bin/env python3
"""
Batch inference script for Wan2.2 T2V-A14B-Diffusers model.
Processes all prompts from dataset while keeping model loaded.
Supports multi-GPU inference with data parallelism (prompts divided among GPUs).
"""

from diffusers.utils import export_to_video
from diffusers import WanPipeline, AutoencoderKLWan
import torch
import yaml
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


# import modelopt.torch.opt as mto


def setup_logging(rank):
    """Setup logging configuration for data parallel (all ranks log)."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(dataset_path):
    """Load prompts from dataset file."""
    with open(dataset_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def generate_videos(args, config):
    """Main generation function with data parallelism."""

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    setup_logging(rank)

    # Generation parameters from config
    height = config['height']
    width = config['width']
    num_frames = config['num_frames']
    fps = config['fps']
    guidance_scale = config['guidance_scale']
    guidance_scale_2 = config['guidance_scale_2']
    boundary_ratio = config['boundary_ratio']
    negative_prompt = config['negative_prompt'].strip()
    sample_steps = config['sample_steps']
    base_seed = config['seed']

    if rank == 0:
        logging.info(f"Model: Wan2.2 T2V-A14B-Diffusers")
        logging.info(f"Model path: {args.model_path}")
        logging.info(f"World size: {world_size}")
        logging.info(f"Sample steps: {sample_steps}")
        logging.info(f"Base seed: {base_seed}")
        logging.info(f"Iterations per prompt: {args.num_iterations}")

    all_prompts = load_prompts(args.dataset)

    if rank == 0:
        logging.info(f"Loaded {len(all_prompts)} prompts from {args.dataset}")

    if args.num_prompts > 0:
        all_prompts = all_prompts[:args.num_prompts]
        if rank == 0:
            logging.info(f"Processing first {args.num_prompts} prompts")

    # Divide prompts among GPUs (data parallelism)
    prompts = all_prompts[rank::world_size]
    logging.info(
        f"This rank will process {len(prompts)} prompts (indices: {rank}, {rank + world_size}, ...)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading Diffusers pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        boundary_ratio=boundary_ratio,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    logging.info("Model loaded successfully!")

    # if args.run_quantized:
    #     logging.info("Loading quantized model...")
    #     transformer_pt = os.path.join(args.quantized_model_path, "transformer.pt")
    #     transformer_2_pt = os.path.join(args.quantized_model_path, "transformer_2.pt")

    #     assert os.path.exists(transformer_pt)
    #     assert os.path.exists(transformer_2_pt)
    #     mto.restore(pipe.transformer, transformer_pt)
    #     mto.restore(pipe.transformer_2, transformer_2_pt)

    #     logging.info("Quantized model loaded successfully!")

    fixed_latent = None
    if args.fixed_latent:
        fixed_latent = torch.load(args.fixed_latent)
        logging.info(
            f"Loaded fixed latent from {args.fixed_latent} with shape: {fixed_latent.shape}")
        logging.info(f"This latent will be reused for all generations")
    else:
        logging.info("No fixed latent provided - using random initial latents")

    if rank == 0:
        logging.info(
            f"Starting batch generation: {len(all_prompts)} total prompts x {args.num_iterations} iterations")
        logging.info(f"Each GPU processes ~{len(prompts)} prompts")

    # Generate videos: iterate through all prompts, then repeat for next
    # iteration
    total_videos = 0
    for iteration in range(args.num_iterations):
        if rank == 0:
            logging.info(f"\n{'='*60}")
            logging.info(f"ITERATION {iteration + 1}/{args.num_iterations}")
            logging.info(f"{'='*60}")

        for local_idx, prompt in enumerate(prompts):
            # Calculate global prompt index
            global_idx = rank + local_idx * world_size

            logging.info(
                f"[Prompt {global_idx+1}/{len(all_prompts)}, Iteration {iteration+1}/{args.num_iterations}] {prompt}")

            # Check if video already exists
            filename = f"{prompt}-{iteration}.mp4"
            save_path = output_dir / filename

            if save_path.exists():
                logging.info(
                    f"Video already exists at {save_path}, skipping generation")
                total_videos += 1
                continue

            # Generate video with seed based on iteration
            current_seed = base_seed + iteration

            # Prepare pipeline arguments
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "guidance_scale_2": guidance_scale_2,
                "num_inference_steps": sample_steps,
                "generator": torch.Generator(device=device).manual_seed(current_seed),
            }

            # Only pass latents if fixed_latent is provided
            if fixed_latent is not None:
                pipeline_kwargs["latents"] = fixed_latent

            output = pipe(**pipeline_kwargs).frames[0]

            # Save video with VBench format: <prompt>-<iteration>.mp4
            logging.info(f"Saving to {save_path} (seed: {current_seed})")
            export_to_video(output, str(save_path), fps=fps)
            total_videos += 1
            logging.info(
                f"Saved! ({total_videos}/{len(prompts) * args.num_iterations} for this GPU)")

            torch.cuda.empty_cache()

    logging.info(f"\n{'='*60}")
    logging.info(f"Batch generation complete for this GPU!")
    logging.info(f"Generated {total_videos} videos in {output_dir}")
    logging.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch T2V inference with Wan2.2-Diffusers")

    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/Wan2.2-T2V-A14B-Diffusers",
        help="Path to model checkpoint directory (default: ./models/Wan2.2-T2V-A14B-Diffusers)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/vbench_prompts.txt",
        help="Path to dataset file (text prompts, one per line) (default: ./data/prompts.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/outputs",
        help="Directory to save generated videos (default: ./data/outputs)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./inference_config.yaml",
        help="Path to inference configuration file (default: ./inference_config.yaml)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of generation iterations per prompt (default: 1)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=-1,
        help="Process only first N prompts (for testing, default: all)"
    )
    parser.add_argument(
        "--fixed-latent",
        type=str,
        default="./data/fixed_latent.pt",
        help="Path to fixed latent .pt file for deterministic generation (default: data/fixed_latent.pt)"
    )
    # parser.add_argument(
    #     "--run-quantized",
    #     action="store_true",
    #     help="Run quantized model"
    # )
    # parser.add_argument(
    #     "--quantized-model-path",
    #     type=str,
    #     default="./models/Wan2.2-T2V-FP8-Torch",
    #     help="Path to quantized model (default: ./models/Wan2.2-T2V-FP8-Torch)"
    # )

    args = parser.parse_args()

    config = load_config(args.config)

    generate_videos(args, config)


if __name__ == "__main__":
    main()
