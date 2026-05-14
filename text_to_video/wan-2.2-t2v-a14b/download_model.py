#!/usr/bin/env python3
"""
Download Wan T2V 14B model from HuggingFace with progress tracking.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed")
    print("Please install it with: pip install huggingface-hub")
    sys.exit(1)


def download_model(download_path: str,
                   model_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"):
    """
    Download Wan T2V model from HuggingFace.

    Args:
        download_path: Directory to download the model
        model_name: HuggingFace model identifier
    """
    download_path = Path(download_path).resolve()
    # Extract model name without org prefix (e.g.,
    # "Wan-AI/Wan2.2-T2V-A14B-Diffusers" -> "Wan2.2-T2V-A14B-Diffusers")
    model_dir_name = model_name.split("/")[-1]
    model_path = download_path / model_dir_name

    # Create download directory
    download_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"{model_name} Model Download")
    print("=" * 60)
    print(f"Download path: {model_path}")
    print("=" * 60)
    print()

    try:
        print("Starting download...")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print()
        print("=" * 60)
        print("? Download completed successfully!")
        print("=" * 60)
        print(f"Model location: {model_path}")

    except Exception as e:
        print()
        print("=" * 60)
        print("? Download failed!")
        print("=" * 60)
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Wan2.2 T2V-A14B-Diffusers model from HuggingFace",
    )

    parser.add_argument(
        "-d", "--download-path",
        default=os.environ.get("DOWNLOAD_PATH", "./models"),
        help="Download directory (default: ./models or $DOWNLOAD_PATH)"
    )

    parser.add_argument(
        "--model-name",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model identifier (default: Wan-AI/Wan2.2-T2V-A14B-Diffusers)"
    )

    args = parser.parse_args()

    download_model(args.download_path, args.model_name)


if __name__ == "__main__":
    main()
