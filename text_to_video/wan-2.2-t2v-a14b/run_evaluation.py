#!/usr/bin/env python3
"""
VBench evaluation script for generated videos.
Evaluates videos across multiple quality dimensions using VBench.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_results(output_path):
    """
    Parse VBench evaluation results and print summary.

    Args:
        output_path: Path to evaluation results directory
    """
    output_path = Path(output_path)

    # Find the most recent eval_results file (contains scores)
    result_files = sorted(output_path.glob("results_*_eval_results.json"))
    if not result_files:
        logging.warning(f"No results found in {output_path}")
        return

    result_file = result_files[-1]

    try:
        with open(result_file, 'r') as f:
            results = json.load(f)

        # Print summary in MLPerf-style format
        print("\n" + "=" * 60)
        print("VBench Evaluation Results")
        print("=" * 60)

        # Extract dimension scores (VBench format: {dimension_name: [avg_score,
        # [video_results]], ...})
        if results:
            print("\nDimension Scores:")
            print("-" * 60)
            total_score = 0
            num_dimensions = 0

            for dimension, value in sorted(results.items()):
                # VBench stores [avg_score, list_of_video_results]
                if isinstance(value, list) and len(value) > 0:
                    score = value[0]  # First element is the average
                    if isinstance(score, (int, float)):
                        total_score += score
                        num_dimensions += 1
                        print(f"  {dimension:30s}: {score:6.4f}")

            if num_dimensions > 0:
                overall_avg = total_score / num_dimensions
                print("-" * 60)
                print(f"  {'Overall Average':30s}: {overall_avg:6.4f}")

        print("=" * 60)
        print(f"Detailed results: {result_file}")
        print("=" * 60 + "\n")

    except Exception as e:
        logging.error(f"Failed to parse results: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="VBench evaluation for Wan2.2 T2V videos")
    parser.add_argument(
        "--videos-path",
        type=str,
        default="./data/outputs",
        help="Path to directory containing generated videos (default: ./data/outputs)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/evaluation_results",
        help="Path to save evaluation results (default: ./data/evaluation_results)"
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=[
            "subject_consistency",
            "dynamic_degree",
            "motion_smoothness",
            "appearance_style",
            "scene",
            "background_consistency"
        ],
        help="Evaluation dimensions (default: 6 core dimensions)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for evaluation (default: 8)"
    )

    args = parser.parse_args()

    setup_logging()

    # Validate inputs
    videos_path = Path(args.videos_path)
    if not videos_path.exists():
        logging.error(f"Videos path does not exist: {videos_path}")
        return 1

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("VBench Evaluation")
    logging.info("=" * 60)
    logging.info(f"Videos path: {videos_path}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"GPUs: {args.num_gpus}")
    logging.info(f"Dimensions: {', '.join(args.dimensions)}")
    logging.info("=" * 60)

    vbench_script = Path(__file__).parent / "submodules" / \
        "VBench" / "evaluate.py"
    cmd = [
        "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={args.num_gpus}",
        str(vbench_script),
        f"--videos_path={videos_path}",
        f"--output_path={output_path}",
        "--load_ckpt_from_local=True",
        "--dimension"
    ] + args.dimensions

    logging.info("\nExecuting VBench evaluation...")
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info("")

    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True)

        # Parse and print results
        logging.info("\nParsing evaluation results...")
        parse_results(output_path)

        return 0

    except subprocess.CalledProcessError as e:
        logging.error(f"Evaluation failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        logging.warning("Evaluation interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
