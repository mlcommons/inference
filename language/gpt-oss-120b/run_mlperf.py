#!/usr/bin/env python3
"""MLPerf inference benchmark runner for gpt-oss.

This script integrates the gpt-oss model with MLPerf LoadGen for
performance and accuracy benchmarking.

Usage:
    # Offline scenario (performance)
    python run_mlperf.py --scenario offline --input-file data/accuracy_eval_tokenized.pkl

    # Server scenario (performance)
    python run_mlperf.py --scenario server --input-file data/accuracy_eval_tokenized.pkl

    # Accuracy mode
    python run_mlperf.py --scenario offline --accuracy --input-file data/accuracy_eval_tokenized.pkl
"""

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import mlperf_loadgen as lg
import pandas as pd
from tqdm import tqdm

from backends import SGLangBackend
from mlperf import OfflineSUT, ServerSUT, QuerySampleLibrary
from utils import load_tokenized_dataset, StandardTokenizer

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_generation_config(config_path: str) -> Dict[str, Any]:
    """Load generation configuration from JSON file.

    Args:
        config_path: Path to generation_config.json

    Returns:
        Dictionary with generation parameters
    """
    logger.info(f"Loading generation config from {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Filter out comment fields (starting with _)
    gen_params = {k: v for k, v in config.items() if not k.startswith('_')}

    return gen_params


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for MLPerf runner."""
    parser = argparse.ArgumentParser(
        description="Run MLPerf inference benchmarks for gpt-oss"
    )

    # Scenario selection
    parser.add_argument(
        "--scenario",
        type=str,
        default="offline",
        choices=["offline", "server"],
        help="MLPerf scenario (offline or server)"
    )

    # Dataset
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to tokenized dataset (parquet or pickle file)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None for all)"
    )

    # MLPerf configuration
    parser.add_argument(
        "--mlperf-conf",
        type=str,
        default="inference/mlperf.conf",
        help="Path to MLPerf configuration file"
    )

    parser.add_argument(
        "--user-conf",
        type=str,
        default="mlperf/user.conf",
        help="Path to user configuration file"
    )

    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy mode instead of performance"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mlperf_results",
        help="Directory for MLPerf output logs"
    )

    # Backend configuration
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        choices=["sglang"],
        help="Backend to use for inference"
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:30000",
        help="Server URL for backend (SGLang)"
    )

    # Generation configuration
    parser.add_argument(
        "--generation-config",
        type=str,
        default="generation_config.json",
        help="Path to generation configuration JSON file"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens from generation config (default: use value from config)"
    )

    # Server scenario specific
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads (for server scenario)"
    )

    # Concurrency control
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=128,
        help="Maximum concurrent requests to backend (SGLang handles batching internally)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Timeout for HTTP requests in seconds (default: 1200)"
    )

    return parser


def configure_loadgen(
    scenario: str,
    accuracy_mode: bool,
    mlperf_conf: Optional[str] = None,
    user_conf: Optional[str] = None,
    log_dir: Optional[str] = None,
    model_name: str = "gpt-oss-120b"
) -> lg.TestSettings:
    """Configure LoadGen test settings.

    Args:
        scenario: MLPerf scenario ("offline" or "server")
        accuracy_mode: Whether to run in accuracy mode
        mlperf_conf: Path to MLPerf config file
        user_conf: Path to user config file
        log_dir: Directory for logs
        model_name: Model name for configuration

    Returns:
        LoadGen TestSettings
    """
    settings = lg.TestSettings()

    # Set scenario
    if scenario.lower() == "offline":
        settings.scenario = lg.TestScenario.Offline
    elif scenario.lower() == "server":
        settings.scenario = lg.TestScenario.Server
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Set mode
    if accuracy_mode:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    # Load configurations if files exist
    # conf_type: 2 = mlperf.conf, 1 = user.conf
    # LoadGen tracks config calls and only allows one user.conf for official
    # submissions
    if mlperf_conf and Path(mlperf_conf).exists():
        logger.debug(f"Loading MLPerf config from {mlperf_conf}")
        settings.FromConfig(mlperf_conf, model_name, scenario.capitalize(), 2)
    else:
        logger.warning(f"MLPerf config not found: {mlperf_conf}")

    if user_conf and Path(user_conf).exists():
        logger.debug(f"Loading user config from {user_conf}")
        settings.FromConfig(user_conf, model_name, scenario.capitalize(), 1)
    else:
        logger.warning(f"User config not found: {user_conf}")

    return settings


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Track resources for cleanup
    sut = None
    qsl = None
    backend = None
    pbar = None
    cleanup_done = False

    def do_cleanup():
        """Perform cleanup once and only once."""
        nonlocal cleanup_done, pbar, sut, qsl, backend

        if cleanup_done:
            return
        cleanup_done = True

        logger.info("Performing cleanup...")

        # 1. Close progress bar first (before any LoadGen cleanup)
        try:
            if pbar is not None:
                pbar.close()
                pbar = None
                logger.debug("  ✓ Progress bar closed")
        except Exception as e:
            logger.debug(f"  ! Error closing progress bar: {e}")

        # Small delay to let LoadGen internal threads finish
        import time
        time.sleep(0.5)

        # 2. Stop SUT (this will stop worker threads and flush)
        try:
            if sut is not None:
                logger.info("  - Stopping SUT and worker threads...")
                sut.stop()
                sut = None
                logger.info("    ✓ SUT stopped")
        except Exception as e:
            logger.warning(f"    ! Error stopping SUT: {e}")

        # 3. Destroy QSL
        try:
            if qsl is not None and qsl.qsl is not None:
                logger.info("  - Destroying Query Sample Library...")
                lg.DestroyQSL(qsl.qsl)
                qsl.qsl = None
                logger.info("    ✓ QSL destroyed")
        except Exception as e:
            logger.warning(f"    ! Error destroying QSL: {e}")

        # 4. Cleanup backend last
        try:
            if backend is not None and backend.initialized:
                logger.info("  - Cleaning up backend connection...")
                backend.cleanup()
                backend = None
                logger.info("    ✓ Backend cleaned up")
        except Exception as e:
            logger.warning(f"    ! Error cleaning up backend: {e}")

    try:
        # Create output directories
        output_dir = Path(args.output_dir)
        log_dir = output_dir / args.scenario / \
            ("accuracy" if args.accuracy else "performance")
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("MLPerf Inference Benchmark Runner for GPT-OSS")
        logger.info("=" * 80)
        logger.info(f"Backend: {args.backend}")
        logger.info(f"Scenario: {args.scenario}")
        logger.info(f"Accuracy: {args.accuracy}")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output directory: {log_dir}")
        logger.info("=" * 80)

        # Load dataset
        logger.debug("Loading tokenized dataset...")
        with tqdm(total=1, desc="Loading dataset", unit="file") as pbar:
            dataset_info = load_tokenized_dataset(
                args.input_file,
                max_samples=args.max_samples
            )
            prompts = dataset_info["prompts"]
            df = dataset_info["dataframe"]
            pbar.update(1)

        logger.info(f"Loaded {len(prompts)} prompts from dataset")

        # Load generation configuration
        logger.info("Loading generation configuration...")
        gen_config = load_generation_config(args.generation_config)

        # Extract generation parameters with defaults
        # CLI override takes precedence over config file
        if args.max_new_tokens is not None:
            max_tokens = args.max_new_tokens
            logger.info(
                f"Using max_new_tokens from CLI override: {max_tokens}")
        else:
            max_tokens = gen_config.get('max_new_tokens', 10240)
            logger.info(f"Using max_new_tokens from config: {max_tokens}")

        temperature = gen_config.get('temperature', 1.0)
        top_k = gen_config.get('top_k', -1)
        top_p = gen_config.get('top_p', 1.0)

        logger.info("Generation parameters:")
        logger.info(f"  max_new_tokens: {max_tokens}")
        logger.info(f"  temperature: {temperature}")
        logger.info(f"  top_k: {top_k}")
        logger.info(f"  top_p: {top_p}")

        # Initialize backend
        logger.debug(f"Initializing {args.backend} backend...")
        if args.backend == "sglang":
            # Set pool size to match max_concurrency with small safety margin
            # This prevents "connection pool is full" warnings
            pool_size = int(args.max_concurrency * 1.1)  # 10% safety margin
            backend = SGLangBackend(
                server_url=args.server_url,
                timeout=args.timeout,
                max_pool_size=pool_size
            )
        else:
            raise ValueError(f"Unknown backend: {args.backend}")

        # Initialize backend
        backend.initialize()

        # Create progress bar early so subsequent logs print below it
        # Total will be dynamically updated by SUT based on actual queries from LoadGen:
        # - Offline: Set once when all queries arrive
        # - Server: Incremented as queries arrive
        pbar = tqdm(
            total=0,  # Will be updated dynamically by SUT
            desc=f"MLPerf {args.scenario}",
            unit="query",
            leave=True,
            position=0,
            mininterval=0.1,
            smoothing=0.1,
            dynamic_ncols=True,
            file=sys.stdout  # Force unbuffered output for async updates
        )

        # Create SUT with progress bar
        logger.debug(f"Creating {args.scenario} SUT...")
        if args.scenario == "offline":
            sut = OfflineSUT(
                backend=backend,
                dataset=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                name=f"gpt-oss-120b_offline_sut",
                progress_bar=pbar,
                max_concurrency=args.max_concurrency
            )
        else:  # server
            sut = ServerSUT(
                backend=backend,
                dataset=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_workers=args.num_workers,
                name=f"gpt-oss-120b_server_sut",
                progress_bar=pbar
            )

        # Create QSL
        logger.info("Creating Query Sample Library...")
        qsl = QuerySampleLibrary(prompts)
        qsl.qsl = lg.ConstructQSL(
            len(prompts),
            len(prompts),
            qsl.load_query_samples,
            qsl.unload_query_samples
        )

        # Configure LoadGen
        settings = configure_loadgen(
            scenario=args.scenario,
            accuracy_mode=args.accuracy,
            mlperf_conf=args.mlperf_conf,
            user_conf=args.user_conf,
            log_dir=str(log_dir)
        )

        # Configure logging
        log_settings = lg.LogSettings()
        log_settings.log_output.outdir = str(log_dir)
        log_settings.log_output.copy_summary_to_stdout = True
        log_settings.enable_trace = False

        # Start the SUT and run test
        logger.info("Running LoadGen test...")
        sut.start()
        lg.StartTestWithLogSettings(
            sut.sut,
            qsl.qsl,
            settings,
            log_settings
        )
        logger.info("LoadGen test completed successfully")

        # Give LoadGen a moment to finish internal cleanup
        import time
        time.sleep(0.2)

        # Flush queries
        logger.info("Flushing queries...")
        with tqdm(total=1, desc="Flushing queries", unit="batch") as pbar:
            sut.flush_queries()
            pbar.update(1)

        # Get results
        logger.info("Retrieving results...")
        with tqdm(total=1, desc="Getting results", unit="batch") as pbar:
            results = sut.get_results()
            pbar.update(1)
        logger.info(f"Retrieved {len(results)} results from SUT")

        logger.info(f"MLPerf results saved to: {log_dir}")

        # If in accuracy mode, prompt user to run evaluation
        if args.accuracy:
            logger.info("=" * 80)
            logger.info("Accuracy mode completed!")
            logger.info("To evaluate accuracy, run:")
            logger.info(
                f"  python eval_accuracy.py --input-file {log_dir}/mlperf_log_accuracy.json")
            logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("⚠️  Test interrupted by user (Ctrl+C)")
        logger.info("=" * 80)
        do_cleanup()
        logger.info("=" * 80)
        logger.info("✓ Cleanup completed successfully")
        logger.info("=" * 80)
        # Exit immediately to prevent finally block from running
        os._exit(130)  # Use os._exit to skip finally block

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error(f"❌ Error during test: {e}")
        logger.error("=" * 80)
        logger.error("Stack trace:", exc_info=True)
        do_cleanup()
        logger.error("=" * 80)
        # Exit immediately to prevent finally block from running
        os._exit(1)

    finally:
        # Only run cleanup if not already done (normal exit path)
        if not cleanup_done:
            do_cleanup()
            logger.info("=" * 80)
            logger.info("✓ Cleanup completed successfully")
            logger.info("=" * 80)


if __name__ == "__main__":
    main()
