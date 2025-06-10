#!/usr/bin/env python3
from eval_accuracy import process_dataframe, print_evaluation_results, process_and_save_dataframe, process_mlperf_log_accuracy
from utils import (
    validate_runner_for_backend, uses_text_input, uses_chat_template,
    load_dataset, save_results, print_runner_header, StandardTokenizer,
    get_backend_instance, create_base_argument_parser,
    setup_output_paths, validate_runner_args, handle_runner_error,
    validate_dataset_extended, generate_timestamped_filename
)
from mlperf import (
    OfflineSUT, ServerSUT, BaseSUT,
    QuerySampleLibrary,
    prepare_mlperf_dataset,
    process_mlperf_results,
    create_mlperf_output_dataframe
)
from backends import BaseBackend
import pandas as pd
import numpy as np
import mlperf_loadgen as lg
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for MLPerf runner."""
    parser = create_base_argument_parser(
        "Run MLPerf inference benchmarks with modular backends (async pattern)"
    )

    # Scenario selection (no backend argument, auto-detected)
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "server"],
                        help="MLPerf scenario mode")

    # MLPerf configuration
    parser.add_argument("--mlperf-conf", type=str, default="/inference/mlperf.conf",
                        help="Path to MLPerf configuration file")

    parser.add_argument("--user-conf", type=str, default="mlperf/user.conf",
                        help="Path to user configuration file")

    parser.add_argument("--scenario", type=str, default=None,
                        choices=["Offline", "Server"],
                        help="MLPerf scenario (overrides --mode)")

    parser.add_argument("--accuracy", action="store_true",
                        help="Run accuracy mode instead of performance")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="mlperf_results",
                        help="Directory for MLPerf output logs")

    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for detailed logs")

    return parser


def configure_loadgen(scenario: str,
                      accuracy_mode: bool,
                      mlperf_conf: Optional[str] = None,
                      user_conf: Optional[str] = None,
                      log_dir: Optional[str] = None,
                      model_name: str = "deepseek-r1") -> lg.TestSettings:
    """Configure LoadGen test settings.

    Args:
        scenario: MLPerf scenario ("offline" or "server")
        accuracy_mode: Whether to run in accuracy mode
        mlperf_conf: Path to MLPerf config file
        user_conf: Path to user config file
        log_dir: Directory for logs
        model_name: Model name for configuration (default: deepseek-r1)

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
    if mlperf_conf and Path(mlperf_conf).exists():
        settings.FromConfig(mlperf_conf, model_name, scenario, 2)
    if user_conf and Path(user_conf).exists():
        settings.FromConfig(user_conf, model_name, scenario, 1)

    return settings


def run_loadgen_test(sut: Union[OfflineSUT, ServerSUT],
                     qsl: QuerySampleLibrary,
                     settings: lg.TestSettings,
                     log_settings: lg.LogSettings) -> None:
    """Run LoadGen test.

    Args:
        sut: System Under Test instance
        qsl: Query Sample Library
        settings: Test settings
        log_settings: Log settings
    """
    # Start the test
    logger.info("Starting LoadGen test with async generation pattern")
    lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings)
    logger.info("LoadGen test completed")


def main():
    """Main function."""
    import gc
    import torch

    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Validate arguments
        validate_runner_args(args, 'mlperf')

        # Detect backend early
        backend_name = validate_runner_for_backend('mlperf')

        # Handle scenario override
        if args.scenario:
            args.mode = args.scenario.lower()

        # Create output directories
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.log_dir:
            log_dir = Path(args.log_dir)
        else:
            log_dir = output_dir / args.mode / \
                ("accuracy" if args.accuracy else "performance")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Set up output paths with mode information
        _, output_file_base = setup_output_paths(args, mode=args.mode)
        if args.output_file is None:
            # Create output file path in the log directory
            mode_str = "accuracy" if args.accuracy else "performance"
            output_file_base = str(
                log_dir / f"{backend_name}_mlperf_{args.mode}_{mode_str}_output.pkl")
        else:
            output_file_base = args.output_file

        # Generate the actual filename with timestamp that will be used for
        # saving
        actual_output_file = generate_timestamped_filename(
            output_file_base, add_timestamp=True)

        # Ensure the parent directory of the output file exists
        output_file_parent = Path(actual_output_file).parent
        output_file_parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Ensured output file directory exists: {output_file_parent}")

        logger.info("=" * 80)
        logger.info("MLPerf Inference Benchmark Runner (Async Pattern)")
        logger.info("=" * 80)
        logger.info(f"Backend: {backend_name}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Accuracy: {args.accuracy}")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Output file: {actual_output_file}")
        logger.info(f"Uses text prompts: {uses_text_input()}")
        logger.info("=" * 80)

        # Initialize tokenizer
        tokenizer = StandardTokenizer()

        # Determine whether to use chat template based on registry
        use_chat_template = uses_chat_template()

        # Prepare dataset using new utility
        logger.info("Preparing MLPerf dataset...")
        dataset_info = prepare_mlperf_dataset(
            args.input_file,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )

        # Extract components
        df = dataset_info['dataframe']
        prompts = dataset_info['prompts']
        tokenized_prompts = dataset_info['tokenized_prompts']
        processed_strings = dataset_info['processed_strings']
        uses_text_prompts = dataset_info['uses_text_prompts']

        logger.info(f"Loaded {len(prompts)} prompts from dataset")

        # For backends that use text prompts, we pass the processed strings
        # For tokenized backends, we pass the tokenized prompts
        if uses_text_prompts:
            logger.info(
                f"Backend {backend_name} will use text prompts directly")
            dataset_for_sut = tokenized_prompts
            strings_for_sut = processed_strings
        else:
            logger.info(f"Backend {backend_name} will use tokenized prompts")
            dataset_for_sut = tokenized_prompts
            strings_for_sut = processed_strings  # This is what gets used for generation now

        # Create backend using registry
        logger.info(f"Initializing {backend_name} backend...")
        backend = get_backend_instance(backend_name)

        # Use backend context manager to ensure initialization and cleanup
        with backend:
            # Create SUT with proper data
            if args.mode == "offline":
                sut = OfflineSUT(
                    backend=backend,
                    dataset=dataset_for_sut,
                    dataset_strings=strings_for_sut,
                    name=f"{backend_name}_offline_sut"
                )
            else:  # server
                sut = ServerSUT(
                    backend=backend,
                    dataset=dataset_for_sut,
                    dataset_strings=strings_for_sut,
                    name=f"{backend_name}_server_sut"
                )

            # Create QSL
            qsl = QuerySampleLibrary(dataset_for_sut, strings_for_sut)

            # Configure LoadGen
            settings = configure_loadgen(
                scenario=args.mode,
                accuracy_mode=args.accuracy,
                mlperf_conf=args.mlperf_conf,
                user_conf=args.user_conf,
                log_dir=str(log_dir)
            )

            # Update settings with dataset info
            # TODO(vir): these should be in mlperf.conf
            settings.max_query_count = len(tokenized_prompts)
            settings.min_query_count = len(tokenized_prompts)
            settings.use_token_latencies = True
            settings.server_coalesce_queries = True

            # Configure logging
            log_settings = lg.LogSettings()
            log_settings.log_output.outdir = str(log_dir)
            log_settings.log_output.copy_summary_to_stdout = True
            log_settings.enable_trace = False

            # Start the SUT
            logger.info("Starting SUT with async generation support...")
            sut.start()

            try:
                # Run test
                logger.info("Running LoadGen test with async backend...")
                run_loadgen_test(sut, qsl, settings, log_settings)
                logger.info("LoadGen test completed successfully")

                # Ensure all queries are flushed and async operations complete
                logger.info("Flushing queries to ensure completion...")
                sut.flush_queries()

                # Get results BEFORE stopping the SUT
                logger.info("Retrieving results from SUT...")
                sut_results = sut.get_results()
                logger.info(f"Retrieved {len(sut_results)} results from SUT")
            finally:
                # Stop the SUT
                logger.info("Stopping SUT...")
                sut.stop()

                # Explicitly destroy QSL to ensure proper cleanup order
                if qsl.qsl is not None:
                    logger.info("Destroying QSL...")
                    lg.DestroyQSL(qsl.qsl)
                    qsl.qsl = None

            logger.info(f"MLPerf results saved to: {log_dir}")

            # Save results to pickle file (always, regardless of mode)
            logger.info("Processing results for output file...")

            # Initialize df_output to None for safety
            df_output = None

            try:
                # Get results from SUT - must have valid results
                if not sut_results:
                    raise RuntimeError(
                        "No results available from SUT - backend failed to generate tokens")

                # Process results using new utility
                processed_results = process_mlperf_results(
                    sut_results, tokenizer
                )

                # Create output dataframe using new utility
                df_output = create_mlperf_output_dataframe(
                    df, processed_results, backend_name
                )

                # Save results
                # FIXME(vir): output pickle is empty so dont save, okay since accuracy run anyways uses mlperf_log_accuracy.json
                # saved_file = save_results(df_output, output_file_base, add_timestamp=True)
                # logger.info(f"Results saved to: {saved_file}")
                # logger.info(f"Output columns: {list(df_output.columns)}")

            except Exception as e:
                logger.error(f"Error processing results: {e}")
                raise

            # If in accuracy mode, run evaluation
            if args.accuracy and df_output is not None:
                logger.info("=" * 80)
                logger.info("Running accuracy evaluation...")
                logger.info("=" * 80)

                # Check for MLPerf log accuracy file first
                mlperf_log_file = log_dir / "mlperf_log_accuracy.json"

                if mlperf_log_file.exists():
                    logger.info(
                        f"Found MLPerf log accuracy file: {mlperf_log_file}")
                    logger.info("Using MLPerf log for accuracy evaluation...")

                    # Get checkpoint path from backend configuration
                    backend_config = get_backend_instance(backend_name).config

                    # Determine checkpoint path based on backend type
                    if hasattr(get_backend_instance(
                            backend_name), 'model_path'):
                        # PyTorch backend has model_path
                        checkpoint_path = str(
                            get_backend_instance(backend_name).model_path)
                    elif 'model' in backend_config:
                        # Other backends use model name directly
                        checkpoint_path = backend_config['model']
                    elif 'model_name' in backend_config:
                        # PyTorch backend config
                        checkpoint_path = backend_config['model_name']
                    else:
                        # Fallback to default
                        checkpoint_path = "deepseek-ai/DeepSeek-R1"

                    # Process MLPerf log accuracy
                    df_evaluated, evaluated_file = process_mlperf_log_accuracy(
                        mlperf_log_file=mlperf_log_file,
                        dataset_file=args.input_file,
                        checkpoint_path=checkpoint_path,
                        output_dir=log_dir,
                        base_filename="mlperf_accuracy_evaluated.pkl"
                    )

                    logger.info(
                        f"MLPerf accuracy evaluation saved to: {evaluated_file}")
                else:
                    logger.info(
                        "No MLPerf log accuracy file found, using standard DataFrame evaluation...")
                    raise RuntimeError(
                        "No MLPerf log accuracy file found, using standard DataFrame evaluation...")

        # Ensure clean exit
        gc.collect()

        # Ensure all CUDA operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        handle_runner_error(e, "run_mlperf.py")


if __name__ == "__main__":
    main()
