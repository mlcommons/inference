#!/usr/bin/env python3
from eval_accuracy import process_dataframe, print_evaluation_results, process_and_save_dataframe, process_mlperf_log_accuracy
from utils.data_utils import (
    load_dataset, save_results,
    generate_timestamped_filename
)
from utils.validation import (
    validate_runner_args, ValidationError,
    validate_dataset_extended
)
from utils.backend_registry import (
    uses_chat_template, get_backend_instance, detect_backend,
    validate_runner_for_backend
)
from utils.runner_utils import create_base_argument_parser, print_runner_header
from utils import (
    StandardTokenizer,
    validate_dataset,
    process_inference_results
)
from mlperf import (
    OfflineSUT, ServerSUT, BaseSUT,
    DistributedQuerySampleLibrary,
    prepare_mlperf_dataset,
    process_mlperf_results,
    create_mlperf_output_dataframe
)
from backends.pytorch_backend import PyTorchBackend
from transformers import AutoTokenizer
import torch.distributed as dist
import torch
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
import builtins
import time

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configure logging - only for rank 0
def setup_logging(rank: int):
    """Setup logging based on rank."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Disable logging for non-rank 0 processes
        logging.disable(logging.CRITICAL)

    return logging.getLogger(__name__)


class DistributedOfflineSUT(BaseSUT):
    """Distributed Offline SUT implementation for PyTorch backend.

    Only rank 0 interacts with LoadGen, but all ranks participate in inference.
    """

    def __init__(self,
                 backend: 'PyTorchBackend',
                 dataset: List[List[int]],
                 prompt_strings: Optional[List[str]] = None,
                 name: str = "DistributedOfflineSUT",
                 rank: int = 0,
                 world_size: int = 1):
        """Initialize the distributed offline SUT.

        Args:
            backend: Backend instance to use for inference
            dataset: List of tokenized prompts
            prompt_strings: Optional list of prompt strings
            name: Name of the SUT
            rank: Process rank
            world_size: Total number of processes
        """
        super().__init__(name)
        self.backend = backend
        self.dataset = dataset
        self.prompt_strings = prompt_strings
        self.rank = rank
        self.world_size = world_size

        # Results storage (only rank 0)
        if self.rank == 0:
            self.results = {}
            self.index_to_id = {}

        # Flag to signal other ranks to exit
        self._should_exit = False

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries for processing.

        Only called on rank 0 by LoadGen.

        Args:
            query_samples: List of MLPerf LoadGen query samples
        """
        if self.rank != 0:
            return

        logger = logging.getLogger(__name__)
        logger.info(f"Issuing {len(query_samples)} queries")

        # Process queries in batches
        batch_size = self.backend.config['batch_size']

        for i in range(0, len(query_samples), batch_size):
            batch_samples = query_samples[i:i + batch_size]

            # Prepare batch tokens
            batch_tokens = []
            batch_ids = []

            for sample in batch_samples:
                # Track index to ID mapping
                self.index_to_id[sample.index] = sample.id

                # Get tokens for this sample
                tokens = self.dataset[sample.index]
                batch_tokens.append(tokens)
                batch_ids.append(sample.id)

            # Signal other ranks to participate in generation
            if self.world_size > 1:
                signal = ["generate"]
                dist.broadcast_object_list(signal, src=0)

            # Generate using distributed backend
            # This will broadcast to all ranks internally
            generated_tokens = self.backend.generate_batch_distributed(
                batch_tokens)

            # Process results and send to LoadGen
            for j, (sample_id, tokens) in enumerate(
                    zip(batch_ids, generated_tokens)):
                # Create a copy of tokens before numpy conversion
                tokens_copy = tokens.copy()

                # Convert tokens to bytes for LoadGen
                token_array = np.array(tokens, dtype=np.int32)
                n_tokens = len(tokens)

                # Create LoadGen response
                response = lg.QuerySampleResponse(
                    sample_id,
                    token_array.ctypes.data,
                    token_array.nbytes,
                    n_tokens,
                )

                # Store result with the tokens copy
                self.results[sample_id] = {
                    'tokens': tokens_copy,
                }

                # Send response to LoadGen
                lg.QuerySamplesComplete([response])

            # Send idle signal to other ranks after batch completes
            if self.world_size > 1:
                idle_signal = [None]
                dist.broadcast_object_list(idle_signal, src=0)

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Nothing to flush in this implementation
        pass

    def start(self) -> lg.ConstructSUT:
        """Start the SUT."""
        # Signal that we're starting
        if self.rank == 0:
            logger = logging.getLogger(__name__)
            logger.info("Starting Distributed Offline SUT")

        return super().start()

    def stop(self) -> None:
        """Stop the SUT."""
        # Signal other ranks to exit is now handled in the main loop
        # after LoadGen test completes

        super().stop()

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results in order of dataset indices.

        Returns:
            List of result dictionaries with model_output, tok_model_output, and tok_model_output_len
        """
        if self.rank != 0:
            return []

        # Create a list to hold results in dataset order
        ordered_results = []

        # Process results in order of dataset indices
        for i in range(len(self.dataset)):
            # Get the sample ID for this index
            sample_id = self.index_to_id.get(i)

            if sample_id is not None and sample_id in self.results:
                result = self.results[sample_id]
                if 'tokens' in result and result['tokens']:
                    tokens = result['tokens']

                    # Decode tokens to get text output
                    output_text = ''
                    if self.backend.tokenizer:
                        output_text = self.backend.tokenizer.decode(
                            tokens, skip_special_tokens=True)

                    ordered_results.append({
                        'model_output': output_text,
                        'tok_model_output': tokens,
                        'tok_model_output_len': len(tokens)
                    })
                else:
                    # Result exists but no tokens - this is an error
                    raise RuntimeError(
                        f"No tokens in result for dataset index {i}, sample_id {sample_id}")
            else:
                # No result for this index - this is an error
                raise RuntimeError(
                    f"No result for dataset index {i}, sample_id {sample_id}")

        return ordered_results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for distributed MLPerf runner."""
    parser = argparse.ArgumentParser(
        description="Run MLPerf inference benchmarks with distributed PyTorch backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument("--input-file", type=str,
                        default="data/final_output.pkl",
                        help="Input pickle file with prompts")

    # MLPerf configuration
    parser.add_argument("--mlperf-conf", type=str, default="/inference/mlperf.conf",
                        help="Path to MLPerf configuration file")

    parser.add_argument("--user-conf", type=str, default="mlperf/user.conf",
                        help="Path to user configuration file")

    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "server"],
                        help="MLPerf scenario mode (only offline supported for distributed)")

    parser.add_argument("--accuracy", action="store_true",
                        help="Run accuracy mode instead of performance")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="mlperf_results",
                        help="Directory for MLPerf output logs")

    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for detailed logs")

    parser.add_argument("--output-file", type=str, default=None,
                        help="Output pickle file path (auto-generated if not specified)")

    # Note: --no-chat-template is removed (chat template usage determined by
    # backend registry)

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


def run_loadgen_test(sut: DistributedOfflineSUT,
                     qsl: DistributedQuerySampleLibrary,
                     settings: lg.TestSettings,
                     log_settings: lg.LogSettings,
                     rank: int,
                     logger) -> None:
    """Run LoadGen test (only on rank 0).

    Args:
        sut: System Under Test instance
        qsl: Query Sample Library
        settings: Test settings
        log_settings: Log settings
        rank: Process rank
        logger: Logger instance
    """
    if rank == 0:
        # Start the test
        logger.info("Starting LoadGen test")
        lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings)
        logger.info("LoadGen test completed")


def main():
    """Main function."""
    _print = builtins.print  # Capture the original built-in print

    # Get distributed environment info
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Override print for non-rank 0 processes
    if rank != 0:
        print = lambda *_, **__: None

    # Setup logging
    logger = setup_logging(rank)

    # Detect backend from environment
    backend_name = detect_backend()

    # Validate backend
    validate_runner_for_backend('mlperf_mpi')

    # Get chat template usage from registry
    use_chat_template = uses_chat_template()

    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    if rank == 0:
        try:
            validate_runner_args(args, 'mlperf_mpi')
        except ValidationError as e:
            _print(f"Argument validation error: {e}")
            sys.exit(1)

    # Validate mode for distributed
    if args.mode != "offline":
        if rank == 0:
            logger.error(
                "Only offline mode is supported for distributed execution")
        sys.exit(1)

    # Create output directories (only rank 0)
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.log_dir:
            log_dir = Path(args.log_dir)
        else:
            log_dir = output_dir / args.mode / \
                ("accuracy" if args.accuracy else "performance")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Determine output file path
        if args.output_file:
            output_file_base = args.output_file
        else:
            mode_str = "accuracy" if args.accuracy else "performance"
            output_file_base = str(
                log_dir / f"{backend_name}_mlperf_{args.mode}_{mode_str}_output.pkl")

        # Generate the actual filename with timestamp
        actual_output_file = generate_timestamped_filename(
            output_file_base, add_timestamp=True)

        # Ensure the parent directory of the output file exists
        output_file_parent = Path(actual_output_file).parent
        output_file_parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Ensured output file directory exists: {output_file_parent}")

        logger.info("=" * 80)
        logger.info("MLPerf Inference Benchmark Runner (Distributed PyTorch)")
        logger.info("=" * 80)
        logger.info(f"Backend: {backend_name}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Accuracy: {args.accuracy}")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Output file: {actual_output_file}")
        logger.info(
            f"Chat template: {'enabled' if use_chat_template else 'disabled'} (from registry)")
        logger.info("=" * 80)
    else:
        log_dir = None
        actual_output_file = None

    try:
        # Initialize PyTorch backend
        backend = PyTorchBackend()
        backend.initialize()

        # Initialize tokenizer (only on rank 0)
        tokenizer = None
        if rank == 0:
            tokenizer = StandardTokenizer()

        # Only rank 0 handles dataset loading
        df = None
        tokenized_prompts = []
        processed_strings = []

        if rank == 0:
            # Prepare dataset using new utility
            logger.info("Preparing MLPerf dataset...")
            dataset_info = prepare_mlperf_dataset(
                args.input_file,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template
            )

            # Extract components
            df = dataset_info['dataframe']
            tokenized_prompts = dataset_info['tokenized_prompts']
            processed_strings = dataset_info['processed_strings']

            logger.info(
                f"Loaded {len(tokenized_prompts)} prompts from dataset")

        # Create SUT
        sut = DistributedOfflineSUT(
            backend=backend,
            dataset=tokenized_prompts if rank == 0 else [],
            prompt_strings=processed_strings if rank == 0 else [],
            name=f"{backend_name}_distributed_offline_sut",
            rank=rank,
            world_size=world_size
        )

        # Create QSL (only rank 0 needs the actual QSL)
        qsl = DistributedQuerySampleLibrary(
            tokenized_prompts if rank == 0 else [],
            processed_strings if rank == 0 else [],
            rank,
            world_size
        )

        # Only rank 0 configures and runs LoadGen
        if rank == 0:
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
        sut.start()

        try:
            if rank == 0:
                # Run test (only rank 0)
                logger.info("Running test...")
                run_loadgen_test(
                    sut, qsl, settings, log_settings, rank, logger)
                logger.info("Completed test...")

                # Ensure all queries are flushed and async operations complete
                logger.info("Flushing queries to ensure completion...")
                sut.flush_queries()

                # Send exit signal to other ranks
                if world_size > 1:
                    exit_signal = [True]
                    dist.broadcast_object_list(exit_signal, src=0)
            else:
                # Non-rank 0 processes participate in distributed generation
                # They wait for signals from rank 0 and participate in
                # generate_batch_distributed
                while True:
                    # First, check if we should exit
                    # We use a separate broadcast to signal exit
                    exit_check = [None]
                    dist.broadcast_object_list(exit_check, src=0)

                    if exit_check[0] is True:
                        # Exit signal received
                        break
                    elif exit_check[0] == "generate":
                        # Signal to participate in generation
                        # The actual batch tokens will be broadcast inside
                        # generate_batch_distributed
                        backend.generate_batch_distributed(None)
                    # If exit_check[0] is None, continue waiting
        finally:
            # Stop the SUT
            sut.stop()

            if rank == 0:
                logger.info(f"Results saved to: {log_dir}")

                # Print summary
                summary_file = log_dir / "mlperf_log_summary.txt"
                if summary_file.exists():
                    logger.info("\nTest Summary:")
                    logger.info("-" * 40)
                    with open(summary_file, 'r') as f:
                        _print(f.read())

                # Save results to pickle file (always, regardless of mode)
                if df is not None:
                    logger.info("Processing results for output file...")

                    # Initialize df_output to None for safety
                    df_output = None

                    try:
                        # Get results from SUT (if available)
                        logger.info(
                            "Retrieving results from distributed SUT...")
                        sut_results = sut.get_results()
                        logger.info(
                            f"Retrieved {len(sut_results)} results from distributed SUT")

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
                            logger.info(
                                "Using MLPerf log for accuracy evaluation...")

                            # For PyTorch backend (only one supported in MPI),
                            # get model path
                            checkpoint_path = str(
                                backend.model_path) if hasattr(
                                backend,
                                'model_path') else backend.config.get(
                                'model_name',
                                'deepseek-ai/DeepSeek-R1')

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

    except KeyboardInterrupt:
        if rank == 0:
            logger.info("Test interrupted by user")
        backend.shutdown()
        if world_size > 1:
            dist.destroy_process_group()
        sys.exit(1)
    except Exception as e:
        if rank == 0:
            logger.error(f"Test failed: {e}", exc_info=True)
        backend.shutdown()
        if world_size > 1:
            dist.destroy_process_group()
        sys.exit(1)

    # Clean up
    backend.shutdown()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
