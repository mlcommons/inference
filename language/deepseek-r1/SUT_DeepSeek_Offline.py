# ============================================================================
# This file was generated and refactored with the help of AI (OpenAI GPT-4),
# with additional modifications and review by the author: <Naveen Miriyalu nmiriyal@redhat.com>
#
# Disclaimer: This code is provided as-is, without warranty of any kind.
# Please review and test before using in production or submitting to MLPerf.
# ============================================================================
"""
SUT_VLLM_SingleReplica.py
-------------------------
Harness for running vLLM models with MLPerf Loadgen in both offline and server scenarios.
Supports local vLLM, vLLM API, and async server batching with multi-worker support.

This module provides three main SUT (System Under Test) implementations:
1. VLLMSingleSUT - Local vLLM model execution
2. VLLMSingleSUTAPI - Remote vLLM API server communication  
3. VLLMSingleSUTServer - Server scenario with async batching
"""

import os
import time
import logging
import numpy as np
from typing import List
from dataset import Dataset
from vllm import TokensPrompt
import sys
import torch
import pkg_resources
from datetime import datetime
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector
from vllm.utils import FlexibleArgumentParser
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import csv
import array
from vllm import AsyncLLMEngine, AsyncEngineArgs
import asyncio
from collections import defaultdict
from random import shuffle


# Import vLLM components with error handling
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed.")
    print("Please install it using: pip install vllm")
    exit(1)

# Import MLPerf Loadgen with error handling
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    exit(1)

# Import NVTX for profiling (optional)
try:
    import nvtx
except ImportError:
    nvtx = None


# ============================================================================
# MLPerf Loadgen Required Functions
# ============================================================================

def load_samples_to_ram(query_samples):
    """Required by MLPerf Loadgen - samples are pre-loaded in Dataset class"""
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    """Required by MLPerf Loadgen - no action needed for our implementation"""
    del query_samples
    return


# ============================================================================
# Main SUT Classes
# ============================================================================

class VLLMSingleSUT:
    """
    Local vLLM SUT Implementation
    
    This class implements the MLPerf SUT interface for local vLLM model execution.
    It handles batch processing, profiling, and various optimization options.
    Uses per-instance logger for proper logging behavior.
    """
    
    def __init__(self, model_name: str, dataset_path: str, test_mode: str = "performance", 
                 enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", 
                 enable_nvtx: bool = False, print_histogram: bool = False, 
                 sort_by_length: bool = False, sort_by_token_contents: bool = False, 
                 print_sorted_tokens: bool = False, print_timing: bool = False, 
                 **engine_args):
        
        # Initialize per-instance logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configuration parameters
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.test_mode = test_mode
        self.engine_args = engine_args
        
        # Performance and debugging options
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir
        self.enable_nvtx = enable_nvtx
        self.print_histogram = print_histogram
        self.sort_by_length = sort_by_length
        self.sort_by_token_contents = sort_by_token_contents
        self.print_sorted_tokens = print_sorted_tokens
        self.print_timing = print_timing
        
        # Runtime state
        self.profiler = None
        self.batch_counter = 0
        self._padded_tokens = {}  # For bucketing and padding
        
        # Load dataset and display statistics
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=4388)
        self.logger.info("Dataset loaded: %d samples", len(self.data_object.input_ids))
        self.logger.info("Dataset statistics - Max Input Tokens: %d, Min Input Tokens: %d, Total Samples: %d", 
                        max(self.data_object.input_lens), 
                        min(self.data_object.input_lens),
                        len(self.data_object.input_lens))
        
        # Initialize the model
        self._load_model()

    def _load_model(self):
        """Load the vLLM model with specified configuration"""
        # Start NVTX range for model loading if enabled
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push("loadmodel")
            
        self.logger.info(f"Loading model '{self.model_name}' with engine args: {self.engine_args}")
        
        # Create LLM instance with engine arguments
        #self.llm = LLM(model=self.model_name, **self.engine_args)
        self.llm = LLM(**self.engine_args)
        
        self.logger.info("Model loaded successfully.")
        
        # Configure sampling parameters for generation
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic generation
            max_tokens=128,   # Maximum output tokens
            min_tokens=1,     # Minimum output tokens
            top_p=1,         # Nucleus sampling parameter
            top_k=1,          # Top-k sampling parameter
            seed=42
        )
        
        # Display model configuration for debugging
        print("-" * 60)
        print("vLLM MODEL CONFIGURATION")
        print("-" * 60)
        try:
            engine_instance = self.llm.llm_engine 
            print("vLLM Config:", engine_instance.vllm_config)
            print("Model Config:", engine_instance.vllm_config.model_config)
            print("Cache Config:", engine_instance.vllm_config.cache_config)
        except Exception as e:
            print(f"Error accessing model configuration: {e}")
        print("-" * 60)

        # End NVTX range for model loading
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()

    def issue_query(self, query_samples: List['lg.QuerySample']):
        """
        Process query samples from MLPerf Loadgen
        
        This is the main entry point called by MLPerf Loadgen. It processes
        queries in batches and returns responses via lg.QuerySamplesComplete().
        """
        batch_size = BATCH_SIZE
        total_samples = len(query_samples)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        self.logger.info(f"Received {total_samples} queries from Loadgen")
        self.logger.info(f"Processing in {num_batches} batches of size {batch_size}")
        
        batch_times = []
        
        # Initialize profiler for all batches if enabled
        if self.enable_profiler and self.profiler is None:
            self._setup_profiler(num_batches)
        
        # Process each batch
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, total_samples)
            batch = query_samples[start:end]
            
            # Apply sorting options if requested
            batch = self._apply_batch_sorting(batch)
            #batch = self.bucket_and_pad_without_batching(batch)
            
            # Print debug information if requested
            if self.print_sorted_tokens or self.logger.isEnabledFor(logging.DEBUG):
                self._print_batch_debug_info(batch_idx, batch)
            
            # Prepare batch data - use padded tokens if available, otherwise original
            prompts_to_process = [TokensPrompt(prompt_token_ids=self._padded_tokens.get(q.index, self.data_object.input_ids[q.index])) 
                                for q in batch]
            original_query_ids = [q.id for q in batch]
            original_query_indexes = [q.index for q in batch]
            
            # Print histogram if requested
            if self.print_histogram:
                self._print_batch_histogram(batch)
            
            # Process the batch
            batch_start = time.time() if self.print_timing else None
            try:
                self._process_single_batch(batch_idx, batch, prompts_to_process, 
                                         original_query_ids, original_query_indexes)
                
                # Record timing if enabled
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'batch_size': len(batch)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                self._handle_batch_error(original_query_ids, batch_start, batch_times, batch_idx, len(batch))
        
        # Cleanup and final statistics
        self._cleanup_profiler()
        if self.print_timing and batch_times:
            self._print_timing_statistics(batch_times)

    def _setup_profiler(self, num_batches):
        """Setup PyTorch profiler for performance analysis"""
        os.makedirs(self.profiler_dir, exist_ok=True)
        trace_file = os.path.join(self.profiler_dir, "vllm_generation_trace.json")
        
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=num_batches, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.start()
        self.logger.info(f"Profiler started for {num_batches} batches. Trace: {trace_file}")

    def _apply_batch_sorting(self, batch):
        """Apply sorting options to the batch if requested"""
        if self.sort_by_length:
            batch = sorted(batch, key=lambda q: len(self.data_object.input_ids[q.index]))
        elif self.sort_by_token_contents:
            batch = sorted(batch, key=lambda q: tuple(self.data_object.input_ids[q.index]))
        return batch

    def bucket_and_pad_without_batching(self, query_samples: List['lg.QuerySample'], 
                                       bucket_sizes: List[int] = [128, 256, 384, 512, 1024, 1536, 2048],
                                       shuffle_within_buckets: bool = False) -> List['lg.QuerySample']:
        """
        Buckets and pads query samples based on token length, but does NOT group into batch_size.

        Args:
        query_samples: List of MLPerf QuerySample objects
        bucket_sizes: Length thresholds for bucketing
        shuffle_within_buckets: Whether to shuffle samples inside buckets

        Returns:
        List of QuerySample objects (potentially reordered) with padded tokens stored
        """
        # Get EOS token for padding
        try:
            tokenizer = self.llm.llm_engine.tokenizer
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                pad_token_id = tokenizer.eos_token_id
            else:
                pad_token_id = 2  # Common EOS token fallback
        except:
            pad_token_id = 2  # Fallback
        
        buckets = defaultdict(list)
        
        # Store original token IDs for each query sample
        self._padded_tokens = {}

        # Assign query samples to the smallest bucket that fits
        for q_sample in query_samples:
            original_tokens = self.data_object.input_ids[q_sample.index]
            prompt_len = len(original_tokens)
            
            # Find appropriate bucket
            assigned_bucket = None
            for b in bucket_sizes:
                if prompt_len <= b:
                    assigned_bucket = b
                    break
            
            if assigned_bucket is None:
                assigned_bucket = bucket_sizes[-1]  # Use largest bucket
            
            buckets[assigned_bucket].append((q_sample, original_tokens))

        reordered_samples = []

        self.logger.info("🔍 Bucket counts:")
        for b in sorted(bucket_sizes):
            bucket_data = buckets[b]
            if not bucket_data:
                continue

            if shuffle_within_buckets:
                shuffle(bucket_data)

            self.logger.info(f"  Bucket {b:>4} tokens → {len(bucket_data)} queries")

            # Pad tokens and store mapping
            for q_sample, original_tokens in bucket_data:
                if len(original_tokens) < b:
                    padded_tokens = original_tokens + [pad_token_id] * (b - len(original_tokens))
                else:
                    padded_tokens = original_tokens[:b]  # Truncate if needed
                
                # Store padded tokens for this query
                self._padded_tokens[q_sample.index] = padded_tokens
                reordered_samples.append(q_sample)

        return reordered_samples

    def _print_batch_debug_info(self, batch_idx, batch):
        """Print debug information for the current batch"""
        print(f"Batch {batch_idx} debug info:")
        for i, q in enumerate(batch):
            print(f"  {i:3d}: idx={q.index}, tokens={self.data_object.input_ids[q.index]}")

    def _print_batch_histogram(self, batch):
        """Print histogram of input lengths and query indexes"""
        input_lens = [len(self.data_object.input_ids[q.index]) for q in batch]
        query_indexes = [q.index for q in batch]
        
        def print_hist_int(data, title, width=50, bins=10):
            data = np.array(data, dtype=int)
            min_val, max_val = int(np.min(data)), int(np.max(data))
            if min_val == max_val:
                bins = 1
            else:
                bins = min(bins, max_val - min_val + 1)
            hist, bin_edges = np.histogram(data, bins=bins, range=(min_val, max_val+1))
            max_count = max(hist)
            print(f"Histogram of {title} (integer bins):")
            for i in range(len(hist)):
                left = int(bin_edges[i])
                right = int(bin_edges[i+1]) - 1
                bar = '#' * int(width * hist[i] / max_count) if max_count > 0 else ''
                print(f"  {left:6d} - {right:6d}: {bar} ({hist[i]})")
        
        print_hist_int(input_lens, "input token lengths")
        print_hist_int(query_indexes, "query indexes")
        
        # Check for duplicates
        from collections import Counter
        counter = Counter(sorted(query_indexes))
        duplicates = {k: v for k, v in counter.items() if v > 1}
        if duplicates:
            print("Duplicate query indexes:")
            for idx, freq in duplicates.items():
                print(f"  Query index {idx} repeated {freq} times")
        else:
            print("No duplicate query indexes in this batch.")

    def _process_single_batch(self, batch_idx, batch, prompts_to_process, 
                            original_query_ids, original_query_indexes):
        """Process a single batch through the vLLM model"""
        batch_label = f"batch_{self.batch_counter:04d}_size_{len(batch)}"
        
        # Start NVTX range if enabled
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(batch_label)
        
        # Start model profiling if enabled
        if self.enable_profiler:
            self.llm.start_profile()
        
        # Generate responses using vLLM
        with torch.profiler.record_function(batch_label):
            gen_start = time.time() if self.print_timing else None
            outputs = self.llm.generate(prompts_to_process, self.sampling_params)
            gen_end = time.time() if self.print_timing else None
        
        # Stop model profiling
        if self.enable_profiler:
            self.llm.stop_profile()
        
        # End NVTX range
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()
        
        # Process outputs and prepare responses for Loadgen
        responses_to_loadgen = []
        for i in range(len(outputs)):
            output = outputs[i]
            token_ids = output.outputs[0].token_ids
            token_count = len(token_ids)
            query_id = original_query_ids[i]
            query_index = original_query_indexes[i]
            
            # Log output information
            self.logger.info(f"Query ID: {query_id}, Index: {query_index:5d}, Tokens: {token_count}")
            self.logger.debug(f"Token IDs: {token_ids}")
            self.logger.debug(f"Token text: {output.outputs[0].text}")
            
            #if self.test_mode == "accuracy" or self.test_mode == "performance":
            # For accuracy testing, include actual token data
            token_array = np.array(token_ids, dtype=np.int32)
            token_bytes = token_array.tobytes()
            response_data = token_array.ctypes.data
            response_size = len(token_bytes)
            response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
            lg.QuerySamplesComplete([response])
            
            #if self.test_mode == "performance":
            #    responses_to_loadgen.append(response)
        
        # Send responses to Loadgen
        #if responses_to_loadgen and self.test_mode == "performance":
        #    lg.QuerySamplesComplete(responses_to_loadgen)
        
        self.batch_counter += 1
        
        # Print metrics if timing is enabled
        if self.print_timing:
            self._print_batch_metrics(gen_start, gen_end)

    def _print_batch_metrics(self, gen_start, gen_end):
        """Print detailed metrics for the current batch"""
        if gen_start and gen_end:
            self.logger.info(f"Batch {self.batch_counter} generation time: {gen_end - gen_start:.1f}s")

    def _handle_batch_error(self, original_query_ids, batch_start, batch_times, batch_idx, batch_size):
        """Handle errors during batch processing"""
        # Send error responses to Loadgen
        for query_id in original_query_ids:
            response = lg.QuerySampleResponse(query_id, 0, 0, 0)
            lg.QuerySamplesComplete([response])
        
        self.batch_counter += 1
        
        # Record timing even for failed batches
        if self.print_timing and batch_start:
            batch_end = time.time()
            batch_times.append({
                'batch_idx': batch_idx,
                'start': batch_start,
                'end': batch_end,
                'duration': batch_end - batch_start,
                'batch_size': batch_size,
                'error': True
            })

    def _cleanup_profiler(self):
        """Stop and cleanup the profiler"""
        if self.enable_profiler and self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
            self.logger.info(f"Profiler stopped after {self.batch_counter} batches")

    def _print_timing_statistics(self, batch_times):
        """Print comprehensive timing statistics"""
        durations = np.array([bt['duration'] for bt in batch_times])
        
        print("\n" + "="*60)
        print("BATCH TIMING STATISTICS")
        print("="*60)
        print(f"Total batches: {len(batch_times)}")
        print(f"Duration (s): min={durations.min():.1f}, max={durations.max():.1f}")
        print(f"             mean={durations.mean():.1f}, std={durations.std():.1f}")
        print("\nPer-batch details:")
        for bt in batch_times:
            error_flag = " [ERROR]" if bt.get('error', False) else ""
            print(f"  Batch {bt['batch_idx']:3d}: size={bt['batch_size']:4d}, "
                  f"duration={bt['duration']:.1f}s{error_flag}")
        print("="*60)

    def flush_queries(self):
        """MLPerf Loadgen callback - flush any pending queries"""
        self.logger.info("Flush queries called (no action needed for offline scenario)")


class VLLMSingleSUTAPI:
    """
    vLLM API Server SUT Implementation
    
    This class communicates with a remote vLLM API server instead of running
    the model locally. It handles API communication, tokenization/detokenization,
    and optional metrics collection.
    """
    
    def __init__(self, model_name: str, dataset_path: str, api_server_url: str, 
                 test_mode: str = "performance", 
                 enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", 
                 enable_nvtx: bool = False, print_histogram: bool = False, 
                 sort_by_length: bool = False, sort_by_token_contents: bool = False, 
                 print_sorted_tokens: bool = False, print_timing: bool = False, 
                 enable_metrics_csv: bool = False, metrics_csv_path: str = "metrics.csv"):
        
        # Initialize per-instance logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configuration
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.api_server_url = api_server_url.rstrip('/')
        self.test_mode = test_mode
        
        # Performance and debugging options
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir
        self.enable_nvtx = enable_nvtx
        self.print_histogram = print_histogram
        self.sort_by_length = sort_by_length
        self.sort_by_token_contents = sort_by_token_contents
        self.print_sorted_tokens = print_sorted_tokens
        self.print_timing = print_timing
        
        # Runtime state
        self.batch_counter = 0
        self.server_ready = False
        self._padded_tokens = {}  # For bucketing and padding
        
        # Metrics collection
        self.enable_metrics_csv = enable_metrics_csv
        self.metrics_csv_path = metrics_csv_path
        self.metrics_thread = None
        self.metrics_stop_event = threading.Event()
        
        # API endpoints
        self.completions_endpoint = f"{self.api_server_url}/v1/completions"
        self.health_endpoint = f"{self.api_server_url}/health"
        self.metrics_endpoint = f"{self.api_server_url}/metrics"
        
        # Load dataset
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=4388)
        self.logger.info("API Dataset loaded: %d samples", len(self.data_object.input_ids))
        self.logger.info("Dataset statistics - Max Inp Tokens: %d, Min Input Tokens: %d, Total Samples: %d", 
                        max(self.data_object.input_lens), 
                        min(self.data_object.input_lens),
                        len(self.data_object.input_lens))
        
        # Wait for server readiness and initialize components
        self._wait_for_server_ready()
        self._initialize_tokenizer()
        
        # Start metrics collection if enabled
        if self.enable_metrics_csv:
            self._start_metrics_thread()

    def _wait_for_server_ready(self, timeout: int = 600):
        """Wait for the vLLM API server to become ready"""
        self.logger.info(f"Waiting for API server at {self.api_server_url} (timeout: {timeout}s)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.health_endpoint, timeout=10)
                if response.status_code == 200:
                    self.logger.info("API server is ready!")
                    self.server_ready = True
                    return
                else:
                    self.logger.warning(f"Health check returned status {response.status_code}")
            except Exception as e:
                self.logger.debug(f"API server not ready: {e}")
            
            time.sleep(1)
        
        raise RuntimeError(f"vLLM API server at {self.api_server_url} did not become ready within {timeout} seconds")
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for text encoding/decoding"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.logger.info("Tokenizer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    def _detokenize_response(self, text_response: str) -> List[int]:
        """Convert text response back to token IDs"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text_response, add_special_tokens=False)
                return tokens
            except Exception as e:
                self.logger.warning(f"Error detokenizing response: {e}")
                return [1, 2, 3]  # Fallback placeholder
        else:
            self.logger.warning("No tokenizer available, using placeholder tokens")
            return [1, 2, 3]  # Fallback placeholder
    
    def issue_query(self, query_samples: List['lg.QuerySample']):
        """Process queries by sending them to the vLLM API server"""
        if not self.server_ready:
            self.logger.error("API server is not ready")
            self._send_error_responses(query_samples)
            return
        
        batch_size = BATCH_SIZE
        total_samples = len(query_samples)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        self.logger.info(f"API processing {total_samples} queries in {num_batches} batches")
        batch_times = []
        
        # Process each batch
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, total_samples)
            batch = query_samples[start:end]
            
            # Apply sorting if requested
            batch = self._apply_batch_sorting(batch)
            #batch = self.bucket_and_pad_without_batching(batch)
            
            # Debug information
            if self.print_sorted_tokens or self.logger.isEnabledFor(logging.DEBUG):
                self._print_batch_debug_info(batch_idx, batch)
            
            # Prepare batch data
            original_query_ids = [q.id for q in batch]
            original_query_indexes = [q.index for q in batch]
            
            # Print histogram if requested
            if self.print_histogram:
                self._print_batch_histogram(batch)
            
            # Process the batch via API
            batch_start = time.time() if self.print_timing else None
            try:
                self._process_api_batch(batch_idx, batch, original_query_ids, original_query_indexes)
                
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'batch_size': len(batch)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing API batch {batch_idx}: {e}")
                self._handle_api_batch_error(original_query_ids, batch_start, batch_times, batch_idx, len(batch))
        
        # Print timing statistics
        if self.print_timing and batch_times:
            self._print_api_timing_statistics(batch_times)

    def _send_error_responses(self, query_samples):
        """Send error responses for all queries"""
        for q_sample in query_samples:
            response = lg.QuerySampleResponse(q_sample.id, 0, 0, 0)
            lg.QuerySamplesComplete([response])

    def _apply_batch_sorting(self, batch):
        """Apply sorting options to the batch"""
        if self.sort_by_length:
            batch = sorted(batch, key=lambda q: len(self.data_object.input_ids[q.index]))
        elif self.sort_by_token_contents:
            batch = sorted(batch, key=lambda q: tuple(self.data_object.input_ids[q.index]))
        return batch

    def bucket_and_pad_without_batching(self, query_samples: List['lg.QuerySample'], 
                                       bucket_sizes: List[int] = [128, 256, 384, 512, 1024, 1536, 2048],
                                       shuffle_within_buckets: bool = False) -> List['lg.QuerySample']:
        """
        Buckets and pads query samples based on token length, but does NOT group into batch_size.

        Args:
        query_samples: List of MLPerf QuerySample objects
        bucket_sizes: Length thresholds for bucketing
        shuffle_within_buckets: Whether to shuffle samples inside buckets

        Returns:
        List of QuerySample objects (potentially reordered) with padded tokens stored
        """
        # Get EOS token for padding (simplified for API)
        try:
            if self.tokenizer and hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                pad_token_id = self.tokenizer.eos_token_id
            else:
                pad_token_id = 2  # Common EOS token fallback
        except:
            pad_token_id = 2  # Fallback
        
        buckets = defaultdict(list)
        
        # Store original token IDs for each query sample
        self._padded_tokens = {}

        # Assign query samples to the smallest bucket that fits
        for q_sample in query_samples:
            original_tokens = self.data_object.input_ids[q_sample.index]
            prompt_len = len(original_tokens)
            
            # Find appropriate bucket
            assigned_bucket = None
            for b in bucket_sizes:
                if prompt_len <= b:
                    assigned_bucket = b
                    break
            
            if assigned_bucket is None:
                assigned_bucket = bucket_sizes[-1]  # Use largest bucket
            
            buckets[assigned_bucket].append((q_sample, original_tokens))

        reordered_samples = []

        self.logger.info("🔍 API Bucket counts:")
        for b in sorted(bucket_sizes):
            bucket_data = buckets[b]
            if not bucket_data:
                continue

            if shuffle_within_buckets:
                shuffle(bucket_data)

            self.logger.info(f"  Bucket {b:>4} tokens → {len(bucket_data)} queries")

            # Pad tokens and store mapping
            for q_sample, original_tokens in bucket_data:
                if len(original_tokens) < b:
                    padded_tokens = original_tokens + [pad_token_id] * (b - len(original_tokens))
                else:
                    padded_tokens = original_tokens[:b]  # Truncate if needed
                
                # Store padded tokens for this query
                self._padded_tokens[q_sample.index] = padded_tokens
                reordered_samples.append(q_sample)

        return reordered_samples


    def _print_batch_debug_info(self, batch_idx, batch):
        """Print debug information for API batch"""
        print(f"API Batch {batch_idx} debug info:")
        for i, q in enumerate(batch):
            print(f"  {i:3d}: idx={q.index}, tokens={self.data_object.input_ids[q.index]}")

    def _print_batch_histogram(self, batch):
        """Print histogram information for API batch"""
        input_lens = [len(self.data_object.input_ids[q.index]) for q in batch]
        self.logger.debug(f"Batch input lengths: min={min(input_lens)}, max={max(input_lens)}")

    def _process_api_batch(self, batch_idx, batch, original_query_ids, original_query_indexes):
        """Process a single batch via the API server"""
        batch_label = f"api_batch_{self.batch_counter:04d}_size_{len(batch)}"
        
        # Start NVTX range if enabled
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(batch_label)
        
        with torch.profiler.record_function(batch_label):
            gen_start = time.time() if self.print_timing else None
            
            # Convert token IDs to text prompts
            text_prompts = self._prepare_text_prompts(batch)
            
            # Prepare API request
            api_payload = {
                "model": self.model_name,
                "prompt": text_prompts,
                "max_tokens": 128,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "stream": False
            }
            
            # Send request to API server
            response = requests.post(self.completions_endpoint, json=api_payload)
            if response.status_code != 200:
                raise RuntimeError(f"API server returned status {response.status_code}: {response.text}")
            
            api_result = response.json()
            choices = api_result.get("choices", [])
            
            gen_end = time.time() if self.print_timing else None
        
        # End NVTX range
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()
        
        # Process API responses
        self._process_api_responses(choices, original_query_ids, original_query_indexes)
        
        self.batch_counter += 1
        
        # Log timing if enabled
        if self.print_timing and gen_start and gen_end:
            self.logger.info(f"API batch {batch_idx} processing time: {gen_end - gen_start:.1f}s")

    def _prepare_text_prompts(self, batch):
        """Convert token IDs to text prompts for API"""
        text_prompts = []
        for q_sample in batch:
            # Use original tokens for decoding (not padded ones) for better text quality
            tokens_to_use = self.data_object.input_ids[q_sample.index]
            
            if self.tokenizer:
                try:
                    text_prompt = self.tokenizer.decode(
                        tokens_to_use, 
                        skip_special_tokens=True
                    )
                    text_prompts.append(text_prompt)
                except Exception as e:
                    self.logger.warning(f"Error decoding tokens for query {q_sample.id}: {e}")
                    # Fallback to token ID string
                    text_prompts.append(" ".join([str(t) for t in tokens_to_use]))
            else:
                # No tokenizer available, use token IDs as string
                text_prompts.append(" ".join([str(t) for t in tokens_to_use]))
        return text_prompts

    def _process_api_responses(self, choices, original_query_ids, original_query_indexes):
        """Process API responses and send to Loadgen"""
        responses_to_loadgen = []
        
        for i in range(len(choices)):
            choice = choices[i]
            query_id = original_query_ids[i]
            query_index = original_query_indexes[i]
            
            # Extract text response from API
            text_response = choice.get("text", "")
            
            # Convert back to token IDs
            token_ids = self._detokenize_response(text_response)
            token_count = len(token_ids)
            
            # Debug logging
            self.logger.info(f"API Query ID: {query_id}, Index: {query_index}, Tokens: {token_count}")
            self.logger.debug(f"API Token IDs: {token_ids}")
            self.logger.debug(f"API Text Response: {text_response}")
            
            # Create response based on test mode
            #if self.test_mode == "accuracy":
            token_array = np.array(token_ids, dtype=np.int32)
            token_bytes = token_array.tobytes()
            response_data = token_array.ctypes.data
            response_size = len(token_bytes)
            response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
            lg.QuerySamplesComplete([response])
            #else:
            #    response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
            
            #if self.test_mode == "performance":
            #    responses_to_loadgen.append(response)
        
        # Send responses to Loadgen
        #if responses_to_loadgen and self.test_mode == "performance":
        #    lg.QuerySamplesComplete(responses_to_loadgen)

    def _handle_api_batch_error(self, original_query_ids, batch_start, batch_times, batch_idx, batch_size):
        """Handle errors in API batch processing"""
        for query_id in original_query_ids:
            response = lg.QuerySampleResponse(query_id, 0, 0, 0)
            lg.QuerySamplesComplete([response])
        
        self.batch_counter += 1
        
        if self.print_timing and batch_start:
            batch_end = time.time()
            batch_times.append({
                'batch_idx': batch_idx,
                'start': batch_start,
                'end': batch_end,
                'duration': batch_end - batch_start,
                'batch_size': batch_size,
                'error': True
            })

    def _print_api_timing_statistics(self, batch_times):
        """Print timing statistics for API processing"""
        durations = np.array([bt['duration'] for bt in batch_times])
        
        print("\n" + "="*60)
        print("API BATCH TIMING STATISTICS")
        print("="*60)
        print(f"Total batches: {len(batch_times)}")
        print(f"Duration (s): min={durations.min():.1f}, max={durations.max():.1f}")
        print(f"             mean={durations.mean():.1f}, std={durations.std():.1f}")
        print("="*60)

    def flush_queries(self):
        """MLPerf Loadgen callback for flushing queries"""
        self.logger.info("API SUT flush queries called")

    def _start_metrics_thread(self):
        """Start background thread for metrics collection"""
        def metrics_worker():
            self.logger.info(f"Starting metrics collection to {self.metrics_csv_path}")
            with open(self.metrics_csv_path, mode='w', newline='') as csvfile:
                writer = None
                while not self.metrics_stop_event.is_set():
                    try:
                        response = requests.get(self.metrics_endpoint, timeout=10)
                        if response.status_code == 200:
                            metrics_data = response.text
                            timestamp = datetime.now().isoformat()
                            
                            # Parse Prometheus format metrics
                            lines = [l for l in metrics_data.splitlines() if l and not l.startswith('#')]
                            metrics_dict = {l.split()[0]: l.split()[1] for l in lines if len(l.split()) == 2}
                            metrics_dict['timestamp'] = timestamp
                            
                            if writer is None:
                                fieldnames = list(metrics_dict.keys())
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                            
                            writer.writerow(metrics_dict)
                            csvfile.flush()
                        else:
                            self.logger.warning(f"Metrics endpoint returned status {response.status_code}")
                    except Exception as e:
                        self.logger.warning(f"Error collecting metrics: {e}")
                    
                    self.metrics_stop_event.wait(1)  # 1 second interval
            
            self.logger.info("Metrics collection stopped")
        
        self.metrics_thread = threading.Thread(target=metrics_worker, daemon=True)
        self.metrics_thread.start()

    def stop_metrics_thread(self):
        """Stop the metrics collection thread"""
        if self.enable_metrics_csv and self.metrics_thread is not None:
            self.metrics_stop_event.set()
            self.metrics_thread.join()
            self.logger.info("Metrics collection thread stopped")



# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Print comprehensive system information
    import sys
    print("=" * 80)
    print("MLPERF vLLM HARNESS - SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Executable: {sys.executable}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Print installed packages for debugging
    print("Key Python packages:")
    pkgs = sorted([(d.project_name, d.version) for d in pkg_resources.working_set], 
                  key=lambda x: x[0].lower())
    for name, version in pkgs:
        if any(keyword in name.lower() for keyword in ['torch', 'vllm', 'mlperf', 'transformers','tokenizers','cuda']):
            print(f"  {name:<30} {version}")
    print("=" * 80)

    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = "16"

    # Set TORCH_CUDA_ARCH_LIST based on device properties
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        arch_str = f"{props.major}.{props.minor}"
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_str
        print(f"Set TORCH_CUDA_ARCH_LIST to {arch_str}")
    else:
        print("CUDA not available. Not setting TORCH_CUDA_ARCH_LIST")

    # ========================================================================
    # Command Line Argument Parsing
    # ========================================================================
    
    parser = FlexibleArgumentParser(
        description="MLPerf vLLM Harness - Run vLLM models with MLPerf Loadgen"
    )
    
    # Add engine args from vLLM
    from vllm import EngineArgs
    EngineArgs.add_cli_args(parser)

    parser.set_defaults(max_model_len=131062,
        trust_remote_code=True)
    
    # Model and Data Configuration
    model_group = parser.add_argument_group('Model and Data')
    model_group.add_argument("--dataset-path", type=str, default=None, 
                           help="Path to the processed dataset pickle file")
    model_group.add_argument("--num-samples", type=int, default=4388, 
                           help="Number of samples for the test")
    
    # Performance Configuration
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument("--batch-size", type=int, default=4388, 
                          help="Batch size for processing")
    perf_group.add_argument("--num-workers", type=int, default=1, 
                          help="Number of worker threads for server scenario")
    
    # Scenario and Testing
    scenario_group = parser.add_argument_group('Scenario and Testing')
    scenario_group.add_argument("--scenario", type=str, default="Offline", 
                              choices=["Offline", "Server"], 
                              help="MLPerf scenario")
    scenario_group.add_argument("--test-mode", type=str, default="performance", 
                              choices=["performance", "accuracy"], 
                              help="Test mode")
    
    # Logging and Output
    log_group = parser.add_argument_group('Logging and Output')
    log_group.add_argument("--log-level", type=str, default="INFO", 
                         choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                         help="Logging level")
    log_group.add_argument("--output-log-dir", type=str, default="./", 
                         help="Directory for log output")
    
    # MLPerf Loadgen Configuration
    lg_group = parser.add_argument_group('MLPerf Loadgen')
    lg_group.add_argument("--user-conf", type=str, default="user.conf", 
                        help="User config for LoadGen settings")
    lg_group.add_argument("--audit-conf", type=str, default="",
                        help="Audit config for LoadGen settings")
    lg_group.add_argument("--lg-model-name", type=str, default="deepseek-r1", 
                        choices=["deepseek-r1", "test-model"], 
                        help="Model name for LoadGen")
    
    # Profiling and Analysis
    prof_group = parser.add_argument_group('Profiling and Analysis')
    prof_group.add_argument("--enable-profiler", action="store_true", 
                          help="Enable torch profiler")
    prof_group.add_argument("--profiler-dir", type=str, default="./torch_profiler_logs", 
                          help="Directory for profiler traces")
    prof_group.add_argument("--enable-nvtx", action="store_true", 
                          help="Enable NVTX profiling")
    prof_group.add_argument("--print-timing", action="store_true", 
                          help="Print timing statistics")
    
    # Data Analysis and Debugging
    debug_group = parser.add_argument_group('Data Analysis and Debugging')
    debug_group.add_argument("--print-histogram", action="store_true", 
                           help="Print histogram of input lengths")
    debug_group.add_argument("--sort-by-length", action="store_true", 
                           help="Sort queries by input token length")
    debug_group.add_argument("--sort-by-token-contents", action="store_true", 
                           help="Sort queries by token contents")
    debug_group.add_argument("--print-sorted-tokens", action="store_true", 
                           help="Print input token lists after sorting")
    
    # API Server Options
    api_group = parser.add_argument_group('API Server')
    api_group.add_argument("--api-server-url", type=str, default=None, 
                         help="URL of vLLM API server")
    api_group.add_argument("--enable-metrics-csv", action="store_true", 
                         help="Enable metrics collection (API only)")
    api_group.add_argument("--metrics-csv-path", type=str, default="metrics.csv", 
                         help="Path for metrics CSV file")
    
    args = parser.parse_args()

    # ========================================================================
    # Environment Setup
    # ========================================================================
    
    # Set profiler directory if enabled
    if args.enable_profiler:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profiler_dir
    os.environ["VLLM_NO_USAGE_STATS"] = "0"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Extract configuration variables
    MODEL_NAME = args.model
    DATASET_PATH = args.dataset_path
    NUM_SAMPLES = args.num_samples
    BATCH_SIZE = args.batch_size
    TEST_MODE = args.test_mode
    SCENARIO = args.scenario
    NUM_WORKERS = args.num_workers
    
    # Extract engine arguments for LLM initialization
    engine_args = {}
    for key, value in vars(args).items():
        if key not in ['dataset_path', 'num_samples', 'batch_size', 'test_mode', 'scenario', 
                      'log_level', 'output_log_dir', 'user_conf', 'audit_conf', 'lg_model_name',
                      'enable_profiler', 'profiler_dir', 'enable_nvtx', 'print_timing',
                      'print_histogram', 'sort_by_length', 'sort_by_token_contents', 
                      'print_sorted_tokens', 'api_server_url', 'enable_metrics_csv', 
                      'metrics_csv_path', 'num_workers']:
            engine_args[key] = value

    # Validation
    if DATASET_PATH is None:
        logging.error("Error: --dataset-path is required")
        exit(1)

    if NUM_SAMPLES <= 0:
        logging.error("Error: --num-samples must be at least 1")
        exit(1)

    # ========================================================================
    # SUT Selection and Initialization
    # ========================================================================
    
    logging.info("=" * 50)
    logging.info("INITIALIZING MLPerf vLLM HARNESS")
    logging.info("=" * 50)

    sut = None
    try:
        if args.api_server_url:
            # Use API server implementation
            logging.info(f"Using vLLM API server at: {args.api_server_url}")
            sut = VLLMSingleSUTAPI(
                model_name=MODEL_NAME,
                dataset_path=DATASET_PATH,
                api_server_url=args.api_server_url,
                test_mode=TEST_MODE,
                enable_profiler=args.enable_profiler,
                profiler_dir=args.profiler_dir,
                enable_nvtx=args.enable_nvtx,
                print_histogram=args.print_histogram,
                sort_by_length=args.sort_by_length,
                sort_by_token_contents=args.sort_by_token_contents,
                print_sorted_tokens=args.print_sorted_tokens,
                print_timing=args.print_timing,
                enable_metrics_csv=args.enable_metrics_csv,
                metrics_csv_path=args.metrics_csv_path
            )
        else:
            # Use local model for offline scenario
            logging.info("Using local vLLM model for Offline scenario")
            sut = VLLMSingleSUT(
                model_name=MODEL_NAME,
                dataset_path=DATASET_PATH,
                test_mode=TEST_MODE,
                enable_profiler=args.enable_profiler,
                profiler_dir=args.profiler_dir,
                enable_nvtx=args.enable_nvtx,
                print_histogram=args.print_histogram,
                sort_by_length=args.sort_by_length,
                sort_by_token_contents=args.sort_by_token_contents,
                print_sorted_tokens=args.print_sorted_tokens,
                print_timing=args.print_timing,
                **engine_args
            )

        # ====================================================================
        # MLPerf Loadgen Configuration and Execution
        # ====================================================================
        
        # Configure test settings
        settings = lg.TestSettings()
        if SCENARIO == "Server":
            settings.scenario = lg.TestScenario.Server
            settings.sample_concatenate_permutation = True
        else:
            settings.scenario = lg.TestScenario.Offline
            
        if TEST_MODE == "accuracy":
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly
            
        settings.use_token_latencies = True
        settings.FromConfig(args.user_conf, args.lg_model_name, SCENARIO, 1)

        # Configure logging settings
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.output_log_dir
        log_output_settings.copy_summary_to_stdout = True
        
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = False

        #Create the output directory if it doesn't exist
        if not os.path.exists(args.output_log_dir):
            os.makedirs(args.output_log_dir)

        # Create Query Sample Library
        qsl = lg.ConstructQSL(4388, NUM_SAMPLES, load_samples_to_ram, unload_samples_from_ram)
        
        # Create SUT for Loadgen
        SUTToTest = lg.ConstructSUT(sut.issue_query, sut.flush_queries)

        # Log test configuration
        logging.info("=" * 50)
        logging.info("STARTING MLPerf TEST")
        logging.info("=" * 50)
        logging.info(f"Model: {MODEL_NAME}")
        logging.info(f"Scenario: {SCENARIO}")
        logging.info(f"Test Mode: {TEST_MODE}")
        logging.info(f"Samples: {NUM_SAMPLES}")
        logging.info(f"Batch Size: {BATCH_SIZE}")
        logging.info(f"Engine Args: {engine_args}")
        if SCENARIO == "Server":
            logging.info(f"Server Workers: {NUM_WORKERS}")
        if args.audit_conf:
            logging.info(f"Audit Config: {args.audit_conf}")
        if args.enable_profiler:
            logging.info(f"Profiling enabled - traces in {args.profiler_dir}")
        if args.enable_nvtx:
            logging.info("NVTX profiling enabled")

        # Start timing measurement
        test_start_time = time.time()
        logging.info(f"Test start time: {datetime.fromtimestamp(test_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run the test
        lg.StartTestWithLogSettings(SUTToTest, qsl, settings, log_settings, args.audit_conf)

        # End timing measurement
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        
        # Test completion
        logging.info("=" * 50)
        logging.info("MLPerf TEST COMPLETED SUCCESSFULLY")
        logging.info("=" * 50)
        logging.info(f"Test end time: {datetime.fromtimestamp(test_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total test execution time: {test_duration:.2f} seconds ({test_duration/60:.2f} minutes)")
        logging.info("=" * 50)

        # Cleanup
        if args.api_server_url and args.enable_metrics_csv and hasattr(sut, 'stop_metrics_thread'):
            sut.stop_metrics_thread()

    except Exception as e:
        logging.critical(f"Critical error in main program: {e}", exc_info=True)
        exit(1) 