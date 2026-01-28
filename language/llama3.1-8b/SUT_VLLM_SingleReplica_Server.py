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
import argparse
import numpy as np
from typing import List
from dataset import Dataset
from vllm import TokensPrompt
import sys
import torch
import pkg_resources
from datetime import datetime
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector
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
from openai import OpenAI


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



class VLLMSingleSUTAPI:
    """
    vLLM API Server SUT Implementation
    
    This class communicates with a remote vLLM API server instead of running
    the model locally. It handles API communication, tokenization/detokenization,
    and optional metrics collection.
    """
    
    def __init__(self, model_name: str, dataset_path: str, api_server_url: str, 
                 max_model_len: int = 2048, test_mode: str = "performance", 
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
        self.max_model_len = max_model_len
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
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=13368)
        self.logger.info("API Dataset loaded: %d samples", len(self.data_object.input_ids))
        self.logger.info("Dataset statistics - Max Inp Tokens: %d, Min Input Tokens: %d, Total Samples: %d", 
                        max(self.data_object.input_lens), 
                        min(self.data_object.input_lens),
                        len(self.data_object.input_lens))
        
        # Wait for server readiness and initialize components
        self._wait_for_server_ready()
        self._initialize_tokenizer()

        self.num_workers = 1
        self.worker_threads = [None] * self.num_workers
        self.first_token_queue = queue.Queue()
        self.query_queue = queue.Queue()


        # Start metrics collection if enabled
        if self.enable_metrics_csv:
            self._start_metrics_thread()

    def start(self):

        print(f"Starting {self.num_workers} workers")
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        # Create first token response thread
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.start()

    def process_first_tokens(self):

        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                self.logger.info("Exiting First token response thread")
                break

            first_tokens, response_id = first_token_item

            response_data = array.array("B", np.array(first_tokens, np.int32).tobytes())
            bi = response_data.buffer_info()
            response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
            lg.FirstTokenComplete(response)
    
    def stream_api_vllm(self, input, response_ids):
        """        
        client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="EMPTY",
        base_url="http://0.0.0.0:8000/v1/"
        )

        models = client.models.list()
        model = models.data[0].id
        self.logger.info(f"{models}")

        # Completion API
        completion = client.completions.create(
            model=self.model_name,
            prompt=input,
            echo=False,
            stream=True,
            max_tokens=128
        )

        for c in completion:
            self.logger.info(f"{c}")
        """

        
        headers = {
            'Content-Type': 'application/json',
        }

        json_data = {
            'model': self.model_name,
            'prompt': input,
            'max_tokens': 128,
            'min_tokens':1,
            'temperature': 0.0,
            'stream': True,
            'top_p':1.0,
            'top_k':1.0,
            'seed':42
            #'stream_options': {'include_usage': True},
            #'logprobs': 1
        }

        while True:
            try:
                token_s_cache = []
                s = requests.Session()
                first = True
                with s.post(
                    f'{self.api_server_url}/v1/completions',
                    headers=headers,
                    json=json_data,
                    verify=False,
                    stream=True
                ) as resp:
                    if resp.status_code != 200:
                        self.logger.error(f"API server returned status {resp.status_code}: {resp.text}")
                        continue
                    #self.logger.info(f"Response: {type(resp)}")

                    for line in resp.iter_lines():
                        if line:
                            #self.logger.info(f"Line: {line}")
                            decoded = line.decode("utf-8")
                            #self.logger.info(f"Decoded: {decoded}")

                            #if decoded.startswith("b'data: "):
                            #   data = decoded[len("b'data: "):]
                            
                            #decoded = json.loads(line.decode())
                            #self.logger.info(f"Decoded: {decoded}")
                            if decoded.startswith("data") and "[DONE]" not in decoded:
                                data = json.loads(decoded[len("data: "):])
                                #self.logger.info(f"Data: {data}")
                                finish_reason = data["choices"][0].get("finish_reason")
                                #self.logger.info(f"Finish reason: {finish_reason}")
                                stop_reason   = data["choices"][0].get("stop_reason")
                                #self.logger.info(f"Stop reason: {stop_reason}")
                                if (finish_reason is not None) or (stop_reason is not None):
                                    if finish_reason == "length":
                                        token_s = self.tokenizer.eos_token
                                        token_s_cache.append(token_s)
                                    else:
                                        self.logger.warning(f"Sequence finished without hitting eos token, finish_reason: {finish_reason}, stop_reason: {stop_reason}")
                                    continue

                                #inter = data["choices"][0]["logprobs"]
                                #if "top_logprobs" in inter:
                                #    token_s = list(inter["top_logprobs"][0].keys())[0]

                                token_s = data["choices"][0]["text"]

                                if token_s == "":
                                    #print(f"Warning: empty token. Last non-empty token was: \"{token_s_cache[-1]}\"")
                                    continue

                                if first:
                                    token_ids = self.tokenizer.encode(token_s)
                                    self.first_token_queue.put((token_ids[0], response_ids[0]))
                                    first = False
                                token_s_cache.append(str(token_s))

                s.close()
                if token_s_cache:
                    self.logger.debug(f"Request completed! {len(token_s_cache)} tokens")
                    #print("Request completed!")
                    #print(token_s_cache)
                    #print("".join(token_s_cache))
                    return self.tokenizer.encode("".join(token_s_cache))
            except Exception as e:
                s.close()
                print(f"Connection failure: {e}")
        
    
    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = self.data_object.input_ids[qitem.index]

            self.logger.debug(f"Number of threads: {threading.active_count()}")
            threading.Thread(target=self.async_process_query, args=(input_ids_tensor, qitem.id)).start()
            #self.async_process_query(input_ids_tensor, qitem.id)
          


    def async_process_query(self, input_ids_tensor, qitem_id):
        decoded = self.tokenizer.decode(input_ids_tensor,skip_special_tokens=True)
        #self.logger.info(f"Decoded: {decoded}")
        response_ids = [qitem_id]
        output_tokens = self.stream_api_vllm(decoded, response_ids)

        n_tokens = len(output_tokens)
        self.logger.info(f"{response_ids} Number of tokens: {n_tokens}")
        if n_tokens <= 1:
            print("WARNING: caught low token count")
            print(input_ids_tensor)
            print(output_tokens)
        response_array = array.array("B", np.array(output_tokens, np.int32).tobytes())
        bi = response_array.buffer_info()
        response = [lg.QuerySampleResponse(
            qitem_id, bi[0], bi[1], n_tokens)]
        lg.QuerySamplesComplete(response)
        sys.exit()

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
        for sample in query_samples:
            self.logger.debug(f"Issuing {sample.index} query")
            self.query_queue.put(sample)
     

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
            if self.test_mode == "accuracy":
                token_array = np.array(token_ids, dtype=np.int32)
                token_bytes = token_array.tobytes()
                response_data = token_array.ctypes.data
                response_size = len(token_bytes)
                response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
                lg.QuerySamplesComplete([response])
            else:
                response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
            
            if self.test_mode == "performance":
                responses_to_loadgen.append(response)
        
        # Send responses to Loadgen
        if responses_to_loadgen and self.test_mode == "performance":
            lg.QuerySamplesComplete(responses_to_loadgen)

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
        #self.stop()

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
    
    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()


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

    # ========================================================================
    # Command Line Argument Parsing
    # ========================================================================
    
    parser = argparse.ArgumentParser(
        description="MLPerf vLLM Harness - Run vLLM models with MLPerf Loadgen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and Data Configuration
    model_group = parser.add_argument_group('Model and Data')
    model_group.add_argument("--model-name", type=str, 
                           default="HuggingFaceH4/tiny-random-LlamaForCausalLM", 
                           help="The name of the LLM model to load")
    model_group.add_argument("--dataset-path", type=str, default=None, 
                           help="Path to the processed dataset pickle file")
    model_group.add_argument("--num-samples", type=int, default=13368, 
                           help="Number of samples for the test")
    
    # Performance Configuration
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument("--max-model-len", type=int, default=131072, 
                          help="Maximum sequence length for the model")
    perf_group.add_argument("--max-num-seqs", type=int, default=512, 
                          help="Maximum sequences processed simultaneously")
    perf_group.add_argument("--gpu-mem-util", type=float, default=0.9, 
                          help="GPU memory utilization factor (0.0 to 1.0)")
    perf_group.add_argument("--batch-size", type=int, default=32, 
                          help="Batch size for processing")
    perf_group.add_argument("--max-num-batched-tokens", type=int, default=None, 
                          help="Maximum number of batched tokens for vLLM")
    perf_group.add_argument("--num-workers", type=int, default=1, 
                          help="Number of worker threads for server scenario")
    perf_group.add_argument("--kv-cache-dtype", type=str, default="auto", 
                          choices=["auto", "fp8", "fp16", "fp32"], 
                          help="Data type for KV cache (fp8 for memory efficiency)")
    
    # Scenario and Testing
    scenario_group = parser.add_argument_group('Scenario and Testing')
    scenario_group.add_argument("--scenario", type=str, default="Server", 
                              choices=["Offline", "Server"], 
                              help="MLPerf scenario")
    scenario_group.add_argument("--test-mode", type=str, default="performance", 
                              choices=["performance", "accuracy"], 
                              help="Test mode")
    
    # Hardware Configuration
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument("--cuda-device", type=str, default=None,
                        help="CUDA device to use (e.g., '0', '1', '0,1'). Sets CUDA_VISIBLE_DEVICES environment variable")
    hw_group.add_argument("--num-gpus", type=int, default=1, 
                        help="Number of GPUs (tensor_parallel_size)")
    hw_group.add_argument("--pipeline-parallel-size", type=int, default=1, 
                        help="Pipeline parallel size")
    hw_group.add_argument("--swap-space", type=int, default=4, 
                        help="Swap space parameter")
    
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
    lg_group.add_argument("--lg-model-name", type=str, default="llama3_1-8b", 
                        choices=["llama3_1-8b", "llama3_1-8b-interactive", "test-model"], 
                        help="Model name for LoadGen")
    lg_group.add_argument("--target-qps", type=float, default=None,
                        help="Target queries per second for LoadGen")
    lg_group.add_argument("--coalesce", action="store_true", default=False,
                        help="Enable coalesce option for LoadGen")
    
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
    
    # Set CUDA device if specified
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
        print(f"Set CUDA_VISIBLE_DEVICES to {args.cuda_device}")
    
    # Set profiler directory if enabled
    if args.enable_profiler:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profiler_dir
    os.environ["VLLM_NO_USAGE_STATS"] = "0"

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

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Extract configuration variables
    MODEL_NAME = args.model_name
    DATASET_PATH = args.dataset_path
    NUM_SAMPLES = args.num_samples
    MAX_MODEL_LEN = args.max_model_len
    MAX_NUM_SEQS = args.max_num_seqs
    GPU_MEM_UTIL = args.gpu_mem_util
    BATCH_SIZE = args.batch_size
    TEST_MODE = args.test_mode
    SCENARIO = args.scenario
    NUM_GPUS = args.num_gpus
    PIPELINE_PARALLEL_SIZE = args.pipeline_parallel_size
    SWAP_SPACE = args.swap_space
    MAX_NUM_BATCHED_TOKENS = args.max_num_batched_tokens
    NUM_WORKERS = args.num_workers
    KV_CACHE_DTYPE = args.kv_cache_dtype

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
                max_model_len=MAX_MODEL_LEN,
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
        # ====================================================================
        # MLPerf Loadgen Configuration and Execution
        # ====================================================================
        
        # Configure test settings
        settings = lg.TestSettings()
        if SCENARIO == "Server":
            settings.scenario = lg.TestScenario.Server
        else:
            settings.scenario = lg.TestScenario.Offline
            
        if TEST_MODE == "accuracy":
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly
            
        settings.use_token_latencies = True
        
        
        #
        
        settings.FromConfig(args.user_conf, args.lg_model_name, SCENARIO, 1)

         # Apply target QPS if specified
        if args.target_qps is not None:
            settings.server_target_qps = args.target_qps
            logging.info(f"Set target QPS to {args.target_qps}")

        # Apply coalesce setting if specified
        if args.coalesce:
            settings.server_coalesce_queries = True
            logging.info("Enabled coalesce queries for LoadGen")

        # Configure logging settings
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.output_log_dir
        log_output_settings.copy_summary_to_stdout = True
        
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = False

        # Create the output directory if it doesn't exist
        if not os.path.exists(args.output_log_dir):
            os.makedirs(args.output_log_dir)
            logging.info(f"Created output log directory: {args.output_log_dir}")

        sut.start()
        # Create Query Sample Library
        qsl = lg.ConstructQSL(13368, NUM_SAMPLES, load_samples_to_ram, unload_samples_from_ram)
        
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
        if args.cuda_device is not None:
            logging.info(f"CUDA Device: {args.cuda_device}")
        if SCENARIO == "Server":
            logging.info(f"Server Workers: {NUM_WORKERS}")
        if args.target_qps is not None:
            logging.info(f"Target QPS: {args.target_qps}")
        if args.coalesce:
            logging.info("Coalesce: Enabled")
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
        lg.StartTestWithLogSettings(SUTToTest, qsl, settings, log_settings,args.audit_conf)

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
        
        sut.stop()

        # Cleanup
        if args.api_server_url and args.enable_metrics_csv and hasattr(sut, 'stop_metrics_thread'):
            sut.stop_metrics_thread()

    except Exception as e:
        logging.critical(f"Critical error in main program: {e}", exc_info=True)
        exit(1) 
