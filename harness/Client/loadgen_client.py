# ============================================================================
# loadgen_client.py
# -----------------
# MLPerf LoadGen client implementation
# Supports both Offline and Server scenarios
# ============================================================================

import os
import sys
import time
import logging
import numpy as np
import requests
import json
import array
import threading
import queue
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

# Add parent directories to path for imports
harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harness_root)

# Import base client and dataset processor
try:
    from Client.base_client import BaseClient
    from data.dataset_processor import DatasetProcessor
except ImportError:
    # Try relative imports if absolute fails
    from .base_client import BaseClient
    import sys
    import os
    sys.path.insert(0, os.path.dirname(harness_root))
    from harness.data.dataset_processor import DatasetProcessor

# Import MLPerf Loadgen
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    sys.exit(1)

# Import tokenizer for API mode
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


class LoadGenClient(BaseClient):
    """
    MLPerf LoadGen client implementation.
    
    Base class for LoadGen clients (Offline and Server scenarios).
    """
    
    def __init__(self,
                 model_name: str,
                 dataset_path: str,
                 scenario: str = "Offline",
                 test_mode: str = "performance",
                 api_server_url: Optional[str] = None,
                 batch_size: int = 13368,
                 num_samples: int = 13368,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize LoadGen client.
        
        Args:
            model_name: Model name or path
            dataset_path: Path to dataset file
            scenario: LoadGen scenario ("Offline" or "Server")
            test_mode: Test mode ("performance" or "accuracy")
            api_server_url: Optional API server URL (if using remote server)
            batch_size: Batch size for processing
            num_samples: Number of samples for testing
            config: Additional configuration
        """
        super().__init__("loadgen", model_name, dataset_path, config)
        
        self.scenario = scenario
        self.test_mode = test_mode
        self.api_server_url = api_server_url
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        # Dataset processor
        self.dataset: Optional[DatasetProcessor] = None
        
        # API mode components
        self.tokenizer = None
        self.completions_endpoint = None
        self.health_endpoint = None
        self.server_ready = False
        
        # Server scenario specific components (for async processing)
        self.num_workers = config.get('num_workers', 1) if config else 1
        self.worker_threads: List[Optional[threading.Thread]] = []
        self.first_token_queue: Optional[queue.Queue] = None
        self.query_queue: Optional[queue.Queue] = None
        self.ft_response_thread: Optional[threading.Thread] = None
        self.workers_started = False
        
        if self.api_server_url:
            self.api_server_url = self.api_server_url.rstrip('/')
            self.completions_endpoint = f"{self.api_server_url}/v1/completions"
            self.health_endpoint = f"{self.api_server_url}/health"
    
    def initialize(self) -> None:
        """Initialize LoadGen client."""
        self.logger.info(f"Initializing LoadGen client (scenario: {self.scenario})")
        
        # Load dataset
        self.logger.info(f"Loading dataset from: {self.dataset_path}")
        self.dataset = DatasetProcessor(
            dataset_path=self.dataset_path,
            model_name=self.model_name,
            total_sample_count=self.num_samples
        )
        
        # Print dataset statistics
        stats = self.dataset.get_statistics()
        self.logger.info("=" * 60)
        self.logger.info("Dataset Statistics")
        self.logger.info("=" * 60)
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)
        
        # Initialize tokenizer if using API mode
        if self.api_server_url:
            self._initialize_tokenizer()
            self._wait_for_server_ready()
        
        # Initialize server scenario components if needed
        if self.scenario == "Server" and self.api_server_url:
            self._initialize_server_components()
            # Note: Workers will be started when first query arrives or can be started explicitly
        
        self.is_initialized = True
        self.logger.info("LoadGen client initialized successfully")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the LoadGen client.
        
        Note: For LoadGen clients, the actual execution is handled by
        the harness calling issue_query() and flush_queries().
        This method marks the client as running.
        
        Returns:
            Dictionary with client status
        """
        self.is_running = True
        self.logger.info("LoadGen client is running")
        return {
            'status': 'running',
            'scenario': self.scenario,
            'test_mode': self.test_mode
        }
    
    def _initialize_server_components(self):
        """Initialize components for server scenario with async processing."""
        self.worker_threads = [None] * self.num_workers
        self.first_token_queue = queue.Queue()
        self.query_queue = queue.Queue()
        self.workers_started = False
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for API mode."""
        if not TOKENIZER_AVAILABLE:
            self.logger.warning("Transformers not available, tokenizer not initialized")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.logger.info("Tokenizer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    def _wait_for_server_ready(self, timeout: int = 600):
        """Wait for API server to become ready."""
        if not self.api_server_url:
            return
        
        self.logger.info(f"Waiting for API server at {self.api_server_url} (timeout: {timeout}s)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.health_endpoint, timeout=10)
                if response.status_code == 200:
                    self.logger.info("API server is ready!")
                    self.server_ready = True
                    return
            except Exception as e:
                self.logger.debug(f"API server not ready: {e}")
            
            time.sleep(2)
        
        raise RuntimeError(f"API server at {self.api_server_url} did not become ready within {timeout} seconds")
    
    @abstractmethod
    def issue_query(self, query_samples: List['lg.QuerySample']) -> None:
        """
        Process query samples from MLPerf Loadgen.
        
        This method must be implemented by subclasses (Offline/Server).
        
        Args:
            query_samples: List of MLPerf QuerySample objects
        """
        pass
    
    @abstractmethod
    def flush_queries(self) -> None:
        """
        Flush any pending queries.
        MLPerf Loadgen callback.
        """
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up LoadGen client")
        self.is_running = False


class LoadGenOfflineClient(LoadGenClient):
    """LoadGen client for Offline scenario."""
    
    def __init__(self, *args, **kwargs):
        # Force scenario to Offline
        kwargs['scenario'] = 'Offline'
        super().__init__(*args, **kwargs)
        self.batch_counter = 0
    
    def issue_query(self, query_samples: List['lg.QuerySample']) -> None:
        """Process queries in batches for offline scenario."""
        total_samples = len(query_samples)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Processing {total_samples} queries in {num_batches} batches")
        
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, total_samples)
            batch = query_samples[start:end]
            
            try:
                if self.api_server_url:
                    self._process_api_batch(batch)
                else:
                    self.logger.warning("Local model processing not yet implemented")
                    # TODO: Implement local model processing
                
                self.batch_counter += 1
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                self._send_error_responses(batch)
    
    def _process_api_batch(self, batch: List['lg.QuerySample']) -> None:
        """Process a batch via API."""
        # Prepare text prompts
        text_prompts = []
        original_query_ids = []
        original_query_indexes = []
        
        for q_sample in batch:
            original_query_ids.append(q_sample.id)
            original_query_indexes.append(q_sample.index)
            
            # Get input IDs from dataset
            input_ids = self.dataset.input_ids[q_sample.index]
            
            # Decode to text if tokenizer available
            if self.tokenizer:
                try:
                    text_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    text_prompts.append(text_prompt)
                except Exception as e:
                    self.logger.warning(f"Error decoding tokens: {e}")
                    text_prompts.append(" ".join([str(t) for t in input_ids]))
            else:
                text_prompts.append(" ".join([str(t) for t in input_ids]))
        
        # Send API request
        api_payload = {
            "model": self.model_name,
            "prompt": text_prompts,
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "stream": False
        }
        
        response = requests.post(self.completions_endpoint, json=api_payload, timeout=None)
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")
        
        api_result = response.json()
        choices = api_result.get("choices", [])
        
        # Process responses
        self._process_api_responses(choices, original_query_ids, original_query_indexes)
    
    def _process_api_responses(self, choices: List[Dict], query_ids: List[int], query_indexes: List[int]) -> None:
        """Process API responses and send to Loadgen."""
        for i, choice in enumerate(choices):
            query_id = query_ids[i]
            query_index = query_indexes[i]
            
            # Extract text response
            text_response = choice.get("text", "")
            
            # Convert back to token IDs
            if self.tokenizer:
                try:
                    token_ids = self.tokenizer.encode(text_response, add_special_tokens=False)
                except Exception as e:
                    self.logger.warning(f"Error encoding response: {e}")
                    token_ids = [1, 2, 3]  # Fallback
            else:
                token_ids = [1, 2, 3]  # Fallback
            
            token_count = len(token_ids)
            
            # Create Loadgen response
            token_array = np.array(token_ids, dtype=np.int32)
            token_bytes = token_array.tobytes()
            response_data = token_array.ctypes.data
            response_size = len(token_bytes)
            
            response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
            self.logger.info(f"Query {query_id}: {token_count} tokens")
            self.logger.debug(f"Query {query_id}: Response: {text_response}")
            lg.QuerySamplesComplete([response])
    
    def _send_error_responses(self, batch: List['lg.QuerySample']) -> None:
        """Send error responses for a batch."""
        for q_sample in batch:
            response = lg.QuerySampleResponse(q_sample.id, 0, 0, 0)
            lg.QuerySamplesComplete([response])
    
    def flush_queries(self) -> None:
        """Flush queries (no-op for offline scenario)."""
        self.logger.debug("Flush queries called (no-op for offline)")


class LoadGenServerClient(LoadGenClient):
    """
    LoadGen client for Server scenario.
    
    Inspired by SUT_VLLM_SingleReplica_Server.py with:
    - Worker threads for async query processing
    - Streaming API support
    - First token handling
    - Query queue management
    """
    
    def __init__(self, *args, **kwargs):
        # Force scenario to Server
        kwargs['scenario'] = 'Server'
        super().__init__(*args, **kwargs)
        self.query_counter = 0
    
    def start_workers(self):
        """Start worker threads for async query processing."""
        if self.workers_started:
            return
        
        if not self.api_server_url:
            self.logger.warning("Workers not started - no API server URL")
            return
        
        self.logger.info(f"Starting {self.num_workers} worker threads")
        
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self._process_queries_worker, daemon=True)
            worker.start()
            self.worker_threads[j] = worker
        
        # Create first token response thread
        self.ft_response_thread = threading.Thread(target=self._process_first_tokens_worker, daemon=True)
        self.ft_response_thread.start()
        
        self.workers_started = True
        self.logger.info("Worker threads started")
    
    def _process_first_tokens_worker(self):
        """Worker thread to process first token responses."""
        while True:
            try:
                first_token_item = self.first_token_queue.get()
                
                if first_token_item is None:
                    self.logger.info("Exiting first token response thread")
                    break
                
                first_token_id, response_id = first_token_item
                
                # Create first token response (single token)
                # Convert to list for array creation
                first_tokens = [first_token_id] if isinstance(first_token_id, int) else first_token_id
                response_data = array.array("B", np.array(first_tokens, dtype=np.int32).tobytes())
                bi = response_data.buffer_info()
                response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
                lg.FirstTokenComplete(response)
                
            except Exception as e:
                self.logger.error(f"Error in first token worker: {e}")
    
    def _process_queries_worker(self):
        """Worker thread to process queued queries."""
        while True:
            try:
                qitem = self.query_queue.get()
                
                if qitem is None:
                    self.logger.debug("Worker thread exiting")
                    break
                
                # Get input IDs from dataset
                input_ids_tensor = self.dataset.input_ids[qitem.index]
                
                # Process query asynchronously
                threading.Thread(
                    target=self._async_process_query,
                    args=(input_ids_tensor, qitem.id),
                    daemon=True
                ).start()
                
            except Exception as e:
                self.logger.error(f"Error in query worker: {e}")
    
    def _async_process_query(self, input_ids_tensor: List[int], query_id: int):
        """Process a single query asynchronously via streaming API."""
        try:
            # Decode input IDs to text
            if self.tokenizer:
                decoded = self.tokenizer.decode(input_ids_tensor, skip_special_tokens=True)
            else:
                decoded = " ".join([str(t) for t in input_ids_tensor])
            
            # Process via streaming API
            response_ids = [query_id]
            output_tokens = self._stream_api_vllm(decoded, response_ids)
            
            n_tokens = len(output_tokens)
            self.logger.debug(f"Query {query_id}: {n_tokens} tokens")
            
            if n_tokens <= 1:
                self.logger.warning(f"Low token count for query {query_id}: {n_tokens}")
            
            # Create final response
            response_array = array.array("B", np.array(output_tokens, dtype=np.int32).tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(query_id, bi[0], bi[1], n_tokens)]
            lg.QuerySamplesComplete(response)
            
        except Exception as e:
            self.logger.error(f"Error processing query {query_id}: {e}")
            self._send_error_response_by_id(query_id)
    
    def _stream_api_vllm(self, input_text: str, response_ids: List[int]) -> List[int]:
        """
        Stream API call to vLLM server with first token handling.
        
        Args:
            input_text: Input text prompt
            response_ids: List of response IDs (for first token handling)
        
        Returns:
            List of output token IDs
        """
        headers = {
            'Content-Type': 'application/json',
        }
        
        json_data = {
            'model': self.model_name,
            'prompt': input_text,
            'max_tokens': 128,
            'min_tokens': 1,
            'temperature': 0.0,
            'stream': True,
            'top_p': 1.0,
            'top_k': 1.0,
            'seed': 42
        }
        
        token_s_cache = []
        first = True
        
        while True:
            try:
                s = requests.Session()
                with s.post(
                    f'{self.api_server_url}/v1/completions',
                    headers=headers,
                    json=json_data,
                    verify=False,
                    stream=True,
                    timeout=None
                ) as resp:
                    if resp.status_code != 200:
                        self.logger.error(f"API server returned status {resp.status_code}: {resp.text}")
                        break
                    
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")
                            
                            if decoded.startswith("data") and "[DONE]" not in decoded:
                                try:
                                    data = json.loads(decoded[len("data: "):])
                                    finish_reason = data["choices"][0].get("finish_reason")
                                    stop_reason = data["choices"][0].get("stop_reason")
                                    
                                    if (finish_reason is not None) or (stop_reason is not None):
                                        if finish_reason == "length":
                                            # Add EOS token
                                            if self.tokenizer and hasattr(self.tokenizer, 'eos_token'):
                                                token_s = self.tokenizer.eos_token
                                                token_s_cache.append(token_s)
                                        else:
                                            self.logger.warning(
                                                f"Sequence finished: finish_reason={finish_reason}, "
                                                f"stop_reason={stop_reason}"
                                            )
                                        continue
                                    
                                    token_s = data["choices"][0]["text"]
                                    
                                    if token_s == "":
                                        continue
                                    
                                    # Handle first token
                                    if first:
                                        if self.tokenizer:
                                            token_ids = self.tokenizer.encode(token_s, add_special_tokens=False)
                                            if token_ids:
                                                self.first_token_queue.put((token_ids[0], response_ids[0]))
                                        first = False
                                    
                                    token_s_cache.append(str(token_s))
                                    
                                except json.JSONDecodeError as e:
                                    self.logger.debug(f"JSON decode error: {e}")
                                    continue
                                except Exception as e:
                                    self.logger.debug(f"Error parsing stream line: {e}")
                                    continue
                    
                    s.close()
                    
                    # Convert accumulated tokens to token IDs
                    if token_s_cache:
                        if self.tokenizer:
                            full_text = "".join(token_s_cache)
                            return self.tokenizer.encode(full_text, add_special_tokens=False)
                        else:
                            # Fallback: return placeholder tokens
                            self.logger.warning("No tokenizer available for encoding response")
                            return [1, 2, 3]
                    
                    break
                    
            except Exception as e:
                self.logger.error(f"Connection failure: {e}")
                s.close()
                # Return fallback tokens
                return [1, 2, 3]
        
        # Fallback if no tokens collected
        return [1, 2, 3]
    
    def issue_query(self, query_samples: List['lg.QuerySample']) -> None:
        """
        Process queries by queuing them for async processing.
        
        In server scenario, queries are queued and processed asynchronously
        by worker threads with streaming API support.
        """
        if not self.server_ready:
            self.logger.error("API server is not ready")
            self._send_error_responses(query_samples)
            return
        
        # Start workers if not already started
        if not self.workers_started:
            self.start_workers()
        
        # Queue queries for processing
        for sample in query_samples:
            self.logger.debug(f"Queuing query {sample.id} (index: {sample.index})")
            self.query_queue.put(sample)
            self.query_counter += 1
    
    def _send_error_responses(self, query_samples: List['lg.QuerySample']) -> None:
        """Send error responses for all queries."""
        for q_sample in query_samples:
            self._send_error_response_by_id(q_sample.id)
    
    def _send_error_response_by_id(self, query_id: int) -> None:
        """Send error response for a query ID."""
        response = lg.QuerySampleResponse(query_id, 0, 0, 0)
        lg.QuerySamplesComplete([response])
    
    def flush_queries(self) -> None:
        """
        Flush queries for server scenario.
        Signals workers to complete pending queries.
        """
        self.logger.info("Flush queries called (server scenario)")
        # Note: In server scenario, we let workers complete their current queries
        # Actual flushing logic can be added if needed
    
    def stop_workers(self):
        """Stop worker threads gracefully."""
        if not self.workers_started:
            return
        
        self.logger.info("Stopping worker threads...")
        
        # Signal workers to stop
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            if worker and worker.is_alive():
                worker.join(timeout=5)
        
        # Signal first token thread to stop
        if self.first_token_queue:
            self.first_token_queue.put(None)
        
        if self.ft_response_thread and self.ft_response_thread.is_alive():
            self.ft_response_thread.join(timeout=5)
        
        self.workers_started = False
        self.logger.info("Worker threads stopped")
    
    def cleanup(self) -> None:
        """Cleanup resources including worker threads."""
        self.stop_workers()
        super().cleanup()


def create_loadgen_client(scenario: str, *args, **kwargs) -> LoadGenClient:
    """
    Factory function to create LoadGen client instances.
    
    Args:
        scenario: "Offline" or "Server"
        *args, **kwargs: Arguments passed to client constructor
    
    Returns:
        LoadGenClient instance
    """
    scenario_lower = scenario.lower()
    
    if scenario_lower == "offline":
        return LoadGenOfflineClient(*args, **kwargs)
    elif scenario_lower == "server":
        return LoadGenServerClient(*args, **kwargs)
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Must be 'Offline' or 'Server'")

