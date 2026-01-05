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
        self.chat_completions_endpoint = None
        self.health_endpoint = None
        self.server_ready = False
        
        # Endpoint type: 'completions' or 'chat_completions'
        self.endpoint_type = config.get('endpoint_type', 'completions') if config else 'completions'
        if self.endpoint_type not in ['completions', 'chat_completions']:
            raise ValueError(f"Invalid endpoint_type: {self.endpoint_type}. Must be 'completions' or 'chat_completions'")
        
        # Max tokens configuration - determine from model name or use config/default
        self.max_tokens = self._determine_max_tokens(model_name, config)
        self.logger.info(f"Using max_tokens: {self.max_tokens} for model: {model_name}")
        
        # Debug mode for accuracy mode
        self.debug_mode = config.get('debug_mode', False) if config else False
        
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
            self.chat_completions_endpoint = f"{self.api_server_url}/v1/chat/completions"
            self.health_endpoint = f"{self.api_server_url}/health"
            
            # Validate endpoint based on backend config
            self._validate_endpoint()
    
    def initialize(self) -> None:
        """Initialize LoadGen client."""
        self.logger.info(f"Initializing LoadGen client (scenario: {self.scenario})")
        
        # Load dataset with configuration support
        self.logger.info(f"Loading dataset from: {self.dataset_path}")
        
        # Extract dataset configuration from config if available
        dataset_name = self.config.get('dataset_name')
        input_column = self.config.get('input_column')
        input_ids_column = self.config.get('input_ids_column')
        output_column = self.config.get('output_column')
        config_dir = self.config.get('config_dir')
        
        self.dataset = DatasetProcessor(
            dataset_path=self.dataset_path,
            model_name=self.model_name,
            total_sample_count=self.num_samples,
            dataset_name=dataset_name,
            input_column=input_column,
            input_ids_column=input_ids_column,
            output_column=output_column,
            config_dir=config_dir
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
    
    def _determine_max_tokens(self, model_name: str, config: Optional[Dict[str, Any]]) -> int:
        """
        Determine max_tokens based on model name or config.
        
        Defaults:
        - deepseek-r1: 20000
        - llama3.1-8b: 1024
        - llama2-70b: 1024
        - default: 1024
        """
        # Check if explicitly set in config
        if config and 'max_tokens' in config:
            return int(config['max_tokens'])
        
        # Determine from model name
        model_lower = model_name.lower()
        if 'deepseek' in model_lower and 'r1' in model_lower:
            return 20000
        elif 'llama3.1' in model_lower or 'llama-3.1' in model_lower or 'llama3_1' in model_lower:
            if '8b' in model_lower or '8-b' in model_lower:
                return 1024
        elif 'llama2' in model_lower or 'llama-2' in model_lower:
            if '70b' in model_lower or '70-b' in model_lower:
                return 1024
        
        # Default
        return 1024
    
    def _validate_endpoint(self):
        """Validate that the requested endpoint exists for the backend."""
        if not self.api_server_url:
            return
        
        backend = self.config.get('backend', 'vllm') if self.config else 'vllm'
        
        # Load backend config to check available endpoints
        try:
            from data.backend_config import BackendConfigLoader
            backend_loader = BackendConfigLoader()
            backend_config = backend_loader.load_backend_config(backend)
            
            # Check if endpoint is available
            available_endpoints = backend_config.get('endpoints', [])
            if self.endpoint_type not in available_endpoints:
                raise ValueError(
                    f"Endpoint '{self.endpoint_type}' is not available for backend '{backend}'. "
                    f"Available endpoints: {available_endpoints}"
                )
            
            self.logger.info(f"Validated endpoint '{self.endpoint_type}' for backend '{backend}'")
        except ImportError:
            # Backend config not available, skip validation
            self.logger.warning("Backend config loader not available, skipping endpoint validation")
        except Exception as e:
            # If backend config doesn't exist or other error, log warning but continue
            self.logger.warning(f"Could not validate endpoint: {e}")
    
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
    
    def get_sut(self):
        """
        Get SystemUnderTest object for LoadGen.
        
        Returns:
            LoadGen SUT object constructed from issue_query and flush_queries methods
        """
        return lg.ConstructSUT(self.issue_query, self.flush_queries)
    
    def get_qsl(self):
        """
        Get QuerySampleLibrary object for LoadGen.
        
        Returns:
            LoadGen QSL object constructed from dataset and callbacks
        """
        if not self.dataset:
            raise RuntimeError("Dataset not initialized. Call initialize() first.")
        
        total_count = self.dataset.total_sample_count
        # Use num_samples as performance_count to ensure all samples are processed
        # In offline scenario, LoadGen will create queries based on samples_per_query
        # but we want to ensure all num_samples are available
        performance_count = min(self.num_samples, total_count)
        
        self.logger.info(f"Constructing QSL: total_count={total_count}, performance_count={performance_count}, num_samples={self.num_samples}")
        
        return lg.ConstructQSL(
            total_count,
            performance_count,
            self._load_samples_to_ram,
            self._unload_samples_from_ram
        )
    
    def _load_samples_to_ram(self, query_sample_indices):
        """
        LoadGen callback: Load samples to RAM.
        
        Args:
            query_sample_indices: List of QuerySampleIndex objects
        """
        # Samples are already loaded in DatasetProcessor
        # This is a no-op as samples are pre-loaded
        pass
    
    def _unload_samples_from_ram(self, query_sample_indices):
        """
        LoadGen callback: Unload samples from RAM.
        
        Args:
            query_sample_indices: List of QuerySampleIndex objects
        """
        # Samples are kept in memory for the duration of the test
        # This is a no-op
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
        
        self.logger.info("=" * 80)
        self.logger.info(f"OFFLINE SCENARIO: Processing {total_samples} queries in {num_batches} batches")
        self.logger.info(f"Batch size: {self.batch_size}, Total samples: {total_samples}")
        self.logger.info("=" * 80)
        
        processed_samples = 0
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, total_samples)
            batch = query_samples[start:end]
            batch_size = len(batch)
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({batch_size} samples)")
            
            try:
                if self.api_server_url:
                    self._process_api_batch(batch)
                    processed_samples += batch_size
                    self.logger.info(f"✓ Batch {batch_idx + 1}/{num_batches} completed ({processed_samples}/{total_samples} samples processed)")
                else:
                    self.logger.warning("Local model processing not yet implemented")
                    # TODO: Implement local model processing
                    self._send_error_responses(batch)
                    processed_samples += batch_size
                
                self.batch_counter += 1
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx + 1}/{num_batches}: {e}", exc_info=True)
                self._send_error_responses(batch)
                processed_samples += batch_size
        
        self.logger.info("=" * 80)
        self.logger.info(f"All batches completed: {processed_samples}/{total_samples} samples processed")
        self.logger.info("=" * 80)
    
    def _process_api_batch(self, batch: List['lg.QuerySample']) -> None:
        """Process a batch via API."""
        batch_size = len(batch)
        self.logger.debug(f"Processing API batch with {batch_size} samples")
        
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
        
        self.logger.debug(f"Prepared {len(text_prompts)} prompts for API batch")
        
        # Determine endpoint based on endpoint_type
        if self.endpoint_type == 'chat_completions':
            endpoint = self.chat_completions_endpoint
            # Format for chat completions API - handle batch requests
            # For chat completions, we need to send each prompt separately or use array format
            # Most APIs support array format for batch processing
            if len(text_prompts) == 1:
                # Single prompt
                api_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": text_prompts[0]}],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False
                }
            else:
                # Batch: send as array of messages arrays
                # Note: Some APIs may require separate requests for batch
                api_payload = {
                    "model": self.model_name,
                    "messages": [[{"role": "user", "content": prompt}] for prompt in text_prompts],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False
                }
        else:
            endpoint = self.completions_endpoint
            # Format for completions API
            api_payload = {
                "model": self.model_name,
                "prompt": text_prompts,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "stream": False
            }
        
        self.logger.debug(f"Sending API batch request to {endpoint} with {len(text_prompts)} prompts")
        response = requests.post(endpoint, json=api_payload, timeout=None)
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")
        
        api_result = response.json()
        self.logger.debug(f"Received API response with {len(api_result.get('choices', []))} choices")
        
        # Extract choices based on endpoint type
        if self.endpoint_type == 'chat_completions':
            # Chat completions: handle batch vs single response
            all_choices = api_result.get("choices", [])
            if isinstance(all_choices, list) and len(all_choices) > 0:
                # Check if first element is a list (batch) or dict (single)
                if isinstance(all_choices[0], list):
                    # Batch response: flatten list of lists
                    choices = []
                    for choice_group in all_choices:
                        for choice in choice_group:
                            if isinstance(choice, dict) and "message" in choice:
                                choices.append({"text": choice["message"].get("content", "")})
                            else:
                                choices.append(choice)
                else:
                    # Single response or array of choices
                    choices = []
                    for choice in all_choices:
                        if isinstance(choice, dict) and "message" in choice:
                            choices.append({"text": choice["message"].get("content", "")})
                        else:
                            choices.append(choice)
            else:
                choices = []
        else:
            # Completions endpoint returns choices with text
            choices = api_result.get("choices", [])
        
        # Process responses
        self._process_api_responses(choices, original_query_ids, original_query_indexes, text_prompts)
    
    def _process_api_responses(self, choices: List[Dict], query_ids: List[int], query_indexes: List[int], text_prompts: Optional[List[str]] = None) -> None:
        """Process API responses and send to Loadgen."""
        self.logger.debug(f"Processing {len(choices)} API responses for {len(query_ids)} queries")
        
        responses = []
        for i, choice in enumerate(choices):
            if i >= len(query_ids):
                self.logger.warning(f"More choices than query IDs: {len(choices)} choices, {len(query_ids)} query IDs")
                break
                
            query_id = query_ids[i]
            query_index = query_indexes[i]
            
            # Extract text response
            text_response = choice.get("text", "")
            
            # Get the original query/prompt if available
            query_prompt = None
            if text_prompts and i < len(text_prompts):
                query_prompt = text_prompts[i]
            
            # Convert back to token IDs
            if self.tokenizer:
                try:
                    token_ids = self.tokenizer.encode(text_response, add_special_tokens=False)
                except Exception as e:
                    self.logger.warning(f"Error encoding response for query {query_id}: {e}")
                    token_ids = [1, 2, 3]  # Fallback
            else:
                token_ids = [1, 2, 3]  # Fallback
            
            token_count = len(token_ids)
            
            # Debug mode: print query, text response and token count in accuracy mode
            if self.debug_mode and self.test_mode == "accuracy":
                # Truncate text for display (first 200 chars)
                query_preview = query_prompt[:200] + "..." if query_prompt and len(query_prompt) > 200 else (query_prompt or "N/A")
                text_preview = text_response[:200] + "..." if len(text_response) > 200 else text_response
                self.logger.info(f"[DEBUG] Query {query_id} (index {query_index}):")
                self.logger.info(f"  Query: {query_preview}")
                self.logger.info(f"  Text Response: {text_preview}")
                self.logger.info(f"  Total Tokens: {token_count}")
            
            # Create Loadgen response
            token_array = np.array(token_ids, dtype=np.int32)
            token_bytes = token_array.tobytes()
            response_data = token_array.ctypes.data
            response_size = len(token_bytes)
            
            response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
            responses.append(response)
            self.logger.debug(f"Query {query_id} (index {query_index}): {token_count} tokens")
        
        # Send all responses to LoadGen
        if responses:
            lg.QuerySamplesComplete(responses)
            self.logger.debug(f"Sent {len(responses)} responses to LoadGen")
        else:
            self.logger.warning(f"No responses to send for batch (expected {len(query_ids)} responses)")
    
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
        
        # Determine endpoint and payload based on endpoint_type
        if self.endpoint_type == 'chat_completions':
            endpoint_url = self.chat_completions_endpoint
            json_data = {
                'model': self.model_name,
                'messages': [{"role": "user", "content": input_text}],
                'max_tokens': self.max_tokens,
                'temperature': 0.0,
                'stream': True,
                'top_p': 1.0,
                'seed': 42
            }
        else:
            endpoint_url = self.completions_endpoint
            json_data = {
                'model': self.model_name,
                'prompt': input_text,
                'max_tokens': self.max_tokens,
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
                    endpoint_url,
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
                                    
                                    # Extract token based on endpoint type
                                    if self.endpoint_type == 'chat_completions':
                                        # Chat completions uses delta.content
                                        delta = data["choices"][0].get("delta", {})
                                        token_s = delta.get("content", "")
                                    else:
                                        # Completions uses text
                                        token_s = data["choices"][0].get("text", "")
                                    
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

