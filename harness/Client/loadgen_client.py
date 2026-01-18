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
import base64
import threading
import queue
import random
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from io import BytesIO

# Try to import pandas for dataset operations
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add parent directories to path for imports
harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harness_root)

# Import base client and dataset processor
try:
    from Client.base_client import BaseClient
    from data.dataset_processor import DatasetProcessor
    # Try to import config availability flag
    try:
        from data.dataset_processor import CONFIG_AVAILABLE
    except ImportError:
        CONFIG_AVAILABLE = False
except ImportError:
    # Try relative imports if absolute fails
    from .base_client import BaseClient
    import sys
    import os
    sys.path.insert(0, os.path.dirname(harness_root))
    from harness.data.dataset_processor import DatasetProcessor
    try:
        from harness.data.dataset_processor import CONFIG_AVAILABLE
    except ImportError:
        CONFIG_AVAILABLE = False

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
                 api_server_urls: Optional[List[str]] = None,
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
            api_server_url: Optional API server URL (if using remote server) - backward compatible
            api_server_urls: Optional list of API server URLs for load balancing
            batch_size: Batch size for processing
            num_samples: Number of samples for testing
            config: Additional configuration
        """
        super().__init__("loadgen", model_name, dataset_path, config)
        
        self.scenario = scenario
        self.test_mode = test_mode
        
        # Handle load balancing: prefer api_server_urls, fall back to api_server_url
        if api_server_urls:
            self.api_server_urls = [url.rstrip('/') for url in api_server_urls]
            self.api_server_url = self.api_server_urls[0]  # Primary URL for backward compatibility
            self.load_balancing = True
            self.load_balance_strategy = config.get('load_balance_strategy', 'round_robin') if config else 'round_robin'
            self.current_server_index = 0
            self.failed_servers = set()  # Track servers that have failed
            self.max_retries_per_server = config.get('max_retries_per_server', 3) if config else 3
            self.logger.info(f"Load balancing enabled with {len(self.api_server_urls)} servers: {self.api_server_urls}")
            self.logger.info(f"Load balance strategy: {self.load_balance_strategy}")
        else:
            self.api_server_urls = None
            self.api_server_url = api_server_url.rstrip('/') if api_server_url else None
            self.load_balancing = False
            self.current_server_index = 0
            self.failed_servers = set()
        
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
        
        # Max tokens configuration - will be updated after dataset is loaded if dataset config has it
        self.max_tokens = self._determine_max_tokens(model_name, config, test_mode)
        self.logger.info(f"Initial max_tokens: {self.max_tokens} for model: {model_name} (test_mode: {test_mode})")
        
        # Sampling parameters - can be different for accuracy vs performance
        # For gpt-oss-120b, use same parameters for both modes: temperature=1.0, top_k=-1, top_p=1.0
        model_lower = model_name.lower()
        is_gpt_oss = 'gpt-oss' in model_lower or 'gpt_oss' in model_lower or 'gptoss' in model_lower
        
        if is_gpt_oss:
            # gpt-oss-120b uses same sampling params for both perf and accuracy
            self.temperature = config.get('temperature', 1.0) if config else 1.0
            self.top_k = config.get('top_k', -1) if config else -1
            self.top_p = config.get('top_p', 1.0) if config else 1.0
            # Set accuracy params to same values
            self.accuracy_temperature = config.get('accuracy_temperature', 1.0) if config else 1.0
            self.accuracy_top_k = config.get('accuracy_top_k', -1) if config else -1
            self.accuracy_top_p = config.get('accuracy_top_p', 1.0) if config else 1.0
        else:
            # Default behavior for other models
            self.temperature = config.get('temperature', 0.0) if config else 0.0
            self.top_k = config.get('top_k', 1) if config else 1
            self.top_p = config.get('top_p', 1.0) if config else 1.0
            # Accuracy mode parameters (if specified, override for accuracy mode)
            self.accuracy_temperature = config.get('accuracy_temperature', None) if config else None
            self.accuracy_top_k = config.get('accuracy_top_k', None) if config else None
            self.accuracy_top_p = config.get('accuracy_top_p', None) if config else None
        
        # SGLang-specific: use input_ids directly instead of text
        # Auto-detect SGLang backend from server_config
        backend = None
        if config and 'server_config' in config:
            backend = config['server_config'].get('backend', 'vllm')
        
        # Set use_input_ids if backend is SGLang or explicitly set in config
        self.use_input_ids = config.get('use_input_ids', False) if config else False
        if backend and backend.lower() == 'sglang' and not self.use_input_ids:
            # Auto-enable input_ids mode for SGLang backend
            self.use_input_ids = True
            self.logger.info("Auto-detected SGLang backend, enabling input_ids mode")
        
        self.sglang_endpoint = config.get('sglang_endpoint', '/generate') if config else '/generate'
        
        # Multimodal-specific: use messages format directly (for qwen3vl)
        self.use_messages = config.get('use_messages', False) if config else False
        self.multimodal = config.get('multimodal', False) if config else False
        self.use_guided_decoding = config.get('use_guided_decoding', False) if config else False
        
        # Offline scenario: send requests back-to-back instead of batching
        self.offline_back_to_back = config.get('offline_back_to_back', False) if config else False
        
        # Debug mode for accuracy mode
        self.debug_mode = config.get('debug_mode', False) if config else False
        
        # Server scenario specific components (for async processing)
        self.num_workers = config.get('num_workers', 1) if config else 1
        self.worker_threads: List[Optional[threading.Thread]] = []
        self.first_token_queue: Optional[queue.Queue] = None
        self.query_queue: Optional[queue.Queue] = None
        self.ft_response_thread: Optional[threading.Thread] = None
        self.workers_started = False
        
        # Initialize endpoints (for load balancing, use primary URL for endpoints)
        if self.load_balancing:
            # For load balancing, endpoints are constructed per-request
            primary_url = self.api_server_urls[0]
            self.completions_endpoint = f"{primary_url}/v1/completions"
            self.chat_completions_endpoint = f"{primary_url}/v1/chat/completions"
            self.health_endpoint = f"{primary_url}/health"
        elif self.api_server_url:
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
        
        # Determine total_sample_count to pass to DatasetProcessor
        # If dataset config exists and has total_sample_count, use None to let config handle it
        # Otherwise, use self.num_samples
        # We'll check for config first by trying to load it
        total_sample_count_for_loader = self.num_samples
        try:
            if CONFIG_AVAILABLE and dataset_name:
                from data.dataset_config import DatasetConfigLoader
                config_loader = DatasetConfigLoader(config_dir=config_dir)
                dataset_config = config_loader.load_dataset_config(dataset_name, self.model_name)
                if dataset_config and dataset_config.total_sample_count is not None:
                    # Config has total_sample_count - pass None to let DatasetProcessor use config value
                    # This ensures we load all samples from file, then limit based on config
                    total_sample_count_for_loader = None
                    self.logger.info(f"Dataset config specifies total_sample_count={dataset_config.total_sample_count}, will load all samples and use config value")
        except Exception as e:
            # If config loading fails, fall back to using self.num_samples
            self.logger.debug(f"Could not pre-check dataset config: {e}, using num_samples={self.num_samples}")
        
        self.dataset = DatasetProcessor(
            dataset_path=self.dataset_path,
            model_name=self.model_name,
            total_sample_count=total_sample_count_for_loader,
            dataset_name=dataset_name,
            input_column=input_column,
            input_ids_column=input_ids_column,
            output_column=output_column,
            config_dir=config_dir
        )
        
        # Update max_tokens from dataset config if available
        # Note: For gpt-oss-120b, max_tokens may be test_mode-dependent
        if hasattr(self.dataset, 'dataset_config') and self.dataset.dataset_config:
            dataset_max_tokens = self.dataset.dataset_config.model_specific.get('max_tokens')
            if dataset_max_tokens is not None:
                # Check if this is a test_mode-specific value or should override
                # For gpt-oss-120b, dataset configs have different max_tokens for perf vs accuracy
                # The dataset name itself indicates which mode (perf_eval_ref vs acc_eval_ref)
                self.max_tokens = int(dataset_max_tokens)
                self.logger.info(f"Updated max_tokens from dataset config: {self.max_tokens}")
        
        # Update num_samples from dataset config if available
        # This ensures we use the correct sample count from config (e.g., 6396 for perf_eval_ref, 4395 for acc_eval_ref)
        # The dataset.total_sample_count may have been set from config, so we should use it
        if hasattr(self.dataset, 'dataset_config') and self.dataset.dataset_config:
            config_total = self.dataset.dataset_config.total_sample_count
            if config_total is not None:
                # Dataset config specified a total_sample_count - use it
                # Limit to actual dataset size
                actual_dataset_size = len(self.dataset.input_ids)
                new_num_samples = min(config_total, actual_dataset_size)
                if new_num_samples != self.num_samples:
                    old_num_samples = self.num_samples
                    self.num_samples = new_num_samples
                    self.logger.info(f"Updated num_samples from dataset config: {old_num_samples} -> {self.num_samples} (config: {config_total}, actual: {actual_dataset_size})")
                else:
                    self.logger.info(f"Using num_samples: {self.num_samples} (matches dataset config: {config_total})")
        elif self.dataset.total_sample_count is not None and self.dataset.total_sample_count != self.num_samples:
            # Fallback: use dataset's total_sample_count if it differs from our num_samples
            # This handles cases where dataset was limited during loading
            actual_dataset_size = len(self.dataset.input_ids)
            new_num_samples = min(self.dataset.total_sample_count, actual_dataset_size)
            if new_num_samples != self.num_samples:
                old_num_samples = self.num_samples
                self.num_samples = new_num_samples
                self.logger.info(f"Updated num_samples to match dataset: {old_num_samples} -> {self.num_samples} (dataset total: {self.dataset.total_sample_count}, actual: {actual_dataset_size})")
        
        # Print dataset statistics
        stats = self.dataset.get_statistics()
        self.logger.info("=" * 60)
        self.logger.info("Dataset Statistics")
        self.logger.info("=" * 60)
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)
        self.logger.info(f"Final max_tokens: {self.max_tokens}")
        self.logger.info("=" * 60)
        
        # Initialize tokenizer if using API mode (needed for vLLM detokenization)
        if self.api_server_url:
            self._initialize_tokenizer()
            
            # For vLLM: if we have input_ids but no text, detokenize them
            # This must happen before waiting for server, as we need text for vLLM API calls
            if not self.use_input_ids and self.tokenizer and self.dataset:
                if len(self.dataset.input_ids) > 0 and len(self.dataset.input) == 0:
                    self.logger.info("Detokenizing input_ids to text for vLLM (dataset has no text field)")
                    self._detokenize_dataset()
            
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
    
    def _detokenize_dataset(self):
        """Detokenize input_ids to text for vLLM when dataset has no text field."""
        if not self.tokenizer or not self.dataset:
            return
        
        if len(self.dataset.input_ids) == 0:
            return
        
        if len(self.dataset.input) > 0:
            # Already has text, no need to detokenize
            return
        
        self.logger.info(f"Detokenizing {len(self.dataset.input_ids)} samples...")
        self.dataset.input = []
        
        for i, input_ids in enumerate(self.dataset.input_ids):
            try:
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                self.dataset.input.append(text)
            except Exception as e:
                self.logger.warning(f"Error detokenizing sample {i}: {e}")
                # Fallback: convert to string representation
                self.dataset.input.append(" ".join([str(t) for t in input_ids]))
        
        self.logger.info(f"Successfully detokenized {len(self.dataset.input)} samples")
    
    def _determine_max_tokens(self, model_name: str, config: Optional[Dict[str, Any]], test_mode: str = "performance") -> int:
        """
        Determine max_tokens based on config, server_config, dataset_config, or model name.
        
        Priority order:
        1. config['max_tokens'] (explicit client config)
        2. config['server_config']['config']['max_tokens'] (server config)
        3. config['server_config']['config']['api_server_args'] with --max-model-len or --max-num-seqs
        4. dataset_config.model_specific.get('max_tokens')
        5. Model name-based defaults (test_mode-aware for gpt-oss-120b)
        6. Default: 1024
        
        Defaults:
        - deepseek-r1: 20000
        - llama3.1-8b: 128
        - llama2-70b: 1024
        - gpt-oss-120b: 10240 (performance), 32768 (accuracy)
        - default: 1024
        """
        # Priority 1: Check if explicitly set in config
        if config and 'max_tokens' in config:
            return int(config['max_tokens'])
        
        # Priority 2: Check server_config
        if config and 'server_config' in config:
            server_config = config['server_config']
            # Check in server config dict directly
            if 'max_tokens' in server_config:
                return int(server_config['max_tokens'])
            # Check in server config['config'] dict
            if 'config' in server_config and isinstance(server_config['config'], dict):
                server_config_dict = server_config['config']
                if 'max_tokens' in server_config_dict:
                    return int(server_config_dict['max_tokens'])
                # Check for max_new_tokens (alternative name)
                if 'max_new_tokens' in server_config_dict:
                    return int(server_config_dict['max_new_tokens'])
                # Check api_server_args for --max-model-len or --max-num-seqs
                if 'api_server_args' in server_config_dict:
                    args = server_config_dict['api_server_args']
                    if isinstance(args, list):
                        for i, arg in enumerate(args):
                            if arg in ['--max-model-len', '--max-num-seqs'] and i + 1 < len(args):
                                try:
                                    return int(args[i + 1])
                                except (ValueError, IndexError):
                                    pass
        
        # Priority 3: Check dataset_config (will be available after initialize)
        # This is checked later in initialize() method after dataset is loaded
        
        # Priority 4: Determine from model name (test_mode-aware for gpt-oss-120b)
        model_lower = model_name.lower()
        if 'deepseek' in model_lower and 'r1' in model_lower:
            return 20000
        elif 'gpt-oss' in model_lower or 'gpt_oss' in model_lower or 'gptoss' in model_lower:
            if '120b' in model_lower or '120-b' in model_lower:
                # gpt-oss-120b has different max_tokens for perf vs accuracy
                if test_mode == "accuracy":
                    return 32768
                else:  # performance
                    return 10240
        elif 'llama3.1' in model_lower or 'llama-3.1' in model_lower or 'llama3_1' in model_lower:
            if '8b' in model_lower or '8-b' in model_lower:
                return 128
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
        """Wait for API server(s) to become ready."""
        if self.load_balancing:
            # Wait for at least one server to be ready
            self.logger.info(f"Waiting for at least one API server to be ready from {len(self.api_server_urls)} servers (timeout: {timeout}s)")
            ready_servers = []
            
            for url in self.api_server_urls:
                health_endpoint = f"{url}/health"
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        response = requests.get(health_endpoint, timeout=10)
                        if response.status_code == 200:
                            self.logger.info(f"API server at {url} is ready!")
                            ready_servers.append(url)
                            break
                    except Exception as e:
                        self.logger.debug(f"API server at {url} not ready: {e}")
                    
                    time.sleep(2)
            
            if ready_servers:
                self.server_ready = True
                self.logger.info(f"{len(ready_servers)}/{len(self.api_server_urls)} servers are ready")
                # Remove failed servers from the list
                self.api_server_urls = [url for url in self.api_server_urls if url in ready_servers]
                if len(self.api_server_urls) != len(ready_servers):
                    self.logger.warning(f"Some servers are not ready. Using {len(self.api_server_urls)} available servers")
            else:
                raise RuntimeError(f"No API servers became ready within {timeout} seconds")
        elif self.api_server_url:
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
    
    def _get_next_server_url(self) -> str:
        """Get the next server URL using load balancing strategy."""
        if not self.load_balancing or not self.api_server_urls:
            return self.api_server_url
        
        # Filter out failed servers (if any are marked as permanently failed)
        available_servers = [url for url in self.api_server_urls if url not in self.failed_servers]
        if not available_servers:
            # If all servers failed, reset and try all again
            self.logger.warning("All servers marked as failed, resetting and trying all servers")
            self.failed_servers.clear()
            available_servers = self.api_server_urls
        
        if self.load_balance_strategy == 'round_robin':
            # Round-robin: cycle through servers
            url = available_servers[self.current_server_index % len(available_servers)]
            self.current_server_index = (self.current_server_index + 1) % len(available_servers)
            return url
        elif self.load_balance_strategy == 'random':
            # Random: pick a random server
            return random.choice(available_servers)
        else:
            # Default to round-robin
            url = available_servers[self.current_server_index % len(available_servers)]
            self.current_server_index = (self.current_server_index + 1) % len(available_servers)
            return url
    
    def _get_endpoints_for_url(self, base_url: str) -> Dict[str, str]:
        """Get endpoint URLs for a given base URL."""
        return {
            'completions': f"{base_url}/v1/completions",
            'chat_completions': f"{base_url}/v1/chat/completions",
            'health': f"{base_url}/health",
            'sglang': f"{base_url}{self.sglang_endpoint}"
        }
    
    def _send_request_with_retry(self, endpoint: str, payload: Dict[str, Any], server_url: str, max_retries: int = 3) -> requests.Response:
        """Send API request with retry logic and load balancing fallback."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, json=payload, timeout=None)
                if response.status_code == 200:
                    return response
                else:
                    # Non-200 status code - might be server error
                    last_exception = RuntimeError(f"API request failed: {response.status_code} - {response.text}")
                    if self.load_balancing and attempt < max_retries - 1:
                        # Try next server
                        self.logger.warning(f"Server {server_url} returned status {response.status_code}, trying next server...")
                        server_url = self._get_next_server_url()
                        endpoints = self._get_endpoints_for_url(server_url)
                        # Update endpoint based on original endpoint type
                        if '/chat/completions' in endpoint:
                            endpoint = endpoints['chat_completions']
                        elif '/completions' in endpoint:
                            endpoint = endpoints['completions']
                        elif '/generate' in endpoint:
                            endpoint = endpoints['sglang']
                        continue
            except (requests.exceptions.RequestException, ConnectionError) as e:
                last_exception = e
                if self.load_balancing and attempt < max_retries - 1:
                    # Try next server on connection error
                    self.logger.warning(f"Connection error to {server_url}: {e}, trying next server...")
                    server_url = self._get_next_server_url()
                    endpoints = self._get_endpoints_for_url(server_url)
                    # Update endpoint based on original endpoint type
                    if '/chat/completions' in endpoint:
                        endpoint = endpoints['chat_completions']
                    elif '/completions' in endpoint:
                        endpoint = endpoints['completions']
                    elif '/generate' in endpoint:
                        endpoint = endpoints['sglang']
                    continue
                else:
                    # No more retries or not load balancing
                    break
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"API request failed after {max_retries} attempts")
    
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
        
        # total_count should be the actual number of samples loaded in the dataset
        # This may be limited by config's total_sample_count or by what was passed to DatasetProcessor
        actual_dataset_size = len(self.dataset.input_ids)
        
        # If dataset config specified total_sample_count, use that for total_count
        # Otherwise use actual dataset size
        if hasattr(self.dataset, 'dataset_config') and self.dataset.dataset_config:
            config_total = self.dataset.dataset_config.total_sample_count
            if config_total is not None:
                # Use config's total_sample_count, but don't exceed actual dataset size
                total_count = min(config_total, actual_dataset_size)
                self.logger.info(f"Using dataset config total_sample_count={config_total} for QSL total_count (actual dataset size: {actual_dataset_size})")
            else:
                total_count = actual_dataset_size
        else:
            total_count = actual_dataset_size
        
        # performance_count should be the number of samples to use for testing
        # This is self.num_samples (which may have been updated from config)
        performance_count = min(self.num_samples, total_count)
        
        self.logger.info(f"Constructing QSL: total_count={total_count}, performance_count={performance_count}, num_samples={self.num_samples}, actual_dataset_size={actual_dataset_size}")
        
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
        
        # Determine sampling parameters based on test_mode
        temperature, top_k, top_p = self._get_sampling_params()
        
        if self.offline_back_to_back:
            # Send requests back-to-back (one at a time)
            self.logger.info("=" * 80)
            self.logger.info(f"OFFLINE SCENARIO: Processing {total_samples} queries back-to-back")
            self.logger.info(f"Total samples: {total_samples}")
            self.logger.info("=" * 80)
            
            processed_samples = 0
            for q_sample in query_samples:
                try:
                    if self.api_server_url:
                        self._process_api_single(q_sample, temperature, top_k, top_p)
                        processed_samples += 1
                        if processed_samples % 100 == 0:
                            self.logger.info(f"Processed {processed_samples}/{total_samples} samples")
                    else:
                        self.logger.warning("Local model processing not yet implemented")
                        self._send_error_responses([q_sample])
                        processed_samples += 1
                except Exception as e:
                    self.logger.error(f"Error processing query {q_sample.id}: {e}", exc_info=True)
                    self._send_error_responses([q_sample])
                    processed_samples += 1
            
            self.logger.info("=" * 80)
            self.logger.info(f"All queries completed: {processed_samples}/{total_samples} samples processed")
            self.logger.info("=" * 80)
        else:
            # Batch processing (original behavior)
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
                        self._process_api_batch(batch, temperature, top_k, top_p)
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
    
    def _get_sampling_params(self):
        """Get sampling parameters based on test_mode."""
        if self.test_mode == "accuracy":
            temperature = self.accuracy_temperature if self.accuracy_temperature is not None else self.temperature
            top_k = self.accuracy_top_k if self.accuracy_top_k is not None else self.top_k
            top_p = self.accuracy_top_p if self.accuracy_top_p is not None else self.top_p
        else:
            temperature = self.temperature
            top_k = self.top_k
            top_p = self.top_p
        return temperature, top_k, top_p
    
    def _process_api_single(self, q_sample: 'lg.QuerySample', temperature: float, top_k: int, top_p: float) -> None:
        """Process a single query via API (for back-to-back mode)."""
        # Check if using multimodal messages format
        if self.use_messages or self.multimodal:
            # Get messages from dataset (multimodal format)
            if hasattr(self.dataset, 'messages') and q_sample.index < len(self.dataset.messages):
                messages = self.dataset.messages[q_sample.index]
            elif hasattr(self.dataset, 'processed_data'):
                # Try to extract messages from dataset
                df = self.dataset.processed_data
                if 'messages' in df.columns:
                    messages = df.iloc[q_sample.index]['messages']
                else:
                    # Fallback: try to construct messages from available fields
                    self.logger.warning(f"Dataset doesn't have 'messages' column, attempting to construct from fields")
                    messages = self._construct_messages_from_dataset(q_sample.index)
            else:
                raise RuntimeError(f"Multimodal mode requires 'messages' in dataset, but not found for index {q_sample.index}")
            
            # Send multimodal request
            self._process_multimodal_request(q_sample.id, q_sample.index, messages, temperature, top_k, top_p)
            return
        
        # Get input IDs from dataset
        input_ids = self.dataset.input_ids[q_sample.index]
        
        # Check if using SGLang with input_ids
        if self.use_input_ids:
            # SGLang format: send input_ids directly
            # Get server URL (with load balancing if enabled)
            server_url = self._get_next_server_url()
            endpoint = f"{server_url}{self.sglang_endpoint}"
            api_payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                }
            }
            
            self.logger.debug(f"Sending SGLang request to {endpoint} for query {q_sample.id}")
            response = self._send_request_with_retry(endpoint, api_payload, server_url)
            
            api_result = response.json()
            output_ids = api_result.get("output_ids", [])
            output_text = api_result.get("text", "")
            
            # Process response
            self._process_sglang_response(q_sample.id, q_sample.index, output_ids, output_text)
        else:
            # Standard format: decode to text
            if self.tokenizer:
                try:
                    text_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                except Exception as e:
                    self.logger.warning(f"Error decoding tokens: {e}")
                    text_prompt = " ".join([str(t) for t in input_ids])
            else:
                text_prompt = " ".join([str(t) for t in input_ids])
            
            # Determine endpoint based on endpoint_type
            if self.endpoint_type == 'chat_completions':
                endpoint = endpoints['chat_completions']
                api_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": text_prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False
                }
            else:
                endpoint = endpoints['completions']
                api_payload = {
                    "model": self.model_name,
                    "prompt": text_prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stream": False
                }
            
            self.logger.debug(f"Sending API request to {endpoint} for query {q_sample.id}")
            response = self._send_request_with_retry(endpoint, api_payload, server_url)
            
            api_result = response.json()
            
            # Extract response
            if self.endpoint_type == 'chat_completions':
                text_response = api_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                text_response = api_result.get("choices", [{}])[0].get("text", "")
            
            # Convert to token IDs
            if self.tokenizer:
                try:
                    token_ids = self.tokenizer.encode(text_response, add_special_tokens=False)
                except Exception as e:
                    self.logger.warning(f"Error encoding response: {e}")
                    token_ids = []
            else:
                token_ids = []
            
            # Process response
            self._process_single_response(q_sample.id, q_sample.index, token_ids, text_response, text_prompt)
    
    def _process_api_batch(self, batch: List['lg.QuerySample'], temperature: float, top_k: int, top_p: float) -> None:
        """Process a batch via API."""
        batch_size = len(batch)
        self.logger.debug(f"Processing API batch with {batch_size} samples")
        
        # Check if using multimodal messages format
        if self.use_messages or self.multimodal:
            # Multimodal: send each request individually (messages format doesn't support batching easily)
            for q_sample in batch:
                self._process_api_single(q_sample, temperature, top_k, top_p)
            return
        
        # Check if using SGLang with input_ids
        if self.use_input_ids:
            # SGLang format: send each request individually (SGLang handles batching internally)
            for q_sample in batch:
                self._process_api_single(q_sample, temperature, top_k, top_p)
            return
        
        # Standard format: prepare text prompts
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
        
        # Get server URL (with load balancing if enabled)
        server_url = self._get_next_server_url()
        endpoints = self._get_endpoints_for_url(server_url)
        
        # Determine endpoint based on endpoint_type
        if self.endpoint_type == 'chat_completions':
            endpoint = endpoints['chat_completions']
            # Format for chat completions API - handle batch requests
            # For chat completions, we need to send each prompt separately or use array format
            # Most APIs support array format for batch processing
            if len(text_prompts) == 1:
                # Single prompt
                api_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": text_prompts[0]}],
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False
                }
            else:
                # Batch: send as array of messages arrays
                # Note: Some APIs may require separate requests for batch
                api_payload = {
                    "model": self.model_name,
                    "messages": [[{"role": "user", "content": prompt}] for prompt in text_prompts],
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False
                }
        else:
            endpoint = endpoints['completions']
            # Format for completions API
            api_payload = {
                "model": self.model_name,
                "prompt": text_prompts,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stream": False
            }
        
        self.logger.debug(f"Sending API batch request to {endpoint} with {len(text_prompts)} prompts")
        response = self._send_request_with_retry(endpoint, api_payload, server_url)
        
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
    
    def _process_sglang_response(self, query_id: int, query_index: int, output_ids: List[int], output_text: str) -> None:
        """Process SGLang response (already has token IDs)."""
        token_count = len(output_ids)
        
        # Debug mode: print query, text response and token count in accuracy mode
        if self.debug_mode and self.test_mode == "accuracy":
            # Get prompt text if available
            prompt_text = None
            if hasattr(self, 'dataset') and self.dataset:
                if query_index < len(self.dataset.input) and self.dataset.input[query_index]:
                    prompt_text = self.dataset.input[query_index]
                elif self.tokenizer and query_index < len(self.dataset.input_ids):
                    try:
                        prompt_text = self.tokenizer.decode(self.dataset.input_ids[query_index], skip_special_tokens=True)
                    except:
                        prompt_text = f"[Token IDs: {self.dataset.input_ids[query_index][:50]}...]"
            
            text_preview = output_text[:200] + "..." if len(output_text) > 200 else output_text
            prompt_preview = prompt_text[:200] + "..." if prompt_text and len(prompt_text) > 200 else (prompt_text or "N/A")
            self.logger.info(f"[DEBUG] Query {query_id} (index {query_index}):")
            if prompt_text:
                self.logger.info(f"  Prompt: {prompt_preview}")
            self.logger.info(f"  Text Response: {text_preview}")
            self.logger.info(f"  Total Tokens: {token_count}")
        
        # Create Loadgen response
        token_array = np.array(output_ids, dtype=np.int32)
        token_bytes = token_array.tobytes()
        response_data = token_array.ctypes.data
        response_size = len(token_bytes)
        
        response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
        lg.QuerySamplesComplete([response])
        self.logger.debug(f"Query {query_id} (index {query_index}): {token_count} tokens")
    
    def _process_single_response(self, query_id: int, query_index: int, token_ids: List[int], text_response: str, text_prompt: Optional[str] = None) -> None:
        """Process a single response."""
        token_count = len(token_ids)
        
        # Debug mode: print query, text response and token count in accuracy mode
        if self.debug_mode and self.test_mode == "accuracy":
            query_preview = text_prompt[:200] + "..." if text_prompt and len(text_prompt) > 200 else (text_prompt or "N/A")
            text_preview = text_response[:200] + "..." if len(text_response) > 200 else text_response
            self.logger.info(f"[DEBUG] Query {query_id} (index {query_index}):")
            self.logger.info(f"  Prompt: {query_preview}")
            self.logger.info(f"  Text Response: {text_preview}")
            self.logger.info(f"  Total Tokens: {token_count}")
        
        # Create Loadgen response
        token_array = np.array(token_ids, dtype=np.int32)
        token_bytes = token_array.tobytes()
        response_data = token_array.ctypes.data
        response_size = len(token_bytes)
        
        response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
        lg.QuerySamplesComplete([response])
        self.logger.debug(f"Query {query_id} (index {query_index}): {token_count} tokens")
    
    def _construct_messages_from_dataset(self, index: int) -> List[Dict[str, Any]]:
        """Construct messages format from dataset fields (fallback for multimodal)."""
        # This is a fallback - ideally the dataset should already have 'messages' column
        if not PANDAS_AVAILABLE or not hasattr(self.dataset, 'processed_data'):
            raise RuntimeError("Pandas not available or dataset not processed")
        
        df = self.dataset.processed_data
        row = df.iloc[index]
        
        # Try to construct messages from common fields
        # This is a simplified version - actual implementation should match the task's formulate_loaded_sample
        messages = []
        
        # Add system message if available
        if 'system_message' in row:
            messages.append({
                "role": "system",
                "content": str(row['system_message'])
            })
        elif 'system' in row:
            messages.append({
                "role": "system",
                "content": str(row['system'])
            })
        
        # Add user message with content
        user_content = []
        
        # Add text content
        text_fields = ['text', 'prompt', 'input', 'product_title', 'product_description']
        text_content = ""
        for field in text_fields:
            if field in row and pd.notna(row[field]):
                if text_content:
                    text_content += "\n\n"
                text_content += str(row[field])
        
        if text_content:
            user_content.append({
                "type": "text",
                "text": text_content
            })
        
        # Add image content if available
        if 'product_image' in row and pd.notna(row['product_image']):
            try:
                from PIL import Image
                image = row['product_image']
                if isinstance(image, Image.Image):
                    image_file = BytesIO()
                    image_format = image.format or 'PNG'
                    image.save(image_file, format=image_format)
                    image_bytes = image_file.getvalue()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format.lower()};base64,{image_base64}"
                        }
                    })
            except Exception as e:
                self.logger.warning(f"Could not process image for index {index}: {e}")
        
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content if len(user_content) > 1 else user_content[0].get("text", "")
            })
        
        return messages
    
    def _process_multimodal_request(self, query_id: int, query_index: int, messages: List[Dict[str, Any]], 
                                   temperature: float, top_k: int, top_p: float) -> None:
        """Process a multimodal request with messages format."""
        # Get server URL (with load balancing if enabled)
        server_url = self._get_next_server_url()
        endpoints = self._get_endpoints_for_url(server_url)
        endpoint = endpoints['chat_completions']
        
        # Build API payload
        api_payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        # Add top_k if specified (via extra_body for vLLM)
        if top_k is not None and top_k > 0:
            api_payload["extra_body"] = {"top_k": top_k}
        
        # Add response_format if guided decoding is enabled
        if self.use_guided_decoding:
            # Note: response_format should be provided in the dataset or config
            # For now, we'll skip it if not available
            if hasattr(self, 'response_format') and self.response_format:
                api_payload["response_format"] = self.response_format
        
        self.logger.debug(f"Sending multimodal request to {endpoint} for query {query_id}")
        response = self._send_request_with_retry(endpoint, api_payload, server_url)
        
        api_result = response.json()
        
        # Extract text response
        content = api_result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content is None:
            content = ""
        
        # Get token count from usage if available
        usage = api_result.get("usage", {})
        token_count = usage.get("completion_tokens", 0)
        
        # If no token count, estimate from text length
        if token_count == 0 and self.tokenizer:
            try:
                token_ids = self.tokenizer.encode(content, add_special_tokens=False)
                token_count = len(token_ids)
            except Exception as e:
                self.logger.warning(f"Error encoding response for token count: {e}")
                token_count = len(content.split())  # Fallback: word count
        
        # Debug mode: print prompt (messages), text response and token count in accuracy mode
        if self.debug_mode and self.test_mode == "accuracy":
            messages_str = json.dumps(messages, indent=2)
            messages_preview = messages_str[:200] + "..." if len(messages_str) > 200 else messages_str
            text_preview = content[:200] + "..." if len(content) > 200 else content
            self.logger.info(f"[DEBUG] Query {query_id} (index {query_index}):")
            self.logger.info(f"  Prompt (Messages): {messages_preview}")
            self.logger.info(f"  Text Response: {text_preview}")
            self.logger.info(f"  Total Tokens: {token_count}")
        
        # Create Loadgen response (text content as bytes)
        bytes_array = array.array("B", content.encode("utf-8"))
        address, length = bytes_array.buffer_info()
        size_in_bytes = length * bytes_array.itemsize
        
        response = lg.QuerySampleResponse(query_id, address, size_in_bytes, token_count)
        lg.QuerySamplesComplete([response])
        self.logger.debug(f"Query {query_id} (index {query_index}): {token_count} tokens")
    
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
        
        # Get server URL (with load balancing if enabled)
        server_url = self._get_next_server_url()
        endpoints = self._get_endpoints_for_url(server_url)
        
        # Determine endpoint and payload based on endpoint_type
        if self.endpoint_type == 'chat_completions':
            endpoint_url = endpoints['chat_completions']
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
            endpoint_url = endpoints['completions']
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

