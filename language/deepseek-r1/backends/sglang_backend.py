"""
SGLang backend for DeepSeek model inference.

This backend starts an SGLang server as a subprocess and communicates with it
using the OpenAI-compatible API.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

import httpx
import numpy as np
import requests
import torch
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from .base_backend import BaseBackend, StreamingChunk
from .utils import (
    find_free_port,
    get_cache_directory,
    set_all_seeds,
    setup_huggingface_cache,
    TerminalDisplay,
    LogMonitor
)
from utils.backend_registry import get_backend_config, apply_backend_env_vars
from utils.validation import require_initialized, validate_prompts_input


class SGLangBackend(BaseBackend):
    """SGLang backend with server management and OpenAI-compatible API."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize SGLang backend with configuration from registry."""
        super().__init__()
        # Get configuration from registry
        self.config = get_backend_config('sglang')
        # Allow override with passed config
        if config:
            self.config.update(config)

        # Dynamic port allocation to avoid conflicts
        self.port = find_free_port(30000)
        self.config['port'] = self.port

        # Server process management
        self.server_process = None
        self.server_log_file = None

        # Client objects
        self.client = None
        self.async_client = None

        # Tokenizer for string conversion
        self.tokenizer = None

        # Log monitoring
        self._log_monitor = None

        # Shared semaphore for async concurrency control
        self._async_semaphore = None

        # Configure logging to suppress httpx INFO logs (only show
        # warnings/errors)
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables and cache directories."""
        # Use the utility function to get cache directory
        cache_base = get_cache_directory()

        # Set up HuggingFace cache environment variables
        setup_huggingface_cache()

        # Apply backend-specific environment variables from registry
        apply_backend_env_vars('sglang')

        # Set seeds for reproducibility
        seed = self.config['random_seed']
        set_all_seeds(seed)

    def _check_server_alive(self) -> None:
        """Check if server process is still alive, raise RuntimeError if not."""
        if self.server_process and self.server_process.poll() is not None:
            exit_code = self.server_process.returncode
            raise RuntimeError(
                f"SGLang server process has died with exit code: {exit_code}. "
                f"Check server logs at: {self.server_log_file}"
            )

    def _build_server_command(self) -> List[str]:
        """Build the SGLang server startup command."""
        cmd = [
            sys.executable, '-m', 'sglang.launch_server',
            '--model-path', self.config['model'],
            '--revision', self.config['model_revision'],
            '--port', str(self.port),
            '--host', self.config['host'],
            '--tp', str(self.config['tensor_parallel_size']),
            '--context-length', str(self.config['context_length']),
            '--mem-fraction-static', str(self.config['mem_fraction_static']),
            '--random-seed', str(self.config['random_seed']),
            '--trust-remote-code',
            '--dtype', self.config['dtype'],
            '--served-model-name', self.config['served_model_name'],
        ]

        # Add optimization flags
        if self.config['enable_torch_compile']:
            cmd.append('--enable-torch-compile')

        if self.config['enable_flashinfer']:
            cmd.append('--attention-backend')
            cmd.append('flashinfer')

        if self.config['enable_dp_attention']:
            cmd.extend(['--enable-dp-attention',
                       '--dp', str(self.config['dp'])])

        # Add performance settings
        cmd.extend([
            '--cuda-graph-max-bs', str(self.config['cuda_graph_max_bs']),
            '--max-running-requests', str(self.config['max_running_requests'])
        ])

        return cmd

    def _wait_for_server_ready(self, timeout: int = None) -> bool:
        """Wait for SGLang server to become available with real-time log monitoring."""
        if timeout is None:
            timeout = self.config['server_startup_timeout']

        start_time = time.time()
        health_url = f"http://localhost:{self.port}/health"
        check_interval = self.config['health_check_interval']

        print(f"\n[SGLANG] Starting server on port {self.port}...")
        print(f"[SGLANG] Server startup timeout: {timeout//60} minutes")
        print(f"[SGLANG] Health check URL: {health_url}")

        # Start log monitoring if log file exists
        if self.server_log_file:
            self._log_monitor = LogMonitor(
                log_file_path=self.server_log_file,
                prefix="SGLANG",
                max_lines=5,
                display_interval=1.0
            )
            # Start monitor and wait for it to set up display area
            self._log_monitor.start(wait_for_file=True, file_wait_timeout=30.0)

        # Reserve a line for progress indicator
        print()  # Empty line for progress

        last_check_time = 0
        last_progress_update = 0
        progress_idx = 0

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            # Update progress indicator every 0.5 seconds
            if time.time() - last_progress_update >= 0.5:
                last_progress_update = time.time()
                progress_idx = (
                    progress_idx + 1) % len(TerminalDisplay.PROGRESS_CHARS)
                minutes = elapsed // 60
                seconds = elapsed % 60
                # Use carriage return to stay on the same line
                progress_msg = f"[SGLANG] {TerminalDisplay.PROGRESS_CHARS[progress_idx]} Waiting for server... ({minutes}m {seconds}s elapsed)"
                # Pad with spaces to clear any previous longer text
                print(f"\r{progress_msg:<80}", end='', flush=True)

            # Run health checks every check_interval seconds
            if elapsed >= last_check_time + check_interval:
                last_check_time = elapsed

                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        # Health check passed, now try a warmup query
                        print(f"\r{' '*80}\r", end='', flush=True)
                        print(
                            f"\n[SGLANG] Health check passed, running warmup query...")

                        # Try to send a simple warmup query using OpenAI client
                        try:
                            base_url = f"http://localhost:{self.port}/v1"
                            api_key = self.config['api_key'] or "dummy-key"

                            # Create a temporary client for warmup
                            warmup_client = OpenAI(
                                base_url=base_url,
                                api_key=api_key,
                                timeout=httpx.Timeout(timeout=30.0),
                                max_retries=3
                            )

                            # Send a simple warmup request
                            warmup_response = warmup_client.chat.completions.create(
                                model=self.config['served_model_name'],
                                messages=[
                                    {"role": "user", "content": "Hello"}],
                                temperature=0.0,
                                max_tokens=10,
                                seed=self.config['seed']
                            )

                            # Check if we got a valid response
                            if warmup_response.choices[0].message.content:
                                print(
                                    f"[SGLANG] ✓ Warmup query successful! Response: {warmup_response.choices[0].message.content[:50]}...")

                                # Stop log monitoring
                                if self._log_monitor:
                                    self._log_monitor.stop()
                                    self._log_monitor = None

                                print(f"\n[SGLANG] " + "=" * 60)
                                print(
                                    f"[SGLANG] ✓ SERVER READY! (startup took {elapsed}s)")
                                print(f"[SGLANG] " + "=" * 60)
                                return True
                            else:
                                print(
                                    f"[SGLANG] Warmup query returned empty response, retrying...")

                        except Exception as warmup_error:
                            print(
                                f"[SGLANG] Warmup query failed: {warmup_error}, retrying...")
                            # Continue waiting, the server might not be fully
                            # ready yet

                except requests.exceptions.RequestException:
                    pass

            # Check if server process is still alive
            if self.server_process and self.server_process.poll() is not None:
                if self._log_monitor:
                    self._log_monitor.stop()
                    self._log_monitor = None
                # Clear progress line
                print(f"\r{' '*80}\r", end='', flush=True)
                print(
                    f"\n[SGLANG] ✗ Server process died with exit code: {self.server_process.returncode}")
                if self.server_log_file:
                    print(
                        f"[SGLANG] Check server logs at: {self.server_log_file}")
                return False

            time.sleep(0.1)  # Check every 100ms for smoother progress

        # Timeout reached
        if self._log_monitor:
            self._log_monitor.stop()
            self._log_monitor = None
        # Clear progress line
        print(f"\r{' '*80}\r", end='', flush=True)
        print(f"\n[SGLANG] ✗ Server failed to start within {timeout} seconds")
        return False

    def _start_server(self) -> None:
        """Start the SGLang server as a subprocess."""
        print(
            f"\n[SGLANG] Starting SGLang server for {self.config['model']}...")
        print(f"[SGLANG] Configuration:")
        print(f"[SGLANG]   - Port: {self.port}")
        print(
            f"[SGLANG]   - Tensor Parallel: {self.config['tensor_parallel_size']}")
        print(
            f"[SGLANG]   - Context Length: {self.config['context_length']:,} tokens")
        print(f"[SGLANG]   - dtype: {self.config['dtype']}")

        # Create log file for server output
        log_dir = Path("/work/logs")
        log_dir.mkdir(exist_ok=True)
        self.server_log_file = log_dir / \
            f"sglang_server_{self.port}_{int(time.time())}.log"

        cmd = self._build_server_command()
        print(f"\n[SGLANG] Command: {' '.join(cmd)}")
        print(f"[SGLANG] Server logs: {self.server_log_file}")

        # Start server process
        with open(self.server_log_file, 'w') as log_file:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )

        # Wait for server to be ready (with log monitoring)
        if not self._wait_for_server_ready():
            self._stop_server()
            raise RuntimeError("Failed to start SGLang server")

    def _stop_server(self) -> None:
        """Stop the SGLang server gracefully."""
        # Stop log monitoring first
        if self._log_monitor:
            self._log_monitor.stop()
            self._log_monitor = None

        if self.server_process:
            print("\n[SGLANG] Stopping SGLang server...")

            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    self.server_process.wait(timeout=30)
                    print("[SGLANG] Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if not stopped
                    print("[SGLANG] Server didn't stop gracefully, forcing...")
                    os.killpg(
                        os.getpgid(
                            self.server_process.pid),
                        signal.SIGKILL)
                    self.server_process.wait()
                    print("[SGLANG] Server force stopped")
            except ProcessLookupError:
                # Process already dead
                pass

            self.server_process = None

    def initialize(self) -> None:
        """Initialize the SGLang backend by starting server and setting up clients."""
        if self.is_initialized:
            return

        try:
            # Load tokenizer for string conversion
            print(f"[SGLANG] Loading tokenizer: {self.config['tokenizer']}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['tokenizer'], revision=self.config['model_revision'])

            # Start SGLang server (with log monitoring)
            self._start_server()

            # Initialize OpenAI clients
            base_url = f"http://localhost:{self.port}/v1"
            api_key = self.config['api_key'] or "dummy-key"

            print(
                f"[SGLANG] Creating OpenAI clients with base URL: {base_url}")

            # Configure timeout settings
            timeout_config = httpx.Timeout(
                timeout=self.config['request_timeout'],
                connect=30.0,
                read=None,
                write=None,
                pool=None
            )

            print(f"[SGLANG] Timeout configuration: {timeout_config}")

            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_config,
                max_retries=10  # Use 10 retries as requested
            )

            print(f"[SGLANG] Created synchronous OpenAI client")

            self.async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_config,
                max_retries=10  # Use 10 retries as requested
            )

            print(f"[SGLANG] Created asynchronous OpenAI client")

            # Create shared semaphore for async concurrency control
            self._async_semaphore = asyncio.Semaphore(
                self.config['max_running_requests'])
            print(
                f"[SGLANG] Created async semaphore with limit: {self.config['max_running_requests']}")

            # Server readiness was already verified by health endpoint in _wait_for_server_ready()
            # No need to check models endpoint

            # Only set initialized to True if we have valid clients
            self.is_initialized = True
            print("[SGLANG] Backend initialized successfully!")

        except Exception as e:
            # Clean up on failure
            print(f"[SGLANG] Initialization failed: {e}")

            # Clear any partially initialized state
            self.client = None
            self.async_client = None
            self.tokenizer = None
            self._async_semaphore = None

            # Stop server if it was started
            self._stop_server()

            # Ensure is_initialized is False
            self.is_initialized = False

            # Re-raise the exception
            raise

    @require_initialized
    def generate(self,
                 tokenized_prompts: Optional[List[List[int]]] = None,
                 text_prompts: Optional[List[str]] = None,
                 **kwargs) -> List[Dict[str, Any]]:
        """Generate responses synchronously."""
        # Check if server process is still alive
        self._check_server_alive()

        # Check if client is properly initialized
        if self.client is None:
            raise RuntimeError(
                "SGLang client is not initialized. Server may have failed to start.")

        # Validate prompts using centralized validation
        validate_prompts_input(
            backend_name='sglang',
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        # SGLang prefers text prompts
        if text_prompts is None:
            # Convert tokenized prompts to strings
            prompt_strings = [
                self.tokenizer.decode(tokens, skip_special_tokens=False)
                for tokens in tokenized_prompts
            ]
        else:
            prompt_strings = text_prompts

        results = []

        # Process prompts with progress bar
        for prompt in tqdm(
                prompt_strings, desc="SGLang sync inference", unit="prompt"):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config['served_model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    max_tokens=self.config['max_tokens'],
                    seed=self.config['seed'],
                )

                # Get generated text
                generated_text = completion.choices[0].message.content

                # Validate response is not empty
                if not generated_text:
                    raise RuntimeError(
                        f"Empty response received from SGLang server for prompt: {prompt[:100]}...")

                # Tokenize the output to get token IDs
                tokens = self.tokenizer.encode(generated_text)

                results.append({
                    'tokens': tokens,
                    'text': generated_text
                })

            except Exception as e:
                print(f"\nError generating completion: {e}")
                raise RuntimeError(
                    f"SGLang backend failed to generate tokens for prompt: {prompt[:100]}...")

        return results

    async def _async_generate_single(
            self, prompt: str, idx: int, semaphore: asyncio.Semaphore) -> Tuple[int, Dict[str, Any]]:
        """Generate a single response asynchronously with semaphore control."""
        # Check if async client is properly initialized
        if self.async_client is None:
            raise RuntimeError(
                f"SGLang async client is not initialized for prompt {idx}")

        async with semaphore:
            try:
                completion = await self.async_client.chat.completions.create(
                    model=self.config['served_model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    max_tokens=self.config['max_tokens'],
                    seed=self.config['seed'],
                )

                # Get generated text
                generated_text = completion.choices[0].message.content

                # Validate response is not empty
                if not generated_text:
                    raise RuntimeError(
                        f"Empty response received from SGLang server for prompt: {prompt[:100]}...")

                # Tokenize the output to get token IDs
                tokens = self.tokenizer.encode(generated_text)

                return idx, {'tokens': tokens, 'text': generated_text}

            except Exception as e:
                print(f"\nError generating completion for prompt {idx}: {e}")
                raise RuntimeError(
                    f"SGLang backend failed to generate tokens for prompt {idx}: {e}")

    @require_initialized
    def generate_async(self,
                       tokenized_prompts: Optional[List[List[int]]] = None,
                       text_prompts: Optional[List[str]] = None,
                       **kwargs) -> List[asyncio.Future]:
        """Generate responses asynchronously using shared semaphore."""
        # Check if server process is still alive
        self._check_server_alive()

        # Check if client is properly initialized
        if self.async_client is None:
            raise RuntimeError(
                "SGLang async client is not initialized. Server may have failed to start.")

        # Validate prompts using centralized validation
        validate_prompts_input(
            backend_name='sglang',
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        # SGLang prefers text prompts
        if text_prompts is None:
            # Convert tokenized prompts to strings
            prompt_strings = [
                self.tokenizer.decode(tokens, skip_special_tokens=False)
                for tokens in tokenized_prompts
            ]
        else:
            prompt_strings = text_prompts

        # Get the current event loop or create one
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        futures = []
        for idx, prompt in enumerate(prompt_strings):
            # Create a task for each prompt using the shared semaphore
            task = asyncio.create_task(
                self._async_generate_single(
                    prompt, idx, self._async_semaphore))

            # Create a future that will hold the result
            future = asyncio.Future()

            # Setup callback to extract just the result (not the index)
            def make_callback(future_obj, expected_idx):
                def callback(task_obj):
                    try:
                        idx, result = task_obj.result()
                        if idx != expected_idx:
                            future_obj.set_exception(
                                Exception(f"Index mismatch: expected {expected_idx}, got {idx}"))
                        else:
                            future_obj.set_result(result)
                    except Exception as e:
                        future_obj.set_exception(e)
                return callback

            task.add_done_callback(make_callback(future, idx))
            futures.append(future)

        return futures

    async def generate_stream(self,
                              tokenized_prompts: Optional[List[List[int]]] = None,
                              text_prompts: Optional[List[str]] = None,
                              **kwargs) -> List[AsyncIterator[StreamingChunk]]:
        """Generate responses for a list of prompts with streaming."""
        if not self.is_initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        # Check if server process is still alive
        self._check_server_alive()

        # Check if async client is properly initialized
        if self.async_client is None:
            raise RuntimeError(
                "SGLang async client is not initialized. Server may have failed to start.")

        # Validate prompts
        validate_prompts_input(
            backend_name='sglang',
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        # SGLang prefers text prompts
        if text_prompts is None:
            # Convert tokenized prompts to strings
            prompt_strings = [
                self.tokenizer.decode(tokens, skip_special_tokens=False)
                for tokens in tokenized_prompts
            ]
        else:
            prompt_strings = text_prompts

        async def stream_single_prompt(
                prompt: str) -> AsyncIterator[StreamingChunk]:
            try:
                stream = await self.async_client.chat.completions.create(
                    model=self.config['served_model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.get('temperature'),
                    top_p=self.config.get('top_p'),
                    max_tokens=self.config.get('max_tokens'),
                    seed=self.config.get('seed'),
                    stream=True
                )

                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason

                    if delta.content:
                        yield StreamingChunk(
                            token=delta.content,
                            token_ids=[],  # SGLang doesn't provide token IDs in streaming
                            is_finished=finish_reason is not None,
                            finish_reason=finish_reason
                        )
                    elif finish_reason:
                        # Final chunk with no content but finish reason
                        yield StreamingChunk(
                            token="",
                            token_ids=[],
                            is_finished=True,
                            finish_reason=finish_reason
                        )
            except Exception as e:
                print(f"[SGLANG] Streaming error for prompt: {e}")
                raise

        return [stream_single_prompt(prompt) for prompt in prompt_strings]

    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        print("[SGLANG] Shutting down SGLang backend...")

        # Stop log monitoring
        if self._log_monitor:
            self._log_monitor.stop()
            self._log_monitor = None

        # Close clients
        self.client = None
        self.async_client = None

        # Clear async semaphore
        self._async_semaphore = None

        # Stop server
        self._stop_server()

        # Clear tokenizer
        self.tokenizer = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False
        print("[SGLANG] Backend shutdown complete")
