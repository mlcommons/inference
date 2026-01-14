#!/usr/bin/env python3
"""vLLM backend implementation for gpt-oss."""

import asyncio
import json
import logging
import requests
import time
from typing import List, Dict, Any, Optional, AsyncIterator
import aiohttp
from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class VLLMBackend(BaseBackend):
    """vLLM inference backend using HTTP API (OpenAI-compatible).

    Connects to a vLLM server running the gpt-oss model.
    Uses text_input field for input instead of token IDs.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: int = 1200,
        max_pool_size: int = 2000,  # Default pool size for high concurrency
        tokenizer=None,  # Tokenizer for converting text to token IDs
        **kwargs
    ):
        """Initialize vLLM backend.

        Args:
            server_url: URL of the vLLM server (default: http://localhost:8000)
            timeout: Request timeout in seconds
            max_pool_size: Maximum connection pool size (should be >= max_concurrency)
            tokenizer: Tokenizer instance for converting output text to token IDs (required for accuracy mode)
            **kwargs: Additional configuration
        """
        config = {
            "server_url": server_url,
            "timeout": timeout,
            "max_pool_size": max_pool_size,
            "tokenizer": tokenizer,
            **kwargs
        }
        super().__init__(config)
        self.server_url = server_url
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.tokenizer = tokenizer
        self.session = None

    def initialize(self) -> None:
        """Initialize connection to vLLM server."""
        if self.initialized:
            logger.warning("Backend already initialized")
            return

        logger.info(f"Connecting to vLLM server at {self.server_url}")
        logger.info(
            f"Configuring connection pool with max_pool_size={self.max_pool_size}")
        # Create session with larger connection pool for high concurrency
        # Default pool size is 10, but we may have 100s-1000s of concurrent
        # requests
        self.session = requests.Session()

        # Increase connection pool size to support high concurrency
        # pool_maxsize should be >= max_concurrency to avoid "pool is full"
        # warnings
        adapter = requests.adapters.HTTPAdapter(
            # Number of connection pools to cache
            pool_connections=min(100, self.max_pool_size // 10),
            pool_maxsize=self.max_pool_size,     # Maximum number of connections in the pool
            max_retries=3,                       # Retry failed requests
            # Don't block when pool is full, create new connections
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # Test connection with a simple request
        try:
            test_response = self._send_request(
                prompt="test",
                max_tokens=5,
                temperature=1.0,
                top_k=1,
                top_p=1.0
            )
            if "error" in test_response:
                raise ConnectionError(
                    f"Failed to connect to vLLM server: {test_response['error']}"
                )
            logger.info("Successfully connected to vLLM server")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize vLLM backend: {e}")
            raise

    def _send_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Dict[str, Any]:
        """Send a single request to the vLLM server.

        Args:
            prompt: Text prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Returns:
            Response dictionary from the server
        """
        # vLLM uses OpenAI-compatible API format
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k if top_k > 0 else -1,  # -1 means no top_k filtering
            "top_p": top_p,
            "stop": []  # No stop sequences by default
        }

        # Log sampling parameters before sending request (debug mode)
        logger.debug(
            f"Sending request with sampling parameters: "
            f"max_tokens={max_tokens}, "
            f"temperature={temperature}, "
            f"top_k={top_k if top_k > 0 else -1}, "
            f"top_p={top_p}, "
            f"prompt_length={len(prompt)} characters"
        )

        try:
            # Try /v1/completions endpoint (OpenAI-compatible)
            response = self.session.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Request failed with status {response.status_code}: {response.text}"
                )
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}

    def generate(
        self,
        prompts: List[str],  # Changed from List[List[int]] to List[str] for text input
        max_tokens: int = 100,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of text prompts.

        Args:
            prompts: List of text prompt strings (not token IDs)
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional parameters (ignored)

        Returns:
            List of response dictionaries with keys:
                - output_ids: List of generated token IDs (empty for vLLM, as we don't get token IDs)
                - output_text: Generated text
                - metadata: Additional metadata (latencies, etc.)
        """
        if not self.initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        results = []
        for prompt_text in prompts:
            start_time = time.time()
            response = self._send_request(
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            end_time = time.time()
            latency = end_time - start_time

            # Extract output from OpenAI-compatible response format
            output_text = ""
            output_ids = []  # Will be populated by tokenizing output_text
            completion_tokens = 0
            
            if "error" not in response:
                # OpenAI-compatible format: {"choices": [{"text": "...", ...}], ...}
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    output_text = choice.get("text", "")
                    
                    # Extract usage info if available
                    if "usage" in response:
                        completion_tokens = response["usage"].get("completion_tokens", 0)
                else:
                    # Fallback: try direct "text" field
                    output_text = response.get("text", "")
                
                # Tokenize output text to get token IDs (required for LoadGen accuracy logging)
                if output_text and self.tokenizer is not None:
                    try:
                        output_ids = self.tokenizer.encode(output_text)
                        logger.debug(f"Tokenized output text: {len(output_ids)} tokens")
                    except Exception as e:
                        logger.warning(f"Failed to tokenize output text: {e}")
                        output_ids = []
                elif output_text and self.tokenizer is None:
                    logger.warning(
                        "Tokenizer not provided to VLLMBackend. "
                        "Output token IDs will be empty. "
                        "This may cause issues with accuracy logging. "
                        "Consider passing a tokenizer to the backend."
                    )

            result = {
                "output_ids": output_ids,  # Tokenized output for LoadGen accuracy logging
                "output_text": output_text,
                "metadata": {
                    "latency": latency,
                    "completion_tokens": completion_tokens if completion_tokens > 0 else len(output_ids) if output_ids else len(output_text.split()) if output_text else 0,
                    "error": response.get("error"),
                }
            }
            results.append(result)

        return results

    async def generate_stream(
        self,
        prompt: str,  # Changed from input_ids to prompt string
        max_tokens: int = 100,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate response with streaming support.

        Yields incremental responses as tokens are generated.

        Args:
            prompt: Text prompt string (not token IDs)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Yields:
            Dict with:
                - delta_token_ids: List of new token IDs in this chunk (empty for vLLM)
                - delta_text: New text in this chunk
                - is_first_token: True if this is the first token
                - is_finished: True if generation is complete
                - accumulated_token_ids: All tokens generated so far (empty for vLLM)
                - metadata: Additional info (TTFT, completion_tokens, etc.)
        """
        if not self.initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k if top_k > 0 else -1,
            "top_p": top_p,
            "stop": [],
            "stream": True  # Enable streaming
        }

        # Log sampling parameters before sending streaming request (debug mode)
        logger.debug(
            f"Sending streaming request with sampling parameters: "
            f"max_tokens={max_tokens}, "
            f"temperature={temperature}, "
            f"top_k={top_k if top_k > 0 else -1}, "
            f"top_p={top_p}, "
            f"prompt_length={len(prompt)} characters"
        )

        start_time = time.time()
        first_token_time = None
        accumulated_text = ""
        accumulated_token_ids = []
        is_first = True
        
        # Initialize streaming state tracking
        if not hasattr(self, '_streaming_states'):
            self._streaming_states = {}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Streaming request failed: {response.status} - {error_text}")
                        yield {
                            "delta_token_ids": [],
                            "delta_text": "",
                            "is_first_token": False,
                            "is_finished": True,
                            "accumulated_token_ids": [],
                            "error": f"HTTP {response.status}: {error_text}",
                            "metadata": {}
                        }
                        return

                    # Read streaming response (SSE format)
                    async for line in response.content:
                        if not line:
                            continue

                        # OpenAI-compatible streaming format: "data: {...}\n\n"
                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith('data:'):
                            continue

                        try:
                            # Remove "data:" prefix
                            json_str = line_str[5:].strip()
                            if json_str == '[DONE]':
                                break

                            chunk = json.loads(json_str)

                            # Extract text delta from OpenAI-compatible format
                            delta_text = ""
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                delta_text = choice.get("delta", {}).get("text", "")
                                
                                # Check if finished
                                finish_reason = choice.get("finish_reason")
                                is_finished = finish_reason is not None
                            else:
                                # Fallback format
                                delta_text = chunk.get("text", "")
                                is_finished = chunk.get("finished", False)

                            # Accumulate text
                            if delta_text:
                                accumulated_text += delta_text

                            # Mark first token timing
                            if is_first and delta_text:
                                first_token_time = time.time()
                                is_first = False

                            # Tokenize accumulated text to get token IDs (for LoadGen accuracy logging)
                            # We tokenize the full accumulated text to ensure correctness
                            # (tokenizing substrings can give different results)
                            if accumulated_text and self.tokenizer is not None:
                                try:
                                    # Tokenize the full accumulated text
                                    new_accumulated_token_ids = self.tokenizer.encode(accumulated_text)
                                    
                                    # Calculate delta: new tokens since last update
                                    prev_count = len(accumulated_token_ids)
                                    if len(new_accumulated_token_ids) > prev_count:
                                        delta_token_ids = new_accumulated_token_ids[prev_count:]
                                    else:
                                        delta_token_ids = []
                                    
                                    accumulated_token_ids = new_accumulated_token_ids
                                except Exception as e:
                                    logger.warning(f"Failed to tokenize accumulated text: {e}")
                                    delta_token_ids = []
                            else:
                                delta_token_ids = []

                            yield {
                                "delta_token_ids": delta_token_ids,
                                "delta_text": delta_text,
                                "is_first_token": (first_token_time is not None and is_first is False and len(accumulated_text) == len(delta_text)),
                                "is_finished": is_finished,
                                "accumulated_token_ids": accumulated_token_ids,
                                "accumulated_text": accumulated_text,
                                "metadata": {
                                    "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else None,
                                    "latency_ms": (time.time() - start_time) * 1000,
                                    "estimated_tokens": len(accumulated_token_ids) if accumulated_token_ids else max(1, len(accumulated_text) // 4) if accumulated_text else 0,
                                }
                            }

                            if is_finished:
                                break

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse streaming chunk: {e}")
                            continue

        except asyncio.TimeoutError:
            logger.error(f"Streaming request timed out after {self.timeout}s")
            yield {
                "delta_token_ids": [],
                "delta_text": "",
                "is_first_token": False,
                "is_finished": True,
                "accumulated_token_ids": [],
                "error": "Timeout",
                "metadata": {}
            }
        except Exception as e:
            logger.error(f"Streaming request failed: {e}", exc_info=True)
            yield {
                "delta_token_ids": [],
                "delta_text": "",
                "is_first_token": False,
                "is_finished": True,
                "accumulated_token_ids": [],
                "error": str(e),
                "metadata": {}
            }

    def cleanup(self) -> None:
        """Clean up backend resources."""
        if self.session:
            self.session.close()
            self.session = None
        self.initialized = False
        logger.info("vLLM backend cleaned up")
