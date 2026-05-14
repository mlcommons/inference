#!/usr/bin/env python3
"""SGLang backend implementation for gpt-oss."""

import asyncio
import json
import logging
import requests
import time
from typing import List, Dict, Any, Optional, AsyncIterator
import aiohttp
from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class SGLangBackend(BaseBackend):
    """SGLang inference backend using HTTP API.

    Connects to an SGLang server running the gpt-oss model.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:30000",
        timeout: int = 1200,
        max_pool_size: int = 2000,  # Default pool size for high concurrency
        **kwargs
    ):
        """Initialize SGLang backend.

        Args:
            server_url: URL of the SGLang server
            timeout: Request timeout in seconds
            max_pool_size: Maximum connection pool size (should be >= max_concurrency)
            **kwargs: Additional configuration
        """
        config = {
            "server_url": server_url,
            "timeout": timeout,
            "max_pool_size": max_pool_size,
            **kwargs
        }
        super().__init__(config)
        self.server_url = server_url
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.session = None

    def initialize(self) -> None:
        """Initialize connection to SGLang server."""
        if self.initialized:
            logger.warning("Backend already initialized")
            return

        logger.info(f"Connecting to SGLang server at {self.server_url}")
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
                input_ids=[1, 2, 3],
                max_tokens=5,
                temperature=0.001,
                top_k=1,
                top_p=1.0
            )
            if "error" in test_response:
                raise ConnectionError(
                    f"Failed to connect to SGLang server: {test_response['error']}"
                )
            logger.info("Successfully connected to SGLang server")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize SGLang backend: {e}")
            raise

    def _send_request(
        self,
        input_ids: List[int],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Dict[str, Any]:
        """Send a single request to the SGLang server.

        Args:
            input_ids: Token IDs for the prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Returns:
            Response dictionary from the server
        """
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        }

        try:
            response = self.session.post(
                f"{self.server_url}/generate",
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
        prompts: List[List[int]],
        max_tokens: int = 100,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of token ID sequences
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional parameters (ignored)

        Returns:
            List of response dictionaries with keys:
                - output_ids: List of generated token IDs
                - output_text: Generated text (if available)
                - metadata: Additional metadata (latencies, etc.)
        """
        if not self.initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        results = []
        for prompt_ids in prompts:
            start_time = time.time()
            response = self._send_request(
                input_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            end_time = time.time()
            latency = end_time - start_time

            # Extract output_ids from response
            output_ids = []
            output_text = ""
            if "error" not in response:
                output_ids = response.get("output_ids", [])
                output_text = response.get("text", "")

            result = {
                "output_ids": output_ids,
                "output_text": output_text,
                "metadata": {
                    "latency": latency,
                    "completion_tokens": response.get("meta_info", {}).get(
                        "completion_tokens", len(output_ids)
                    ),
                    "error": response.get("error"),
                }
            }
            results.append(result)

        return results

    async def generate_stream(
        self,
        input_ids: List[int],
        max_tokens: int = 100,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate response with streaming support.

        Yields incremental responses as tokens are generated.

        Args:
            input_ids: Token IDs for the prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Yields:
            Dict with:
                - delta_token_ids: List of new token IDs in this chunk
                - delta_text: New text in this chunk
                - is_first_token: True if this is the first token
                - is_finished: True if generation is complete
                - accumulated_token_ids: All tokens generated so far
                - metadata: Additional info (TTFT, completion_tokens, etc.)

        Note:
            SGLang's streaming API behavior:
            - Returns 'output_ids', 'text', and 'meta_info' in each chunk
            - 'output_ids' can have retractions (length can decrease between chunks)
            - 'meta_info.completion_tokens' is the RELIABLE cumulative token count
            - 'finish_reason' in meta_info indicates completion (not a 'finished' flag)
            - We use completion_tokens for accurate LoadGen token/sec metrics
        """
        if not self.initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
            "stream": True  # Enable streaming
        }

        start_time = time.time()
        first_token_time = None
        accumulated_token_ids = []
        accumulated_text = ""
        is_first = True

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/generate",
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

                    # Read streaming response
                    async for line in response.content:
                        if not line:
                            continue

                        # SGLang sends data as "data: {...}\n\n"
                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith('data:'):
                            continue

                        try:
                            # Remove "data:" prefix
                            json_str = line_str[5:].strip()
                            if json_str == '[DONE]':
                                break

                            chunk = json.loads(json_str)

                            # Extract text delta
                            delta_text = chunk.get("text", "")

                            # Check if this is the final chunk
                            # SGLang uses 'finish_reason' in meta_info, not
                            # 'finished' flag
                            meta_info = chunk.get("meta_info", {})
                            finish_reason = meta_info.get("finish_reason")
                            is_finished = (
                                finish_reason is not None and finish_reason != "null") or chunk.get(
                                "finished", False)

                            # Extract token information from chunk
                            # SGLang's output_ids can have retractions, so use meta_info.completion_tokens
                            # which is the reliable cumulative count
                            chunk_output_ids = chunk.get("output_ids", [])
                            completion_tokens = meta_info.get(
                                "completion_tokens", 0)

                            if completion_tokens > 0:
                                # Use completion_tokens as the authoritative
                                # count
                                previous_count = len(accumulated_token_ids)

                                if completion_tokens > previous_count:
                                    # New tokens generated
                                    num_new_tokens = completion_tokens - previous_count

                                    if chunk_output_ids and len(
                                            chunk_output_ids) >= num_new_tokens:
                                        # Use actual token IDs from chunk
                                        delta_token_ids = chunk_output_ids[-num_new_tokens:] if num_new_tokens > 0 else [
                                        ]
                                    else:
                                        # Fallback: create placeholder tokens
                                        # for counting
                                        delta_token_ids = list(
                                            range(previous_count, completion_tokens))

                                    accumulated_token_ids.extend(
                                        delta_token_ids)
                                else:
                                    delta_token_ids = []

                            else:
                                # No completion_tokens - fallback to output_ids
                                # or text estimation
                                if chunk_output_ids:
                                    delta_token_ids = chunk_output_ids
                                    accumulated_token_ids.extend(
                                        delta_token_ids)
                                elif delta_text:
                                    # Estimate from text length
                                    estimated_tokens = max(
                                        1, len(delta_text) // 4)
                                    delta_token_ids = [0] * estimated_tokens
                                    accumulated_token_ids.extend(
                                        delta_token_ids)
                                else:
                                    delta_token_ids = []

                            # Accumulate text
                            if delta_text:
                                accumulated_text += delta_text

                            # Mark first token timing
                            if is_first and (delta_token_ids or delta_text):
                                first_token_time = time.time()
                                is_first = False

                            yield {
                                "delta_token_ids": delta_token_ids,
                                "delta_text": delta_text,
                                "is_first_token": (first_token_time is not None and is_first is False and len(accumulated_token_ids) <= len(delta_token_ids)),
                                "is_finished": is_finished,
                                "accumulated_token_ids": accumulated_token_ids.copy(),
                                "accumulated_text": accumulated_text,
                                "metadata": {
                                    "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else None,
                                    "latency_ms": (time.time() - start_time) * 1000,
                                    **chunk.get("meta_info", {})
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
                "accumulated_token_ids": accumulated_token_ids,
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
                "accumulated_token_ids": accumulated_token_ids,
                "error": str(e),
                "metadata": {}
            }

    def cleanup(self) -> None:
        """Clean up backend resources."""
        if self.session:
            self.session.close()
            self.session = None
        self.initialized = False
        logger.info("SGLang backend cleaned up")
