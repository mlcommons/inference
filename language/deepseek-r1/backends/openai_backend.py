"""
OpenAI backend variant that connects to an existing server.

Unlike the standard OpenAIBackend, this backend does not launch or manage a
server process. It expects an environment variable pointing to a running
OpenAI server and simply creates OpenAI-compatible clients against it.
"""

import asyncio
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import requests
import torch
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from .base_backend import BaseBackend, StreamingChunk
from .utils import (
    get_cache_directory,
    set_all_seeds,
    setup_huggingface_cache,
)
from utils.backend_registry import apply_backend_env_vars, get_backend_config
from utils.validation import require_initialized, validate_prompts_input


class OpenAIBackend(BaseBackend):
    """Connect to a pre-existing OpenAI compatible server."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        self.config = get_backend_config(self.backend_name)
        if config:
            self.config.update(config)

        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        self.tokenizer = None
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._server_root: Optional[str] = None

        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables and cache directories."""
        get_cache_directory()  # Ensure cache directory exists
        setup_huggingface_cache()
        apply_backend_env_vars(self.backend_name)

        seed = self.config.get('random_seed', self.config.get('seed', 42))
        set_all_seeds(seed)

    def _resolve_server_root(self) -> str:
        """Resolve server root URL and OpenAI base URL from env/config."""
        return f"http://{self.config.get('host', '127.0.0.1')}:{self.config.get('port', 8000)}"

    def _check_remote_health(self, server_root: str) -> None:
        """Ping the remote server health endpoint to fail fast on misconfig."""
        health_url = f"{server_root}/health"
        try:
            response = requests.get(health_url, timeout=10)
            response.raise_for_status()
            print(f"[OPENAI] Health check OK at {health_url}")
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI server at {health_url} is unavailable: {exc}"
            ) from exc

    def _run_warmup_query(self) -> None:
        """Send a simple warmup query to verify the connection."""
        if self.client is None:
            return

        if not self.config.get('run_warmup_query', True):
            return

        try:
            warmup_response = self.client.chat.completions.create(
                model=self.config['served_model_name'],
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.0,
                max_tokens=10,
                seed=self.config['seed']
            )

            if warmup_response.choices[0].message.content:
                preview = warmup_response.choices[0].message.content[:50]
                print(f"[OPENAI] Warmup query successful: {preview}...")
            else:
                raise RuntimeError("Warmup query returned empty content")
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI warmup query failed: {exc}"
            ) from exc

    def initialize(self) -> None:
        """Initialize clients against an existing OpenAI server."""
        if self.is_initialized:
            return

        try:
            print(f"[OPENAI] Loading tokenizer: {self.config['tokenizer']}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['tokenizer'],
                revision=self.config['model_revision']
            )

            server_root = self._resolve_server_root()
            base_url = f"{server_root}/v1"
            self._server_root = server_root

            api_key = (
                self.config.get('api_key')
                or os.environ.get('OPENAI_API_KEY')
                or "dummy-key"
            )
            timeout_config = httpx.Timeout(
                timeout=self.config['request_timeout'],
                connect=30.0,
                read=None,
                write=None,
                pool=None
            )

            print(f"[OPENAI] Connecting to server at: {base_url}")
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_config,
                max_retries=10,
                http_client=httpx.Client(
                    limits=httpx.Limits(
                        max_connections=self.config['max_running_requests'],
                        max_keepalive_connections=self.config['max_running_requests'] // 2
                    )
                )
            )
            self.async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_config,
                max_retries=10,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=self.config['max_running_requests'],
                        max_keepalive_connections=self.config['max_running_requests'] // 2
                    )
                )
            )
            self._async_semaphore = asyncio.Semaphore(
                self.config['max_running_requests'])

            self._check_remote_health(server_root)
            self._run_warmup_query()

            self.is_initialized = True
            print("[OPENAI] Backend initialized successfully!")

        except Exception as exc:
            print(f"[OPENAI] Initialization failed: {exc}")
            self.client = None
            self.async_client = None
            self.tokenizer = None
            self._async_semaphore = None
            self.is_initialized = False
            raise

    @require_initialized
    def generate(self,
                 tokenized_prompts: Optional[List[List[int]]] = None,
                 text_prompts: Optional[List[str]] = None,
                 **kwargs) -> List[Dict[str, Any]]:
        """Generate responses synchronously."""
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        validate_prompts_input(
            backend_name=self.backend_name,
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        if text_prompts is None:
            prompt_strings = [
                self.tokenizer.decode(tokens, skip_special_tokens=False)
                for tokens in tokenized_prompts
            ]
        else:
            prompt_strings = text_prompts

        results = []
        for prompt in tqdm(prompt_strings,
                           desc="OpenAI sync inference",
                           unit="prompt"):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config['served_model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    max_tokens=self.config['max_tokens'],
                    seed=self.config['seed'],
                )

                generated_text = completion.choices[0].message.content
                if not generated_text:
                    raise RuntimeError(
                        f"Empty response received from OpenAI server for prompt: {prompt[:100]}..."
                    )

                tokens = self.tokenizer.encode(generated_text)
                results.append({'tokens': tokens, 'text': generated_text})

            except Exception as exc:
                print(f"\nError generating completion: {exc}")
                raise RuntimeError(
                    f"OpenAI failed to generate tokens for prompt: {prompt[:100]}..."
                )

        return results

    async def _async_generate_single(
            self, prompt: str, idx: int,
            semaphore: asyncio.Semaphore) -> Tuple[int, Dict[str, Any]]:
        """Generate a single response asynchronously with semaphore control."""
        if self.async_client is None:
            raise RuntimeError(
                f"OpenAI async client is not initialized for prompt {idx}")

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

                generated_text = completion.choices[0].message.content
                if not generated_text:
                    raise RuntimeError(
                        f"Empty response received from OpenAI server for prompt: {prompt[:100]}..."
                    )

                tokens = self.tokenizer.encode(generated_text)
                return idx, {'tokens': tokens, 'text': generated_text}

            except Exception as exc:
                print(f"\nError generating completion for prompt {idx}: {exc}")
                raise RuntimeError(
                    f"OpenAI failed to generate tokens for prompt {idx}: {exc}"
                )

    @require_initialized
    def generate_async(self,
                       tokenized_prompts: Optional[List[List[int]]] = None,
                       text_prompts: Optional[List[str]] = None,
                       **kwargs) -> List[asyncio.Future]:
        """Generate responses asynchronously using shared semaphore."""
        if self.async_client is None:
            raise RuntimeError("OpenAI async client is not initialized.")

        validate_prompts_input(
            backend_name=self.backend_name,
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        if text_prompts is None:
            prompt_strings = [
                self.tokenizer.decode(tokens, skip_special_tokens=False)
                for tokens in tokenized_prompts
            ]
        else:
            prompt_strings = text_prompts

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        futures = []
        for idx, prompt in enumerate(prompt_strings):
            task = asyncio.create_task(
                self._async_generate_single(prompt, idx, self._async_semaphore))

            future = asyncio.Future()

            def make_callback(future_obj, expected_idx):
                def callback(task_obj):
                    try:
                        i, result = task_obj.result()
                        if i != expected_idx:
                            future_obj.set_exception(
                                Exception(
                                    f"Index mismatch: expected {expected_idx}, got {i}"))
                        else:
                            future_obj.set_result(result)
                    except Exception as exc:
                        future_obj.set_exception(exc)
                return callback

            task.add_done_callback(make_callback(future, idx))
            futures.append(future)

        return futures

    async def generate_stream(self,
                              tokenized_prompts: Optional[List[List[int]]] = None,
                              text_prompts: Optional[List[str]] = None,
                              **kwargs) -> List[AsyncIterator[StreamingChunk]]:
        """Generate responses for a list of prompts with streaming."""
        if self.async_client is None:
            raise RuntimeError("OpenAI async client is not initialized.")

        validate_prompts_input(
            backend_name=self.backend_name,
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        if text_prompts is None:
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
                            token_ids=[],
                            is_finished=finish_reason is not None,
                            finish_reason=finish_reason
                        )
                    elif finish_reason:
                        yield StreamingChunk(
                            token="",
                            token_ids=[],
                            is_finished=True,
                            finish_reason=finish_reason
                        )
            except Exception as exc:
                print(f"[OPENAI] Streaming error for prompt: {exc}")
                raise

        return [stream_single_prompt(prompt) for prompt in prompt_strings]

    def shutdown(self) -> None:
        """Clean up client resources."""
        print("[OPENAI] Shutting down OpenAI backend...")
        self.client = None
        self.async_client = None
        self._async_semaphore = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False
        print("[OPENAI] Backend shutdown complete")
