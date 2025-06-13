import asyncio
import os
import random
import subprocess
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm
from transformers import set_seed
from vllm import LLM, SamplingParams

from .base_backend import BaseBackend
from .utils import get_cache_directory, setup_huggingface_cache, set_all_seeds, validate_prompts
from utils.backend_registry import get_backend_config, apply_backend_env_vars
from utils.validation import require_initialized, validate_prompts_input


class VLLMBackend(BaseBackend):
    """vLLM backend using synchronous LLM class with async wrapper support.

    This backend only accepts text prompts, not tokenized prompts.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize vLLM backend with configuration from registry."""
        super().__init__()  # Initialize base class
        # Get configuration from registry
        self.config = get_backend_config('vllm')
        # Allow override with passed config
        if config:
            self.config.update(config)

        # Set model and tokenizer names
        self.model_name = self.config['model']
        self.tokenizer_name = self.config['tokenizer']

        self.llm = None
        self.sampling_params = None
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables and cache directories."""
        # Use the utility function to get cache directory
        cache_base = get_cache_directory()

        # Use models subdirectory to match user's example paths
        self.cache_dir = cache_base.parent / 'models'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config['model_cache_dir'] = str(self.cache_dir)

        # Set up HuggingFace cache environment variables
        setup_huggingface_cache()

        # Apply backend-specific environment variables from registry
        apply_backend_env_vars('vllm')

        # Set seeds for reproducibility
        seed = self.config['seed']
        set_all_seeds(seed)

    def _ensure_model_cached(self) -> Path:
        """Ensure model is available locally and return path."""
        model_name = self.config['model']

        # Create safe directory name from model path
        model_dir_name = model_name.replace("/", "_")
        checkpoint_path = self.cache_dir / model_dir_name

        if not checkpoint_path.exists():
            print(f"Model not found at {checkpoint_path}")
            print(f"Downloading {model_name} from HuggingFace...")

            # Create download command following user's exact steps
            cmd = [
                "huggingface-cli", "download",
                model_name,
                "--revision", self.config['model_revision'],
                "--local-dir", str(checkpoint_path)
            ]

            # Set environment variable for faster downloads
            env = os.environ.copy()
            env['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

            try:
                # Run download command
                print(
                    f"Running command: HF_HUB_ENABLE_HF_TRANSFER=1 {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"Model downloaded successfully to {checkpoint_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading model: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                raise RuntimeError(f"Failed to download model: {e}")
        else:
            print(f"Using cached model at {checkpoint_path}")

        return checkpoint_path

    def initialize(self) -> None:
        """Initialize the vLLM backend with LLM class."""
        # Ensure model is cached locally
        checkpoint_path = self._ensure_model_cached()

        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            n=1,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            max_tokens=self.config['max_output_len'],
            seed=self.config['seed'],
        )

        # Create LLM instance
        print(f"Initializing vLLM with model: {self.model_name}")

        try:
            self.llm = LLM(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                tensor_parallel_size=self.config['tensor_parallel_size'],
                max_model_len=self.config['max_model_len'],
                max_num_seqs=self.config['max_num_seqs'],
                gpu_memory_utilization=self.config['gpu_memory_utilization'],
                trust_remote_code=self.config['trust_remote_code'],
                dtype=self.config['dtype'],
                seed=self.config['seed'],
                enforce_eager=self.config['enforce_eager'],
                enable_prefix_caching=self.config['enable_prefix_caching'],
                enable_chunked_prefill=self.config['enable_chunked_prefill'],
            )
            self.is_initialized = True
            print("vLLM backend initialized successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM: {e}")

    @require_initialized
    def generate(self,
                 tokenized_prompts: Optional[List[List[int]]] = None,
                 text_prompts: Optional[List[str]] = None,
                 **kwargs) -> List[Dict[str, Any]]:
        """Generate responses synchronously using LLM.generate().

        Note: vLLM backend only accepts text_prompts parameter.
        """
        # Validate prompts using centralized validation
        validate_prompts_input(
            backend_name='vllm',
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        # Run generation
        outputs = self.llm.generate(text_prompts, self.sampling_params)

        # Convert outputs to standard format
        results = []
        for output in outputs:
            # output is of type RequestOutput
            # We assume n=1 in sampling_params, so take outputs[0]
            completion = output.outputs[0]

            # Validate response is not empty
            if not completion.text:
                # Get the corresponding prompt for context
                prompt_idx = outputs.index(output)
                prompt_preview = text_prompts[prompt_idx][:100] if len(
                    text_prompts[prompt_idx]) > 100 else text_prompts[prompt_idx]
                raise RuntimeError(
                    f"Empty response received from vLLM for prompt: {prompt_preview}...")

            results.append({
                # Convert tuple to list for .copy() compatibility
                'tokens': list(completion.token_ids),
                'text': completion.text,
                'finish_reason': completion.finish_reason
            })

        return results

    @require_initialized
    def generate_async(self,
                       tokenized_prompts: Optional[List[List[int]]] = None,
                       text_prompts: Optional[List[str]] = None,
                       **kwargs) -> List[asyncio.Future]:
        """Generate responses asynchronously, returning futures immediately.

        Note: vLLM backend only accepts text_prompts parameter.
        """
        # Validate prompts using centralized validation
        validate_prompts_input(
            backend_name='vllm',
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type='text'
        )

        if not text_prompts:
            return []

        # Create and return a future that will run the synchronous generation
        loop = asyncio.get_event_loop()

        # Run the synchronous generation in executor
        future = loop.run_in_executor(None, self._generate_batch, text_prompts)

        # Convert the single future to a list of futures, one per prompt
        async def split_results():
            results = await future
            return results

        # Create individual futures for each prompt
        main_future = asyncio.ensure_future(split_results())

        # Create a future for each prompt that extracts its result
        prompt_futures = []
        for i in range(len(text_prompts)):
            async def get_result(idx=i):
                results = await main_future
                return results[idx]
            prompt_futures.append(asyncio.ensure_future(get_result()))

        return prompt_futures

    def _generate_batch(self, text_prompts: List[str]) -> List[Dict[str, Any]]:
        """Helper method to run synchronous generation for async wrapper."""
        outputs = self.llm.generate(text_prompts, self.sampling_params)

        # Convert outputs to standard format
        results = []
        for output in outputs:
            completion = output.outputs[0]

            # Validate response is not empty
            if not completion.text:
                # Get the corresponding prompt for context
                prompt_idx = outputs.index(output)
                prompt_preview = text_prompts[prompt_idx][:100] if len(
                    text_prompts[prompt_idx]) > 100 else text_prompts[prompt_idx]
                raise RuntimeError(
                    f"Empty response received from vLLM for prompt: {prompt_preview}...")

            results.append({
                # Convert tuple to list for .copy() compatibility
                'tokens': list(completion.token_ids),
                'text': completion.text,
                'finish_reason': completion.finish_reason
            })

        return results

    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        print("Shutting down vLLM backend...")

        # Delete the LLM instance first
        if self.llm is not None:
            # Access internal executor to ensure proper cleanup
            if self.llm.llm_engine is not None:
                try:
                    # This helps cleanup vLLM's internal Ray/multiprocessing
                    # resources
                    del self.llm.llm_engine.model_executor
                except Exception as e:
                    print(f"Warning: Failed to cleanup model executor: {e}")

            del self.llm
            self.llm = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete

        # Clean up distributed resources if they exist
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
                print("Destroyed process group")
            except Exception as e:
                print(f"Warning: Failed to destroy process group: {e}")

        self.sampling_params = None
        self.is_initialized = False

        # Perform garbage collection
        gc.collect()

        # Check if Ray is initialized and shutdown if needed
        # This helps with multiprocessing shared memory cleanup
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                print("Ray shutdown completed")
        except Exception as e:
            print(f"Warning: Failed to shutdown Ray: {e}")

        # Wait for cleanup to complete
        time.sleep(0.5)

        print("vLLM backend shutdown complete")
