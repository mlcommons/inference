#!/usr/bin/env python3
"""vLLM backend implementation for gpt-oss - MINIMAL VERSION."""

import logging
from typing import List, Dict, Any

from vllm import LLM, SamplingParams

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class VLLMBackend(BaseBackend):
    """vLLM inference backend using native LLM class with default settings."""

    def __init__(
        self,
        model_name: str = "/mnt/models/gpt-oss-120b",
        tensor_parallel_size: int = 2,
        max_model_len: int = 4096,
        max_num_seqs: int = 64,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        **kwargs
    ):
        """Initialize vLLM backend with minimal configuration.

        Args:
            model_name: Path to model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            max_num_seqs: Maximum number of sequences to process in parallel
            gpu_memory_utilization: Fraction of GPU memory to use
            trust_remote_code: Whether to trust remote code
            dtype: Data type (auto, float16, bfloat16)
        """
        config = {
            "model_name": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            **kwargs
        }
        super().__init__(config)
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.llm = None

    def initialize(self) -> None:
        """Initialize the vLLM backend with default settings."""
        if self.initialized:
            logger.warning("Backend already initialized")
            return

        logger.info(f"Initializing vLLM with model: {self.model_name}")
        logger.info(f"Tensor parallel: {self.tensor_parallel_size} GPUs")
        logger.info(f"Max model length: {self.max_model_len}")
        logger.info(f"Max num seqs: {self.max_num_seqs}")
        logger.info("Using vLLM default settings (no custom env vars)")

        try:
            # Use vLLM defaults - no environment variable overrides
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype,
            )
            self.initialized = True
            logger.info("vLLM backend initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

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
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            List of response dictionaries
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        if not prompts:
            return []

        sampling_params = SamplingParams(
            n=1,
            temperature=max(temperature, 0.001),
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

        outputs = self.llm.generate(
            prompt_token_ids=prompts,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        results = []
        for output in outputs:
            completion = output.outputs[0]
            results.append({
                "output_ids": completion.token_ids,
                "output_text": completion.text,
                "metadata": {
                    "finish_reason": completion.finish_reason,
                    "num_tokens": len(completion.token_ids),
                }
            })
        return results

    def cleanup(self) -> None:
        """Clean up vLLM backend resources."""
        logger.info("Cleaning up vLLM backend")

        if self.llm is not None:
            del self.llm
            self.llm = None

        # Optional: Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self.initialized = False
        logger.info("vLLM backend cleanup complete")
