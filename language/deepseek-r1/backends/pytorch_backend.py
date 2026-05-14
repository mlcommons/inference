from utils.validation import require_initialized, BackendNotInitializedError
from utils.backend_registry import get_backend_config
from .utils import get_cache_directory
from .base_backend import BaseBackend
from transformers import AutoTokenizer
import torch.distributed as dist
import torch
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional
import logging
import json
from ref_dsinfer.inference.model import Transformer, ModelArgs
from safetensors.torch import load_model
import os
import sys

# Add ref_dsinfer to path BEFORE importing from ref_dsinfer
ref_dsinfer_path = os.environ.get(
    'REF_DSINFER_PATH', '/opt/ref_dsinfer/inference')
sys.path.append(ref_dsinfer_path)


logger = logging.getLogger(__name__)


class PyTorchBackend(BaseBackend):
    """PyTorch backend for DeepSeek model inference with MPI support."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize PyTorch backend for MPI/distributed inference.

        Args:
            config: Optional configuration dict (ignored, uses registry config)
        """
        super().__init__()  # Initialize base class
        # Use config from registry
        self.config = get_backend_config('pytorch')

        # Get distributed setup from environment
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # Get cache directory and model paths
        cache_base = get_cache_directory()
        self.cache_dir = cache_base.parent / 'models'
        model_dir_name = self.config['model_name'].replace('/', '_')
        self.model_path = self.cache_dir / f"{model_dir_name}-Demo"
        self.config_path = Path(
            '/opt/ref_dsinfer/inference/configs/config_671B.json')

        # Model components
        self.model = None
        self.tokenizer = None
        self.model_args = None

        # Convert dtype string to torch dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        self.dtype = dtype_map.get(self.config['dtype'], torch.bfloat16)

        # Check if model exists
        if not self.model_path.exists():
            raise RuntimeError(f"Model not found at {self.model_path}")

        # Check if checkpoint exists for this rank
        checkpoint_file = self.model_path / \
            f"model{self.rank}-mp{self.world_size}.safetensors"
        if not checkpoint_file.exists():
            raise RuntimeError(f"Checkpoint not found: {checkpoint_file}")

    def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        if self.rank == 0:
            logger.info(
                f"Initializing PyTorch backend with {self.world_size} ranks")

        # Initialize distributed if needed and not already initialized
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")

        # Set device and dtype
        torch.cuda.set_device(self.local_rank)
        torch.set_default_dtype(self.dtype)
        torch.set_num_threads(self.config['num_threads'])
        torch.manual_seed(self.config['seed'])

        # Load model configuration
        if not self.config_path.exists():
            raise RuntimeError(
                f"Model configuration not found at {self.config_path}")

        with open(self.config_path) as f:
            model_args_data = json.load(f)

        # Override with our config
        model_args_data['max_batch_size'] = self.config['batch_size']
        model_args_data['max_seq_len'] = self.config['max_seq_len']

        self.model_args = ModelArgs(**model_args_data)

        if self.rank == 0:
            logger.info(f"Model config: batch_size={self.model_args.max_batch_size}, "
                        f"seq_len={self.model_args.max_seq_len}")

        # Initialize model
        with torch.device(self.config['device']):
            self.model = Transformer(self.model_args)

        # Load tokenizer (only rank 0 needs it for MLPerf, but all ranks need
        # it for run_eval_mpi)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), revision=self.config['model_revision'])

        # Load model weights
        checkpoint_file = self.model_path / \
            f"model{self.rank}-mp{self.world_size}.safetensors"
        load_model(self.model, str(checkpoint_file))

        if self.rank == 0:
            logger.info("Model loaded successfully")

        # Mark as initialized
        self.is_initialized = True

    def sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample from logits with temperature."""
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(
            probs).exponential_(1)).argmax(dim=-1)

    @torch.inference_mode()
    def _generate_batch(
        self,
        prompt_tokens: List[List[int]],
    ) -> List[List[int]]:
        """Generate tokens for a batch of prompts."""
        prompt_lens = [len(t) for t in prompt_tokens]
        max_prompt_len = max(prompt_lens)

        # Check prompt length
        if max_prompt_len > self.model.max_seq_len:
            raise ValueError(
                f"Prompt length {max_prompt_len} exceeds model maximum "
                f"sequence length {self.model.max_seq_len}"
            )

        # Calculate total length
        total_len = min(self.model.max_seq_len,
                        self.config['max_new_tokens'] + max_prompt_len)

        # Initialize tokens tensor
        tokens = torch.full(
            (len(prompt_tokens), total_len),
            -1,
            dtype=torch.long,
            device=self.config['device']
        )

        # Fill in prompt tokens
        for i, t in enumerate(prompt_tokens):
            tokens[i, :len(t)] = torch.tensor(
                t, dtype=torch.long, device=self.config['device'])

        # Generation loop
        prev_pos = 0
        finished = torch.tensor(
            [False] * len(prompt_tokens), device=self.config['device'])
        prompt_mask = tokens != -1
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1

        for cur_pos in range(min(prompt_lens), total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if self.config['temperature'] > 0:
                next_token = self.sample(logits, self.config['temperature'])
            else:
                next_token = logits.argmax(dim=-1)

            # Only generate for positions after prompt
            next_token = torch.where(
                prompt_mask[:, cur_pos],
                tokens[:, cur_pos],
                next_token
            )
            tokens[:, cur_pos] = next_token

            # Check for EOS
            finished |= torch.logical_and(
                ~prompt_mask[:, cur_pos],
                next_token == eos_id
            )

            prev_pos = cur_pos

            if finished.all():
                break

        # Extract completion tokens
        completion_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_lens[i]:prompt_lens[i] +
                        self.config['max_new_tokens']]
            if eos_id in toks and eos_id != -1:
                toks = toks[:toks.index(eos_id)]

            # Validate response is not empty
            if not toks:
                prompt_preview = prompt_tokens[i][:50] if len(
                    prompt_tokens[i]) > 50 else prompt_tokens[i]
                raise RuntimeError(
                    f"Empty response generated from PyTorch backend for prompt tokens: {prompt_preview}...")

            completion_tokens.append(toks)

        return completion_tokens

    @require_initialized
    def generate(
            self, tokenized_prompts: List[List[int]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for a list of pre-tokenized prompts.

        This method should only be called by rank 0 in distributed mode.

        Args:
            tokenized_prompts: List of pre-tokenized prompts (token IDs)
            **kwargs: Additional generation parameters (ignored)

        Returns:
            List of dictionaries with generation results
        """
        # In distributed mode, only rank 0 should call this
        if self.world_size > 1 and self.rank != 0:
            raise RuntimeError(
                "generate() should only be called by rank 0 in distributed mode")

        results = []

        # Process in batches
        batch_size = self.config['batch_size']
        for i in range(0, len(tokenized_prompts), batch_size):
            batch_tokens = tokenized_prompts[i:i + batch_size]

            # For distributed mode, we need to broadcast to other ranks
            if self.world_size > 1:
                objects_to_broadcast = [batch_tokens]
                dist.broadcast_object_list(objects_to_broadcast, src=0)

            # Generate tokens
            generated_tokens = self._generate_batch(batch_tokens)

            # Create results
            for tokens in generated_tokens:
                result = {
                    'tokens': tokens,
                }
                results.append(result)

        return results

    @require_initialized
    def generate_batch_distributed(
            self, batch_tokens: List[List[int]]) -> List[List[int]]:
        """
        Generate tokens for a batch in distributed mode.

        This method is called by all ranks and handles the distributed communication.
        Used by run_eval_mpi.py where all ranks participate.

        Args:
            batch_tokens: Tokenized prompts for this batch (None for non-rank-0)

        Returns:
            Generated tokens (only meaningful for rank 0)
        """
        # In distributed mode, broadcast the batch from rank 0
        if self.world_size > 1:
            if self.rank == 0:
                objects_to_broadcast = [batch_tokens]
            else:
                objects_to_broadcast = [None]
            dist.broadcast_object_list(objects_to_broadcast, src=0)

            if self.rank != 0:
                batch_tokens = objects_to_broadcast[0]

        # All ranks generate
        if batch_tokens:
            return self._generate_batch(batch_tokens)
        else:
            return []

    @require_initialized
    def generate_async(
            self, tokenized_prompts: List[List[int]], **kwargs) -> List[asyncio.Future]:
        """
        Generate responses asynchronously.

        For PyTorch backend, this is just a wrapper around sync generation.
        In distributed mode, only rank 0 should call this method.
        """
        if self.world_size > 1 and self.rank != 0:
            raise RuntimeError(
                "generate_async() should only be called by rank 0 in distributed mode")

        loop = asyncio.get_event_loop()

        # Run sync generation in executor
        future = loop.run_in_executor(
            None,
            self.generate,
            tokenized_prompts
        )

        # Create individual futures for each prompt
        futures = []
        num_prompts = len(tokenized_prompts)

        async def extract_result(idx):
            results = await future
            return results[idx]

        for i in range(num_prompts):
            prompt_future = loop.create_task(extract_result(i))
            futures.append(prompt_future)

        return futures

    @require_initialized
    def generate_batch_distributed_async(
            self, batch_tokens: List[List[int]]) -> asyncio.Future:
        """
        Generate tokens for a batch in distributed mode asynchronously.

        This method is called by all ranks and handles the distributed communication.
        Used by run_mlperf_mpi.py where all ranks participate.

        Args:
            batch_tokens: Tokenized prompts for this batch (None for non-rank-0)

        Returns:
            Future containing generated tokens (only meaningful for rank 0)
        """
        loop = asyncio.get_event_loop()

        # Run distributed generation in executor
        future = loop.run_in_executor(
            None,
            self.generate_batch_distributed,
            batch_tokens
        )

        return future

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.rank == 0:
            logger.info("Shutting down PyTorch backend")

        # Clear model from memory
        self.model = None
        self.tokenizer = None

        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Mark as not initialized
        self.is_initialized = False

        # Note: We don't destroy the process group here because it might be
        # managed by the calling script (run_eval_mpi.py)
