"""Shared tokenization utilities for all runners."""

from typing import List, Tuple, Optional
from transformers import AutoTokenizer
from utils.backend_registry import uses_chat_template, detect_backend, get_backend_config, get_supported_backends


class StandardTokenizer:
    """Standard tokenizer for DeepSeek models."""

    # Standard configuration used across all runners
    DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1"
    DEFAULT_MAX_LENGTH = 32 * 1024

    def __init__(self, model_name: str = None, max_length: int = None):
        """
        Initialize tokenizer.

        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length or self.DEFAULT_MAX_LENGTH
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            print(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision="56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad")
        return self._tokenizer

    def tokenize_prompts(self, prompts: List[str],
                         use_chat_template: Optional[bool] = None,
                         backend_name: Optional[str] = None) -> Tuple[List[List[int]], List[str]]:
        """
        Tokenize prompts with backend-specific handling.

        Args:
            prompts: List of text prompts
            use_chat_template: Whether to use chat template (if None and backend_name provided, uses registry)
            backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

        Returns:
            Tuple of (tokenized_prompts, processed_strings)
        """
        # Auto-detect backend if not provided
        if backend_name is None:
            backend_name = detect_backend()

        # Determine chat template usage from registry if backend_name provided
        if use_chat_template is None:
            use_chat_template = uses_chat_template(backend_name)
            print(
                f"[{backend_name}] Using chat template from registry: {use_chat_template}")

        tokenized = []
        processed_strings = []

        for prompt in prompts:
            if use_chat_template and hasattr(
                    self.tokenizer, 'apply_chat_template'):
                tokens = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    max_length=self.max_length,
                    truncation=True
                )
                processed_string = self.tokenizer.decode(
                    tokens, skip_special_tokens=False)
            else:
                tokens = self.tokenizer.encode(
                    prompt,
                    truncation=True,
                    max_length=self.max_length
                )
                processed_string = prompt

            tokenized.append(tokens)
            processed_strings.append(processed_string)

        return tokenized, processed_strings

    def decode_tokens(self, tokens: List[int],
                      skip_special_tokens: bool = True) -> str:
        """Decode tokens to text."""
        return self.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_lists: List[List[int]],
                     skip_special_tokens: bool = True) -> List[str]:
        """Batch decode multiple token lists."""
        return self.tokenizer.batch_decode(
            token_lists, skip_special_tokens=skip_special_tokens)


def process_inference_results(raw_results: List[dict],
                              tokenizer: Optional[StandardTokenizer] = None,
                              backend_name: Optional[str] = None,
                              uses_text_prompts: bool = False) -> List[dict]:
    """
    Process raw inference results into standardized format.

    Args:
        raw_results: Raw results from backend
        tokenizer: Tokenizer for decoding
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
        uses_text_prompts: Whether backend uses text prompts

    Returns:
        List of standardized result dictionaries
    """
    # Auto-detect backend if not provided
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in get_supported_backends():
        raise ValueError(f"Backend {backend_name} is not supported")

    backend_config = get_backend_config(backend_name)

    standardized_results = []

    for raw_result in raw_results:
        # Handle text-prompt backends
        if uses_text_prompts and 'text' in raw_result:
            text = raw_result['text']
            tokens = raw_result.get('tokens', [])
        else:
            # Decode tokens to get text
            tokens = raw_result.get('tokens', [])
            text = ''
            if tokenizer and tokens:
                try:
                    text = tokenizer.decode_tokens(tokens)
                except BaseException:
                    pass

        standardized = {
            'model_output': text,
            'tok_model_output': tokens,
            'tok_model_output_len': len(tokens),
            'model_backend': backend_name,
        }
        standardized_results.append(standardized)

    return standardized_results
