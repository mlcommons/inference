from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator
import asyncio
from dataclasses import dataclass


@dataclass
class StreamingChunk:
    """Standardized streaming response chunk."""
    token: str
    token_ids: List[int]
    is_finished: bool
    finish_reason: Optional[str] = None


class BaseBackend(ABC):
    """Abstract base class for all inference backends."""

    def __init__(self):
        """Initialize base backend attributes."""
        self.is_initialized = False

    @property
    def backend_name(self) -> str:
        """Get backend name from class name or registry.

        This property provides a consistent way to get the backend name.
        Subclasses don't need to set self.backend_name anymore.
        """
        # Extract backend name from class name (e.g., VLLMBackend -> vllm)
        class_name = self.__class__.__name__
        if class_name.endswith('Backend'):
            name = class_name[:-7].lower()  # Remove 'Backend' suffix
        else:
            name = class_name.lower()

        # Validate against registry
        from utils.backend_registry import BACKEND_REGISTRY
        if name not in BACKEND_REGISTRY:
            # Fallback to class name if not in registry
            return class_name.lower()
        return name

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (load model, tokenizer, etc.).

        Subclasses should call super().initialize() at the end to set is_initialized=True,
        or set it manually if they don't call super().
        """
        pass

    @abstractmethod
    def generate(self,
                 tokenized_prompts: Optional[List[List[int]]] = None,
                 text_prompts: Optional[List[str]] = None,
                 **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for a list of prompts synchronously.

        Args:
            tokenized_prompts: List of pre-tokenized prompts (token IDs) - used by most backends
            text_prompts: List of text prompts - used by vLLM and SGLang backends

        Returns:
            List of dictionaries with standardized output format
        """
        pass

    @abstractmethod
    def generate_async(self,
                       tokenized_prompts: Optional[List[List[int]]] = None,
                       text_prompts: Optional[List[str]] = None,
                       **kwargs) -> List[asyncio.Future]:
        """
        Generate responses for a list of prompts asynchronously.

        This method returns immediately with a list of futures that will resolve
        to the generation results.

        Args:
            tokenized_prompts: List of pre-tokenized prompts (token IDs) - used by most backends
            text_prompts: List of text prompts - used by vLLM and SGLang backends

        Returns:
            List of futures that will resolve to dictionaries with standardized output format
        """
        pass

    async def generate_stream(self,
                              tokenized_prompts: Optional[List[List[int]]] = None,
                              text_prompts: Optional[List[str]] = None,
                              **kwargs) -> List[AsyncIterator[StreamingChunk]]:
        """
        Generate responses for a list of prompts with streaming.

        Args:
            tokenized_prompts: List of tokenized prompts
            text_prompts: List of text prompts
            **kwargs: Additional generation parameters

        Returns:
            List of async iterators, one per prompt, yielding StreamingChunk objects

        Raises:
            NotImplementedError: If backend doesn't support streaming
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming generation")

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources and shut down the backend.

        Subclasses should call super().shutdown() at the end to set is_initialized=False,
        or set it manually if they don't call super().
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
