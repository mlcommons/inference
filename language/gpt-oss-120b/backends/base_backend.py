#!/usr/bin/env python3
"""Base backend class for gpt-oss inference."""

import abc
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseBackend(abc.ABC):
    """Abstract base class for inference backends.

    All backends must implement this interface to work with the MLPerf SUT.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the backend.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        logger.info(f"Initializing {self.__class__.__name__}")

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (load model, connect to server, etc.)."""
        raise NotImplementedError("Subclasses must implement initialize()")

    @abc.abstractmethod
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
            **kwargs: Additional backend-specific parameters

        Returns:
            List of response dictionaries with keys:
                - output_ids: List of generated token IDs
                - output_text: Generated text (optional)
                - metadata: Additional metadata (latencies, etc.)
        """
        raise NotImplementedError("Subclasses must implement generate()")

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Clean up backend resources."""
        raise NotImplementedError("Subclasses must implement cleanup()")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self.initialized
