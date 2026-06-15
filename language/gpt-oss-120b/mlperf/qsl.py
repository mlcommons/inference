#!/usr/bin/env python3
"""Query Sample Library for gpt-oss MLPerf integration."""

import logging
from typing import List
import mlperf_loadgen as lg

logger = logging.getLogger(__name__)


class QuerySampleLibrary:
    """Query Sample Library implementation.

    This class manages the dataset of samples that LoadGen will query.
    """

    def __init__(self, dataset: List[List[int]]):
        """Initialize the Query Sample Library.

        Args:
            dataset: List of tokenized prompts (list of token ID lists)
        """
        self.dataset = dataset
        self.qsl = None
        logger.info(f"Initializing QSL with {len(dataset)} samples")

    def load_query_samples(self, sample_indices: List[int]) -> None:
        """Load specified query samples into memory.

        Args:
            sample_indices: List of sample indices to load
        """
        # For this implementation, all samples are already in memory
        logger.info(f"Loading {len(sample_indices)} query samples")

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        """Unload specified query samples from memory.

        Args:
            sample_indices: List of sample indices to unload
        """
        # For this implementation, we keep all samples in memory
        logger.info(f"Unloading {len(sample_indices)} query samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __enter__(self):
        """Context manager entry."""
        self.qsl = lg.ConstructQSL(
            len(self.dataset),
            len(self.dataset),  # performance sample count
            self.load_query_samples,
            self.unload_query_samples
        )
        logger.info("QSL constructed")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.qsl:
            lg.DestroyQSL(self.qsl)
            self.qsl = None
            logger.info("QSL destroyed")
