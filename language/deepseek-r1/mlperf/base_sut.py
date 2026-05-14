#!/usr/bin/env python3
"""Base System Under Test (SUT) class for MLPerf inference benchmarks."""

import abc
import logging
from typing import List, Dict, Any, Optional
import mlperf_loadgen as lg


logger = logging.getLogger(__name__)


class BaseSUT(abc.ABC):
    """Base class for MLPerf inference System Under Test (SUT).

    This class defines the interface that all SUTs must implement for MLPerf
    inference benchmarks. It provides two main methods:
    - issue_queries: to enqueue prompt tokens
    - flush_queries: to await completion of all issued queries
    """

    def __init__(self, name: str = "BaseSUT"):
        """Initialize the base SUT.

        Args:
            name: Name of the SUT for logging purposes
        """
        self.name = name
        self.sut = None
        logger.info(f"Initializing {self.name}")

    @abc.abstractmethod
    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries to the SUT.

        This method should enqueue the provided query samples for processing.
        It should return immediately without waiting for completion.

        Args:
            query_samples: List of MLPerf LoadGen query samples to process
        """
        raise NotImplementedError("Subclasses must implement issue_queries")

    @abc.abstractmethod
    def flush_queries(self) -> None:
        """Flush all pending queries.

        This method should wait for all previously issued queries to complete
        before returning. It's called by LoadGen to ensure all work is done.
        """
        raise NotImplementedError("Subclasses must implement flush_queries")

    def start(self) -> lg.ConstructSUT:
        """Start the SUT and return the LoadGen SUT handle.

        Returns:
            LoadGen SUT handle for use with LoadGen
        """
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        logger.info(f"{self.name} started")
        return self.sut

    def stop(self) -> None:
        """Stop the SUT and clean up resources."""
        if self.sut:
            lg.DestroySUT(self.sut)
            self.sut = None
            logger.info(f"{self.name} stopped")

    def __enter__(self):
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
