"""QuerySampleLibrary implementations for MLPerf LoadGen."""

import logging
from typing import List, Optional
import mlperf_loadgen as lg


class QuerySampleLibrary:
    """MLPerf QuerySampleLibrary implementation for single-process execution."""

    def __init__(self, dataset: List[List[int]], dataset_strings: List[str],
                 name: str = "QSL"):
        """
        Initialize QSL with dataset.

        Args:
            dataset: List of tokenized prompts
            dataset_strings: List of original prompt strings
            name: Name for logging
        """
        self.dataset = dataset
        self.dataset_strings = dataset_strings
        self.count = len(dataset)
        self.perf_count = self.count
        self.name = name
        self.logger = logging.getLogger(__name__)

        # Create LoadGen QSL
        self.qsl = lg.ConstructQSL(
            self.count,
            self.perf_count,
            lambda x: None,  # LoadSamplesToRam
            lambda x: None   # UnloadSamplesFromRam
        )
        self.logger.info(f"Created {self.name} with {self.count} samples")

    def __del__(self):
        """Cleanup QSL."""
        if self.qsl is not None:
            lg.DestroyQSL(self.qsl)
            self.logger.info(f"{self.name} destroyed")


class DistributedQuerySampleLibrary:
    """QuerySampleLibrary for distributed execution (MPI/torchrun)."""

    def __init__(self, dataset: List[List[int]], dataset_strings: List[str],
                 rank: int, world_size: int, name: str = "DistributedQSL"):
        """
        Initialize distributed QSL.

        Args:
            dataset: List of tokenized prompts
            dataset_strings: List of original prompt strings
            rank: Process rank
            world_size: Total number of processes
            name: Name for logging
        """
        self.dataset = dataset
        self.dataset_strings = dataset_strings
        self.count = len(dataset)
        self.perf_count = self.count
        self.rank = rank
        self.world_size = world_size
        self.name = name
        self.logger = logging.getLogger(__name__)

        # Track if this is rank zero explicitly
        self.is_rank_zero = (self.rank == 0)

        # Only rank 0 creates the actual QSL
        if self.is_rank_zero:
            self.qsl = lg.ConstructQSL(
                self.count,
                self.perf_count,
                lambda x: None,
                lambda x: None
            )
            self.logger.info(
                f"Created {self.name} with {self.count} samples on rank 0")
        else:
            self.qsl = None

    def __del__(self):
        """Cleanup QSL on rank 0."""
        if self.is_rank_zero and self.qsl is not None:
            lg.DestroyQSL(self.qsl)
            self.logger.info(f"{self.name} destroyed on rank 0")
