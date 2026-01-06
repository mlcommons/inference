#!/usr/bin/env python3
"""Offline scenario SUT implementation for gpt-oss."""

import logging
import numpy as np
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlperf_loadgen as lg
from tqdm import tqdm
from .base_sut import BaseSUT

logger = logging.getLogger(__name__)


class OfflineSUT(BaseSUT):
    """Offline scenario System Under Test.

    In the Offline scenario, all queries are issued at once and can be
    processed in any order. This allows for maximum batching and throughput.
    """

    def __init__(
        self,
        backend,
        dataset: List[List[int]],
        max_tokens: int = 32768,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        name: str = "OfflineSUT",
        progress_bar=None,
        max_concurrency: int = 128
    ):
        """Initialize the Offline SUT.

        Args:
            backend: Backend instance for inference
            dataset: List of tokenized prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            name: Name of the SUT
            progress_bar: Optional tqdm progress bar for real-time updates
            max_concurrency: Maximum concurrent requests to backend (SGLang does in-flight batching)
        """
        super().__init__(backend, dataset, name, progress_bar)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.pending_queries = []
        self.max_concurrency = max_concurrency

        logger.info(
            f"OfflineSUT configured with max_concurrency={max_concurrency} (backend handles batching)")

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries to the SUT.

        In Offline mode, we accumulate all queries and process them in batch.

        Args:
            query_samples: List of MLPerf LoadGen query samples
        """
        logger.info(f"Received {len(query_samples)} queries")

        # Update progress bar total by accumulating (for repeats_per_sample > 1)
        # LoadGen may call issue_queries multiple times for repeated sampling
        if self.progress_bar is not None:
            self.progress_bar.total = (
                self.progress_bar.total or 0) + len(query_samples)
            self.progress_bar.refresh()

        # Store queries for batch processing
        for qs in query_samples:
            self.pending_queries.append(qs)

    def flush_queries(self) -> None:
        """Process all accumulated queries with concurrent requests.

        Sends individual requests concurrently up to max_concurrency limit.
        SGLang handles batching internally via continuous batching.
        """
        if not self.pending_queries:
            logger.info("No pending queries to flush")
            return

        logger.info(
            f"Flushing {len(self.pending_queries)} queries with max_concurrency={self.max_concurrency}")
        start_time = time.time()

        def process_single_query(query_sample):
            """Process a single query (backend batches automatically via continuous batching)."""
            # Check if we should stop (e.g., KeyboardInterrupt)
            if self.should_stop.is_set():
                logger.info(
                    f"Skipping query {query_sample.id} due to shutdown")
                return None, None, None

            query_id = query_sample.id
            input_ids = self.dataset[query_sample.index]

            # Call backend with single query
            # SGLang will batch this with other concurrent requests
            # automatically
            responses = self.backend.generate(
                prompts=[input_ids],  # Single query as list
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )

            return query_id, query_sample, responses[0]

        try:
            # Process queries in parallel with max_concurrency
            logger.info(
                f"Submitting {len(self.pending_queries)} queries to {self.max_concurrency} concurrent workers...")
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                # Submit all queries at once
                futures = [
                    executor.submit(
                        process_single_query,
                        qs) for qs in self.pending_queries]

                # Process results as they complete
                completed_count = 0
                cancelled_count = 0

                for future in as_completed(futures):
                    # Check if shutdown was requested
                    if self.should_stop.is_set():
                        logger.info(
                            "Shutdown requested, cancelling remaining futures...")
                        for f in futures:
                            f.cancel()
                        cancelled_count = sum(
                            1 for f in futures if f.cancelled())
                        logger.info(
                            f"Cancelled {cancelled_count} pending futures")
                        break
                    try:
                        query_id, query_sample, response = future.result()

                        # Skip if query was cancelled/skipped
                        if query_id is None:
                            continue

                        output_ids = response.get("output_ids", [])

                        # Store results
                        self.results[query_id] = {
                            "output_ids": output_ids,
                            "output_text": response.get("output_text", ""),
                            "metadata": response.get("metadata", {})
                        }

                        # Convert output_ids to numpy array for LoadGen
                        # LoadGen expects int32 token IDs as a contiguous array
                        if output_ids:
                            token_array = np.ascontiguousarray(
                                output_ids, dtype=np.int32)
                            output_data_ptr = token_array.ctypes.data
                            output_data_size = token_array.nbytes
                            n_tokens = len(output_ids)
                        else:
                            # Empty response
                            token_array = np.array([], dtype=np.int32)
                            output_data_ptr = 0
                            output_data_size = 0
                            n_tokens = 0

                        # Create response for LoadGen with token count
                        response_array = [
                            lg.QuerySampleResponse(
                                query_id,
                                output_data_ptr,
                                output_data_size,
                                n_tokens  # Number of output tokens for tokens/sec metric
                            )
                        ]

                        # Report completion to LoadGen
                        lg.QuerySamplesComplete(response_array)

                        # Update progress bar
                        if self.progress_bar is not None:
                            self.progress_bar.update(1)
                            self.progress_bar.refresh()

                        completed_count += 1
                        # Log progress at debug level only (tqdm shows
                        # progress)
                        if completed_count % 100 == 0:
                            logger.debug(
                                f"Completed {completed_count}/{len(self.pending_queries)} queries")

                    except Exception as e:
                        logger.error(
                            f"Error processing query: {e}", exc_info=True)

            elapsed = time.time() - start_time
            if cancelled_count > 0:
                logger.info(
                    f"Completed {completed_count} queries, cancelled {cancelled_count} queries "
                    f"in {elapsed:.2f}s"
                )
            else:
                logger.info(
                    f"Completed {len(self.pending_queries)} queries in {elapsed:.2f}s "
                    f"({len(self.pending_queries)/elapsed:.2f} QPS)"
                )

        except Exception as e:
            logger.error(f"Error during concurrent flush: {e}", exc_info=True)
            raise
        finally:
            # Clear pending queries
            self.pending_queries = []
