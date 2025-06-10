#!/usr/bin/env python3
"""Offline SUT implementation for MLPerf inference benchmarks."""

import asyncio
import logging
import threading
from typing import List, Dict, Any, Optional
import numpy as np
import mlperf_loadgen as lg

from .base_sut import BaseSUT
from backends import BaseBackend
from utils.backend_registry import uses_text_input


logger = logging.getLogger(__name__)


class OfflineSUT(BaseSUT):
    """Offline scenario SUT implementation.

    This SUT uses async inference APIs similar to run_eval.py pattern:
    - Collects all queries into a batch
    - Calls backend.generate_async() once with all prompts
    - Processes futures as they complete out-of-order
    """

    def __init__(self,
                 backend: BaseBackend,
                 dataset: List[List[int]],
                 dataset_strings: Optional[List[str]] = None,
                 name: str = "OfflineSUT"):
        """Initialize the offline SUT.

        Args:
            backend: Backend instance to use for inference
            dataset: List of tokenized prompts
            dataset_strings: List of original text prompts (for backends that use text)
            name: Name of the SUT
        """
        super().__init__(name)
        self.backend = backend
        self.dataset = dataset
        self.dataset_strings = dataset_strings

        # Determine backend type using registry
        self.backend_name = getattr(
            backend,
            'backend_name',
            type(backend).__name__.lower())
        self.uses_text_prompts = uses_text_input(self.backend_name)

        if self.uses_text_prompts and dataset_strings is None:
            raise ValueError(
                f"Backend {self.backend_name} requires text prompts but dataset_strings was not provided")

        # Async event loop and thread
        self.loop = None
        self.loop_thread = None

        # Query collection for batch processing (like run_eval.py)
        self.query_samples = []
        self.queries_lock = threading.Lock()
        self.batch_processed = threading.Event()

        # Results storage
        self.results = {}

        # Track index to sample ID mapping
        self.index_to_id = {}

        # Initialize batch future to None
        self.batch_future = None

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries - collect all queries for batch processing like run_eval.py.

        Args:
            query_samples: List of MLPerf LoadGen query samples
        """
        logger.info(f"Issuing {len(query_samples)} queries")

        # Store queries for batch processing
        with self.queries_lock:
            self.query_samples = query_samples
            # Track index to ID mapping
            for sample in query_samples:
                self.index_to_id[sample.index] = sample.id

        # Schedule batch processing in the async loop (like run_eval.py)
        future = asyncio.run_coroutine_threadsafe(
            self._process_all_queries_async(),
            self.loop
        )

        # Store the future so we can wait for it in flush_queries
        self.batch_future = future

    def flush_queries(self) -> None:
        """Wait for all queries to complete."""
        logger.info("Flushing queries...")

        # Wait for the batch processing to complete
        if self.batch_future is not None:
            try:
                self.batch_future.result()  # Wait for completion
                logger.info("Batch processing completed")
            except Exception as e:
                logger.error(f"Error waiting for batch completion: {e}")
            finally:
                # Reset batch_future
                self.batch_future = None

    async def _process_all_queries_async(self):
        """Process all queries in a single batch using run_eval.py pattern."""
        try:
            with self.queries_lock:
                query_samples = self.query_samples.copy()

            if not query_samples:
                logger.warning("No queries to process")
                return

            logger.info(f"Processing {len(query_samples)} queries in batch")

            # Prepare prompts for batch processing (like run_eval.py)
            if self.uses_text_prompts:
                # Use text prompts for vLLM and SGLang
                prompts = [self.dataset_strings[sample.index]
                           for sample in query_samples]
                futures = self.backend.generate_async(text_prompts=prompts)
            else:
                # Use tokenized prompts for other backends
                prompts = [self.dataset[sample.index]
                           for sample in query_samples]
                futures = self.backend.generate_async(
                    tokenized_prompts=prompts)

            logger.info(f"Got {len(futures)} futures from backend")

            # Process futures as they complete (same pattern as run_eval.py)
            results = [None] * len(futures)
            indexed_futures = [(i, future) for i, future in enumerate(futures)]
            completed_indices = set()

            # Use asyncio.wait with FIRST_COMPLETED to handle out-of-order
            # completion
            pending = {future for _, future in indexed_futures}

            while pending:
                # Wait for at least one future to complete
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                # Process all completed futures in this batch
                for completed_future in done:
                    # Find the original index for this completed future
                    original_idx = None
                    for idx, future in indexed_futures:
                        if future is completed_future:
                            original_idx = idx
                            break

                    if original_idx is None:
                        logger.error(
                            "Could not find original index for completed future")
                        continue

                    # Check for duplicate completion
                    if original_idx in completed_indices:
                        logger.warning(
                            f"Prompt {original_idx} completed multiple times!")
                        continue

                    try:
                        # Get the result from the completed future
                        result = await completed_future

                        # Store the result in the correct position
                        results[original_idx] = result
                        completed_indices.add(original_idx)

                        # Send result to LoadGen immediately
                        sample = query_samples[original_idx]
                        await self._send_result_to_loadgen(sample, result)

                    except Exception as e:
                        logger.error(
                            f"Error processing prompt {original_idx}: {type(e).__name__}: {e}")
                        # Raise the error instead of handling empty responses
                        raise RuntimeError(
                            f"Backend failed to generate tokens for prompt {original_idx}: {e}")

            # Verify all results are populated
            if len(completed_indices) != len(futures):
                missing_count = len(futures) - len(completed_indices)
                raise RuntimeError(
                    f"Missing results: completed {len(completed_indices)} != {len(futures)} total ({missing_count} missing)")

            for i, result in enumerate(results):
                if result is None:
                    raise RuntimeError(f"Missing result for prompt {i}")

            logger.info(
                f"Completed all {len(completed_indices)} prompts successfully")

        except Exception as e:
            logger.error(
                f"Error during batch processing: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise instead of sending empty responses

    async def _send_result_to_loadgen(
            self, sample: lg.QuerySample, result: Dict[str, Any]):
        """Send a single result to LoadGen."""
        try:
            # Validate that tokens exist - raise error if missing
            tokens = result.get('tokens')
            if tokens is None:
                raise ValueError(
                    f"Backend result missing 'tokens' key for query {sample.id}")
            if not isinstance(tokens, (list, tuple)) or len(tokens) == 0:
                raise ValueError(
                    f"Backend returned empty or invalid tokens for query {sample.id}: {tokens}")

            # Create a copy of tokens before numpy conversion
            tokens_copy = tokens.copy()

            # Convert tokens to bytes for LoadGen
            token_array = np.array(tokens, dtype=np.int32)
            n_tokens = len(tokens)

            # Create LoadGen response
            response = lg.QuerySampleResponse(
                sample.id,
                token_array.ctypes.data if token_array.size > 0 else 0,
                token_array.nbytes,
                n_tokens,
            )

            # Store result with the tokens copy for later access
            self.results[sample.id] = {
                'tokens': tokens_copy,
                'text': result.get('text', '')
            }

            # Send response to LoadGen
            lg.QuerySamplesComplete([response])

            logger.debug(
                f"Sent {n_tokens} tokens to LoadGen for query {sample.id}")

        except Exception as e:
            logger.error(
                f"Error sending result to LoadGen for query {sample.id}: {e}")
            # Raise the error instead of sending empty response
            raise RuntimeError(
                f"Failed to send result to LoadGen for query {sample.id}: {e}")

    def _run_event_loop(self):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> lg.ConstructSUT:
        """Start the SUT and async event loop."""
        # Create and start event loop in separate thread
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop)
        self.loop_thread.start()

        # Call parent start
        return super().start()

    def stop(self) -> None:
        """Stop the SUT and clean up."""
        # Stop the event loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

        # Wait for thread to finish
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join()

        # Close the loop
        if self.loop:
            self.loop.close()
            self.loop = None

        # Call parent stop
        super().stop()

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results in order of dataset indices using stored backend results.

        Returns:
            List of result dictionaries with model_output, tok_model_output, and tok_model_output_len
        """
        # Create a list to hold results in dataset order
        ordered_results = []

        # Only process results for samples that were actually queried
        # Sort by index to maintain dataset order
        queried_indices = sorted(self.index_to_id.keys())

        logger.info(
            f"Retrieving results for {len(queried_indices)} queried samples")

        # Process results in order of dataset indices using stored results
        for i in queried_indices:
            # Get the sample ID for this index
            sample_id = self.index_to_id[i]

            # Get the stored result from backend
            result = self.results.get(sample_id, {})

            if result:
                tokens = result['tokens']
                output_text = result.get('text', '')
                if not output_text and self.backend.tokenizer:
                    output_text = self.backend.tokenizer.decode(
                        result['tokens'], skip_special_tokens=True)

                ordered_results.append({
                    'model_output': output_text,
                    'tok_model_output': tokens,
                    'tok_model_output_len': len(tokens)
                })
            else:
                # No backend result for this sample
                raise RuntimeError(
                    f"No backend result stored for dataset index {i}, sample_id {sample_id}")

        return ordered_results
