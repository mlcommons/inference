#!/usr/bin/env python3
"""Server SUT implementation for MLPerf inference benchmarks."""

import asyncio
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Set, AsyncIterator
from dataclasses import dataclass, field
import numpy as np
import mlperf_loadgen as lg
from utils import supports_streaming

from .base_sut import BaseSUT
from backends import BaseBackend
from backends.base_backend import StreamingChunk
from utils.backend_registry import uses_text_input


logger = logging.getLogger(__name__)


@dataclass
class QueryInfo:
    """Information about a single query from LoadGen."""
    query_id: int                     # Unique query ID from LoadGen
    index: int                        # Index into the dataset
    issued_time: float               # When the query was issued
    sample: lg.QuerySample           # Original LoadGen query sample
    result: Optional[Dict[str, Any]] = None  # Results after completion


@dataclass
class StreamingQueryState:
    """Track state of a streaming query."""
    query_info: QueryInfo
    stream_gen: AsyncIterator[StreamingChunk]
    first_token_time: Optional[float] = None
    first_token_sent: bool = False
    accumulated_text: str = ""
    accumulated_tokens: List[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    task: Optional[asyncio.Task] = None  # Track the processing task


class ServerSUT(BaseSUT):
    """Server scenario SUT implementation with streaming support.

    This SUT requires backends to support streaming for server mode.
    All processing happens asynchronously without blocking LoadGen.
    """

    def __init__(self,
                 backend: BaseBackend,
                 dataset: List[List[int]],
                 dataset_strings: Optional[List[str]] = None,
                 name: str = "ServerSUT"):
        """Initialize the server SUT.

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

        # Track active streaming queries
        self.active_streams: Dict[int, StreamingQueryState] = {}
        self.active_streams_lock = asyncio.Lock()

        # Track all active tasks for proper cleanup
        self.active_tasks: Set[asyncio.Task] = set()
        self.active_tasks_lock = asyncio.Lock()

        # Results storage
        self.all_results: Dict[int, Dict[str, Any]] = {}
        self.results_lock = asyncio.Lock()

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries in streaming mode with batching."""
        if not supports_streaming():
            # Fallback to async if streaming not supported
            super().issue_queries(query_samples)
            return

        logger.debug(f"Issuing {len(query_samples)} queries (non-blocking)")

        # Schedule async processing without blocking
        for sample in query_samples:
            query_info = QueryInfo(
                query_id=sample.id,
                index=sample.index,
                issued_time=time.time(),
                sample=sample
            )

            # Fire and forget - start streaming without blocking
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._start_streaming_query(query_info),
                    self.loop
                )

    async def _start_streaming_query(self, query_info: QueryInfo) -> None:
        """Start streaming for a single query."""
        try:
            # Verify streaming support
            if not supports_streaming():
                raise RuntimeError(
                    f"Backend {self.backend_name} does not support streaming required for server mode")

            # Prepare prompt based on backend type
            if self.uses_text_prompts:
                prompt = [self.dataset_strings[query_info.index]]
                stream_gens = await self.backend.generate_stream(text_prompts=prompt)
            else:
                prompt = [self.dataset[query_info.index]]
                stream_gens = await self.backend.generate_stream(tokenized_prompts=prompt)

            # Create streaming state
            state = StreamingQueryState(
                query_info=query_info,
                stream_gen=stream_gens[0]
            )

            # Register active stream
            async with self.active_streams_lock:
                self.active_streams[query_info.query_id] = state

            # Start processing the stream and track the task
            task = asyncio.create_task(self._process_stream(state))
            state.task = task

            # Track the task for proper cleanup
            async with self.active_tasks_lock:
                self.active_tasks.add(task)

            # Add callback to remove task from active set when done
            task.add_done_callback(self._remove_task_from_active)

        except Exception as e:
            logger.error(
                f"Error starting stream for query {query_info.query_id}: {e}")
            raise RuntimeError(
                f"Failed to start streaming for query {query_info.query_id}: {e}")

    def _remove_task_from_active(self, task: asyncio.Task) -> None:
        """Remove a completed task from the active set."""
        # This runs in the event loop thread context
        asyncio.create_task(self._async_remove_task_from_active(task))

    async def _async_remove_task_from_active(self, task: asyncio.Task) -> None:
        """Async version of task removal."""
        async with self.active_tasks_lock:
            self.active_tasks.discard(task)

    async def _process_stream(self, state: StreamingQueryState) -> None:
        """Process a streaming response, reporting first token immediately."""
        try:
            async for chunk in state.stream_gen:
                current_time = time.perf_counter()

                # Accumulate text and tokens
                if chunk.token:
                    state.accumulated_text += chunk.token
                if chunk.token_ids:
                    state.accumulated_tokens.extend(chunk.token_ids)

                # Report first token immediately for TTFT measurement
                if not state.first_token_sent and (
                        chunk.token or chunk.token_ids):
                    state.first_token_time = current_time - state.start_time
                    state.first_token_sent = True

                    # Send first token response to LoadGen immediately
                    # This allows LoadGen to measure TTFT accurately
                    await self._send_first_token_response(state)

                # Check if generation is complete
                if chunk.is_finished:
                    # Send final response with complete text
                    await self._send_final_response(state)
                    break

        except asyncio.CancelledError:
            # Task was cancelled, clean up gracefully
            logger.debug(
                f"Stream processing cancelled for query {state.query_info.query_id}")
            # Close the async generator properly (assume aclose exists in our
            # containerized environment)
            try:
                await state.stream_gen.aclose()
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(
                f"Error processing stream for query {state.query_info.query_id}: {e}")
            raise RuntimeError(
                f"Stream processing failed for query {state.query_info.query_id}: {e}")
        finally:
            # Clean up active stream
            async with self.active_streams_lock:
                self.active_streams.pop(state.query_info.query_id, None)

    async def _send_first_token_response(
            self, state: StreamingQueryState) -> None:
        """Send first token notification to LoadGen for TTFT measurement."""
        logger.debug(
            f"First token received for query {state.query_info.query_id} at {state.first_token_time:.3f}s")

        # Convert first tokens to proper format for LoadGen
        if state.accumulated_tokens:
            output_tokens = np.ascontiguousarray(
                state.accumulated_tokens, dtype=np.int32)
        else:
            # If no token IDs available, encode the text
            if hasattr(self.backend, 'tokenizer') and state.accumulated_text:
                tokens = self.backend.tokenizer.encode(state.accumulated_text)
                output_tokens = np.ascontiguousarray(tokens, dtype=np.int32)
            else:
                raise RuntimeError(
                    f"No token IDs available for first token response for query {state.query_info.query_id}")

        output_seq_len = len(output_tokens)
        output_toks_ptr = output_tokens.ctypes.data if output_seq_len > 0 else 0
        output_toks_size = output_seq_len * output_tokens.itemsize

        # Send first token response using LoadGen API
        lg.FirstTokenComplete([
            lg.QuerySampleResponse(
                state.query_info.query_id,
                output_toks_ptr,
                output_toks_size,
                output_seq_len
            )
        ])

    async def _send_final_response(self, state: StreamingQueryState) -> None:
        """Send final response with complete generated text."""
        try:
            # For MLPerf server mode, we need complete tokens
            if state.accumulated_tokens:
                # Create a copy of tokens before numpy conversion
                tokens_to_send = state.accumulated_tokens.copy()
                token_array = np.array(
                    state.accumulated_tokens, dtype=np.int32)
            else:
                # If no tokens, encode the text
                if hasattr(self.backend,
                           'tokenizer') and state.accumulated_text:
                    tokens = self.backend.tokenizer.encode(
                        state.accumulated_text)
                    # Create a copy of tokens before numpy conversion
                    tokens_to_send = tokens.copy()
                    token_array = np.array(tokens, dtype=np.int32)
                else:
                    raise RuntimeError(
                        f"No tokens or tokenizer available for query {state.query_info.query_id}")

            # Validate we have tokens
            if len(token_array) == 0:
                raise RuntimeError(
                    f"No tokens generated for query {state.query_info.query_id}")

            # Create LoadGen response
            response = lg.QuerySampleResponse(
                state.query_info.query_id,
                token_array.ctypes.data if token_array.size > 0 else 0,
                token_array.nbytes,
                len(token_array)
            )

            # Send response to LoadGen
            lg.QuerySamplesComplete([response])

            # Store result for later retrieval with the tokens copy
            async with self.results_lock:
                state.query_info.result = {
                    'text': state.accumulated_text,
                    'tokens': tokens_to_send,  # Use the copy
                    'first_token_time': state.first_token_time,
                    'total_time': time.perf_counter() - state.start_time,
                    'index': state.query_info.index  # Store the dataset index
                }
                self.all_results[state.query_info.query_id] = state.query_info.result

            logger.debug(
                f"Sent {len(token_array)} tokens to LoadGen for query {state.query_info.query_id}")

        except Exception as e:
            logger.error(
                f"Error sending final response for query {state.query_info.query_id}: {e}")
            raise RuntimeError(
                f"Failed to send final response for query {state.query_info.query_id}: {e}")

    def flush_queries(self) -> None:
        """Wait for all active streams to complete."""
        logger.info("Flushing queries...")

        # Create a task to wait for all streams
        async def wait_for_streams():
            max_wait = 300  # 5 minutes timeout
            start_time = time.time()

            while time.time() - start_time < max_wait:
                async with self.active_streams_lock:
                    if not self.active_streams:
                        break
                    active_count = len(self.active_streams)

                logger.debug(f"Waiting for {active_count} active streams...")
                await asyncio.sleep(0.1)

            async with self.active_streams_lock:
                if self.active_streams:
                    logger.warning(
                        f"Timeout: {len(self.active_streams)} streams still active")

        # Run the wait task in the event loop
        if self.loop and not self.loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                wait_for_streams(), self.loop)
            try:
                # Slightly longer than internal timeout
                future.result(timeout=310)
            except Exception as e:
                logger.error(f"Error waiting for streams to complete: {e}")

    def _run_event_loop(self):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)

        # Run until stopped
        self.loop.run_forever()

    def start(self) -> lg.ConstructSUT:
        """Start the SUT and async event loop."""
        # Create new event loop
        self.loop = asyncio.new_event_loop()

        # Start event loop thread
        self.loop_thread = threading.Thread(target=self._run_event_loop)
        self.loop_thread.start()

        # Call parent start
        return super().start()

    def stop(self) -> None:
        """Stop the SUT and clean up."""
        # Cancel all active tasks before stopping the loop
        if self.loop and self.loop_thread:
            # Cancel all active tasks
            async def cancel_all_tasks():
                async with self.active_tasks_lock:
                    tasks_to_cancel = list(self.active_tasks)

                if tasks_to_cancel:
                    logger.info(
                        f"Cancelling {len(tasks_to_cancel)} active streaming tasks...")
                    for task in tasks_to_cancel:
                        task.cancel()

                    # Wait for all tasks to complete cancellation
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                    logger.info("All streaming tasks cancelled")

                # Clear the active tasks set
                async with self.active_tasks_lock:
                    self.active_tasks.clear()

            # Run the cancellation in the event loop
            future = asyncio.run_coroutine_threadsafe(
                cancel_all_tasks(), self.loop)
            try:
                future.result(timeout=10.0)  # Give tasks time to cancel
            except Exception as e:
                logger.error(f"Error cancelling tasks: {e}")

            # Now stop the loop
            self.loop.call_soon_threadsafe(self.loop.stop)

            # Wait for thread to finish
            self.loop_thread.join(timeout=5.0)

            # Close loop
            if not self.loop.is_closed():
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

        # Create index to result mapping from all stored results
        index_to_result = {}

        # Map results by their stored dataset index
        for query_id, result in self.all_results.items():
            if result and 'index' in result:
                index_to_result[result['index']] = result

        # Only process results for samples that were actually queried
        # Sort by index to maintain dataset order
        queried_indices = sorted(index_to_result.keys())

        logger.info(
            f"Retrieving results for {len(queried_indices)} queried samples")

        # Process results in order of dataset indices using stored backend
        # results
        for i in queried_indices:
            result = index_to_result[i]

            # Get tokens from the backend result
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

        return ordered_results
