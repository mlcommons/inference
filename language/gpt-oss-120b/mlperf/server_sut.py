#!/usr/bin/env python3
"""Server scenario SUT implementation with streaming support for gpt-oss."""

import asyncio
import logging
import numpy as np
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import mlperf_loadgen as lg
from tqdm import tqdm

from .base_sut import BaseSUT

logger = logging.getLogger(__name__)


@dataclass
class StreamingQueryState:
    """State for a streaming query."""
    query_sample: lg.QuerySample
    query_id: int
    input_ids: List[int]
    accumulated_tokens: List[int]
    accumulated_text: str
    first_token_received: bool
    first_token_time: Optional[float]
    start_time: float
    finished: bool


class ServerSUT(BaseSUT):
    """Server scenario SUT with streaming support.

    Properly reports FirstTokenComplete and QuerySamplesComplete to LoadGen.
    """

    def __init__(
        self,
        backend,
        dataset: List[List[int]],
        max_tokens: int = 32768,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        num_workers: int = 1,
        name: str = "ServerSUT",
        progress_bar=None
    ):
        """Initialize the Server SUT.

        Args:
            backend: Backend instance for inference (must support streaming)
            dataset: List of tokenized prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_workers: Number of worker threads
            name: Name of the SUT
            progress_bar: Optional tqdm progress bar for real-time updates
        """
        super().__init__(backend, dataset, name, progress_bar)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_workers = num_workers

        # Query queue and streaming state
        self.query_queue = queue.Queue()
        self.active_streams: Dict[int, StreamingQueryState] = {}
        self.active_streams_lock = threading.Lock()

        # Track active async tasks for cancellation on KeyboardInterrupt
        self.active_tasks = set()
        self.active_tasks_lock = threading.Lock()

        # Worker threads
        self.workers = []

        # Progress tracking
        self.queries_completed = 0
        self.progress_lock = threading.Lock()

        # Event loop for async streaming
        self.loop = None
        self.loop_thread = None

        logger.info(
            f"ServerSUT configured with num_workers={num_workers} (streaming enabled)")

    def start(self) -> lg.ConstructSUT:
        """Start the SUT and worker threads."""
        # Start event loop thread for async streaming
        self._start_event_loop()

        # Start worker threads
        self._start_workers()

        # Create LoadGen SUT
        self.sut = lg.ConstructSUT(
            self.issue_queries,
            self.flush_queries)
        logger.info(f"{self.name} started with streaming support")
        return self.sut

    def _start_event_loop(self):
        """Start the asyncio event loop in a separate thread."""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.001)

        logger.info("Async event loop started")

    def _start_workers(self):
        """Start worker threads for processing queries."""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"ServerWorker-{i}",
                daemon=True
            )
            self.workers.append(worker)
            worker.start()
        logger.info(f"Started {self.num_workers} worker threads")

    def _worker_thread(self):
        """Worker thread that processes queries from the queue."""
        try:
            while not self.should_stop.is_set():
                try:
                    query_sample = self.query_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    logger.info(
                        "Worker thread interrupted, exiting gracefully...")
                    break

                # Schedule async streaming processing and track task
                if self.loop and not self.should_stop.is_set():
                    # Create the coroutine
                    coro = self._process_streaming_query_tracked(query_sample)
                    # Schedule it on the event loop
                    future = asyncio.run_coroutine_threadsafe(coro, self.loop)
                    # Don't wait for completion - it happens asynchronously

        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)

    async def _process_streaming_query_tracked(
            self, query_sample: lg.QuerySample):
        """Wrapper that tracks the async task for cancellation."""
        task = asyncio.current_task()

        # Add to active tasks
        with self.active_tasks_lock:
            self.active_tasks.add(task)

        try:
            await self._process_streaming_query(query_sample)
        finally:
            # Remove from active tasks
            with self.active_tasks_lock:
                self.active_tasks.discard(task)

    async def _process_streaming_query(self, query_sample: lg.QuerySample):
        """Process a single query with streaming support.

        Token reporting to LoadGen:
        1. When first token arrives → lg.FirstTokenComplete([token_0])
        2. When generation finishes → lg.QuerySamplesComplete([token_1, token_2, ..., token_n])
        Args:
            query_sample: MLPerf LoadGen query sample
        """
        query_id = query_sample.id
        sample_idx = query_sample.index
        input_ids = self.dataset[sample_idx]

        # Initialize streaming state
        state = StreamingQueryState(
            query_sample=query_sample,
            query_id=query_id,
            input_ids=input_ids,
            accumulated_tokens=[],
            accumulated_text="",
            first_token_received=False,
            first_token_time=None,
            start_time=time.time(),
            finished=False
        )

        with self.active_streams_lock:
            self.active_streams[query_id] = state

        try:
            # Stream tokens from backend
            async for chunk in self.backend.generate_stream(
                input_ids=input_ids,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            ):
                # Update state
                if chunk.get("delta_token_ids"):
                    state.accumulated_tokens.extend(chunk["delta_token_ids"])
                if chunk.get("delta_text"):
                    state.accumulated_text += chunk["delta_text"]

                # Send FirstTokenComplete on first token
                if chunk.get(
                        "is_first_token") and not state.first_token_received:
                    state.first_token_received = True
                    state.first_token_time = time.time()
                    await self._send_first_token_complete(state)

                # Check if finished
                if chunk.get("is_finished"):
                    state.finished = True
                    await self._send_final_response(state)
                    break

            # If no explicit finish signal, send final response
            if not state.finished:
                state.finished = True
                await self._send_final_response(state)

        except asyncio.CancelledError:
            # Task was cancelled (e.g., KeyboardInterrupt during graceful
            # shutdown)
            logger.info(
                f"Streaming query {query_id} cancelled during shutdown")
            # Don't send response to LoadGen - we're shutting down
            raise  # Re-raise to mark task as cancelled
        except Exception as e:
            logger.error(
                f"Error processing streaming query {query_id}: {e}",
                exc_info=True)
            # Send empty response to unblock LoadGen
            try:
                await self._send_final_response(state)
            except BaseException:
                pass
        finally:
            # Clean up
            with self.active_streams_lock:
                self.active_streams.pop(query_id, None)

    async def _send_first_token_complete(self, state: StreamingQueryState):
        """Send FirstTokenComplete to LoadGen for TTFT measurement.

        Only sends the first token for TTFT measurement.
        """
        try:
            logger.debug(
                f"First token for query {state.query_id} at {state.first_token_time - state.start_time:.3f}s")

            # LoadGen uses this to measure Time To First Token (TTFT)
            if state.accumulated_tokens and len(state.accumulated_tokens) > 0:
                # Extract only the first token
                first_token_only = [state.accumulated_tokens[0]]
                token_array = np.ascontiguousarray(
                    first_token_only, dtype=np.int32)
            else:
                # No tokens yet - this shouldn't happen but handle gracefully
                token_array = np.array([], dtype=np.int32)
                logger.warning(
                    f"FirstTokenComplete called but no tokens accumulated for query {state.query_id}")

            # Create response
            response = lg.QuerySampleResponse(
                state.query_id,
                token_array.ctypes.data if token_array.size > 0 else 0,
                token_array.nbytes,
                len(token_array)
            )

            # Report to LoadGen
            lg.FirstTokenComplete([response])
            logger.debug(
                f"Sent FirstTokenComplete for query {state.query_id}: 1 token")

        except Exception as e:
            logger.error(
                f"Error sending FirstTokenComplete for query {state.query_id}: {e}",
                exc_info=True)

    async def _send_final_response(self, state: StreamingQueryState):
        """Send final QuerySamplesComplete to LoadGen. (send all tokens except the first one)
        """
        try:
            num_total_tokens = len(state.accumulated_tokens)
            logger.debug(
                f"Final response for query {state.query_id}: {num_total_tokens} total tokens")

            # Store results (all tokens for internal tracking)
            self.results[state.query_id] = {
                "output_ids": state.accumulated_tokens,
                "output_text": state.accumulated_text,
                "metadata": {
                    "latency": time.time() - state.start_time,
                    "ttft": state.first_token_time - state.start_time if state.first_token_time else None,
                }
            }

            if state.accumulated_tokens and len(state.accumulated_tokens) > 1:
                remaining_tokens = state.accumulated_tokens[1:]
                token_array = np.ascontiguousarray(
                    remaining_tokens, dtype=np.int32)
            else:
                token_array = np.array([], dtype=np.int32)

            # Create response
            response = lg.QuerySampleResponse(
                state.query_id,
                token_array.ctypes.data if token_array.size > 0 else 0,
                token_array.nbytes,
                len(token_array)
            )

            # Report to LoadGen
            lg.QuerySamplesComplete([response])
            logger.debug(
                f"Sent QuerySamplesComplete for query {state.query_id}: "
                f"{len(token_array)} remaining tokens (total: {num_total_tokens})"
            )

            # Update progress bar (force refresh for async updates)
            if self.progress_bar is not None:
                with self.progress_lock:
                    self.queries_completed += 1
                    self.progress_bar.update(1)
                    self.progress_bar.refresh()  # Force redraw from async context
                    sys.stdout.flush()  # Force flush for immediate display in async/threaded context

        except Exception as e:
            logger.error(
                f"Error sending final response for query {state.query_id}: {e}",
                exc_info=True)

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries to the SUT.

        In Server mode, queries are added to a queue for worker threads.

        Args:
            query_samples: List of MLPerf LoadGen query samples
        """
        # Update progress bar total dynamically as queries arrive
        if self.progress_bar is not None:
            with self.progress_lock:
                self.progress_bar.total = (
                    self.progress_bar.total or 0) + len(query_samples)
                self.progress_bar.refresh()

        for qs in query_samples:
            self.query_queue.put(qs)

    def flush_queries(self) -> None:
        """Flush all pending queries.

        Wait for all issued queries to complete.
        """
        logger.info("Flushing server queries...")

        # Wait for queue to empty and all streams to complete
        while True:
            queue_empty = self.query_queue.empty()

            with self.active_streams_lock:
                no_active_streams = len(self.active_streams) == 0

            if queue_empty and no_active_streams:
                break

            time.sleep(0.01)

        logger.info("Server queries flushed")

    def stop(self) -> None:
        """Stop the SUT and clean up resources."""
        if self.should_stop.is_set():
            logger.info(f"{self.name} already stopping or stopped.")
            return

        super().stop()

        # Cancel all active streaming tasks
        logger.info("Cancelling active streaming tasks...")
        tasks_to_cancel = []
        with self.active_tasks_lock:
            tasks_to_cancel = list(self.active_tasks)

        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} active tasks")
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()

        # Clear pending queries from queue
        pending_count = 0
        try:
            while True:
                self.query_queue.get_nowait()
                pending_count += 1
        except queue.Empty:
            pass

        if pending_count > 0:
            logger.info(f"Cleared {pending_count} pending queries from queue")

        # Wait for workers with progress bar
        with tqdm(total=len(self.workers), desc="Stopping workers", unit="worker") as pbar:
            for i, worker in enumerate(self.workers):
                worker.join(timeout=5)
                if worker.is_alive():
                    logger.warning(
                        f"Worker {i+1} did not terminate gracefully")
                pbar.update(1)

        # Stop event loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=2)

        logger.info("All workers stopped")

        # Destroy LoadGen SUT
        super().stop()
