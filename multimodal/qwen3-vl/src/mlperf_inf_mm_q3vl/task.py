"""Task definitions for the Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

import array
import asyncio
import base64
import json
import random
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import httpx
import mlperf_loadgen as lg
from datasets import load_dataset
from loguru import logger
from openai import AsyncOpenAI, DefaultAioHttpClient
from pympler import asizeof

from .schema import (
    ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES,
    MAX_NUM_ESTIMATION_PERFORMANCE_SAMPLES,
    Dataset,
    Endpoint,
    LoadedSample,
    ProductMetadata,
    TestScenario,
    TestSettings,
)


class Task(ABC):
    """Abstract base class for defining a task."""

    def __init__(
        self,
        dataset: Dataset,
        endpoint: Endpoint,
        settings: TestSettings,
        random_seed: int = 12345,
    ) -> None:
        """Initialize the task.

        Args:
            dataset: The dataset configuration passed in from the CLI.
            endpoint: The endpoint configuration passed in from the CLI.
            settings: Parameters of the current benchmark.
            random_seed: The random seed to use for the task.
        """
        random.seed(random_seed)
        self.dataset = load_dataset(
            dataset.repo_id,
            token=dataset.token,
            revision=dataset.revision,
            split="+".join(dataset.split),
        )
        logger.info(
            "Imported {} samples from the dataset splits {}.",
            len(self.dataset),
            dataset.split,
        )
        self.endpoint = endpoint
        request_timeout_seconds = endpoint.request_timeout.total_seconds()
        self.openai_api_client = AsyncOpenAI(
            base_url=endpoint.url,
            http_client=DefaultAioHttpClient(
                timeout=httpx.Timeout(
                    timeout=request_timeout_seconds, connect=5.0),
            ),
            api_key=endpoint.api_key,
            timeout=request_timeout_seconds,
        )
        self.event_loop, self.event_loop_thread = (
            self._create_event_loop_in_separate_thread()
        )
        self.loaded_samples: dict[int, LoadedSample] = {}
        self.settings = settings

    def __del__(self) -> None:
        """Clean up the resources used by the task."""
        logger.trace("Cleaning up the resources used by the task...")

        # Cancel all pending tasks in the event loop
        async def _cancel_all_tasks() -> None:
            """Cancel all pending tasks in the event loop."""
            tasks = [
                task
                for task in asyncio.all_tasks(self.event_loop)
                if task is not asyncio.current_task() and not task.done()
            ]
            for task in tasks:
                task.cancel()
            # Wait briefly for tasks to cancel
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            # Cancel any pending tasks first
            asyncio.run_coroutine_threadsafe(
                _cancel_all_tasks(),
                self.event_loop,
            ).result(timeout=5.0)
        except Exception as e:  # noqa: BLE001
            logger.trace("Error cancelling tasks during cleanup: {}", e)

        # Try to close the OpenAI client gracefully
        try:
            asyncio.run_coroutine_threadsafe(
                self.openai_api_client.close(),
                self.event_loop,
            ).result(timeout=5.0)
        except Exception as e:  # noqa: BLE001
            logger.trace("Error closing OpenAI client during cleanup: {}", e)

        # Stop the event loop and join the thread
        self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        self.event_loop_thread.join(timeout=5.0)

    @staticmethod
    def _create_event_loop_in_separate_thread() -> (
        tuple[asyncio.AbstractEventLoop, threading.Thread]
    ):
        """Create a dedicated event loop in a separate thread.

        This event loop is where async calls to the serving endpoint via OpenAI API
        client happens. The reason why a separate thread and event loop is needed is to
        avoid blocking the `issue_query` callback when the callback is called by the
        LoadGen C++ code.
        """
        event_loop = asyncio.new_event_loop()

        def _run_event_loop_forever() -> None:
            asyncio.set_event_loop(event_loop)
            event_loop.run_forever()

        event_loop_thread = threading.Thread(
            target=_run_event_loop_forever,
            daemon=True,
        )
        event_loop_thread.start()
        return event_loop, event_loop_thread

    @staticmethod
    @abstractmethod
    def formulate_loaded_sample(
        sample: dict[str, Any],
        *,
        use_guided_decoding: bool = False,
    ) -> LoadedSample:
        """Formulate the sample to be loaded into host memory before testing.

        Args:
            sample: The sample from the dataset to be formulated into a loaded sample.
            use_guided_decoding: Whether to use guided decoding for the sample.

        Returns:
            The loaded sample to be used for issuing queries to the inference endpoint.
        """
        raise NotImplementedError

    @property
    def total_num_samples(self) -> int:
        """The total number of samples in the dataset.

        This is used to set the `total_sample_count` parameter in the LoadGen QSL
        constructor.
        """
        return min(len(self.dataset), self.settings.min_query_count)

    @property
    def estimated_num_performance_samples(self) -> int:
        """The estimated number of performance samples.

        This is used to set the `performance_sample_count` parameter in the LoadGen QSL
        constructor. In performance mode, the performance samples will be loaded into
        host memory before testing, and LoadGen will keep sending queries using these
        performance samples (and repeating them if necessary) until the min_duration
        and min_query_count are met.

        We estimate this value by assuming that we reserve
        `self.ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES` bytes of host memory for
        storing the performance samples, and we try to estimate how many samples can fit
        in that amount of memory. If this value is bigger than the total number of
        samples, we will just load all samples into host memory.
        """
        estimation_indices = random.sample(
            range(self.total_num_samples),
            k=min(
                MAX_NUM_ESTIMATION_PERFORMANCE_SAMPLES,
                self.total_num_samples),
        )
        estimation_samples = [
            self.formulate_loaded_sample(
                self.dataset[i],
                use_guided_decoding=self.endpoint.use_guided_decoding,
            )
            for i in estimation_indices
        ]
        avg_messages_footprint = sum(
            asizeof.asizeof(m) for m in estimation_samples
        ) / len(estimation_samples)
        result = min(
            round(
                ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES / avg_messages_footprint,
            ),
            self.total_num_samples,
        )
        logger.debug(
            "Estimated number of performance samples that can be loaded into {} GB host"
            " memory before testing is {}.",
            ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES / 1024 / 1024 / 1024,
            result,
        )
        if self.settings.performance_sample_count_override > 0:
            logger.debug(
                "However, performance_sample_count_override is set to {} and will "
                "override the estimated number of performance samples inside the "
                "LoadGen.",
                self.settings.performance_sample_count_override,
            )
        return result

    def construct_qsl(self) -> int:
        """Construct the LoadGen QSL for the task."""

        def _load_samples_to_ram(query_sample_indices: list[int]) -> None:
            """Called by LoadGen to load samples to host memory before testing.

            Args:
                query_sample_indices: The indices of the samples to load to host memory.
            """
            logger.info(
                "Starting to load {} samples to RAM...",
                len(query_sample_indices),
            )
            tic = time.perf_counter()
            for index in query_sample_indices:
                self.loaded_samples[index] = self.formulate_loaded_sample(
                    self.dataset[index],
                    use_guided_decoding=self.endpoint.use_guided_decoding,
                )
            logger.info(
                "Loaded {} samples to RAM, which took {} seconds and {} GB in total.",
                len(query_sample_indices),
                time.perf_counter() - tic,
                asizeof.asizeof(self.loaded_samples) / 1024 / 1024 / 1024,
            )

        def _unload_samples_from_ram(query_sample_indices: list[int]) -> None:
            """Called by LoadGen to unload samples from host memory after testing.

            Args:
                query_sample_indices: The indices of the samples to unload from host
                    memory.
            """
            logger.info(
                "Starting to unload {} samples from RAM...",
                len(query_sample_indices),
            )
            tic = time.perf_counter()
            for index in query_sample_indices:
                sample_to_unload = self.loaded_samples.pop(index, None)
                del sample_to_unload
            logger.info(
                "Unloaded {} samples from RAM, which took {} seconds.",
                len(query_sample_indices),
                time.perf_counter() - tic,
            )

        return lg.ConstructQSL(
            self.total_num_samples,
            self.estimated_num_performance_samples,
            _load_samples_to_ram,
            _unload_samples_from_ram,
        )

    async def _query_endpoint_async_batch(
            self, query_sample: lg.QuerySample) -> None:
        """Query the endpoint through the async OpenAI API client."""
        try:
            sample = self.loaded_samples[query_sample.index]
            logger.debug(
                "Issuing query sample index: {} with response ID: {}",
                query_sample.index,
                query_sample.id,
            )
            logger.trace(
                "The sample (query sample index: {}, response ID: {}) to be "
                "issued to the endpoint is: {}",
                query_sample.index,
                query_sample.id,
                sample,
            )
            tic = time.perf_counter()
            response = await self.openai_api_client.chat.completions.create(  # type: ignore[call-overload, misc]
                model=self.endpoint.model.repo_id,
                messages=sample.messages,
                response_format=(
                    sample.response_format.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                    if sample.response_format is not None
                    else None
                ),
                frequency_penalty=self.endpoint.sampling_params.frequency_penalty,
                presence_penalty=self.endpoint.sampling_params.presence_penalty,
                temperature=self.endpoint.sampling_params.temperature,
                top_p=self.endpoint.sampling_params.top_p,
                extra_body={
                    k: getattr(self.endpoint.sampling_params, k)
                    for k in ("top_k", "min_p", "repetition_penalty")
                    if getattr(self.endpoint.sampling_params, k) is not None
                },
            )
            logger.debug(
                "Received response (ID: {}) from endpoint after {} seconds.",
                query_sample.id,
                time.perf_counter() - tic,
            )
            logger.trace(
                "The response (ID: {}) from the endpoint is: {}",
                query_sample.id,
                response,
            )
            content = response.choices[0].message.content
            if content is None:
                content = ""
            bytes_array = array.array("B", content.encode("utf-8"))
            address, length = bytes_array.buffer_info()
            size_in_bytes = length * bytes_array.itemsize
            lg.QuerySamplesComplete(
                [
                    lg.QuerySampleResponse(
                        query_sample.id,
                        address,
                        size_in_bytes,
                        int(response.usage.completion_tokens),
                    ),
                ],
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error processing query sample index {} with response ID {}.",
                query_sample.index,
                query_sample.id,
            )
            # Send empty response to LoadGen to avoid hanging.
            empty_content = ""
            bytes_array = array.array("B", empty_content.encode("utf-8"))
            address, length = bytes_array.buffer_info()
            size_in_bytes = length * bytes_array.itemsize
            lg.QuerySamplesComplete(
                [
                    lg.QuerySampleResponse(
                        query_sample.id,
                        address,
                        size_in_bytes,
                        0,
                    ),
                ],
            )

    async def _query_endpoint_async_stream(
            self, query_sample: lg.QuerySample) -> None:
        """Query the endpoint through the async OpenAI API client."""
        ttft_set = False
        try:
            sample = self.loaded_samples[query_sample.index]
            logger.debug(
                "Issuing query sample index: {} with response ID: {}",
                query_sample.index,
                query_sample.id,
            )
            logger.trace(
                "The sample (query sample index: {}, response ID: {}) to be "
                "issued to the endpoint is: {}",
                query_sample.index,
                query_sample.id,
                sample,
            )
            word_array = []
            stream = await self.openai_api_client.chat.completions.create(  # type: ignore[call-overload, misc]
                stream=True,
                model=self.endpoint.model.repo_id,
                messages=sample.messages,
                stream_options={"include_usage": True},
                response_format=(
                    sample.response_format.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                    if sample.response_format is not None
                    else None
                ),
                frequency_penalty=self.endpoint.sampling_params.frequency_penalty,
                presence_penalty=self.endpoint.sampling_params.presence_penalty,
                temperature=self.endpoint.sampling_params.temperature,
                top_p=self.endpoint.sampling_params.top_p,
                extra_body={
                    k: getattr(self.endpoint.sampling_params, k)
                    for k in ("top_k", "min_p", "repetition_penalty")
                    if getattr(self.endpoint.sampling_params, k) is not None
                },
            )
            # iterate asynchronously
            total_tokens = 0
            async for chunk in stream:

                # This is the final chunk and will not have 'choices'
                if chunk.usage is not None:
                    total_tokens = int(chunk.usage.completion_tokens)

                # If it's not the usage chunk, process it as a content
                # chunk
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                # first non-empty token -> TTFT
                delta = choices[0].delta
                text = getattr(delta, "content", None)
                if not text:
                    continue
                if ttft_set is False:
                    bytes_array = array.array("B", text.encode("utf-8"))
                    address, length = bytes_array.buffer_info()
                    size_in_bytes = length * bytes_array.itemsize
                    lg.FirstTokenComplete(
                        [
                            lg.QuerySampleResponse(
                                query_sample.id,
                                address,
                                size_in_bytes,
                                1,
                            ),
                        ],
                    )
                    ttft_set = True
                word_array.append(text)

            # when the stream ends, total latency
            content = "".join(word_array)
            bytes_array = array.array("B", content.encode("utf-8"))
            address, length = bytes_array.buffer_info()
            size_in_bytes = length * bytes_array.itemsize
            lg.QuerySamplesComplete(
                [
                    lg.QuerySampleResponse(
                        query_sample.id,
                        address,
                        size_in_bytes,
                        total_tokens,
                    ),
                ],
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error processing query sample index {} with response ID {}.",
                query_sample.index,
                query_sample.id,
            )
            # Send empty response to LoadGen to avoid hanging.
            empty_content = ""
            bytes_array = array.array("B", empty_content.encode("utf-8"))
            address, length = bytes_array.buffer_info()
            size_in_bytes = length * bytes_array.itemsize
            # If TTFT was not set, we still need to complete that.
            if not ttft_set:
                lg.FirstTokenComplete(
                    [
                        lg.QuerySampleResponse(
                            query_sample.id,
                            address,
                            size_in_bytes,
                            0,
                        ),
                    ],
                )
            lg.QuerySamplesComplete(
                [
                    lg.QuerySampleResponse(
                        query_sample.id,
                        address,
                        size_in_bytes,
                        0,
                    ),
                ],
            )

    def construct_sut(self) -> int:
        """Construct the LoadGen SUT for the task."""

        def _issue_queries(query_samples: list[lg.QuerySample]) -> None:
            """Called by the LoadGen to issue queries to the inference endpoint.

            Args:
                query_samples: The list of query samples to issue to the inference
                    endpoint. Each query sample contains (1) `id`: `lg.ResponseId`
                    (i.e., unique identifier for the response), and (2) `index`:
                    `lg.QuerySampleIndex` (i.e., the sample index into the dataset).
            """
            for query_sample in query_samples:
                asyncio.run_coroutine_threadsafe(
                    (
                        self._query_endpoint_async_stream(query_sample)
                        if (
                            self.settings.scenario is TestScenario.SERVER
                            and self.settings.use_token_latencies
                        )
                        else self._query_endpoint_async_batch(query_sample)
                    ),
                    self.event_loop,
                )

        def _flush_queries() -> None:
            """Called by the LoadGen to indicate that all queries have been issued."""
            logger.info(
                "LoadGen has indicated that all queries have been issued. "
                "Waiting for all pending queries to complete...",
            )

            async def _wait_for_pending_queries_async() -> None:
                """Wait for all pending queries to complete."""
                results = await asyncio.gather(
                    *[
                        t
                        for t in asyncio.all_tasks(self.event_loop)
                        # Current task is this one that waits for pending
                        # queries.
                        if t is not asyncio.current_task()
                    ],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        raise result

            future = asyncio.run_coroutine_threadsafe(
                _wait_for_pending_queries_async(),
                self.event_loop,
            )
            future.result()
            logger.info("All pending queries has completed.")

        return lg.ConstructSUT(_issue_queries, _flush_queries)


class ShopifyGlobalCatalogue(Task):
    """The Shopify Global Catalogue task."""

    def __init__(
        self,
        dataset: Dataset,
        endpoint: Endpoint,
        settings: TestSettings,
        random_seed: int = 12345,
    ) -> None:
        """Initialize the task.

        Args:
            dataset: The dataset configuration passed in from the CLI.
            endpoint: The endpoint configuration passed in from the CLI.
            settings: Parameters of the current benchmark.
            random_seed: The random seed to use for the task.
        """
        super().__init__(
            dataset=dataset,
            endpoint=endpoint,
            settings=settings,
            random_seed=random_seed,
        )

    @staticmethod
    def formulate_loaded_sample(
        sample: dict[str, Any],
        *,
        use_guided_decoding: bool = False,
    ) -> LoadedSample:
        """Formulate the sample to be loaded into host memory before testing.

        Args:
            sample: The sample from the dataset to be formulated into a loaded sample.
            use_guided_decoding: Whether to use guided decoding for the sample.

        Returns:
            The loaded sample to be used for issuing queries to the inference endpoint.
        """
        image_file = BytesIO()
        image_format = sample["product_image"].format
        sample["product_image"].save(image_file, format=image_format)
        image_bytes = image_file.getvalue()
        image_base64 = base64.b64encode(image_bytes)
        image_base64_string = image_base64.decode("utf-8")
        messages = [
            {
                "role": "system",
                "content": f"""Please analyze the product from the user prompt
and provide the following fields in a valid JSON object:
- category
- brand
- is_secondhand

You must choose only one, which is the most appropriate, correct, and specifc
category out of the list of possible product categories.

The description of the product sometimes contains various types of source code
(e.g., JavaScript, CSS, HTML, etc.), where useful product information is embedded
somewhere inside the source code. For this task, you should extract the useful
product information from the source code and leverage it, and discard the
programmatic parts of the source code.

Your response should only contain a valid JSON object and nothing more, e.g.,
you should not fence the JSON object inside a ```json code block.
The JSON object should match the followng JSON schema:
```json
{json.dumps(ProductMetadata.model_json_schema(), indent=2)}
```
""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""The title of the product is the following:
```text
{sample['product_title']}
```

The description of the product is the following:
```text
{sample['product_description']}
```

The following are the possible product categories:
```json
{sample['potential_product_categories']}
```
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,"
                            f"{image_base64_string}",
                        },
                    },
                ],
            },
        ]

        return LoadedSample(
            messages=messages,
            response_format=(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "product_metadata",
                        "schema": ProductMetadata.model_json_schema(),
                        "strict": True,
                    },
                }
                if use_guided_decoding
                else None
            ),
        )
