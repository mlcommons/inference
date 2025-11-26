"""Task definitions for the VL2L benchmark."""

from __future__ import annotations

import array
import asyncio
import base64
import random
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any

import mlperf_loadgen as lg
from datasets import load_dataset
from loguru import logger
from openai import AsyncOpenAI, DefaultAioHttpClient
from pympler import asizeof

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )

    from .cli import Dataset as DatasetCLI
    from .cli import Endpoint as EndpointCLI
    from .cli import Model as ModelCLI
    from .cli import TestSettings


class Task(ABC):
    """Abstract base class for defining a task."""

    def __init__(
        self,
        dataset_cli: DatasetCLI,
        model_cli: ModelCLI,
        endpoint_cli: EndpointCLI,
        settings: TestSettings,
        random_seed: int = 12345,
    ) -> None:
        """Initialize the task.

        Args:
            dataset_cli: The dataset configuration passed in from the CLI.
            model_cli: The model configuration passed in from the CLI.
            endpoint_cli: The endpoint configuration passed in from the CLI.
            settings: Parameters of the current benchmark.
            random_seed: The random seed to use for the task.
        """
        random.seed(random_seed)
        self.scenario = settings.scenario
        self.dataset = load_dataset(
            dataset_cli.repo_id,
            token=dataset_cli.token,
            split="+".join(dataset_cli.split),
        )
        logger.debug(
            "Loaded {} samples from the dataset splits {}.",
            len(self.dataset),
            dataset_cli.split,
        )
        self.model_cli = model_cli
        self.openai_api_client = AsyncOpenAI(
            base_url=endpoint_cli.url,
            http_client=DefaultAioHttpClient(),
            api_key=endpoint_cli.api_key,
        )
        self.event_loop, self.event_loop_thread = (
            self._create_event_loop_in_separate_thread()
        )
        self.loaded_messages: dict[int, list[ChatCompletionMessageParam]] = {}
        self.settings = settings

    def __del__(self) -> None:
        """Clean up the resources used by the task."""
        logger.trace("Cleaning up the resources used by the task...")
        asyncio.run_coroutine_threadsafe(
            self.openai_api_client.close(),
            self.event_loop,
        ).result()
        self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        self.event_loop_thread.join()

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
    def formulate_messages(
            sample: dict[str, Any]) -> list[ChatCompletionMessageParam]:
        """Formulate the messages for chat completion.

        Args:
            sample: The sample to formulate the messages for.

        Returns:
            The messages for chat completion.
        """
        raise NotImplementedError

    @property
    def total_num_samples(self) -> int:
        """The total number of samples in the dataset.

        This is used to set the `total_sample_count` parameter in the LoadGen QSL
        constructor.
        """
        return min(len(self.dataset), self.settings.min_query_count)

    MAX_NUM_ESTIMATION_PERFORMANCE_SAMPLES = 100
    ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES = 1 * 1024 * 1024 * 1024  # 1GB

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
                self.MAX_NUM_ESTIMATION_PERFORMANCE_SAMPLES,
                self.total_num_samples),
        )
        estimation_samples = [
            self.formulate_messages(self.dataset[i]) for i in estimation_indices
        ]
        avg_messages_footprint = sum(
            asizeof.asizeof(m) for m in estimation_samples
        ) / len(estimation_samples)
        result = min(
            round(
                self.ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES
                / avg_messages_footprint,
            ),
            self.total_num_samples,
        )
        logger.debug(
            "Estimated number of performance samples that will be loaded into the host"
            " memory before testing is {}.",
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
            for index in query_sample_indices:
                self.loaded_messages[index] = self.formulate_messages(
                    self.dataset[index],
                )

        def _unload_samples_from_ram(query_sample_indices: list[int]) -> None:
            """Called by LoadGen to unload samples from host memory after testing.

            Args:
                query_sample_indices: The indices of the samples to unload from host
                    memory.
            """
            for index in query_sample_indices:
                sample_to_unload = self.loaded_messages.pop(index, None)
                del sample_to_unload

        return lg.ConstructQSL(
            self.total_num_samples,
            self.estimated_num_performance_samples,
            _load_samples_to_ram,
            _unload_samples_from_ram,
        )

    def construct_sut(self) -> int:
        """Construct the LoadGen SUT for the task."""
        # avoid circular import
        from .cli import TestScenario

        def _issue_queries(query_samples: list[lg.QuerySample]) -> None:
            """Called by the LoadGen to issue queries to the inference endpoint.

            Args:
                query_samples: The list of query samples to issue to the inference
                    endpoint. Each query sample contains (1) `id`: `lg.ResponseId`
                    (i.e., unique identifier for the response), and (2) `index`:
                    `lg.QuerySampleIndex` (i.e., the sample index into the dataset).
            """

            async def _query_endpoint_async(
                    query_sample: lg.QuerySample) -> None:
                """Query the endpoint through the async OpenAI API client."""
                try:
                    messages = self.loaded_messages[query_sample.index]
                    logger.trace(
                        "Issuing query sample index: {} with response ID: {}",
                        query_sample.index,
                        query_sample.id,
                    )
                    tic = time.perf_counter()
                    response = await self.openai_api_client.chat.completions.create(
                        model=self.model_cli.repo_id,
                        messages=messages,
                    )
                    logger.trace(
                        "Received response (ID: {}) from endpoint after {} seconds: {}",
                        query_sample.id,
                        time.perf_counter() - tic,
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
                    bytes_array = array.array(
                        "B", empty_content.encode("utf-8"))
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

            for query_sample in query_samples:
                asyncio.run_coroutine_threadsafe(
                    _query_endpoint_async(query_sample),
                    self.event_loop,
                )

        def _issue_streaming_queries(
                query_samples: list[lg.QuerySample]) -> None:
            """Called by the LoadGen to issue queries to the inference endpoint.

            Args:
                query_samples: The list of query samples to issue to the inference
                    endpoint. Each query sample contains (1) `id`: `lg.ResponseId`
                    (i.e., unique identifier for the response), and (2) `index`:
                    `lg.QuerySampleIndex` (i.e., the sample index into the dataset).
            """

            async def _query_endpoint_async(
                    query_sample: lg.QuerySample) -> None:
                """Query the endpoint through the async OpenAI API client."""
                ttft_set = False
                try:
                    messages = self.loaded_messages[query_sample.index]
                    logger.trace(
                        "Issuing query sample index: {} with response ID: {}",
                        query_sample.index,
                        query_sample.id,
                    )
                    word_array = []
                    stream = await self.openai_api_client.chat.completions.create(
                        stream=True,
                        model=self.model_cli.repo_id,
                        messages=messages,
                        stream_options={"include_usage": True},
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
                            bytes_array = array.array(
                                "B", text.encode("utf-8"))
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
                    bytes_array = array.array(
                        "B", empty_content.encode("utf-8"))
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

            for query_sample in query_samples:
                asyncio.run_coroutine_threadsafe(
                    _query_endpoint_async(query_sample),
                    self.event_loop,
                )

        def _flush_queries() -> None:
            """Called by the LoadGen to indicate that all queries have been issued."""

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

        return lg.ConstructSUT(
            (
                _issue_streaming_queries
                if self.scenario is TestScenario.SERVER
                else _issue_queries
            ),
            _flush_queries,
        )


class ShopifyGlobalCatalogue(Task):
    """The Shopify Global Catalogue task."""

    def __init__(
        self,
        dataset_cli: DatasetCLI,
        model_cli: ModelCLI,
        endpoint_cli: EndpointCLI,
        settings: TestSettings,
        random_seed: int = 12345,
    ) -> None:
        """Initialize the task.

        Args:
            dataset_cli: The dataset configuration passed in from the CLI.
            model_cli: The model configuration passed in from the CLI.
            endpoint_cli: The endpoint configuration passed in from the CLI.
            settings: Parameters of the current benchmark.
            random_seed: The random seed to use for the task.
        """
        super().__init__(
            dataset_cli=dataset_cli,
            model_cli=model_cli,
            endpoint_cli=endpoint_cli,
            settings=settings,
            random_seed=random_seed,
        )

    @staticmethod
    def formulate_messages(
            sample: dict[str, Any]) -> list[ChatCompletionMessageParam]:
        """Formulate the messages for chat completion.

        Args:
            sample: The sample to formulate the messages for.

        Returns:
            The messages for chat completion.
        """
        image_file = BytesIO()
        image_format = sample["product_image"].format
        sample["product_image"].save(image_file, format=image_format)
        image_bytes = image_file.getvalue()
        image_base64 = base64.b64encode(image_bytes)
        image_base64_string = image_base64.decode("utf-8")
        return [
            {
                "role": "system",
                "content": """Please analyze the product from the user prompt
and provide the following fields in a valid JSON object:
- category
- brand
- is_secondhand

You must choose only one, which is the most appropriate/correct,
category out of the list of possible product categories.

Your response should only contain a valid JSON object and nothing more.
The JSON object should match the followng JSON schema:
```json
{
  "type": "object",
  "properties": {
    "category": {"type": "string"},
    "brand": {"type": "string"},
    "is_secondhand": {"type": "boolean"}
  }
}
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
