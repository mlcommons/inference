"""Task definitions for the VL2L benchmark."""

from __future__ import annotations

import array
import asyncio
import base64
import threading
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any

import mlperf_loadgen as lg
from datasets import load_dataset
from loguru import logger
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from .cli import Dataset as DatasetCLI
    from .cli import Model as ModelCLI


class Task(ABC):
    """Abstract base class for defining a task."""

    def __init__(
        self,
        dataset_cli: DatasetCLI,
        model_cli: ModelCLI,
        openai_api_client: AsyncOpenAI,
    ) -> None:
        """Initialize the task.

        Args:
            dataset_cli: The dataset configuration passed in from the CLI.
            model_cli: The model configuration passed in from the CLI.
            openai_api_client: The OpenAI API client to use for the task.
        """
        self.dataset = load_dataset(
            dataset_cli.repo_id,
            token=dataset_cli.token,
        )
        self.model_cli = model_cli
        self.loaded_samples: dict[int, Any] = {}
        self.openai_api_client = openai_api_client
        self.event_loop = self._create_event_loop_in_separate_thread()

    @staticmethod
    def _create_event_loop_in_separate_thread() -> asyncio.EventLoop:
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
        return event_loop

    @staticmethod
    @abstractmethod
    def formulate_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
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
        return len(self.dataset)

    @property
    def max_num_samples_in_host_memory(self) -> int:
        """The maximum number of samples that are guaranteed to fit in host memory.

        This is used to set the `performance_sample_count` parameter in the LoadGen QSL
        constructor.
        """
        return round(len(self.dataset) * 0.1)

    def construct_qsl(self) -> int:
        """Construct the LoadGen QSL for the task."""

        def _load_samples_to_ram(query_sample_indices: list[int]) -> None:
            """Called by LoadGen to load samples to host memory before testing.

            Args:
                query_sample_indices: The indices of the samples to load to host memory.
            """
            for index in query_sample_indices:
                self.loaded_samples[index] = self.formulate_messages(
                    self.dataset[index],
                )

        def _unload_samples_from_ram(query_sample_indices: list[int]) -> None:
            """Called by LoadGen to unload samples from host memory after testing.

            Args:
                query_sample_indices: The indices of the samples to unload from host memory.
            """
            for index in query_sample_indices:
                sample_to_unload = self.loaded_samples.pop(index, None)
                del sample_to_unload

        return lg.ConstructQSL(
            self.total_num_samples,
            self.max_num_samples_in_host_memory,
            _load_samples_to_ram,
            _unload_samples_from_ram,
        )

    def construct_sut(self) -> int:
        """Construct the LoadGen SUT for the task."""

        def _issue_queries(query_samples: list[lg.QuerySample]) -> None:
            """Called by the LoadGen to issue queries to the inference endpoint.

            Args:
                query_samples: The list of query samples to issue to the inference endpoint.
                    Each query sample contains (1) `id`: `lg.ResponseId` (i.e., unique
                    identifier for the response), and (2) `index`: `lg.QuerySampleIndex`
                    (i.e., the sample index into the dataset).
            """

            async def _query_endpoint_async(
                    query_sample: lg.QuerySample) -> None:
                """Query the endpoint through the async OpenAI API client."""
                messages = self.loaded_samples[query_sample.index]
                response = await self.openai_api_client.chat.completions.create(
                    model=self.model_cli.repo_id,
                    messages=messages,
                )
                logger.info("Response: {}", response)
                content = response.choices[0].message.content
                bytes_array = array.array("B", content.encode("utf-8"))
                address, length = bytes_array.buffer_info()
                size_in_bytes = length * bytes_array.itemsize
                return lg.QuerySampleResponse(
                    query_sample.id,
                    address,
                    size_in_bytes,
                    len(content),
                )

            async def _issue_queries_async(
                    query_samples: list[lg.QuerySample]) -> None:
                """Issue queries to the inference endpoint."""
                query_sample_responses = await asyncio.gather(
                    *[
                        _query_endpoint_async(query_sample)
                        for query_sample in query_samples
                    ],
                )
                lg.QuerySamplesComplete(query_sample_responses)

            asyncio.run_coroutine_threadsafe(
                _issue_queries_async(query_samples),
                self.event_loop,
            )

        def _flush_queries() -> None:
            """Called by the LoadGen to indicate that all queries have been issued."""

            async def _wait_for_pending_queries_async() -> None:
                """Wait for all pending queries to complete."""
                await asyncio.gather(
                    *[
                        t
                        for t in asyncio.all_tasks(self.event_loop)
                        # Current task is this one that waits for pending
                        # queries.
                        if t is not asyncio.current_task()
                    ],
                    return_exceptions=True,
                )

            future = asyncio.run_coroutine_threadsafe(
                _wait_for_pending_queries_async(),
                self.event_loop,
            )
            future.result()  # Block until all pending queries are complete.

        return lg.ConstructSUT(_issue_queries, _flush_queries)


class MMMU(Task):
    """The MMMU task."""

    @staticmethod
    def formulate_messages(
            self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Formulate the messages for chat completion.

        Args:
            sample: The sample to formulate the messages for.

        Returns:
            The messages for chat completion.
        """


class ShopifyGlobalCatalogue(Task):
    """The Shopify Global Catalogue task."""

    def __init__(
        self,
        dataset_cli: DatasetCLI,
        model_cli: ModelCLI,
        openai_api_client: AsyncOpenAI,
    ) -> None:
        """Initialize the task.

        Args:
            dataset_cli: The dataset configuration passed in from the CLI.
            model_cli: The model configuration passed in from the CLI.
            openai_api_client: The OpenAI API client to use for the task.
        """
        super().__init__(dataset_cli, model_cli, openai_api_client)
        # Shopify only released the train split so far.
        self.dataset = self.dataset["train"]

    @staticmethod
    def formulate_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Formulate the messages for chat completion.

        Args:
            sample: The sample to formulate the messages for.

        Returns:
            The messages for chat completion.
        """
        image_file = BytesIO()
        sample["image"].save(image_file, format="PNG")
        image_bytes = image_file.getvalue()
        image_base64 = base64.b64encode(image_bytes)
        image_base64_string = image_base64.decode("utf-8")
        return [
            {
                "role": "system",
                "content": (
                    "Please analyze the following product and provide the following "
                    "fields in JSON format:\n"
                    "- catagory\n"
                    "- standardized_title\n"
                    "- standardized_description\n"
                    "- brands\n"
                    "- is_secondhand\n"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"The title of the product is: {sample['title']}\n\n"
                            f"The description of the product is: {sample['description']}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64_string}",
                        },
                    },
                ],
            },
        ]
