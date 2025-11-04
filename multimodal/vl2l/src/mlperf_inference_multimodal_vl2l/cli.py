"""The CLI definition for the VL2L benchmark."""

from __future__ import annotations

import sys
from datetime import timedelta
from enum import StrEnum, auto
from typing import Annotated

import mlperf_loadgen as lg
from loguru import logger
from openai import AsyncOpenAI, DefaultAioHttpClient
from pydantic import BaseModel, Field
from pydantic_typer import Typer
from typer import Option

from .task import ShopifyGlobalCatalogue

app = Typer()


class TestScenario(StrEnum):
    """The test scenario for the MLPerf inference LoadGen."""

    SERVER = auto()
    """Run the benchmark in server/interactive scenario."""

    OFFLINE = auto()
    """Run the benchmark in offline/batch scenario."""

    class UnknownValueError(ValueError):
        """The exception raised when an unknown test scenario is encountered."""

        def __init__(self, test_scenario: TestScenario) -> None:
            """Initialize the exception."""
            super().__init__(f"Unknown test scenario: {test_scenario}")

    def to_lgtype(self) -> lg.TestScenario:
        """Convert the test scenario to its corresponding LoadGen type."""
        match self:
            case TestScenario.SERVER:
                return lg.TestScenario.Server
            case TestScenario.OFFLINE:
                return lg.TestScenario.Offline
            case _:
                raise TestScenario.UnknownValueError(self)


class TestMode(StrEnum):
    """The test mode for the MLPerf inference LoadGen."""

    PERFORMANCE_ONLY = auto()
    """Run the benchmark to evaluate performance."""

    ACCURACY_ONLY = auto()
    """Run the benchmark to evaluate model quality."""

    class UnknownValueError(ValueError):
        """The exception raised when an unknown test mode is encountered."""

        def __init__(self, test_mode: TestMode) -> None:
            """Initialize the exception."""
            super().__init__(f"Unknown test mode: {test_mode}")

    def to_lgtype(self) -> lg.TestMode:
        """Convert the test mode to its corresponding LoadGen type."""
        match self:
            case TestMode.PERFORMANCE_ONLY:
                return lg.TestMode.PerformanceOnly
            case TestMode.ACCURACY_ONLY:
                return lg.TestMode.AccuracyOnly
            case _:
                raise TestMode.UnknownValueError(self)


class TestSettings(BaseModel):
    """The test settings for the MLPerf inference LoadGen."""

    senario: Annotated[
        TestScenario,
        Field(
            description=(
                "The MLPerf inference benchmarking scenario to run the benchmark in."
            ),
        ),
    ] = TestScenario.OFFLINE

    mode: Annotated[
        TestMode,
        Field(
            description=(
                "Whether you want to run the benchmark for performance or accuracy."
            ),
        ),
    ] = TestMode.PERFORMANCE_ONLY

    offline_expected_qps: Annotated[
        float,
        Field(
            description="The expected QPS for the offline scenario.",
        ),
    ] = 100

    min_duration: Annotated[
        timedelta,
        Field(
            description="The minimum testing duration.",
        ),
    ] = timedelta(seconds=5)

    def to_lgtype(self) -> lg.TestSettings:
        """Convert the test settings to its corresponding LoadGen type."""
        settings = lg.TestSettings()
        settings.scenario = self.senario.to_lgtype()
        settings.mode = self.mode.to_lgtype()
        settings.offline_expected_qps = self.offline_expected_qps
        settings.min_duration_ms = round(
            self.min_duration.total_seconds() * 1000)
        settings.use_token_latencies = True
        return settings


class Model(BaseModel):
    """Specifies the model to use for the VL2L benchmark."""

    repo_id: Annotated[
        str,
        Field(description="The HuggingFace repository ID of the model."),
    ] = "Qwen/Qwen3-VL-235B-A22B-Instruct"


class Dataset(BaseModel):
    """Specifies a dataset on HuggingFace."""

    repo_id: Annotated[
        str,
        Field(description="The HuggingFace repository ID of the dataset."),
    ] = "Shopify/the-catalogue-public-beta"

    token: Annotated[
        str | None,
        Field(
            description=(
                "The token to access the HuggingFace repository of the dataset."
            ),
        ),
    ] = None


class Verbosity(StrEnum):
    """The verbosity level of the logger."""

    TRACE = auto()
    """The trace verbosity level."""

    DEBUG = auto()
    """The debug verbosity level."""

    INFO = auto()
    """The info verbosity level (default)."""


class Endpoint(BaseModel):
    """Specifies the OpenAI API endpoint to use for the VL2L benchmark."""

    url: Annotated[
        str,
        Field(
            description=(
                "The URL of the OpenAI API endpoint that the inference requests will be"
                " sent to."
            ),
        ),
    ] = "http://localhost:8000/v1"
    api_key: Annotated[
        str,
        Field(description="The API key to authenticate the inference requests."),
    ] = ""


@app.command()
def main(
    *,
    settings: TestSettings,
    model: Model,
    dataset: Dataset,
    endpoint: Endpoint,
    random_seed: Annotated[
        int,
        Option(help="The seed for the random number generator used by the benchmark."),
    ] = 12345,
    verbosity: Annotated[
        Verbosity,
        Option(help="The verbosity level of the logger."),
    ] = Verbosity.INFO,
) -> None:
    """Main CLI for running the VL2L benchmark."""
    logger.remove()
    logger.add(sys.stdout, level=verbosity.value.upper())
    logger.info("Running VL2L benchmark with settings: {}", settings)
    logger.info("Running VL2L benchmark with model: {}", model)
    logger.info("Running VL2L benchmark with dataset: {}", dataset)
    logger.info(
        "Running VL2L benchmark with OpenAI API endpoint: {}",
        endpoint)
    logger.info("Running VL2L benchmark with random seed: {}", random_seed)
    lg_settings = settings.to_lgtype()
    task = ShopifyGlobalCatalogue(
        dataset_cli=dataset,
        model_cli=model,
        openai_api_client=AsyncOpenAI(
            base_url=endpoint.url,
            http_client=DefaultAioHttpClient(),
            api_key=endpoint.api_key,
        ),
        random_seed=random_seed,
    )
    sut = task.construct_sut()
    qsl = task.construct_qsl()
    logger.info("Starting the VL2L benchmark with LoadGen...")
    lg.StartTest(sut, qsl, lg_settings)
    logger.info("The VL2L benchmark with LoadGen completed.")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
