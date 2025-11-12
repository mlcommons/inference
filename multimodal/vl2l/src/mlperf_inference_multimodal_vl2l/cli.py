"""The CLI definition for the VL2L benchmark."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from enum import StrEnum, auto
from typing import Annotated

import mlperf_loadgen as lg
from loguru import logger
from pydantic import BaseModel, Field, field_validator
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

    def to_lgtype(self) -> lg.TestScenario:
        """Convert the test scenario to its corresponding LoadGen type."""
        match self:
            case TestScenario.SERVER:
                return lg.TestScenario.Server
            case TestScenario.OFFLINE:
                return lg.TestScenario.Offline
            case _:
                raise UnknownTestScenarioValueError(self)


class UnknownTestScenarioValueError(ValueError):
    """The exception raised when an unknown test scenario is encountered."""

    def __init__(self, test_scenario: TestScenario) -> None:
        """Initialize the exception."""
        super().__init__(f"Unknown test scenario: {test_scenario}")


class TestMode(StrEnum):
    """The test mode for the MLPerf inference LoadGen."""

    PERFORMANCE_ONLY = auto()
    """Run the benchmark to evaluate performance."""

    ACCURACY_ONLY = auto()
    """Run the benchmark to evaluate model quality."""

    def to_lgtype(self) -> lg.TestMode:
        """Convert the test mode to its corresponding LoadGen type."""
        match self:
            case TestMode.PERFORMANCE_ONLY:
                return lg.TestMode.PerformanceOnly
            case TestMode.ACCURACY_ONLY:
                return lg.TestMode.AccuracyOnly
            case _:
                raise UnknownTestModeValueError(self)


class UnknownTestModeValueError(ValueError):
    """The exception raised when an unknown test mode is encountered."""

    def __init__(self, test_mode: TestMode) -> None:
        """Initialize the exception."""
        super().__init__(f"Unknown test mode: {test_mode}")


class TestSettings(BaseModel):
    """The test settings for the MLPerf inference LoadGen."""

    scenario: Annotated[
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

    server_expected_qps: Annotated[
        float,
        Field(
            description="The expected QPS for the server scenario.",
        ),
    ] = 1

    server_target_latency: Annotated[
        float,
        Field(description="Expected latency for Server scenario "
              "(will be converted to ns)"),
    ] = 0.1

    server_ttft_latency: Annotated[
        float,
        Field(description="token ttft latency parameter "
              "(used when use_token_latencies is enabled). "
              "Will be converted to ns"),
    ] = 0.1

    server_tpot_latency: Annotated[
        float,
        Field(description="token tpot latency parameter "
              "(used when use_token_latencies is enabled). "
              "Will be converted to ns"),
    ] = 0.1

    # The test runs until both min duration and min query count have been met
    min_duration: Annotated[
        timedelta,
        Field(
            description=(
                "The minimum testing duration (in seconds or ISO 8601 format like"
                " PT5S)."
            ),
        ),
    ] = timedelta(seconds=5)

    min_query_count: Annotated[
        int,
        Field(
            description="The minimum testing query count",
        ),
    ] = 100

    use_token_latencies: Annotated[
        bool,
        Field(
            description="When set to True, LoadGen will track TTFT and TPOT.",
        ),
    ] = False

    @field_validator("min_duration", mode="before")
    @classmethod
    def parse_min_duration(cls, value: timedelta |
                           float | str) -> timedelta | str:
        """Parse timedelta from seconds (int/float/str) or ISO 8601 format."""
        if isinstance(value, timedelta):
            return value
        if isinstance(value, (int, float)):
            return timedelta(seconds=value)
        if isinstance(value, str):
            # Try to parse as a number first
            try:
                return timedelta(seconds=float(value))
            except ValueError:
                # If it fails, it might be ISO 8601 format
                # Let pydantic's default parser handle it
                pass
        return value

    def to_lgtype(self) -> lg.TestSettings:
        """Convert the test settings to its corresponding LoadGen type."""
        settings = lg.TestSettings()
        settings.scenario = self.scenario.to_lgtype()
        settings.mode = self.mode.to_lgtype()
        settings.offline_expected_qps = self.offline_expected_qps
        settings.server_target_qps = self.server_expected_qps
        settings.server_target_latency_ns = round(
            self.server_target_latency * 1e9)
        settings.ttft_latency = round(self.server_ttft_latency * 1e9)
        settings.tpot_latency = round(self.server_tpot_latency * 1e9)
        settings.min_duration_ms = round(
            self.min_duration.total_seconds() * 1000)
        settings.min_query_count = self.min_query_count
        settings.use_token_latencies = self.use_token_latencies
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
    output_log_dir: Annotated[
        str,
        Option(help="Location of output logs"),
    ] = "output",
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
        endpoint_cli=endpoint,
        scenario=settings.scenario,
        random_seed=random_seed,
    )
    sut = task.construct_sut()
    qsl = task.construct_qsl()
    # log settings
    os.makedirs(output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    logger.info("Starting the VL2L benchmark with LoadGen...")
    lg.StartTestWithLogSettings(sut, qsl, lg_settings, log_settings)
    logger.info("The VL2L benchmark with LoadGen completed.")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
