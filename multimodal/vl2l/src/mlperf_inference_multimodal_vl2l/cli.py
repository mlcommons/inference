"""The CLI definition for the VL2L benchmark."""

from __future__ import annotations

import sys
from datetime import timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Literal

import mlperf_loadgen as lg
from loguru import logger
from pydantic import BaseModel, DirectoryPath, Field, FilePath, field_validator
from pydantic_typer import Typer
from typer import Option

from .evaluation import run_evaluation
from .task import ShopifyGlobalCatalogue

app = Typer()
SplitType = Literal["train", "test"]

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


class LoggingMode(StrEnum):
    """Specifies when logging should be sampled and stringified."""

    ASYNC_POLL = auto()
    """ Logs are serialized and output on an IOThread that polls for new logs
      at a fixed interval. This is the only mode currently implemented."""

    END_OF_TEST_ONLY = auto()
    """ Not implemented """

    SYNCHRONOUS = auto()
    """ Not implemented """

    def to_lgtype(self) -> lg.LoggingMode:
        """Convert logging mode to its corresponding LoadGen type."""
        match self:
            case LoggingMode.ASYNC_POLL:
                return lg.LoggingMode.AsyncPoll
            case _:
                raise UnknownLoggingModeValueError


class UnknownLoggingModeValueError(ValueError):
    """The exception raised when an unknown logging mode is encountered."""

    def __init__(self, test_mode: TestMode) -> None:
        """Initialize the exception."""
        super().__init__(f"Unknown logging mode: {test_mode}")


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
            description="The expected QPS for the server scenario. "
            "Loadgen will try to send as many request as necessary "
            "to achieve this value.",
        ),
    ] = 1

    server_target_latency: Annotated[
        timedelta,
        Field(description="""Expected latency constraint for Server scenario.
        This is a constraint that we expect depending
              on the argument server_expected_qps.
        When server_expected_qps increases, we expect the latency to also increase.
        When server_expected_qps decreases, we expect the latency to also decrease."""),
    ] = timedelta(seconds=1)

    server_ttft_latency: Annotated[
        timedelta,
        Field(description="""Time to First Token (TTFT)
              latency constraint result validation"
              (used when use_token_latencies is enabled)."""),
    ] = timedelta(seconds=1)

    server_tpot_latency: Annotated[
        timedelta,
        Field(description="""Time per Output Token (TPOT)
              latency constraint result validation"
              (used when use_token_latencies is enabled)."""),
    ] = timedelta(seconds=1)

    min_duration: Annotated[
        timedelta,
        Field(
            description="""The minimum testing duration
                (in seconds or ISO 8601 format like PT5S).
                The benchmark runs until this value has been met.""",
        ),
    ] = timedelta(seconds=5)

    min_query_count: Annotated[
        int,
        Field(
            description="""The minimum testing query count.
            The benchmark runs until this value has been met.
            if min_query_count is less than the total number of samples in the dataset,
            only the first min_query_count samples will be used during testing.""",
        ),
    ] = 100

    use_token_latencies: Annotated[
        bool,
        Field(
            description="""By default,
            the Server scenario will use server_target_latency as the constraint.
            When set to True, the Server scenario will use server_ttft_latency
            and server_tpot_latency as the constraint.""",
        ),
    ] = False

    @field_validator("server_target_latency",
                     "server_ttft_latency",
                     "server_tpot_latency",
                     "min_duration",
                     mode="before")
    @classmethod
    def parse_timedelta(cls, value: timedelta |
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
            self.server_target_latency.total_seconds() * 1e9)
        settings.ttft_latency = round(
            self.server_ttft_latency.total_seconds() * 1e9)
        settings.tpot_latency = round(
            self.server_tpot_latency.total_seconds() * 1e9)
        settings.min_duration_ms = round(
            self.min_duration.total_seconds() * 1000)
        settings.min_query_count = self.min_query_count
        settings.use_token_latencies = self.use_token_latencies
        return settings


class LogOutputSettings(BaseModel):
    """The test log output settings for the MLPerf inference LoadGen."""
    outdir: Annotated[
        DirectoryPath,
        Field(
            description="Where to save the output files from the benchmark.",
        ),
    ] = DirectoryPath("output")
    prefix: Annotated[
        str,
        Field(
            description="Modify the filenames of the logs with a prefix.",
        ),
    ] = "mlperf_log_"
    suffix: Annotated[
        str,
        Field(
            description="Modify the filenames of the logs with a suffix.",
        ),
    ] = ""
    prefix_with_datetime: Annotated[
        bool,
        Field(
            description="Modify the filenames of the logs with a datetime.",
        ),
    ] = False
    copy_detail_to_stdout: Annotated[
        bool,
        Field(
            description="Print details of performance test to stdout.",
        ),
    ] = False
    copy_summary_to_stdout: Annotated[
        bool,
        Field(
            description="Print results of performance test to terminal.",
        ),
    ] = True

    @field_validator("outdir", mode="before")
    @classmethod
    def parse_directory_field(cls, value: str) -> None:
        """Verify and create the output directory to store log files."""
        path = Path(value)
        path.mkdir(exist_ok=True)
        return path

    def to_lgtype(self) -> lg.LogOutputSettings:
        """Convert the log output settings to its corresponding LoadGen type."""
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = self.outdir.as_posix()
        log_output_settings.prefix = self.prefix
        log_output_settings.suffix = self.suffix
        log_output_settings.prefix_with_datetime = self.prefix_with_datetime
        log_output_settings.copy_detail_to_stdout = self.copy_detail_to_stdout
        log_output_settings.copy_summary_to_stdout = self.copy_summary_to_stdout
        return log_output_settings


class LogSettings(BaseModel):
    """The test log settings for the MLPerf inference LoadGen."""
    log_output: Annotated[
        LogOutputSettings,
        Field(
            description="Log output settings",
        ),
    ] = LogOutputSettings
    log_mode: Annotated[
        LoggingMode,
        Field(
            description="""How and when logging should be
            sampled and stringified at runtime""",
        ),
    ] = LoggingMode.ASYNC_POLL
    enable_trace: Annotated[
        bool,
        Field(
            description="Enable trace",
        ),
    ] = True

    def to_lgtype(self) -> lg.LogSettings:
        """Convert log settings to its corresponding LoadGen type."""
        log_settings = lg.LogSettings()
        log_settings.log_output = self.log_output.to_lgtype()
        log_settings.log_mode = self.log_mode.to_lgtype()
        log_settings.enable_trace = self.enable_trace
        return log_settings


class Settings(BaseModel):
    """Combine the settings for the test and logging of LoadGen."""
    test: Annotated[
        TestSettings,
        Field(
            description="Test settings parameters.",
        ),
    ] = TestSettings

    logging: Annotated[
        LogSettings,
        Field(
            description="Test logging parameters",
        ),
    ] = LogSettings

    def to_lgtype(self) -> tuple[lg.TestSettings, lg.LogSettings]:
        """Return test and log settings for LoadGen."""
        test_settings = self.test.to_lgtype()
        log_settings = self.logging.to_lgtype()
        return (test_settings, log_settings)


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

    split: Annotated[
        list[str],
        Field(
            description=(
                "List of splits in order (e.g. ['train', 'test']). "
                "Allowed values: 'train', 'test'."
            ),
        ),
    ] = ["train"]

    @field_validator("split", mode="before")
    @classmethod
    def normalize_and_validate_split(cls, v: str) -> list[str]:
        """Normalize and validate the input string of field split."""
        # Allow a single string like "train" or "train,test"
        if isinstance(v, str):
            v = [part.strip() for part in v.split(",") if part.strip()]

        if not isinstance(v, list):
            err="split must be a string or a list of strings"
            raise TypeError(err)

        allowed = {"train", "test"}
        for item in v:
            if item not in allowed:
                msg_err =  f"Invalid split {item!r}. Must be one of: {sorted(allowed)}"
                raise ValueError(msg_err)
        return v


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
def evaluate(
    filename: Annotated[
        FilePath,
        Option(
            help="Location of the accuracy file.",
        ),
    ],
    dataset: Dataset,
) -> None:
    """Evaluate the accuracy of the VLM responses."""
    logger.info("Evaluating the accuracy file")
    run_evaluation(filename=filename, dataset=dataset)


@app.command()
def benchmark(
    *,
    settings: Settings,
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
    test_settings, log_settings = settings.to_lgtype()
    task = ShopifyGlobalCatalogue(
        dataset_cli=dataset,
        model_cli=model,
        endpoint_cli=endpoint,
        settings=settings.test,
        random_seed=random_seed,
    )
    sut = task.construct_sut()
    qsl = task.construct_qsl()
    logger.info("Starting the VL2L benchmark with LoadGen...")
    lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)
    logger.info("The VL2L benchmark with LoadGen completed.")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
