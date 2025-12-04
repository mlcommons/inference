"""Schema definitions of various data structures in the VL2L benchmark."""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, ClassVar, Self

import mlperf_loadgen as lg
from openai.types import ResponseFormatJSONSchema
from openai.types.chat import ChatCompletionMessageParam
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)

MAX_NUM_ESTIMATION_PERFORMANCE_SAMPLES = 100
ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES = 1 * 1024 * 1024 * 1024  # 1GB


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
                raise UnknownLoggingModeValueError(self)


class UnknownLoggingModeValueError(ValueError):
    """The exception raised when an unknown logging mode is encountered."""

    def __init__(self, logging_mode: LoggingMode) -> None:
        """Initialize the exception."""
        super().__init__(f"Unknown logging mode: {logging_mode}")


class BaseModelWithAttributeDescriptionsFromDocstrings(BaseModel):
    """Base model that automatically adds attribute descriptions from docstrings."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Pydantic settings for
    - Automatically add the attribute descriptions from docstrings.
    - Forbid extra attributes.
    """


class TestSettings(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The test settings for the MLPerf inference LoadGen."""

    scenario: TestScenario = TestScenario.OFFLINE
    """The MLPerf inference benchmarking scenario to run the benchmark in."""

    mode: TestMode = TestMode.PERFORMANCE_ONLY
    """Whether you want to run the benchmark for performance measurement or accuracy
    evaluation.
    """

    offline_expected_qps: float = 100
    """The expected QPS for the offline scenario."""

    server_expected_qps: float = 1
    """The expected QPS for the server scenario. Loadgen will try to send as many
    request as necessary to achieve this value.
    """

    server_target_latency: timedelta = timedelta(seconds=1)
    """Expected latency constraint for Server scenario. This is a constraint that we
    expect depending on the argument server_expected_qps. When server_expected_qps
    increases, we expect the latency to also increase. When server_expected_qps
    decreases, we expect the latency to also decrease.
    """

    server_ttft_latency: timedelta = timedelta(seconds=1)
    """Time to First Token (TTFT) latency constraint result validation (used when
    use_token_latencies is enabled).
    """

    server_tpot_latency: timedelta = timedelta(seconds=1)
    """Time per Output Token (TPOT) latency constraint result validation (used when
    use_token_latencies is enabled).
    """

    min_duration: timedelta = timedelta(seconds=5)
    """The minimum testing duration (in seconds or ISO 8601 format like `PT5S`). The
    benchmark runs until this value has been met.
    """

    min_query_count: int = 100
    """The minimum testing query count. The benchmark runs until this value has been
    met. If min_query_count is less than the total number of samples in the dataset,
    only the first min_query_count samples will be used during testing.
    """

    performance_sample_count_override: Annotated[
        NonNegativeInt,
        Field(
            description="The number of samples to use for the performance test. In the "  # noqa: S608
            "performance mode, the benchmark will select P random samples from the "
            "dataset, then send enough queries using these P samples (and repeating "
            "them if necessary) to reach the min_duration and min_query_count. If a "
            "non-zero value is passed to this flag, the P will be this value. "
            "Otherwise, the benchmark will estimate how many samples can be loaded into"
            f" {ALLOWED_MEMORY_FOOTPRINT_PERFORMANCE_SAMPLES} bytes of memory "
            "based on the memory footprint of randomly selected "
            f"{MAX_NUM_ESTIMATION_PERFORMANCE_SAMPLES} samples (at most), and then"
            " use this estimation as the value P.",
        ),
    ] = 0

    use_token_latencies: bool = False
    """By default, the Server scenario will use `server_target_latency` as the
    constraint. When set to True, the Server scenario will use `server_ttft_latency` and
    `server_tpot_latency` as the constraint.
    """

    @field_validator(
        "server_target_latency",
        "server_ttft_latency",
        "server_tpot_latency",
        "min_duration",
        mode="before",
    )
    @classmethod
    def parse_timedelta(cls, value: timedelta | float |
                        str) -> timedelta | str:
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
            self.server_target_latency.total_seconds() * 1e9,
        )
        settings.ttft_latency = round(
            self.server_ttft_latency.total_seconds() * 1e9)
        settings.tpot_latency = round(
            self.server_tpot_latency.total_seconds() * 1e9)
        settings.min_duration_ms = round(
            self.min_duration.total_seconds() * 1000)
        settings.min_query_count = self.min_query_count
        settings.performance_sample_count_override = (
            self.performance_sample_count_override
        )
        settings.use_token_latencies = self.use_token_latencies
        return settings


class LogOutputSettings(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The test log output settings for the MLPerf inference LoadGen."""

    outdir: DirectoryPath = DirectoryPath("./output")
    """Where to save the output files from the benchmark."""

    prefix: str = "mlperf_log_"
    """Modify the filenames of the logs with a prefix."""

    suffix: str = ""
    """Modify the filenames of the logs with a suffix."""

    prefix_with_datetime: bool = False
    """Modify the filenames of the logs with a datetime."""

    copy_detail_to_stdout: bool = False
    """Print details of performance test to stdout."""

    copy_summary_to_stdout: bool = True
    """Print results of performance test to terminal."""

    @field_validator("outdir", mode="before")
    @classmethod
    def parse_directory_field(cls, value: str) -> Path:
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


class LogSettings(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The test log settings for the MLPerf inference LoadGen."""

    log_output: LogOutputSettings = LogOutputSettings()
    """Log output settings"""

    log_mode: LoggingMode = LoggingMode.ASYNC_POLL
    """How and when logging should be sampled and stringified at runtime"""

    enable_trace: bool = True
    """Enable trace"""

    def to_lgtype(self) -> lg.LogSettings:
        """Convert log settings to its corresponding LoadGen type."""
        log_settings = lg.LogSettings()
        log_settings.log_output = self.log_output.to_lgtype()
        log_settings.log_mode = self.log_mode.to_lgtype()
        log_settings.enable_trace = self.enable_trace
        return log_settings


class Settings(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Combine the settings for the test and logging of LoadGen."""

    test: TestSettings
    """Test settings parameters."""

    logging: LogSettings
    """Test logging parameters."""

    def to_lgtype(self) -> tuple[lg.TestSettings, lg.LogSettings]:
        """Return test and log settings for LoadGen."""
        test_settings = self.test.to_lgtype()
        log_settings = self.logging.to_lgtype()
        return (test_settings, log_settings)


class Model(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Specifies the model to use for the VL2L benchmark."""

    repo_id: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    """The HuggingFace repository ID of the model."""


class Dataset(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Specifies a dataset on HuggingFace."""

    repo_id: str = "Shopify/the-catalogue-public-beta"
    """The HuggingFace repository ID of the dataset."""

    token: str | None = None
    """The token to access the HuggingFace repository of the dataset."""

    revision: str | None = None
    """The revision of the dataset."""

    split: list[str] = ["train", "test"]
    """Dataset splits to use for the benchmark, e.g., "train" and "test". You can add
    multiple splits by repeating the same CLI flag multiple times, e.g.:
    --dataset.split test --dataset.split train
    The testing dataset is a concatenation of these splits in the same order.
    """


class Verbosity(StrEnum):
    """The verbosity level of the logger."""

    TRACE = auto()
    """The trace verbosity level."""

    DEBUG = auto()
    """The debug verbosity level."""

    INFO = auto()
    """The info verbosity level (default)."""


class Endpoint(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Specifies the OpenAI API endpoint to use for the VL2L benchmark."""

    url: str = "http://localhost:8000/v1"
    """The URL of the OpenAI API endpoint that the inference requests are sent to."""

    api_key: str = ""
    """The API key to authenticate the inference requests."""

    model: Model
    """The model to use for the VL2L benchmark, i.e., the model that was deployed behind
    this OpenAI API endpoint.
    """

    use_guided_decoding: bool = True
    """If True, the benchmark will enable guided decoding for the requests. This
    requires the endpoint (and the inference engine behind it) to support guided
    decoding. If False, the response from the endpoint might not be directly parsable
    by the response JSON schema (e.g., the JSON object might be fenced in a
    ```json ... ``` code block).
    """


class EndpointToDeploy(Endpoint):
    """Specifies the endpoint to deploy for the VL2L benchmark."""

    startup_timeout: timedelta = timedelta(minutes=20)
    """The timeout for the endpoint to start up."""

    shutdown_timeout: timedelta = timedelta(minutes=1)
    """The timeout for the endpoint to shut down."""

    poll_interval: timedelta = timedelta(seconds=60)
    """The interval to poll the endpoint for readiness."""

    healthcheck_timeout: timedelta = timedelta(seconds=5)
    """The timeout for the healthcheck request to the endpoint."""


class VllmEndpoint(EndpointToDeploy):
    """Specifies how to deploy an OpenAI API endpoint in vLLM for benchmarking."""

    cli: list[str] = []
    """The CLI arguments to pass to `vllm serve`. This excludes vllm's `--host`,
    `--port`, --api-key` and `--model` CLI arguments which will be determined by
    the `url`, `api_key` and `model` fields of this schema."""

    @model_validator(mode="after")
    def validate_cli(self) -> Self:
        """Validate the vllm CLI arguments."""
        for flag in self.cli:
            if not flag.startswith(("--", "-")):
                raise PositionalVllmCliFlagError(flag)
            if flag.split("=", 1)[0] in BlacklistedVllmCliFlagError.BLACKLIST:
                raise BlacklistedVllmCliFlagError(flag)
        return self


class PositionalVllmCliFlagError(ValueError):
    """The exception raised when a positional vllm CLI flag is encountered."""

    def __init__(self, flag: str) -> None:
        """Initialize the exception."""
        super().__init__(
            f"Positional vllm CLI flag: {flag} is not allowed. Only optional flags are "
            "allowed to be passed to `--vllm.cli`.",
        )


class BlacklistedVllmCliFlagError(ValueError):
    """The exception raised when a blacklisted vllm CLI flag is encountered."""

    BLACKLIST: ClassVar[list[str]] = [
        "--model", "--host", "--port", "--api-key"]

    def __init__(self, flag: str) -> None:
        """Initialize the exception."""
        super().__init__(
            f"Blacklisted vllm CLI flag: {flag} is not allowed. The blacklisted flags"
            f"are {self.BLACKLIST}.",
        )


class ProductMetadata(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Json format for the expected responses from the VLM."""

    category: str
    """The complete category of the product, e.g.,
    "Clothing & Accessories > Clothing > Shirts > Polo Shirts".
    Each categorical level is separated by " > ".
    """

    brand: str
    """The brand of the product, e.g., "giorgio armani"."""

    is_secondhand: bool
    """True if the product is second-hand, False otherwise."""


class LoadedSample(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Sample format to be used by LoadGen."""

    messages: list[ChatCompletionMessageParam]
    """The messages to be sent for chat completion to the VLM inference endpoint."""

    response_format: ResponseFormatJSONSchema | None = None
    """The response format to be used during guided decoding."""

    @field_validator("messages", mode="after")
    @classmethod
    def ensure_content_is_list(
        cls,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """If the content is a `ValidatorIterator`, convert it back to a list.

        This is to workaround a Pydantic bug. See
        https://github.com/pydantic/pydantic/issues/9467 for more details.
        """
        for message in messages:
            if (
                "content" in message
                and message["content"].__class__.__module__
                == "pydantic_core._pydantic_core"
                and message["content"].__class__.__name__ == "ValidatorIterator"
            ):
                message["content"] = list(
                    message["content"])  # type: ignore[arg-type]
        return messages
