"""Schema definitions of various data structures in the Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, ClassVar, Self

import mlperf_loadgen as lg
from loguru import logger
from openai.types import ResponseFormatJSONSchema
from openai.types.chat import ChatCompletionMessageParam
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
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

    @staticmethod
    def from_lgtype(lgtype: lg.TestScenario) -> TestScenario:
        """Convert the LoadGen's test scenario to the TestScenario schema."""
        match lgtype:
            case lg.TestScenario.Server:
                return TestScenario.SERVER
            case lg.TestScenario.Offline:
                return TestScenario.OFFLINE
            case _:
                raise UnknownTestScenarioValueError(lgtype)


class UnknownTestScenarioValueError(ValueError):
    """The exception raised when an unknown test scenario is encountered."""

    def __init__(self, test_scenario: TestScenario | lg.TestScenario) -> None:
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

    @staticmethod
    def from_lgtype(lgtype: lg.TestMode) -> TestMode:
        """Convert the LoadGen's test mode to the TestMode schema."""
        match lgtype:
            case lg.TestMode.PerformanceOnly:
                return TestMode.PERFORMANCE_ONLY
            case lg.TestMode.AccuracyOnly:
                return TestMode.ACCURACY_ONLY
            case _:
                raise UnknownTestModeValueError(lgtype)


class UnknownTestModeValueError(ValueError):
    """The exception raised when an unknown test mode is encountered."""

    def __init__(self, test_mode: TestMode | lg.TestMode) -> None:
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


_DEFAULT_DATASET_SIZE = 48289
_DEFAULT_MIN_DURATION = timedelta(minutes=10)
_DEFAULT_OFFLINE_EXPECTED_QPS = (
    _DEFAULT_DATASET_SIZE / _DEFAULT_MIN_DURATION.total_seconds()
)


class TestSettings(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The test settings for the MLPerf inference LoadGen."""

    scenario: TestScenario = TestScenario.OFFLINE
    """The MLPerf inference benchmarking scenario to run the benchmark in."""

    mode: TestMode = TestMode.PERFORMANCE_ONLY
    """Whether you want to run the benchmark for performance measurement or accuracy
    evaluation.
    """

    """Server-specific settings"""

    server_target_qps: float = 5
    """The average QPS of the poisson distribution. Note: This field is used as a
    FindPeakPerformance's lower bound. When you run FindPeakPerformanceMode, you should
    make sure that this value satisfies performance constraints.
    """

    server_target_latency: timedelta = timedelta(seconds=12)
    """The latency constraint for the Server scenario."""

    server_target_latency_percentile: float = 0.99
    """The latency percentile for server mode. This value is combined with
    server_target_latency to determine if a run is valid.
    """

    server_coalesce_queries: bool = False
    """If this flag is set to True, LoadGen will combine samples from
    multiple queries into a single query if their scheduled issue times have
    passed.
    """

    server_find_peak_qps_decimals_of_precision: int = 1
    """The decimal places of QPS precision used to terminate
    FindPeakPerformance mode.
    """

    server_find_peak_qps_boundary_step_size: float = 1
    """The step size (as a fraction of the QPS) used to widen the lower and
    upper bounds to find the initial boundaries of binary search.
    """

    server_max_async_queries: int = 0
    """The maximum number of outstanding queries to allow before earlying out from a
    performance run. Useful for performance tuning and speeding up the
    FindPeakPerformance mode.
    """

    server_num_issue_query_threads: int = 0
    """The number of issue query threads that will be registered and used
    to call SUT's IssueQuery(). If this is 0, the same thread calling
    StartTest() will be used to call IssueQuery(). See also
    mlperf::RegisterIssueQueryThread().
    """

    """Offline-specific settings"""

    offline_expected_qps: float = _DEFAULT_OFFLINE_EXPECTED_QPS
    """Specifies the QPS the SUT expects to hit for the offline load.
    The LoadGen generates 10% more queries than it thinks it needs to meet
    the minimum test duration.
    """

    sample_concatenate_permutation: bool = True
    """Affects the order in which the samples of the dataset are chosen.
    If False, it concatenates a single permutation of the dataset (or part
    of it depending on performance_sample_count_override) several times up to the
    number of samples requested.
    If True, it concatenates a multiple permutation of the dataset (or a
    part of it depending on `performance_sample_count_override`) several times
    up to the number of samples requested.
    """

    """Test duration settings"""

    min_duration: timedelta = _DEFAULT_MIN_DURATION
    """The minimum testing duration (in seconds or ISO 8601 format like `PT5S`). The
    benchmark runs until this value has been met.
    """

    max_duration: timedelta = timedelta(seconds=0)
    """The maximum testing duration (in seconds or ISO 8601 format like `PT5S`). The
    benchmark will exit before this value has been met. 0 means infinity.
    """

    min_query_count: int = _DEFAULT_DATASET_SIZE
    """The minimum testing query count. The benchmark runs until this value has been
    met. If min_query_count is less than the total number of samples in the dataset,
    only the first min_query_count samples will be used during testing.
    """

    max_query_count: int = 0
    """The maximum testing query count. The benchmark will exit before this value has
    been met. 0 means infinity.
    """

    """Random number generation settings"""

    qsl_rng_seed: int = 0
    """Affects which subset of samples from the QSL are chosen for
    the performance sample set and accuracy sample sets."""

    sample_index_rng_seed: int = 0
    """Affects the order in which samples from the performance set will
    be included in queries."""

    schedule_rng_seed: int = 0
    """Affects the poisson arrival process of the Server scenario.
    Different seeds will appear to "jitter" the queries
    differently in time, but should not affect the average issued QPS.
    """

    accuracy_log_rng_seed: int = 0
    """Affects which samples have their query returns logged to the
    accuracy log in performance mode."""

    accuracy_log_probability: float = 0.0
    """The probability of the query response of a sample being logged to the
    accuracy log in performance mode."""

    accuracy_log_sampling_target: int = 0
    """The target number of samples that will have their results printed to
    accuracy log in performance mode for compliance testing."""

    """Test05 settings"""

    test05: bool = False
    """Whether or not to run test05."""

    test05_qsl_rng_seed: int = 0
    """Test05 seed for which subset of samples from the QSL are chosen for
    the performance sample set and accuracy sample sets."""

    test05_sample_index_rng_seed: int = 0
    """Test05 seed for the order in which samples from the performance set will
    be included in queries."""

    test05_schedule_rng_seed: int = 0
    """Test05 seed for the poisson arrival process of the Server scenario.
    Different seeds will appear to "jitter" the queries
    differently in time, but should not affect the average issued QPS.
    """

    """Performance Sample modifiers"""

    print_timestamps: bool = False
    """Prints measurement interval start and stop timestamps to stdout
    for the purpose of comparison against an external timer."""

    performance_issue_unique: bool = False
    """Allows issuing only unique queries in Performance mode of any
    scenario. This can be used to send non-repeat & hence unique
    samples to SUT.
    """

    performance_issue_same: bool = False
    """If True, the same query is chosen repeatedley for Inference.
    In offline scenario, the query is filled with the same sample.
    """

    performance_issue_same_index: int = 0
    """Offset to control which sample is repeated in
    performance_issue_same mode. Value should be within [0, performance_sample_count).
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
    ] = _DEFAULT_DATASET_SIZE

    use_token_latencies: bool = False
    """By default, the Server scenario will use `server_target_latency` as the
    constraint. When set to True, the Server scenario will use `server_ttft_latency` and
    `server_tpot_latency` as the constraint.
    """

    server_ttft_latency: timedelta = timedelta(milliseconds=100)
    """Time to First Token (TTFT) latency constraint result validation (used when
    use_token_latencies is enabled).
    """

    server_tpot_latency: timedelta = timedelta(milliseconds=100)
    """Time per Output Token (TPOT) latency constraint result validation (used when
    use_token_latencies is enabled).
    """

    infer_token_latencies: bool = False
    """Infer token latencies from the response time."""

    token_latency_scaling_factor: int = 1
    """Only used when infer_token_latencies is enabled. The scaling factor inferring
    token latencies from the response time.
    """

    @field_validator(
        "server_target_latency",
        "min_duration",
        "max_duration",
        "server_ttft_latency",
        "server_tpot_latency",
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

        # Server-specific settings
        settings.server_target_qps = self.server_target_qps
        settings.server_target_latency_ns = round(
            self.server_target_latency.total_seconds() * 1e9,
        )
        settings.server_target_latency_percentile = (
            self.server_target_latency_percentile
        )
        settings.server_coalesce_queries = self.server_coalesce_queries
        settings.server_find_peak_qps_decimals_of_precision = (
            self.server_find_peak_qps_decimals_of_precision
        )
        settings.server_find_peak_qps_boundary_step_size = (
            self.server_find_peak_qps_boundary_step_size
        )
        settings.server_max_async_queries = self.server_max_async_queries
        settings.server_num_issue_query_threads = self.server_num_issue_query_threads

        # Offline-specific settings
        settings.offline_expected_qps = self.offline_expected_qps
        settings.sample_concatenate_permutation = self.sample_concatenate_permutation

        # Test duration settings
        settings.min_duration_ms = round(
            self.min_duration.total_seconds() * 1000)
        settings.max_duration_ms = round(
            self.max_duration.total_seconds() * 1000)
        settings.min_query_count = self.min_query_count
        settings.max_query_count = self.max_query_count

        # Random number generation settings
        settings.qsl_rng_seed = self.qsl_rng_seed
        settings.sample_index_rng_seed = self.sample_index_rng_seed
        settings.schedule_rng_seed = self.schedule_rng_seed
        settings.accuracy_log_rng_seed = self.accuracy_log_rng_seed
        settings.accuracy_log_probability = self.accuracy_log_probability
        settings.accuracy_log_sampling_target = self.accuracy_log_sampling_target

        # Test05 settings
        settings.test05 = self.test05
        settings.test05_qsl_rng_seed = self.test05_qsl_rng_seed
        settings.test05_sample_index_rng_seed = self.test05_sample_index_rng_seed
        settings.test05_schedule_rng_seed = self.test05_schedule_rng_seed

        # Performance Sample modifiers
        settings.print_timestamps = self.print_timestamps
        settings.performance_issue_unique = self.performance_issue_unique
        settings.performance_issue_same = self.performance_issue_same
        settings.performance_issue_same_index = self.performance_issue_same_index
        settings.performance_sample_count_override = (
            self.performance_sample_count_override
        )
        settings.use_token_latencies = self.use_token_latencies
        settings.ttft_latency = round(
            self.server_ttft_latency.total_seconds() * 1e9)
        settings.tpot_latency = round(
            self.server_tpot_latency.total_seconds() * 1e9)
        settings.infer_token_latencies = self.infer_token_latencies
        settings.token_latency_scaling_factor = self.token_latency_scaling_factor

        return settings

    @staticmethod
    def from_lgtype(lgtype: lg.TestSettings) -> TestSettings:
        """Convert the LoadGen's test settings to the TestSettings schema."""
        return TestSettings(
            scenario=TestScenario.from_lgtype(lgtype.scenario),
            mode=TestMode.from_lgtype(lgtype.mode),
            server_target_qps=lgtype.server_target_qps,
            server_target_latency=timedelta(
                seconds=lgtype.server_target_latency_ns / 1e9,
            ),
            server_target_latency_percentile=lgtype.server_target_latency_percentile,
            server_coalesce_queries=lgtype.server_coalesce_queries,
            server_find_peak_qps_decimals_of_precision=lgtype.server_find_peak_qps_decimals_of_precision,
            server_find_peak_qps_boundary_step_size=lgtype.server_find_peak_qps_boundary_step_size,
            server_max_async_queries=lgtype.server_max_async_queries,
            server_num_issue_query_threads=lgtype.server_num_issue_query_threads,
            offline_expected_qps=lgtype.offline_expected_qps,
            sample_concatenate_permutation=lgtype.sample_concatenate_permutation,
            min_duration=timedelta(milliseconds=lgtype.min_duration_ms),
            max_duration=timedelta(milliseconds=lgtype.max_duration_ms),
            min_query_count=lgtype.min_query_count,
            max_query_count=lgtype.max_query_count,
            qsl_rng_seed=lgtype.qsl_rng_seed,
            sample_index_rng_seed=lgtype.sample_index_rng_seed,
            schedule_rng_seed=lgtype.schedule_rng_seed,
            accuracy_log_rng_seed=lgtype.accuracy_log_rng_seed,
            accuracy_log_probability=lgtype.accuracy_log_probability,
            accuracy_log_sampling_target=lgtype.accuracy_log_sampling_target,
            test05=lgtype.test05,
            test05_qsl_rng_seed=lgtype.test05_qsl_rng_seed,
            test05_sample_index_rng_seed=lgtype.test05_sample_index_rng_seed,
            test05_schedule_rng_seed=lgtype.test05_schedule_rng_seed,
            print_timestamps=lgtype.print_timestamps,
            performance_issue_unique=lgtype.performance_issue_unique,
            performance_issue_same=lgtype.performance_issue_same,
            performance_issue_same_index=lgtype.performance_issue_same_index,
            performance_sample_count_override=lgtype.performance_sample_count_override,
            use_token_latencies=lgtype.use_token_latencies,
            server_ttft_latency=timedelta(seconds=lgtype.ttft_latency / 1e9),
            server_tpot_latency=timedelta(seconds=lgtype.tpot_latency / 1e9),
            infer_token_latencies=lgtype.infer_token_latencies,
            token_latency_scaling_factor=lgtype.token_latency_scaling_factor,
        )


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


class UserConf(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The user.conf file for specifying LoadGen test settings."""

    path: FilePath | None = None
    """The path to the user.conf file. If provided, the test settings will be overridden
    with the settings from the provided user.conf file and the mlperf.conf file from
    inside LoadGen.
    """

    model: str = "qwen3-vl-235b-a22b"
    """The model name that corresponds to the entries in the mlperf.conf file (in the
    LoadGen) which defines the benchmark-wide constraints.
    """


class Settings(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Combine the settings for the test and logging of LoadGen."""

    test: TestSettings
    """Test settings parameters."""

    user_conf: UserConf
    """The user.conf file for specifying LoadGen test settings."""

    logging: LogSettings
    """Test logging parameters."""

    @model_validator(mode="after")
    def override_test_settings_from_user_conf(self) -> Self:
        """Override the test settings from the user.conf file."""
        if self.user_conf.path:
            lg_test_settings = self.test.to_lgtype()
            lg_test_settings.FromConfig(
                str(self.user_conf.path),
                self.user_conf.model,
                self.test.scenario.value.capitalize(),
            )
            self.test = TestSettings.from_lgtype(lg_test_settings)
            logger.info(
                "Loaded test settings from the user.conf and mlperf.conf files: {}",
                self.test,
            )
        return self

    def to_lgtype(self) -> tuple[lg.TestSettings, lg.LogSettings]:
        """Return test and log settings for LoadGen."""
        test_settings = self.test.to_lgtype()
        log_settings = self.logging.to_lgtype()
        return (test_settings, log_settings)


class Model(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Specifies the model to use for the Qwen3-VL (Q3VL) benchmark."""

    repo_id: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    """The HuggingFace repository ID of the model."""

    token: str | None = None
    """The token to access the HuggingFace repository of the model."""

    revision: str = "710c13861be6c466e66de3f484069440b8f31389"
    """The revision of the model."""


class Dataset(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Specifies a dataset on HuggingFace."""

    repo_id: str = "Shopify/product-catalogue"
    """The HuggingFace repository ID of the dataset."""

    token: str | None = None
    """The token to access the HuggingFace repository of the dataset."""

    revision: str = "d5c517c509f5aca99053897ef1de797d6d7e5aa5"
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


class SamplingParams(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Specifies the sampling parameters for the inference request to the endpoint."""

    frequency_penalty: float | None = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their
    existing frequency in the text so far, decreasing the model's likelihood to repeat
    the same line verbatim. See
    https://platform.openai.com/docs/api-reference/chat/create#chat_create-frequency_penalty
    """

    presence_penalty: float | None = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
    they appear in the text so far, increasing the model's likelihood to talk about new
    topics. See
    https://platform.openai.com/docs/api-reference/chat/create#chat_create-presence_penalty
    """

    temperature: float | None = None
    """What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
    make the output more random, while lower values like 0.2 will make it more focused
    and deterministic. We generally recommend altering this or top_p but not both. See
    https://platform.openai.com/docs/api-reference/chat/create#chat_create-temperature
    """

    top_p: float | None = None
    """An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1 means
    only the tokens comprising the top 10% probability mass are considered. We generally
    recommend altering this or temperature but not both.
    See https://platform.openai.com/docs/api-reference/chat/create#chat_create-top_p
    """

    top_k: int | None = None
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens.
    Note that this is not part of the OpenAI API spec. Therefore, this field will be
    passed in via the `extra_body` field of the inference request to the endpoint.
    The inference engine therefore needs to support this field, such as what vLLM does
    here:
    https://github.com/vllm-project/vllm/blob/83a317f650f210b86572b13b8198b7d38aaacb7e/vllm/entrypoints/openai/protocol.py#L566
    """

    min_p: float | None = None
    """Represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token. Must be in [0, 1].
    Set to 0 to disable this.
    Note that this is not part of the OpenAI API spec. Therefore, this field will be
    passed in via the `extra_body` field of the inference request to the endpoint.
    The inference engine therefore needs to support this field, such as what vLLM does
    here:
    https://github.com/vllm-project/vllm/blob/83a317f650f210b86572b13b8198b7d38aaacb7e/vllm/entrypoints/openai/protocol.py#L567
    """

    repetition_penalty: float | None = None
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens.
    Note that this is not part of the OpenAI API spec. Therefore, this field will be
    passed in via the `extra_body` field of the inference request to the endpoint.
    The inference engine therefore needs to support this field, such as what vLLM does
    here:
    https://github.com/vllm-project/vllm/blob/83a317f650f210b86572b13b8198b7d38aaacb7e/vllm/entrypoints/openai/protocol.py#L568
    """


class Endpoint(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Specifies the OpenAI API endpoint to use for the Qwen3-VL (Q3VL) benchmark."""

    url: str = "http://localhost:8000/v1"
    """The URL of the OpenAI API endpoint that the inference requests are sent to."""

    api_key: str = ""
    """The API key to authenticate the inference requests."""

    model: Model
    """The model to use for the Qwen3-VL (Q3VL) benchmark, i.e., the model that was
    deployed behind this OpenAI API endpoint.
    """

    use_guided_decoding: bool = False
    """If True, the benchmark will enable guided decoding for the requests. This
    requires the endpoint (and the inference engine behind it) to support guided
    decoding. If False, the response from the endpoint might not be directly parsable
    by the response JSON schema (e.g., the JSON object might be fenced in a
    ```json ... ``` code block).
    """

    request_timeout: timedelta = timedelta(hours=2)
    """The timeout for the inference request to the endpoint. The default value for
    OpenAI API client is 10 minutes
    (https://github.com/openai/openai-python?tab=readme-ov-file#timeouts) which might
    not be sufficient for the offline scenario.
    """

    sampling_params: SamplingParams
    """The sampling parameters to use for the inference request to the endpoint."""


class EndpointToDeploy(Endpoint):
    """Specifies the endpoint to deploy for the Qwen3-VL (Q3VL) benchmark."""

    startup_timeout: timedelta = timedelta(hours=1)
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
        "--model",
        "--revision",
        "--host",
        "--port",
        "--hf-token",
        "--api-key",
    ]

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
