from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
import os


class PerformanceCheck(BaseCheck):
    """Validate performance-related submission artifacts and metrics.

    The `PerformanceCheck` class performs a comprehensive set of validations
    on submission performance outputs. It inspects the parsed MLPerf log,
    system JSON, and configuration to ensure that performance runs meet
    required constraints such as sample counts, latency limits, seed values,
    minimum durations, and scenario-specific rules. It also handles result
    inference for edge cases and validates network mode configurations.

    Attributes:
        submission_logs (SubmissionLogs): Holder for submission log paths
            and parsed contents (performance logs, system JSON, loader data).
        mlperf_log: Parsed MLPerf log object for inspecting run metadata and
            results.
        system_json (dict): Parsed system description JSON for hardware
            validation.
        config (Config): Configuration provider for targets, constraints,
            and feature toggles.
    """

    def __init__(self, log, path, config: Config,
                 submission_logs: SubmissionLogs):
        """Initialize the performance checker.

        Args:
            log: Logger instance used to report messages.
            path: Path to the submission being checked.
            config (Config): Configuration provider for performance targets
                and constraints.
            submission_logs (SubmissionLogs): Parsed submission logs and
                artifact paths (performance logs, system JSON, loader data).
        """
        super().__init__(log, path)
        self.name = "performance checks"
        self.submission_logs = submission_logs
        self.mlperf_log = self.submission_logs.performance_log
        self.system_json = self.submission_logs.system_json
        self.config = config
        self.model = self.submission_logs.loader_data.get("benchmark", "")
        self.model_mapping = self.submission_logs.loader_data.get(
            "model_mapping", {})
        self.model = self.config.get_mlperf_model(
            self.model, self.model_mapping)
        self.scenario_fixed = self.submission_logs.loader_data.get(
            "scenario", "")
        self.scenario = self.mlperf_log["effective_scenario"]
        self.division = self.submission_logs.loader_data.get("division", "")
        self.setup_checks()

    def setup_checks(self):
        """Register individual performance-related checks.

        Adds the per-submission validation callables to `self.checks` in
        the order they should be executed.
        """
        self.checks.append(self.missing_check)
        self.checks.append(self.scenarios_check)
        self.checks.append(self.loadgen_errors_check)
        self.checks.append(self.equal_issue_check)
        self.checks.append(self.performance_sample_count_check)
        self.checks.append(self.seeds_check)
        self.checks.append(self.latency_check)
        self.checks.append(self.min_query_count_check)
        self.checks.append(self.min_duration_check)
        self.checks.append(self.network_check)
        self.checks.append(self.llm_check)
        self.checks.append(self.inferred_check)
        self.checks.append(self.get_performance_metric_check)

    def missing_check(self):
        """Ensure the performance log was provided.

        Returns:
            bool: True if `mlperf_log` is present, False otherwise.
        """
        if self.mlperf_log is None:
            self.log.error("Performance log missing at %s", self.path)
            return False
        return True
    
    def scenarios_check(self):
        if self.submission_logs.loader_data.get("check_scenarios", False):
            return True
        else:
            missing_scenarios = self.submission_logs.loader_data.get("missing_scenarios", [])
            unknown_scenarios = self.submission_logs.loader_data.get("unknown_scenarios", [])
            if len(missing_scenarios) > 0:
                self.log.error(
                    "%s does not have all required scenarios, missing %s",
                    self.path,
                    missing_scenarios,
                )
            if len(unknown_scenarios) > 0:
                self.log.error(
                    "%s has all unknown scenarios for this benchmark %s",
                    self.path,
                    unknown_scenarios,
                )
            return False

    def loadgen_errors_check(self):
        """Detect Loadgen errors reported in the MLPerf log.

        If errors are present and not ignored by configuration, logs the
        error messages and returns False to indicate failure.

        Returns:
            bool: True if no blocking Loadgen errors are present,
                False otherwise.
        """
        if self.mlperf_log.has_error():
            if self.config.ignore_uncommited:
                has_other_errors = False
                for error in self.mlperf_log.get_errors():
                    if "Loadgen built with uncommitted changes!" not in error["value"]:
                        has_other_errors = True
            self.log.error("%s contains errors:", self.path)
            for error in self.mlperf_log.get_errors():
                self.log.error("%s", error["value"])

            if not self.config.ignore_uncommited or has_other_errors:
                self.log.error(
                    "%s has loadgen errors, number of errors: %s", self.path, self.mlperf_log.num_errors()
                )
                return False
        return True

    def equal_issue_check(self):
        """Verify equal-issue mode is enabled for required models.

        For models requiring equal-issue mode, checks that
        `sample_concatenate_permutation` is True.

        Returns:
            bool: True if equal-issue mode is correctly set or not required,
                False otherwise.
        """
        if self.config.requires_equal_issue(
                self.model, self.division) and self.mlperf_log["effective_sample_concatenate_permutation"]:
            self.log.error(
                "%s requires equal issue mode (sample_concatenate_permutation), expected=true, found=false",
                self.path)
            return False
        return True

    def performance_sample_count_check(self):
        """Ensure the performance run used sufficient samples.

        Compares the effective performance sample count against the
        configured minimum for the model.

        Returns:
            bool: True if the sample count meets or exceeds the requirement,
                False otherwise.
        """
        required_performance_sample_count = self.config.get_performance_sample_count(
            self.model)
        performance_sample_count = self.mlperf_log["effective_performance_sample_count"]
        if performance_sample_count < required_performance_sample_count:
            self.log.error(
                "%s performance_sample_count, found %d, needs to be >= %d",
                self.path,
                performance_sample_count,
                required_performance_sample_count,
            )
            return False
        return True

    def seeds_check(self):
        """Validate RNG seeds match the submission fixed values.

        Checks that QSL, sample index, and schedule RNG seeds from the log
        match the expected values from `config.seeds`.

        Returns:
            bool: True if all seeds match, False if any mismatch.
        """
        config_seeds = self.config.seeds
        qsl_rng_seed = self.mlperf_log["effective_qsl_rng_seed"]
        sample_index_rng_seed = self.mlperf_log["effective_sample_index_rng_seed"]
        schedule_rng_seed = self.mlperf_log["effective_schedule_rng_seed"]
        is_valid = True
        if qsl_rng_seed != config_seeds["qsl_rng_seed"]:
            self.log.error(
                "%s qsl_rng_seed is wrong, expected=%s, found=%s",
                self.path,
                config_seeds["qsl_rng_seed"],
                qsl_rng_seed,
            )
            is_valid = False
        if sample_index_rng_seed != config_seeds["sample_index_rng_seed"]:
            self.log.error(
                "%s sample_index_rng_seed is wrong, expected=%s, found=%s",
                self.path,
                config_seeds["sample_index_rng_seed"],
                sample_index_rng_seed,
            )
            is_valid = False
        if schedule_rng_seed != config_seeds["schedule_rng_seed"]:
            self.log.error(
                "%s schedule_rng_seed is wrong, expected=%s, found=%s",
                self.path,
                config_seeds["schedule_rng_seed"],
                schedule_rng_seed,
            )
            is_valid = False
        return is_valid

    def latency_check(self):
        """Enforce latency constraints based on scenario and early stopping.

        For scenarios using early stopping, verifies the condition was met
        and target latency constraints. For others, checks 99th percentile
        latency against configured limits.

        Returns:
            bool: True if latency constraints are satisfied, False otherwise.
        """
        uses_early_stopping = self.config.uses_early_stopping(self.scenario)
        if uses_early_stopping:
            # check if early_stopping condition was met
            if not self.mlperf_log["early_stopping_met"]:
                early_stopping_result = self.mlperf_log["early_stopping_result"]
                self.log.error(
                    "Early stopping condition was not met, msg=%s",
                    early_stopping_result,
                )
                return False
            # If the scenario has a target latency (Server scenario), check
            # that the target latency that was passed to the early stopping
            # is less than the target latency.
            target_latency = self.config.latency_constraint.get(
                self.model, dict()).get(self.scenario)
            if target_latency:
                early_stopping_latency_ns = self.mlperf_log["effective_target_latency_ns"]
                self.log.info(
                    "Target latency: %s, Early Stopping Latency: %s, Scenario: %s",
                    target_latency,
                    early_stopping_latency_ns,
                    self.scenario,
                )
                if early_stopping_latency_ns > target_latency:
                    self.log.error(
                        "%s Latency constraint with early stopping not met, expected=%s, found=%s",
                        self.path,
                        target_latency,
                        early_stopping_latency_ns,
                    )
                    return False
        else:
            # check if the benchmark meets latency constraint
            latency_99_percentile = self.mlperf_log["result_99.00_percentile_latency_ns"]
            target_latency = self.config.latency_constraint.get(
                self.model, dict()).get(self.scenario)
            self.log.info(
                "Target latency: %s, Latency: %s, Scenario: %s",
                target_latency,
                latency_99_percentile,
                self.scenario,
            )
            if target_latency:
                if latency_99_percentile > target_latency:
                    self.log.error(
                        "%s Latency constraint not met, expected=%s, found=%s",
                        self.path,
                        target_latency,
                        latency_99_percentile,
                    )
                    return False
        return True

    def min_query_count_check(self):
        """Verify minimum query counts and samples per query are met.

        Checks minimum query count for non-early-stopping scenarios and
        enforces minimum samples per query for Offline scenarios in closed
        division.

        Returns:
            bool: True if all minimum requirements are satisfied,
                False otherwise.
        """
        uses_early_stopping = self.config.uses_early_stopping(self.scenario)
        min_query_count = self.mlperf_log["effective_min_query_count"]
        samples_per_query = self.mlperf_log["effective_samples_per_query"]
        if not uses_early_stopping:
            required_min_query_count = self.config.get_min_query_count(
                self.model, self.scenario)
            if required_min_query_count and min_query_count < required_min_query_count:
                self.log.error(
                    "%s Required minimum Query Count not met by user config, Expected=%s, Found=%s",
                    self.path,
                    required_min_query_count,
                    min_query_count,
                )
                return False
        if self.scenario.lower() == "offline" and (
                samples_per_query < OFFLINE_MIN_SPQ_SINCE_V4[self.model]) and self.division.lower() == "closed":
            self.log.error(
                "%s Required minimum samples per query not met by user config, Expected=%s, Found=%s",
                self.path,
                OFFLINE_MIN_SPQ_SINCE_V4[self.model],
                samples_per_query,
            )
            return False
        return True

    def min_duration_check(self):
        """Ensure the test duration meets the minimum requirement.

        Verifies that the effective minimum duration is at least
        `TEST_DURATION_MS` (600 seconds).

        Returns:
            bool: True if duration meets the minimum, False otherwise.
        """
        required_min_duration = TEST_DURATION_MS
        min_duration = self.mlperf_log["effective_min_duration_ms"]
        if min_duration < required_min_duration:
            self.log.error(
                "%s Test duration less than 600s in user config. expected=%s, found=%s",
                self.path,
                required_min_duration,
                min_duration,
            )
            return False
        return True

    def network_check(self):
        """Validate network mode settings and SUT naming.

        Ensures the system JSON indicates the correct network mode for the
        division and that SUT names comply with network mode requirements.

        Returns:
            bool: True if network mode and naming are valid, False otherwise.
        """
        if self.system_json is None:
            self.log.error(
                "%s system json file not found",
                self.path
            )
            return False
        is_network_mode_sys_spec_str = self.system_json.get(
            SYSTEM_DESC_IS_NETWORK_MODE)
        is_network_system = (
            is_network_mode_sys_spec_str.lower() == "true"
            if is_network_mode_sys_spec_str is not None
            else False
        )
        # verify that the system corresponds the division
        is_valid = True
        expected_state_by_division = {"network": True, "closed": False}
        if self.division in expected_state_by_division:
            is_valid = expected_state_by_division[self.division] is is_network_system
        if not is_valid:
            self.log.error(
                f"{self.path} incorrect network mode(={is_network_system}) "
                f"for division {self.division}"
            )
            return False

        sut_name = self.mlperf_log["sut_name"]
        if is_network_system:
            # for network mode verify the SUT name is valid, according to the rules
            # (must include "Network SUT" in name)
            if NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME not in sut_name:
                self.log.error(
                    f"{self.path} invalid sut name for network mode. expecting the substring '{NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME}' got '{sut_name}'"
                )
                return False

        return True

    def llm_check(self):
        """Perform LLM-specific latency validations for token latencies.

        For LLM models, ensures token latencies are enabled and that TTFT
        and TPOT metrics meet configured limits for applicable scenarios.

        Returns:
            bool: True if LLM checks pass or model is not an LLM,
                False otherwise.
        """
        if self.model in self.config.get_llm_models():
            if self.mlperf_log["requested_use_token_latencies"]:
                if self.scenario not in ["Server", "Interactive"]:
                    # For offline, singlestream and multistream no further checks are
                    # necessary
                    return True
                else:
                    limits = LLM_LATENCY_LIMITS[self.model][self.scenario]
                    if (
                        self.mlperf_log["result_first_token_99.00_percentile_latency_ns"]
                        < limits["ttft"]
                        and self.mlperf_log["result_time_per_output_token_99.00_percentile_ns"]
                        < limits["tpot"]
                    ):
                        return True
            else:
                self.log.error(
                    f"use_token_latencies flag needs to be enabled for Llama2 benchmark")
                return False

            self.log.error(
                'Failed extra check for TTFT and TPOT. Obtained: TTFT 99-tile: %.4f, TPOT 99-tile: %.4f. Required: TTFT 99-tile: %.4f, TPOT 99-tile: %.4f',
                self.mlperf_log["result_first_token_99.00_percentile_latency_ns"],
                self.mlperf_log["result_time_per_output_token_99.00_percentile_ns"],
                limits["ttft"],
                limits["tpot"]
            )
            return False
        return True

    def inferred_check(self):
        """Validate rules for inferring results across scenarios.

        Ensures that result inference is only allowed for edge systems and
        specific scenario pairs, preventing invalid cross-scenario reuse.

        Returns:
            bool: True if inference is valid or not attempted, False otherwise.
        """
        if self.scenario.lower() != self.scenario_fixed.lower() and (
                self.scenario.lower(), self.scenario_fixed.lower()) != ("server", "interactive"):
            if "edge" not in self.system_json["system_type"].lower():
                self.log.error(
                    "Result can not be inferred for %s suite for: %s. Scenario: %s, Scenario fixed: %s",
                    self.system_json["system_type"],
                    self.path,
                    self.scenario,
                    self.scenario_fixed)
                return False
            list_inferred = [
                ("singlestream", "multistream"),
                ("multistream", "offline"),
                ("singlestream", "offline")
            ]
            if (self.scenario.lower(), self.scenario_fixed.lower()
                ) not in list_inferred:
                self.log.error(
                    "Result for scenario %s can not be inferred from %s for: %s",
                    self.scenario_fixed,
                    self.scenario,
                    self.path)
                return False
        return True

    def get_performance_metric_check(self):
        """Extract and validate the primary performance metric.

        Parses the performance result from the log, applies any benchmark-
        specific overwrites, and handles inferred results. Records the
        metric in `submission_logs.loader_data`.

        Returns:
            bool: True if the metric is valid, False otherwise.
        """
        # Assumes new logging format
        is_valid = True
        version = self.config.version
        if (
            "result_validity" in self.mlperf_log.get_keys()
            and self.mlperf_log["result_validity"] == "VALID"
        ):
            is_valid = True
        scenario = self.mlperf_log["effective_scenario"]

        res = float(self.mlperf_log[RESULT_FIELD_NEW[version][scenario]])
        if (
            version in RESULT_FIELD_BENCHMARK_OVERWRITE
            and self.model in RESULT_FIELD_BENCHMARK_OVERWRITE[version]
            and scenario in RESULT_FIELD_BENCHMARK_OVERWRITE[version][self.model]
        ):
            res = float(
                self.mlperf_log[RESULT_FIELD_BENCHMARK_OVERWRITE[version]
                                [self.model][scenario]]
            )

        inferred = False
        if self.scenario.lower() != self.scenario_fixed.lower() and (
                self.scenario.lower(), self.scenario_fixed.lower()) != ("server", "interactive"):
            res, is_valid = self.get_inferred_result(res)
        self.submission_logs.loader_data["performance_metric"] = res
        return is_valid

    def get_inferred_result(self, res):
        """Compute inferred performance result for cross-scenario reuse.

        Calculates the performance metric for the fixed scenario based on
        the run scenario's results, applying scenario-specific formulas.

        Args:
            res (float): The raw performance result from the log.

        Returns:
            tuple: (inferred_result, is_valid) where is_valid indicates if
                inference was successful.
        """
        inferred = False
        is_valid = True
        # Check if current scenario (and version) uses early stopping
        uses_early_stopping = self.config.uses_early_stopping(self.scenario)

        latency_mean = self.mlperf_log["result_mean_latency_ns"]
        if self.scenario in ["MultiStream"]:
            latency_99_percentile = self.mlperf_log[
                "result_99.00_percentile_per_query_latency_ns"
            ]
            latency_mean = self.mlperf_log["result_mean_query_latency_ns"]
        samples_per_query = self.mlperf_log["effective_samples_per_query"]
        if self.scenario == "SingleStream":
            # qps_wo_loadgen_overhead is only used for inferring Offline from
            # SingleStream; only for old submissions
            qps_wo_loadgen_overhead = self.mlperf_log["result_qps_without_loadgen_overhead"]

        # special case for results inferred from different scenario
        if self.scenario_fixed in [
                "Offline"] and self.scenario in ["SingleStream"]:
            inferred = True
            res = qps_wo_loadgen_overhead

        if (self.scenario_fixed in ["Offline"]
            ) and self.scenario in ["MultiStream"]:
            inferred = True
            res = samples_per_query * S_TO_MS / (latency_mean / MS_TO_NS)

        if (self.scenario_fixed in ["MultiStream"]
            ) and self.scenario in ["SingleStream"]:
            inferred = True
            # samples_per_query does not match with the one reported in the logs
            # when inferring MultiStream from SingleStream
            samples_per_query = 8
            if uses_early_stopping:
                early_stopping_latency_ms = self.mlperf_log["early_stopping_latency_ms"]
                if early_stopping_latency_ms == 0:
                    self.log.error(
                        "Not enough samples were processed for early stopping to make an estimate"
                    )
                    is_valid = False
                res = (early_stopping_latency_ms *
                       samples_per_query) / MS_TO_NS
            else:
                res = (latency_99_percentile * samples_per_query) / MS_TO_NS
        if (self.scenario_fixed in ["Interactive"]
            ) and self.scenario not in ["Server"]:
            is_valid = False
        return res, is_valid
