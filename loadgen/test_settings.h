/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/// \file
/// \brief Provides ways for a client to change the behavior and
/// constraints of the load generator.
/// \details Note: The MLPerf specification takes precedent over any of the
/// comments in this file if there are inconsistencies in regards to how the
/// loadgen *should* work.
/// The comments in this file are indicative of the loadgen implementation.

#ifndef MLPERF_LOADGEN_TEST_SETTINGS_H
#define MLPERF_LOADGEN_TEST_SETTINGS_H

#include <cstdint>
#include <string>

namespace mlperf {

/// \addtogroup LoadgenAPI
/// @{

/// \addtogroup LoadgenAPITestSettings Test Settings
/// \brief This page contains a description of all the scenarios, modes,
/// and log settings as implemented by the LoadGen.
/// @{

///
/// \enum TestScenario
/// * **SingleStream**
///  + Issues queries containing a single sample.
///  + The next query is only issued once the previous one has completed.
///  + Internal LoadGen latency between queries is not included in the
///    latency results.
///  + **Final performance result is:** a percentile of the latency.
///   - Percentile will be programmable via TestSettings soon.
/// * **MultiStream**
///  + Attempts to issue queries containing N samples each at a uniform rate.
///   - N is specified by \link
///   mlperf::TestSettings::multi_stream_samples_per_query
///   multi_stream_samples_per_query \endlink.
///   - The rate is specified by \link
///   mlperf::TestSettings::multi_stream_target_qps multi_stream_target_qps
///   \endlink.
///  + The loadgen will skip sending for one interval if the SUT falls behind
///    too much.
///  + By default, only a single query may be outstanding at a time.
///  + Latency is tracked on a per-sample basis, as opposed to per-query.
///  + The samples of each query are guaranteed to be contiguous with respect
///    to the order they were loaded in the QuerySampleLibrary.
///  + **Final performance result is:** PASS if a percentile of the latency is
///    under a given threshold. FAIL otherwise.
///   - Percentile will be programmable via TestSettings soon.
///   - Threshold is specified by \link
///   mlperf::TestSettings::multi_stream_target_latency_ns
///   multi_stream_target_latency_ns \endlink.
/// * **MultiStreamFree**
///  + Behaves similar to MultiStream, with the exceptions that it:
///   - Allows up to N async queries where N is limited only by the latency
///     target.
///   - Issues queries at a variable rate corresponding to when the N'th
///     oldest query completes.
///  + Not an official MLPerf scenario, but is maintained for evaluation
///    and testing purposes.
///  + Compared to MultiStream, there is no frequency quantization, which
///    allows the results to reflect small performance improvements.
///  + **Final performance result is:** samples per second.
/// * **Server**
///  + Sends queries with a single sample.
///  + Queries have a random poisson (non-uniform) arrival rate that, when
///    averaged, hits the target QPS.
///  + **Final performance result is:** PASS if the a percentile of the latency
///    is under a given threshold. FAIL otherwise.
///   - Percentile will be programmable via TestSettings soon.
///   - Threshold is specified by \link
///   mlperf::TestSettings::server_target_latency_ns server_target_latency_ns
///   \endlink.
/// * **Offline**
///  + Sends all N samples to the SUT inside of a single query.
///  + The samples of the query are guaranteed to be contiguous with respect
///    to the order they were loaded in the QuerySampleLibrary.
///    (WIP. Not true yet.)
///  + **Final performance result is:** samples per second.
///
enum class TestScenario {
  SingleStream,
  MultiStream,
  MultiStreamFree,
  Server,
  Offline,
};

///
/// \enum TestMode
/// * **SubmissionRun**
///  + Runs accuracy mode followed by performance mode.
///  + TODO: Implement further requirements as decided by MLPerf.
/// * **AccuracyOnly**
///  + Runs each sample from the QSL through the SUT a least once.
///  + Outputs responses to an accuracy json that can be parsed by a model +
///    sample library specific script.
/// * **PerformanceOnly**
///  + Runs the performance traffic for the given scenario, as described in
///    the comments for TestScenario.
/// * **FindPeakPerformance**
///  + TODO: Not implemented yet.
///  + Determines the maximumum QPS for the Server scenario.
///  + Determines the maximum samples per query for the MultiStream and
///    MultiStreamFree scenarios.
///  + Not applicable for SingleStream or Offline scenarios.
///
enum class TestMode {
  SubmissionRun,
  AccuracyOnly,
  PerformanceOnly,
  FindPeakPerformance,
};

///
/// \brief Top-level struct specifing the modes and parameters of the test.
///
/// \todo Create TestSetting from a config file.
struct TestSettings {
  TestScenario scenario = TestScenario::SingleStream;
  TestMode mode = TestMode::PerformanceOnly;

  // ==================================
  /// \name SingleStream-specific
  /**@{*/
  /// \brief A hint used by the loadgen to pre-generate enough samples to
  ///        meet the minimum test duration.
  uint64_t single_stream_expected_latency_ns = 1000000;
  double single_stream_target_latency_percentile = 0.90;
  /**@}*/

  // ==================================
  /// \name MultiStream-specific
  /**@{*/
  /// \brief The uniform rate at which queries are produced.
  ///        Does not apply to the MultiStreamFree scenario.
  double multi_stream_target_qps = 10.0;
  /// \brief The latency constraint for the MultiStream scenario.
  uint64_t multi_stream_target_latency_ns = 100000000;
  /// \brief The latency percentile for multistream mode. This value is combined
  /// with multi_stream_target_latency_ns to determine if a run is valid.
  double multi_stream_target_latency_percentile = 0.9;
  /// \brief The number of samples in each query.
  int multi_stream_samples_per_query = 4;
  /// \brief The maximum number of queries, to which a SUT has not responded,
  /// before the loadgen will throttle issuance of new queries.
  int multi_stream_max_async_queries = 1;
  /**@}*/

  // ==================================
  /// \name Server-specific
  /**@{*/
  /// \brief The average QPS of the poisson distribution.
  double server_target_qps = 1;
  /// \brief The latency constraint for the Server scenario.
  uint64_t server_target_latency_ns = 100000000;
  /// \brief The latency percentile for server mode. This value is combined with
  /// server_target_latency_ns to determine if a run is valid.
  /// \details 99% is the default value, which is correct for image models. GNMT
  /// should be set to 0.97 (97%) in v0.5.(As always, check the policy page for
  /// updated values for the benchmark you are running.)
  double server_target_latency_percentile = 0.99;
  /// \brief TODO: Implement this.
  bool server_coalesce_queries = false;
  /// \brief The decimal places of QPS precision used to terminate
  /// FindPeakPerformance mode.
  int server_find_peak_qps_decimals_of_precision = 1;
  /// \brief A step size used to widen lower & upper bound to find the initial
  /// boundaries of binary search
  double server_find_peak_qps_boundary_step_size = 1;
  /**@}*/

  // ==================================
  /// \name Offline-specific
  /**@{*/
  /// \brief Specifies the QPS the SUT expects to hit for the offline load.
  ///        The loadgen generates 10% more queries than it thinks it
  ///        needs to meet the minimum test duration.
  double offline_expected_qps = 1;
  /**@}*/

  // ==================================
  /// \name Test duration
  /// The test runs until **both** min duration and min query count have been
  /// met. However, it will exit before that point if **either** max duration or
  /// max query count have been reached.
  /**@{*/
  uint64_t min_duration_ms = 10000;
  uint64_t max_duration_ms = 0;  ///< 0: Infinity.
  uint64_t min_query_count = 100;
  uint64_t max_query_count = 0;  ///< 0: Infinity.
  /**@}*/

  // ==================================
  /// \name Random number generation
  /// There are 3 separate seeds, so each dimension can be changed
  /// independently.
  /**@{*/
  /// \brief Affects which subset of samples from the QSL are chosen for
  /// the performance sample set and accuracy sample sets.
  uint64_t qsl_rng_seed = 0;
  /// \brief Affects the order in which samples from the performance set will
  /// be included in queries.
  uint64_t sample_index_rng_seed = 0;
  /// \brief Affects the poisson arrival process of the Server scenario.
  /// \details Different seeds will appear to "jitter" the queries
  /// differently in time, but should not affect the average issued QPS.
  uint64_t schedule_rng_seed = 0;
  /**@}*/
};

///
/// \enum LoggingMode
/// Specifies how and when logging should be sampled and stringified at
/// runtime.
/// * **AsyncPoll**
///  + Logs are serialized and output on an IOThread that polls for new logs at
///  a fixed interval. This is the only mode currently implemented.
/// * **EndOfTestOnly**
///  + TODO: Logs are serialzied and output only at the end of the test.
/// * **Synchronous**
///  + TODO: Logs are serialized and output inline.
enum class LoggingMode {
  AsyncPoll,
  EndOfTestOnly,
  Synchronous,
};

///
/// \brief Specifies where log outputs should go.
///
/// By default, the loadgen outputs its log files to outdir and
/// modifies the filenames of its logs with a prefix and suffix.
/// Filenames will take the form:
/// "<outdir>/<datetime><prefix>summary<suffix>.txt"
///
/// Affordances for outputing logs to stdout are also provided.
///
struct LogOutputSettings {
  std::string outdir = ".";
  std::string prefix = "mlperf_log_";
  std::string suffix = "";
  bool prefix_with_datetime = false;
  bool copy_detail_to_stdout = false;
  bool copy_summary_to_stdout = false;
};

///
/// \brief Top-level log settings.
///
struct LogSettings {
  LogOutputSettings log_output;
  LoggingMode log_mode = LoggingMode::AsyncPoll;
  uint64_t log_mode_async_poll_interval_ms = 1000;  ///< TODO: Implement this.
  bool enable_trace = true;                         ///< TODO: Implement this.
};

/// @}

/// @}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_H
