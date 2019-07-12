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

// This file provides ways for the client to change the behavior and
// constraints of the load generator.
//
// Note: The MLPerf specification takes precedent over any of the comments in
// this file if there are inconsistencies in regards to how the loadgen
// *should* work.
// The comments in this file are indicative of the loadgen implementation.

#ifndef MLPERF_LOADGEN_TEST_SETTINGS_H
#define MLPERF_LOADGEN_TEST_SETTINGS_H

#include <cstdint>
#include <string>

namespace mlperf {

enum class TestScenario {
  // SingleStream issues queries containing a single sample. The next query is
  // only issued once the previous one has completed. Internal LoadGen latency
  // between queries is not included in the latency calculations.
  // Final performance result is 90 percentile latency.
  SingleStream,

  // MultiStream ideally issues queries containing N samples each at a uniform
  // rate of 20 Hz. However, the loadgen will skip sending for one interval if
  // the SUT falls behind too much. By default, the loadgen will only allow a
  // single outstanding query at a time.
  // TODO: Some SUTs may benefit from pipelining multiple queries while still
  //       hitting the specified latency thresholds. In those cases, the user
  //       may request to have up to Q outstanding queries instead via
  //       |multi_stream_max_async_queries|. Should this be officially
  //       allowed?
  // Final performance result is PASS if the 90 percentile latency is under
  // a given threshold (model-specific) for a given N.
  MultiStream,

  // MultiStreamFree is not an official MLPerf scenario, but is implemented
  // for evaluation purposes.
  // It is the same as MultiStream, but allows for up to P async queries where
  // N is limited only by the latency target. Instead of attempting to issue
  // queries at a fixed rate, this scenario issues a query as soon as the P'th
  // oldest query completes.
  // Final performance result is PASS if the 90th percentile latency is under
  // a given threashold (model-specific) for a given value of N and P.
  MultiStreamFree,

  // Server sends queries with a single sample. Queries have a random poisson
  // (non-uniform) arrival rate that, when averaged, hits the target QPS.
  // Final performance result is 90 percentile latency.
  Server,

  // Offline sends all the samples to the SUT inside of a single query.
  // Final performance result is QPS.
  Offline,
};

enum class TestMode {
  // Runs accuracy mode followed by performance mode.
  // Overriding settings in ways that are not compatible with the MLPerf
  // rules is not allowed in this mode.
  SubmissionRun,

  // Runs each sample from the QSL through the SUT exactly once.
  // Calculates and logs the results for the quality metric.
  // TODO: Determine the metrics for each model.
  AccuracyOnly,

  // Runs the performance traffic for the given scenario, as described in
  // the comments for TestScenario.
  PerformanceOnly,

  // Determines the maximumum QPS for the Server scenario.
  // Determines the maximum samples per query for the MultiStream scenario.
  // Not applicable for SingleStream or Offline.
  FindPeakPerformance,
};

struct TestSettings {
  TestScenario scenario = TestScenario::SingleStream;
  TestMode mode = TestMode::PerformanceOnly;

  // SingleStream-specific settings.
  uint64_t single_stream_expected_latency_ns = 1000000;

  // MultiStream-specific settings.
  // |multi_stream_target_qps| is the rate at which "frames" are produced.
  // |multi_stream_target_latency_ns| is the latency constraint.
  // |multi_stream_samples_per_query| is only used as a hint in
  // SearchForPeakPerformance mode.
  // multi_stream_max_async_queries
  double multi_stream_target_qps = 10.0;
  uint64_t multi_stream_target_latency_ns = 100000000;
  int multi_stream_samples_per_query = 4;
  int multi_stream_max_async_queries = 1;

  // Server-specific settings.
  // |server_target_qps| is only used as a hint in SearchForPeakPerformance
  // mode.
  // |server_target_latency_ns| is the latency constraint.
  double server_target_qps = 1;
  uint64_t server_target_latency_ns = 100000000;
  bool server_coalesce_queries = false;  // TODO: Use this.

  // Offline-specific settings.
  // Used to specify the qps the SUT expects to hit for the offline load.
  // In the offline scenario, all queries will be coalesced into a single
  // query; in this sense, "samples per second" is equivalent to "queries per
  // second." We go with QPS for consistency.
  double offline_expected_qps = 1;

  // The test runs until both min duration and min query count have been met.
  // However, it will exit before that point if either max duration or
  // max query count have been reached.
  uint64_t min_duration_ms = 10000;
  uint64_t max_duration_ms = 0;  // 0: Infinity.
  uint64_t min_query_count = 100;
  uint64_t max_query_count = 0;  // 0: Infinity.

  // Random number generation seeds.
  // There are 3 separate seeds, so each dimension can be changed independently.

  // |qsl_rng_seed| affects which subset of samples in the QSL
  // are chosen for the performance set, as well as the order in which samples
  // are processed in AccuracyOnly mode.
  uint64_t qsl_rng_seed = 0;

  // |sample_index_rng_seed| affects the order in which samples
  // from the performance set will be included in queries.
  uint64_t sample_index_rng_seed = 0;

  // |schedule_rng_seed| affects the poisson arrival process of
  // the Server scenario. Different seeds will appear to "jitter" the queries
  // differently in time, but should not affect the average issued QPS.
  uint64_t schedule_rng_seed = 0;
};

enum class LoggingMode {
  AsyncPoll,      // Logs are serialized and output on an IOThread that polls
                  // for new logs at a fixed interval.
  EndOfTestOnly,  // TODO: Logs are serialzied and output only at the end of
                  // the test.
  Synchronous,    // TODO: Logs are serialized and output inline.
};

struct LogOutputSettings {
  // By default, the loadgen outputs its log files to outdir and
  // modifies the filenames of its logs with a prefix and suffix.
  // Filenames will take the form:
  // "<outdir>/<datetime><prefix>summary<suffix>.txt"
  std::string outdir = ".";
  std::string prefix = "mlperf_log_";
  std::string suffix = "";
  bool prefix_with_datetime = false;
  bool copy_detail_to_stdout = false;
  bool copy_summary_to_stdout = false;
};

struct LogSettings {
  LogOutputSettings log_output;
  LoggingMode log_mode = LoggingMode::AsyncPoll;
  uint64_t log_mode_async_poll_interval_ms = 1000;  // TODO.
  bool enable_trace = true;  // TODO: Allow trace to be disabled.
};

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_H
