#ifndef MLPERF_LOADGEN_TEST_SETTINGS_H
#define MLPERF_LOADGEN_TEST_SETTINGS_H

// Note: The MLPerf specification takes precedent over any of the comments in
// this file if there are inconsistencies in regards to how the loadgen
// *should* work.
// The comments in this file are indicative of the loadgen implementation.

#include <cstdint>

namespace mlperf {

enum TestScenario {
  // SingleStream issues queries containing a single sample. The next query is
  // only issued once the previous one has completed. Internal LoadGen latency
  // between queries is not included in the latency calculations.
  // Final performance result is 90 percentile latency.
  SingleStream,

  // MultiStream ideally issues queries containing N samples each at a uniform
  // rate of 60 Hz. However, the loadgen will skip sending for one interval if
  // the SUT falls behind too much. By default and spec, the loadgen will only
  // allow 1 outstanding query at a time.
  // TODO: Some SUTs may benefit from pipelining multiple queries while still
  //       hitting the specified latency thresholds. In those cases, the user
  //       may request to have up to Q outstanding queries instead via
  //       |override_multi_stream_max_async_queries|. Should this be officially
  //       allowed?
  // Final performance result is PASS if the 90 percentile latency is under
  // a given threshold (model-specific) for a given N.
  MultiStream,

  // Server sends queries with a single sample. Queries have a random poisson
  // (non-uniform) arrival rate that, when averaged, hits the target QPS.
  // Final performance result is 90 percentile latency.
  Server,

  // Offline sends all the samples to the SUT inside of a single query.
  // Final performance result is QPS.
  Offline,
};

enum TestMode {
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

// TODO: Logging settings. e.g.: sync vs. async; log frequency;
struct TestSettings {
  TestScenario scenario = TestScenario::SingleStream;
  TestMode mode = TestMode::PerformanceOnly;

  // SingleStream-specific settings.
  uint64_t single_stream_expected_latency_ns = 100000;

  // MultiStream-specific settings.
  // |multi_stream_samples_per_query| is only used as a hint in
  // SearchForPeakPerformance mode.
  int multi_stream_samples_per_query = 4;

  // Server-specific settings.
  // |server_target_qps| is only used as a hint in SearchForPeakPerformance
  // mode.
  double server_target_qps = 10;
  bool server_coallesce_queries = false;  // TODO: Use this.

  // Offline-specific settings.
  double offline_expected_qps = 10;

  // |enable_spec_overrides| is useful for experimentation and
  // for shortening testing feedback loops.
  // Must be false if mode is SubmissionRun.
  bool enable_spec_overrides = false;

  // Settings after this point only have an effect if
  // |enable_spec_overrides| is true.
  uint64_t override_target_latency_ns = 0;  // 0: Use spec default.
  // TODO: Should this be an official setting?
  int override_multi_stream_max_async_queries = 0;  // 0: Use spec default.

  // Test runs until both min duration and query count have been met, but
  // will exit before that point if either max duration or query count have
  // been reached.
  uint64_t override_min_duration_ms = 0;  // 0: Use spec defaults.
  uint64_t override_max_duration_ms = 0;  // 0: Infinity. TODO: Use this.
  uint64_t override_min_query_count = 0;  // 0: Use spec defaults.
  uint64_t override_max_query_count = 0;  // 0: Infinity. TODO: Use this.

  // Random number generation seeds. Values of 0 disable overrides.
  // There are 3 separate seeds, so each dimension can be changed independently.

  // |override_qsl_rng_seed| affects which subset of samples in the QSL
  // are chosen for the performance set, as well as the order in which samples
  // are processed in AccuracyOnly mode.
  uint64_t override_qsl_rng_seed = 0;

  // |override_sample_index_rng_seed| affects the order in which samples
  // from the performance set will be included in queries.
  uint64_t override_sample_index_rng_seed = 0;

  // |override_schedule_rng_seed| affects the poisson arrival process of
  // the Server scenario. Different seeds will appear to "jitter" the queries
  // differently in time, but should not affect the average issued QPS.
  uint64_t override_schedule_rng_seed = 0;  // 0: Use spec default.
};

}  // namespace mlperf

#endif // MLPERF_LOADGEN_TEST_SETTINGS_H
