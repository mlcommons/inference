#ifndef MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
#define MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H

#include <chrono>

#include "test_settings.h"

namespace mlperf {

// TestSettingsInternal takes the mode and scenario requested by the
// user-provided TestSettings and chooses the proper test parameters based
// on the spec-defined defaults and any user-requesed overrides.
struct TestSettingsInternal {
  explicit TestSettingsInternal(const TestSettings &requested_settings);
  void LogSettings();

  const TestSettings requested;
  const TestScenario scenario;  // Copied here for convenience.
  const TestMode mode;          // Copied here for convenience.

  int samples_per_query;
  double target_qps;
  std::chrono::nanoseconds target_latency;
  int max_async_queries;

  std::chrono::milliseconds min_duration;
  std::chrono::milliseconds max_duration;
  uint64_t min_query_count;
  uint64_t max_query_count;

  uint64_t qsl_rng_seed;
  uint64_t sample_index_rng_seed;
  uint64_t schedule_rng_seed;
};

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
