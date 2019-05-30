#ifndef MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
#define MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H

#include <chrono>
#include <string>

#include "test_settings.h"

namespace mlperf {

class AsyncLog;

std::string ToString(TestScenario scenario);
std::string ToString(TestMode mode);

// TestSettingsInternal takes the mode and scenario requested by the
// user-provided TestSettings and chooses the proper test parameters based
// on the spec-defined defaults and any user-requesed overrides.
struct TestSettingsInternal {
  explicit TestSettingsInternal(const TestSettings& requested_settings);
  void LogEffectiveSettings() const;
  void LogAllSettings() const;
  void LogSummary(AsyncLog& log) const;

  const TestSettings requested;
  const TestScenario scenario;  // Copied here for convenience.
  const TestMode mode;          // Copied here for convenience.

  int samples_per_query;
  double target_qps;
  std::chrono::nanoseconds target_latency;
  int max_async_queries;

  // Taget duration is used to generate queries of a minimum duration before
  // the test run.
  std::chrono::milliseconds target_duration;

  // Min duration/query_count/sample_count are used to validate the test
  // duration at the end of the run.
  std::chrono::milliseconds min_duration;
  std::chrono::milliseconds max_duration;
  uint64_t min_query_count;
  uint64_t max_query_count;
  uint64_t min_sample_count;  // Offline only.

  uint64_t qsl_rng_seed;
  uint64_t sample_index_rng_seed;
  uint64_t schedule_rng_seed;

 private:
  void ApplyOverrides();
};

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
