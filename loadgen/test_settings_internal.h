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
/// \brief The internal representation of user-provided settings.

#ifndef MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
#define MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H

#include <chrono>
#include <string>

#include "test_settings.h"

namespace mlperf {

namespace logging {
class AsyncSummary;
}

namespace loadgen {

using AsyncSummary = logging::AsyncSummary;

std::string ToString(TestScenario scenario);
std::string ToString(TestMode mode);

/// \brief takes the user-friendly TestSettings and normalizes it
/// for consumption by the loadgen.
/// \details It does things like remove scenario-specific naming and introduce
/// the concept of target_duration used to pre-generate queries.
struct TestSettingsInternal {
  explicit TestSettingsInternal(const TestSettings& requested_settings);
  void LogEffectiveSettings() const;
  void LogAllSettings() const;
  void LogSummary(AsyncSummary& summary) const;

  const TestSettings requested;
  const TestScenario scenario;  // Copied here for convenience.
  const TestMode mode;          // Copied here for convenience.

  int samples_per_query;
  double target_qps;
  std::chrono::nanoseconds target_latency{0};
  int max_async_queries;

  // Target duration is used to generate queries of a minimum duration before
  // the test run.
  std::chrono::milliseconds target_duration{0};

  // Min duration/query_count/sample_count are used to validate the test
  // duration at the end of the run.
  std::chrono::milliseconds min_duration{0};
  std::chrono::milliseconds max_duration{0};
  uint64_t min_query_count;
  uint64_t max_query_count;
  uint64_t min_sample_count;  // Offline only.

  uint64_t qsl_rng_seed;
  uint64_t sample_index_rng_seed;
  uint64_t schedule_rng_seed;
};

}  // namespace loadgen
}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
