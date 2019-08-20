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

#include "logging.h"
#include "math.h"

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
  double target_latency_percentile; // Single, multistream and server mode
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

namespace find_peak_performance {

template <TestScenario scenario>
TestSettingsInternal MidOfBoundaries(
    const TestSettingsInternal &lower_bound_settings,
    const TestSettingsInternal &upper_bound_settings) {
  TestSettingsInternal mid_settings = lower_bound_settings;
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    assert(lower_bound_settings.samples_per_query <
           upper_bound_settings.samples_per_query);
    mid_settings.samples_per_query = lower_bound_settings.samples_per_query +
                                     (upper_bound_settings.samples_per_query -
                                      lower_bound_settings.samples_per_query) /
                                         2;
  } else if (scenario == TestScenario::Server) {
    assert(lower_bound_settings.target_qps < upper_bound_settings.target_qps);
    mid_settings.target_qps =
        lower_bound_settings.target_qps +
        (upper_bound_settings.target_qps - lower_bound_settings.target_qps) / 2;
  } else {
    LogDetail([](AsyncDetail &detail) {
      detail(
          "other types of scenarios (SingleStream & Offline) does not support "
          "the method.");
    });
  }
  return mid_settings;
}

template <TestScenario scenario>
bool IsFinished(const TestSettingsInternal &lower_bound_settings,
                const TestSettingsInternal &upper_bound_settings) {
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    return lower_bound_settings.samples_per_query + 1 >=
           upper_bound_settings.samples_per_query;
  } else if (scenario == TestScenario::Server) {
    uint8_t precision =
        lower_bound_settings.requested.server_target_qps_precision;
    double l = floor(lower_bound_settings.target_qps * pow(10, precision));
    double u = floor(upper_bound_settings.target_qps * pow(10, precision));
    return l + 1 >= u;
  } else {
    LogDetail([](AsyncDetail &detail) {
      detail(
          "other types of scenarios (SingleStream & Offline) does not support "
          "the method.");
    });
    return true;
  }
}

template <TestScenario scenario>
std::string ToStringPerformanceField(const TestSettingsInternal &settings) {
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    return std::to_string(settings.samples_per_query);
  } else if (scenario == TestScenario::Server) {
    return std::to_string(settings.target_qps);
  } else {
    // Unreachable
    assert(false);
    return ToString(settings.scenario);
  }
}

template <TestScenario scenario>
void WidenPerformanceField(TestSettingsInternal &settings) {
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    settings.samples_per_query = settings.samples_per_query * 2;
  } else if (scenario == TestScenario::Server) {
    settings.target_qps = settings.target_qps * 2;
  } else {
    LogDetail([](AsyncDetail &detail) {
      detail(
          "other types of scenarios (SingleStream & Offline) does not support "
          "the method.");
    });
  }
}

}  // namespace find_peak_performance
}  // namespace loadgen
}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
