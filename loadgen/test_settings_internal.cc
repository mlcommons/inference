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

#include "test_settings_internal.h"

#include "logging.h"
#include "utils.h"

namespace mlperf {

TestSettingsInternal::TestSettingsInternal(
    const TestSettings &requested_settings)
    : requested(requested_settings),
      scenario(requested.scenario),
      mode(requested.mode),
      samples_per_query(1),
      target_qps(1),
      max_async_queries(-1),
      target_duration(std::chrono::milliseconds(requested.min_duration_ms)),
      min_duration(std::chrono::milliseconds(requested.min_duration_ms)),
      max_duration(std::chrono::milliseconds(requested.max_duration_ms)),
      min_query_count(requested.min_query_count),
      max_query_count(requested.max_query_count),
      min_sample_count(0),
      qsl_rng_seed(requested.qsl_rng_seed),
      sample_index_rng_seed(requested.sample_index_rng_seed),
      schedule_rng_seed(requested.schedule_rng_seed) {
  // Target QPS, target latency, and max_async_queries.
  switch (requested.scenario) {
    case TestScenario::SingleStream:
      target_qps = static_cast<double>(std::nano::den) /
                   requested.single_stream_expected_latency_ns;
      max_async_queries = 1;
      break;
    case TestScenario::MultiStream:
    case TestScenario::MultiStreamFree:
      target_qps = requested.multi_stream_target_qps;
      target_latency =
          std::chrono::nanoseconds(requested.multi_stream_target_latency_ns);
      max_async_queries = requested.multi_stream_max_async_queries;
      break;
    case TestScenario::Server:
      if (requested.server_target_qps >= 0.0) {
        target_qps = requested.server_target_qps;
      } else {
        LogError([server_target_qps = requested.server_target_qps,
                  target_qps = target_qps](AsyncLog &log) {
          log.LogDetail("Invalid value for server_target_qps requested.",
                        "requested", server_target_qps, "using", target_qps);
        });
      }
      target_latency =
          std::chrono::nanoseconds(requested.server_target_latency_ns);
      max_async_queries =
          std::numeric_limits<decltype(max_async_queries)>::max();
      break;
    case TestScenario::Offline:
      if (requested.offline_expected_qps >= 0.0) {
        target_qps = requested.offline_expected_qps;
      } else {
        LogError([offline_expected_qps = requested.offline_expected_qps,
                  target_qps = target_qps](AsyncLog &log) {
          log.LogDetail("Invalid value for offline_expected_qps requested.",
                        "requested", offline_expected_qps, "using", target_qps);
        });
      }
      max_async_queries =
          std::numeric_limits<decltype(max_async_queries)>::max();
      break;
  }

  // Samples per query.
  if (requested.scenario == TestScenario::MultiStream ||
      requested.scenario == TestScenario::MultiStreamFree) {
    samples_per_query = requested.multi_stream_samples_per_query;
  }

  // In the offline scenario, coalesce all queries into a single query.
  if (requested.scenario == TestScenario::Offline) {
    // TODO: Should the spec require a max duration for large query counts?
    // kSlack is used to make sure we generate enough samples for the SUT
    // to take longer than than the minimum test duration required by the
    // MLPerf spec.
    constexpr double kSlack = 1.1;
    int target_sample_count =
        kSlack * DurationToSeconds(target_duration) * target_qps;
    samples_per_query = std::max<int>(min_query_count, target_sample_count);
    min_query_count = 1;
    target_duration = std::chrono::milliseconds(0);
  }

  min_sample_count = min_query_count * samples_per_query;
}

std::string ToString(TestScenario scenario) {
  switch (scenario) {
    case TestScenario::SingleStream:
      return "Single Stream";
    case TestScenario::MultiStream:
      return "Multi Stream";
    case TestScenario::MultiStreamFree:
      return "Multi Stream Free";
    case TestScenario::Server:
      return "Server";
    case TestScenario::Offline:
      return "Offline";
  }
  assert(false);
  return "InvalidScenario";
}

std::string ToString(TestMode mode) {
  switch (mode) {
    case TestMode::SubmissionRun:
      return "Submission";
    case TestMode::AccuracyOnly:
      return "Accuracy";
    case TestMode::PerformanceOnly:
      return "Performance";
    case TestMode::FindPeakPerformance:
      return "Find Peak Performance";
  }
  assert(false);
  return "InvalidMode";
}

void LogRequestedTestSettings(const TestSettings &s) {
  LogDetail([s](AsyncLog &log) {
    log.LogDetail("");
    log.LogDetail("Requested Settings:");
    log.LogDetail("Scenario : " + ToString(s.scenario));
    log.LogDetail("Test mode : " + ToString(s.mode));

    // Scenario-specific
    switch (s.scenario) {
      case TestScenario::SingleStream:
        log.LogDetail("single_stream_expected_latency_ns : ",
                      s.single_stream_expected_latency_ns);
        break;
      case TestScenario::MultiStream:
      case TestScenario::MultiStreamFree:
        log.LogDetail("multi_stream_target_qps : ", s.multi_stream_target_qps);
        log.LogDetail("multi_stream_target_latency_ns : ",
                      s.multi_stream_target_latency_ns);
        log.LogDetail("multi_stream_samples_per_query : ",
                      s.multi_stream_samples_per_query);
        log.LogDetail("multi_stream_max_async_queries : ",
                      s.multi_stream_max_async_queries);
        break;
      case TestScenario::Server:
        log.LogDetail("server_target_qps : ", s.server_target_qps);
        log.LogDetail("server_target_latency_ns : ",
                      s.server_target_latency_ns);
        log.LogDetail("server_coalesce_queries : ", s.server_coalesce_queries);
        break;
      case TestScenario::Offline:
        log.LogDetail("offline_expected_qps : ", s.offline_expected_qps);
        break;
    }

    // Overrides
    log.LogDetail("min_duration_ms : ", s.min_duration_ms);
    log.LogDetail("max_duration_ms : ", s.max_duration_ms);
    log.LogDetail("min_query_count : ", s.min_query_count);
    log.LogDetail("max_query_count : ", s.max_query_count);
    log.LogDetail("qsl_rng_seed : ", s.qsl_rng_seed);
    log.LogDetail("sample_index_rng_seed : ", s.sample_index_rng_seed);
    log.LogDetail("schedule_rng_seed : ", s.schedule_rng_seed);

    log.LogDetail("");
  });
}

void TestSettingsInternal::LogEffectiveSettings() const {
  LogDetail([s = *this](AsyncLog &log) {
    log.LogDetail("");
    log.LogDetail("Effective Settings:");

    log.LogDetail("Scenario : " + ToString(s.scenario));
    log.LogDetail("Test mode : " + ToString(s.mode));

    log.LogDetail("samples_per_query : ", s.samples_per_query);
    log.LogDetail("target_qps : ", s.target_qps);
    log.LogDetail("target_latency (ns): ", s.target_latency.count());
    log.LogDetail("max_async_queries : ", s.max_async_queries);
    log.LogDetail("target_duration (ms): ", s.target_duration.count());
    log.LogDetail("min_duration (ms): ", s.min_duration.count());
    log.LogDetail("max_duration (ms): ", s.max_duration.count());
    log.LogDetail("min_query_count : ", s.min_query_count);
    log.LogDetail("max_query_count : ", s.max_query_count);
    log.LogDetail("min_sample_count : ", s.min_sample_count);
    log.LogDetail("qsl_rng_seed : ", s.qsl_rng_seed);
    log.LogDetail("sample_index_rng_seed : ", s.sample_index_rng_seed);
    log.LogDetail("schedule_rng_seed : ", s.schedule_rng_seed);
  });
}

void TestSettingsInternal::LogAllSettings() const {
  LogEffectiveSettings();
  LogRequestedTestSettings(requested);
}

void TestSettingsInternal::LogSummary(AsyncLog &log) const {
  log.LogSummary("samples_per_query : ", samples_per_query);
  log.LogSummary("target_qps : ", target_qps);
  log.LogSummary("target_latency (ns): ", target_latency.count());
  log.LogSummary("max_async_queries : ", max_async_queries);
  log.LogSummary("min_duration (ms): ", min_duration.count());
  log.LogSummary("max_duration (ms): ", max_duration.count());
  log.LogSummary("min_query_count : ", min_query_count);
  log.LogSummary("max_query_count : ", max_query_count);
  log.LogSummary("qsl_rng_seed : ", qsl_rng_seed);
  log.LogSummary("sample_index_rng_seed : ", sample_index_rng_seed);
  log.LogSummary("schedule_rng_seed : ", schedule_rng_seed);
}

}  // namespace mlperf
