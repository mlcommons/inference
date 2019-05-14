#ifndef MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
#define MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H

#include "logging.h"
#include "test_settings.h"

namespace mlperf {

constexpr uint64_t kDefaultQslSeed = 0xABCD1234;
constexpr uint64_t kDefaultSampleSeed = 0x1234ABCD;
constexpr uint64_t kDefaultScheduleSeed = 0xA1B2C3D4;

// TestSettingsInternal takes the mode and scenario requested by the
// user-provided TestSettings and chooses the proper test parameters based
// on the spec-defined defaults and any user-requesed overrides.
struct TestSettingsInternal {
  const TestSettings requested;

  const TestScenario scenario;  // Copied here for convenience.
  const TestMode mode;          // Copied here for convenience.

  int samples_per_query = 1;
  double target_qps = 60.0;
  std::chrono::nanoseconds target_latency{1000000000};  // TODO: Spec.
  int max_async_queries = 1;

  std::chrono::milliseconds min_duration{60000};
  std::chrono::milliseconds max_duration{0};
  uint64_t min_query_count;
  uint64_t max_query_count = std::numeric_limits<uint64_t>::max();

  uint64_t qsl_rng_seed = kDefaultQslSeed;
  uint64_t sample_index_rng_seed = kDefaultSampleSeed;
  uint64_t schedule_rng_seed = kDefaultScheduleSeed;

  explicit TestSettingsInternal(const TestSettings &requested_settings)
      : requested(requested_settings),
        scenario(requested.scenario),
        mode(requested.mode),
        min_query_count(
            requested.scenario == TestScenario::SingleStream ? 1024 : 24576) {
    // Target QPS.
    switch (requested.scenario) {
      case TestScenario::SingleStream:
        target_qps = 1000000000.0 / requested.single_stream_expected_latency_ns;
        break;
      case TestScenario::MultiStream:
        target_qps = 60.0;
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
        break;
      case TestScenario::Offline:
        if (requested.offline_expected_qps >= 0.0) {
          target_qps = requested.offline_expected_qps;
        } else {
          LogError([offline_expected_qps = requested.offline_expected_qps,
                    target_qps = target_qps](AsyncLog &log) {
            log.LogDetail("Invalid value for offline_expected_qps requested.",
                          "requested", offline_expected_qps, "using",
                          target_qps);
          });
        }
        break;
    }

    // Samples per query.
    if (requested.scenario == TestScenario::MultiStream) {
      samples_per_query = requested.multi_stream_samples_per_query;
    }

    // In the offline scenario, coalesce all queries into a single query.
    if (requested.scenario == TestScenario::Offline) {
      // TODO: Should the spec require a max duration for large query counts?
      // kSlack is used to make sure we generate enough samples for the SUT
      // to take longer than than the minimum test duration required by the
      // MLPerf spec.
      constexpr double kSlack = 1.1;
      samples_per_query =
          std::max<int>(min_query_count, (60.0 / target_qps) * kSlack);
      min_query_count = 1;
      min_duration = std::chrono::milliseconds(0);
    }

    // Exit here if we are using defaults.
    if (!requested.enable_spec_overrides) {
      return;
    }

    // Do not allow overrides for a submission run.
    if (requested.mode == TestMode::SubmissionRun) {
      LogError([](AsyncLog &log) {
        log.LogDetail(
            "Overriding defaults for a SubmissionRun not allowed. \
                       All overrides ignored.");
      });
      return;
    }

    if (requested.override_target_latency_ns != 0) {
      target_latency =
          std::chrono::nanoseconds(requested.override_target_latency_ns);
    }

    if (requested.override_multi_stream_max_async_queries != 0) {
      if (requested.scenario == TestScenario::MultiStream) {
        max_async_queries = requested.override_multi_stream_max_async_queries;
      } else {
        LogError([](AsyncLog &log) {
          log.LogDetail(
              "Overriding max async queries outside of the \
                         MultiStream scenario has no effect.");
        });
      }
    }

    // Test duration.
    if (requested.override_min_duration_ms != 0) {
      min_duration =
          std::chrono::milliseconds(requested.override_min_duration_ms);
    }
    if (requested.override_max_duration_ms != 0) {
      max_duration =
          std::chrono::milliseconds(requested.override_max_duration_ms);
    }
    if (requested.override_min_query_count != 0) {
      min_query_count = requested.override_min_query_count;
    }
    if (requested.override_max_query_count != 0) {
      max_query_count = requested.override_max_query_count;
    }

    // Random number generation.
    if (requested.override_qsl_rng_seed != 0) {
      qsl_rng_seed = requested.override_qsl_rng_seed;
    }
    if (requested.override_sample_index_rng_seed != 0) {
      sample_index_rng_seed = requested.override_sample_index_rng_seed;
    }
    if (requested.override_schedule_rng_seed != 0) {
      schedule_rng_seed = requested.override_schedule_rng_seed;
    }
  }
};

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
