#ifndef MLPERF_LOADGEN_TEST_SETTINGS_H
#define MLPERF_LOADGEN_TEST_SETTINGS_H

#include <cstdint>

namespace mlperf {

enum TestScenario {
  SingleStream,
  MultiStream,
  Server,
  Offline,
};

enum TestMode {
  SubmissionRun,
  AccuracyOnly,
  PerformanceOnly,
  SearchForQps,
};

// TODO: Logging settings. e.g.: sync vs. async; log frequency;
struct TestSettings {
  TestScenario scenario = TestScenario::MultiStream;
  TestMode mode = TestMode::AccuracyOnly;
  int samples_per_query = 4;
  double target_qps = 100;
  uint64_t target_latency_ns = 1000000000;
};

}  // namespace mlperf

#endif // MLPERF_LOADGEN_TEST_SETTINGS_H
