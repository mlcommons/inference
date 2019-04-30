#ifndef MLPERF_LOADGEN_H_
#define MLPERF_LOADGEN_H_

#include <stddef.h>
#include <stdint.h>

#include "query_sample.h"

namespace mlperf {

struct QuerySampleResponse;
class QuerySampleLibrary;
class SystemUnderTest;

enum TestScenario {
  SingleStream,
  MultiStream,
  Cloud,
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

// QuerySamplesComplete must be called by the SUT once it completes samples of
// a query issued by SystemUnderTest::IssueQuery().
// The samples may be from any combination of queries or partial queries.
// TODO(brianderson): This API assumes the response will be allocated and
// owend by the SUT. This necessarily requires the allocation to be timed,
// which will benefit SUTs that efficiently recycle response memory.
void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count);

// Starts the test against |sut| with the specified |settings|.
// This is the C++ entry point. See mlperf::c::StartTest for the C entry point.
void StartTest(SystemUnderTest* sut,
               QuerySampleLibrary* qsl,
               const TestSettings& settings);

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_H_
