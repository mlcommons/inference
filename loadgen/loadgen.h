#ifndef TEST_HARNESS_H_
#define TEST_HARNESS_H_

#include <stddef.h>
#include <stdint.h>

#include "query_sample.h"

namespace mlperf {

struct QuerySampleResponse;
class QuerySampleLibrary;
class SystemUnderTest;

enum TestScenario {
  StreamOneAtATime,
  StreamNAtFixedRate,
  Cloud,
  Offline,
};

enum TestMode {
  SubmissionRun,
  AccuracyOnly,
  PerformanceOnly,
  SearchForQps,
};

struct TestSettings {
  TestScenario scenario = TestScenario::StreamNAtFixedRate;
  TestMode mode = TestMode::AccuracyOnly;
  int samples_per_query = 4;
  double target_qps = 100;
};

// Defined in parse_command_line.cc
TestSettings ParseCommandLine(const char* command);        // For Python.
TestSettings ParseCommandLineArgs(int argc, char** argv);  // For C.

// QueryComplete must be called by the SUT once it completes a query issued
// by SystemUnderTest::IssueQuery().
// |query_id| corresponds to the id provided to SystemUnderTest::IssueQuery.
// TODO(brianderson): This API assumes the response will be allocated and
// owend by the SUT. This necessarily requires the allocation to be timed,
// which will benefit SUTs that efficiently recycle response memory.
void QueryComplete(QueryId query_id, QuerySampleResponse* responses,
                   size_t response_count);

// Starts the test against |sut| with the specified |settings|.
// This is the C++ entry point. See mlperf::c::StartTest for the C entry point.
void StartTest(SystemUnderTest* sut,
               QuerySampleLibrary* qsl,
               const TestSettings& settings);

}  // namespace mlperf

#endif  // TEST_HARNESS_H_
