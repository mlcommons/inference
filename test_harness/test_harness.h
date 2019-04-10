#ifndef TEST_HARNESS_H_
#define TEST_HARNESS_H_

#include <stddef.h>
#include <stdint.h>

namespace mlperf {

struct QuerySampleResponse;

enum TestScenario {
  All, Cloud, Edge, Offline,
};

enum TestMode {
  SubmissionRun, AccuracyOnly, PerformanceOnly, SearchForQps,
};

struct TestSettings {
  char* query_sample_library_name;
  size_t query_sample_library_name_length;
  TestScenario scenario;
  TestMode mode;
  int samples_per_query;
  double target_qps;
};

// Defined in parse_command_line.cc
TestSettings ParseCommandLine(const char* command);  // For Python.
TestSettings ParseCommandLineArgs(int argc, char** argv);  // For C.

// QueryComplete must be called by the SUT once it completes a query issued
// by SystemUnderTest::IssueQuery().
// |query_id| corresponds to the id provided to SystemUnderTest::IssueQuery.
// TODO(brianderson): This API assumes the response will be allocated and
// owend by the SUT. This necessarily requires the allocation to be timed,
// which will benefit SUTs that efficiently recycle response memory.
// The recycling logic might be able to live in the test harness logic though,
// which would pull the response allocation out of the critical path for all
// SUTs.
void QueryComplete(intptr_t query_id, QuerySampleResponse* responses,
                   size_t response_count);

// Note: StartTest() would normally be declared here, but which version of
// StartTest() to use depends on how the SystemUnderTest is created.
// Therefore StartTest() is declared in the system_under_test.h and
// system_under_test_c_api.h.

}  // namespace mlperf

#endif  // TEST_HARNESS_H_
