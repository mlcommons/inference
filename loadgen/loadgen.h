#ifndef MLPERF_LOADGEN_LOADGEN_H_
#define MLPERF_LOADGEN_LOADGEN_H_

#include <cstddef>

namespace mlperf {

struct QuerySampleResponse;
class QuerySampleLibrary;
class SystemUnderTest;
struct TestSettings;
struct LogSettings;

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
void StartTest(SystemUnderTest* sut, QuerySampleLibrary* qsl,
               const TestSettings& requested_settings,
               const LogSettings& log_settings);

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_LOADGEN_H_
