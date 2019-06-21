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
