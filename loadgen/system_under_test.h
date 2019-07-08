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

#ifndef MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H
#define MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H

#include <string>
#include <vector>

#include "query_sample.h"

namespace mlperf {

// SystemUnderTest provides the interface to:
//  1) Allocate, preprocess, and issue queries.
//  2) Warm up the system.
class SystemUnderTest {
 public:
  virtual ~SystemUnderTest() {}

  // A human-readable string for loggin purposes.
  virtual const std::string& Name() const = 0;

  // Issues a N samples to the SUT.
  // The SUT may either a) return immediately and signal completion at a later
  // time on another thread or b) it may block and signal completion on the
  // current stack. The load generator will handle both cases properly.
  // Note: The data for neighboring samples are not contiguous.
  virtual void IssueQuery(const std::vector<QuerySample>& samples) = 0;

  // FlushQueries is called immediately after the last call to IssueQuery
  // in a series is made. This doesn't necessarily signify the end of the
  // test since there may be multiple series involved during a test.
  // Clients can use this to flush any staged queries immediately, rather
  // than waiting for some timeout.
  // This is especially useful in the server scenario.
  virtual void FlushQueries() = 0;

  // Reports the raw latency results to the SUT of each sample issued as
  // recorded by the load generator.
  // Units are nanoseconds.
  virtual void ReportLatencyResults(
      const std::vector<QuerySampleLatency>& latencies_ns) = 0;
};

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H
