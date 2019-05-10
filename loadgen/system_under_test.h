#ifndef MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H
#define MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H

#include <stdint.h>

#include <memory>

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

  // Reports the raw latency results to the SUT of each sample issued as
  // recorded by the load generator.
  // Units are nanoseconds.
  virtual void ReportLatencyResults(
      const std::vector<QuerySampleLatency>& latencies_ns) = 0;
};

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H
