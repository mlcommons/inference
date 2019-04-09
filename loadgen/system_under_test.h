#ifndef SYSTEM_UNDER_TEST_H
#define SYSTEM_UNDER_TEST_H

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
  // current stack. The laod generator will handle both cases properly.
  // |query_id| is a unique id associated with this instance of the query.
  // The SUT must reference the associated query_id when triggering
  // QueryComplete().
  // Note: The data for neighboring samples are not contiguous.
  virtual void IssueQuery(QueryId query_id, QuerySample* samples,
                          size_t sample_count) = 0;
};

}  // namespace mlperf

#endif  // SYSTEM_UNDER_TEST_H
