#ifndef SYSTEM_UNDER_TEST_H
#define SYSTEM_UNDER_TEST_H

#include <stdint.h>
#include <memory>

namespace mlperf {

struct QuerySample;
struct TestSettings;

// SystemUnderTest provides the interface to:
//  1) Allocate, preprocess, and issue queries.
//  2) Warm up the system.
class SystemUnderTest {
 public:
  virtual ~SystemUnderTest() {}

  // A human-readable string for loggin purposes.
  virtual std::string Name() = 0;

  // UntimedWarmUp will be called by the test harness before it starts
  // issuing any queries. The SUT may block in this method and perform any
  // warm up activities it wants, such as feeding all zeros to the model's
  // inputs.
  // The query sample library isn't loaded until after UntimedWarmUp returns.
  virtual void UntimedWarmUp() = 0;

  // Allocates memory for samples passed to IssueQuery().
  // This allows the system under test to do things like provide pre-pinned
  // accelerator-visiable memory or shared cross-process memory for queries
  // to be written to.
  virtual void* AllocateQuerySample(size_t size_in_bytes) = 0;

  // Frees memory allocated by AllocateQuerySample.
  virtual void FreeQuerySample(void* mem) = 0;

  // PreprocessQuerySample() enables the SUT to crop/transpose/pad a sample
  // before it is issued. See MLPerf inference rules regarding what kinds of
  // preprocessing is allowed. The preprocessing is not timed.
  // Implementations must allocate the necessary memory for
  // |processed_data|. Then read from |source_data|, processes it into
  // |processed_data|, and return the size of |processed_data| in
  // processed_size|.
  // Caller takes ownership of |processed_data| which will be freed by
  // a call to FreeQuerySample().
  // Caller maintains ownership of |source_data| as well.
  virtual void PreprocessQuerySample(const void* source_data,
                                     const size_t source_size,
                                     void** processed_data,
                                     size_t* processed_size) = 0;

  // Issues a N samples to the SUT.
  // The SUT may either a) return immediately and signal completion at a later
  // time on another thread or b) it may block and signal completion on the
  // current stack. The test harness will handle both cases properly.
  // |query_id| is a unique id associated with this instance of the query.
  // The SUT must reference the associated query_id when triggering
  // QueryComplete().
  // Note: The data for neighboring samples are not contiguous.
  virtual void IssueQuery(intptr_t query_id, QuerySample* samples,
                          size_t sample_count) = 0;
};

// Starts the test against |sut| with the specified |settings|.
// This is the C++ entry point. See mlperf::c::StartTest for the C entry point.
void StartTest(SystemUnderTest* sut, const TestSettings& settings);

}  // namespace mlperf

#endif  // SYSTEM_UNDER_TEST_H
