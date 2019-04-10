#include "test_harness.h"

#include <iostream>
#include <mutex>
#include <stdint.h>

#include "query_allocator.h"
#include "query_residency_manager.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "query_sample_library_registry.h"
#include "system_under_test.h"

namespace mlperf {

// QueryResponseMetadata is used by the TestHarness to coordinate
// response data and completion.
struct QueryResponseMetadata {
  size_t query_index;
  std::vector<uint8_t> data;

  void ResetCompletion() { complete = true; }

  void NotifyCompletion() {
    std::unique_lock<std::mutex> lock(mutex);
    complete = true;
    cv.notify_all();
  }

  void WaitForCompletion() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return complete; });
  }

 private:
  std::mutex mutex;
  std::condition_variable cv;
  bool complete;
};

void QueryComplete(intptr_t query_id, QuerySampleResponse* responses,
                   size_t response_count) {
  QueryResponseMetadata* metadata =
      reinterpret_cast<QueryResponseMetadata*>(query_id);
  // TODO(brianderson): Don't copy data in performance mode.
  auto* src_begin = reinterpret_cast<uint8_t*>(responses[0].data);
  auto* src_end = src_begin + responses[0].size;
  metadata->data = std::vector<uint8_t>(src_begin, src_end);
  std::cout << "Query complete ID: " << query_id << "\n";
  std::cout << "Query library index: " << metadata->query_index << "\n";
  metadata->NotifyCompletion();
}

void RunVerificationMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                         const TestSettings& settings) {
  qsl->Initialize(QueryLibraryMode::Verification);
  SutQueryAllocator query_allocator(sut);

  // TODO(brianderson): Remove this call to UntimedWarmUp.
  // Warm up isn't needed for verification, but is included here for
  // reference during initial development.
  sut->UntimedWarmUp();

  // Don't specify a library allocator since we don't want to pre-allocate
  // memory for the whole library in accuracy verification mode.
  QueryResidencyManager query_manager(qsl, nullptr, &query_allocator);
  size_t library_query_count = query_manager.LibrarySize();

  // Re-use the same response over and over since we only ever have
  // a single response outstanding in verification mode right now.
  QueryResponseMetadata response;

  // Issue every query in the library and calculate the accuracy metric.
  qsl->ResetAccuracyMetric();
  for (size_t i = 0; i < library_query_count; i++) {
    response.ResetCompletion();
    response.query_index = i;
    intptr_t id = reinterpret_cast<intptr_t>(&response);
    // TODO(brianderson): Pipeline staging, issuance, and completion to make
    // verification faster.
    query_manager.StageQuery(i, id);
    query_manager.IssueQuery(id);
    response.WaitForCompletion();
    query_manager.RetireQuery(id);
    qsl->UpdateAccuracyMetric(response.query_index, response.data.data(),
                              response.data.size());
  }
  std::cout << "SUT accuracy metric:" << qsl->GetAccuracyMetric();
}

void RunPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                        const TestSettings& settings) {
  std::cerr << "RunPerformanceMode not implemented.";
}

void StartTest(SystemUnderTest* sut, const TestSettings& settings) {
  std::string qsl_name(settings.query_sample_library_name,
                       settings.query_sample_library_name);
  QuerySampleLibrary* qsl = QslRegistry::GetQslInstance(qsl_name);
  if (!qsl) {
    std::cerr << "Did not find the requested query sample library: " << qsl_name
              << "\n";
    return;
  }

  switch (settings.mode) {
    case TestMode::SubmissionRun:
      RunVerificationMode(sut, qsl, settings);
      RunPerformanceMode(sut, qsl, settings);
      break;
    case TestMode::AccuracyOnly:
      RunVerificationMode(sut, qsl, settings);
      break;
    case TestMode::PerformanceOnly:
      RunPerformanceMode(sut, qsl, settings);
      break;
    case TestMode::SearchForQps:
      std::cerr << "TestMode::SearchForQps not implemented.";
      break;
  }
}

}  // namespace mlperf
