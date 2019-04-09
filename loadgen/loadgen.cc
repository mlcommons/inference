#include "loadgen.h"

#include <stdint.h>

#include <cassert>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"

namespace mlperf {

namespace {
constexpr uint64_t kSeed = 0xABCD1234;
}  // namespace

// QueryResponseMetadata is used by the load generator to coordinate
// response data and completion.
struct QueryResponseMetadata {
  std::vector<QuerySample> query;
  std::vector<std::vector<uint8_t>> data;

  void Reset() {
    query.clear();
    data.clear();
    complete = false;
  }

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
  bool complete = false;
};

void QueryComplete(QueryId query_id, QuerySampleResponse* responses,
                   size_t response_count) {
  QueryResponseMetadata* metadata =
      reinterpret_cast<QueryResponseMetadata*>(query_id);
  // TODO(brianderson): Don't copy data in performance mode.
  for (size_t i = 0; i < response_count; i++) {
    auto* src_begin = reinterpret_cast<uint8_t*>(responses[i].data);
    auto* src_end = src_begin + responses[i].size;
    metadata->data.push_back(std::vector<uint8_t>(src_begin, src_end));
  }
  std::cout << "Query complete ID: " << query_id << "\n";
  std::cout << "Query library index: " << metadata->query[0] << "\n";
  metadata->NotifyCompletion();
}

// Generates random sets of indicies where indicies are selected from
// the rage [0,|total_count|) without replacement.
// The sets are limitted in size by |loadable_count|.
// TODO: Choosing bins randomly, rather than samples randomly, would be more
// straightforward and faster.
std::vector<std::vector<QuerySample>> GenerateDisjointSets(
    const std::vector<QuerySample> &samples_src,
    size_t set_size, std::mt19937 *gen) {
  constexpr float kGarbageCollectRatio = 0.5;
  constexpr size_t kUsedIndex = std::numeric_limits<size_t>::max();

  std::vector<std::vector<size_t>> result;

  std::vector<QuerySample> samples(samples_src);

  std::vector<size_t> loadable_set;
  loadable_set.reserve(set_size);
  size_t remaining_count = samples.size();
  size_t garbage_collect_count = remaining_count * kGarbageCollectRatio;
  std::uniform_int_distribution<> uniform_distribution(0, remaining_count-1);
  while (remaining_count > 0) {
    size_t candidate_index = uniform_distribution(*gen);
    if (samples[candidate_index] == kUsedIndex) {
      continue;
    }
    loadable_set.push_back(samples[candidate_index]);
    if (loadable_set.size() == set_size) {
      result.push_back(std::move(loadable_set));
      loadable_set.clear();
      loadable_set.reserve(set_size);
    }
    samples[candidate_index] = kUsedIndex;
    remaining_count--;
    if (garbage_collect_count != 0) {
      garbage_collect_count--;
      continue;
    }

    // Garbage collect used indicies.
    std::vector<size_t> gc_samples;
    gc_samples.reserve(remaining_count);
    for (auto s : samples) {
      if (s != kUsedIndex) {
        gc_samples.push_back(s);
      }
    }
    assert(remaining_count == gc_samples.size());
    samples = std::move(gc_samples);
    uniform_distribution.param(
          std::uniform_int_distribution<>::param_type(0, remaining_count-1));
    garbage_collect_count = remaining_count * kGarbageCollectRatio;
  }

  if (!loadable_set.empty()) {
    result.push_back(std::move(loadable_set));
  }
  return result;
}

void RunVerificationMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                         const TestSettings& settings) {
  constexpr size_t kMaxAsyncQueries = 2;
  constexpr int64_t kNanosecondsPerSecond = 1000000000;

  std::mt19937 gen(kSeed);

  if (settings.scenario != TestScenario::StreamNAtFixedRate) {
    std::cerr << "Unsupported scenario. Only StreamNAtFixedRate supported.\n";
  }

  size_t qsl_total_count = qsl->TotalSampleCount();
  std::vector<QuerySample> qsl_samples(qsl_total_count);
  for(size_t i = 0; i < qsl_total_count; i++) {
    qsl_samples[i] = static_cast<QuerySample>(i);
  }

  std::vector<std::vector<QuerySample>> loadable_sets =
      GenerateDisjointSets(qsl_samples,
                           qsl->PerformanceSampleCount(),
                           &gen);

  auto start = std::chrono::high_resolution_clock::now();
  size_t i_query = 0;
  uint64_t tick_count = 0;
  QueryResponseMetadata issued_query_infos[kMaxAsyncQueries];
  for (auto &loadable_set : loadable_sets) {
    qsl->LoadSamplesToRam(loadable_set.data(), loadable_set.size());

    std::vector<std::vector<QuerySample>> queries =
        GenerateDisjointSets(loadable_set, settings.samples_per_query, &gen);

    for (auto &query : queries) {
      size_t i_issued_query_info = i_query % kMaxAsyncQueries;
      auto& issued_query_info = issued_query_infos[i_issued_query_info];
      if (i_query > kMaxAsyncQueries) {
        issued_query_info.WaitForCompletion();
      }

      auto query_start_time = start;
      do {
        tick_count++;
        std::chrono::nanoseconds delta(static_cast<int64_t>(
            (tick_count * kNanosecondsPerSecond) / settings.target_qps));
        query_start_time = start + delta;
      } while (query_start_time < std::chrono::high_resolution_clock::now());
      std::this_thread::sleep_until(query_start_time);

      issued_query_info.Reset();
      issued_query_info.query = query;
      intptr_t query_id = reinterpret_cast<intptr_t>(&issued_query_info);
      sut->IssueQuery(query_id, query.data(), query.size());
      i_query++;
    }
    qsl->UnloadSamplesFromRam(loadable_set.data(), loadable_set.size());
  }

  // TODO: WaitForCompletion on tail queries. Then process.

  std::cout << "SUT accuracy metric:" << qsl->GetAccuracyMetric();
}

void RunPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                        const TestSettings& settings) {
  std::cerr << "RunPerformanceMode not implemented.\n";
}

void StartTest(SystemUnderTest* sut,
               QuerySampleLibrary* qsl,
               const TestSettings& settings) {
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
      std::cerr << "TestMode::SearchForQps not implemented.\n";
      break;
  }
}

}  // namespace mlperf
