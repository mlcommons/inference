#include "loadgen.h"

#include <stdint.h>

#include <atomic>
#include <cassert>
#include <future>
#include <iostream>
#include <random>
#include <thread>

#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"

namespace mlperf {

namespace {
constexpr uint64_t kSeed = 0xABCD1234;
}  // namespace

struct QueryMetadata;

// QueryResponseMetadata is used by the load generator to coordinate
// response data and completion.
struct SampleMetadata {
  QueryMetadata* query_metadata;
  QuerySampleIndex sample_index;
  std::vector<uint8_t> data;
  // TODO: Timestamp.
};

struct QueryMetadata {
  std::vector<SampleMetadata> samples;

  void Prepare(std::vector<QuerySample>* query_to_send) {
    assert(wait_count.load() == 0);
    size_t query_count = query_to_send->size();
    samples.clear();
    samples.resize(query_count);
    for (size_t i = 0; i < query_count; i++) {
      SampleMetadata* sample_metadata = &samples[i];
      sample_metadata->query_metadata = this;
      sample_metadata->sample_index = (*query_to_send)[i].index;
      (*query_to_send)[i].id = reinterpret_cast<ResponseId>(sample_metadata);
    }
    wait_count.store(query_count);
    all_samples_done = std::promise<void>();
  }

  void NotifySampleCompleted() {
    size_t old_count = wait_count.fetch_sub(1);
    if (old_count == 1) {
      all_samples_done.set_value();
    }
  }

  void WaitForCompletion() {
    all_samples_done.get_future().wait();
  }

 private:
  // Initializing as fulfilled prevents code from having to special
  // case initial use vs reuse.
  std::promise<void> init_promise_as_fulfilled() {
    std::promise<void> init;
    init.set_value();
    return init;
  }

  std::atomic<size_t> wait_count { 0 };
  std::promise<void> all_samples_done { init_promise_as_fulfilled() };
};

void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count) {
  for (size_t i = 0; i < response_count; i++) {
    QuerySampleResponse &response = responses[i];
    SampleMetadata* sample_metadata =
        reinterpret_cast<SampleMetadata*>(response.id);
    // TODO(brianderson): Don't copy data in performance mode.
    auto* src_begin = reinterpret_cast<uint8_t*>(response.data);
    auto* src_end = src_begin + response.size;
    sample_metadata->data = std::vector<uint8_t>(src_begin, src_end);
    sample_metadata->query_metadata->NotifySampleCompleted();
  }
}

// Generates random sets of indicies where indicies are selected from
// |samples_src| without replacement.
// The sets are limitted in size by |set_size|.
// TODO: Choosing bins randomly, rather than samples randomly, would be more
// straightforward and faster.
std::vector<std::vector<QuerySampleIndex>> GenerateDisjointSets(
    const std::vector<QuerySampleIndex> &samples_src,
    size_t set_size, std::mt19937 *gen) {
  constexpr float kGarbageCollectRatio = 0.5;
  constexpr size_t kUsedIndex = std::numeric_limits<size_t>::max();

  std::vector<std::vector<QuerySampleIndex>> result;

  std::vector<QuerySampleIndex> samples(samples_src);

  std::vector<QuerySampleIndex> loadable_set;
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

    // Garbage collect used indicies as probability of hitting one
    // increases.
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

std::vector<QuerySample> SampleIndexesToQuery(
    std::vector<QuerySampleIndex> src_set) {
  std::vector<QuerySample> dst_set;
  dst_set.reserve(src_set.size());
  for (const auto &index : src_set) {
    dst_set.push_back({0, index});
  }
  return dst_set;
}

void RunVerificationMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                         const TestSettings& settings) {
  constexpr size_t kMaxAsyncQueries = 4;
  constexpr int64_t kNanosecondsPerSecond = 1000000000;

  std::mt19937 gen(kSeed);

  if (settings.scenario != TestScenario::MultiStream) {
    std::cerr << "Unsupported scenario. Only MultiStream supported.\n";
  }

  // Generate indicies for all available samples in the QSL.
  size_t qsl_total_count = qsl->TotalSampleCount();
  std::vector<QuerySampleIndex> qsl_samples(qsl_total_count);
  for(size_t i = 0; i < qsl_total_count; i++) {
    qsl_samples[i] = static_cast<QuerySampleIndex>(i);
  }

  // Generate random sets of samples that we can load into RAM.
  std::vector<std::vector<QuerySampleIndex>> loadable_sets =
      GenerateDisjointSets(qsl_samples,
                           qsl->PerformanceSampleCount(),
                           &gen);

  // Iterate through each loadable set.
  auto start = std::chrono::high_resolution_clock::now();
  size_t i_query = 0;
  uint64_t tick_count = 0;
  QueryMetadata issued_query_infos[kMaxAsyncQueries];
  for (auto &loadable_set : loadable_sets) {
    qsl->LoadSamplesToRam(loadable_set.data(), loadable_set.size());

    // Split the set up into random queries.
    std::vector<std::vector<QuerySampleIndex>> queries =
        GenerateDisjointSets(loadable_set, settings.samples_per_query, &gen);
    for (auto &query : queries) {
      // Limit the number of oustanding async queries by waiting for
      // old queries here.
      size_t i_issued_query_info = i_query % kMaxAsyncQueries;
      auto& issued_query_info = issued_query_infos[i_issued_query_info];
      issued_query_info.WaitForCompletion();

      // Sleep until the next tick, skipping ticks that have already passed.
      auto query_start_time = start;
      do {
        tick_count++;
        std::chrono::nanoseconds delta(static_cast<int64_t>(
            (tick_count * kNanosecondsPerSecond) / settings.target_qps));
        query_start_time = start + delta;
      } while (query_start_time < std::chrono::high_resolution_clock::now());
      std::this_thread::sleep_until(query_start_time);

      // Issue the query to the SUT.
      std::vector<QuerySample> query_to_send = SampleIndexesToQuery(query);
      issued_query_info.Prepare(&query_to_send);
      sut->IssueQuery(query_to_send.data(), query_to_send.size());
      i_query++;
    }

    qsl->UnloadSamplesFromRam(loadable_set.data(), loadable_set.size());
  }

  // Wait for tail queries to complete and process them.
  // We have to keep the synchronization primitives alive until the SUT
  // is done with them.
  for (size_t i = 0; i < kMaxAsyncQueries; i++) {
    size_t i_issued_query_info = i_query % kMaxAsyncQueries;
    auto& issued_query_info = issued_query_infos[i_issued_query_info];
    issued_query_info.WaitForCompletion();
    i_query++;
  }

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
