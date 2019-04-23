#include "loadgen.h"

#include <stdint.h>

#include <atomic>
#include <cassert>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <fstream>

#include "logging.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"

namespace mlperf {

namespace {
constexpr uint64_t kSeed = 0xABCD1234;
}  // namespace

struct SampleMetadata;
struct QueryMetadata;

struct SampleCompleteDelagate {
  virtual void Notify(QueryMetadata*, SampleMetadata*,
                      QuerySampleResponse*, PerfClock::time_point) = 0;
};

// SampleMetadata is used by the load generator to coordinate
// response data and completion.
struct SampleMetadata {
  QueryMetadata* query_metadata;
  uint64_t sequence_id;
  QuerySampleIndex sample_index;
};

class QueryMetadata {
 public:
  void Prepare(std::vector<QuerySample>* query_to_send,
               uint64_t* sample_sequence_id) {
    assert(wait_count_.load() == 0);
    size_t query_count = query_to_send->size();
    samples_.clear();
    samples_.resize(query_count);
    for (size_t i = 0; i < query_count; i++) {
      SampleMetadata* sample_metadata = &samples_[i];
      sample_metadata->query_metadata = this;
      sample_metadata->sequence_id = (*sample_sequence_id)++;
      sample_metadata->sample_index = (*query_to_send)[i].index;
      (*query_to_send)[i].id = reinterpret_cast<ResponseId>(sample_metadata);
    }
    slot_free_count_.store(query_count, std::memory_order_relaxed);
    wait_count_.store(query_count, std::memory_order_relaxed);
    all_samples_done_ = std::promise<void>();
  }

  void NotifyOneSampleCompleted() {
    size_t old_count = wait_count_.fetch_sub(1);
    if (old_count == 1) {
      all_samples_done_.set_value();
    }
  }

  void WaitForAllSamplesCompleted() {
    all_samples_done_.get_future().wait();
  }

  void RemoveOneSampleRef() {
    slot_free_count_.fetch_sub(1);
  }

  bool ReadyForNewQuery() {
    return 0 == slot_free_count_.load();
  }

 public:
  SampleCompleteDelagate* sample_complete_logger;
  uint64_t sequence_id;
  PerfClock::time_point scheduled_time;
  PerfClock::time_point wait_for_slot_time;
  PerfClock::time_point issued_start_time;

 private:
  // Initializing as fulfilled prevents code from having to special
  // case initial use vs reuse.
  std::promise<void> init_promise_as_fulfilled() {
    std::promise<void> init;
    init.set_value();
    return init;
  }

  // |slot_free_count_| is used to verify that the logging logic is done
  //   reading the data from this class.
  // |wait_count_| is used to notify of completion asap.
  std::atomic<size_t> slot_free_count_ { 0 };
  std::atomic<size_t> wait_count_ { 0 };
  std::promise<void> all_samples_done_ { init_promise_as_fulfilled() };
  std::vector<SampleMetadata> samples_;
};

struct DurationGeneratorNs {
  PerfClock::time_point start;
  int64_t delta(PerfClock::time_point end) const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
          end - start).count();
  }
};

void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count) {
  PerfClock::time_point timestamp = PerfClock::now();
  const QuerySampleResponse* end = responses + response_count;

  // Notify first to unblock loadgen production ASAP.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    SampleMetadata* sample = reinterpret_cast<SampleMetadata*>(response->id);
    QueryMetadata* query = sample->query_metadata;
    query->NotifyOneSampleCompleted();
  }

  // Log samples and release query metadata for re-use.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    SampleMetadata* sample = reinterpret_cast<SampleMetadata*>(response->id);
    QueryMetadata* query = sample->query_metadata;
    query->sample_complete_logger->Notify(query, sample, response, timestamp);
    query->RemoveOneSampleRef();
  }
}


// Right now, this is the only implementation of SampleCompleteDelagate,
// but more will be coming soon.
// TODO: Versions that don't copy data.
// TODO: Versions that have less detailed logs.
// TODO: Versions that do a delayed notification.
struct SampleCompleteDetailed : public SampleCompleteDelagate {
  SampleCompleteDetailed(Logger* logger) : logger(logger) {}

  Logger* logger;

  void Notify(QueryMetadata* query,
              SampleMetadata* sample,
              QuerySampleResponse* response,
              PerfClock::time_point complete_begin_time) override {
    auto* src_begin = reinterpret_cast<uint8_t*>(response->data);
    auto* src_end = src_begin + response->size;
    std::vector<uint8_t>sample_data_copy(src_begin, src_end);
    Log(logger,
        // Copy lots of data to the lambda since it may be overwritten soon.
        // TODO: Avoid the extra copies here.
        [sample_sequence_id = sample->sequence_id,
         query_sequence_id = query->sequence_id,
         sample_index = sample->sample_index,
         origin_time = logger->origin_time(),
         scheduled_time = query->scheduled_time,
         issued_start_time = query->issued_start_time,
         wait_for_slot_time = query->wait_for_slot_time,
         complete_begin_time = complete_begin_time,
         sample_data = std::move(sample_data_copy)](AsyncLog& trace) {
          DurationGeneratorNs origin { origin_time };
          DurationGeneratorNs sched { scheduled_time };
          trace.AsyncEvent("Sample",
                sample_sequence_id,
                origin.delta(scheduled_time),
                sched.delta(complete_begin_time), // This is the latency.
                "sample_seq", sample_sequence_id,
                "query_seq", query_sequence_id,
                "idx", sample_index,
                "sched_ns", origin.delta(scheduled_time),
                "wait_for_slot_ns", sched.delta(wait_for_slot_time),
                "issue_start_ns", sched.delta(issued_start_time),
                "complete_ns", sched.delta(complete_begin_time));
        });

  }
};

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

    // Garbage collect used indicies as probability of hitting one increases.
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
                         const TestSettings& settings, Logger* logger) {
  // kQueryPoolSize should be big enough that we don't need to worry about
  // logging falling too far behind.
  constexpr size_t kQueryPoolSize = 1024;
  constexpr size_t kMaxAsyncQueries = 4;
  constexpr int64_t kNanosecondsPerSecond = 1000000000;

  logger->SetExpectedLatencies(qsl->TotalSampleCount());

  SampleCompleteDetailed sample_complete_logger(logger);
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
  auto start = PerfClock::now();
  size_t i_query = 0;
  uint64_t sample_sequence_id = 0;
  uint64_t tick_count = 0;
  QueryMetadata query_info_pool[kQueryPoolSize];
  QueryMetadata* issued_query_infos[kMaxAsyncQueries] = {}; // zeroes.
  for (auto &loadable_set : loadable_sets) {
    qsl->LoadSamplesToRam(loadable_set.data(), loadable_set.size());

    // Split the set up into random queries.
    std::vector<std::vector<QuerySampleIndex>> queries =
        GenerateDisjointSets(loadable_set, settings.samples_per_query, &gen);
    for (auto &query : queries) {
      // Sleep until the next tick, skipping ticks that have already passed.
      auto query_start_time = start;
      do {
        tick_count++;
        std::chrono::nanoseconds delta(static_cast<int64_t>(
            (tick_count * kNanosecondsPerSecond) / settings.target_qps));
        query_start_time = start + delta;
      } while (query_start_time < PerfClock::now());
      std::this_thread::sleep_until(query_start_time);

      // Limit the number of oustanding async queries by waiting for
      // old queries here.
      size_t i_query_pool = i_query % kQueryPoolSize;
      QueryMetadata& issued_query_info = query_info_pool[i_query_pool];

      auto wait_for_slot_time = PerfClock::now();
      size_t i_limitting_query = i_query % kMaxAsyncQueries;
      QueryMetadata*& limitting_query = issued_query_infos[i_limitting_query];
      if (limitting_query != nullptr) {
        limitting_query->WaitForAllSamplesCompleted();
      }
      limitting_query = &issued_query_info;

      assert(issued_query_info.ReadyForNewQuery());

      // Issue the query to the SUT.
      std::vector<QuerySample> query_to_send = SampleIndexesToQuery(query);
      issued_query_info.Prepare(&query_to_send, &sample_sequence_id);
      issued_query_info.sample_complete_logger = &sample_complete_logger;
      issued_query_info.sequence_id = i_query;
      issued_query_info.scheduled_time = query_start_time;
      issued_query_info.wait_for_slot_time = wait_for_slot_time;
      issued_query_info.issued_start_time = PerfClock::now();
      sut->IssueQuery(query_to_send.data(), query_to_send.size());
      i_query++;
    }

    qsl->UnloadSamplesFromRam(loadable_set.data(), loadable_set.size());
  }

  // Wait for tail queries to complete and collect all the latencies.
  // We have to keep the synchronization primitives alive until the SUT
  // is done with them.
  std::vector<std::chrono::nanoseconds> latencies =
      logger->GetLatenciesBlocking();

  // Compute percentile.  
  std::sort(latencies.begin(), latencies.end());
  size_t i90 = latencies.size() * .9;
  std::cout << "90th percentile latency: " << latencies[i90].count() << "ns\n";
}

void RunPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                        const TestSettings& settings) {
  std::cerr << "RunPerformanceMode not implemented.\n";
}

void StartTest(SystemUnderTest* sut,
               QuerySampleLibrary* qsl,
               const TestSettings& settings) {
  constexpr size_t kMaxLoggingThreads = 512;
  std::string log_filename = "mlperf_log.json";
  std::ofstream out_file(log_filename);
  Logger logger(&out_file, std::chrono::nanoseconds(1), kMaxLoggingThreads);
  switch (settings.mode) {
    case TestMode::SubmissionRun:
      RunVerificationMode(sut, qsl, settings, &logger);
      RunPerformanceMode(sut, qsl, settings);
      break;
    case TestMode::AccuracyOnly:
      RunVerificationMode(sut, qsl, settings, &logger);
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
