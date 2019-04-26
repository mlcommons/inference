#include "loadgen.h"

#include <stdint.h>

#include <atomic>
#include <cassert>
#include <cstring>
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
  virtual void Notify(SampleMetadata*,
                      QuerySampleResponse*,
                      PerfClock::time_point) = 0;
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
    query->sample_complete_logger->Notify(sample, response, timestamp);
  }
}

// Right now, this is the only implementation of SampleCompleteDelagate,
// but more will be coming soon.
// TODO: Versions that don't copy data.
// TODO: Versions that have less detailed logs.
// TODO: Versions that do a delayed notification.
struct SampleCompleteDetailed : public SampleCompleteDelagate {
  void Notify(SampleMetadata* sample,
              QuerySampleResponse* response,
              PerfClock::time_point complete_begin_time) override {
    // Using a raw pointer here should help us hit the std::function
    // small buffer optimization code path when we aren't copying data.
    // For some reason, using std::unique_ptr<std::vector> wasn't moving
    // into the lambda; even with C++14.
    auto sample_data_copy = new uint8_t[response->size];
    auto* src_begin = reinterpret_cast<uint8_t*>(response->data);
    std::memcpy(sample_data_copy, src_begin, response->size);
    Log([sample, sample_data_copy, complete_begin_time](AsyncLog& log) {
          QueryMetadata* query = sample->query_metadata;
          DurationGeneratorNs sched { query->scheduled_time };
          QuerySampleLatency latency = sched.delta(complete_begin_time);
          log.RecordLatency(sample->sequence_id, latency);
          log.TraceSample(
              "Sample",
              sample->sequence_id,
              query->scheduled_time,
              complete_begin_time,
              "sample_seq", sample->sequence_id,
              "query_seq", query->sequence_id,
              "idx", sample->sample_index,
              "wait_for_slot_ns", sched.delta(query->wait_for_slot_time),
              "issue_start_ns", sched.delta(query->issued_start_time),
              "complete_ns", sched.delta(complete_begin_time));
          query->RemoveOneSampleRef();
          if (sample_data_copy != nullptr) {
            delete [] sample_data_copy;
          }
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
                         const TestSettings& settings) {
  // kQueryPoolSize should be big enough that we don't need to worry about
  // logging falling too far behind.
  constexpr size_t kQueryPoolSize = 1024;
  constexpr size_t kMaxAsyncQueries = 4;
  constexpr int64_t kNanosecondsPerSecond = 1000000000;

  SampleCompleteDetailed sample_complete_logger;
  std::mt19937 gen(kSeed);

  if (settings.scenario != TestScenario::MultiStream) {
    std::cerr << "Unsupported scenario. Only MultiStream supported.\n";
    return;
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
  PerfClock::time_point start = PerfClock::now();
  size_t i_query = 0;
  uint64_t sample_sequence_id = 0;
  uint64_t tick_count = 0;
  QueryMetadata query_info_pool[kQueryPoolSize];
  QueryMetadata* issued_query_infos[kMaxAsyncQueries] = {}; // zeroes.
  for (auto &loadable_set : loadable_sets) {
    qsl->LoadSamplesToRam(loadable_set);
    GlobalLogger().RestartLatencyRecording();

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
      QueryMetadata& query_info = query_info_pool[i_query_pool];

      PerfClock::time_point wait_for_slot_time = PerfClock::now();
      size_t i_limitting_query = i_query % kMaxAsyncQueries;
      QueryMetadata*& limitting_query = issued_query_infos[i_limitting_query];
      if (limitting_query != nullptr) {
        limitting_query->WaitForAllSamplesCompleted();
      }
      limitting_query = &query_info;

      assert(query_info.ReadyForNewQuery());

      // Issue the query to the SUT.
      std::vector<QuerySample> query_to_send = SampleIndexesToQuery(query);
      query_info.Prepare(&query_to_send, &sample_sequence_id);
      query_info.sample_complete_logger = &sample_complete_logger;
      query_info.sequence_id = i_query;
      query_info.scheduled_time = query_start_time;
      query_info.wait_for_slot_time = wait_for_slot_time;
      auto trace = MakeScopedTracer(
            [](AsyncLog &log){ log.ScopedTrace("IssueQuery"); });
      query_info.issued_start_time = PerfClock::now();
      sut->IssueQuery(query_to_send);
      i_query++;
    }

    // Wait for tail queries to complete and collect all the latencies.
    // We have to keep the synchronization primitives and loaded samples
    // alive until the SUT is done with them.
    GlobalLogger().GetLatenciesBlocking(loadable_set.size());
    qsl->UnloadSamplesFromRam(loadable_set);
  }
}

// TODO: Share logic duplicated in RunVerificationMode.
// TODO: Simplify logic that will not be shared with RunVerificationMode.
void RunPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                        const TestSettings& settings) {
  GlobalLogger().RestartLatencyRecording();

  // kQueryInfoPoolSize should be big enough that we don't need to worry
  // about logging falling too far behind. And queries being processed
  // too far out of order.
  // TODO: Make this a TestSetting.
  constexpr size_t kQueryInfoPoolSize = 32 * 1024;
  constexpr size_t kMaxAsyncQueries = 1;
  constexpr size_t kMinQueryCount = 1024;
  constexpr std::chrono::seconds kMinDuration(60);

  SampleCompleteDetailed sample_complete_logger;
  std::mt19937 gen(kSeed);

  if (settings.scenario != TestScenario::SingleStream) {
    std::cerr << "Unsupported scenario. Only SingleStream supported.\n";
    return;
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

  // Use first loadable set as the performance set.
  std::vector<QuerySampleIndex>& performance_set = loadable_sets.front();

  size_t i_query = 0;
  uint64_t sample_sequence_id = 0;
  QueryMetadata query_info_pool[kQueryInfoPoolSize];
  QueryMetadata* issued_query_infos[kMaxAsyncQueries] = {}; // zeroes.

  qsl->LoadSamplesToRam(performance_set);

  // Split the set up into random queries.
  std::vector<std::vector<QuerySampleIndex>> queries =
      GenerateDisjointSets(performance_set, settings.samples_per_query, &gen);

  std::uniform_int_distribution<> uniform_distribution(
        0, performance_set.size()-1);

  PerfClock::time_point start = PerfClock::now();
  PerfClock::time_point last_now = start;
  PerfClock::time_point run_to_time_point_min = start + kMinDuration;
  while (i_query < kMinQueryCount || last_now < run_to_time_point_min) {
    auto trace1 = MakeScopedTracer(
        [](AsyncLog &log){ log.ScopedTrace("SampleLoop"); });
    std::vector<QuerySampleIndex> query(1, uniform_distribution(gen));

    // Limit the number of oustanding async queries by waiting for
    // old queries here.
    size_t i_query_pool = i_query % kQueryInfoPoolSize;
    QueryMetadata& query_info = query_info_pool[i_query_pool];

    PerfClock::time_point wait_for_slot_time = last_now;
    size_t i_limitting_query = i_query % kMaxAsyncQueries;
    QueryMetadata*& limitting_query = issued_query_infos[i_limitting_query];
    if (limitting_query != nullptr) {
      auto trace2 = MakeScopedTracer(
          [](AsyncLog &log){ log.ScopedTrace("WaitOnPrev"); });
      limitting_query->WaitForAllSamplesCompleted();
    }
    limitting_query = &query_info;

    assert(query_info.ReadyForNewQuery());

    // Issue the query to the SUT.
    std::vector<QuerySample> query_to_send = SampleIndexesToQuery(query);
    query_info.Prepare(&query_to_send, &sample_sequence_id);
    query_info.sample_complete_logger = &sample_complete_logger;
    query_info.sequence_id = i_query;
    query_info.wait_for_slot_time = wait_for_slot_time;
    last_now = PerfClock::now();
    query_info.scheduled_time = last_now;
    query_info.issued_start_time = last_now;
    auto trace3 = MakeScopedTracer(
        [](AsyncLog &log){ log.ScopedTrace("IssueQuery"); });
    sut->IssueQuery(query_to_send);
    i_query++;
  }

  // Wait for tail queries to complete and collect all the latencies.
  // We have to keep the synchronization primitives alive until the SUT
  // is done with them.
  const size_t expected_latencies = i_query;
  std::vector<QuerySampleLatency> latencies(
      GlobalLogger().GetLatenciesBlocking(expected_latencies));

  sut->ReportLatencyResults(latencies);

  // Compute percentile.
  // TODO: Output these to a summary log.
  std::cout << "Loadgen results summary:\n";
  std::sort(latencies.begin(), latencies.end());
  int64_t accumulator = 0;
  for (auto l : latencies) {
    accumulator += l;
  }
  std::cout << "Mean latency: " << accumulator / latencies.size() << "ns\n";
  size_t i90 = latencies.size() * .9;
  std::cout << "90th percentile latency: " << latencies[i90] << "ns\n";

  qsl->UnloadSamplesFromRam(performance_set);
}

void StartTest(SystemUnderTest* sut,
               QuerySampleLibrary* qsl,
               const TestSettings& settings) {
  std::string trace_filename = "mlperf_trace.json";
  std::ofstream trace_out(trace_filename);
  GlobalLogger().StartNewTrace(&trace_out, PerfClock::now());
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
  // Redirect traces to an overflow target.
  GlobalLogger().StopTracing();
}

}  // namespace mlperf
