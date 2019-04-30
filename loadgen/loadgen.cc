#include "loadgen.h"

#include <stdint.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>
#include <future>
#include <queue>
#include <random>
#include <thread>

#include "logging.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "utils.h"

namespace mlperf {

namespace {
constexpr uint64_t kDefaultQslSeed = 0xABCD1234;
constexpr uint64_t kDefaultSampleSeed = 0xABCD1234;
constexpr uint64_t kDefaultScheduleSeed = 0x1234ABCD;
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
  QueryMetadata(const std::vector<QuerySampleIndex>& query_sample_indicies,
                std::chrono::nanoseconds scheduled_delta,
                SampleCompleteDelagate* sample_complete_delegate,
                uint64_t query_sequence_id,
                uint64_t* sample_sequence_id)
    : scheduled_delta(scheduled_delta),
      sample_complete_delegate(sample_complete_delegate),
      sequence_id(query_sequence_id),
      wait_count_(query_sample_indicies.size()) {
    for (QuerySampleIndex i : query_sample_indicies) {
      samples_.push_back({ this, (*sample_sequence_id)++, i });
      query_to_send.push_back(
          { reinterpret_cast<ResponseId>(&samples_.back()), i });
    }
  }

  QueryMetadata(QueryMetadata&& src)
    : query_to_send(std::move(src.query_to_send)),
      scheduled_delta(src.scheduled_delta),
      sample_complete_delegate(src.sample_complete_delegate),
      sequence_id(src.sequence_id),
      wait_count_(src.samples_.size()),
      samples_(std::move(src.samples_)) {
    // The move constructor should only be called while generating a
    // vector of QueryMetadata, before it's been used.
    // Assert that wait_count_ is in its initial state.
    assert(wait_count_.load() == samples_.size());
    // Update the "parent" of each sample to be this query; the old query
    // address will no longer be valid.
    for (auto& s : samples_) {
      s.query_metadata = this;
    }
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

 public:
  std::vector<QuerySample> query_to_send;
  const std::chrono::nanoseconds scheduled_delta;
  SampleCompleteDelagate* const sample_complete_delegate;
  const uint64_t sequence_id;

  // Tracing timestamps.
  PerfClock::time_point scheduled_time;
  PerfClock::time_point wait_for_slot_time;
  PerfClock::time_point issued_start_time;

 private:
  std::atomic<size_t> wait_count_;
  std::promise<void> all_samples_done_;
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
    query->sample_complete_delegate->Notify(sample, response, timestamp);
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
          if (sample_data_copy != nullptr) {
            delete [] sample_data_copy;
          }
        });
  }
};

// ScheduleDistribution templates by test scenario.
template <TestScenario scenario>
auto ScheduleDistribution(double qps) {
  return [period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / qps))] (auto& gen) {
    return period;
  };
}

template <>
auto ScheduleDistribution<TestScenario::Cloud>(double qps) {
  // Poisson arrival process corresponds to exponentially distributed
  // interarrival times.
  return [dist = std::exponential_distribution<>(qps)](auto& gen) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

// SampleDistribution templates by test mode.
template <TestMode mode>
auto SampleDistribution(size_t sample_count) {
  return [sample_count, i = size_t(0)]
      (auto& gen) mutable {
        return (i++) % sample_count;
      };
}

template <>
auto SampleDistribution<TestMode::PerformanceOnly>(size_t sample_count) {
  return [dist = std::uniform_int_distribution<>(0, sample_count - 1)]
      (auto& gen) mutable {
        return dist(gen);
      };
}

template <TestScenario scenario, TestMode mode>
std::vector<QueryMetadata> GenerateQueries(
    const TestSettings& settings,
    const std::vector<QuerySampleIndex>& loaded_samples,
    SampleCompleteDelagate* sample_complete_delegate) {

  constexpr std::chrono::seconds kMinDuration(
        mode == TestMode::AccuracyOnly ? 0 : 60);
  // TODO: Pull MLPerf spec constants into a common area.
  constexpr size_t kMinQueriesPerformance =
      scenario == TestScenario::SingleStream ? 1024 : 24576;
  const size_t samples_per_query =
      scenario == TestScenario::SingleStream ?
        1 : settings.samples_per_query;
  const size_t min_queries_accuracy =
      (loaded_samples.size() + samples_per_query - 1) / samples_per_query;
  const size_t min_queries = (mode == TestMode::AccuracyOnly) ?
        min_queries_accuracy : kMinQueriesPerformance;
  std::vector<QueryMetadata> queries;

  assert(scenario == settings.scenario);

  // TODO: Allow overriding of seeds for experimentation.
  const uint64_t sample_seed = kDefaultSampleSeed;
  const uint64_t schedule_seed = kDefaultScheduleSeed;

  // Using the std::mt19937 pseudo-random number generator ensures a modicum of
  // cross platform reproducibility for trace generation.
  std::mt19937 sample_rng(sample_seed);
  std::mt19937 schedule_rng(schedule_seed);

  auto sample_distribution = SampleDistribution<mode>(loaded_samples.size());
  auto schedule_distribution =
      ScheduleDistribution<scenario>(settings.target_qps);

  std::chrono::nanoseconds timestamp(0);
  std::vector<QuerySampleIndex> samples(samples_per_query);
  uint64_t query_sequence_id = 0;
  uint64_t sample_sequence_id = 0;
  while (timestamp < kMinDuration || queries.size() < min_queries) {
    for (auto& s: samples) {
      s = loaded_samples[sample_distribution(sample_rng)];
    }
    timestamp += schedule_distribution(schedule_rng);
    queries.emplace_back(samples,
                         timestamp,
                         sample_complete_delegate,
                         query_sequence_id,
                         &sample_sequence_id);
    query_sequence_id++;
  }
  return queries;
}

void RunVerificationMode(
    SystemUnderTest* sut,
    QuerySampleLibrary* qsl,
    const TestSettings& settings,
    const std::vector<std::vector<QuerySampleIndex>>& loadable_sets) {

  LogDetail([](AsyncLog &log) {
    log.LogDetail("Starting verification mode:"); });

  // kQueryPoolSize should be big enough that we don't need to worry about
  // logging falling too far behind.
  constexpr size_t kMaxAsyncQueries = 4;
  constexpr int64_t kNanosecondsPerSecond = 1000000000;

  SampleCompleteDetailed sample_complete_logger;

  if (settings.scenario != TestScenario::MultiStream) {
    LogError([](AsyncLog &log) {
      log.LogDetail("Unsupported scenario. Only MultiStream supported."); });
    return;
  }

  // Iterate through each loadable set.
  PerfClock::time_point start = PerfClock::now();
  uint64_t tick_count = 0;
  for (auto &loadable_set : loadable_sets) {
    qsl->LoadSamplesToRam(loadable_set);
    GlobalLogger().RestartLatencyRecording();

    // Split the set up into queries.
    std::vector<QueryMetadata> queries =
        GenerateQueries<TestScenario::MultiStream, TestMode::AccuracyOnly>(
          settings, loadable_set, &sample_complete_logger);

    std::queue<QueryMetadata*> prev_queries;
    for (auto& query : queries) {
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
      PerfClock::time_point wait_for_slot_time = PerfClock::now();
      if (prev_queries.size() > kMaxAsyncQueries) {
        QueryMetadata* limitting_query = prev_queries.front();
        limitting_query->WaitForAllSamplesCompleted();
        prev_queries.pop();
      }
      prev_queries.push(&query);

      // Issue the query to the SUT.
      query.scheduled_time = query_start_time;
      query.wait_for_slot_time = wait_for_slot_time;
      auto trace = MakeScopedTracer(
            [](AsyncLog &log){ log.ScopedTrace("IssueQuery"); });
      query.issued_start_time = PerfClock::now();
      sut->IssueQuery(query.query_to_send);
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
void RunPerformanceMode(
    SystemUnderTest* sut,
    QuerySampleLibrary* qsl,
    const TestSettings& settings,
    const std::vector<std::vector<QuerySampleIndex>>& loadable_sets) {

  LogDetail([](AsyncLog &log) {
    log.LogDetail("Starting performance mode:"); });

  GlobalLogger().RestartLatencyRecording();

  SampleCompleteDetailed sample_complete_logger;

  if (settings.scenario != TestScenario::SingleStream) {
    LogError([](AsyncLog &log) {
      log.LogDetail("Unsupported scenario. Only SingleStream supported."); });
    return;
  }

  // Use first loadable set as the performance set.
  const std::vector<QuerySampleIndex>& performance_set = loadable_sets.front();

  qsl->LoadSamplesToRam(performance_set);
  std::vector<QueryMetadata> queries =
      GenerateQueries<TestScenario::SingleStream, TestMode::PerformanceOnly>(
        settings, performance_set, &sample_complete_logger);

  PerfClock::time_point start = PerfClock::now();
  PerfClock::time_point last_now = start;
  QueryMetadata* prev_query = nullptr;
  for(auto& query : queries) {
    auto trace1 = MakeScopedTracer(
        [](AsyncLog &log){ log.ScopedTrace("SampleLoop"); });

    PerfClock::time_point wait_for_slot_time = last_now;
    if (prev_query != nullptr) {
      auto trace2 = MakeScopedTracer(
          [](AsyncLog &log){ log.ScopedTrace("WaitOnPrev"); });
      prev_query->WaitForAllSamplesCompleted();
    }
    prev_query = &query;

    // Issue the query to the SUT.
    query.wait_for_slot_time = wait_for_slot_time;
    last_now = PerfClock::now();
    query.scheduled_time = last_now;
    query.issued_start_time = last_now;
    auto trace3 = MakeScopedTracer(
        [](AsyncLog &log){ log.ScopedTrace("IssueQuery"); });
    sut->IssueQuery(query.query_to_send);
  }

  // Wait for tail queries to complete and collect all the latencies.
  // We have to keep the synchronization primitives alive until the SUT
  // is done with them.
  const size_t expected_latencies = queries.size();
  std::vector<QuerySampleLatency> latencies(
      GlobalLogger().GetLatenciesBlocking(expected_latencies));

  sut->ReportLatencyResults(latencies);

  // Compute percentile.
  std::sort(latencies.begin(), latencies.end());
  int64_t accumulator = 0;
  for (auto l : latencies) {
    accumulator += l;
  }
  int64_t mean_latency = accumulator / latencies.size();
  size_t i90 = latencies.size() * .9;
  Log([mean_latency, l90 = latencies[i90]](AsyncLog &log) {
    log.LogSummary(
          "Loadgen results summary:"
          "\n  Mean latency (ns) : " + std::to_string(mean_latency) +
          "\n  90th percentile latency (ns) : " + std::to_string(l90));
  });

  qsl->UnloadSamplesFromRam(performance_set);
}

// Generates random sets of samples in the QSL that we can load into RAM
// at the same time.
// Choosing samples randomly to go into a set naturally avoids biasing some
// samples to a particular set.
// TODO: Choosing bins randomly, rather than samples randomly, would avoid the
//       garbage collection logic, but we'd need to avoid later samples being
//       less likely to be in the smallest set. This may not be an important
//       requirement though.
std::vector<std::vector<QuerySampleIndex>> GenerateLoadableSets(
    QuerySampleLibrary* qsl) {
  constexpr float kGarbageCollectRatio = 0.5;
  constexpr size_t kUsedIndex = std::numeric_limits<size_t>::max();

  auto trace = MakeScopedTracer(
      [](AsyncLog &log){ log.ScopedTrace("GenerateLoadableSets"); });

  std::vector<std::vector<QuerySampleIndex>> result;
  std::mt19937 qsl_rng(kDefaultQslSeed);

  // Generate indicies for all available samples in the QSL.
  const size_t qsl_total_count = qsl->TotalSampleCount();
  std::vector<QuerySampleIndex> samples(qsl_total_count);
  for(size_t i = 0; i < qsl_total_count; i++) {
    samples[i] = static_cast<QuerySampleIndex>(i);
  }

  const size_t set_size = qsl->PerformanceSampleCount();
  std::vector<QuerySampleIndex> loadable_set;
  loadable_set.reserve(set_size);
  size_t remaining_count = samples.size();
  size_t garbage_collect_count = remaining_count * kGarbageCollectRatio;
  std::uniform_int_distribution<> dist(0, remaining_count-1);

  while (remaining_count > 0) {
    size_t candidate_index = dist(qsl_rng);
    // Skip indicies we've already used.
    if (samples[candidate_index] == kUsedIndex) {
      continue;
    }

    // Update loadable sets and mark index as used.
    loadable_set.push_back(samples[candidate_index]);
    if (loadable_set.size() == set_size) {
      result.push_back(std::move(loadable_set));
      loadable_set.clear();
      loadable_set.reserve(set_size);
    }
    samples[candidate_index] = kUsedIndex;
    remaining_count--;

    // Garbage collect used indicies as probability of hitting one increases.
    if (garbage_collect_count != 0) {
      garbage_collect_count--;
    } else {
      RemoveValue(&samples, kUsedIndex);
      assert(remaining_count == samples.size());
      dist.param(std::uniform_int_distribution<>::param_type(
          0, remaining_count-1));
      garbage_collect_count = remaining_count * kGarbageCollectRatio;
    }
  }

  if (!loadable_set.empty()) {
    result.push_back(std::move(loadable_set));
  }
  return result;
}

void StartTest(SystemUnderTest* sut,
               QuerySampleLibrary* qsl,
               const TestSettings& settings) {
  std::ofstream summary_out("mlperf_log_summary.txt");
  std::ofstream detail_out("mlperf_log_detail.txt");
  GlobalLogger().StartLogging(&summary_out, &detail_out);
  std::ofstream trace_out("mlperf_trace.json");
  GlobalLogger().StartNewTrace(&trace_out, PerfClock::now());

  std::vector<std::vector<QuerySampleIndex>> loadable_sets(
      GenerateLoadableSets(qsl));

  switch (settings.mode) {
    case TestMode::SubmissionRun:
      RunVerificationMode(sut, qsl, settings, loadable_sets);
      RunPerformanceMode(sut, qsl, settings, loadable_sets);
      break;
    case TestMode::AccuracyOnly:
      RunVerificationMode(sut, qsl, settings, loadable_sets);
      break;
    case TestMode::PerformanceOnly:
      RunPerformanceMode(sut, qsl, settings, loadable_sets);
      break;
    case TestMode::SearchForQps:
      LogError([](AsyncLog &log) {
        log.LogDetail("TestMode::SearchForQps not implemented."); });
      break;
  }

  GlobalLogger().StopTracing();
  GlobalLogger().StopLogging();
}

}  // namespace mlperf
