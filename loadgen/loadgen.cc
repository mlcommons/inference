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

#include "loadgen.h"

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>
#include <future>
#include <queue>
#include <random>
#include <string>
#include <thread>

#include "logging.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "test_settings_internal.h"
#include "utils.h"
#include "version.h"

namespace mlperf {

struct SampleMetadata;
struct QueryMetadata;

// Every query and sample within a call to StartTest gets a unique sequence id
// for easy cross reference.
struct SequenceGen {
  uint64_t NextQueryId() { return query_id++; }
  uint64_t NextSampleId() { return sample_id++; }
  uint64_t CurrentSampleId() { return sample_id; }

 private:
  uint64_t query_id = 0;
  uint64_t sample_id = 0;
};

struct LoadableSampleSet {
  std::vector<QuerySampleIndex> set;
  const size_t sample_distribution_end;  // Excludes padding in multi-stream.
};

struct ResponseDelegate {
  virtual ~ResponseDelegate() = default;
  virtual void SampleComplete(SampleMetadata*, QuerySampleResponse*,
                              PerfClock::time_point) = 0;
  virtual void QueryComplete() = 0;
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
                ResponseDelegate* response_delegate, SequenceGen* sequence_gen)
      : scheduled_delta(scheduled_delta),
        response_delegate(response_delegate),
        sequence_id(sequence_gen->NextQueryId()),
        wait_count_(query_sample_indicies.size()) {
    samples_.reserve(query_sample_indicies.size());
    for (QuerySampleIndex qsi : query_sample_indicies) {
      samples_.push_back({this, sequence_gen->NextSampleId(), qsi});
    }
    query_to_send.reserve(query_sample_indicies.size());
    for (auto& s : samples_) {
      query_to_send.push_back(
          {reinterpret_cast<ResponseId>(&s), s.sample_index});
    }
  }

  QueryMetadata(QueryMetadata&& src)
      : query_to_send(std::move(src.query_to_send)),
        scheduled_delta(src.scheduled_delta),
        response_delegate(src.response_delegate),
        sequence_id(src.sequence_id),
        wait_count_(src.samples_.size()),
        samples_(std::move(src.samples_)) {
    // The move constructor should only be called while generating a
    // vector of QueryMetadata, before it's been used.
    // Assert that wait_count_ is in its initial state.
    assert(src.wait_count_.load() == samples_.size());
    // Update the "parent" of each sample to be this query; the old query
    // address will no longer be valid.
    // TODO: Only set up the sample parenting once after all the queries have
    //       been created, rather than re-parenting on move here.
    for (size_t i = 0; i < samples_.size(); i++) {
      SampleMetadata* s = &samples_[i];
      s->query_metadata = this;
      query_to_send[i].id = reinterpret_cast<ResponseId>(s);
    }
  }

  void NotifyOneSampleCompleted(PerfClock::time_point timestamp) {
    size_t old_count = wait_count_.fetch_sub(1, std::memory_order_relaxed);
    if (old_count == 1) {
      all_samples_done_time = timestamp;
      all_samples_done_.set_value();
      response_delegate->QueryComplete();
    }
  }

  void WaitForAllSamplesCompleted() { all_samples_done_.get_future().wait(); }

  PerfClock::time_point WaitForAllSamplesCompletedWithTimestamp() {
    all_samples_done_.get_future().wait();
    return all_samples_done_time;
  }

 public:
  std::vector<QuerySample> query_to_send;
  const std::chrono::nanoseconds scheduled_delta;
  ResponseDelegate* const response_delegate;
  const uint64_t sequence_id;

  // Performance information.

  int scheduled_intervals = 0;  // Number of intervals between queries, as
                                // actually scheduled during the run.
                                // For the multi-stream scenario only.
  PerfClock::time_point scheduled_time;
  PerfClock::time_point issued_start_time;
  PerfClock::time_point all_samples_done_time;

 private:
  std::atomic<size_t> wait_count_;
  std::promise<void> all_samples_done_;
  std::vector<SampleMetadata> samples_;
};

void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count) {
  PerfClock::time_point timestamp = PerfClock::now();

  auto tracer = MakeScopedTracer(
      [](AsyncTrace& trace) { trace("QuerySamplesComplete"); });

  const QuerySampleResponse* end = responses + response_count;

  // Notify first to unblock loadgen production ASAP.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    SampleMetadata* sample = reinterpret_cast<SampleMetadata*>(response->id);
    QueryMetadata* query = sample->query_metadata;
    query->NotifyOneSampleCompleted(timestamp);
  }

  // Log samples.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    SampleMetadata* sample = reinterpret_cast<SampleMetadata*>(response->id);
    QueryMetadata* query = sample->query_metadata;
    query->response_delegate->SampleComplete(sample, response, timestamp);
  }
}

struct DurationGeneratorNs {
  const PerfClock::time_point start;
  int64_t delta(PerfClock::time_point end) const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
        .count();
  }
};

// Right now, this is the only implementation of ResponseDelegate,
// but more will be coming soon.
template <TestScenario scenario, TestMode mode>
struct ResponseDelegateDetailed : public ResponseDelegate {
  std::atomic<size_t> queries_completed{0};

  void SampleComplete(SampleMetadata* sample, QuerySampleResponse* response,
                      PerfClock::time_point complete_begin_time) override {
    // Using a raw pointer here should help us hit the std::function
    // small buffer optimization code path when we aren't copying data.
    // For some reason, using std::unique_ptr<std::vector> wasn't moving
    // into the lambda; even with C++14.
    std::vector<uint8_t>* sample_data_copy = nullptr;
    if (mode == TestMode::AccuracyOnly) {
      // TODO: Verify accuracy with the data copied here.
      uint8_t* src_begin = reinterpret_cast<uint8_t*>(response->data);
      uint8_t* src_end = src_begin + response->size;
      sample_data_copy = new std::vector<uint8_t>(src_begin, src_end);
    }
    Log([sample, complete_begin_time, sample_data_copy](AsyncLog& log) {
      QueryMetadata* query = sample->query_metadata;
      DurationGeneratorNs sched{query->scheduled_time};

      if (scenario == TestScenario::Server) {
        // Trace the server scenario as a stacked graph via counter events.
        DurationGeneratorNs issued{query->issued_start_time};
        log.TraceCounterEvent("Latency", query->scheduled_time, "issue_delay",
                              sched.delta(query->issued_start_time),
                              "issue_to_done",
                              issued.delta(complete_begin_time));
      } else if (scenario != TestScenario::Offline) {
        // Disable tracing of each sample in offline mode, where visualizing
        // all samples overlapping isn't practical.
        log.TraceSample("Sample", sample->sequence_id, query->scheduled_time,
                        complete_begin_time, "sample_seq", sample->sequence_id,
                        "query_seq", query->sequence_id, "sample_idx",
                        sample->sample_index, "issue_start_ns",
                        sched.delta(query->issued_start_time), "complete_ns",
                        sched.delta(complete_begin_time));
      }

      if (sample_data_copy) {
        log.LogAccuracy(sample->sequence_id, sample->sample_index,
                        LogBinaryAsHexString{sample_data_copy});
        delete sample_data_copy;
      }

      // Record the latency at the end, since it will unblock the issuing
      // thread and potentially destroy the metadata being used above.
      QuerySampleLatency latency = sched.delta(complete_begin_time);
      log.RecordLatency(sample->sequence_id, latency);
    });
  }

  void QueryComplete() override {
    // We only need to track outstanding queries in the server scenario to
    // detect when the SUT has fallen too far behind.
    if (scenario == TestScenario::Server) {
      queries_completed.fetch_add(1, std::memory_order_relaxed);
    }
  }
};

// ScheduleDistribution templates by test scenario.
template <TestScenario scenario>
auto ScheduleDistribution(double qps) {
  return [period = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::duration<double>(1.0 / qps))](auto& /*gen*/) {
    return period;
  };
}

template <>
auto ScheduleDistribution<TestScenario::Server>(double qps) {
  // Poisson arrival process corresponds to exponentially distributed
  // interarrival times.
  return [dist = std::exponential_distribution<>(qps)](auto& gen) mutable {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

// SampleDistribution templates by test mode.
template <TestMode mode>
auto SampleDistribution(size_t sample_count, size_t samples_per_query) {
  return
      [sample_count, samples_per_query, i = size_t(0)](auto& /*gen*/) mutable {
        size_t result = i;
        i += samples_per_query;
        return result % sample_count;
      };
}

template <>
auto SampleDistribution<TestMode::PerformanceOnly>(
    size_t sample_count, size_t /*samples_per_query*/) {
  return [dist = std::uniform_int_distribution<>(0, sample_count - 1)](
             auto& gen) mutable { return dist(gen); };
}

template <TestScenario scenario, TestMode mode>
std::vector<QueryMetadata> GenerateQueries(
    const TestSettingsInternal& settings,
    const LoadableSampleSet& loaded_sample_set, SequenceGen* sequence_gen,
    ResponseDelegate* response_delegate) {
  auto tracer =
      MakeScopedTracer([](AsyncTrace& trace) { trace("GenerateQueries"); });

  auto& loaded_samples = loaded_sample_set.set;

  // Generate 2x more samples than we think we'll need given the expected
  // QPS. We should exit before issuing all queries.
  std::chrono::microseconds k2xTargetDuration = 2 * settings.target_duration;
  size_t min_queries = settings.min_query_count;

  // We should not exit early in accuracy mode.
  if (mode == TestMode::AccuracyOnly) {
    k2xTargetDuration = std::chrono::microseconds(0);
    // Integer truncation here is intentional.
    // For MultiStream, loaded samples is properly padded.
    // For Offline, we create a 'remainder' query at the end of this function.
    min_queries = loaded_samples.size() / settings.samples_per_query;
  }

  std::vector<QueryMetadata> queries;

  // Using the std::mt19937 pseudo-random number generator ensures a modicum of
  // cross platform reproducibility for trace generation.
  std::mt19937 sample_rng(settings.sample_index_rng_seed);
  std::mt19937 schedule_rng(settings.schedule_rng_seed);

  auto sample_distribution = SampleDistribution<mode>(
      loaded_sample_set.sample_distribution_end, settings.samples_per_query);
  auto schedule_distribution =
      ScheduleDistribution<scenario>(settings.target_qps);

  std::vector<QuerySampleIndex> samples(settings.samples_per_query);
  std::chrono::nanoseconds timestamp(0);
  while (timestamp <= k2xTargetDuration || queries.size() < min_queries) {
    if (scenario == TestScenario::MultiStream ||
        scenario == TestScenario::MultiStreamFree) {
      QuerySampleIndex sample_i = sample_distribution(sample_rng);
      for (auto& s : samples) {
        // Select contiguous samples in the MultiStream scenario.
        // This will not overflow, since GenerateLoadableSets adds padding at
        // the end of the loadable sets in the MultiStream scenario.
        // The padding allows the starting samples to be the same for each
        // query as the value of samples_per_query increases.
        s = loaded_samples[sample_i++];
      }
    } else {
      for (auto& s : samples) {
        s = loaded_samples[sample_distribution(sample_rng)];
      }
    }
    queries.emplace_back(samples, timestamp, response_delegate, sequence_gen);
    timestamp += schedule_distribution(schedule_rng);
  }

  // See if we need to create a "remainder" query for offline+accuracy to
  // ensure we issue all samples in loaded_samples. Offline doesn't pad
  // loaded_samples like MultiStream does.
  if (scenario == TestScenario::Offline && mode == TestMode::AccuracyOnly) {
    size_t remaining_samples =
        loaded_samples.size() % settings.samples_per_query;
    if (remaining_samples != 0) {
      samples.resize(remaining_samples);
      for (auto& s : samples) {
        s = loaded_samples[sample_distribution(sample_rng)];
      }
      queries.emplace_back(samples, timestamp, response_delegate, sequence_gen);
    }
  }

  LogDetail([count = queries.size(), spq = settings.samples_per_query,
             duration = timestamp.count()](AsyncDetail& detail) {
    detail("GeneratedQueries: ", "queries", count, "samples per query", spq,
           "duration", duration);
  });

  return queries;
}

// Template for the QueryScheduler. This base template should never be used
// since each scenario has its own specialization.
template <TestScenario scenario>
struct QueryScheduler {
  static_assert(scenario != scenario, "Unhandled TestScenario");
};

// SingleStream QueryScheduler
template <>
struct QueryScheduler<TestScenario::SingleStream> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto tracer = MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
    if (prev_query != nullptr) {
      prev_query->WaitForAllSamplesCompleted();
    }
    prev_query = next_query;

    auto now = PerfClock::now();
    next_query->scheduled_time = now;
    next_query->issued_start_time = now;
    return now;
  }

  QueryMetadata* prev_query = nullptr;
};

enum class MultiStreamFrequency { Fixed, Free };

// MultiStream QueryScheduler
template <>
struct QueryScheduler<TestScenario::MultiStream> {
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point start)
      : qps(settings.target_qps),
        max_async_queries(settings.max_async_queries),
        start_time(start) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    {
      prev_queries.push(next_query);
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
      if (prev_queries.size() > max_async_queries) {
        prev_queries.front()->WaitForAllSamplesCompleted();
        prev_queries.pop();
      }
    }

    {
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Scheduling"); });
      // TODO(brianderson): Skip ticks based on the query complete time,
      //     before the query snchronization + notification thread hop,
      //     rather than after.
      PerfClock::time_point now = PerfClock::now();
      auto i_period_old = i_period;
      PerfClock::time_point tick_time;
      do {
        i_period++;
        tick_time =
            start_time + SecondsToDuration<PerfClock::duration>(i_period / qps);
        Log([tick_time](AsyncLog& log) {
          log.TraceAsyncInstant("QueryInterval", 0, tick_time);
        });
      } while (tick_time < now);
      next_query->scheduled_intervals = i_period - i_period_old;
      next_query->scheduled_time = tick_time;
      std::this_thread::sleep_until(tick_time);
    }

    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  size_t i_period = 0;
  double qps;
  const size_t max_async_queries;
  PerfClock::time_point start_time;
  std::queue<QueryMetadata*> prev_queries;
};

// MultiStreamFree QueryScheduler
template <>
struct QueryScheduler<TestScenario::MultiStreamFree> {
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point /*start*/)
      : max_async_queries(settings.max_async_queries) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    bool schedule_time_needed = true;
    {
      prev_queries.push(next_query);
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
      if (prev_queries.size() > max_async_queries) {
        next_query->scheduled_time =
            prev_queries.front()->WaitForAllSamplesCompletedWithTimestamp();
        schedule_time_needed = false;
        prev_queries.pop();
      }
    }

    auto now = PerfClock::now();
    if (schedule_time_needed) {
      next_query->scheduled_time = now;
    }
    next_query->issued_start_time = now;
    return now;
  }

  const size_t max_async_queries;
  std::queue<QueryMetadata*> prev_queries;
};

// Server QueryScheduler
template <>
struct QueryScheduler<TestScenario::Server> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point start)
      : start(start) {}

  // TODO: Coalesce all queries whose scheduled timestamps have passed.
  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto tracer =
        MakeScopedTracer([](AsyncTrace& trace) { trace("Scheduling"); });

    auto scheduled_time = start + next_query->scheduled_delta;
    next_query->scheduled_time = scheduled_time;
    std::this_thread::sleep_until(scheduled_time);

    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  const PerfClock::time_point start;
};

// Offline QueryScheduler
template <>
struct QueryScheduler<TestScenario::Offline> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point start)
      : start(start) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    next_query->scheduled_time = start;
    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  const PerfClock::time_point start;
};

// Provides performance results that are independent of scenario
// and other context.
struct PerformanceResult {
  std::vector<QuerySampleLatency> latencies;
  size_t queries_issued;
  double max_latency;
  double final_query_scheduled_time;         // seconds from start.
  double final_query_issued_time;            // seconds from start.
  double final_query_all_samples_done_time;  // seconds from start.
};

// TODO: Templates for scenario and mode are overused, given the loadgen
//       no longer generates queries on the fly. Should we reduce the
//       use of templates?
template <TestScenario scenario, TestMode mode>
PerformanceResult IssueQueries(SystemUnderTest* sut,
                               const TestSettingsInternal& settings,
                               const LoadableSampleSet& loaded_sample_set,
                               SequenceGen* sequence_gen) {
  GlobalLogger().RestartLatencyRecording(sequence_gen->CurrentSampleId());
  ResponseDelegateDetailed<scenario, mode> response_logger;

  std::vector<QueryMetadata> queries = GenerateQueries<scenario, mode>(
      settings, loaded_sample_set, sequence_gen, &response_logger);

  size_t queries_issued = 0;
  // TODO: Replace the constant 5 below with a TestSetting.
  const double query_seconds_outstanding_threshold =
      5 * std::chrono::duration_cast<std::chrono::duration<double>>(
              settings.target_latency)
              .count();
  const size_t max_queries_outstanding =
      settings.target_qps * query_seconds_outstanding_threshold;

  const PerfClock::time_point start = PerfClock::now();
  PerfClock::time_point last_now = start;
  QueryScheduler<scenario> query_scheduler(settings, start);

  for (auto& query : queries) {
    auto tracer1 =
        MakeScopedTracer([](AsyncTrace& trace) { trace("SampleLoop"); });
    last_now = query_scheduler.Wait(&query);

    // Issue the query to the SUT.
    {
      auto tracer3 =
          MakeScopedTracer([](AsyncTrace& trace) { trace("IssueQuery"); });
      sut->IssueQuery(query.query_to_send);
    }

    queries_issued++;
    if (mode == TestMode::AccuracyOnly) {
      // TODO: Rate limit in accuracy mode.
      continue;
    }

    auto duration = (last_now - start);
    if (queries_issued >= settings.min_query_count &&
        duration > settings.min_duration) {
      LogDetail([](AsyncDetail& detail) {
        detail("Ending naturally: Minimum query count and test duration met.");
      });
      break;
    }
    if (settings.max_query_count != 0 &&
        queries_issued >= settings.max_query_count) {
      LogDetail([queries_issued](AsyncDetail& detail) {
        detail.Error("Ending early: Max query count reached.", "query_count",
                     queries_issued);
      });
      break;
    }
    if (settings.max_duration.count() != 0 &&
        duration > settings.max_duration) {
      LogDetail([duration](AsyncDetail& detail) {
        detail.Error("Ending early: Max test duration reached.", "duration_ns",
                     duration.count());
      });
      break;
    }
    if (scenario == TestScenario::Server) {
      size_t queries_outstanding =
          queries_issued -
          response_logger.queries_completed.load(std::memory_order_relaxed);
      if (queries_outstanding > max_queries_outstanding) {
        LogDetail([queries_issued, queries_outstanding](AsyncDetail& detail) {
          detail.Error("Ending early: Too many outstanding queries.", "issued",
                       queries_issued, "outstanding", queries_outstanding);
        });
        break;
      }
    }
    // TODO: Use GetMaxLatencySoFar here if we decide to have a hard latency
    //       limit.
  }

  // Let the SUT know it should not expect any more queries.
  sut->FlushQueries();

  // The offline scenario always only has a single query, so this check
  // doesn't apply.
  if (scenario != TestScenario::Offline && mode == TestMode::PerformanceOnly &&
      queries_issued >= queries.size()) {
    LogDetail([](AsyncDetail& detail) {
      detail.Error(
          "Ending early: Ran out of generated queries to issue before the "
          "minimum query count and test duration were reached.");
      detail(
          "Please update the relevant expected latency or target qps in the "
          "TestSettings so they are more accurate.");
    });
  }

  // Wait for tail queries to complete and collect all the latencies.
  // We have to keep the synchronization primitives alive until the SUT
  // is done with them.
  auto& final_query = queries[queries_issued - 1];
  const size_t expected_latencies = queries_issued * settings.samples_per_query;
  std::vector<QuerySampleLatency> latencies(
      GlobalLogger().GetLatenciesBlocking(expected_latencies));

  // Log contention counters after every test as a sanity check.
  GlobalLogger().LogContentionCounters();

  double max_latency =
      QuerySampleLatencyToSeconds(GlobalLogger().GetMaxLatencySoFar());
  double final_query_scheduled_time =
      DurationToSeconds(final_query.scheduled_delta);
  double final_query_issued_time =
      DurationToSeconds(final_query.issued_start_time - start);
  double final_query_all_samples_done_time =
      DurationToSeconds(final_query.all_samples_done_time - start);
  return PerformanceResult{std::move(latencies),
                           queries_issued,
                           max_latency,
                           final_query_scheduled_time,
                           final_query_issued_time,
                           final_query_all_samples_done_time};
}

// Takes the raw PerformanceResult and uses relevant context to determine
// how to interpret and report it.
struct PerformanceSummary {
  std::string sut_name;
  TestSettingsInternal settings;
  PerformanceResult pr;

  // Set by ProcessLatencies.
  size_t sample_count = 0;
  QuerySampleLatency latency_min = 0;
  QuerySampleLatency latency_max = 0;
  QuerySampleLatency latency_mean = 0;
  struct PercentileEntry {
    const double percentile;
    QuerySampleLatency value = 0;
  };
  // TODO: Make .90 a spec constant and have that affect relevant strings.
  PercentileEntry latency_target{.90};
  PercentileEntry latency_percentiles[5] = {{.50}, {.90}, {.95}, {.99}, {.999}};

  void ProcessLatencies();

  bool MinDurationMet(std::string* recommendation);
  bool MinQueriesMet();
  bool MinSamplesMet();
  bool HasPerfConstraints();
  bool PerfConstraintsMet(std::string* recommendation);
  void Log(AsyncSummary& summary);
};

void PerformanceSummary::ProcessLatencies() {
  if (pr.latencies.empty()) {
    return;
  }

  sample_count = pr.latencies.size();

  QuerySampleLatency accumulated_latency = 0;
  for (auto latency : pr.latencies) {
    accumulated_latency += latency;
  }
  latency_mean = accumulated_latency / sample_count;

  std::sort(pr.latencies.begin(), pr.latencies.end());

  latency_target.value = pr.latencies[sample_count * latency_target.percentile];
  latency_min = pr.latencies.front();
  latency_max = pr.latencies.back();
  for (auto& lp : latency_percentiles) {
    assert(lp.percentile >= 0.0);
    assert(lp.percentile < 1.0);
    lp.value = pr.latencies[sample_count * lp.percentile];
  }

  // Clear latencies since we are done processing them.
  pr.latencies = std::vector<QuerySampleLatency>();
}

bool PerformanceSummary::MinDurationMet(std::string* recommendation) {
  recommendation->clear();
  const double min_duration = DurationToSeconds(settings.min_duration);
  bool min_duration_met = (settings.scenario == TestScenario::Offline)
                              ? pr.max_latency > min_duration
                              : pr.final_query_issued_time >= min_duration;
  if (min_duration_met) {
    return true;
  }

  switch (settings.scenario) {
    case TestScenario::SingleStream:
      *recommendation =
          "Decrease the expected latency so the loadgen pre-generates more "
          "queries.";
      break;
    case TestScenario::MultiStream:
      *recommendation =
          "MultiStream should always meet the minimum duration. "
          "Please file a bug.";
      break;
    case TestScenario::MultiStreamFree:
      *recommendation =
          "Increase the target QPS so the loadgen pre-generates more queries.";
      break;
    case TestScenario::Server:
      *recommendation =
          "Increase the target QPS so the loadgen pre-generates more queries.";
      break;
    case TestScenario::Offline:
      *recommendation =
          "Increase expected QPS so the loadgen pre-generates more queries.";
      break;
  }
  return false;
}

bool PerformanceSummary::MinQueriesMet() {
  return pr.queries_issued >= settings.min_query_count;
}

bool PerformanceSummary::MinSamplesMet() {
  return sample_count >= settings.min_sample_count;
}

bool PerformanceSummary::HasPerfConstraints() {
  return settings.scenario == TestScenario::MultiStream ||
         settings.scenario == TestScenario::MultiStreamFree ||
         settings.scenario == TestScenario::Server;
}

bool PerformanceSummary::PerfConstraintsMet(std::string* recommendation) {
  recommendation->clear();
  bool perf_constraints_met = true;
  switch (settings.scenario) {
    case TestScenario::SingleStream:
      break;
    case TestScenario::MultiStream:
    case TestScenario::MultiStreamFree:
      // TODO: Finalize multi-stream performance targets with working group.
      ProcessLatencies();
      if (latency_target.value > settings.target_latency.count()) {
        *recommendation = "Reduce samples per query to improve latency.";
        perf_constraints_met = false;
      }
      break;
    case TestScenario::Server:
      ProcessLatencies();
      if (latency_target.value > settings.target_latency.count()) {
        *recommendation = "Reduce target QPS to improve latency.";
        perf_constraints_met = false;
      }
      break;
    case TestScenario::Offline:
      break;
  }
  return perf_constraints_met;
}

void PerformanceSummary::Log(AsyncSummary& summary) {
  ProcessLatencies();

  summary(
      "================================================\n"
      "MLPerf Results Summary\n"
      "================================================");
  summary("SUT name : ", sut_name);
  summary("Scenario : ", ToString(settings.scenario));
  summary("Mode     : ", ToString(settings.mode));

  switch (settings.scenario) {
    case TestScenario::SingleStream: {
      summary("90th percentile latency (ns) : ", latency_target.value);
      break;
    }
    case TestScenario::MultiStream: {
      summary("Samples per query : ", settings.samples_per_query);
      break;
    }
    case TestScenario::MultiStreamFree: {
      double samples_per_second = pr.queries_issued *
                                  settings.samples_per_query /
                                  pr.final_query_all_samples_done_time;
      summary("Samples per second : ", samples_per_second);
      break;
    }
    case TestScenario::Server: {
      // Subtract 1 from sample count since the start of the final sample
      // represents the open end of the time range: i.e. [begin, end).
      // This makes sense since:
      // a) QPS doesn't apply if there's only one sample; it's pure latency.
      // b) If you have precisely 1k QPS, there will be a sample exactly on
      //    the 1 second time point; but that would be the 1001th sample in
      //    the stream. Given the first 1001 queries, the QPS is
      //    1000 queries / 1 second.
      double qps_as_scheduled =
          (sample_count - 1) / pr.final_query_scheduled_time;
      summary("Scheduled samples per second : ",
              DoubleToString(qps_as_scheduled));
      break;
    }
    case TestScenario::Offline: {
      double samples_per_second = sample_count / pr.max_latency;
      summary("Samples per second: ", samples_per_second);
      break;
    }
  }

  std::string min_duration_recommendation;
  std::string perf_constraints_recommendation;

  bool min_duration_met = MinDurationMet(&min_duration_recommendation);
  bool min_queries_met = MinQueriesMet() && MinSamplesMet();
  bool perf_constraints_met =
      PerfConstraintsMet(&perf_constraints_recommendation);
  bool all_constraints_met =
      min_duration_met && min_queries_met && perf_constraints_met;
  summary("Result is : ", all_constraints_met ? "VALID" : "INVALID");
  if (HasPerfConstraints()) {
    summary("  Performance constraints satisfied : ",
            perf_constraints_met ? "Yes" : "NO");
  }
  summary("  Min duration satisfied : ", min_duration_met ? "Yes" : "NO");
  summary("  Min queries satisfied : ", min_queries_met ? "Yes" : "NO");

  if (!all_constraints_met) {
    summary("Recommendations:");
    if (!perf_constraints_met) {
      summary(" * " + perf_constraints_recommendation);
    }
    if (!min_duration_met) {
      summary(" * " + min_duration_recommendation);
    }
    if (!min_queries_met) {
      summary(
          " * The test exited early, before enough queries were issued.\n"
          "   See the detailed log for why this may have occurred.");
    }
  }

  summary(
      "\n"
      "================================================\n"
      "Additional Stats\n"
      "================================================");

  if (settings.scenario == TestScenario::SingleStream) {
    double qps_w_lg = (sample_count - 1) / pr.final_query_issued_time;
    double qps_wo_lg = 1 / QuerySampleLatencyToSeconds(latency_min);
    summary("QPS w/ loadgen overhead         : " + DoubleToString(qps_w_lg));
    summary("QPS w/o loadgen overhead        : " + DoubleToString(qps_wo_lg));
    summary("");
  } else if (settings.scenario == TestScenario::Server) {
    double qps_as_completed =
        (sample_count - 1) / pr.final_query_all_samples_done_time;
    summary("Completed samples per second    : ",
            DoubleToString(qps_as_completed));
    summary("");
  }

  summary("Min latency (ns)                : ", latency_min);
  summary("Max latency (ns)                : ", latency_max);
  summary("Mean latency (ns)               : ", latency_mean);
  for (auto& lp : latency_percentiles) {
    summary(
        DoubleToString(lp.percentile * 100) + " percentile latency (ns)   : ",
        lp.value);
  }

  summary(
      "\n"
      "================================================\n"
      "Test Parameters Used\n"
      "================================================");
  settings.LogSummary(summary);
}

void LoadSamplesToRam(QuerySampleLibrary* qsl,
                      const std::vector<QuerySampleIndex>& samples) {
  LogDetail([&samples](AsyncDetail& detail) {
    std::string set("\"[");
    for (auto i : samples) {
      set += std::to_string(i) + ",";
    }
    set.resize(set.size() - 1);
    set += "]\"";
    detail("Loading QSL : ", "set", set);
  });
  qsl->LoadSamplesToRam(samples);
}

template <TestScenario scenario>
void RunPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                        const TestSettingsInternal& settings,
                        const std::vector<LoadableSampleSet>& loadable_sets,
                        SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) { detail("Starting performance mode:"); });

  // Use first loadable set as the performance set.
  const LoadableSampleSet& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set.set);

  PerformanceResult pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, settings, performance_set, sequence_gen));

  sut->ReportLatencyResults(pr.latencies);

  LogSummary(
      [perf_summary = PerformanceSummary{sut->Name(), settings, std::move(pr)}](
          AsyncSummary& summary) mutable { perf_summary.Log(summary); });

  qsl->UnloadSamplesFromRam(performance_set.set);
}

template <TestScenario scenario>
void FindPeakPerformanceMode(
    SystemUnderTest* sut, QuerySampleLibrary* qsl,
    const TestSettingsInternal& settings,
    const std::vector<LoadableSampleSet>& loadable_sets,
    SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) {
    detail("Starting FindPeakPerformance mode:");
  });

  // Use first loadable set as the performance set.
  const LoadableSampleSet& performance_set = loadable_sets.front();

  LoadSamplesToRam(qsl, performance_set.set);

  TestSettingsInternal search_settings = settings;

  bool still_searching = true;
  while (still_searching) {
    PerformanceResult pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
        sut, search_settings, performance_set, sequence_gen));
    PerformanceSummary perf_summary{sut->Name(), search_settings,
                                    std::move(pr)};
  }

  qsl->UnloadSamplesFromRam(performance_set.set);
}

template <TestScenario scenario>
void RunAccuracyMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                     const TestSettingsInternal& settings,
                     const std::vector<LoadableSampleSet>& loadable_sets,
                     SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) { detail("Starting accuracy mode:"); });

  for (auto& loadable_set : loadable_sets) {
    {
      auto tracer = MakeScopedTracer(
          [count = loadable_set.set.size()](AsyncTrace& trace) {
            trace("LoadSamples", "count", count);
          });
      LoadSamplesToRam(qsl, loadable_set.set);
    }

    PerformanceResult pr(IssueQueries<scenario, TestMode::AccuracyOnly>(
        sut, settings, loadable_set, sequence_gen));

    {
      auto tracer = MakeScopedTracer(
          [count = loadable_set.set.size()](AsyncTrace& trace) {
            trace("UnloadSampes", "count", count);
          });
      qsl->UnloadSamplesFromRam(loadable_set.set);
    }
  }
}

// Routes runtime scenario requests to the corresponding instances of its
// templated mode functions.
struct RunFunctions {
  using Signature = void(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                         const TestSettingsInternal& settings,
                         const std::vector<LoadableSampleSet>& loadable_sets,
                         SequenceGen* sequence_gen);

  template <TestScenario compile_time_scenario>
  static RunFunctions GetCompileTime() {
    return {(RunAccuracyMode<compile_time_scenario>),
            (RunPerformanceMode<compile_time_scenario>),
            (FindPeakPerformanceMode<compile_time_scenario>)};
  }

  static RunFunctions Get(TestScenario run_time_scenario) {
    switch (run_time_scenario) {
      case TestScenario::SingleStream:
        return GetCompileTime<TestScenario::SingleStream>();
      case TestScenario::MultiStream:
        return GetCompileTime<TestScenario::MultiStream>();
      case TestScenario::MultiStreamFree:
        return GetCompileTime<TestScenario::MultiStreamFree>();
      case TestScenario::Server:
        return GetCompileTime<TestScenario::Server>();
      case TestScenario::Offline:
        return GetCompileTime<TestScenario::Offline>();
    }
    // We should not reach this point.
    assert(false);
    return GetCompileTime<TestScenario::SingleStream>();
  }

  const Signature& accuracy;
  const Signature& performance;
  const Signature& find_peak_performance;
};

// Generates random sets of samples in the QSL that we can load into RAM
// at the same time.
// Choosing samples randomly to go into a set naturally avoids biasing some
// samples to a particular set.
// TODO: Choosing bins randomly, rather than samples randomly, would avoid the
//       garbage collection logic, but we'd need to avoid later samples being
//       less likely to be in the smallest set. This may not be an important
//       requirement though.
std::vector<LoadableSampleSet> GenerateLoadableSets(
    QuerySampleLibrary* qsl, const TestSettingsInternal& settings) {
  auto tracer = MakeScopedTracer(
      [](AsyncTrace& trace) { trace("GenerateLoadableSets"); });

  std::vector<LoadableSampleSet> result;
  std::mt19937 qsl_rng(settings.qsl_rng_seed);

  // Generate indicies for all available samples in the QSL.
  const size_t qsl_total_count = qsl->TotalSampleCount();
  std::vector<QuerySampleIndex> samples(qsl_total_count);
  for (size_t i = 0; i < qsl_total_count; i++) {
    samples[i] = static_cast<QuerySampleIndex>(i);
  }

  // Randomize the order of the samples.
  std::shuffle(samples.begin(), samples.end(), qsl_rng);

  // Partition the samples into loadable sets.
  const size_t set_size = qsl->PerformanceSampleCount();
  const size_t set_padding =
      (settings.scenario == TestScenario::MultiStream ||
       settings.scenario == TestScenario::MultiStreamFree)
          ? settings.samples_per_query - 1
          : 0;
  std::vector<QuerySampleIndex> loadable_set;
  loadable_set.reserve(set_size + set_padding);

  for (auto s : samples) {
    loadable_set.push_back(s);
    if (loadable_set.size() == set_size) {
      result.push_back({std::move(loadable_set), set_size});
      loadable_set.clear();
      loadable_set.reserve(set_size + set_padding);
    }
  }

  if (!loadable_set.empty()) {
    // Copy the size since it will become invalid after the move.
    size_t loadable_set_size = loadable_set.size();
    result.push_back({std::move(loadable_set), loadable_set_size});
  }

  // Add padding for the multi stream scenario. Padding allows the
  // startings sample to be the same for all SUTs, independent of the value
  // of samples_per_query, while enabling samples in a query to be contiguous.
  for (auto& loadable_set : result) {
    auto& set = loadable_set.set;
    for (size_t i = 0; i < set_padding; i++) {
      // It's not clear in the spec if the STL deallocates the old container
      // before assigning, which would invalidate the source before the
      // assignment happens. Even though we should have reserved enough
      // elements above, copy the source first anyway since we are just moving
      // integers around.
      QuerySampleIndex p = set[i];
      set.push_back(p);
    }
  }

  return result;
}

struct LogOutputs {
  LogOutputs(const LogOutputSettings& output_settings,
             const std::string& test_date_time) {
    std::string prefix = output_settings.outdir;
    prefix += "/" + output_settings.prefix;
    if (output_settings.prefix_with_datetime) {
      prefix += test_date_time + "_";
    }
    const std::string& suffix = output_settings.suffix;

    summary_out.open(prefix + "summary" + suffix + ".txt");
    detail_out.open(prefix + "detail" + suffix + ".txt");
    accuracy_out.open(prefix + "accuracy" + suffix + ".json");
    trace_out.open(prefix + "trace" + suffix + ".json");
  }

  bool CheckOutputs() {
    bool all_ofstreams_good = true;
    if (!summary_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open summary file.";
    }
    if (!detail_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open detailed log file.";
    }
    if (!accuracy_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open accuracy log file.";
    }
    if (!trace_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open trace file.";
    }
    return all_ofstreams_good;
  }

  std::ofstream summary_out;
  std::ofstream detail_out;
  std::ofstream accuracy_out;
  std::ofstream trace_out;
};

void StartTest(SystemUnderTest* sut, QuerySampleLibrary* qsl,
               const TestSettings& requested_settings,
               const LogSettings& log_settings) {
  GlobalLogger().StartIOThread();

  const std::string test_date_time = CurrentDateTimeISO8601();

  LogOutputs log_outputs(log_settings.log_output, test_date_time);
  if (!log_outputs.CheckOutputs()) {
    return;
  }

  GlobalLogger().StartLogging(&log_outputs.summary_out, &log_outputs.detail_out,
                              &log_outputs.accuracy_out,
                              log_settings.log_output.copy_detail_to_stdout,
                              log_settings.log_output.copy_summary_to_stdout);
  GlobalLogger().StartNewTrace(&log_outputs.trace_out, PerfClock::now());

  LogLoadgenVersion();
  LogDetail([sut, qsl, test_date_time](AsyncDetail& detail) {
    detail("Date + time of test: ", test_date_time);
    detail("System Under Test (SUT) name: ", sut->Name());
    detail("Query Sample Library (QSL) name: ", qsl->Name());
    detail("QSL total size: ", qsl->TotalSampleCount());
    detail("QSL performance size: ", qsl->PerformanceSampleCount());
  });
  TestSettingsInternal sanitized_settings(requested_settings);
  sanitized_settings.LogAllSettings();

  std::vector<LoadableSampleSet> loadable_sets(
      GenerateLoadableSets(qsl, sanitized_settings));

  RunFunctions run_funcs = RunFunctions::Get(sanitized_settings.scenario);

  SequenceGen sequence_gen;
  switch (sanitized_settings.mode) {
    case TestMode::SubmissionRun:
      run_funcs.accuracy(sut, qsl, sanitized_settings, loadable_sets,
                         &sequence_gen);
      run_funcs.performance(sut, qsl, sanitized_settings, loadable_sets,
                            &sequence_gen);
      break;
    case TestMode::AccuracyOnly:
      run_funcs.accuracy(sut, qsl, sanitized_settings, loadable_sets,
                         &sequence_gen);
      break;
    case TestMode::PerformanceOnly:
      run_funcs.performance(sut, qsl, sanitized_settings, loadable_sets,
                            &sequence_gen);
      break;
    case TestMode::FindPeakPerformance:
      run_funcs.find_peak_performance(sut, qsl, sanitized_settings,
                                      loadable_sets, &sequence_gen);
      break;
  }

  // Stop tracing after logging so all logs are captured in the trace.
  GlobalLogger().StopLogging();
  GlobalLogger().StopTracing();
  GlobalLogger().StopIOThread();
}

}  // namespace mlperf
