#include "loadgen.h"

#include <stdint.h>

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

struct ResponseDelegate {
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
                ResponseDelegate* response_delegate, uint64_t query_sequence_id,
                uint64_t* sample_sequence_id)
      : scheduled_delta(scheduled_delta),
        response_delegate(response_delegate),
        sequence_id(query_sequence_id),
        wait_count_(query_sample_indicies.size()) {
    samples_.reserve(query_sample_indicies.size());
    for (QuerySampleIndex qsi : query_sample_indicies) {
      samples_.push_back({this, (*sample_sequence_id)++, qsi});
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

  auto trace = MakeScopedTracer(
      [](AsyncLog& log) { log.ScopedTrace("QuerySamplesComplete"); });

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
// TODO: Versions that don't copy data.
// TODO: Versions that have less detailed logs.
// TODO: Versions that do a delayed notification.
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
      QuerySampleLatency latency = sched.delta(complete_begin_time);
      log.RecordLatency(sample->sequence_id, latency);
      // Disable tracing each sample in offline mode. Since thousands of
      // samples could be overlapping when visualized, it's not very useful.
      // TODO: Should we disable for cloud mode as well? Sufficiently
      // out-of-order processing could have lots of overlap too.
      if (scenario != TestScenario::Offline) {
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
    });
  }

  void QueryComplete() override {
    // We only need to track oustanding queries in the server scenario to
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
              std::chrono::duration<double>(1.0 / qps))](auto& gen) {
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
  return [sample_count, samples_per_query, i = size_t(0)](auto& gen) mutable {
    size_t result = i;
    i += samples_per_query;
    return result % sample_count;
  };
}

template <>
auto SampleDistribution<TestMode::PerformanceOnly>(size_t sample_count,
                                                   size_t samples_per_query) {
  return [dist = std::uniform_int_distribution<>(0, sample_count - 1)](
             auto& gen) mutable { return dist(gen); };
}

template <TestScenario scenario, TestMode mode>
std::vector<QueryMetadata> GenerateQueries(
    const TestSettingsInternal& settings,
    const std::vector<QuerySampleIndex>& loaded_samples,
    ResponseDelegate* response_delegate) {
  auto trace = MakeScopedTracer(
      [](AsyncLog& log) { log.ScopedTrace("GenerateQueries"); });

  // Generate 2x more samples than we think we'll need given the expected
  // QPS. We should exit before issuing all queries.
  std::chrono::microseconds k2xTargetDuration = 2 * settings.target_duration;
  size_t min_queries = settings.min_query_count;

  // We should not exit early in accuracy mode.
  if (mode == TestMode::AccuracyOnly) {
    k2xTargetDuration = std::chrono::microseconds(0);
    // Integer truncation here is intentional.
    // Loaded samples is properly padded in the MultiStream scenario.
    min_queries = loaded_samples.size() / settings.samples_per_query;
  }

  std::vector<QueryMetadata> queries;

  assert(scenario == settings.scenario);
  assert(mode == settings.mode);

  // Using the std::mt19937 pseudo-random number generator ensures a modicum of
  // cross platform reproducibility for trace generation.
  std::mt19937 sample_rng(settings.sample_index_rng_seed);
  std::mt19937 schedule_rng(settings.schedule_rng_seed);

  auto sample_distribution = SampleDistribution<mode>(
      loaded_samples.size(), settings.samples_per_query);
  auto schedule_distribution =
      ScheduleDistribution<scenario>(settings.target_qps);

  std::vector<QuerySampleIndex> samples(settings.samples_per_query);
  std::chrono::nanoseconds timestamp(0);
  uint64_t query_sequence_id = 0;
  uint64_t sample_sequence_id = 0;
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
    queries.emplace_back(samples, timestamp, response_delegate,
                         query_sequence_id, &sample_sequence_id);
    timestamp += schedule_distribution(schedule_rng);
    query_sequence_id++;
  }

  LogDetail([count = queries.size(), spq = settings.samples_per_query,
             duration = timestamp.count()](AsyncLog& log) {
    log.LogDetail("GeneratedQueries: ", "queries", count, "samples per query",
                  spq, "duration", duration);
  });

  return queries;
}

// Template for the QueryScheduler. This base template should never be used
// since each scenario has its own specialization.
template <TestScenario scenario>
struct QueryScheduler {
  static_assert(scenario != scenario, "Unhandled TestScenario");
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point) {}
  PerfClock::time_point Wait(QueryMetadata* next_query) {
    return PerfClock::now();
  }
};

// SingleStream QueryScheduler
template <>
struct QueryScheduler<TestScenario::SingleStream> {
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto trace =
        MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("Waiting"); });
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
      auto trace =
          MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("Waiting"); });
      if (prev_queries.size() > max_async_queries) {
        prev_queries.front()->WaitForAllSamplesCompleted();
        prev_queries.pop();
      }
    }

    {
      auto trace = MakeScopedTracer(
          [](AsyncLog& log) { log.ScopedTrace("Scheduling"); });
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
                 const PerfClock::time_point start)
      : max_async_queries(settings.max_async_queries) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    bool schedule_time_needed = true;
    {
      prev_queries.push(next_query);
      auto trace =
          MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("Waiting"); });
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
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point start)
      : start(start) {}

  // TODO: Coalesce all queries whose scheduled timestamps have passed.
  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto trace =
        MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("Scheduling"); });

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
  QueryScheduler(const TestSettingsInternal& settings,
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
PerformanceResult IssueQueries(
    SystemUnderTest* sut, const TestSettingsInternal& settings,
    const std::vector<QuerySampleIndex>& sample_set) {
  GlobalLogger().RestartLatencyRecording();
  ResponseDelegateDetailed<scenario, mode> response_logger;

  std::vector<QueryMetadata> queries =
      GenerateQueries<scenario, mode>(settings, sample_set, &response_logger);

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
    auto trace1 =
        MakeScopedTracer([](AsyncLog& log) { log.ScopedTrace("SampleLoop"); });
    last_now = query_scheduler.Wait(&query);

    // Issue the query to the SUT.
    {
      auto trace3 = MakeScopedTracer(
          [](AsyncLog& log) { log.ScopedTrace("IssueQuery"); });
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
      LogDetail([](AsyncLog& log) {
        log.LogDetail(
            "Ending naturally: Minimum query count and test duration met.");
      });
      break;
    }
    if (queries_issued >= settings.max_query_count) {
      LogError([queries_issued](AsyncLog& log) {
        log.LogDetail("Ending early: Max query count reached.", "query_count",
                      queries_issued);
      });
      break;
    }
    if (settings.max_duration.count() != 0 &&
        duration > settings.max_duration) {
      LogError([duration](AsyncLog& log) {
        log.LogDetail("Ending early: Max test duration reached.", "duration_ns",
                      duration.count());
      });
      break;
    }
    if (scenario == TestScenario::Server) {
      size_t queries_outstanding =
          queries_issued -
          response_logger.queries_completed.load(std::memory_order_relaxed);
      if (queries_outstanding > max_queries_outstanding) {
        LogError([queries_issued, queries_outstanding](AsyncLog& log) {
          log.LogDetail("Ending early: Too many oustanding queries.", "issued",
                        queries_issued, "outstanding", queries_outstanding);
        });
        break;
      }
    }
    // TODO: Use GetMaxLatencySoFar here if we decide to have a hard latency
    //       limit.
  }

  // The offline scenario always only has a single query, so this check
  // doesn't apply.
  if (scenario != TestScenario::Offline && mode == TestMode::PerformanceOnly &&
      queries_issued >= queries.size()) {
    LogError([](AsyncLog& log) {
      log.LogDetail(
          "Ending early: Ran out of generated queries to issue before the "
          "minimum query count and test duration were reached.");
      log.LogDetail(
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
  size_t sample_count;
  QuerySampleLatency latency_min;
  QuerySampleLatency latency_max;
  QuerySampleLatency latency_mean;
  struct PercentileEntry {
    const double percentile;
    QuerySampleLatency value = 0;
  };
  // TODO: Make .90 a spec constant and have that affect relevant strings.
  PercentileEntry latency_target{.90};
  PercentileEntry latency_percentiles[5] = {{.50}, {.90}, {.95}, {.99}, {.999}};

  void ProcessLatencies();

  bool MinDurationMet();
  bool MinQueriesMet();
  bool MinSamplesMet();
  bool HasPerfConstraints();
  bool PerfConstraintsMet();
  void Log(AsyncLog& log);
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

bool PerformanceSummary::MinDurationMet() {
  if (settings.scenario == TestScenario::Offline) {
    return pr.max_latency > DurationToSeconds(settings.min_duration);
  }
  return pr.final_query_issued_time >= DurationToSeconds(settings.min_duration);
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

bool PerformanceSummary::PerfConstraintsMet() {
  switch (settings.scenario) {
    case TestScenario::SingleStream:
      return true;
    case TestScenario::MultiStream:
    case TestScenario::MultiStreamFree: {
      // TODO: Finalize multi-stream performance targets with working group.
      ProcessLatencies();
      return latency_target.value <= settings.target_latency.count();
    }
    case TestScenario::Server: {
      ProcessLatencies();
      return latency_target.value <= settings.target_latency.count();
      break;
    }
    case TestScenario::Offline:
      return true;
  }
  assert(false);
  return false;
}

void PerformanceSummary::Log(AsyncLog& log) {
  ProcessLatencies();

  log.LogSummary(
      "================================================\n"
      "MLPerf Results Summary\n"
      "================================================");
  log.LogSummary("SUT name : ", sut_name);
  log.LogSummary("Scenario : ", ToString(settings.scenario));
  log.LogSummary("Mode     : ", ToString(settings.mode));

  switch (settings.scenario) {
    case TestScenario::SingleStream: {
      log.LogSummary("90th percentile latency (ns) : ", latency_target.value);
      break;
    }
    case TestScenario::MultiStream: {
      log.LogSummary("Samples per query : ", settings.samples_per_query);
      break;
    }
    case TestScenario::MultiStreamFree: {
      double ips = pr.queries_issued * settings.samples_per_query /
                   pr.final_query_all_samples_done_time;
      log.LogSummary("Inferences per second : ", ips);
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
      log.LogSummary("Scheduled QPS : ", qps_as_scheduled);
      break;
    }
    case TestScenario::Offline: {
      double qps = sample_count / pr.max_latency;
      log.LogSummary("QPS: ", qps);
      break;
    }
  }

  bool min_duration_met = MinDurationMet();
  bool min_queries_met = MinQueriesMet() && MinSamplesMet();
  bool perf_constraints_met = PerfConstraintsMet();
  bool all_constraints_met =
      min_duration_met && min_queries_met && perf_constraints_met;
  log.LogSummary("Result is : ", all_constraints_met ? "VALID" : "INVALID");
  if (HasPerfConstraints()) {
    log.LogSummary("  Performance constraints satisfied : ",
                   perf_constraints_met ? "Yes" : "NO");
  }
  log.LogSummary("  Min duration satisfied : ",
                 min_duration_met ? "Yes" : "NO");
  log.LogSummary("  Min queries satisfied : ", min_queries_met ? "Yes" : "NO");

  log.LogSummary(
      "\n"
      "================================================\n"
      "Additional Stats\n"
      "================================================");
  log.LogSummary("Min latency (ns)                : ", latency_min);
  log.LogSummary("Max latency (ns)                : ", latency_max);
  log.LogSummary("Mean latency (ns)               : ", latency_mean);
  for (auto& lp : latency_percentiles) {
    log.LogSummary(
        DoubleToString(lp.percentile * 100) + " percentile latency (ns)   : ",
        lp.value);
  }
  if (settings.scenario == TestScenario::SingleStream) {
    double qps_w_lg = (sample_count - 1) / pr.final_query_issued_time;
    double qps_wo_lg = 1 / QuerySampleLatencyToSeconds(latency_min);
    log.LogSummary("");
    log.LogSummary("QPS w/ loadgen overhead  : " + DoubleToString(qps_w_lg));
    log.LogSummary("QPS w/o loadgen overhead : " + DoubleToString(qps_wo_lg));
  }

  log.LogSummary(
      "\n"
      "================================================\n"
      "Test Parameters Used\n"
      "================================================");
  settings.LogSummary(log);
}

void LoadSamplesToRam(QuerySampleLibrary* qsl,
                      const std::vector<QuerySampleIndex>& samples) {
  LogDetail([samples](AsyncLog& log) {
    std::string set("\"[");
    for (auto i : samples) {
      set += std::to_string(i) + ",";
    }
    set.resize(set.size() - 1);
    set += "]\"";
    log.LogDetail("Loading QSL : ", "set", set);
  });
  qsl->LoadSamplesToRam(samples);
}

template <TestScenario scenario>
void RunPerformanceMode(
    SystemUnderTest* sut, QuerySampleLibrary* qsl,
    const TestSettingsInternal& settings,
    const std::vector<std::vector<QuerySampleIndex>>& loadable_sets) {
  LogDetail([](AsyncLog& log) { log.LogDetail("Starting performance mode:"); });

  // Use first loadable set as the performance set.
  const std::vector<QuerySampleIndex>& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set);

  PerformanceResult pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, settings, performance_set));

  sut->ReportLatencyResults(pr.latencies);

  Log([perf_summary = PerformanceSummary{sut->Name(), settings, std::move(pr)}](
          AsyncLog& log) mutable { perf_summary.Log(log); });

  qsl->UnloadSamplesFromRam(performance_set);
}

template <TestScenario scenario>
void FindPeakPerformanceMode(
    SystemUnderTest* sut, QuerySampleLibrary* qsl,
    const TestSettingsInternal& settings,
    const std::vector<std::vector<QuerySampleIndex>>& loadable_sets) {
  LogDetail([](AsyncLog& log) {
    log.LogDetail("Starting FindPeakPerformance mode:");
  });

  // Use first loadable set as the performance set.
  const std::vector<QuerySampleIndex>& performance_set = loadable_sets.front();

  LoadSamplesToRam(qsl, performance_set);

  TestSettingsInternal search_settings = settings;

  bool still_searching = true;
  while (still_searching) {
    PerformanceResult pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
        sut, search_settings, performance_set));
    PerformanceSummary perf_summary{sut->Name(), search_settings,
                                    std::move(pr)};
  }

  qsl->UnloadSamplesFromRam(performance_set);
}

template <TestScenario scenario>
void RunAccuracyMode(
    SystemUnderTest* sut, QuerySampleLibrary* qsl,
    const TestSettingsInternal& settings,
    const std::vector<std::vector<QuerySampleIndex>>& loadable_sets) {
  LogDetail(
      [](AsyncLog& log) { log.LogDetail("Starting verification mode:"); });

  for (auto& loadable_set : loadable_sets) {
    {
      auto trace =
          MakeScopedTracer([count = loadable_set.size()](AsyncLog& log) {
            log.ScopedTrace("LoadSamples", "count", count);
          });
      LoadSamplesToRam(qsl, loadable_set);
    }

    PerformanceResult pr(IssueQueries<scenario, TestMode::AccuracyOnly>(
        sut, settings, loadable_set));

    {
      auto trace =
          MakeScopedTracer([count = loadable_set.size()](AsyncLog& log) {
            log.ScopedTrace("UnloadSampes", "count", count);
          });
      qsl->UnloadSamplesFromRam(loadable_set);
    }
  }
}

// Routes runtime scenario requests to the corresponding instances of its
// mode functions.
struct RunFunctions {
  using Signature =
      void(SystemUnderTest* sut, QuerySampleLibrary* qsl,
           const TestSettingsInternal& settings,
           const std::vector<std::vector<QuerySampleIndex>>& loadable_sets);

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
std::vector<std::vector<QuerySampleIndex>> GenerateLoadableSets(
    QuerySampleLibrary* qsl, const TestSettingsInternal& settings) {
  constexpr float kGarbageCollectRatio = 0.5;
  constexpr size_t kUsedIndex = std::numeric_limits<size_t>::max();

  auto trace = MakeScopedTracer(
      [](AsyncLog& log) { log.ScopedTrace("GenerateLoadableSets"); });

  std::vector<std::vector<QuerySampleIndex>> result;
  std::mt19937 qsl_rng(settings.qsl_rng_seed);

  // Generate indicies for all available samples in the QSL.
  const size_t qsl_total_count = qsl->TotalSampleCount();
  std::vector<QuerySampleIndex> samples(qsl_total_count);
  for (size_t i = 0; i < qsl_total_count; i++) {
    samples[i] = static_cast<QuerySampleIndex>(i);
  }

  const size_t set_size = qsl->PerformanceSampleCount();
  const size_t set_padding =
      (settings.scenario == TestScenario::MultiStream ||
       settings.scenario == TestScenario::MultiStreamFree)
          ? settings.samples_per_query - 1
          : 0;
  std::vector<QuerySampleIndex> loadable_set;
  loadable_set.reserve(set_size + set_padding);
  size_t remaining_count = samples.size();
  size_t garbage_collect_count = remaining_count * kGarbageCollectRatio;
  std::uniform_int_distribution<> dist(0, remaining_count - 1);

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
      loadable_set.reserve(set_size + set_padding);
    }
    samples[candidate_index] = kUsedIndex;
    remaining_count--;

    // Garbage collect used indicies as probability of hitting one increases.
    if (garbage_collect_count != 0) {
      garbage_collect_count--;
    } else {
      RemoveValue(&samples, kUsedIndex);
      assert(remaining_count == samples.size());
      dist.param(
          std::uniform_int_distribution<>::param_type(0, remaining_count - 1));
      garbage_collect_count = remaining_count * kGarbageCollectRatio;
    }
  }

  if (!loadable_set.empty()) {
    result.push_back(std::move(loadable_set));
  }

  // Add padding for the multi stream scenario. Padding allows the
  // startings sample to be the same for all SUTs, independent of the value
  // of samples_per_query, while enabling samples in a query to be contiguous.
  for (auto& set : result) {
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

void StartTest(SystemUnderTest* sut, QuerySampleLibrary* qsl,
               const TestSettings& requested_settings) {
  GlobalLogger().StartIOThread();

  std::ofstream summary_out("mlperf_log_summary.txt");
  std::ofstream detail_out("mlperf_log_detail.txt");
  std::ofstream accuracy_out("mlperf_log_accuracy.json");
  GlobalLogger().StartLogging(&summary_out, &detail_out, &accuracy_out);
  std::ofstream trace_out("mlperf_trace.json");
  GlobalLogger().StartNewTrace(&trace_out, PerfClock::now());

  LogLoadgenVersion();
  LogDetail([sut, qsl](AsyncLog& log) {
    log.LogDetail("System Under Test (SUT) name: ", sut->Name());
    log.LogDetail("Query Sample Library (QSL) name: ", qsl->Name());
    log.LogDetail("QSL total size: ", qsl->TotalSampleCount());
    log.LogDetail("QSL performance size: ", qsl->PerformanceSampleCount());
  });
  TestSettingsInternal sanitized_settings(requested_settings);
  sanitized_settings.LogAllSettings();

  std::vector<std::vector<QuerySampleIndex>> loadable_sets(
      GenerateLoadableSets(qsl, sanitized_settings));

  RunFunctions run_funcs = RunFunctions::Get(sanitized_settings.scenario);

  switch (sanitized_settings.mode) {
    case TestMode::SubmissionRun:
      run_funcs.accuracy(sut, qsl, sanitized_settings, loadable_sets);
      run_funcs.performance(sut, qsl, sanitized_settings, loadable_sets);
      break;
    case TestMode::AccuracyOnly:
      run_funcs.accuracy(sut, qsl, sanitized_settings, loadable_sets);
      break;
    case TestMode::PerformanceOnly:
      run_funcs.performance(sut, qsl, sanitized_settings, loadable_sets);
      break;
    case TestMode::FindPeakPerformance:
      run_funcs.find_peak_performance(sut, qsl, sanitized_settings,
                                      loadable_sets);
      break;
  }

  // Stop tracing after logging so all logs are captured in the trace.
  GlobalLogger().StopLogging();
  GlobalLogger().StopTracing();
  GlobalLogger().StopIOThread();
}

}  // namespace mlperf
