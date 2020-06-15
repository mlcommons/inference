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

/// \file
/// \brief TODO

#ifndef MLPERF_LOADGEN_ISSUE_QUERY_CONTROLLER_H_
#define MLPERF_LOADGEN_ISSUE_QUERY_CONTROLLER_H_

#include "loadgen.h"
#include "logging.h"
#include "query_sample.h"
#include "system_under_test.h"
#include "test_settings_internal.h"
#include "utils.h"

#include <stdint.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace mlperf {

namespace loadgen {

struct SampleMetadata;
class QueryMetadata;

/// \brief Every query and sample within a call to StartTest gets a unique
/// sequence id for easy cross reference, and a random number which is used to
/// determine accuracy logging when it is enabled.
struct SequenceGen {
  uint64_t NextQueryId() { return query_id++; }
  uint64_t NextSampleId() { return sample_id++; }
  uint64_t CurrentSampleId() { return sample_id; }
  double NextAccLogRng() { return accuracy_log_dist(accuracy_log_rng); }
  void InitAccLogRng(uint64_t accuracy_log_rng_seed) {
    accuracy_log_rng = std::mt19937(accuracy_log_rng_seed);
  }

 private:
  uint64_t query_id = 0;
  uint64_t sample_id = 0;
  std::mt19937 accuracy_log_rng;
  std::uniform_real_distribution<double> accuracy_log_dist =
      std::uniform_real_distribution<double>(0, 1);
};

/// \brief An interface for a particular scenario + mode to implement for
/// extended hanlding of sample completion.
struct ResponseDelegate {
  virtual ~ResponseDelegate() = default;
  virtual void SampleComplete(SampleMetadata*, QuerySampleResponse*,
                              PerfClock::time_point) = 0;
  virtual void QueryComplete() = 0;
  std::atomic<size_t> queries_completed{0};
};

/// \brief Used by the loadgen to coordinate response data and completion.
struct SampleMetadata {
  QueryMetadata* query_metadata;
  uint64_t sequence_id;
  QuerySampleIndex sample_index;
  double accuracy_log_val;
};

/// \brief Maintains data and timing info for a query and all its samples.
class QueryMetadata {
 public:
  QueryMetadata(const std::vector<QuerySampleIndex>& query_sample_indices,
                std::chrono::nanoseconds scheduled_delta,
                ResponseDelegate* response_delegate, SequenceGen* sequence_gen);
  QueryMetadata(QueryMetadata&& src);

  void NotifyOneSampleCompleted(PerfClock::time_point timestamp);

  void WaitForAllSamplesCompleted();

  PerfClock::time_point WaitForAllSamplesCompletedWithTimestamp();

  // When server_coalesce_queries is set to true in Server scenario, we
  // sometimes coalesce multiple queries into one query. This is done by moving
  // the other query's sample into current query, while maintaining their
  // original scheduled_time.
  void CoalesceQueries(QueryMetadata* queries, size_t first, size_t last,
                       size_t stride);

  void Decoalesce();

 public:
  std::vector<QuerySample> query_to_send;
  const std::chrono::nanoseconds scheduled_delta;
  ResponseDelegate* const response_delegate;
  const uint64_t sequence_id;

  // Performance information.

  size_t scheduled_intervals = 0;  // Number of intervals between queries, as
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

struct IssueQueryState {
  SystemUnderTest* sut;
  std::vector<QueryMetadata>* queries;
  ResponseDelegate* response_delegate;
  const TestSettingsInternal* settings;
  TestMode mode;
  std::chrono::system_clock::time_point start_for_power;
  PerfClock::time_point start_time;
  bool ran_out_of_generated_queries;
  size_t queries_issued;
  size_t expected_latencies;
  std::mutex mtx;
};

/// \brief TODO
class IssueQueryController {
 public:
  static IssueQueryController& GetInstance();
  IssueQueryController(IssueQueryController const&) = delete;
  void operator=(IssueQueryController const&) = delete;
  void RegisterThread();
  void SetNumThreads(size_t n);
  template <TestScenario scenario>
  void StartIssueQueries(IssueQueryState* s);
  void EndThreads();

 private:
  /// \note Hide constructor so that it cannot be explicitly created.
  IssueQueryController() {}
  template <TestScenario scenario, bool multi_thread>
  void IssueQueriesInternal(size_t query_stride, size_t thread_idx);

  IssueQueryState* state;
  std::mutex mtx;
  std::condition_variable cond_var;
  std::vector<std::thread::id> thread_ids;
  size_t num_threads{0};
  bool issuing{false};
  std::vector<bool> thread_complete;
  bool end_test{false};
};

}  // namespace loadgen

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_ISSUE_QUERY_CONTROLLER_H_
