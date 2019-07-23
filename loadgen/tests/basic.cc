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

#include <algorithm>
#include <future>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

#include "../loadgen.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"
#include "../test_settings.h"
#include "loadgen_test.h"

class SystemUnderTestBasic : public mlperf::QuerySampleLibrary,
                             public mlperf::SystemUnderTest {
 public:
  SystemUnderTestBasic(size_t total_sample_count,
                       size_t performance_sample_count)
      : total_sample_count_(total_sample_count),
        performance_sample_count_(performance_sample_count),
        samples_load_count_(total_sample_count, 0),
        samples_issue_count_(total_sample_count, 0) {}

  ~SystemUnderTestBasic() override = default;
  const std::string& Name() const override { return name_; }

  size_t TotalSampleCount() override { return total_sample_count_; }
  size_t PerformanceSampleCount() override { return performance_sample_count_; }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for (auto s : samples) {
      samples_load_count_.at(s)++;
      loaded_samples_.push(s);
    }
  }

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for (auto s : samples) {
      FAIL_IF(loaded_samples_.front() != s) &&
          FAIL_EXP(loaded_samples_.front()) && FAIL_EXP(s);
      loaded_samples_.pop();
      size_t prev_load_count = samples_load_count_.at(s)--;
      FAIL_IF(prev_load_count == 0) && FAIL_EXP(prev_load_count);
    }
  }

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    for (auto s : samples) {
      FAIL_IF(samples_load_count_.at(s.index) == 0) &&
          FAIL_MSG("Issued unloaded sample:") && FAIL_EXP(s.index);
      samples_issue_count_.at(s.index)++;
      issued_samples_.push_back(s.index);
      responses.push_back({s.id, 0, 0});
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }

  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

  std::vector<mlperf::QuerySampleIndex> GetIssuedSamples() {
    return std::move(issued_samples_);
  }

 private:
  std::string name_{"BasicSUT"};
  const size_t total_sample_count_;
  const size_t performance_sample_count_;
  std::vector<mlperf::QuerySampleIndex> issued_samples_;
  std::queue<mlperf::QuerySampleIndex> loaded_samples_;
  std::vector<size_t> samples_load_count_;
  std::vector<size_t> samples_issue_count_;
};

void TestAccuracyIncludesAllSamples(mlperf::TestScenario scenario) {
  const size_t kSamplesPerQuery = 4;
  const size_t kPerformanceSampleCount = kSamplesPerQuery * 16;
  // Make sure performance set size doesn't divide total size evenly.
  const size_t kSampleRemainder = 7;
  const size_t kTotalSampleCount =
      kPerformanceSampleCount * 32 + kSampleRemainder;
  const size_t kExpectedSets = kTotalSampleCount / kPerformanceSampleCount;

  SystemUnderTestBasic sut(kTotalSampleCount, kPerformanceSampleCount);

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = true;

  mlperf::TestSettings ts;
  ts.scenario = scenario;
  ts.mode = mlperf::TestMode::AccuracyOnly;
  ts.multi_stream_samples_per_query = kSamplesPerQuery;

  double qps = 1e3;
  ts.server_target_qps = qps;
  ts.multi_stream_target_qps = qps;

  mlperf::StartTest(&sut, &sut, ts, log_settings);

  std::vector<mlperf::QuerySampleIndex> issued_samples(sut.GetIssuedSamples());

  std::sort(issued_samples.begin(), issued_samples.end());

  FAIL_IF(issued_samples.size() < kTotalSampleCount) &&
      FAIL_EXP(issued_samples.size()) && FAIL_EXP(kTotalSampleCount);
  FAIL_IF(issued_samples.front() != 0) && FAIL_EXP(issued_samples.front());
  FAIL_IF(issued_samples.back() != kTotalSampleCount - 1) &&
      FAIL_EXP(issued_samples.back()) && FAIL_EXP(kTotalSampleCount);

  mlperf::QuerySampleIndex prev = -1;
  size_t discontinuities = 0;
  size_t dupes = 0;
  for (auto s : issued_samples) {
    if (s == prev) {
      dupes++;
    } else if (s - prev > 1) {
      discontinuities++;
    }
    prev = s;
  }

  FAIL_IF(discontinuities != 0) && FAIL_EXP(discontinuities);
  if (scenario == mlperf::TestScenario::MultiStream ||
      scenario == mlperf::TestScenario::MultiStreamFree) {
    FAIL_IF(dupes >= kSamplesPerQuery * kExpectedSets) && FAIL_EXP(dupes);
  } else {
    FAIL_IF(dupes != 0) && FAIL_EXP(dupes);
  }
}

REGISTER_TEST_ALL_SCENARIOS(AccuracyIncludesAllSamples,
                            TestAccuracyIncludesAllSamples);
