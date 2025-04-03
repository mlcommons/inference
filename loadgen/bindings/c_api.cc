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

#include "c_api.h"

#include <string>
#include <cassert>

#include "../loadgen.h"
#include "../query_sample.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"
#include "../test_settings.h"

namespace mlperf {
namespace c {
namespace {

// Forwards SystemUnderTest calls to relevant callbacks.
class SystemUnderTestTrampoline : public SystemUnderTest {
 public:
  SystemUnderTestTrampoline(ClientData client_data, std::string name,
                            IssueQueryCallback issue_cb,
                            FlushQueriesCallback flush_queries_cb)
      : client_data_(client_data),
        name_(std::move(name)),
        issue_cb_(issue_cb),
        flush_queries_cb_(flush_queries_cb) {}
  ~SystemUnderTestTrampoline() override = default;

  const std::string& Name() override { return name_; }

  void IssueQuery(const std::vector<QuerySample>& samples) override {
    (*issue_cb_)(client_data_, samples.data(), samples.size());
  }

  void FlushQueries() override { (*flush_queries_cb_)(); }

 private:
  ClientData client_data_;
  std::string name_;
  IssueQueryCallback issue_cb_;
  FlushQueriesCallback flush_queries_cb_;
};

}  // namespace

void* ConstructSUT(ClientData client_data, const char* name, size_t name_length,
                   IssueQueryCallback issue_cb,
                   FlushQueriesCallback flush_queries_cb) {
  SystemUnderTestTrampoline* sut = new SystemUnderTestTrampoline(
      client_data, std::string(name, name_length), issue_cb, flush_queries_cb);
  return reinterpret_cast<void*>(sut);
}

void DestroySUT(void* sut) {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  delete sut_cast;
}

namespace {

// Forwards QuerySampleLibrary calls to relevant callbacks.
class QuerySampleLibraryTrampoline : public QuerySampleLibrary {
 public:
  QuerySampleLibraryTrampoline(
      ClientData client_data, std::string name, size_t total_sample_count,
      size_t performance_sample_count,
      LoadSamplesToRamCallback load_samples_to_ram_cb,
      UnloadSamplesFromRamCallback unload_samples_from_ram_cb)
      : client_data_(client_data),
        name_(std::move(name)),
        total_sample_count_(total_sample_count),
        performance_sample_count_(performance_sample_count),
        load_samples_to_ram_cb_(load_samples_to_ram_cb),
        unload_samples_from_ram_cb_(unload_samples_from_ram_cb) {}
  ~QuerySampleLibraryTrampoline() override = default;

  const std::string& Name() override { return name_; }
  size_t TotalSampleCount() override { return total_sample_count_; }
  size_t PerformanceSampleCount() override { return performance_sample_count_; }
  size_t GroupSize(size_t i) override { return 1; }
  size_t GroupOf(size_t i) override { return i; }
  size_t NumberOfGroups() override { return total_sample_count_; }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {
    (*load_samples_to_ram_cb_)(client_data_, samples.data(), samples.size());
  }
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {
    (*unload_samples_from_ram_cb_)(client_data_, samples.data(),
                                   samples.size());
  }

 private:
  ClientData client_data_;
  std::string name_;
  size_t total_sample_count_;
  size_t performance_sample_count_;
  LoadSamplesToRamCallback load_samples_to_ram_cb_;
  UnloadSamplesFromRamCallback unload_samples_from_ram_cb_;
};

}  // namespace

void* ConstructQSL(ClientData client_data, const char* name, size_t name_length,
                   size_t total_sample_count, size_t performance_sample_count,
                   LoadSamplesToRamCallback load_samples_to_ram_cb,
                   UnloadSamplesFromRamCallback unload_samples_from_ram_cb) {
  QuerySampleLibraryTrampoline* qsl = new QuerySampleLibraryTrampoline(
      client_data, std::string(name, name_length), total_sample_count,
      performance_sample_count, load_samples_to_ram_cb,
      unload_samples_from_ram_cb);
  return reinterpret_cast<void*>(qsl);
}

void DestroyQSL(void* qsl) {
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  delete qsl_cast;
}

namespace {

// 
class GroupedQuerySampleLibraryTrampoline : public QuerySampleLibrary {
 public:
  GroupedQuerySampleLibraryTrampoline(
      ClientData client_data,
      std::string name,
      size_t performance_sample_count,
      LoadSamplesToRamCallback load_samples_to_ram_cb,
      UnloadSamplesFromRamCallback unload_samples_from_ram_cb,
      std::vector<size_t>& group_sizes)
      : name_(std::move(name)),
        performance_sample_count_(performance_sample_count),
        load_samples_to_ram_cb_(load_samples_to_ram_cb),
        unload_samples_from_ram_cb_(unload_samples_from_ram_cb) {

      total_sample_count_ = 0;

      for(size_t i = 0; i < group_sizes.size(); i++){
        group_sizes_.push_back(group_sizes[i]);
        total_sample_count_ += group_sizes[i];
        for(size_t j = 0; j < group_sizes[i]; j++){
          group_idx_.push_back(i);
        }
      }
    }
  ~GroupedQuerySampleLibraryTrampoline() override = default;

  const std::string& Name() override { return name_; }
  size_t TotalSampleCount() override { return total_sample_count_; }
  size_t PerformanceSampleCount() override { return performance_sample_count_; }
  size_t GroupSize(size_t i) override { return group_sizes_[i]; }
  size_t GroupOf(size_t i) override { return group_idx_[i]; }
  size_t NumberOfGroups() override { return group_sizes_.size(); }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {
    (*load_samples_to_ram_cb_)(client_data_, samples.data(), samples.size());
  }
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {
    (*unload_samples_from_ram_cb_)(client_data_, samples.data(),
                                   samples.size());
  }

 private:
  std::string name_;
  ClientData client_data_;
  std::vector<size_t> group_sizes_;
  std::vector<size_t> group_idx_;
  size_t total_sample_count_;
  size_t performance_sample_count_;
  LoadSamplesToRamCallback load_samples_to_ram_cb_;
  UnloadSamplesFromRamCallback unload_samples_from_ram_cb_;
};

} // namespace

void* ConstructGroupedQSL(ClientData client_data, const char* name, size_t name_length,
                   size_t total_sample_count, size_t performance_sample_count,
                   LoadSamplesToRamCallback load_samples_to_ram_cb,
                   UnloadSamplesFromRamCallback unload_samples_from_ram_cb,
                   std::vector<size_t>& group_sizes) {
  GroupedQuerySampleLibraryTrampoline* qsl = new GroupedQuerySampleLibraryTrampoline(
      client_data, std::string(name, name_length),
      performance_sample_count, load_samples_to_ram_cb,
      unload_samples_from_ram_cb, group_sizes);
  return reinterpret_cast<void*>(qsl);
}

void DestroyGroupedQSL(void* qsl) {
  GroupedQuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<GroupedQuerySampleLibraryTrampoline*>(qsl);
  delete qsl_cast;
}


// mlperf::c::StartTest just forwards to mlperf::StartTest after doing the
// proper cast.
void StartTest(void* sut, void* qsl, const TestSettings& settings,
               const std::string& audit_config_filename = "audit.config") {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  LogSettings default_log_settings;
  mlperf::StartTest(sut_cast, qsl_cast, settings, default_log_settings,
                    audit_config_filename);
}

void StartTestWithGroupedQSL(void* sut, void* qsl, const TestSettings& settings,
               const std::string& audit_config_filename = "audit.config") {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  GroupedQuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<GroupedQuerySampleLibraryTrampoline*>(qsl);
  assert(settings.use_grouped_qsl);
  LogSettings default_log_settings;
  mlperf::StartTest(sut_cast, qsl_cast, settings, default_log_settings,
                    audit_config_filename);
}

void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count) {
  mlperf::QuerySamplesComplete(responses, response_count);
}

void QuerySamplesCompleteResponseCb(QuerySampleResponse* responses,
                                    size_t response_count,
                                    ResponseCallback response_cb,
                                    ClientData client_data) {
  mlperf::QuerySamplesComplete(
      responses, response_count,
      [client_data, response_cb](QuerySampleResponse* response) {
        response_cb(client_data, response);
      });
}

void FirstTokenComplete(QuerySampleResponse* responses, size_t response_count) {
  mlperf::FirstTokenComplete(responses, response_count);
}

void FirstTokenCompleteResponseCb(QuerySampleResponse* responses,
                                  size_t response_count,
                                  ResponseCallback response_cb,
                                  ClientData client_data) {
  mlperf::FirstTokenComplete(
      responses, response_count,
      [client_data, response_cb](QuerySampleResponse* response) {
        response_cb(client_data, response);
      });
}

void RegisterIssueQueryThread() { mlperf::RegisterIssueQueryThread(); }

}  // namespace c
}  // namespace mlperf
