#include "c_api.h"

#include <string>

#include "../query_sample.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"

namespace mlperf {
namespace c {
namespace {

// Forwards SystemUnderTest calls to relevant callbacks.
class SystemUnderTestTrampoline : public SystemUnderTest {
 public:
  SystemUnderTestTrampoline(ClientData client_data, std::string name,
                            IssueQueryCallback issue_cb)
      : client_data_(client_data),
        name_(std::move(name)),
        issue_cb_(issue_cb) {}
  ~SystemUnderTestTrampoline() override = default;

  const std::string& Name() const override { return name_; }

  void IssueQuery(intptr_t query_id, QuerySample* samples,
                  size_t sample_count) override {
    (*issue_cb_)(client_data_, query_id, samples, sample_count);
  }

 private:
  ClientData client_data_;
  std::string name_;
  IssueQueryCallback issue_cb_;
};

}  // namespace

void* ConstructSUT(ClientData client_data, const char* name, size_t name_length,
                   IssueQueryCallback issue_cb) {
  SystemUnderTestTrampoline* sut = new SystemUnderTestTrampoline(
      client_data, std::string(name, name_length), issue_cb);
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
  QuerySampleLibraryTrampoline(ClientData client_data, std::string name,
                               IssueQueryCallback issue_cb)
      : client_data_(client_data),
        name_(std::move(name)),
        issue_cb_(issue_cb) {}
  ~QuerySampleLibraryTrampoline() override = default;

  const std::string& Name() const override { return name_; }


 const size_t TotalSampleCount() { return 0; }
 const size_t PerformanceSampleCount() { return 0; }

  void LoadSamplesToRam(QuerySample* samples,
                        size_t sample_count) override {}
  void UnloadSamplesFromRam(QuerySample* samples,
                            size_t sample_count) override {}
  void ResetAccuracyMetric() override {}
  void UpdateAccuracyMetric(uint64_t sample_index, void* response_data,
                            size_t response_size) override {}
  double GetAccuracyMetric() override {return 0;}
  std::string HumanReadableAccuracyMetric(double metric_value) override { return "Todo."; }

 private:
  ClientData client_data_;
  std::string name_;

  // TODO: QSL callbacks.
  IssueQueryCallback issue_cb_;
};

}  // namespace

void* ConstructQSL(ClientData client_data, const char* name, size_t name_length,
                   IssueQueryCallback issue_cb) {
  QuerySampleLibraryTrampoline* qsl = new QuerySampleLibraryTrampoline(
      client_data, std::string(name, name_length), issue_cb);
  return reinterpret_cast<void*>(qsl);
}

void DestroyQSL(void* qsl) {
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  delete qsl_cast;
}

// mlperf::c::StartTest just forwards to mlperf::StartTest after doing the
// proper cast.
void StartTest(void* sut, void* qsl, const TestSettings& settings) {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  mlperf::StartTest(sut_cast, qsl_cast, settings);
}

}  // namespace c
}  // namespace mlperf
