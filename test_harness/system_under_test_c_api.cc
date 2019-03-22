#include "system_under_test_c_api.h"

#include <string>

#include "system_under_test.h"

namespace mlperf {
namespace c {
namespace {

// Forwards SystemUnderTest calls to relevant callbacks.
class SystemUnderTestTrampoline : public SystemUnderTest {
 public:
  SystemUnderTestTrampoline(ClientData client_data, std::string name,
                            UntimedWarmUpCallback warm_up_cb,
                            AllocateQuerySampleCallback allocate_cb,
                            FreeQuerySampleCallback free_cb,
                            PreprocessQuerySampleCallback preprocess_cb,
                            IssueQueryCallback issue_cb)
      : client_data_(client_data),
        name_(std::move(name)),
        warm_up_cb_(warm_up_cb),
        allocate_cb_(allocate_cb),
        free_cb_(free_cb),
        preprocess_cb_(preprocess_cb),
        issue_cb_(issue_cb) {}
  ~SystemUnderTestTrampoline() override = default;

  std::string Name() override { return name_; }
  void UntimedWarmUp() override { (*warm_up_cb_)(client_data_); }
  void* AllocateQuerySample(size_t size_in_bytes) override {
    return (*allocate_cb_)(client_data_, size_in_bytes);
  }
  void FreeQuerySample(void* mem) override { (*free_cb_)(client_data_, mem); }
  void PreprocessQuerySample(const void* source_data, const size_t source_size,
                             void** processed_data,
                             size_t* processed_size) override {
    (*preprocess_cb_)(client_data_, source_data, source_size, processed_data,
                      processed_size);
  }
  void IssueQuery(intptr_t query_id, QuerySample* samples,
                  size_t sample_count) override {
    (*issue_cb_)(client_data_, query_id, samples, sample_count);
  }

 private:
  ClientData client_data_;
  std::string name_;
  UntimedWarmUpCallback warm_up_cb_;
  AllocateQuerySampleCallback allocate_cb_;
  FreeQuerySampleCallback free_cb_;
  PreprocessQuerySampleCallback preprocess_cb_;
  IssueQueryCallback issue_cb_;
};

}  // namespace

void* ConstructSUT(ClientData client_data, const char* name, size_t name_length,
                   UntimedWarmUpCallback warm_up_cb,
                   AllocateQuerySampleCallback allocate_cb,
                   FreeQuerySampleCallback free_cb,
                   PreprocessQuerySampleCallback preprocess_cb,
                   IssueQueryCallback issue_cb) {
  SystemUnderTestTrampoline* sut = new SystemUnderTestTrampoline(
      client_data, std::string(name, name_length), warm_up_cb, allocate_cb,
      free_cb, preprocess_cb, issue_cb);
  return reinterpret_cast<void*>(sut);
}

void DestroySUT(void* sut) {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  delete sut_cast;
}

// mlperf::c::StartTest just forwards to mlperf::StartTest after doing the
// proper cast.
void StartTest(void* sut, const TestSettings& settings) {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  mlperf::StartTest(sut_cast, settings);
}

}  // namespace c
}  // namespace mlperf
