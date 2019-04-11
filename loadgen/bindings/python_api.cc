#ifndef PYTHON_BINDINGS_H
#define PYTHON_BINDINGS_H

#include <functional>

#include "third_party/pybind/include/pybind11/functional.h"
#include "third_party/pybind/include/pybind11/pybind11.h"
#include "third_party/pybind/include/pybind11/stl.h"
#include "third_party/pybind/include/pybind11/stl_bind.h"

#include "../loadgen.h"
#include "../query_sample.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"


namespace mlperf {

namespace {

using IssueQueryCallback = std::function<void(QueryId, std::vector<QuerySample>)>;

// Forwards SystemUnderTest calls to relevant callbacks.
class SystemUnderTestTrampoline : public SystemUnderTest {
 public:
  SystemUnderTestTrampoline(std::string name, IssueQueryCallback issue_cb)
      : name_(std::move(name)),
        issue_cb_(issue_cb) {}
  ~SystemUnderTestTrampoline() override = default;

  const std::string& Name() const override { return name_; }

  void IssueQuery(QueryId query_id, QuerySample* samples,
                  size_t sample_count) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    // TODO: Get rid of copies.
    issue_cb_(query_id, std::vector<QuerySample>(samples, samples+sample_count));
  }

 private:
  std::string name_;
  IssueQueryCallback issue_cb_;
};

using LoadSamplesToRamCallback =
    std::function<void(std::vector<QuerySample>)>;
using UnloadSamplesFromRamCallback =
    std::function<void(std::vector<QuerySample>)>;

// Forwards QuerySampleLibrary calls to relevant callbacks.
class QuerySampleLibraryTrampoline : public QuerySampleLibrary {
 public:
  QuerySampleLibraryTrampoline(std::string name,
                               size_t total_sample_count, size_t performance_sample_count,
                               LoadSamplesToRamCallback load_samples_to_ram_cb,
                               UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb)
      : name_(std::move(name)),
        total_sample_count_(total_sample_count),
        performance_sample_count_(performance_sample_count),
        load_samples_to_ram_cb_(load_samples_to_ram_cb),
        unload_samlpes_from_ram_cb_(unload_samlpes_from_ram_cb) {}
  ~QuerySampleLibraryTrampoline() override = default;

  const std::string& Name() const override { return name_; }
  const size_t TotalSampleCount() { return total_sample_count_; }
  const size_t PerformanceSampleCount() { return performance_sample_count_; }

  void LoadSamplesToRam(QuerySample* samples,
                        size_t sample_count) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    // TODO: Get rid of copies.
    load_samples_to_ram_cb_(
          std::vector<QuerySample>(samples, samples+sample_count));
  }
  void UnloadSamplesFromRam(QuerySample* samples,
                            size_t sample_count) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    // TODO: Get rid of copies.
    unload_samlpes_from_ram_cb_(
          std::vector<QuerySample>(samples, samples+sample_count));
  }

  // TODO(brianderson): Accuracy Metric API.
  void ResetAccuracyMetric() override {}
  void UpdateAccuracyMetric(uint64_t sample_index, void* response_data,
                            size_t response_size) override {}
  double GetAccuracyMetric() override {return 0;}
  std::string HumanReadableAccuracyMetric(double metric_value) override {
    return "TODO: AccuracyMetric";
  }

 private:
  std::string name_;
  size_t total_sample_count_;
  size_t performance_sample_count_;
  LoadSamplesToRamCallback load_samples_to_ram_cb_;
  UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb_;
};

}  // namespace

namespace py {
  void* ConstructSUT(IssueQueryCallback issue_cb) {
    SystemUnderTestTrampoline* sut = new SystemUnderTestTrampoline("PySUT", issue_cb);
    return reinterpret_cast<void*>(sut);
  }

  void DestroySUT(void* sut) {
    SystemUnderTestTrampoline* sut_cast =
        reinterpret_cast<SystemUnderTestTrampoline*>(sut);
    delete sut_cast;
  }

  void* ConstructQSL(size_t total_sample_count, size_t performance_sample_count,
                     LoadSamplesToRamCallback load_samples_to_ram_cb,
                     UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb) {
    QuerySampleLibraryTrampoline* qsl = new QuerySampleLibraryTrampoline(
        "PyQSL", total_sample_count, performance_sample_count,
        load_samples_to_ram_cb, unload_samlpes_from_ram_cb);
    return reinterpret_cast<void*>(qsl);
  }

  void DestroyQSL(void* qsl) {
    QuerySampleLibraryTrampoline* qsl_cast =
        reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
    delete qsl_cast;
  }

  // Parses commandline.
  void StartTest(void* sut, void* qsl, mlperf::TestSettings settings) {
    pybind11::gil_scoped_release gil_releaser;
    SystemUnderTestTrampoline* sut_cast =
        reinterpret_cast<SystemUnderTestTrampoline*>(sut);
    QuerySampleLibraryTrampoline* qsl_cast =
        reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
    mlperf::StartTest(sut_cast, qsl_cast, settings);
  }

  // TODO: Get rid of copies.
  void QueryComplete(QueryId query_id,
                     std::vector<QuerySampleResponse> responses) {
    pybind11::gil_scoped_release gil_releaser;
    mlperf::QueryComplete(query_id, responses.data(), responses.size());
  }

PYBIND11_MODULE(mlperf_loadgen, m) {
  m.doc() = "MLPerf Inference load generator.";

  pybind11::enum_<mlperf::TestScenario>(m, "TestScenario").
    value("SingleStream", mlperf::TestScenario::SingleStream).
    value("MultiStream", mlperf::TestScenario::MultiStream).
    value("Cloud", mlperf::TestScenario::Cloud).
    value("Offline", mlperf::TestScenario::Offline);

  pybind11::enum_<mlperf::TestMode>(m, "TestMode").
    value("SubmissionRun", mlperf::TestMode::SubmissionRun).
    value("AccuracyOnly", mlperf::TestMode::AccuracyOnly).
    value("PerformanceOnly", mlperf::TestMode::PerformanceOnly).
    value("SearchForQps", mlperf::TestMode::SearchForQps);

  pybind11::class_<mlperf::TestSettings>(m, "TestSettings").
    def(pybind11::init<>()).
    def(pybind11::init<
        mlperf::TestScenario, mlperf::TestMode, int, double, uint64_t>()).
    def_readwrite("scenario", &TestSettings::scenario).
    def_readwrite("mode", &TestSettings::mode).
    def_readwrite("samples_per_query", &TestSettings::samples_per_query).
    def_readwrite("target_qps", &TestSettings::target_qps).
    def_readwrite("target_latency_ns", &TestSettings::target_latency_ns);

  pybind11::class_<mlperf::QuerySampleResponse>(m, "QuerySampleResponse")
      .def(pybind11::init<>())
      .def(pybind11::init<intptr_t, size_t>())
      .def_readwrite("data", &mlperf::QuerySampleResponse::data)
      .def_readwrite("size", &mlperf::QuerySampleResponse::size);

  // TODO: Use PYBIND11_MAKE_OPAQUE for the following vector types.
  pybind11::bind_vector<std::vector<mlperf::QuerySample>>(m, "VectorQuerySample");
  pybind11::bind_vector<std::vector<mlperf::QuerySampleResponse>>(m, "VectorQuerySampleResponse");

  m.def("ConstructSUT", &mlperf::py::ConstructSUT,
        pybind11::return_value_policy::reference,
        "Construct the system under test.");
  m.def("DestroySUT", &mlperf::py::DestroySUT,
        "Destroy the object created by ConstructSUT.");

  m.def("ConstructQSL", &mlperf::py::ConstructQSL,
        pybind11::return_value_policy::reference,
        "Construct the query sample library.");
  m.def("DestroyQSL", &mlperf::py::DestroyQSL,
        "Destroy the object created by ConstructQSL.");

  m.def("StartTest", &mlperf::py::StartTest,
        "Run tests on a SUT created by ConstructSUT() with the provided QSL.");
  m.def("QueryComplete", &mlperf::py::QueryComplete,
        "Called by the SUT to indicate the query_id from the"
        "IssueQuery callback is finished.");
}

}  // namespace py
}  // namespace mlperf

#endif  // PYTHON_BINDINGS_H
