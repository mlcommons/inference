#include "../loadgen.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"
#include "../test_settings.h"

class SystemUnderTestNull : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestNull() = default;
  ~SystemUnderTestNull() override = default;
  const std::string& Name() const override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    for (auto s : samples) {
      responses.push_back({s.id, 0, 0});
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

 private:
  std::string name_{"NullSUT"};
};

class QuerySampleLibraryNull : public mlperf::QuerySampleLibrary {
 public:
  QuerySampleLibraryNull() = default;
  ~QuerySampleLibraryNull() = default;
  const std::string& Name() const override { return name_; }

  const size_t TotalSampleCount() override { return 1024 * 1024; }

  const size_t PerformanceSampleCount() override { return 1024; }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    return;
  }

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    return;
  }

 private:
  std::string name_{"NullQSL"};
};

int main(int argc, char* argv[]) {
  SystemUnderTestNull null_sut;
  QuerySampleLibraryNull null_qsl;

  mlperf::TestSettings test_settings;
  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = true;

  mlperf::StartTest(&null_sut, &null_qsl, test_settings, log_settings);
  return 0;
}
