#ifndef QUERY_SAMPLE_LIBRARY_H
#define QUERY_SAMPLE_LIBRARY_H

#include <memory>
#include <vector>

namespace mlperf {

// QueryLibraryMode indicates which library should be used.
enum class QueryLibraryMode {
  Verification = 0,  // Contains the entire library of queries used for
                     // accuracy verification.
  Performance = 1,   // A library of queries that can fit into memory and
                     // will be used for performance measurements.
};

// QuerySampleLibrary provides the interface to:
//  1) load queries from the query library and
//  2) calculate the accuracy of the query sample responses.
// Register instances of derived classes with the test harness via
// QslRegistry::Register().
class QuerySampleLibrary {
 public:
  virtual ~QuerySampleLibrary() {}

  // A human readable name for the model.
  virtual const std::string& Name() const = 0;

  // Initializes the model under test with the specified settings.
  // Returns false if the specified settings are not supported.
  virtual bool Initialize(QueryLibraryMode library_mode) = 0;

  // Returns a vector of sizes which correspond to the size of each
  // query sample in bytes. The size of the vector corresponds to the number
  // of samples in the library.
  // These sizes correspond to the original query sample sizes, before any SUT
  // preprocessing takes place.
  virtual std::vector<size_t> GetQuerySampleSizes() = 0;

  // Loads the requested query sample into the provided memory.
  // Returns the actual size of the loaded query sample.
  // A return value of 0 indicates error.
  virtual size_t LoadQuerySample(uint64_t sample_index, void* data,
                                 size_t max_size) = 0;

  // Starts an accuracy verification cycle.
  virtual void ResetAccuracyMetric() = 0;

  // Updates the accuracy metric, one query sample at a time.
  virtual void UpdateAccuracyMetric(uint64_t sample_index, void* response_data,
                                    size_t response_size) = 0;

  // Calculates and returns the current value for the accuracy metric.
  virtual double GetAccuracyMetric() = 0;

  // Returns a string that contains the metric suffixed by the proper units
  // and formatted with any relevant rounding.
  virtual std::string HumanReadableAccuracyMetric(double metric_value) = 0;
};

}  // namespace mlperf

#endif  // QUERY_SAMPLE_LIBRARY_H
