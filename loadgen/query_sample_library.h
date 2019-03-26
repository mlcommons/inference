#ifndef QUERY_SAMPLE_LIBRARY_H
#define QUERY_SAMPLE_LIBRARY_H

#include <memory>
#include <vector>

namespace mlperf {

// QuerySampleLibrary provides the interface to:
//  1) load queries from the query library and
//  2) calculate the accuracy of the query sample responses.
// A parallel struct, QuerySampleLibrarySettings, describes the number of
// samples in the library.
class QuerySampleLibrary {
 public:
  virtual ~QuerySampleLibrary() {}

  // A human readable name for the model.
  virtual const std::string& Name() const = 0;

  // Total number of samples in library.
  virtual const size_t TotalSampleCount() = 0;

  // The number of samples that are guaranteed to fit in RAM.
  virtual const size_t PerformanceSampleCount() = 0;

  // Loads the requested query samples into memory.
  // Paired with calls to UnloadSamplesFromRam.
  // The load generator will never load a currently loaded sample.
  virtual void LoadSamplesToRam(QuerySample* samples,
                                size_t sample_count) = 0;

  // Unloads the requested query samples from memory.
  // The load generator will never unload a currently unloaded sample.
  virtual void UnloadSamplesFromRam(QuerySample* samples,
                                    size_t sample_count) = 0;

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
