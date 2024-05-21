#ifndef _THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_UTIL_ROC_METRICS_ROC_METRICS_H_
#define _THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_UTIL_ROC_METRICS_ROC_METRICS_H_

#include <Python.h>

#include <vector>

#include "absl/types/span.h"
#include "numpy/core/include/numpy/arrayobject.h"

namespace rocmetrics {

// Computes ROC-based metrics for single-class binary classifiers.

struct PredictElem {
  float score;
  int target;
};

struct RocData {
  std::vector<int> tps;
  std::vector<int> fps;
};

class RocMetrics {
public:
  // Create an RocMetrics object with predictions py_scores and targets
  // py_objects are numpy array objects.
  explicit RocMetrics(PyObject *py_scores, PyObject *py_targets);
  // RocMetrics is designed to be used in a roughly singleton fashion.
  RocMetrics(const RocMetrics &other) = delete;
  RocMetrics &operator=(const RocMetrics &other) = delete;

  // Computes the area under the ROC curve.
  float ComputeRocAuc();

  // Computes the raw ROC vectors: TPS and FPS. These can be normalized to TPR
  // and FPR via element-wise division with the final value, vec.back().
  // The algorithm is a combination of a cumulative sum on the targets, a
  // filtering operation, and an averaging of duplicated score contributions.
  RocData BinaryRoc() const;

private:
  // Container for the full set of predictions and targets.
  std::vector<PredictElem> full_data_;
};

} // namespace rocmetrics

#endif // _THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_UTIL_ROC_METRICS_ROC_METRICS_H_
