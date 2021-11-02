#include "third_party/tensorflow_models/mlperf/models/rough/util/roc_metrics/roc_metrics.h" //NOLINT

#include <Python.h>

#include <vector>

#include "absl/types/span.h"
#include "base/logging.h"
#include "boost/boost/boost/sort/block_indirect_sort/block_indirect_sort.hpp" //NOLINT
#include "numpy/core/include/numpy/ndarrayobject.h"
#include "numpy/core/include/numpy/ndarraytypes.h"
#include "numpy/core/include/numpy/npy_common.h"

namespace rocmetrics {
namespace {

template <typename T, typename Compare>
void ParallelSort(absl::Span<T> elements, const Compare &comp) {
  // As of 2020-May, boost's block_indirect_sort library is not allowed in
  // repo.  If using this library in a non-third_party project, this will
  // need to be replaced.
  boost::sort::block_indirect_sort(elements.begin(), elements.end(), comp);
}

// Integrates the y vector along the x vector using the trapezoidal rule.
// Performance of this function can be improved via parallelization.
double Trapz(const std::vector<float> &y, const std::vector<float> &x) {
  // Performance can be improved via parallelization.
  double ret = 0.0;
  float x_prev = x[0];
  float y_prev = y[0];
  auto trap_area = [](float x0, float y0, float x1, float y1) {
    float retval = 0.5 * (x1 - x0) * (y0 + y1);
    return double{retval};
  };

  for (int i = 1; i < y.size(); ++i) {
    if (x_prev == 1.0) {
      // Early stop criteria.
      break;
    }
    if (x[i] != x_prev) {
      ret += trap_area(x_prev, y_prev, x[i], y[i]);
    }
    x_prev = x[i];
    y_prev = y[i];
  }
  return ret;
}

} // namespace

RocMetrics::RocMetrics(PyObject *py_scores, PyObject *py_targets) {
  CHECK(PyArray_Check(py_scores)) << "scores object is not a PyArrayObject.";
  CHECK(PyArray_Check(py_targets)) << "targets object is not a PyArrayObject.";
  PyArrayObject *p_scores = reinterpret_cast<PyArrayObject *>(py_scores);
  PyArrayObject *p_targets = reinterpret_cast<PyArrayObject *>(py_targets);
  int p_data_len = PyArray_SHAPE(p_scores)[0];
  int t_data_len = PyArray_SHAPE(p_targets)[0];
  CHECK(p_data_len > 1) << "Scores array must be of length greater than 1.";
  CHECK(t_data_len > 1) << "Targets array must be of length greater than 1.";
  CHECK(p_data_len == t_data_len)
      << "Scores and targets must be of same length.";
  full_data_.reserve(p_data_len);
  auto scores_ptr = static_cast<const float *>(PyArray_GETPTR1(p_scores, 0));
  auto tgts_ptr = static_cast<const int *>(PyArray_GETPTR1(p_targets, 0));
  for (int i = 0; i < p_data_len; ++i) {
    if (tgts_ptr[i] >= 0) {
      full_data_.push_back({scores_ptr[i], tgts_ptr[i]});
    }
  }
  LOG(INFO) << "== roc_metrics: number of valid eval samples: "
            << full_data_.size();
}

RocData RocMetrics::BinaryRoc() const {
  // TPR and FPR should begin at point (0, 0).
  std::vector<int> tps = {0};
  std::vector<int> fps = {0};

  float prev_score = full_data_[0].score;
  int accum = 0, thresh_idx = 0;

  for (const PredictElem &d : full_data_) {
    float cur_score = d.score;
    if (cur_score != prev_score) {
      tps.push_back(accum);
      fps.push_back(thresh_idx - accum);
    }
    prev_score = cur_score;
    accum += d.target;
    thresh_idx++;
  }
  // Include full sum, for normalization to 1.0.
  tps.push_back(accum);
  fps.push_back(thresh_idx - accum);
  return {std::move(tps), std::move(fps)};
}

float RocMetrics::ComputeRocAuc() {
  ParallelSort(absl::MakeSpan(full_data_),
               [](const PredictElem &t1, const PredictElem &t2) {
                 return t1.score > t2.score;
               });

  // Generate TPR and FPR.
  const auto [tps, fps] = BinaryRoc();
  std::vector<float> tpr(tps.size());
  std::vector<float> fpr(fps.size());
  const float tp_count = static_cast<float>(tps.back());
  const float fp_count = static_cast<float>(fps.back());
  for (int i = 0; i < tps.size(); ++i) {
    tpr[i] = static_cast<float>(tps[i]) / tp_count;
    fpr[i] = static_cast<float>(fps[i]) / fp_count;
  }

  // Trapezoidal integration to compute the AUC.
  double auc = Trapz(tpr, fpr);

  return static_cast<float>(auc);
}

} // namespace rocmetrics
