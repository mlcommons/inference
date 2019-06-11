#ifndef MLPERF_LOADGEN_UTILS_H
#define MLPERF_LOADGEN_UTILS_H

#include <algorithm>
#include <chrono>
#include <string>

#include "query_sample.h"

namespace mlperf {

template <typename T>
void RemoveValue(T* container, const typename T::value_type& value_to_remove) {
  container->erase(std::remove_if(container->begin(), container->end(),
                                  [&](typename T::value_type v) {
                                    return v == value_to_remove;
                                  }),
                   container->end());
}

template <typename CountT, typename RatioT>
double DurationToSeconds(
    const std::chrono::duration<CountT, RatioT>& chrono_duration) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             chrono_duration)
      .count();
}

inline double QuerySampleLatencyToSeconds(QuerySampleLatency qsl) {
  return static_cast<double>(qsl) / std::nano::den;
}

template <typename DurationT>
inline DurationT SecondsToDuration(double seconds) {
  return std::chrono::duration_cast<DurationT>(
      std::chrono::duration<double>(seconds));
}

std::string CurrentDateTimeISO8601();

std::string DoubleToString(double value, int precision = 2);

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_UTILS_H
