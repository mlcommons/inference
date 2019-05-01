#ifndef MLPERF_LOADGEN_UTILS_H
#define MLPERF_LOADGEN_UTILS_H

namespace mlperf {

template <typename T>
void RemoveValue(T* container, const typename T::value_type& value_to_remove) {
  container->erase(
      std::remove_if(container->begin(), container->end(),
                     [&](typename T::value_type v) {
                       return v == value_to_remove;
                     }),
      container->end());
}

}  // namespace mlperf

#endif // MLPERF_LOADGEN_UTILS_H
