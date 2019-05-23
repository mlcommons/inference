#include "utils.h"

#include <sstream>

namespace mlperf {

std::string DoubleToString(double value, int precision) {
  std::stringstream ss;
  ss.precision(precision);
  ss << std::fixed << value;
  return ss.str();
}

}  // namespace mlperf
