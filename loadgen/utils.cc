#include "utils.h"

#include <ctime>
#include <sstream>

namespace mlperf {

std::string DoubleToString(double value, int precision) {
  std::stringstream ss;
  ss.precision(precision);
  ss << std::fixed << value;
  return ss.str();
}

std::string CurrentDateTimeISO8601() {
  std::time_t current_time = std::time(nullptr);
  std::tm date_time = *std::localtime(&current_time);
  constexpr size_t kDateTimeMaxSize = 256;
  char date_time_cstring[kDateTimeMaxSize];
  std::strftime(date_time_cstring, kDateTimeMaxSize, "%FT%TZ", &date_time);
  return date_time_cstring;
}

}  // namespace mlperf
