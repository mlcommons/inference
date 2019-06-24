/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
