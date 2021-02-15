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

/// \file
/// \brief Prints out system details.

#include <array>
#include <iostream>
#include <memory>

#include "logging.h"
#include "utils.h"

namespace mlperf {

std::string RunCommand(const std::string& cmd) {
  std::string result;
  try {
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
      throw;
    }
    std::array<char, 128> buf;
    while (fgets(buf.data(), buf.size(), pipe.get())) {
      result.append(buf.data());
    }
    auto last_char = result.find_last_not_of(" \t\f\v\n\r");
    if (last_char != std::string::npos) {
      result.erase(last_char + 1);
    }
    else {
      result.clear();
    }
    result = EscapeStringJson(result);
  }
  catch (...) {
    result = "Command failed";
  }
  return result;
}

void LogSystemDetails() {
  LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    // TODO: Load these commands from a file instead.
    // TODO: Support vendor-specific commands.
    MLPERF_LOG(detail, "system_os_version", RunCommand("cat /etc/lsb-release |grep DISTRIB_RELEASE |cut -f 2 -d \"=\""));
#endif
  });
}

}  // namespace mlperf
