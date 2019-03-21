#include "test_harness.h"

#include <string>

namespace mlperf {

TestSettings ParseCommandLine(const char* command) {
  // TODO(brianderson): Figure out natural hooks for python.
  // TODO(brianderson): Handoff to a common helper that ParseCommandLineArgs
  //                    also uses.
}

TestSettings ParseCommandLineArgs(int argc, char** argv) {
  TestSettings settings;
  for (int i = 0; i < argc - 1; i++) {
    std::string n = argv[i];
    if (n == "--mlperf_scenario") {
      std::string v(argv[i + 1]);
      if (n == "cloud") {
        settings.scenario = Scenario::Cloud;
      } else if (n == "edge") {
        settings.scenario = Scenario::Edge;
      } else if (n == "offline") {
        settings.scenario = Scenario::Offline;
      }
    }
    // TODO(brianderson): More settings.
  }
  return settings;
}

}  // namespace mlperf
