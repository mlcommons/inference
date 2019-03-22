#include "test_harness.h"

#include <iostream>
#include <string>

namespace mlperf {

TestSettings ParseCommandLine(const char* command) {
  // TODO(brianderson): Figure out natural hooks for python.
  // TODO(brianderson): Handoff to a common helper that ParseCommandLineArgs
  //                    also uses.
  return TestSettings();
}

TestSettings ParseCommandLineArgs(int argc, char** argv) {
  TestSettings settings;
  for (int i = 0; i < argc - 1; i++) {
    std::string n = argv[i];
    if (n == "--mlperf_scenario") {
      std::string v(argv[i + 1]);
      if (v == "cloud") {
        settings.scenario = TestScenario::Cloud;
      } else if (v == "edge") {
        settings.scenario = TestScenario::Edge;
      } else if (v == "offline") {
        settings.scenario = TestScenario::Offline;
      } else {
        std::cerr << "Bad scenario: " << v;
      }
    }
    // TODO(brianderson): More settings.
  }
  return settings;
}

}  // namespace mlperf
