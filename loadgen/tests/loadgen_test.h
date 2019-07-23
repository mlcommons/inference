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

// A minimal test framework.

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

class Test {
  using TestMap = std::multimap<const char*, std::function<void()>>;
  static TestMap& tests() {
    static TestMap tests_;
    return tests_;
  }

  static size_t& test_fails() {
    static size_t test_fails_ = 0;
    return test_fails_;
  }

 public:
  struct StaticRegistrant {
    template <typename... Args>
    StaticRegistrant(Args&&... args) {
      Test::Register(std::forward<Args>(args)...);
    }
  };

  template <typename TestF, typename... Args>
  static void Register(const char* name, TestF test, Args&&... args) {
    std::function<void()> test_closure =
        std::bind(test, std::forward<Args>(args)...);
    tests().insert({std::move(name), std::move(test_closure)});
  }

  static int Run(std::function<bool(const char*)> filter) {
    // Determine which tests are enabled.
    std::vector<TestMap::value_type*> enabled_tests;
    for (auto& test : tests()) {
      if (filter(test.first)) {
        enabled_tests.push_back(&test);
      }
    }
    const size_t enabled_count = enabled_tests.size();
    std::cout << enabled_count << " of " << tests().size()
              << " tests match regex filters.\n";

    // Run the tests.
    std::vector<const char*> failures;
    for (size_t i = 0; i < enabled_count; i++) {
      const char* name = enabled_tests[i]->first;
      std::cout << "[" << i << "/" << enabled_count << "] : " << name << " : ";
      std::cout.flush();
      test_fails() = 0;
      enabled_tests[i]->second();  // Run the test.
      if (test_fails() > 0) {
        failures.push_back(name);
        std::cerr << "\n FAILED: " << name << "\n";
      } else {
        std::cout << "SUCCESS\n";
      }
    }

    // Summarize.
    if (enabled_tests.empty()) {
      std::cerr << "Check your regexes.\n";
    } else if (failures.empty()) {
      std::cout << "All " << enabled_count << " tests passed! \\o/\n";
    } else {
      std::cout << failures.size() << " of " << enabled_count
                << " tests failed:\n";
      for (auto failed_test_name : failures) {
        std::cout << "  " << failed_test_name << "\n";
      }
    }
    return failures.size();
  }

  static void AddFailure() { test_fails()++; }

  static void Log() {}
  template <typename T, typename... Args>
  static void Log(T&& v, Args&&... args) {
    std::cerr << v;
    Log(std::forward<Args>(args)...);
  }
};

#define REGISTER_TEST(name, ...) \
  static Test::StaticRegistrant test##name(#name, __VA_ARGS__);

#define REGISTER_TEST_SCENARIO(name, scenario, test, ...) \
  static Test::StaticRegistrant t##name##scenario(        \
      #name "_" #scenario, test, mlperf::TestScenario::scenario)

#define REGISTER_TEST_ALL_SCENARIOS(name, test)        \
  REGISTER_TEST_SCENARIO(name, SingleStream, test);    \
  REGISTER_TEST_SCENARIO(name, MultiStream, test);     \
  REGISTER_TEST_SCENARIO(name, MultiStreamFree, test); \
  REGISTER_TEST_SCENARIO(name, Server, test);          \
  REGISTER_TEST_SCENARIO(name, Offline, test);

#define FAIL_IF(exp)                                              \
  [&]() {                                                         \
    const bool v = exp;                                           \
    if (v) {                                                      \
      std::cerr << "\n   ERROR: (" << __FILE__ << "@" << __LINE__ \
                << ") : " #exp;                                   \
      Test::AddFailure();                                         \
    }                                                             \
    return v;                                                     \
  }()

#define FAIL_MSG(...)                                                      \
  [&]() {                                                                  \
    std::cerr << "\n    Info: (" << __FILE__ << "@" << __LINE__ << ") : "; \
    Test::Log(__VA_ARGS__);                                                \
    return true;                                                           \
  }()

#define FAIL_EXP(exp)                                                          \
  [&]() {                                                                      \
    std::cerr << "\n    Info: (" << __FILE__ << "@" << __LINE__ << ") : " #exp \
              << " is " << (exp);                                              \
    return true;                                                               \
  }()
