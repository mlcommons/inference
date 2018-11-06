// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "trace_generator.h"

#include <iostream>

int main(int, char **) {
  std::vector<int> query_library(10);
  std::iota(query_library.begin(), query_library.end(), 0);

  TraceGenerator::EnqueueFn<int> enqueue =
      [](std::chrono::nanoseconds start, int query,
         std::function<void(void)> completion_callback) {
        // TODO(tjablin): Right now, the demo inference just sleeps for one
        // hundred microseconds to demonstrate the API. This demo should be more
        // realistic.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        completion_callback();
      };

  // TODO(tjablin): These parameters should all be read from the command-line or
  // maybe a configuration file?

  // TODO(tjablin): The latency bound of 400 us is unphysically low.
  std::chrono::nanoseconds latency_bound(400000);
  uint64_t min_queries = 4096;
  std::chrono::seconds min_duration(2);
  double latency_bound_percentile = 0.95;
  uint64_t seed = 0;
  double max_qps = TraceGenerator::FindMaxQPS(
      query_library, enqueue, seed, latency_bound, min_queries, min_duration,
      latency_bound_percentile);
  std::cout << "Max QPS subject to " << latency_bound.count() << " ns "
            << std::roundl(100 * latency_bound_percentile)
            << "% latency bound: " << max_qps << "\n";

  return 0;
}
