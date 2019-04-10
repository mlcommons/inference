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

#ifndef TRACE_GENERATOR_H
#define TRACE_GENERATOR_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// Theory of Operations
//
// 1. Generate a trace with Poisson distributed arrival times satisfying all
// specified requirements in terms of queries, QPS, and duration.
//
// 2. Attempt to run the trace capturing the latency between query arrival time
// and inference completion.
//
// 3. Increase QPS until the latency constraint is violated. Use binary search
// to find the maximum QPS achieved without violating the latency constraint.

namespace TraceGenerator {

// TODO(tjablin): Instead of all of this template stuff, maybe just pass around
// the indices of the queries in the query library? If the "natural" QueryType
// expensive to copy, people will want to use std::shared_ptrs<QueryType>
// instead.

template <class QueryType>
using QueryLibrary = std::vector<QueryType>;

template <class QueryType>
using TraceEntry = std::pair<std::chrono::nanoseconds, QueryType>;

template <class QueryType>
using Trace = std::vector<TraceEntry<QueryType>>;

// The first argument is the arrival time of the query. The second argument is
// the query. The third argument is a callback used to record the end of
// inferrence. Typical implementations will enqueue work to be done by another
// thread rather than doing it themselves.
template <class QueryType>
using EnqueueFn = std::function<void(std::chrono::nanoseconds, QueryType,
                                     std::function<void(void)>)>;

template <class QueryType>
double CalculateQPS(const Trace<QueryType> &trace) {
  int queries = trace.size();
  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        trace.rbegin()->first)
                        .count();
  return queries / duration;
}

// TODO(tjablin): Many of these methods have long argument lists. Maybe use
// fluent APIs instead?

// TODO(tjablin): Instead of generating the whole trace at once, maybe generate
// trace entries dynamically?

// TODO(tjablin): Add accuracy-mode versus performance mode.

// Generate a trace from a query library based on a seed with a given minimum
// number of queries, miniumum duration, and qps.
template <class QueryType>
Trace<QueryType> GenerateTrace(const QueryLibrary<QueryType> &query_library,
                               uint64_t seed, int min_queries,
                               std::chrono::seconds min_duration, double qps) {
  // Using the std::mt19937 pseudo-random number generator ensures a modicum of
  // cross platform reproducibility for trace generation.
  std::mt19937 gen(seed);
  std::exponential_distribution<> exponential_distribution(qps);
  std::uniform_int_distribution<> uniform_distribution(
      0, query_library.size() - 1);

  std::chrono::nanoseconds timestamp(0);
  Trace<QueryType> trace;
  while (timestamp < min_duration || trace.size() < min_queries) {
    // Poisson arrival process corresponds to exponentially distributed
    // interarrival times.
    timestamp += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(exponential_distribution(gen)));
    trace.emplace_back(timestamp, query_library[uniform_distribution(gen)]);
  }
  return trace;
}

// TODO(tjablin): Maybe return a histogram instead of just one latency?

// Replay a trace using a user provided work enqueueing function. Returns the
// 99-percentile latency.
template <class QueryType>
std::chrono::nanoseconds ReplayTrace(const Trace<QueryType> &trace,
                                     EnqueueFn<QueryType> enqueue,
                                     double latency_bound_percentile) {
  if (trace.empty()) {
    return std::chrono::nanoseconds(0);
  }

  // TODO(tjablin): Support Start-up time.
  // TODO(tjablin): Support Warm-up time.

  std::vector<std::chrono::nanoseconds> latencies(trace.size());
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < trace.size(); ++i) {
    const auto &trace_entry = trace[i];
    auto query_start_time = start + trace_entry.first;
    std::this_thread::sleep_until(query_start_time);

    // TODO(tjablin): The completion callback should record the result of the
    // inference.
    auto completion_callback = [i, query_start_time, &latencies] {
      latencies[i] =
          std::chrono::high_resolution_clock::now() - query_start_time;
    };

    enqueue(trace_entry.first, trace_entry.second, completion_callback);
  }

  // TODO(tjablin): Some sort of user supplied BlockUntilDone method should be
  // called here.

  std::sort(latencies.begin(), latencies.end());
  return latencies[std::ceil(latency_bound_percentile *
                             (latencies.size() - 1))];
}

// TODO(tjablin): Maybe return an interval instead?

// Returns the maximum throughput (QPS) subject to a 99-percentile latency
// bound.
template <class QueryType>
double FindMaxQPS(const QueryLibrary<QueryType> &query_library,
                  EnqueueFn<QueryType> enqueue, uint64_t seed,
                  std::chrono::nanoseconds latency_bound, int min_queries,
                  std::chrono::seconds min_duration,
                  double latency_bound_percentile) {
  double qps_lower_bound = 0;
  double qps_upper_bound = INFINITY;
  double relative_qps_tolerance = 0.01;
  // TODO(tjablin): This halt condition seems kind of arbitrary.
  while ((qps_upper_bound - qps_lower_bound) / qps_lower_bound >
         relative_qps_tolerance) {
    double target_qps;
    if (qps_lower_bound == 0 && qps_upper_bound == INFINITY) {
      target_qps = 512;
    } else if (qps_upper_bound == INFINITY) {
      target_qps = 2 * qps_lower_bound;
    } else {
      target_qps = (qps_lower_bound + qps_upper_bound) / 2;
    }

    auto trace = GenerateTrace(query_library, ++seed, min_queries, min_duration,
                               target_qps);
    double trace_qps = CalculateQPS(trace);
    if (qps_lower_bound < trace_qps && trace_qps < qps_upper_bound) {
      std::chrono::nanoseconds measured_latency =
          ReplayTrace(trace, enqueue, latency_bound_percentile);
      std::cout << "QPS: " << trace_qps << "\t"
                << std::roundl(100 * latency_bound_percentile)
                << "% latency: " << measured_latency.count() << " ns\n";
      if (measured_latency > latency_bound) {
        qps_upper_bound = std::min(qps_upper_bound, trace_qps);
      } else {
        qps_lower_bound = std::max(trace_qps, qps_lower_bound);
      }
    }
  }

  return std::min(qps_upper_bound, qps_lower_bound);
}

}  // namespace TraceGenerator

#endif  // TRACE_GENERATOR_H
