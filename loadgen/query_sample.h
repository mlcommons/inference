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

#ifndef MLPERF_LOADGEN_QUERY_SAMPLE_H_
#define MLPERF_LOADGEN_QUERY_SAMPLE_H_

// Defines the structs involved in issuing a query and responding to a query.
// These are broken out into their own files since they are exposed as part
// of the C API and we want to avoid C clients including C++ code.

#include <stddef.h>
#include <stdint.h>

namespace mlperf {

// ResponseId represents a unique identifier for a sample of an issued query.
typedef uintptr_t ResponseId;

typedef size_t QuerySampleIndex;

// QuerySample represents the smallest unit of input inference can run on.
// A query will consist of one or more samples.
struct QuerySample {
  ResponseId id;
  QuerySampleIndex index;
};

// QuerySampleResponse represents a single response to QuerySample
struct QuerySampleResponse {
  ResponseId id;
  uintptr_t data;
  size_t size;  // Size in bytes.
};

typedef int64_t QuerySampleLatency;  // In nanoseconds.

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_QUERY_SAMPLE_H_
