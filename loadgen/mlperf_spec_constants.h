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

#ifndef MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H
#define MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H

#include <stddef.h>
#include <stdint.h>

namespace mlperf {

constexpr uint64_t kDefaultQslSeed = 0xABCD1234;
constexpr uint64_t kDefaultSampleSeed = 0x1234ABCD;
constexpr uint64_t kDefaultScheduleSeed = 0xA1B2C3D4;

constexpr size_t kMinQueryCountSingleStream = 1024;
constexpr size_t kMinQueryCountNotSingleStream = 24576;

constexpr double kMinPerformanceRunTargetLatencySeconds = 0.05;

constexpr double kMinPerformanceRunDurationSeconds = 60.0;

constexpr double kMultiStreamTargetQPS = 60.0;

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H
