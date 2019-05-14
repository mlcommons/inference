#ifndef MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H
#define MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H

namespace mlperf {

constexpr uint64_t kDefaultQslSeed = 0xABCD1234;
constexpr uint64_t kDefaultSampleSeed = 0x1234ABCD;
constexpr uint64_t kDefaultScheduleSeed = 0xA1B2C3D4;

constexpr size_t kMinQueryCountSingleStream = 1024;
constexpr size_t kMinQueryCountNotSingleStream = 24576;

constexpr double kMinPerformanceRunDurationSeconds = 60.0;

constexpr double kMultiStreamTargetQPS = 60.0;

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H
