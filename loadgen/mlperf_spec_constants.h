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

// Units are seconds.
constexpr double kMultiStreamTargetLatencyMobileNet = .05;
constexpr double kMultiStreamTargetLatencyResNet = .05;
constexpr double kMultiStreamTargetLatencySsdSmall = .05;
constexpr double kMultiStreamTargetLatencySsdLarge = .05;
constexpr double kMultiStreamTargetLatencyNMT = .1;

// Units are seconds.
constexpr double kServerTargetLatencyMobileNet = .05;
constexpr double kServerTargetLatencyResNet = .05;
constexpr double kServerTargetLatencySsdSmall = .05;
constexpr double kServerTargetLatencySsdLarge = .05;
constexpr double kServerTargetLatencyNMT = .1;

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H
