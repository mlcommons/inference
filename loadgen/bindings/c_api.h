#ifndef SYSTEM_UNDER_TEST_C_API_H_
#define SYSTEM_UNDER_TEST_C_API_H_

// The C API allows a C or Python client to easily create
// a SystemUnderTest without having to expose the SystemUnderTest class
// directly.
// ConstructSUT works with a bunch of function poitners instead that are
// called from an underlying trampoline class.

#include <stddef.h>
#include <stdint.h>

#include "../query_sample.h"

namespace mlperf {

struct TestSettings;

namespace c {

// Optional opaque client data creators of SUTs and QSLs can pass to their
// callback invocations. Helps avoids global variables.
typedef uintptr_t ClientData;

// Create and destroy an opaque SUT pointer based on C callbacks.
typedef void (*IssueQueryCallback)(ClientData, const QuerySample*, size_t);
typedef void (*ReportLatencyResultsCallback)(
    ClientData, const int64_t*, size_t);
void* ConstructSUT(ClientData client_data,
                   const char* name,
                   size_t name_length,
                   IssueQueryCallback issue_cb,
                   ReportLatencyResultsCallback report_latency_results_cb);
void DestroySUT(void* sut);

// Create and destroy an opaque QSL pointer based on C callbacks.
typedef void (*LoadSamplesToRamCallback)(
    ClientData, const QuerySampleIndex*, size_t);
typedef void (*UnloadSamplesFromRamCallback)(
    ClientData, const QuerySampleIndex*, size_t);
void* ConstructQSL(ClientData client_data, const char* name, size_t name_length,
                   size_t total_sample_count, size_t performance_sample_count,
                   LoadSamplesToRamCallback load_samples_to_ram_cb,
                   UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb);
void DestroyQSL(void* qsl);

// Run tests on a SUT created by ConstructSUT().
// This is the C entry point. See mlperf::StartTest for the C++ entry point.
// TODO(brianderson): Implement query sample library callbacks.
void StartTest(void* sut, void* qsl, const TestSettings& settings);

}  // namespace c
}  // namespace mlperf

#endif  // SYSTEM_UNDER_TEST_C_API_H_
