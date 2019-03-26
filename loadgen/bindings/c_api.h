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

typedef intptr_t ClientData;
typedef void (*IssueQueryCallback)(ClientData, intptr_t, QuerySample*, size_t);

// Create and destroy an opaque SUT pointer based on C callbacks.
void* ConstructSUT(ClientData client_data, const char* name, size_t name_length,
                   IssueQueryCallback issue_cb);
void DestroySUT(void* sut);

// Run tests on a SUT created by ConstructSUT().
// This is the C entry point. See mlperf::StartTest for the C++ entry point.
// TODO(brianderson): Implement query sample library callbacks.
void StartTest(void* sut, void* qsl, const TestSettings& settings);

}  // namespace c
}  // namespace mlperf

#endif  // SYSTEM_UNDER_TEST_C_API_H_
