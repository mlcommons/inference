#ifndef SYSTEM_UNDER_TEST_C_API_H_
#define SYSTEM_UNDER_TEST_C_API_H_

// The C API allows a C or Python client to easily create
// a SystemUnderTest without having to expose the SystemUnderTest class
// directly.
// ConstructSUT works with a bunch of function poitners instead that are
// called from an underlying trampoline class.

// TODO(brianderson): Use pybind instead? The intention here was to
// run swig on the C interface for python and future language bindings.
// If C and other language bindings aren't necessary, we can remove this.

#include <stddef.h>
#include <stdint.h>

namespace mlperf {

struct QuerySample;
struct TestSettings;

namespace c {

typedef intptr_t ClientData;  // Equivalent to C++'s this pointer.

typedef void(*UntimedWarmUpCallback)(ClientData);
typedef void*(*AllocateQuerySampleCallback)(ClientData, size_t);
typedef void(*FreeQuerySampleCallback)(ClientData, void*);
typedef void(*PreprocessQuerySampleCallback)(ClientData, const void*,
                                             const size_t, void**, size_t*);
typedef void(*IssueQueryCallback)(ClientData, intptr_t, QuerySample*, size_t);

// Create and destroy an opaque SUT pointer based on C callbacks.
void* ConstructSUT(ClientData client_data, const char* name, size_t name_length,
                   UntimedWarmUpCallback warm_up_cb,
                   AllocateQuerySampleCallback allocate_cb,
                   FreeQuerySampleCallback free_cb,
                   PreprocessQuerySampleCallback preprocess_cb,
                   IssueQueryCallback issue_cb);
void DestroySUT(void* sut);

// Run tests on a SUT created by ConstructSUT().
// This is the C entry point. See mlperf::StartTest for the C++ entry point.
void StartTest(void* sut, const TestSettings& settings);

}  // namespace c
}  // namespace mlperf

#endif  // SYSTEM_UNDER_TEST_C_API_H_
